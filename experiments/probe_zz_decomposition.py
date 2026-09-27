"""Decompose the ZZ-correlator gain into a classical part and an entangled part.

Reviewers 4 (#2) and 6 object that the paper's ZZ ablation is not
dimension-matched (9 features against 45) and that raw <Z_iZ_j> certifies
neither entanglement nor even correlation. Both objections are answered by one
exact identity:

    <Z_i Z_j>  =  <Z_i><Z_j>  +  C_ij ,       C_ij = <Z_i Z_j> - <Z_i><Z_j>

The first term is a deterministic function of the nine single-qubit marginals,
so it requires no correlation at all: a classical practitioner holding only the
Z-only cache could compute it for free. The second is the connected correlator,
the genuinely correlated part. Because the simulated states are pure (unitary
evolution from a product state, analytic expectations, no shots and no
decoherence), C_ij != 0 implies the state is not a product across the
{i} | rest bipartition, i.e. it is entangled.

This script builds four dimension-controlled feature sets per topology:

    z     nine <Z_i>                                       (  9 channels)
    zz    nine <Z_i> + 36 raw <Z_iZ_j>                      ( 45 channels)
    prod  nine <Z_i> + 36 products <Z_i><Z_j>               ( 45 channels)
    conn  nine <Z_i> + 36 connected C_ij                    ( 45 channels)

zz = prod + conn channel-wise, so the three 45-channel arms are mutually
dimension-matched and the comparison isolates what the correlations add over
their classically free shadow.

All four are derived from an existing ZZ cache with no quantum recomputation.

Run from the repo root:
    python experiments/probe_zz_decomposition.py --dataset breast_mnist
Writes outputs/camera_ready/zz_decomposition_<dataset>.csv (gitignored).
"""
import argparse
import csv
import json
import os
import zipfile

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Same protocol as linear_probing_topology_sweep.py, except that C is tuned on
# validation rather than fixed at 1.0 (the project's fairness rule: an untuned
# C under-regularises high-dimensional maps and would penalise the 45-channel
# arms relative to the 9-channel one).
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


def topology_channels(meta, topology):
    """Z channel indices and (channel, i, j) ZZ triples for one topology."""
    z_channels, zz_triples = [], []
    for entry in meta["channel_kernel_map"]:
        if entry["kernel"] != topology:
            continue
        if entry["measurement"] == "Z":
            z_channels.append((entry["qubit"], entry["channel"]))
        else:
            zz_triples.append((entry["channel"], *entry["qubit_pair"]))
    # order by qubit index so position k in the stack is qubit k
    z_channels.sort()
    return [ch for _, ch in z_channels], zz_triples


def build_arms(features, z_channels, zz_triples):
    """Four (N, channels, H, W) feature stacks from one topology's channels."""
    z = features[:, z_channels, :, :]
    zz_raw = features[:, [ch for ch, _, _ in zz_triples], :, :]
    # product of marginals, pair by pair, in the cache's own pair order
    prod = np.stack([z[:, i] * z[:, j] for _, i, j in zz_triples], axis=1)
    conn = zz_raw - prod
    return {
        "z": z,
        "zz": np.concatenate([z, zz_raw], axis=1),
        "prod": np.concatenate([z, prod], axis=1),
        "conn": np.concatenate([z, conn], axis=1),
    }


def stream_subsample(zf, member, keep, rows_per_block=256):
    """Read rows `keep` (sorted indices) out of a compressed .npy zip member.

    The Tissue caches are deflate-compressed, so there is no random access: the
    member has to be decompressed from the start. We walk it in blocks and copy
    out only the wanted rows, which keeps peak memory at the size of the
    subsample rather than the 38.6 GB full array.
    """
    with zf.open(member) as fh:
        version = np.lib.format.read_magic(fh)
        shape, fortran, dtype = np.lib.format._read_array_header(fh, version)
        assert not fortran, "Fortran-ordered cache not supported"
        row_items = int(np.prod(shape[1:]))
        row_bytes = row_items * dtype.itemsize

        out = np.empty((len(keep), row_items), dtype=np.float32)
        ki = 0          # next position in `keep` still to be filled
        row = 0         # index of the first row in the current block
        while row < shape[0] and ki < len(keep):
            n = min(rows_per_block, shape[0] - row)
            buf = read_exact(fh, row_bytes * n)
            block = np.frombuffer(buf, dtype=dtype).reshape(n, row_items)
            while ki < len(keep) and keep[ki] < row + n:
                out[ki] = block[keep[ki] - row]
                ki += 1
            row += n
    assert ki == len(keep), f"{member}: only filled {ki} of {len(keep)} rows"
    return out.reshape((len(keep),) + shape[1:])


def read_exact(fh, nbytes):
    """zipfile streams may return short reads; loop until we have the block."""
    chunks, got = [], 0
    while got < nbytes:
        chunk = fh.read(nbytes - got)
        if not chunk:
            break
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def load_splits(cache_path, max_rows, seed=0):
    """Load train/val/test, subsampling rows when max_rows is given.

    max_rows is (n_train, n_val, n_test); None means take the whole split.
    Subsampling is a fixed-seed random draw over the whole split, not a prefix,
    so it cannot be biased by any class ordering in the cache.
    """
    zf = zipfile.ZipFile(cache_path)
    meta = json.loads(str(np.load(zf.open("metadata.npy"), allow_pickle=True)))
    splits = {}
    for split, cap in zip(("train", "val", "test"), max_rows):
        labels = np.load(zf.open(f"{split}_labels.npy")).ravel()
        total = len(labels)
        if cap is None or cap >= total:
            keep = np.arange(total)
        else:
            keep = np.sort(np.random.default_rng(seed).choice(total, cap, replace=False))
        features = stream_subsample(zf, f"{split}_features.npy", keep)
        splits[split] = (features, labels[keep])
        print(f"  {split}: {features.shape} of {total} rows", flush=True)
    return meta, splits


def flat(x):
    return x.reshape(len(x), -1)


def make_clf(C, estimator):
    """The two linear probes, sharing one C grid.

    LinearSVC is liblinear, which is single-threaded and fits one binary
    problem per class. On the 25k-row 8-class TissueMNIST blocks a single arm
    took over two hours, so multiclass runs use LogisticRegression instead:
    lbfgs solves all eight classes in one multinomial fit. The same choice is
    already made, for the same reason, in probe_tissue_topologies.py.
    """
    if estimator == "logreg":
        return LogisticRegression(C=C, max_iter=2000)
    return LinearSVC(C=C, max_iter=50_000, dual="auto")


def probe(X_tr, y_tr, X_va, y_va, X_te, y_te, estimator="svc"):
    """Standardise, tune C on validation, refit, report test accuracy and AUC."""
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_va, X_te = (scaler.transform(x) for x in (X_tr, X_va, X_te))

    def fit_score(C, X, y):
        svm = make_clf(C, estimator).fit(X_tr, y_tr)
        scores = svm.decision_function(X)
        if scores.ndim == 1:
            auc = roc_auc_score(y, scores)
        else:
            # Macro one-vs-rest AUC straight from the OVR decision function.
            # sklearn's multi_class="ovr" path insists on normalised
            # probabilities, which a margin classifier does not produce, so we
            # binarise the labels and average the per-class AUCs ourselves.
            y_bin = label_binarize(y, classes=svm.classes_)
            auc = roc_auc_score(y_bin, scores, average="macro")
        return svm, accuracy_score(y, svm.predict(X)), auc

    best_C, best_val_auc = None, -np.inf
    for C in C_GRID:
        _, _, val_auc = fit_score(C, X_va, y_va)
        if val_auc > best_val_auc:
            best_C, best_val_auc = C, val_auc
    _, test_acc, test_auc = fit_score(best_C, X_te, y_te)
    return best_C, best_val_auc, test_acc, test_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="path to a *_zz[_analog].npz cache")
    ap.add_argument("--topologies", nargs="*", default=None,
                    help="subset of topologies; default all in the cache")
    ap.add_argument("--tag", default=None, help="label for the output file")
    ap.add_argument("--estimator", choices=["svc", "logreg"], default="svc",
                    help="logreg is required for multiclass; see make_clf")
    ap.add_argument("--max-rows", nargs=3, type=int, default=None,
                    metavar=("TRAIN", "VAL", "TEST"),
                    help="subsample each split; needed for the 38.6 GB Tissue cache")
    args = ap.parse_args()

    cache_path = os.path.join(ROOT, args.cache)
    tag = args.tag or os.path.basename(args.cache).replace(".npz", "")
    print(f"{tag}: loading", flush=True)

    if args.max_rows is None:
        data = np.load(cache_path, allow_pickle=True)
        meta = json.loads(str(data["metadata"]))
        splits = {s: (data[f"{s}_features"], data[f"{s}_labels"].ravel())
                  for s in ("train", "val", "test")}
    else:
        meta, splits = load_splits(cache_path, args.max_rows)

    assert meta["include_correlators"], "not a ZZ cache"
    topologies = args.topologies or meta["kernel_topology_names"]
    print(f"{tag}: train {splits['train'][0].shape}, {len(topologies)} topologies")

    out_dir = os.path.join(ROOT, "outputs", "camera_ready")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"zz_decomposition_{tag}.csv")

    rows = []
    for topology in topologies:
        z_channels, zz_triples = topology_channels(meta, topology)
        assert len(z_channels) == 9 and len(zz_triples) == 36, \
            f"{topology}: got {len(z_channels)} Z and {len(zz_triples)} ZZ channels"
        arms = {s: build_arms(f, z_channels, zz_triples) for s, (f, _) in splits.items()}

        # How much of the raw ZZ signal is genuinely correlated? Ratio of the
        # connected part's spread to the product part's, on the train split.
        conn_only = arms["train"]["conn"][:, 9:]
        prod_only = arms["train"]["prod"][:, 9:]
        conn_share = float(conn_only.std() / (conn_only.std() + prod_only.std()))

        print(f"\n  {topology} (connected share of spread {conn_share:.3f})")
        for name in ("z", "zz", "prod", "conn"):
            C, val_auc, test_acc, test_auc = probe(
                flat(arms["train"][name]), splits["train"][1],
                flat(arms["val"][name]), splits["val"][1],
                flat(arms["test"][name]), splits["test"][1],
                estimator=args.estimator)
            dim = arms["train"][name].shape[1]
            rows.append({"cache": tag, "topology": topology, "arm": name,
                         "estimator": args.estimator,
                         "channels": dim, "C": C, "val_auc": val_auc,
                         "test_acc": test_acc, "test_auc": test_auc,
                         "connected_share": conn_share})
            print(f"    {name:<5} {dim:>3} ch  C={C:<6g} val AUC {val_auc:.4f}  "
                  f"test AUC {test_auc:.4f}  acc {test_acc:.4f}", flush=True)

        # rewrite after every topology: a multi-hour run must never be
        # all-or-nothing if it is interrupted
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
