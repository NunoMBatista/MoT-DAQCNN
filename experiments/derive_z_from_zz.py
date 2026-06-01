"""
derive_z_from_zz.py

Derive a Z-only quantum-feature cache from an existing ZZ cache by selecting the
single-qubit <Z_i> channels. The <Z_i> expectation values are computed from the
same final state regardless of whether <Z_iZ_j> is also measured, so the Z-only
features are exactly the Z channels of the ZZ cache -- no quantum recomputation.

For analog caches this is bit-exact (analog simulation is deterministic). For
digital caches the sliced Z equals a standalone digital_z run to ~1e-16 (intrinsic
cross-process float nondeterminism), which is irrelevant downstream.

Usage:
    python experiments/derive_z_from_zz.py --zz path/to/<cache>_zz[_analog].npz
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.create_quantum_dataset import generate_output_filename


def derive_z(zz_path, out_dir=None):
    zz_path = Path(zz_path)
    z = np.load(zz_path, allow_pickle=True)
    meta = json.loads(str(z["metadata"]))
    assert meta["include_correlators"], f"{zz_path.name} is not a ZZ cache"

    cmap = meta["channel_kernel_map"]
    z_idx = [e["channel"] for e in cmap if e["measurement"] == "Z"]
    n_topo = len(meta["kernel_topology_names"])
    n_qubits = meta["kernel_size"] ** 2
    assert len(z_idx) == n_topo * n_qubits, (
        f"expected {n_topo * n_qubits} Z channels, found {len(z_idx)}"
    )
    # Z indices must be ascending (sanity: we slice in cache order)
    assert z_idx == sorted(z_idx)

    # New channel map: the Z entries, renumbered 0..len-1.
    new_cmap = []
    for new_ch, old in enumerate(z_idx):
        e = dict(cmap[old])
        e["channel"] = new_ch
        new_cmap.append(e)

    out = {}
    for s in ["train", "val", "test"]:
        out[f"{s}_features"] = z[f"{s}_features"][:, z_idx, :, :].copy()
        out[f"{s}_labels"] = z[f"{s}_labels"]

    new_meta = dict(meta)
    new_meta["include_correlators"] = False
    new_meta["out_channels"] = len(z_idx)
    new_meta["quantum_out_channels"] = len(z_idx)
    new_meta["channel_kernel_map"] = new_cmap
    new_meta["created_at"] = datetime.now().isoformat()
    new_meta["derived_from"] = zz_path.name  # provenance

    out_dir = Path(out_dir) if out_dir else zz_path.parent
    out_path = out_dir / generate_output_filename(new_meta)

    np.savez_compressed(
        out_path,
        train_features=out["train_features"], train_labels=out["train_labels"],
        val_features=out["val_features"], val_labels=out["val_labels"],
        test_features=out["test_features"], test_labels=out["test_labels"],
        metadata=json.dumps(new_meta),
    )
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(new_meta, f, indent=2)
    print(f"  {zz_path.name}")
    print(f"   -> {out_path.name}  ({len(z_idx)} channels, "
          f"train {out['train_features'].shape})")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zz", required=True, help="Path to a *_zz[_analog].npz cache")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    derive_z(args.zz, args.out_dir)
