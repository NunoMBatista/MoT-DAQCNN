"""Merge the A9 evolution-time sweep chunks, one arm at a time.

The chunk directory holds chunks from several (encoding, T) arms mixed
together, and merge_quantum_chunks.py requires that every chunk in a directory
belong to ONE cache (its first gate is cache-identity agreement). This groups
chunks by the cache prefix encoded in their filename and runs the merger once
per arm, in a temporary per-arm directory of symlinks.
"""
import argparse, os, re, subprocess, sys, tempfile
from collections import defaultdict

CHUNK_RE = re.compile(r"^(?P<base>.+?)__(?P<split>train|val|test)_\d{6}-\d{6}\.chunk\.npz$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-dir", required=True)
    ap.add_argument("--out-dir", default="data/quantum_datasets")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arms = defaultdict(list)
    for f in sorted(os.listdir(args.chunk_dir)):
        m = CHUNK_RE.match(f)
        if m:
            arms[m.group("base")].append(f)
    print(f"{len(arms)} arms in {args.chunk_dir}")

    failures = []
    for base, files in sorted(arms.items()):
        print(f"\n=== {base}  ({len(files)} chunks) ===")
        if args.dry_run:
            continue
        with tempfile.TemporaryDirectory() as d:
            for f in files:
                os.symlink(os.path.abspath(os.path.join(args.chunk_dir, f)),
                           os.path.join(d, f))
            r = subprocess.run([args.python, "experiments/merge_quantum_chunks.py",
                                "--chunk-dir", d, "--out-dir", args.out_dir],
                               capture_output=True, text=True)
            tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
            for line in tail:
                print("   ", line)
            if r.returncode != 0:
                failures.append(base)
    if failures:
        print(f"\nFAILED arms ({len(failures)}):")
        for f in failures:
            print("   ", f)
        sys.exit(1)
    print("\nall arms merged, all gates passed")


if __name__ == "__main__":
    main()
