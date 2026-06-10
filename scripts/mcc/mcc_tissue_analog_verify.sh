#!/bin/bash
#SBATCH -A mxs42
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH -J ta_verify
#SBATCH --output=logs/ta_verify_%j.out
# Post-merge value-sanity check: per split, confirm no NaN/Inf and that all
# <Z>/<ZZ> values lie in the physical [-1,1] range. Loads one split at a time
# to bound memory.
cd $HOME/MoT-DAQCNN
.venv/bin/python - <<'PYEOF'
import numpy as np
f = "data/quantum_datasets/tissue_mnist__k3_s3_tkin-hor-ver-cro-rin-cha-sta-gri_ev2.50_sc1_gray_zz_analog.npz"
z = np.load(f, allow_pickle=True)
ok = True
for split in ["train", "val", "test"]:
    a = z[f"{split}_features"]
    nan = int(np.isnan(a).sum()); inf = int(np.isinf(a).sum())
    amin = float(np.nanmin(a)); amax = float(np.nanmax(a))
    oob = int(((a < -1.0001) | (a > 1.0001)).sum())
    bad = nan or inf or oob
    ok = ok and not bad
    print(f"{split:5s} shape={a.shape} dtype={a.dtype} "
          f"NaN={nan} Inf={inf} min={amin:.4f} max={amax:.4f} out_of_[-1,1]={oob} "
          f"{'OK' if not bad else 'FAIL'}")
    del a
print("VERIFY_RESULT", "PASS" if ok else "FAIL")
PYEOF
echo "TA_VERIFY_EXIT $?"
