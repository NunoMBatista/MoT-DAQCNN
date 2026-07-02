# MCC SLURM launchers

One-shot `sbatch` scripts used to run the heavy jobs on the Mary Coombs Cluster
(Hartree/STFC). They are cluster-specific (account placeholder `<your-account>`, the project venv at
`~/MoT-DAQCNN/.venv`, 200 G for the ZZ-cache jobs) and assume the repo is synced
to `~/MoT-DAQCNN`. See the "Remote HPC: Mary Coombs Cluster" section of
`CLAUDE.md` for setup and gotchas.

Kept here for provenance — how each paper artifact was actually produced:

- `mcc_tissue_e2e.sh` / `mcc_tissue_e2e_launch.sh` — TissueMNIST end-to-end HP
  searches (digital + classical), one job per `MODEL`.
- `mcc_tissue_analog_e2e.sh` — analog-ZZ end-to-end rows (200 G for the analog
  ZZ cache).
- `mcc_tissue_analog_capsweep.sh` — analog column of the capacity sweep.
- `mcc_tissue_classical_cache.sh` — build the poly-2/RFF classical feature caches.
- `mcc_tissue_analog_merge.sh` — merge the analog chunk files into one cache
  (runs the four integrity gates).
- `mcc_tissue_analog_verify.sh` — post-merge value-sanity check (NaN/Inf/range).
- `mcc_tissue_analog_topo_probe.sh` — validation-selected analog topology probe.
- `mcc_tissue_derive_z.sh` — slice the analog-Z cache from the analog-ZZ cache.
- `mcc_breast_classical.sh` — BreastMNIST classical baselines.

Self-perpetuating analog chunk-generation chains (`tissue_analog_chain2.sh`) and
the per-run recovery/top-up scripts were one-offs and are not kept.
