# Archived experiments

Superseded or exploratory scripts, kept for reference. Not maintained. To run
one, move it back to `experiments/` first (relative paths assume that
directory).

- `plot_capacity_{sweep,facets,heatmap,delta}.py`, `plot_capacity_pareto.py` —
  earlier versions of the capacity figures, replaced by `plot_capacity_grid.py`
  and `plot_capacity_delta_1col.py`
- `plot_atom_topologies.py` — matplotlib topology figure, replaced by the TikZ
  version (`gen_atom_topologies_tikz.py`)
- `feature_probing_ablation.py`, `analyze_feature_importance.py` — early
  probing studies, replaced by `linear_probing_topology_sweep.py` and the
  `probe_*.py` scripts
- `kernel_cka_similarity.py`, `sparse_reconstruction.py`,
  `ablation_kernel_routing.py`, `ts_moe_vs_individual_kernels.py` — abandoned
  directions (CKA similarity, TS-MoE routing)
- `update_tissue_configs.py` — one-off config writer for the TissueMNIST runs
- `00`–`04` scripts, `benchmark_*.py`, rest — early smoke tests and
  explorations
