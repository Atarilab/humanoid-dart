# Provenance

This code was assembled from the development repositories:

- `Atarilab/mjlab@folm_cluster_rebuttal` (3ec05a6): pipeline (`FoLM`),
  tracking task, train/play scripts.
- `pran-d/diffusion-planner@Cluster_branch_rebuttal` (1c8fa83): planner
  (runtime modules only).

Differences from those sources: imports are moved into the `humanoid_dart`
package, the branch's edits inside mjlab are re-implemented in the plugin
(`envs.py`, `assets/`, `tracking/`), planner normalization stats are computed
per run under `out_dir`, and W&B logging is opt-in.
