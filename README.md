# Humanoid-DART

Code for **Humanoid-DART: Humanoid Loco-Manipulation using Diffusion-guided
Augmentation through Relabeling and Tracking**, IEEE Robotics and Automation
Letters (RA-L), 2026.

Kanish Thiagarajan\*, Pranav Debbad\*, Victor Dhédin\*, Shafeef Omar, Majid Khadiv
(\* equal contribution)

[Project page](https://pran-d.github.io/Humanoid-DART/) ·
[arXiv:2606.26855](https://arxiv.org/abs/2606.26855)

Starting from a few demonstrations, DART grows an archive of physically valid
loco-manipulation behaviours for the Unitree G1:

1. a goal-conditioned diffusion planner generates reference trajectories for
   goals sampled by a frontier curriculum over the task space,
2. an RL tracking policy executes them in simulation (MuJoCo Warp via
   [mjlab](https://github.com/mujocolab/mjlab)),
3. rollouts that pass the fitness threshold are relabelled with the goal they
   actually achieved and added to the archive,
4. the planner is retrained on the archive, and the tracker is fine-tuned on it
   once per generation.

This repository is an mjlab plugin: it installs stock mjlab as a dependency
and adds the tracking task, the planner and the pipeline on top.

## Layout

```
configs/pick_place.yaml      pick-and-place experiment
checkpoints/model_3999.pt    tracking policy pretrained on the seeds (starting point)
motions/                     seed demonstrations (format: motions/README.md)
src/humanoid_dart/
  dart/                      pipeline: archive, curriculum, fitness, relabeling, rollouts
  planner/                   goal-conditioned diffusion planner (dual-branch DiT1D)
  tracking/                  G1 + box motion-tracking task (registered with mjlab on import)
  assets/                    G1 without hands, pick-and-place box
  envs.py                    DartRlEnv (reset events run after the command reset)
  scripts/                   train / play the tracking policy
tests/                       CPU smoke tests
```

## Installation

Requires Linux x86_64 with an NVIDIA GPU and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Atarilab/humanoid-dart.git
cd humanoid-dart
uv sync
```

`uv.lock` pins the environment the paper code ran with: mjlab at
`mujocolab/mjlab@2309ffb`, MuJoCo 3.8.1, MuJoCo Warp 3.8.0.2, Warp 1.13,
PyTorch 2.9.0 (CUDA 12.8).

## Pick-and-place experiment

The seed demonstrations are in `motions/pick_place.npz`
(format: [motions/README.md](motions/README.md)), and
`checkpoints/model_3999.pt` is the tracking policy pretrained on these seeds,
which the pick-and-place runs start from. From the repository root:

```bash
MUJOCO_GL=egl uv run dart-run --config configs/pick_place.yaml \
  --checkpoint_file checkpoints/model_3999.pt
```

Without `--checkpoint_file`, the tracker is first trained from scratch on the
seeds (`training.initial_max_iterations`, 4000 PPO iterations).

Useful overrides (any config key can be overridden on the CLI):

```bash
# 4 GPUs: tracker training via torch.distributed, rollouts split across GPUs
uv run dart-run --config configs/pick_place.yaml --num_gpus 4

# log to Weights & Biases
uv run dart-run --config configs/pick_place.yaml --wandb_project humanoid-dart
```

### Outputs

```
results/pick_place/
  run_config.yaml              resolved pipeline config
  planner_config.yaml          planner config used for this run
  planner/.../norm_stats.npz   planner normalization stats (from the seeds)
  diffusion_initial_*.pt       planner after the initial fit
  gen_XXX/
    elites.json, elites.npz    archive after the generation
    diffusion_gen_*.pt         planner checkpoints
    iter_XXX/                  per-iteration curriculum map, targets vs. achieved,
                               iteration_summary.json, iter_elites.npz
logs/rsl_rl/g1_object_tracking/<timestamp>/model_*.pt   tracker checkpoints
```

## Tracking policy on its own

```bash
# train on a motion file
uv run dart-train Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation \
  --motion-file motions/pick_place.npz --env.scene.num-envs 4096

# play a checkpoint
uv run dart-play Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation \
  --motion-file motions/pick_place.npz \
  --checkpoint-file checkpoints/model_3999.pt --num-envs 4
```

## Tests

The smoke tests run on CPU (no GPU needed) with synthetic motions:

```bash
uv sync --extra cpu
uv run --extra cpu pytest
```

## Citation

```bibtex
@article{thiagarajan2026humanoiddart,
  title         = {Humanoid-{DART}: Humanoid Loco-Manipulation using Diffusion-guided Augmentation through Relabeling and Tracking},
  author        = {Thiagarajan, Kanish and Debbad, Pranav and Dh{\'e}din, Victor and Omar, Shafeef and Khadiv, Majid},
  journal       = {IEEE Robotics and Automation Letters},
  year          = {2026},
  eprint        = {2606.26855},
  archivePrefix = {arXiv}
}
```

## License

Apache-2.0 (see [LICENSE](LICENSE)). The tracking task and train/play scripts
are derived from [mjlab](https://github.com/mujocolab/mjlab) (Apache-2.0).
