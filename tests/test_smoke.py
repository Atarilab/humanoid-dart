"""CPU smoke tests with synthetic motions (no seed data or GPU required)."""

import importlib
import pkgutil
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from synthetic import batch, standing_box_trajectory  # noqa: E402

TASK = "Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation"
REPO = Path(__file__).parents[1]
SEED = REPO / "motions" / "pick_place.npz"
CHECKPOINT = REPO / "checkpoints" / "model_3999.pt"


@pytest.fixture(scope="module")
def motion_file(tmp_path_factory) -> Path:
  from humanoid_dart.dart.utils.trajectory_utils import regenerate_sim_data

  trajs = regenerate_sim_data(
    [standing_box_trajectory(box_x=0.4), standing_box_trajectory(box_x=0.5)],
    device="cpu",
  )
  path = tmp_path_factory.mktemp("motion") / "base_motions.npz"
  np.savez(path, **batch(trajs))
  return path


def test_all_modules_import():
  import humanoid_dart

  for mod in pkgutil.walk_packages(humanoid_dart.__path__, "humanoid_dart."):
    importlib.import_module(mod.name)


def test_box_tasks_registered():
  import humanoid_dart.tracking  # noqa: F401
  from mjlab.tasks.registry import list_tasks

  assert TASK in list_tasks()
  assert "Mjlab-Tracking-Flat-Unitree-G1-Box" in list_tasks()


def test_g1_has_no_hands():
  from humanoid_dart.assets.g1 import get_g1_no_hands_spec

  model = get_g1_no_hands_spec().compile()

  def names(obj, n):
    return {mujoco.mj_id2name(model, obj, i) for i in range(n)}

  geoms = names(mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
  assert {"left_wrist_collision", "right_wrist_collision"} <= geoms
  assert not {"left_hand_collision", "right_hand_collision"} & geoms
  assert not {"left_palm", "right_palm"} & names(mujoco.mjtObj.mjOBJ_SITE, model.nsite)
  assert not {"left_rubber_hand", "right_rubber_hand"} & names(
    mujoco.mjtObj.mjOBJ_MESH, model.nmesh
  )


def test_seed_motions_load():
  from humanoid_dart.dart.utils.trajectory_utils import load_base_motions

  data, _ = load_base_motions(SEED)
  assert len(data) == 87
  for key in ("joint_pos", "body_pos_w", "object_pos_w", "contact_indicators"):
    assert all(key in d for d in data)
  assert data[0]["joint_pos"].shape[1] == 29


def test_motion_file_schema(motion_file):
  data = np.load(motion_file)
  assert data["joint_pos"].shape == (2, 60, 29)
  assert data["body_pos_w"].shape == (2, 60, 30, 3)
  assert data["contact_indicators"].shape == (2, 60, 2)


def test_tracking_env_steps(motion_file):
  import humanoid_dart.tracking  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg

  from humanoid_dart.envs import DartRlEnv

  cfg = load_env_cfg(TASK, play=True)
  cfg.commands["motion"].motion_file = str(motion_file)
  cfg.scene.num_envs = 2
  env = DartRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset()
    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    for _ in range(5):
      obs, reward, *_ = env.step(action)
    assert torch.isfinite(reward).all()
    assert torch.isfinite(obs["policy"]).all()
  finally:
    env.close()


def test_planner_fit_and_generate(tmp_path, motion_file):
  from humanoid_dart.dart.run_evolutionary_pipeline import (
    DEFAULT_PLANNER_CONFIG,
    _write_run_planner_config,
  )
  from humanoid_dart.dart.utils.trajectory_utils import load_base_motions
  from humanoid_dart.planner.motion_generator import MotionGenerator

  data, _ = load_base_motions(motion_file)
  cfg_path = _write_run_planner_config(DEFAULT_PLANNER_CONFIG, tmp_path)
  mg = MotionGenerator(config_path=str(cfg_path), device="cpu")
  mg.data_cfg["batch_size"] = 16
  mg.fit(data_source=data, epochs=1, save_path=str(tmp_path / "planner.pt"), calc_stats=False)
  assert list(tmp_path.rglob("norm_stats.npz")), "norm stats must be written inside the run dir"

  H = mg.data_cfg["state_history"]
  d = data[0]
  robot = np.concatenate([d["body_pos_w"][:H, 0], d["body_quat_w"][:H, 0], d["joint_pos"][:H]], -1)
  obj = np.concatenate([d["object_pos_w"][:H], d["object_quat_w"][:H]], -1)
  out, lengths = mg.generate_trajectory(
    initial_condition={"robot": robot[None], "obj": obj[None]},
    goal_condition=np.array([[0.5, -0.3]]),
    num_samples=1,
    target_traj_length=60,
    use_last_frame_wp=False,
    enable_goal_stop=False,
    end_error_threshold=0.25,
    cfg_w=2.0,
  )
  assert out.shape == (1, 60, 43)
  assert np.isfinite(out).all()


@pytest.mark.slow
def test_pipeline_end_to_end(tmp_path, motion_file, monkeypatch):
  from humanoid_dart.dart import run_evolutionary_pipeline

  cfg = yaml.safe_load((REPO / "configs" / "pick_place.yaml").read_text())
  cfg.update(
    base_motion=str(motion_file),
    checkpoint_file=str(CHECKPOINT),
    out_dir=str(tmp_path / "run"),
    device="cpu",
    num_samples=4,
    iterations=1,
    generations=1,
  )
  cfg["training"].update(num_envs=4, initial_max_iterations=1, elite_max_iterations=1)
  cfg["diffusion"].update(initial_epochs=1, train_epochs=1, reset_epochs=1, chunk_size=4)
  cfg["evolution"]["max_traj_length"] = 60
  cfg_file = tmp_path / "smoke.yaml"
  cfg_file.write_text(yaml.safe_dump(cfg))

  monkeypatch.chdir(tmp_path)  # tracker logs go to ./logs
  monkeypatch.setattr(sys, "argv", ["dart-run", "--config", str(cfg_file)])
  run_evolutionary_pipeline.main()

  gen = tmp_path / "run" / "gen_000"
  assert (gen / "elites.json").exists()
  assert list((tmp_path / "logs" / "rsl_rl").rglob("model_*.pt"))
