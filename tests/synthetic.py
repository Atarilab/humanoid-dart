"""Synthetic motions for smoke tests (no real seed data needed)."""

import numpy as np

NUM_JOINTS = 29


def standing_box_trajectory(num_frames: int = 60, fps: float = 50.0, box_x: float = 0.4) -> dict:
  """G1 standing still (zero joint angles) with the box resting in front of it.

  Only the fields the pipeline's diffusion output provides are filled in; the
  rest (all body poses, velocities, contacts) come from regenerate_sim_data.
  """
  body_pos = np.zeros((num_frames, 1, 3))
  body_pos[:, 0, 2] = 0.793
  body_quat = np.zeros((num_frames, 1, 4))
  body_quat[:, 0, 0] = 1.0
  obj_pos = np.tile([box_x, 0.0, 0.17], (num_frames, 1))
  obj_quat = np.tile([1.0, 0.0, 0.0, 0.0], (num_frames, 1))
  return {
    "fps": np.array([fps]),
    "joint_pos": np.zeros((num_frames, NUM_JOINTS)),
    "body_pos_w": body_pos,
    "body_quat_w": body_quat,
    "object_pos_w": obj_pos,
    "object_quat_w": obj_quat,
  }


def batch(trajs: list[dict]) -> dict:
  """Stack per-trajectory dicts into a batched (B, T, ...) motion file dict."""
  out = {}
  for key, val in trajs[0].items():
    if isinstance(val, np.ndarray) and val.ndim >= 1 and all(key in t for t in trajs):
      try:
        out[key] = np.stack([t[key] for t in trajs], axis=0)
      except ValueError:
        pass
  return out
