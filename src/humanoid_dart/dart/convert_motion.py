import os
from typing import Any

import mujoco
import numpy as np
import torch
import tyro
from tqdm import tqdm

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from humanoid_dart.tracking.config.g1.env_cfgs import (
  unitree_g1_flat_tracking_env_cfg,
  unitree_g1_flat_tracking_env_cfg_box,
)
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
  axis_angle_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
)


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
  return q / torch.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)


def _coerce_per_eef_thresholds(
  threshold: float | np.ndarray,
  num_eefs: int,
) -> np.ndarray:
  threshold_arr = np.asarray(threshold, dtype=np.float32).reshape(-1)
  if threshold_arr.size == 1:
    return np.full(num_eefs, float(threshold_arr[0]), dtype=np.float32)
  if threshold_arr.size != num_eefs:
    raise ValueError(
      f"Expected 1 or {num_eefs} contact thresholds, got {threshold_arr.size}"
    )
  return threshold_arr.astype(np.float32, copy=False)


def _normalize_quat_np(quat: np.ndarray | None) -> np.ndarray | None:
  if quat is None:
    return None
  quat_arr = np.asarray(quat, dtype=np.float32)
  norm = float(np.linalg.norm(quat_arr))
  if norm <= 0.0:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
  return quat_arr / norm


def _resolve_contact_box_pose(
  object_position: np.ndarray,
  object_rotation: np.ndarray | None,
  object_geom_local_position: np.ndarray | None = None,
  object_geom_local_rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
  effective_position = np.asarray(object_position, dtype=np.float32)
  effective_rotation = _normalize_quat_np(object_rotation)

  if object_geom_local_position is None and object_geom_local_rotation is None:
    return effective_position, effective_rotation

  geom_local_position = np.zeros(3, dtype=np.float32)
  if object_geom_local_position is not None:
    geom_local_position = np.asarray(object_geom_local_position, dtype=np.float32)

  geom_local_rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
  if object_geom_local_rotation is not None:
    normalized_geom_local_rotation = _normalize_quat_np(object_geom_local_rotation)
    assert normalized_geom_local_rotation is not None
    geom_local_rotation = normalized_geom_local_rotation

  if effective_rotation is None:
    return effective_position + geom_local_position, geom_local_rotation

  object_rotation_t = torch.from_numpy(effective_rotation).unsqueeze(0)
  geom_local_position_t = torch.from_numpy(geom_local_position).unsqueeze(0)
  geom_local_rotation_t = torch.from_numpy(geom_local_rotation).unsqueeze(0)
  effective_position = (
    quat_apply(object_rotation_t, geom_local_position_t)[0].numpy() + effective_position
  )
  effective_rotation = quat_mul(object_rotation_t, geom_local_rotation_t)[0].numpy()
  return effective_position, effective_rotation


def extract_contacts_from_distance(
  eef_positions: np.ndarray,
  object_position: np.ndarray,
  object_rotation: np.ndarray | None = None,
  object_size: np.ndarray | None = None,
  object_geom_local_position: np.ndarray | None = None,
  object_geom_local_rotation: np.ndarray | None = None,
  threshold: float | np.ndarray = 0.05,
) -> dict[str, Any]:
  """Extract contact information using distance threshold with object size consideration.

  This is a simpler fallback method that checks if end effector positions are within
  the object's bounding box (accounting for size) plus a threshold margin.

  Args:
    eef_positions: (num_eefs, 3) array of end effector positions in world frame
    object_position: (3,) array of object position (center) in world frame
    object_rotation: (4,) array of object quaternion (w, x, y, z) in world frame, or None for axis-aligned
    object_size: (3,) array of object half-extents (size in MuJoCo convention), or None to use threshold only
    threshold: Distance threshold in meters for contact detection (default: 0.05m = 5cm).
           May be a scalar applied to all end effectors or one value per end effector.
           If object_size is None, this is used as distance from center.
           If object_size is provided, this is added as margin around the bounding box.

  Returns:
    Dictionary with contact information:
      - contact_positions: (num_eefs, 3) array of contact positions (closest point on object if in contact, NaN otherwise)
      - contact_indicators: (num_eefs,) boolean array indicating if contact exists
  """
  num_eefs = eef_positions.shape[0]
  thresholds = _coerce_per_eef_thresholds(threshold, num_eefs)
  effective_object_position, effective_object_rotation = _resolve_contact_box_pose(
    object_position,
    object_rotation,
    object_geom_local_position=object_geom_local_position,
    object_geom_local_rotation=object_geom_local_rotation,
  )
  contact_indicators = np.zeros(num_eefs, dtype=bool)
  contact_positions = np.full((num_eefs, 3), np.nan, dtype=np.float32)

  # If object size is not provided, use simple distance from center
  if object_size is None:
    for i in range(num_eefs):
      dist = np.linalg.norm(eef_positions[i] - effective_object_position)
      if dist <= thresholds[i]:
        contact_indicators[i] = True
        contact_positions[i] = effective_object_position.copy()
  else:
    # Transform end effector positions to object's local frame
    if effective_object_rotation is not None:
      # Convert quaternion to rotation matrix and transform
      # quat_apply_inverse: transform from world to object frame
      # Ensure float32 dtype for torch operations
      eef_pos_local = quat_apply_inverse(
        torch.from_numpy(effective_object_rotation.astype(np.float32)).unsqueeze(0),
        torch.from_numpy(
          (eef_positions - effective_object_position.reshape(1, 3)).astype(np.float32)
        ),
      ).numpy()
    else:
      # No rotation, just translate
      eef_pos_local = eef_positions - effective_object_position.reshape(1, 3)

    # Check if end effector is within bounding box + threshold
    # object_size is half-extents, so bounds are [-size-threshold, +size+threshold]
    for i in range(num_eefs):
      # Skip if position is NaN (end effector not found)
      if np.isnan(eef_pos_local[i]).any():
        continue

      # Check if within bounding box (with threshold margin)
      bounds = object_size + float(thresholds[i])
      within_bounds = np.all(np.abs(eef_pos_local[i]) <= bounds)

      if within_bounds:
        contact_indicators[i] = True
        # Find closest point on the cube surface
        # Clamp to bounding box bounds
        closest_local = np.clip(eef_pos_local[i], -object_size, object_size)
        # Transform back to world frame
        if effective_object_rotation is not None:
          # Ensure float32 dtype for torch operations
          closest_world = (
            quat_apply(
              torch.from_numpy(effective_object_rotation.astype(np.float32)).unsqueeze(0),
              torch.from_numpy(closest_local.astype(np.float32)).unsqueeze(0),
            ).numpy()[0]
            + effective_object_position
          )
        else:
          closest_world = closest_local + effective_object_position
        contact_positions[i] = closest_world

  return {
    "contact_positions": contact_positions,
    "contact_indicators": contact_indicators,
  }


def _detect_batch_size(data: dict[str, Any]) -> int:
  """Return the batch size B if the NPZ contains (B,T,D) arrays, else 1."""
  for key in ("root_pos", "base_xyz_quat", "dof_pos", "actuator_pos", "x", "qpos"):
    if key in data:
      arr = data[key] if isinstance(data[key], np.ndarray) else np.asarray(data[key])
      if arr.ndim == 3:  # (B, T, D)
        return int(arr.shape[0])
  return 1


def _slice_batch(data: dict[str, Any], idx: int) -> dict[str, np.ndarray]:
  """Extract the *idx*-th trajectory from a possibly-batched NPZ dict.

  For arrays with 3+ dims whose first axis equals the detected batch size,
  slice along dim-0.  Everything else (scalars, 1-D time arrays, etc.) is
  passed through unchanged.
  """
  B = _detect_batch_size(data)
  out: dict[str, np.ndarray] = {}
  for k, v in data.items():
    arr = np.asarray(v)
    if arr.ndim >= 2 and arr.shape[0] == B and B > 1:
      out[k] = arr[idx]
    else:
      out[k] = arr
  return out


def _coerce_time_array(times: np.ndarray, input_file: str) -> np.ndarray:
  """Normalize supported time array layouts to shape (T,)."""
  times = np.asarray(times, dtype=np.float32)
  if times.ndim == 2 and times.shape[1] == 1:
    times = times[:, 0]
  elif times.ndim != 1:
    raise AssertionError(
      f"time must be 1D or (T,1) after batch slicing in {input_file}; got {times.shape}"
    )
  return times


def _coerce_fps_scalar(fps_value: Any, input_file: str) -> float:
  fps_arr = np.asarray(fps_value, dtype=np.float32).reshape(-1)
  if fps_arr.size != 1:
    raise AssertionError(
      f"fps must be scalar in {input_file}; got shape {np.asarray(fps_value).shape}"
    )
  fps = float(fps_arr[0])
  if fps <= 0:
    raise AssertionError(f"fps must be positive in {input_file}; got {fps}")
  return fps


EEF_BODY_NAMES = (
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
)


def _resolve_contact_thresholds(
  default_threshold: float,
  hand_contact_threshold: float | None,
  leg_contact_threshold: float | None,
) -> np.ndarray:
  hand_threshold = default_threshold if hand_contact_threshold is None else hand_contact_threshold
  leg_threshold = default_threshold if leg_contact_threshold is None else leg_contact_threshold
  return np.asarray(
    [hand_threshold, hand_threshold, leg_threshold, leg_threshold],
    dtype=np.float32,
  )


def _resolve_eef_indices(robot: Entity) -> list[int]:
  return [robot.body_names.index(name) if name in robot.body_names else -1 for name in EEF_BODY_NAMES]


def _get_object_geom_props(
  sim: Simulation,
  box: Entity | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
  if box is None:
    return None, None, None

  object_body_names = ["largebox_link", "box", "object"]
  object_body_id = -1
  for body_name in object_body_names:
    object_body_id = mujoco.mj_name2id(
      sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
    )
    if object_body_id >= 0:
      break

  if object_body_id >= 0:
    for geom_id in range(sim.mj_model.ngeom):
      geom_idx = int(geom_id)
      if sim.mj_model.geom_bodyid[geom_idx] != object_body_id:
        continue
      if sim.mj_model.geom_type[geom_idx] == mujoco.mjtGeom.mjGEOM_BOX:
        return (
          sim.mj_model.geom_size[geom_idx].copy(),
          sim.mj_model.geom_pos[geom_idx].copy(),
          sim.mj_model.geom_quat[geom_idx].copy(),
        )

  if hasattr(box, "spec"):
    try:
      geom_ids = box.indexing.geom_ids
      if len(geom_ids) > 0:
        geom_id = geom_ids[0].item() if hasattr(geom_ids[0], "item") else geom_ids[0]
        geom_idx = int(geom_id)
        if sim.mj_model.geom_type[geom_idx] == mujoco.mjtGeom.mjGEOM_BOX:
          return (
            sim.mj_model.geom_size[geom_idx].copy(),
            sim.mj_model.geom_pos[geom_idx].copy(),
            sim.mj_model.geom_quat[geom_idx].copy(),
          )
    except Exception:
      pass

  return None, None, None


def _init_log(output_fps: float, log_object: bool) -> dict[str, Any]:
  log: dict[str, Any] = {
    "fps": [int(round(output_fps))],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  if log_object:
    log["object_pos_w"] = []
    log["object_quat_w"] = []
    log["object_lin_vel_w"] = []
    log["object_ang_vel_w"] = []
    log["contact_positions"] = []
    log["contact_indicators"] = []
  return log


def _expand_to_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
  if tensor.shape[0] == batch_size:
    return tensor.clone()
  if tensor.shape[0] == 1:
    return tensor.repeat(batch_size, *([1] * (tensor.ndim - 1)))
  raise ValueError(f"Expected leading dim 1 or {batch_size}, got {tuple(tensor.shape)}")


def _get_env_origin(scene: Scene, env_idx: int) -> torch.Tensor:
  if scene.env_origins.shape[0] == 1:
    return scene.env_origins[0]
  return scene.env_origins[env_idx]


def _finalize_and_save_log(
  log: dict[str, Any],
  output_file: str,
  robot: Entity,
  eef_indices: list[int],
  contact_method: str,
  contact_threshold: float | np.ndarray,
) -> dict[str, Any]:
  for key in (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    log[key] = np.stack(log[key], axis=0)

  if "object_pos_w" in log and len(log["object_pos_w"]) > 0:
    log["object_pos_w"] = np.stack(log["object_pos_w"], axis=0)
    log["object_quat_w"] = np.stack(log["object_quat_w"], axis=0)
    log["object_lin_vel_w"] = np.stack(log["object_lin_vel_w"], axis=0)
    log["object_ang_vel_w"] = np.stack(log["object_ang_vel_w"], axis=0)

    if len(log["contact_positions"]) > 0:
      log["contact_positions"] = np.stack(log["contact_positions"], axis=0)
      log["contact_indicators"] = np.stack(log["contact_indicators"], axis=0)

      num_frames = log["contact_indicators"].shape[0]
      num_eefs = log["contact_indicators"].shape[1]

      print(f"\n{'=' * 60}")
      print("Contact Extraction Summary")
      print(f"{'=' * 60}")
      print(f"Method: {contact_method}")
      if contact_method == "distance":
        thresholds = _coerce_per_eef_thresholds(contact_threshold, num_eefs)
        print("Distance thresholds:")
        for eef_idx, eef_name in enumerate(EEF_BODY_NAMES[:num_eefs]):
          print(f"  {eef_name}: {float(thresholds[eef_idx]):.3f}m")
      print("\nContact data shape:")
      print(f"  Positions: {log['contact_positions'].shape} (T, num_eefs, 3)")
      print(f"  Indicators: {log['contact_indicators'].shape} (T, num_eefs)")

      if contact_method == "distance" and log["contact_indicators"].sum() == 0:
        print("\nNo contacts detected; sample distances:")
        sample_frames = min(10, num_frames)
        if sample_frames > 0:
          for frame_idx in range(sample_frames):
            for eef_idx, eef_name in enumerate(EEF_BODY_NAMES):
              if eef_idx >= len(eef_indices) or eef_indices[eef_idx] < 0:
                continue
              eef_pos = log["body_pos_w"][frame_idx, eef_indices[eef_idx]]
              obj_pos = log["object_pos_w"][frame_idx]
              dist = np.linalg.norm(eef_pos - obj_pos)
              print(f"  Frame {frame_idx}, {eef_name}: distance = {dist:.4f}m")

      print("\nPer-end-effector contact statistics:")
      for eef_idx, eef_name in enumerate(EEF_BODY_NAMES):
        if eef_idx >= num_eefs:
          continue
        contact_count = log["contact_indicators"][:, eef_idx].sum()
        contact_percentage = 100.0 * contact_count / num_frames if num_frames > 0 else 0.0
        print(f"  {eef_name}:")
        print(f"    Contact frames: {contact_count}/{num_frames} ({contact_percentage:.1f}%)")
        contact_frames = np.where(log["contact_indicators"][:, eef_idx])[0]
        if len(contact_frames) > 0:
          num_samples = min(5, len(contact_frames))
          print(f"    Sample contact positions (first {num_samples}):")
          for frame_idx in contact_frames[:num_samples]:
            pos = log["contact_positions"][frame_idx, eef_idx]
            if not np.isnan(pos).any():
              print(f"      Frame {frame_idx}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
          if len(contact_frames) > num_samples:
            print(f"    ... and {len(contact_frames) - num_samples} more contact frames")

      total_contacts = log["contact_indicators"].sum()
      max_possible = num_frames * num_eefs
      overall_percentage = 100.0 * total_contacts / max_possible if max_possible > 0 else 0.0
      print("\nOverall statistics:")
      print(f"  Total contact detections: {total_contacts}/{max_possible} ({overall_percentage:.1f}%)")
      print(
        f"  Frames with at least one contact: {(log['contact_indicators'].sum(axis=1) > 0).sum()}/{num_frames}"
      )
      print(
        f"  Frames with all four contacts: {(log['contact_indicators'].sum(axis=1) == num_eefs).sum()}/{num_frames}"
      )
      print(f"{'=' * 60}\n")

  os.makedirs(os.path.dirname(output_file), exist_ok=True)
  np.savez(output_file, **log)  # type: ignore[arg-type]
  print(f"\nSaved mjlab motion to: {output_file}")
  print(
    f"Output contains {log['body_pos_w'].shape[0]} frames with {log['body_pos_w'].shape[1]} body links"
  )
  print(f"  All robot bodies: {robot.body_names}")
  return log


class TrajectoryNpzSimLoader:
  def __init__(
    self,
    input_file: str,
    output_fps: int,
    device: torch.device | str,
    repeat_last_frame: int = 0,
    repeat_first_frame: int = 0,
    append_reverse: bool = False,
    preloaded_data: dict[str, Any] | None = None,
  ):
    self.input_file = input_file
    self.output_fps = int(output_fps)
    self.output_dt = 1.0 / float(self.output_fps)
    self.device = device
    self.current_idx = 0
    self.repeat_last_frame = max(0, int(repeat_last_frame))
    self.repeat_first_frame = max(0, int(repeat_first_frame))
    self.append_reverse = bool(append_reverse)

    self._load(preloaded_data)
    self._resample_to_output_fps_using_times()
    self._repeat_first_frame()
    self._repeat_last_frame()
    self._append_reverse()
    self._compute_velocities_from_resampled()

  def _load(self, preloaded_data: dict[str, Any] | None = None) -> None:
    if preloaded_data is not None:
      data = preloaded_data
    else:
      data = dict(np.load(self.input_file, allow_pickle=True))

    # Check if this is Victor format or standard format
    # New key mapping:
    #   root_pos (T,3), root_rot (T,4), dof_pos (T,29)
    #   object_pos (T,3), object_rot (T,4)
    # Legacy key mapping:
    #   base_xyz_quat (T,7), actuator_pos (T,29), obj_0_xyz_quat (T,7)
    # OmniRetarget HF mapping:
    #   qpos (T,36|43) with [quat_wxyz(4), pos_xyz(3), joints(29), obj_quat_wxyz(4), obj_pos_xyz(3)]
    #   fps scalar
    is_new_victor = "root_pos" in data and "dof_pos" in data
    is_legacy_victor = "base_xyz_quat" in data and "actuator_pos" in data
    is_victor_format = is_new_victor or is_legacy_victor
    is_omniretarget_qpos = "qpos" in data and "fps" in data

    if is_victor_format:
      assert "time" in data, f"Key 'time' not found in {self.input_file}"
      times_np = _coerce_time_array(data["time"], self.input_file)

      if is_new_victor:
        # New format: separate pos / rot arrays
        root_pos = data["root_pos"].astype(np.float32)   # (T, 3)
        root_rot = data["root_rot"].astype(np.float32)   # (T, 4) wxyz
        dof_pos = data["dof_pos"].astype(np.float32)     # (T, 29)
        assert root_pos.shape[-1] == 3, f"root_pos last dim must be 3, got {root_pos.shape}"
        assert root_rot.shape[-1] == 4, f"root_rot last dim must be 4, got {root_rot.shape}"
        assert dof_pos.shape[-1] == 29, f"dof_pos last dim must be 29, got {dof_pos.shape}"
      else:
        # Legacy format: combined base_xyz_quat
        base_xyz_quat = data["base_xyz_quat"].astype(np.float32)  # (T, 7)
        assert base_xyz_quat.shape[-1] == 7, (
          f"base_xyz_quat must have 7 columns, got {base_xyz_quat.shape}"
        )
        root_pos = base_xyz_quat[..., 0:3]
        root_rot = base_xyz_quat[..., 3:7]
        dof_pos = data["actuator_pos"].astype(np.float32)  # (T, 29)
        assert dof_pos.shape[-1] == 29, (
          f"actuator_pos must have 29 columns, got {dof_pos.shape}"
        )

      T = times_np.shape[0]
      assert root_pos.shape[0] == T, "root_pos length must match time length"
      assert dof_pos.shape[0] == T, "dof_pos length must match time length"

      self.input_times_np = times_np
      self.input_times = torch.from_numpy(times_np).to(self.device)

      self.motion_base_poss_input = torch.from_numpy(root_pos).to(self.device)
      self.motion_base_rots_input = _normalize_quat(
        torch.from_numpy(root_rot).to(self.device)
      )
      self.motion_dof_poss_input = torch.from_numpy(dof_pos).to(self.device)

      # Check for object data (new keys first, then legacy)
      self.has_object = False
      if "object_pos" in data and "object_rot" in data:
        obj_pos = data["object_pos"].astype(np.float32)  # (T, 3)
        obj_rot = data["object_rot"].astype(np.float32)  # (T, 4)
        assert obj_pos.shape == (T, 3), (
          f"object_pos must be shape ({T}, 3), got {obj_pos.shape}"
        )
        assert obj_rot.shape == (T, 4), (
          f"object_rot must be shape ({T}, 4), got {obj_rot.shape}"
        )
        self.has_object = True
        self.object_pos_input = torch.from_numpy(obj_pos).to(self.device)
        self.object_rots_input = _normalize_quat(
          torch.from_numpy(obj_rot).to(self.device)
        )
      elif "obj_0_xyz_quat" in data:
        obj_xyz_quat = data["obj_0_xyz_quat"].astype(np.float32)  # (T, 7)
        assert obj_xyz_quat.shape == (T, 7), (
          f"obj_0_xyz_quat must be shape ({T}, 7), got {obj_xyz_quat.shape}"
        )
        self.has_object = True
        self.object_pos_input = torch.from_numpy(obj_xyz_quat[:, 0:3]).to(self.device)
        self.object_rots_input = _normalize_quat(
          torch.from_numpy(obj_xyz_quat[:, 3:7]).to(self.device)
        )
    elif is_omniretarget_qpos:
      qpos = np.asarray(data["qpos"], dtype=np.float32)
      assert qpos.ndim == 2 and qpos.shape[1] in (36, 43), (
        f"qpos must be (T,36) or (T,43); got {qpos.shape}"
      )

      fps = _coerce_fps_scalar(data["fps"], self.input_file)
      T = qpos.shape[0]
      times_np = np.arange(T, dtype=np.float32) / fps

      self.input_times_np = times_np
      self.input_times = torch.from_numpy(times_np).to(self.device)

      # OmniRetarget stores quaternion before position; mjlab expects them split.
      self.motion_base_rots_input = _normalize_quat(
        torch.from_numpy(qpos[:, 0:4]).to(self.device)
      )
      self.motion_base_poss_input = torch.from_numpy(qpos[:, 4:7]).to(self.device)
      self.motion_dof_poss_input = torch.from_numpy(qpos[:, 7:36]).to(self.device)

      self.has_object = qpos.shape[1] == 43
      if self.has_object:
        self.object_rots_input = _normalize_quat(
          torch.from_numpy(qpos[:, 36:40]).to(self.device)
        )
        self.object_pos_input = torch.from_numpy(qpos[:, 40:43]).to(self.device)
    else:
      # Standard format: single 'x' array
      assert "x" in data, f"Key 'x' not found in {self.input_file}"
      x_np = data["x"]
      assert x_np.ndim == 2 and x_np.shape[1] in (71, 84), (
        f"x must be (T,71) or (T,84); got {x_np.shape}"
      )

      assert "time" in data, f"Key 'time' not found in {self.input_file}"
      times_np = _coerce_time_array(data["time"], self.input_file)
      assert times_np.shape[0] == x_np.shape[0], (
        "times must be 1D and match x length"
      )

      self.input_times_np = times_np
      self.input_times = torch.from_numpy(times_np).to(self.device)

      T = x_np.shape[0]
      rs = torch.from_numpy(x_np.astype(np.float32)).to(self.device)

      # Use only qpos portions: [pos(3), quat wxyz(4), joint pos(29)]
      # New format (T,84) has x = (qpos(43), qvel(41)). Legacy (T,71) keeps only qpos fields.
      self.motion_base_poss_input = rs[:, 0:3]
      self.motion_base_rots_input = _normalize_quat(rs[:, 3:7])
      self.motion_dof_poss_input = rs[:, 7:36]

      # Optional object
      # Priority: if x has embedded object in qpos (shape 84), use that; otherwise fallback to separate 'object_states'
      self.has_object = False
      if x_np.shape[1] == 84:
        # x = (qpos(43), qvel(41)) where:
        # qpos[0:7] - first 7 entries (base pos(3) + base quat(4))
        # qpos[7:36] - next 29 entries (joint positions)
        # qpos[36:40] = obj_quat_wxyz (4 values: w, x, y, z)
        # qpos[40:43] = obj_pos (3 values: x, y, z)
        self.has_object = True
        self.object_rots_input = _normalize_quat(rs[:, 36:40])
        self.object_pos_input = rs[:, 40:43]
        assert self.object_rots_input.shape == (T, 4), (
          f"object_rots_input shape mismatch: expected ({T}, 4), got {self.object_rots_input.shape}"
        )
        assert self.object_pos_input.shape == (T, 3), (
          f"object_pos_input shape mismatch: expected ({T}, 3), got {self.object_pos_input.shape}"
        )
      elif "object_states" in data:
        obj_np = data["object_states"].astype(np.float32)
        assert obj_np.ndim == 2 and obj_np.shape[0] == T and obj_np.shape[1] == 7, (
          "object_states must be shape (T,7)"
        )
        self.has_object = True
        self.object_pos_input = torch.from_numpy(obj_np[:, 0:3]).to(self.device)
        self.object_rots_input = _normalize_quat(
          torch.from_numpy(obj_np[:, 3:7]).to(self.device)
        )

    T = len(self.input_times_np)
    self.input_frames = int(T)
    self.duration = (
      float(self.input_times_np[-1] - self.input_times_np[0]) if T > 1 else 0.0
    )
    self.input_dt = float(self.duration / max(1, T - 1)) if T > 1 else self.output_dt
    self.input_fps = int(round(1.0 / max(1e-8, self.input_dt)))

  def _resample_to_output_fps_using_times(self) -> None:
    if self.input_frames <= 1:
      self.motion_base_poss = self.motion_base_poss_input
      self.motion_base_rots = self.motion_base_rots_input
      self.motion_dof_poss = self.motion_dof_poss_input
      if self.has_object:
        self.object_poss = self.object_pos_input
        self.object_rots = self.object_rots_input
      self.output_frames = self.input_frames
      return

    t0 = float(self.input_times_np[0])
    duration = float(self.input_times_np[-1] - self.input_times_np[0])
    if duration <= 0.0:
      self.motion_base_poss = self.motion_base_poss_input
      self.motion_base_rots = self.motion_base_rots_input
      self.motion_dof_poss = self.motion_dof_poss_input
      if self.has_object:
        self.object_poss = self.object_pos_input
        self.object_rots = self.object_rots_input
      self.output_frames = self.input_frames
      return

    times_out = np.arange(0.0, duration + 1e-8, self.output_dt, dtype=np.float32) + t0
    self.output_frames = int(times_out.shape[0])

    idx1 = np.searchsorted(self.input_times_np, times_out, side="right")
    idx1 = np.clip(idx1, 1, self.input_frames - 1)
    idx0 = idx1 - 1
    t0s = self.input_times_np[idx0]
    t1s = self.input_times_np[idx1]
    denom = np.maximum(1e-8, (t1s - t0s))
    blend_np = (times_out - t0s) / denom

    index_0 = torch.from_numpy(idx0.astype(np.int64)).to(self.device)
    index_1 = torch.from_numpy(idx1.astype(np.int64)).to(self.device)
    blend = torch.from_numpy(blend_np.astype(np.float32)).to(self.device)

    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    q0 = self.motion_base_rots_input[index_0]
    q1 = self.motion_base_rots_input[index_1]
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    q = q0 * (1 - blend.unsqueeze(1)) + q1 * blend.unsqueeze(1)
    self.motion_base_rots = _normalize_quat(q)
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )

    if self.has_object:
      self.object_poss = self._lerp(
        self.object_pos_input[index_0],
        self.object_pos_input[index_1],
        blend.unsqueeze(1),
      )
      oq0 = self.object_rots_input[index_0]
      oq1 = self.object_rots_input[index_1]
      odot = (oq0 * oq1).sum(-1, keepdim=True)
      oq1 = torch.where(odot < 0, -oq1, oq1)
      oq = oq0 * (1 - blend.unsqueeze(1)) + oq1 * blend.unsqueeze(1)
      self.object_rots = _normalize_quat(oq)

  def _repeat_first_frame(self) -> None:
    """Prepend the first frame N times to extend the trajectory."""
    if self.repeat_first_frame <= 0:
      return

    first_pos = self.motion_base_poss[:1].repeat(self.repeat_first_frame, 1)
    first_rot = self.motion_base_rots[:1].repeat(self.repeat_first_frame, 1)
    first_dof = self.motion_dof_poss[:1].repeat(self.repeat_first_frame, 1)

    self.motion_base_poss = torch.cat([first_pos, self.motion_base_poss], dim=0)
    self.motion_base_rots = torch.cat([first_rot, self.motion_base_rots], dim=0)
    self.motion_dof_poss = torch.cat([first_dof, self.motion_dof_poss], dim=0)

    if self.has_object:
      first_obj_pos = self.object_poss[:1].repeat(self.repeat_first_frame, 1)
      first_obj_rot = self.object_rots[:1].repeat(self.repeat_first_frame, 1)
      self.object_poss = torch.cat([first_obj_pos, self.object_poss], dim=0)
      self.object_rots = torch.cat([first_obj_rot, self.object_rots], dim=0)

    self.output_frames += self.repeat_first_frame

  def _repeat_last_frame(self) -> None:
    """Append the last frame N times to extend the trajectory."""
    if self.repeat_last_frame <= 0:
      return

    last_pos = self.motion_base_poss[-1:].repeat(self.repeat_last_frame, 1)
    last_rot = self.motion_base_rots[-1:].repeat(self.repeat_last_frame, 1)
    last_dof = self.motion_dof_poss[-1:].repeat(self.repeat_last_frame, 1)

    self.motion_base_poss = torch.cat([self.motion_base_poss, last_pos], dim=0)
    self.motion_base_rots = torch.cat([self.motion_base_rots, last_rot], dim=0)
    self.motion_dof_poss = torch.cat([self.motion_dof_poss, last_dof], dim=0)

    if self.has_object:
      last_obj_pos = self.object_poss[-1:].repeat(self.repeat_last_frame, 1)
      last_obj_rot = self.object_rots[-1:].repeat(self.repeat_last_frame, 1)
      self.object_poss = torch.cat([self.object_poss, last_obj_pos], dim=0)
      self.object_rots = torch.cat([self.object_rots, last_obj_rot], dim=0)

    self.output_frames += self.repeat_last_frame

  def _append_reverse(self) -> None:
    """Append the reversed motion trajectory at the end."""
    if not self.append_reverse:
      return

    # Store original frame count before appending
    original_frames = self.output_frames

    # Reverse all motion data along the time dimension (dim=0)
    reversed_pos = torch.flip(self.motion_base_poss, dims=[0])
    reversed_rot = torch.flip(self.motion_base_rots, dims=[0])
    reversed_dof = torch.flip(self.motion_dof_poss, dims=[0])

    self.motion_base_poss = torch.cat([self.motion_base_poss, reversed_pos], dim=0)
    self.motion_base_rots = torch.cat([self.motion_base_rots, reversed_rot], dim=0)
    self.motion_dof_poss = torch.cat([self.motion_dof_poss, reversed_dof], dim=0)

    if self.has_object:
      reversed_obj_pos = torch.flip(self.object_poss, dims=[0])
      reversed_obj_rot = torch.flip(self.object_rots, dims=[0])
      self.object_poss = torch.cat([self.object_poss, reversed_obj_pos], dim=0)
      self.object_rots = torch.cat([self.object_rots, reversed_obj_rot], dim=0)

    self.output_frames += original_frames

  def _compute_velocities_from_resampled(self) -> None:
    # lin vel from pos derivative and ang vel from quaternion finite difference
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]

    q = self.motion_base_rots
    if q.shape[0] >= 3:
      q_prev, q_next = q[:-2], q[2:]
      q_rel = quat_mul(q_next, quat_conjugate(q_prev))
      omega = axis_angle_from_quat(q_rel) / (2.0 * self.output_dt)
      omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
    else:
      omega = torch.zeros_like(self.motion_base_poss)
    self.motion_base_ang_vels = omega

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    return a * (1 - blend) + b * blend

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ],
    bool,
  ]:
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= getattr(self, "output_frames", self.input_frames):
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def _convert_batched_trajectories(
  motions: list[TrajectoryNpzSimLoader],
  output_files: list[str],
  output_fps: float,
  device: str,
  sim: Simulation,
  sim_cfg: SimulationCfg,
  scene: Scene,
  robot: Entity,
  box: Entity | None,
  extract_object_states: bool,
  contact_method: str,
  contact_threshold: float | np.ndarray,
) -> list[dict[str, Any]]:
  """Run one batched sim pass and write one output file per trajectory."""
  batch_size = len(motions)
  if batch_size == 0:
    return []

  eef_indices = _resolve_eef_indices(robot)
  contact_thresholds = _coerce_per_eef_thresholds(contact_threshold, len(eef_indices))
  log_object_flags = [
    extract_object_states
    and getattr(motion, "has_object", False)
    and (box is not None and not box.data.is_fixed_base)
    for motion in motions
  ]
  logs = [_init_log(output_fps, log_object) for log_object in log_object_flags]
  frames_per_motion = [getattr(motion, "output_frames", motion.input_frames) for motion in motions]
  max_frames = max(frames_per_motion)

  object_size = None
  object_geom_local_position = None
  object_geom_local_rotation = None
  if any(log_object_flags):
    (
      object_size,
      object_geom_local_position,
      object_geom_local_rotation,
    ) = _get_object_geom_props(sim, box)
  if any(log_object_flags):
    print("[Contact Detection] Distance thresholds:")
    for eef_name, threshold_value in zip(EEF_BODY_NAMES, contact_thresholds, strict=True):
      print(f"  {eef_name}: {float(threshold_value):.3f}m")
    if object_size is not None:
      print(f"[Contact Detection] Object size (half-extents): {object_size}")
      for eef_name, threshold_value in zip(EEF_BODY_NAMES, contact_thresholds, strict=True):
        print(
          f"[Contact Detection] Effective bounding box for {eef_name}: ±{object_size + float(threshold_value)}"
        )
    else:
      print(
        "[Contact Detection] Warning: Object size not found, using simple distance thresholds from center"
      )

  default_box_pose = None
  if box is not None and any(log_object_flags):
    default_box_pose = box.data.default_root_state[:, :7].clone()

  prev_obj_pos_motion: list[torch.Tensor | None] = [None] * batch_size
  prev_obj_rot_motion: list[torch.Tensor | None] = [None] * batch_size
  current_obj_lin_vels: list[np.ndarray | None] = [None] * batch_size
  current_obj_ang_vels: list[np.ndarray | None] = [None] * batch_size

  scene.reset()
  pbar = tqdm(total=max_frames, desc="Converting batch", unit="frame", ncols=100)

  for frame_idx in range(max_frames):
    root_states = _expand_to_batch(robot.data.default_root_state, batch_size)
    joint_pos = _expand_to_batch(robot.data.default_joint_pos, batch_size)
    joint_vel = _expand_to_batch(robot.data.default_joint_vel, batch_size)
    obj_pose = (
      _expand_to_batch(default_box_pose, batch_size)
      if default_box_pose is not None
      else None
    )

    for env_idx, motion in enumerate(motions):
      sample_idx = min(frame_idx, frames_per_motion[env_idx] - 1)

      motion_base_pos = motion.motion_base_poss[sample_idx]
      motion_base_rot = motion.motion_base_rots[sample_idx]
      motion_base_lin_vel = motion.motion_base_lin_vels[sample_idx]
      motion_base_ang_vel = motion.motion_base_ang_vels[sample_idx]
      motion_dof_pos = motion.motion_dof_poss[sample_idx]
      motion_dof_vel = motion.motion_dof_vels[sample_idx]

      env_origin = _get_env_origin(scene, env_idx)
      root_states[env_idx, 0:3] = motion_base_pos
      root_states[env_idx, 0:2] += env_origin[0:2]
      root_states[env_idx, 3:7] = motion_base_rot
      root_states[env_idx, 7:10] = motion_base_lin_vel
      root_states[env_idx, 10:13] = quat_apply_inverse(
        motion_base_rot.unsqueeze(0), motion_base_ang_vel.unsqueeze(0)
      )[0]

      joint_pos[env_idx, :] = motion_dof_pos
      joint_vel[env_idx, :] = motion_dof_vel

      if log_object_flags[env_idx]:
        if not hasattr(motion, "object_poss") or not hasattr(motion, "object_rots"):
          raise RuntimeError(
            "Object data not found: object_poss or object_rots missing. This should not happen if has_object is True."
          )

        curr_obj_pos_motion = motion.object_poss[sample_idx].clone()
        curr_obj_rot_motion = motion.object_rots[sample_idx].clone()
        prev_pos = prev_obj_pos_motion[env_idx]
        prev_rot = prev_obj_rot_motion[env_idx]

        if prev_pos is not None and prev_rot is not None:
          obj_lin_vel_motion = (
            curr_obj_pos_motion - prev_pos
          ) / sim_cfg.mujoco.timestep
          q_rel = quat_mul(
            curr_obj_rot_motion.unsqueeze(0),
            quat_conjugate(prev_rot.unsqueeze(0)),
          )[0]
          obj_ang_vel_motion = axis_angle_from_quat(q_rel.unsqueeze(0))[0] / sim_cfg.mujoco.timestep
        else:
          obj_lin_vel_motion = torch.zeros(3, device=device, dtype=torch.float32)
          obj_ang_vel_motion = torch.zeros(3, device=device, dtype=torch.float32)

        current_obj_lin_vels[env_idx] = obj_lin_vel_motion.cpu().numpy().copy()
        current_obj_ang_vels[env_idx] = obj_ang_vel_motion.cpu().numpy().copy()
        prev_obj_pos_motion[env_idx] = curr_obj_pos_motion.clone()
        prev_obj_rot_motion[env_idx] = curr_obj_rot_motion.clone()

        if obj_pose is not None:
          obj_pose[env_idx, 0:3] = curr_obj_pos_motion
          obj_pose[env_idx, 0:2] += env_origin[0:2]
          obj_pose[env_idx, 3:7] = curr_obj_rot_motion

    robot.write_root_state_to_sim(root_states)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    if obj_pose is not None and box is not None:
      box.write_root_link_pose_to_sim(obj_pose)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    for env_idx, log in enumerate(logs):
      if frame_idx >= frames_per_motion[env_idx]:
        continue

      log["joint_pos"].append(robot.data.joint_pos[env_idx, :].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[env_idx, :].cpu().numpy().copy())

      body_pos_w = robot.data.body_link_pos_w[env_idx, :].cpu().numpy().copy()
      log["body_pos_w"].append(body_pos_w)
      log["body_quat_w"].append(
        robot.data.body_link_quat_w[env_idx, :].cpu().numpy().copy()
      )
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[env_idx, :].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[env_idx, :].cpu().numpy().copy()
      )

      if not log_object_flags[env_idx]:
        continue

      assert box is not None
      curr_obj_pos = box.data.body_link_pos_w[env_idx, 0].cpu().numpy().copy()
      curr_obj_rot = box.data.body_link_quat_w[env_idx, 0].cpu().numpy().copy()
      obj_lin_vel = current_obj_lin_vels[env_idx]
      obj_ang_vel = current_obj_ang_vels[env_idx]
      assert obj_lin_vel is not None
      assert obj_ang_vel is not None
      log["object_pos_w"].append(curr_obj_pos)
      log["object_quat_w"].append(curr_obj_rot)
      log["object_lin_vel_w"].append(obj_lin_vel)
      log["object_ang_vel_w"].append(obj_ang_vel)

      eef_positions = np.array(
        [
          body_pos_w[eef_body_idx] if eef_body_idx >= 0 else np.array([np.nan, np.nan, np.nan])
          for eef_body_idx in eef_indices
        ]
      )
      contact_data = extract_contacts_from_distance(
        eef_positions,
        curr_obj_pos,
        object_rotation=curr_obj_rot,
        object_size=object_size,
        object_geom_local_position=object_geom_local_position,
        object_geom_local_rotation=object_geom_local_rotation,
        threshold=contact_thresholds,
      )
      log["contact_positions"].append(contact_data["contact_positions"])
      log["contact_indicators"].append(contact_data["contact_indicators"])

    pbar.update(1)

  pbar.close()

  return [
    _finalize_and_save_log(
      log=log,
      output_file=output_file,
      robot=robot,
      eef_indices=eef_indices,
      contact_method=contact_method,
      contact_threshold=contact_thresholds,
    )
    for log, output_file in zip(logs, output_files, strict=True)
  ]


def convert(
  input_file: str,
  output_dir: str,
  output_name: str,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  repeat_last_frame: int = 0,
  repeat_first_frame: int = 0,
  append_reverse: bool = False,
  extract_object_states: bool = True,
  contact_method: str = "distance",
  contact_threshold: float = 0.05,
  hand_contact_threshold: float | None = None,
  leg_contact_threshold: float | None = None,
):
  """Convert dataset NPZ (qpos in x) to mjlab motion.npz using simulation states.

  Extracts data for all robot bodies to match the expected mjlab motion format.
  Supports both Victor format (separate arrays) and standard format (x array).
  If the input NPZ contains a batch of trajectories (arrays with shape (B,T,D)),
  all trajectories are replayed together in one batched simulation pass and then
  saved independently as motion_000.npz, etc.

  When object data is present, also extracts contact information for end effectors.

  Args:
    input_file: Path to input NPZ file
    output_dir: Base output directory
    output_name: Name for the output (used to create <output_dir>/<output_name>/...)
    output_fps: Output frame rate
    device: Device to use for simulation
    repeat_last_frame: Number of times to repeat the last frame
    repeat_first_frame: Number of times to repeat the first frame
    append_reverse: If True, append the reversed motion trajectory at the end
    extract_object_states: If False, skip extracting object states and contact information
    contact_method: Method to extract contacts - "mujoco" (preferred, uses physics) or "distance" (uses threshold)
    contact_threshold: Default distance threshold in meters for "distance" contacts
    hand_contact_threshold: Optional override for the two wrist contact thresholds
    leg_contact_threshold: Optional override for the two ankle contact thresholds
  """
  effective_contact_threshold = _resolve_contact_thresholds(
    contact_threshold,
    hand_contact_threshold,
    leg_contact_threshold,
  )

  # ---- Load NPZ once and detect batch ----
  raw_data = dict(np.load(input_file, allow_pickle=True))
  batch_size = _detect_batch_size(raw_data)
  print(f"[INFO] Detected batch size: {batch_size}")

  # ---- Build per-trajectory loaders once ----
  motions: list[TrajectoryNpzSimLoader] = []
  output_files: list[str] = []
  for b_idx in range(batch_size):
    single_data = _slice_batch(raw_data, b_idx) if batch_size > 1 else raw_data
    file_name = f"motion_{b_idx:03d}.npz" if batch_size > 1 else "motion.npz"
    out_file = os.path.join(output_dir, output_name, file_name)

    if batch_size > 1:
      print(f"\n{'─' * 60}")
      print(f"[Batch {b_idx + 1}/{batch_size}] Preparing loader")

    motions.append(
      TrajectoryNpzSimLoader(
        input_file=input_file,
        output_fps=int(round(output_fps)),
        device=device,
        repeat_last_frame=repeat_last_frame,
        repeat_first_frame=repeat_first_frame,
        append_reverse=append_reverse,
        preloaded_data=single_data,
      )
    )
    output_files.append(out_file)

  # ---- Set up one batched simulation (expensive) ----
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / float(output_fps)
  if extract_object_states:
    scene = Scene(unitree_g1_flat_tracking_env_cfg_box().scene, device=device)
  else:
    scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=batch_size, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot: Entity = scene["robot"]
  box: Entity | None = scene.entities.get("box") if hasattr(scene, "entities") else None
  print(f"Robot has {len(robot.body_names)} bodies: {robot.body_names}")

  _convert_batched_trajectories(
      motions=motions,
      output_files=output_files,
      output_fps=output_fps,
      device=device,
      sim=sim,
      sim_cfg=sim_cfg,
      scene=scene,
      robot=robot,
      box=box,
      extract_object_states=extract_object_states,
      contact_method=contact_method,
        contact_threshold=effective_contact_threshold,
  )


def main(
  input_file: str,
  output_name: str,
  repeat_last_frame: int = 0,
  repeat_first_frame: int = 0,
  append_reverse: bool = False,
  extract_object_states: bool = True,
  output_fps: float = 50.0,
  contact_threshold: float = 0.05,
  hand_contact_threshold: float | None = None,
  leg_contact_threshold: float | None = None,
  contact_method: str = "distance",
  output_dir: str = "motions/",
  device: str = "cuda:0",
):
  """Convert trajectory NPZ file to mjlab motion format with contact extraction.

  Args:
    input_file: Path to input NPZ file
    output_dir: Base output directory
    output_name: Name for the output (used to create <output_dir>/<output_name>/motion.npz)
    output_fps: Output frame rate
    device: Device to use for simulation
    repeat_last_frame: Number of times to repeat the last frame
    repeat_first_frame: Number of times to repeat the first frame
    append_reverse: If True, append the reversed motion trajectory at the end
    extract_object_states: If False, skip extracting object states and contact information
    contact_method: Method to extract contacts - "mujoco" (preferred) or "distance"
    contact_threshold: Default distance threshold in meters for "distance" method
    hand_contact_threshold: Optional override for the two wrist contact thresholds
    leg_contact_threshold: Optional override for the two ankle contact thresholds

  Example usage:
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run python -m humanoid_dart.dart.convert_motion
    --input-file path/to/raw_trajectories.npz
    --output-name pick_place_raw
    --output-fps 70
    --repeat-last-frame 150 --repeat-first-frame 100
  """
  convert(
    input_file=input_file,
    output_dir=output_dir,
    output_name=output_name,
    output_fps=output_fps,
    device=device,
    repeat_last_frame=repeat_last_frame,
    repeat_first_frame=repeat_first_frame,
    append_reverse=append_reverse,
    extract_object_states=extract_object_states,
    contact_method=contact_method,
    contact_threshold=contact_threshold,
    hand_contact_threshold=hand_contact_threshold,
    leg_contact_threshold=leg_contact_threshold,
  )


if __name__ == "__main__":
  tyro.cli(main)
