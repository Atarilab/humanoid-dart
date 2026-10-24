from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactData, ContactSensor
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_eef_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  eef_body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Specific reward for tracking end-effector positions relative to the anchor."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  
  if eef_body_names is None:
      eef_body_names = ("left_wrist_yaw_link", "right_wrist_yaw_link")

  body_indexes = _get_body_indexes(command, eef_body_names)
  
  if not body_indexes:
      return torch.zeros(env.num_envs, device=env.device)

  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Cost that returns the number of self-collisions detected by a sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)


def object_global_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_asset_cfg: SceneEntityCfg,
  std: float,
) -> torch.Tensor:
  """Tracks object world position.

  Args:
    object_asset_cfg: SceneEntityCfg identifying the object entity (e.g., "box").
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  box: Entity = env.scene[object_asset_cfg.name]
  target_pos = command.object_pos_w  # (N, 3)
  current_pos = box.data.body_link_pos_w[:, 0]  # (N, 3) root body
  error = torch.sum(torch.square(target_pos - current_pos), dim=-1)
  # breakpoint()
  return torch.exp(-error / (std**2))


def object_global_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_asset_cfg: SceneEntityCfg,
  std: float,
) -> torch.Tensor:
  """Tracks object world orientation using quaternion error magnitude."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  box: Entity = env.scene[object_asset_cfg.name]
  target_quat = command.object_quat_w  # (N, 4)
  current_quat = box.data.body_link_quat_w[:, 0]  # (N, 4)
  error = quat_error_magnitude(target_quat, current_quat) ** 2
  return torch.exp(-error / (std**2))


def combined_object_motion_global_pos_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_asset_cfg: SceneEntityCfg,
  object_pos_std: float,
  motion_pos_std: float,
) -> torch.Tensor:
  """Combined reward term that multiplies object and motion global position/orientation tracking.

  This reward multiplies:
  - Object global position tracking
  - Object global orientation tracking
  - Motion global root position tracking
  - Motion global root orientation tracking

  Args:
    command_name: Name of the motion command.
    object_asset_cfg: SceneEntityCfg identifying the object entity (e.g., "box").
    object_pos_std: Standard deviation for object position error.
    object_ori_std: Standard deviation for object orientation error.
    motion_pos_std: Standard deviation for motion root position error.
    motion_ori_std: Standard deviation for motion root orientation error.
  """
  # Compute individual reward terms
  object_pos_reward = object_global_position_error_exp(
    env, command_name, object_asset_cfg, object_pos_std
  )
  motion_pos_reward = motion_global_anchor_position_error_exp(
    env, command_name, motion_pos_std
  )

  # Multiply all terms together
  return object_pos_reward * motion_pos_reward


def combined_object_motion_global_ori_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_asset_cfg: SceneEntityCfg,
  object_ori_std: float,
  motion_ori_std: float,
) -> torch.Tensor:
  """Combined reward term that multiplies object and motion global position/orientation tracking.

  This reward multiplies:
  - Object global position tracking
  - Object global orientation tracking
  - Motion global root position tracking
  - Motion global root orientation tracking

  Args:
    command_name: Name of the motion command.
    object_asset_cfg: SceneEntityCfg identifying the object entity (e.g., "box").
    object_pos_std: Standard deviation for object position error.
    object_ori_std: Standard deviation for object orientation error.
    motion_pos_std: Standard deviation for motion root position error.
    motion_ori_std: Standard deviation for motion root orientation error.
  """
  # Compute individual reward terms
  object_ori_reward = object_global_orientation_error_exp(
    env, command_name, object_asset_cfg, object_ori_std
  )
  motion_ori_reward = motion_global_anchor_orientation_error_exp(
    env, command_name, motion_ori_std
  )
  # Multiply all terms together
  return object_ori_reward * motion_ori_reward


def eef_contact_indicator_match(
  env: ManagerBasedRlEnv,
  command_name: str,
  eef_body_names: list[str],
  sensor_names: list[str] | None = None,
  gain: float = 1.0,
  force_threshold: float = 10.0,
  force_penalty_std: float = 10.0,
) -> torch.Tensor:
  """Reward for matching contact indicators from command with actual end effector contact.

  This function gives reward when:
  1. The contact indicator from the command is True (ref_object_contact)
  2. AND the corresponding end effector is actually in contact (detected via contact sensors)
  3. AND the contact force is within acceptable limits (penalized if exceeds threshold)

  The reward is multiplied by a force penalty term:
    min(exp((||F_contact||_2 - F_thres) / σ_frc), 1)
  This penalizes forces above the threshold.

  Args:
    command_name: Name of the motion command
    eef_body_names: List of end effector body names (e.g., ["left_wrist_yaw_link", "right_wrist_yaw_link"])
    sensor_names: List of contact sensor names corresponding to each end effector.
                 If None, defaults to ["left_eef_contact", "right_eef_contact"]
    gain: Scaling factor for the reward
    force_threshold: Force threshold in Newtons (default: 10.0)
    force_penalty_std: Standard deviation for force penalty exponential (default: 5.0)

  Returns:
    Reward tensor of shape (num_envs,)
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  # Check if contact data is available
  if command.ref_object_contact is None:
    return torch.zeros(env.num_envs, device=env.device)

  # Get contact indicators from command: (num_envs, num_contacts)
  contact_indicators = command.ref_object_contact  # (num_envs, num_contacts)

  # Default sensor names if not provided
  if sensor_names is None:
    sensor_names = ["left_eef_contact", "right_eef_contact"]

  num_eefs = len(eef_body_names)
  num_contacts = contact_indicators.shape[1]
  num_matches = min(num_contacts, num_eefs, len(sensor_names))

  # Pre-allocate contact detection tensor (GPU-friendly)
  contact_detected = torch.zeros(
    env.num_envs, num_matches, device=env.device, dtype=torch.bool
  )

  # Batch all sensor data access upfront (vectorized, GPU-friendly)
  # Collect all sensor data into a list first
  sensor_data_list = []
  valid_sensor_indices = []

  for i in range(num_matches):
    sensor_name = sensor_names[i]
    if sensor_name in env.scene.sensors:
      sensor_data_list.append(env.scene.sensors[sensor_name].data)
      valid_sensor_indices.append(i)

  if not sensor_data_list:
    # No valid sensors found
    return torch.zeros(env.num_envs, device=env.device)

  # Process all sensors in parallel (vectorized operations)
  # Extract "found" values and force magnitudes from ContactData objects
  # With reduce="netforce" and fields=("found", "force", "pos"):
  # - found: [num_envs, 1] or [num_envs] - contact count
  # - force: [num_envs, 1, 3] or [num_envs, 3] - force vector
  has_contact_all = torch.zeros(
    env.num_envs, num_matches, device=env.device, dtype=torch.bool
  )
  force_magnitudes = torch.zeros(
    env.num_envs, num_matches, device=env.device, dtype=torch.float32
  )

  for idx, sensor_idx in enumerate(valid_sensor_indices):
    sensor_data: ContactData = sensor_data_list[idx]

    # Check if found field is available and not None
    if sensor_data.found is not None:
      # found is [num_envs, N] where N is number of slots/primary matches
      # Check if any slot has contact (found > 0)
      found = sensor_data.found  # [num_envs, N]
      if found.dim() == 2:
        # Multiple slots: check if any slot has contact
        has_contact = (found > 0).any(dim=1)  # [num_envs,] - True if any contact found
        # print(has_contact)
      else:
        # Single dimension: [num_envs]
        has_contact = found > 0  # [num_envs,] - True if contact found
    else:
      has_contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    # Compute force magnitude from force vector
    if sensor_data.force is not None:
      # force is [num_envs, N, 3] where N is number of slots/primary matches
      force = sensor_data.force  # [num_envs, N, 3] or [num_envs, 3]
      if force.dim() == 3:
        # Multiple slots: compute norm for each slot, then take max (most conservative)
        force_norms = torch.norm(force, dim=-1)  # [num_envs, N]
        force_mag = force_norms.max(dim=1)[0]  # [num_envs,] - max force across slots
      elif force.dim() == 2:
        # Single slot: [num_envs, 3]
        force_mag = torch.norm(force, dim=-1)  # [num_envs,]
      else:
        # Unexpected shape
        force_mag = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    else:
      force_mag = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    has_contact_all[:, sensor_idx] = has_contact
    force_magnitudes[:, sensor_idx] = force_mag

  # Vectorized matching: contact indicator AND sensor contact (all at once)
  # contact_indicators: (num_envs, num_contacts)
  # has_contact_all: (num_envs, num_matches)
  # Only match up to num_matches
  contact_detected = (
    contact_indicators[:, :num_matches] & has_contact_all[:, :num_matches]
  )  # (num_envs, num_matches)

  # Compute force penalty term: min(exp((||F||_2 - F_thres) / σ_frc), 1)
  # As per the image formula, but to penalize forces above threshold, we invert:
  # Use exp(-(||F||_2 - F_thres) / σ_frc) so penalty decreases when force exceeds threshold
  # Or implement exactly as image: min(exp((||F||_2 - F_thres) / σ_frc), 1) and use as multiplier
  # For penalizing high forces, we want: penalty = 1 when force <= threshold, penalty < 1 when force > threshold

  # Compute force excess: (||F||_2 - F_thres)
  force_excess = force_magnitudes - force_threshold  # (num_envs, num_matches)

  # Apply formula from image: min(exp((||F||_2 - F_thres) / σ_frc), 1)
  # But to penalize high forces, we use the inverse: exp(-max(0, force_excess) / std)
  # This gives: penalty = 1 when force <= threshold, penalty < 1 when force > threshold
  force_penalty_term = torch.exp(
    -torch.clamp(force_excess, min=0.0) / force_penalty_std
  )  # (num_envs, num_matches)
  force_penalty = torch.clamp(
    force_penalty_term, min=0.0, max=1.0
  )  # Ensure [0, 1] range

  # Apply penalty only where contact is detected, keep 1.0 elsewhere
  force_penalty = torch.where(
    contact_detected, force_penalty, torch.ones_like(force_penalty)
  )  # (num_envs, num_matches)

  # Vectorized reward computation with force penalty
  # Reward = gain * sum(contact_detected * force_penalty) for each environment
  # The force_penalty multiplies the reward, reducing it when forces exceed threshold
  reward = gain * (contact_detected.float() * force_penalty).mean(dim=1)  # (num_envs,)
  # print(reward)
  return reward


def object_height_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_asset_cfg: SceneEntityCfg,
  std: float,
  threshold: float = 0.3,
  sensor_names: list[str] | None = None,
) -> torch.Tensor:
  """Tracks object height (Z) error against the target height (max Z of motion).

  Only applies reward during the last portion of the trajectory (defined by threshold).
  If sensor_names are provided, reward is zero if no contact is detected.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  box: Entity = env.scene[object_asset_cfg.name]

  target_h = command.target_height.squeeze(-1)  # (N,)
  current_h = box.data.body_link_pos_w[:, 0, 2]  # (N,) Z coordinate of root

  error = torch.square(target_h - current_h)
  reward = torch.exp(-error / (std**2))

  # Mask based on trajectory progress
  total_steps = command.motion.time_step_total
  start_step = int(total_steps * (1.0 - threshold))

  mask = (command.time_steps >= start_step).float()


  if sensor_names is None:
    sensor_names = ["left_eef_contact", "right_eef_contact"]
  
  # Filter by contact if sensors provided
  if sensor_names is not None:
    has_contact_any = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for name in sensor_names:
      if name in env.scene.sensors:
        sensor = env.scene.sensors[name]
        sensor_data = sensor.data

        if sensor_data.found is not None:
          found = sensor_data.found
          if found.dim() == 2:
            has_contact = (found > 0).any(dim=1)
          else:
            has_contact = found > 0
        else:
          has_contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

        has_contact_any |= has_contact

    reward = reward * has_contact_any.float()

  return reward * mask
