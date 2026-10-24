from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def object_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.object_pos_w,
    command.object_quat_w,
  )
  # breakpoint()
  return pos.view(env.num_envs, -1)


def object_lin_vel_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  object = env.scene[asset_cfg.name]
  return object.data.body_link_lin_vel_w[:, 0].view(env.num_envs, -1)


def object_ang_vel_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  object = env.scene[asset_cfg.name]
  return object.data.body_link_ang_vel_w[:, 0].view(env.num_envs, -1)


def object_position_error(
  env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Compute the error between actual and desired object position.

  This is crucial for box pushing - the robot needs to know how far the box
  is from where it should be to learn to push it correctly.
  """

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  box: Entity = env.scene[asset_cfg.name]

  # Desired object position from motion data
  desired_pos = command.object_pos_w  # (N, 3)

  # Actual object position from simulation
  actual_pos = box.data.body_link_pos_w[:, 0]  # (N, 3)
  # Position error: actual - desired
  error = actual_pos - desired_pos
  return error


def object_orientation_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Compute the error between actual and desired object orientation."""

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  box: Entity = env.scene[asset_cfg.name]

  _, ori = subtract_frame_transforms(
    command.object_pos_w,
    command.object_quat_w,
    box.data.body_link_pos_w[:, 0],
    box.data.body_link_quat_w[:, 0],
  )
  mat = matrix_from_quat(ori)
  # breakpoint()
  return mat[..., :2].reshape(mat.shape[0], -1)


def target_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.target_height

def future_commands(
  env: ManagerBasedRlEnv,
  command_name: str,
  horizon: int = 5,
  stride: int = 2,
  step_offsets: list[int] | None = None,
) -> torch.Tensor:
  """Get future commands from the motion command."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.get_future_commands(
    horizon=horizon, stride=stride, step_offsets=step_offsets
  )
