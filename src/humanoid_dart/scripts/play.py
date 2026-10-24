"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

import humanoid_dart.tracking.mdp as mdp
from humanoid_dart.envs import DartRlEnv
from humanoid_dart.utils.checkpoint import smart_load_weights
from mjlab.managers.manager_term_config import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from humanoid_dart.tracking.mdp import MotionCommandCfg
from humanoid_dart.tracking.rl import MotionTrackingOnPolicyRunner
from humanoid_dart.tracking.rl.exporter import (
  attach_onnx_metadata as attach_tracking_onnx_metadata,
)
from humanoid_dart.tracking.rl.exporter import (
  export_motion_policy_as_onnx,
)
from mjlab.tasks.velocity.rl.exporter import (
  attach_onnx_metadata as attach_velocity_onnx_metadata,
)
from mjlab.third_party.isaaclab.isaaclab_rl.rsl_rl.exporter import export_policy_as_onnx
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
  quat_apply,
  quat_apply_inverse,
  quat_mul,
)


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser", "headless"] = "auto"
  max_steps: int | None = None
  task_params: list[float] | None = None
  env_spacing: float | None = None
  
  # Camera config overrides
  camera_lookat: tuple[float, float, float] | None = None
  camera_distance: float | None = None
  camera_elevation: float | None = None
  camera_azimuth: float | None = None

  # Directory to write recorded_positions.npz into (None = CWD).
  output_dir: tyro.conf.Suppress[str | None] = None

  # Skip ONNX export (saves time when called from pipeline).
  skip_onnx_export: tyro.conf.Suppress[bool] = False

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


class RecordingNativeMujocoViewer(NativeMujocoViewer):
  def __init__(self, env, policy, *args, output_dir: str | None = None, **kwargs):
    super().__init__(env, policy, *args, **kwargs)
    self._output_dir = output_dir
    # Recorded data from simulation
    self.box_pos = []
    self.box_quat = []
    self.box_lin_vel = []
    self.box_ang_vel = []
    self.body_pos = []
    self.body_quat = []
    self.body_lin_vel = []
    self.body_ang_vel = []
    self.joint_pos = []
    self.joint_vel = []
    self.root_pos = []
    self.root_quat = []
    self.contact_pos = []
    self.contact_ind = []

    # Referece body positions from motion command
    self.ref_body_positions = []

    # For live vizualization
    self.current_body_pos = None
    self.current_ref_body_pos = None
    
    # Reward terms
    self.reward_terms = {}

  def step_simulation(self) -> None:
    super().step_simulation()
    # Get motion command
    if hasattr(self.env.unwrapped, "command_manager"):
      cmd_manager = self.env.unwrapped.command_manager
      for name, term in cmd_manager._terms.items():
        if isinstance(term.cfg, MotionCommandCfg):
          # Found the motion command term
          ref_object_pos = term.object_pos_w
          ref_object_quat = term.object_quat_w
          ref_body_pos = term.body_pos_w  
          body_pos_viz = term.robot_body_pos_w
          
          # Get actual object position from simulation if available
          if hasattr(self.env.unwrapped.scene, "entities") and "box" in self.env.unwrapped.scene.entities:
             box_entity = self.env.unwrapped.scene.entities["box"]
             # Use root_link_pose_w for consistent object position/orientation
             box_pose = box_entity.data.root_link_pose_w.cpu().numpy()
             
             # Subtract env origins to get local position
             if hasattr(self.env.unwrapped.scene, "env_origins"):
                 env_origins = self.env.unwrapped.scene.env_origins.cpu().numpy()
                 # Only subtract X and Y (Z is usually global/shared)
                 box_pose[:, :2] -= env_origins[:, :2]

             self.box_pos.append(box_pose[:, :3])
             self.box_quat.append(box_pose[:, 3:])
             self.box_lin_vel.append(box_entity.data.root_link_lin_vel_w.cpu().numpy())
             self.box_ang_vel.append(box_entity.data.root_link_ang_vel_w.cpu().numpy())
             
          # Capture robot state
          if hasattr(self.env.unwrapped.scene, "entities"):
              robot = self.env.unwrapped.scene.entities.get("robot")
              
              if robot is not None:
                  self.joint_pos.append(robot.data.joint_pos.cpu().numpy())   
                  self.joint_vel.append(robot.data.joint_vel.cpu().numpy())               
                  root_pose = robot.data.root_link_pose_w.cpu().numpy()
                  self.root_pos.append(root_pose[:, :3])
                  self.root_quat.append(root_pose[:, 3:])

                  # Numpy copy of body positions
                  body_pos = robot.data.body_link_pos_w
                  body_pos_np = body_pos.cpu().numpy()
                  ref_body_pos_np = ref_body_pos.cpu().numpy()
                  
                  # Subtract env origins to get local body positions
                  if hasattr(self.env.unwrapped.scene, "env_origins"):
                      env_origins = self.env.unwrapped.scene.env_origins.cpu().numpy()
                      # body_pos is (Num_Envs, Num_Bodies, 3), env_origins is (Num_Envs, 3)
                      # We need to broadcast env_origins: (Num_Envs, 1, 3)
                      # Get Body positions in local frame
                      body_pos_np[:, :, :2] -= env_origins[:, None, :2]
                      ref_body_pos_np[:, :, :2] -= env_origins[:, None, :2]
                  
                  self.body_pos.append(body_pos_np)
                  self.body_quat.append(robot.data.body_link_quat_w.cpu().numpy())
                  self.body_lin_vel.append(robot.data.body_link_lin_vel_w.cpu().numpy())
                  self.body_ang_vel.append(robot.data.body_link_ang_vel_w.cpu().numpy())
                  self.ref_body_positions.append(ref_body_pos_np)

          # Compute contacts
          if hasattr(self.env.unwrapped.scene, "entities") and "box" in self.env.unwrapped.scene.entities:
              # We need to do this for each environment
              step_contact_pos = []
              step_contact_ind = []
              
              # Get data for all envs
              # Use EEF positions if available
              if hasattr(term, "robot_eef_pos_w"):
                   all_eef_pos = term.robot_eef_pos_w.cpu().numpy() # (N, E, 3)
              else:
                   raise AttributeError("Motion command term does not have 'robot_eef_pos_w'. Ensure the task config provides end-effector information.")
              
              # Subtract env origins from EEF positions so they match box_pos (local frame)
              if hasattr(self.env.unwrapped.scene, "env_origins"):
                   eo = self.env.unwrapped.scene.env_origins.cpu().numpy()
                   all_eef_pos[:, :, :2] -= eo[:, None, :2]

              # Box data was already appended to self.box_pos/quat, retrieve last
              all_box_pos = self.box_pos[-1] # (N, 3)
              all_box_quat = self.box_quat[-1] # (N, 4)
              if True:
                object_size, object_geom_local_position, object_geom_local_rotation = _get_box_geom_props(self.env)

              for i in range(all_eef_pos.shape[0]):
                  res = extract_contacts_from_distance(
                      eef_positions=all_eef_pos[i],
                      object_position=all_box_pos[i],
                      object_rotation=all_box_quat[i],
                      object_size=object_size,
                      object_geom_local_position=object_geom_local_position,
                      object_geom_local_rotation=object_geom_local_rotation,
                      threshold=0.05
                  )
                  step_contact_pos.append(res["contact_positions"])
                  step_contact_ind.append(res["contact_indicators"])
              
              self.contact_pos.append(np.stack(step_contact_pos))
              self.contact_ind.append(np.stack(step_contact_ind))
          
          # Store for visualization (take first 5 envs)
          num_vis = min(5, body_pos_viz.shape[0])
          self.current_body_pos = body_pos_viz[:num_vis].cpu().numpy()
          self.current_ref_body_pos = ref_body_pos[:num_vis].cpu().numpy()
        
          break
    
    # Record rewards
    if hasattr(self.env.unwrapped, "reward_manager"):
      reward_manager = self.env.unwrapped.reward_manager
      step_rewards = reward_manager._step_reward.cpu().numpy()
      for idx, name in enumerate(reward_manager._term_names):
        if name not in self.reward_terms:
          self.reward_terms[name] = []
        self.reward_terms[name].append(step_rewards[:, idx])

  def sync_env_to_viewer(self) -> None:
    super().sync_env_to_viewer()
    
    if self.current_body_pos is not None and self.viewer is not None:
      v = self.viewer
      
      # Iterate over the visualized environments (up to 5)
      for env_idx in range(self.current_body_pos.shape[0]):
          # Add spheres for body points
          for pos in self.current_body_pos[env_idx]:
            if v.user_scn.ngeom < v.user_scn.maxgeom:
              mujoco.mjv_initGeom(
                v.user_scn.geoms[v.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.04, 0, 0], dtype=np.float64),
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=np.array([1, 0, 0, 1], dtype=np.float32)  # Red color
              )
              v.user_scn.ngeom += 1

          for pos in self.current_ref_body_pos[env_idx]:
            if v.user_scn.ngeom < v.user_scn.maxgeom:
              mujoco.mjv_initGeom(
                v.user_scn.geoms[v.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.04, 0, 0], dtype=np.float64),
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=np.array([0, 0, 1, 1], dtype=np.float32)  # Blue color
              )
              v.user_scn.ngeom += 1
      
      v.sync(state_only=True)

  def close(self):
    super().close()
    if self.box_pos and self.body_pos:
      box_positions = np.stack(self.box_pos)
      body_positions = np.stack(self.body_pos)
      ref_body_positions = np.stack(self.ref_body_positions)
      print(f"[INFO] Recorded box positions shape: {box_positions.shape}")
      print(f"[INFO] Recorded body positions shape: {body_positions.shape}")
      print(f"[INFO] Recorded ref body positions shape: {ref_body_positions.shape}")
      
      save_dict = {
        "object_pos_w": box_positions,
        "body_pos_w": body_positions,
        "ref_body_pos_w": ref_body_positions,
      }
      
      # Save body names for mapping
      if hasattr(self.env.unwrapped.scene, "entities") and "robot" in self.env.unwrapped.scene.entities:
          robot = self.env.unwrapped.scene.entities["robot"]
          if hasattr(robot, "body_names"):
              save_dict["body_names"] = np.array(robot.body_names)

      if self.joint_pos:
          save_dict["joint_pos"] = np.stack(self.joint_pos)
          save_dict["joint_vel"] = np.stack(self.joint_vel)
          # save_dict["root_pos"] = np.stack(self.root_pos)
          # save_dict["root_quat"] = np.stack(self.root_quat)
          save_dict["object_quat_w"] = np.stack(self.box_quat)
          save_dict["object_lin_vel_w"] = np.stack(self.box_lin_vel)
          save_dict["object_ang_vel_w"] = np.stack(self.box_ang_vel)

          save_dict["body_quat_w"] = np.stack(self.body_quat)
          save_dict["body_lin_vel_w"] = np.stack(self.body_lin_vel)
          save_dict["body_ang_vel_w"] = np.stack(self.body_ang_vel)
      
      if self.contact_pos:
          save_dict["contact_pos_w"] = np.stack(self.contact_pos)
          save_dict["contact_indices"] = np.stack(self.contact_ind)
          # Also save under the canonical keys used by the RL pipeline
          # (overwrite_with_rollouts expects these for time-dim consistency).
          save_dict["contact_indicators"] = np.stack(self.contact_ind)
          save_dict["contact_positions"] = np.stack(self.contact_pos)
          
      if self.reward_terms:
        print(f"[INFO] Recording {len(self.reward_terms)} reward terms.")
        for name, values in self.reward_terms.items():
          save_dict[f"reward_{name}"] = np.stack(values)
          
      _out = os.path.join(self._output_dir, "recorded_positions.npz") if self._output_dir else "recorded_positions.npz"
      np.savez(_out, **save_dict)
      print(f"[INFO] Saved positions and rewards to {_out}")


class HeadlessRecordingViewer(RecordingNativeMujocoViewer):
  """Headless variant of RecordingNativeMujocoViewer.

  Runs the exact same simulation stepping and data recording logic but
  without opening a GUI window or doing any rendering.  This gives
  identical ``recorded_positions.npz`` output while eliminating all
  rendering / vsync overhead.
  """

  # ── skip GUI setup ──────────────────────────────────────────

  def setup(self) -> None:
    """No-op — no viewer window to create."""

  def is_running(self) -> bool:
    return True

  def sync_env_to_viewer(self) -> None:
    """No-op — nothing to render."""

  def sync_viewer_to_env(self) -> None:
    """No-op — no perturbations from a viewer."""

  # ── tight run loop (no frame-rate throttling) ───────────────

  def run(self, num_steps: int | None = None) -> None:
    """Step simulation as fast as possible, then save recorded data."""
    import torch as _torch

    self.setup()
    try:
      with _torch.no_grad():
        while num_steps is None or self._step_count < num_steps:
          self.step_simulation()
    finally:
      self.close()

  # ── save data without touching the (non-existent) viewer ────

  def close(self) -> None:
    """Save recorded data. Skips NativeMujocoViewer's window cleanup."""
    # Bypass NativeMujocoViewer.close() (which touches self.viewer)
    # and go straight to the recording-save logic.
    if self.box_pos and self.body_pos:
      box_positions = np.stack(self.box_pos)
      body_positions = np.stack(self.body_pos)
      ref_body_positions = np.stack(self.ref_body_positions)
      print(f"[INFO] Recorded box positions shape: {box_positions.shape}")
      print(f"[INFO] Recorded body positions shape: {body_positions.shape}")
      print(f"[INFO] Recorded ref body positions shape: {ref_body_positions.shape}")

      save_dict = {
        "object_pos_w": box_positions,
        "body_pos_w": body_positions,
        "ref_body_pos_w": ref_body_positions,
      }

      if hasattr(self.env.unwrapped.scene, "entities") and "robot" in self.env.unwrapped.scene.entities:
          robot = self.env.unwrapped.scene.entities["robot"]
          if hasattr(robot, "body_names"):
              save_dict["body_names"] = np.array(robot.body_names)

      if self.joint_pos:
          save_dict["joint_pos"] = np.stack(self.joint_pos)
          save_dict["joint_vel"] = np.stack(self.joint_vel)
          save_dict["object_quat_w"] = np.stack(self.box_quat)
          save_dict["object_lin_vel_w"] = np.stack(self.box_lin_vel)
          save_dict["object_ang_vel_w"] = np.stack(self.box_ang_vel)
          save_dict["body_quat_w"] = np.stack(self.body_quat)
          save_dict["body_lin_vel_w"] = np.stack(self.body_lin_vel)
          save_dict["body_ang_vel_w"] = np.stack(self.body_ang_vel)

      if self.contact_pos:
          save_dict["contact_pos_w"] = np.stack(self.contact_pos)
          save_dict["contact_indices"] = np.stack(self.contact_ind)
          # Also save under the canonical keys used by the RL pipeline
          save_dict["contact_indicators"] = np.stack(self.contact_ind)
          save_dict["contact_positions"] = np.stack(self.contact_pos)

      if self.reward_terms:
        print(f"[INFO] Recording {len(self.reward_terms)} reward terms.")
        for name, values in self.reward_terms.items():
          save_dict[f"reward_{name}"] = np.stack(values)

      _out = os.path.join(self._output_dir, "recorded_positions.npz") if self._output_dir else "recorded_positions.npz"
      np.savez(_out, **save_dict)
      print(f"[INFO] Saved positions and rewards to {_out}")


def extract_contacts_from_distance(
  eef_positions: np.ndarray,
  object_position: np.ndarray,
  object_rotation: np.ndarray | None = None,
  object_size: np.ndarray | None = None,
  object_geom_local_position: np.ndarray | None = None,
  object_geom_local_rotation: np.ndarray | None = None,
  threshold: float = 0.05,
) -> dict[str, np.ndarray]:
  """Extract contact information using distance threshold with object size consideration."""
  def _normalize_quat_np(quat: np.ndarray | None) -> np.ndarray | None:
    if quat is None:
      return None
    quat_arr = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat_arr))
    if norm <= 0.0:
      return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat_arr / norm

  effective_object_position = np.asarray(object_position, dtype=np.float32)
  effective_object_rotation = _normalize_quat_np(object_rotation)
  if object_geom_local_position is not None or object_geom_local_rotation is not None:
    geom_local_position = np.zeros(3, dtype=np.float32)
    if object_geom_local_position is not None:
      geom_local_position = np.asarray(object_geom_local_position, dtype=np.float32)
    geom_local_rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if object_geom_local_rotation is not None:
      normalized_geom_local_rotation = _normalize_quat_np(object_geom_local_rotation)
      assert normalized_geom_local_rotation is not None
      geom_local_rotation = normalized_geom_local_rotation
    if effective_object_rotation is None:
      effective_object_position = effective_object_position + geom_local_position
      effective_object_rotation = geom_local_rotation
    else:
      object_rotation_t = torch.from_numpy(effective_object_rotation).unsqueeze(0)
      effective_object_position = (
        quat_apply(
          object_rotation_t,
          torch.from_numpy(geom_local_position).unsqueeze(0),
        )[0].numpy()
        + effective_object_position
      )
      effective_object_rotation = quat_mul(
        object_rotation_t,
        torch.from_numpy(geom_local_rotation).unsqueeze(0),
      )[0].numpy()

  num_eefs = eef_positions.shape[0]
  contact_indicators = np.zeros(num_eefs, dtype=bool)
  contact_positions = np.full((num_eefs, 3), np.nan, dtype=np.float32)

  if object_size is None:
    for i in range(num_eefs):
      dist = np.linalg.norm(eef_positions[i] - effective_object_position)
      if dist <= threshold:
        contact_indicators[i] = True
        contact_positions[i] = effective_object_position.copy()
  else:
    if effective_object_rotation is not None:
      eef_pos_local = quat_apply_inverse(
        torch.from_numpy(effective_object_rotation.astype(np.float32)).unsqueeze(0),
        torch.from_numpy(
          (eef_positions - effective_object_position.reshape(1, 3)).astype(np.float32)
        ),
      ).numpy()
    else:
      eef_pos_local = eef_positions - effective_object_position.reshape(1, 3)

    for i in range(num_eefs):
      if np.isnan(eef_pos_local[i]).any():
        continue

      bounds = object_size + threshold
      within_bounds = np.all(np.abs(eef_pos_local[i]) <= bounds)

      if within_bounds:
        contact_indicators[i] = True
        closest_local = np.clip(eef_pos_local[i], -object_size, object_size)
        if effective_object_rotation is not None:
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


def _get_box_geom_props(env: Any) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
  box_entity = env.unwrapped.scene.entities["box"]
  object_size = None
  object_geom_local_position = np.zeros(3, dtype=np.float32)
  object_geom_local_rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

  if hasattr(box_entity, "indexing"):
    mj_model = env.unwrapped.sim.mj_model
    for gid in box_entity.indexing.geom_ids.cpu().numpy():
      if mj_model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
        continue
      object_size = mj_model.geom_size[gid].copy()
      object_geom_local_position = mj_model.geom_pos[gid].copy()
      object_geom_local_rotation = mj_model.geom_quat[gid].copy()
      break

  return object_size, object_geom_local_position, object_geom_local_rotation


def run_play(task: str, cfg: PlayConfig, batch_target_values: list[dict[str, float]] | None = None):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task, play=True)
  agent_cfg = load_rl_cfg(task)

  # Modify env_cfg based on task parameters if provided
  target_box_tensor = None
  if "reset_object_pose" in env_cfg.events:
    # Check if we should override object pose (explicit targets OR motion file available)
    should_override = (batch_target_values is not None) or (cfg.motion_file and os.path.exists(cfg.motion_file))
    
    if should_override:
        # Construct Tensor for Per-Env Initialization
        try:
             # Default Values
             default_x = 0.4
             default_y = 0.0
             default_z = 0.17 # Box height / 2

             # Determine number of environments
             if batch_target_values:
                 num_envs = len(batch_target_values)
             elif cfg.num_envs:
                 num_envs = cfg.num_envs
             else:
                 num_envs = env_cfg.scene.num_envs
                 
             targets_np = np.zeros((num_envs, 7), dtype=np.float32)
             
             targets_np[:, 0] = default_x
             targets_np[:, 1] = default_y
             targets_np[:, 2] = default_z
             # Default orientation (w, x, y, z) = identity
             targets_np[:, 3] = 1.0
             
             # Attempt to load from motion file if available
             if cfg.motion_file and os.path.exists(cfg.motion_file):
                 try:
                     print(f"[INFO] Loading initial object poses from {cfg.motion_file}")
                     with np.load(cfg.motion_file) as motion_data:
                         if "object_pos_w" in motion_data:
                             obj_pos = motion_data["object_pos_w"]
                             
                             # Shape: (Batch, Time, 3) or (Time, 3)
                             if obj_pos.ndim == 3:
                                 limit = min(num_envs, obj_pos.shape[0])
                                 targets_np[:limit, :3] = obj_pos[:limit, 0, :]
                             elif obj_pos.ndim == 2:
                                 targets_np[:, :3] = obj_pos[0, :]
                         else:
                             print("[WARN] 'object_pos_w' not found in motion file. Using defaults.")
                         
                         if "object_quat_w" in motion_data:
                             obj_quat = motion_data["object_quat_w"]
                             # Shape: (Batch, Time, 4) or (Time, 4)
                             if obj_quat.ndim == 3:
                                 limit = min(num_envs, obj_quat.shape[0])
                                 targets_np[:limit, 3:7] = obj_quat[:limit, 0, :]
                             elif obj_quat.ndim == 2:
                                 targets_np[:, 3:7] = obj_quat[0, :]
                         else:
                             # Keep default identity quaternion if not found
                             pass

                 except Exception as e:
                     print(f"[ERROR] Failed to load motion file for object spawn: {e}")

             if batch_target_values:
                 for i, t in enumerate(batch_target_values):
                     if "initial_box_x" in t:
                         targets_np[i, 0] = t["initial_box_x"]
                     if "initial_box_y" in t:
                         targets_np[i, 1] = t["initial_box_y"]
             
             # Store for injection later
             target_box_tensor = torch.tensor(targets_np, device=device)
             print(f"[INFO] Configured per-environment box initialization for {num_envs} envs.")
             
             # Override the reset event to use our custom function
             env_cfg.events["reset_object_pose"] = EventTermCfg(
                func=mdp.reset_root_state_from_tensor,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("box"),
                    "buffer_name": "target_box_pos",
                },
            )

        except Exception as e:
            print(f"[ERROR] Failed to setup per-env initialization: {e}")

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = (
    env_cfg.commands is not None
    and "motion" in env_cfg.commands
    and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    assert env_cfg.commands is not None
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    assert env_cfg.commands is not None
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    if DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require `registry_name` when using dummy agents."
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width
  if cfg.env_spacing is not None:
    print(f"[INFO] Overriding env_spacing to {cfg.env_spacing}")
    env_cfg.scene.env_spacing = cfg.env_spacing

  # Override camera settings if provided
  if cfg.camera_lookat is not None:
      env_cfg.viewer.lookat = cfg.camera_lookat
  if cfg.camera_distance is not None:
      env_cfg.viewer.distance = cfg.camera_distance
  if cfg.camera_elevation is not None:
      env_cfg.viewer.elevation = cfg.camera_elevation
  if cfg.camera_azimuth is not None:
      env_cfg.viewer.azimuth = cfg.camera_azimuth

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = DartRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  
  # Inject target tensor if created
  if target_box_tensor is not None:
      # We attach it to the env instance so the event function can find it
      env.target_box_pos = target_box_tensor

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    if is_tracking_task:
      runner = MotionTrackingOnPolicyRunner(
        env, asdict(agent_cfg), log_dir=str(log_dir), device=device
      )
    else:
      runner = OnPolicyRunner(
        env, asdict(agent_cfg), log_dir=str(log_dir), device=device
      )
    # runner.load(str(resume_path), map_location=device)
    smart_load_weights(runner.alg.policy, str(resume_path), device)
    policy = runner.get_inference_policy(device=device)

    if not cfg.skip_onnx_export:
      # Export policy as ONNX before running play
      assert log_dir is not None  # log_dir is set in TRAINED_MODE block
      onnx_path = str(log_dir)
      onnx_filename = "policy.onnx"

      # Get normalizer if it exists
      if runner.alg.policy.actor_obs_normalization:
        normalizer = runner.alg.policy.actor_obs_normalizer
      else:
        normalizer = None

      print(f"[INFO]: Exporting policy as ONNX to {onnx_path}/{onnx_filename}")
      if is_tracking_task:
        export_motion_policy_as_onnx(
          env.unwrapped,
          runner.alg.policy,
          normalizer=normalizer,
          path=onnx_path,
          filename=onnx_filename,
        )
      else:
        export_policy_as_onnx(
          runner.alg.policy,
          normalizer=normalizer,
          path=onnx_path,
          filename=onnx_filename,
        )
      print("[INFO]: ONNX export completed")

      # Attach ONNX metadata
      # Use wandb_run_path if available, otherwise use log_dir name
      if cfg.wandb_run_path is not None:
        run_path = cfg.wandb_run_path
      else:
        # Use the log_dir name as fallback
        run_path = str(log_dir.name) if log_dir else "unknown"

      print(f"[INFO]: Attaching ONNX metadata (run_path: {run_path})")
      if is_tracking_task:
        attach_tracking_onnx_metadata(
          env.unwrapped,
          run_path,
          path=onnx_path,
          filename=onnx_filename,
        )
      else:
        attach_velocity_onnx_metadata(
          env.unwrapped,
          run_path,
          path=onnx_path,
          filename=onnx_filename,
        )
      print("[INFO]: ONNX metadata attachment completed")

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "headless":
    HeadlessRecordingViewer(env, policy, output_dir=cfg.output_dir).run(num_steps=cfg.max_steps)
  elif resolved_viewer == "native":
    RecordingNativeMujocoViewer(env, policy, output_dir=cfg.output_dir).run(num_steps=cfg.max_steps)
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run(num_steps=cfg.max_steps)
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import humanoid_dart.tracking  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
