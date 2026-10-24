from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


def reset_root_state_from_tensor(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    buffer_name: str = "target_box_pos", # Name of attribute in env to look for
) -> None:
    """
    Reset root state using per-environment targets stored in the environment instance.
    This allows precise initialization for curriculum learning / evaluation where each env has a specific problem.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    # Check if we have the buffer
    if not hasattr(env, buffer_name):
        return

    # Buffer should be (Num_Envs, 3) for Pos or (Num_Envs, 7) for Pose
    targets = getattr(env, buffer_name) 
    
    # Extract for current envs
    # targets is likely a tensor on the correct device
    target_subset = targets[env_ids]
    
    asset: Entity = env.scene[asset_cfg.name]
    
    # Floating-base entities.
    default_root_state = asset.data.default_root_state
    root_states = default_root_state[env_ids].clone() # (N, 13)

    # Apply Position Overrides
    # If target is (N, 3), we assume it is Position (x,y,z) in World Frame? 
    # Or needs to be added to env origin?
    # Usually targets are "global" in the sense of the task (relative to env origin)
    # The standard behavior is pos + env_origin
    
    # Let's assume the tensor stored in env is [x, y, z] relative to env origin
    
    target_pos = target_subset[:, :3]
    
    positions = target_pos + env.scene.env_origins[env_ids]
    
    if target_subset.shape[1] >= 7:
        # Use provided orientation (assuming it's formatted as quaternion w, x, y, z or similar)
        # Mujoco uses [w, x, y, z] for quat
        orientations = target_subset[:, 3:7]
    else:
        # Keep default orientation
        orientations = root_states[:, 3:7] 
    
    asset.write_root_link_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    
    # Reset velocities to zero
    velocities = torch.zeros_like(root_states[:, 7:13])
    asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)
