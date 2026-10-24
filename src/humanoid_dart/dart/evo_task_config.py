from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import random
import torch
from scipy.spatial.transform import Rotation as R

@dataclass
class TaskDimension:
    name: str
    min_val: float
    max_val: float
    bin_size: float
    
    # Function to extract this metric from recorded data
    # Signature: (recorded_data: Dict[str, np.ndarray], start_idx: int) -> np.ndarray (shape: [num_envs])
    metric_fn: Any 

@dataclass
class TaskConfig:
    dimensions: List[TaskDimension]
    
    def get_bins(self) -> List[Tuple[int, ...]]:
        """Generate all possible bin combinations."""
        ranges = []
        for dim in self.dimensions:
            num_bins = int((dim.max_val - dim.min_val) / dim.bin_size) + 1
            ranges.append(np.arange(num_bins))
        
        # Create grid
        grids = np.meshgrid(*ranges, indexing='ij')
        # Flatten and stack
        flat_grids = [g.flatten() for g in grids]
        return list(zip(*flat_grids))

    def get_bin_tuple(self, values: Dict[str, float]) -> Tuple[int, ...]:
        """Convert metric values to a bin tuple."""
        bin_indices = []
        for dim in self.dimensions:
            val = values[dim.name]
            # Clamp value
            val = max(dim.min_val, min(dim.max_val, val))
            # Compute bin
            idx = int((val - dim.min_val) / dim.bin_size)
            bin_indices.append(idx)
        return tuple(bin_indices)

    def sample_target(self, bin_tuple: Tuple[int, ...]) -> Dict[str, float]:
        """Sample a random target value within the specified bin."""
        targets = {}
        for i, dim in enumerate(self.dimensions):
            bin_idx = bin_tuple[i]
            # Range for this bin
            low = dim.min_val + bin_idx * dim.bin_size
            high = low + dim.bin_size
            # Sample
            val = random.uniform(low, high)
            val = max(dim.min_val, min(dim.max_val, val))
            targets[dim.name] = val
        return targets

    def get_bin_center(self, bin_tuple: Tuple[int, ...]) -> List[float]:
        """Get the center values of the bin."""
        center_vals = []
        for i, dim in enumerate(self.dimensions):
            bin_idx = bin_tuple[i]
            low = dim.min_val + bin_idx * dim.bin_size
            center = low + dim.bin_size / 2.0
            # Clamp to range just in case
            center = max(dim.min_val, min(dim.max_val, center))
            center_vals.append(center)
        return center_vals

    def distance(self, elite_metrics: Dict[str, float], target: Dict[str, float]) -> float:
        """Compute Euclidean distance between elite metrics and target."""
        sq_dist = 0.0
        for dim in self.dimensions:
            diff = elite_metrics[dim.name] - target[dim.name]
            sq_dist += diff**2
        return np.sqrt(sq_dist)

# --- Metric Extraction Functions ---

def _get_valid_ends(recorded_data, T):
    """Return per-env last valid frame index (inclusive). Shape (Envs,)."""
    traj_lengths = recorded_data.get('traj_lengths')
    if traj_lengths is not None:
        return np.clip(traj_lengths.astype(int) - 1, 0, T - 1)
    return None

def extract_box_height(recorded_data, start_idx):
    # (T, Envs, 3)
    box_pos = recorded_data['object_pos_w']
    if box_pos.ndim == 2: box_pos = box_pos[:, None, :]
    T = box_pos.shape[0]
    valid_ends = _get_valid_ends(recorded_data, T)
    if valid_ends is not None:
        # Per-env masked mean of z from start_idx to valid end
        time_idx = np.arange(T)[:, None]  # (T, 1)
        mask = (time_idx >= start_idx) & (time_idx <= valid_ends[None, :])  # (T, Envs)
        masked_z = box_pos[:, :, 2] * mask
        counts = np.maximum(mask.sum(axis=0), 1)
        return masked_z.sum(axis=0) / counts
    return np.mean(box_pos[start_idx:, :, 2], axis=0)

def extract_box_x(recorded_data, start_idx):
    box_pos = recorded_data['object_pos_w']
    if box_pos.ndim == 2: box_pos = box_pos[:, None, :]
    return np.mean(box_pos[start_idx-2:start_idx+2, :, 0], axis=0)

def extract_box_y(recorded_data, start_idx):
    box_pos = recorded_data['object_pos_w']
    if box_pos.ndim == 2: box_pos = box_pos[:, None, :]
    return np.mean(box_pos[start_idx-2:start_idx+2, :, 1], axis=0)

def extract_box_z(recorded_data, start_idx):
    box_pos = recorded_data['object_pos_w']
    if box_pos.ndim == 2: box_pos = box_pos[:, None, :]
    return np.mean(box_pos[start_idx-2:start_idx+2, :, 2], axis=0)

def extract_final_box_x(recorded_data, start_idx):
    box_pos = recorded_data['object_pos_w']
    if box_pos.ndim == 2: box_pos = box_pos[:, None, :]
    T = box_pos.shape[0]
    valid_ends = _get_valid_ends(recorded_data, T)
    if valid_ends is not None:
        # Per-env: average x over [valid_end-4, valid_end] (last 5 valid frames)
        time_idx = np.arange(T)[:, None]  # (T, 1)
        starts = np.maximum(valid_ends - 4, 0)[None, :]  # (1, Envs)
        mask = (time_idx >= starts) & (time_idx <= valid_ends[None, :])  # (T, Envs)
        masked_x = box_pos[:, :, 0] * mask
        counts = np.maximum(mask.sum(axis=0), 1)
        return masked_x.sum(axis=0) / counts
    return np.mean(box_pos[-5:, :, 0], axis=0)

def extract_final_box_y(recorded_data, start_idx):
    box_pos = recorded_data['object_pos_w']
    if box_pos.ndim == 2: box_pos = box_pos[:, None, :]
    T = box_pos.shape[0]
    valid_ends = _get_valid_ends(recorded_data, T)
    if valid_ends is not None:
        time_idx = np.arange(T)[:, None]
        starts = np.maximum(valid_ends - 4, 0)[None, :]
        mask = (time_idx >= starts) & (time_idx <= valid_ends[None, :])
        masked_y = box_pos[:, :, 1] * mask
        counts = np.maximum(mask.sum(axis=0), 1)
        return masked_y.sum(axis=0) / counts
    return np.mean(box_pos[-5:, :, 1], axis=0)

def extract_initial_box_x(recorded_data, start_idx):
    # (T, Envs, 3)
    box_pos = recorded_data['object_pos_w']
    if box_pos.ndim == 2: box_pos = box_pos[:, None, :]
    # Use first 5 frames to estimate initial position
    return np.mean(box_pos[:5, :, 0], axis=0)

def extract_initial_box_y(recorded_data, start_idx):
    # (T, Envs, 3)
    box_pos = recorded_data['object_pos_w']
    if box_pos.ndim == 2: box_pos = box_pos[:, None, :]
    # Use first 5 frames to estimate initial position
    return np.mean(box_pos[:5, :, 1], axis=0)

def _extract_relative_displacement(recorded_data, start_idx):
    # (T, Envs, 3) - Object pos
    box_pos = recorded_data['object_pos_w']
    if box_pos.ndim == 2: box_pos = box_pos[:, None, :]
    
    # (T, Envs, 30, 4) - Body pos. Assume body 0 is base.
    body_quat = recorded_data['body_quat_w']
    # If (T, Envs, Bodies, 4)
    if body_quat.ndim == 4:
        base_quat = body_quat[0, :, 0, :] # Initial frame, all envs, base body
    elif body_quat.ndim == 3: # (T, Envs, 4) if only base logged
        base_quat = body_quat[0, :, :]
    else:
        print("Check the body_quat_w, unexpected shape!")
        return np.zeros((box_pos.shape[1], 3))

    # Initial and Final Box Pos
    p_init = box_pos[0, :, :3] # (Envs, 3)
    
    T = box_pos.shape[0]
    valid_ends = _get_valid_ends(recorded_data, T)
    if valid_ends is not None:
        # Per-env last valid frame
        env_idx = np.arange(box_pos.shape[1])
        p_final = box_pos[valid_ends, env_idx, :3]  # (Envs, 3)
    else:
        p_final = box_pos[-1, :, :3] # (Envs, 3)
    
    delta_pos = p_final - p_init # (Envs, 3)
    
    # Assuming input is [w, x, y, z] (standard Mujoco)
    base_quat_scipy = np.roll(base_quat, -1, axis=-1) # (w,x,y,z) -> (x,y,z,w)
    
    r = R.from_quat(base_quat_scipy)
    
    euler = r.as_euler('xyz', degrees=False)
    yaw = np.asarray(euler[..., 2]).ravel()  # (Envs,) — guard against scipy 1.17+ shape changes

    # Inverse rotation around Z (negative yaw).
    # Use (N, 1) shape — scipy 1.15+ requires 2D (N, num_axes) for batched single-axis rotations.
    r_inv = R.from_euler('z', (-yaw).reshape(-1, 1), degrees=False)
    
    # Apply rotation
    delta_rot = r_inv.apply(delta_pos) # (Envs, 3)
    
    return delta_rot # x, y, z

def extract_relative_x(recorded_data, start_idx):
    deltas = _extract_relative_displacement(recorded_data, start_idx)
    return deltas[:, 0]

def extract_relative_y(recorded_data, start_idx):
    deltas = _extract_relative_displacement(recorded_data, start_idx)
    return deltas[:, 1]

def extract_relative_z(recorded_data, start_idx):
    deltas = _extract_relative_displacement(recorded_data, start_idx)
    return deltas[:, 2]


# ---------------------------------------------------------------------------
# 6-D relative pose extraction (x, y, z, roll, pitch, yaw in robot yaw frame)
# ---------------------------------------------------------------------------

_CACHE_KEY_6D = "__relative_pose_6d_cache__"

def _extract_relative_pose_6d(recorded_data: dict, start_idx: int) -> np.ndarray:
    """Return (Envs, 6): [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw].

    All values are expressed in the robot's initial yaw frame so they are
    invariant to the robot's heading.  Angles are in radians and wrapped to
    [-π, π] (pitch clamped to [-π/2, π/2]).

    The result is cached on the recorded_data dict under ``_CACHE_KEY_6D`` so
    multiple per-dimension extractor calls only compute it once per evaluation.
    """
    if _CACHE_KEY_6D in recorded_data:
        return recorded_data[_CACHE_KEY_6D]  # type: ignore[return-value]

    # --- object position ---
    box_pos = recorded_data['object_pos_w']
    if box_pos.ndim == 2:
        box_pos = box_pos[:, None, :]   # (T, 1, 3)

    # --- robot base quaternion at t=0 [w, x, y, z] (MuJoCo convention) ---
    body_quat = recorded_data['body_quat_w']
    if body_quat.ndim == 4:
        base_quat = body_quat[0, :, 0, :]   # (Envs, 4)
    elif body_quat.ndim == 3:
        base_quat = body_quat[0, :, :]       # (Envs, 4)
    else:
        n_envs = box_pos.shape[1]
        result = np.zeros((n_envs, 6))
        recorded_data[_CACHE_KEY_6D] = result
        return result

    # --- object quaternion [w, x, y, z] ---
    obj_quat = recorded_data.get('object_quat_w')  # (T, Envs, 4) or None
    if obj_quat is None:
        # Fallback: attempt body quat channel 1 if available
        if body_quat.ndim == 4 and body_quat.shape[2] > 1:
            obj_quat = body_quat[:, :, 1, :]
        else:
            obj_quat = None

    n_envs = box_pos.shape[1]
    T = box_pos.shape[0]
    valid_ends = _get_valid_ends(recorded_data, T)
    env_idx = np.arange(n_envs)

    # Initial and final object positions
    p_init = box_pos[0, :, :]   # (Envs, 3)
    if valid_ends is not None:
        p_final = box_pos[valid_ends, env_idx, :]   # (Envs, 3)
    else:
        p_final = box_pos[-1, :, :]

    delta_pos = p_final - p_init  # (Envs, 3)

    # --- robot yaw from initial base quaternion ---
    # MuJoCo [w,x,y,z] → scipy [x,y,z,w]
    base_quat_scipy = np.roll(base_quat, -1, axis=-1)
    r_base = R.from_quat(base_quat_scipy)
    euler_base = r_base.as_euler('xyz', degrees=False)
    yaw = np.asarray(euler_base[..., 2]).ravel()   # (Envs,) — ravel() guards against scipy 1.17+ returning extra dims

    # Inverse yaw rotation matrix applied to position delta.
    # Use shape (Envs, 1) for from_euler — scipy 1.15+ requires 2D (N, num_axes)
    # for batched single-axis rotations; (N,) 1D raises ValueError in scipy 1.17.
    r_inv = R.from_euler('z', (-yaw).reshape(-1, 1), degrees=False)
    delta_pos_local = r_inv.apply(delta_pos)   # (Envs, 3)

    # --- orientation delta ---
    if obj_quat is not None:
        # Initial and final object quaternion [w,x,y,z]
        q_init = obj_quat[0, :, :]   # (Envs, 4)
        if valid_ends is not None:
            q_final = obj_quat[valid_ends, env_idx, :]
        else:
            q_final = obj_quat[-1, :, :]

        # scipy convention: [x,y,z,w]
        q_init_s  = np.roll(q_init,  -1, axis=-1)
        q_final_s = np.roll(q_final, -1, axis=-1)

        r_init  = R.from_quat(q_init_s)
        r_final = R.from_quat(q_final_s)

        # Express both in robot yaw frame, then compute delta
        r_init_rob  = r_inv * r_init
        r_final_rob = r_inv * r_final

        euler_init  = r_init_rob.as_euler('xyz', degrees=False)   # (Envs, 3)
        euler_final = r_final_rob.as_euler('xyz', degrees=False)  # (Envs, 3)

        def _wrap(a: np.ndarray) -> np.ndarray:
            return (a + np.pi) % (2 * np.pi) - np.pi

        delta_roll  = _wrap(euler_final[:, 0] - euler_init[:, 0])
        delta_pitch = np.clip(
            _wrap(euler_final[:, 1] - euler_init[:, 1]),
            -np.pi / 2, np.pi / 2
        )
        delta_yaw   = _wrap(euler_final[:, 2] - euler_init[:, 2])
    else:
        delta_roll  = np.zeros(n_envs)
        delta_pitch = np.zeros(n_envs)
        delta_yaw   = np.zeros(n_envs)

    result = np.stack(
        [delta_pos_local[:, 0], delta_pos_local[:, 1], delta_pos_local[:, 2],
         delta_roll, delta_pitch, delta_yaw],
        axis=-1,
    )   # (Envs, 6)

    recorded_data[_CACHE_KEY_6D] = result
    return result


def extract_relative_x_6d(recorded_data, start_idx):
    return _extract_relative_pose_6d(recorded_data, start_idx)[:, 0]

def extract_relative_y_6d(recorded_data, start_idx):
    return _extract_relative_pose_6d(recorded_data, start_idx)[:, 1]

def extract_relative_z_6d(recorded_data, start_idx):
    return _extract_relative_pose_6d(recorded_data, start_idx)[:, 2]

def extract_relative_roll(recorded_data, start_idx):
    return _extract_relative_pose_6d(recorded_data, start_idx)[:, 3]

def extract_relative_pitch(recorded_data, start_idx):
    return _extract_relative_pose_6d(recorded_data, start_idx)[:, 4]

def extract_relative_yaw(recorded_data, start_idx):
    return _extract_relative_pose_6d(recorded_data, start_idx)[:, 5]


# --- Registry ---

TASK_REGISTRY = {
    "box_height_xy": TaskConfig(
        dimensions=[
            TaskDimension("height", 0.3, 1.4, 0.02, extract_box_height),
            TaskDimension("initial_box_x", -0.45, 0.4, 0.05, extract_initial_box_x),
            TaskDimension("initial_box_y", -0.5, 0.5, 0.05, extract_initial_box_y)
        ]
    ),
    "box_height_x": TaskConfig(
        dimensions=[
            TaskDimension("height", 0.3, 1.4, 0.02, extract_box_height),
            TaskDimension("initial_box_x", -0.45, 0.4, 0.05, extract_initial_box_x)
        ]
    ),
    "box_height": TaskConfig(
        dimensions=[
            TaskDimension("height", 0.3, 1.4, 0.02, extract_box_height)
        ]
    ),
    "pick_place": TaskConfig(
        dimensions=[
            TaskDimension("height", 0.3, 1.4, 0.02, extract_box_height),
            TaskDimension("box_x", -0.8, 0.0, 0.02, extract_box_x),
            TaskDimension("box_y", 0.5, 1.1, 0.02, extract_box_y)
        ]
    ),
    "pick_place_final_box_pose": TaskConfig(
        dimensions=[
            TaskDimension("final_box_x", -1.6, 0.6, 0.02, extract_final_box_x),
            TaskDimension("final_box_y", 0.4, 2.4, 0.02, extract_final_box_y)
        ]
    ),
    # Main pick-place task now targets final object displacement in 3D.
    # Use the *_xy alias to reproduce the legacy 2D archive/task space.
    "pick_place_relative_box_pose_xyz": TaskConfig(
        dimensions=[
            TaskDimension("relative_x", -2.0, 2.0, 0.1, extract_relative_x),
            TaskDimension("relative_y", -2.0, 2.0, 0.1, extract_relative_y),
            TaskDimension("relative_z", 0.2, 1.2, 0.1, extract_relative_z)
        ]
    ),
    "pick_place_relative_box_pose": TaskConfig(
        dimensions=[
            TaskDimension("relative_x", -3.0, 5, 0.08, extract_relative_x),
            TaskDimension("relative_y", -4.4, 1.6, 0.08, extract_relative_y)
        ]
    ),
    # 6-D task: relative object pose (x, y, z, roll, pitch, yaw) in robot yaw frame.
    # Bin counts: x=16, y=16, z=6, roll=12, pitch=6, yaw=12  → ~663k total bins.
    # Use curriculum="random" until the sparse-archive guard (Phase 0-A) is merged.
    "pick_place_relative_box_pose_6d": TaskConfig(
        dimensions=[
            TaskDimension("relative_x",     -1.5,      1.5,       0.2,       extract_relative_x_6d),
            TaskDimension("relative_y",     -1.5,      1.5,       0.2,       extract_relative_y_6d),
            TaskDimension("relative_z",      0.5,      1.0,       0.1,       extract_relative_z_6d),
            TaskDimension("relative_roll",  -np.pi,    np.pi,     np.pi/6,   extract_relative_roll),
            TaskDimension("relative_pitch", -np.pi/2,  np.pi/2,   np.pi/6,   extract_relative_pitch),
            TaskDimension("relative_yaw",   -np.pi,    np.pi,     np.pi/6,   extract_relative_yaw),
        ]
    ),
}

def get_task_config(name: str) -> TaskConfig:
    if name not in TASK_REGISTRY:
        raise KeyError(f"Unknown evo_task {name!r}; available: {sorted(TASK_REGISTRY)}")
    return TASK_REGISTRY[name]
