import os
import numpy as np
import torch
import mujoco
import time
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, cast

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from humanoid_dart.dart.archive import Elite
from mjlab.tasks.registry import load_env_cfg
from humanoid_dart.dart.evo_task_config import get_task_config
from humanoid_dart.dart.utils.evaluate_fitness import compute_laplacian
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
from humanoid_dart.dart.convert_motion import (
    extract_contacts_from_distance as _shared_extract_contacts_from_distance,
)


# ---------------------------------------------------------------------------
# Memory diagnostics
# ---------------------------------------------------------------------------

def log_memory(tag: str = "") -> None:
    """Print current CPU (RSS) and GPU memory usage."""
    parts = [f"[MEM {tag}]"]
    if _HAS_PSUTIL:
        rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9  # type: ignore[possibly-unbound]
        parts.append(f"CPU RSS {rss_gb:.2f} GB")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        resrv = torch.cuda.memory_reserved() / 1e9
        parts.append(f"GPU alloc {alloc:.2f} GB  reserved {resrv:.2f} GB")
    print("  ".join(parts))


# ---------------------------------------------------------------------------
# Data loading / schema helpers
# ---------------------------------------------------------------------------

def load_npz_data(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def ensure_rl_schema(data: dict) -> dict:
    """Convert internal diffusion schema (base/joints/obj) back to RL schema if needed.

    RL schema keys: ``body_pos_w``, ``body_quat_w``, ``joint_pos``, optionally
    ``object_pos_w`` / ``object_quat_w``.
    """
    if 'body_pos_w' in data:
        return data  # Already RL schema
    if 'base' not in data or 'joints' not in data:
        return data  # Unknown schema, pass through

    out = {k: v for k, v in data.items() if k not in ('base', 'joints', 'obj')}
    base = np.asarray(data['base'])   # (T, 7)  root pos(3) + quat(4)
    out['body_pos_w']  = base[:, :3].reshape(-1, 1, 3)
    out['body_quat_w'] = base[:, 3:7].reshape(-1, 1, 4)
    out['joint_pos']   = np.asarray(data['joints'])
    if 'base_vel' in data:
        base_vel = np.asarray(data['base_vel'])
        out['body_lin_vel_w'] = base_vel[:, :3].reshape(-1, 1, 3)
        out['body_ang_vel_w'] = base_vel[:, 3:6].reshape(-1, 1, 3)
    if 'joints_vel' in data:
        out['joint_vel'] = np.asarray(data['joints_vel'])
    if 'obj' in data:
        obj = np.asarray(data['obj'])  # (T, 7)
        out['object_pos_w']  = obj[:, :3]
        out['object_quat_w'] = obj[:, 3:7]
    if 'obj_vel' in data:
        obj_vel = np.asarray(data['obj_vel'])
        out['object_lin_vel_w'] = obj_vel[:, :3]
        out['object_ang_vel_w'] = obj_vel[:, 3:6]
    return out


def get_trajectory_length(traj_data: Dict[str, Any]) -> int:
    """Return the real trajectory length, preferring explicit metadata when available."""
    for key in ('traj_length', 'traj_lengths'):
        if key in traj_data:
            arr = np.asarray(traj_data[key]).reshape(-1)
            if arr.size > 0:
                return int(arr[0])
    return int(np.asarray(traj_data['joint_pos']).shape[0])


CONTACT_EEF_BODY_NAMES = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)


def _normalize_contact_channel_count(traj_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Trim mixed contact-channel tensors to a shared leading prefix.

    This keeps directory merges compatible when some trajectories only contain
    wrist contacts and others contain wrist+ankle contacts. The box-tracking
    task consumes the leading wrist channels, so trimming to the smallest shared
    prefix preserves the compatible subset instead of dropping contact tensors.
    """
    indicator_counts = {
        int(np.asarray(traj['contact_indicators']).shape[-1])
        for traj in traj_list
        if 'contact_indicators' in traj
    }
    position_counts = {
        int(np.asarray(traj['contact_positions']).shape[-2])
        for traj in traj_list
        if 'contact_positions' in traj
    }
    present_counts = indicator_counts | position_counts
    if len(present_counts) <= 1:
        return traj_list

    target_count = min(present_counts)
    print(
        "[INFO] Normalizing contact channels across base motions "
        f"from {sorted(present_counts)} to {target_count}"
    )

    normalized: List[Dict[str, Any]] = []
    for traj in traj_list:
        updated = traj.copy()
        if 'contact_indicators' in updated:
            updated['contact_indicators'] = np.asarray(updated['contact_indicators'])[..., :target_count].copy()
        if 'contact_positions' in updated:
            updated['contact_positions'] = np.asarray(updated['contact_positions'])[..., :target_count, :].copy()
        normalized.append(updated)
    return normalized


def _desired_contact_eef_count(traj_list: List[Dict[str, Any]]) -> int:
    """Infer how many contact channels regenerated trajectories should preserve."""
    max_count = 0
    for traj in traj_list:
        if 'contact_indicators' in traj:
            max_count = max(max_count, int(np.asarray(traj['contact_indicators']).shape[-1]))
        if 'contact_positions' in traj:
            max_count = max(max_count, int(np.asarray(traj['contact_positions']).shape[-2]))
    if max_count <= 0:
        return 2
    return min(max_count, len(CONTACT_EEF_BODY_NAMES))


def _coerce_metadata_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0]
        else:
            return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value)


def _infer_source_folder_name(file_path: Any) -> Optional[str]:
    file_path_str = _coerce_metadata_string(file_path)
    if not file_path_str:
        return None
    npz_idx = file_path_str.find(".npz")
    if npz_idx >= 0:
        file_path_str = file_path_str[: npz_idx + 4]
    path = Path(file_path_str)
    return path.parent.name or None


def _restore_source_metadata(single: Dict[str, Any], fallback_file_path: Any) -> Dict[str, Any]:
    file_path = _coerce_metadata_string(single.get('file_path'))
    if file_path is None:
        file_path = _coerce_metadata_string(fallback_file_path)
    if file_path is not None:
        single['file_path'] = file_path

    source_folder_name = _coerce_metadata_string(single.get('source_folder_name'))
    if source_folder_name is None:
        source_folder_name = _infer_source_folder_name(file_path)
    if source_folder_name is not None:
        single['source_folder_name'] = source_folder_name
    return single


def _trim_single_trajectory(single: Dict[str, Any], real_len: int, padded_len: int) -> Dict[str, Any]:
    trimmed: Dict[str, Any] = {}
    for key, value in single.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == padded_len:
            trimmed[key] = value[:real_len].copy()
        else:
            trimmed[key] = value
    trimmed['traj_length'] = np.array([real_len], dtype=np.int64)
    return trimmed


def _merge_trajectories_to_batched(traj_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    traj_lengths = [get_trajectory_length(traj) for traj in traj_list]
    max_len = max(traj_lengths)
    merged: Dict[str, np.ndarray] = {
        'traj_lengths': np.asarray(traj_lengths, dtype=np.int64),
    }

    shared_keys = set(traj_list[0].keys())
    for traj in traj_list[1:]:
        shared_keys &= set(traj.keys())
    shared_keys.discard('file_path')
    shared_keys.discard('traj_length')
    shared_keys.discard('traj_lengths')

    for key in sorted(shared_keys):
        values = [traj[key] for traj in traj_list]
        if not all(isinstance(value, np.ndarray) for value in values):
            continue

        arrays = [np.asarray(value) for value in values]
        first = arrays[0]

        if all(arr.shape == first.shape for arr in arrays):
            merged[key] = np.stack(arrays, axis=0)
            continue

        trailing_shapes = [arr.shape[1:] for arr in arrays]
        time_indexed = all(
            arr.ndim >= 1 and arr.shape[0] == traj_len
            for arr, traj_len in zip(arrays, traj_lengths)
        )
        if time_indexed and all(shape == trailing_shapes[0] for shape in trailing_shapes):
            padded = []
            for arr, traj_len in zip(arrays, traj_lengths):
                if traj_len < max_len:
                    deficit = max_len - traj_len
                    pad = np.repeat(arr[-1:], deficit, axis=0)
                    arr = np.concatenate([arr, pad], axis=0)
                padded.append(arr)
            merged[key] = np.stack(padded, axis=0)
            continue

        print(f"  [WARN] Skipping key '{key}' while merging base motions due to incompatible shapes")

    file_paths = [_coerce_metadata_string(traj.get('file_path')) for traj in traj_list]
    if any(path is not None for path in file_paths):
        merged['file_path'] = np.asarray(file_paths, dtype=object)

    source_folder_names = [
        _coerce_metadata_string(traj.get('source_folder_name')) or _infer_source_folder_name(traj.get('file_path'))
        for traj in traj_list
    ]
    if any(name is not None for name in source_folder_names):
        merged['source_folder_name'] = np.asarray(source_folder_names, dtype=object)

    return merged


def load_base_motions(
    base_motion_path: Path,
) -> tuple[list[dict[str, Any]], Path]:
    """Load base motion trajectories from a file or directory of ``.npz`` files.

    Parameters
    ----------
    base_motion_path:
        Path to a single ``.npz`` file (batched or single) **or** a directory
        containing multiple ``.npz`` files.

    Returns
    -------
    all_base_data:
        List of trajectory dicts, each already converted to RL schema
        (``body_pos_w``, ``body_quat_w``, ``joint_pos``, …).
    base_motion_path:
        The (possibly updated) path — when a directory is provided the merged
        batched file is created and its path returned.
    """
    all_base_data: list[dict[str, Any]] = []
    base_motion_path = Path(base_motion_path)
    should_merge_directory = base_motion_path.is_dir()

    if should_merge_directory:
        print(f"[INFO] Loading all base motions from directory: {base_motion_path}")
        for f in sorted(base_motion_path.rglob("*.npz")):
            if f.name.startswith("_"):  # skip our own merged files
                continue
            try:
                raw = dict(np.load(f, allow_pickle=True))
                if 'joint_pos' in raw and raw['joint_pos'].ndim == 3:
                    ref_key = 'joint_pos'
                elif 'base' in raw and raw['base'].ndim == 3:
                    ref_key = 'base'
                else:
                    ref_key = None

                if ref_key is not None:
                    num_trajs = raw[ref_key].shape[0]
                    padded_len = raw[ref_key].shape[1]
                    traj_lengths = raw.get('traj_lengths')
                    print(f"  [INFO] Batched file {f.name}: {num_trajs} trajectories")
                    keys_to_split = [
                        k for k, v in raw.items()
                        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == num_trajs
                    ]
                    keys_to_copy = [k for k in raw if k not in keys_to_split]
                    for i in range(num_trajs):
                        single = {k: raw[k][i] for k in keys_to_split}
                        for k in keys_to_copy:
                            single[k] = raw[k]
                        real_len = int(traj_lengths[i]) if traj_lengths is not None else padded_len
                        single = _trim_single_trajectory(single, real_len, padded_len)
                        single = _restore_source_metadata(single, f"{f}_{i}")
                        all_base_data.append(single)
                elif (
                    ('joint_pos' in raw and 'body_pos_w' in raw and 'body_quat_w' in raw)
                    or ('base' in raw and 'joints' in raw)
                ):
                    if 'traj_length' in raw:
                        real_len = get_trajectory_length(raw)
                        raw_time_key = 'joint_pos'
                        if raw_time_key not in raw:
                            raw_time_key = 'base'
                        raw = _trim_single_trajectory(raw, real_len, raw[raw_time_key].shape[0])
                    raw = _restore_source_metadata(raw, str(f))
                    all_base_data.append(raw)
                else:
                    print(f"  [SKIP] {f.name}: missing required keys")
            except Exception as e:
                print(f"[WARN] Failed to load {f}: {e}")

    elif base_motion_path.suffix == '.npz':
        raw_data = dict(np.load(base_motion_path, allow_pickle=True))

        if 'joint_pos' in raw_data and raw_data['joint_pos'].ndim == 3:
            ref_key = 'joint_pos'
        elif 'base' in raw_data and raw_data['base'].ndim == 3:
            ref_key = 'base'
        else:
            ref_key = None

        if ref_key is not None:
            num_trajs = raw_data[ref_key].shape[0]
            padded_len = raw_data[ref_key].shape[1]
            traj_lengths = raw_data.get('traj_lengths')
            print(f"[INFO] Detected batched .npz: {num_trajs} trajectories.")

            keys_to_split = []
            keys_to_copy = []
            for k, v in raw_data.items():
                if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == num_trajs:
                    keys_to_split.append(k)
                else:
                    keys_to_copy.append(k)

            for i in range(num_trajs):
                single_traj = {k: raw_data[k][i] for k in keys_to_split}
                for k in keys_to_copy:
                    single_traj[k] = raw_data[k]
                real_len = int(traj_lengths[i]) if traj_lengths is not None else padded_len
                single_traj = _trim_single_trajectory(single_traj, real_len, padded_len)
                single_traj = _restore_source_metadata(single_traj, f"{base_motion_path}_{i}")
                all_base_data.append(single_traj)
        else:
            print(f"[INFO] Loading single base motion: {base_motion_path}")
            if 'traj_length' in raw_data:
                real_len = get_trajectory_length(raw_data)
                raw_time_key = 'joint_pos'
                if raw_time_key not in raw_data:
                    raw_time_key = 'base'
                raw_data = _trim_single_trajectory(raw_data, real_len, raw_data[raw_time_key].shape[0])
            raw_data = _restore_source_metadata(raw_data, str(base_motion_path))
            all_base_data.append(raw_data)
    else:
        raise ValueError(f"Unknown file format or path: {base_motion_path}")

    if not all_base_data:
        raise ValueError(f"No valid motion data found in {base_motion_path}")

    # Normalise to RL schema
    all_base_data = [ensure_rl_schema(d) for d in all_base_data]

    for idx, traj in enumerate(all_base_data):
        traj = _restore_source_metadata(traj, traj.get('file_path'))
        if 'traj_length' not in traj:
            traj['traj_length'] = np.array([get_trajectory_length(traj)], dtype=np.int64)
        all_base_data[idx] = traj

    print(f"[INFO] Loaded {len(all_base_data)} base trajectories.")

    first = all_base_data[0]
    for key in ('joint_pos', 'body_pos_w', 'body_quat_w'):
        if key not in first:
            raise ValueError(f"Base motion must contain '{key}'")

    needs_regeneration = any(
        ('joint_vel' not in traj)
        or ('body_lin_vel_w' not in traj)
        or ('body_ang_vel_w' not in traj)
        or (np.asarray(traj['body_pos_w']).shape[1] <= 1)
        for traj in all_base_data
    )
    if needs_regeneration:
        print("[INFO] Regenerating simulation data for base motions to build full RL tracking schema...")
        all_base_data = cast(List[Dict[str, Any]], regenerate_sim_data(all_base_data))

    all_base_data = _normalize_contact_channel_count(all_base_data)

    for idx, traj in enumerate(all_base_data):
        traj = _restore_source_metadata(traj, traj.get('file_path'))
        if 'traj_length' not in traj:
            traj['traj_length'] = np.array([get_trajectory_length(traj)], dtype=np.int64)
        all_base_data[idx] = traj

    if should_merge_directory:
        merged_path = base_motion_path / "_merged_batched.npz"
        print(f"[INFO] Merging {len(all_base_data)} trajectories into {merged_path} ...")
        merged = _merge_trajectories_to_batched(all_base_data)
        np.savez(merged_path, **cast(Dict[str, Any], merged))
        base_motion_path = merged_path
        print(f"[INFO] Saved merged batched file ({len(all_base_data)} trajs).")

    return all_base_data, base_motion_path

def weighted_sum_trajectory(neighbors: List[Elite], output_path: Path):
    try:
        # Weights = fitness scores
        weights = np.array([e.fitness for e in neighbors])
        total_weight = np.sum(weights)
        if total_weight < 1e-6:
            # Avoid division by zero if all fitnesses are 0 (unlikely given threshold)
            normalized_weights = np.ones_like(weights) / len(weights)
        else:
            normalized_weights = weights / total_weight
        
        # Use first neighbor as template
        template = neighbors[0].data
        new_data = template.copy()
        
        T = template['joint_pos'].shape[0]
        
        # 1. Accumulate Linear Terms
        acc_joint_pos = np.zeros_like(template['joint_pos'])
        acc_root_pos = np.zeros((T, 3))
        
        for i, elite in enumerate(neighbors):
            w = normalized_weights[i]
            
            # Joint Pos
            if 'joint_pos' in elite.data:
                acc_joint_pos += w * elite.data['joint_pos']
                
            # Root Pos (Index 0)
            if 'body_pos_w' in elite.data:
                acc_root_pos += w * elite.data['body_pos_w'][:, 0, :]
                
        # Assign linear results
        new_data['joint_pos'] = acc_joint_pos
        if 'body_pos_w' in new_data:
            new_bp = new_data['body_pos_w'].copy()
            new_bp[:, 0, :] = acc_root_pos
            new_data['body_pos_w'] = new_bp

        # 2. Markley's Method for Weighted Quaternion Average
        # See: "Averaging Quaternions", F. Landis Markley
        if 'body_quat_w' in neighbors[0].data:
            # Gather all root quaternions: (M, T, 4)
            quats_list = [e.data['body_quat_w'][:, 0, :] for e in neighbors]
            quats = np.stack(quats_list, axis=0) # (M, T, 4)
            
            # Canonicalize: if w < 0, flip q to -q
            # quats[..., 0] is the scalar part w
            w_negative = quats[..., 0] < 0 # (M, T)
            signs = np.where(w_negative, -1.0, 1.0) # (M, T)
            signs = signs[..., None] # (M, T, 1)
            Q = quats * signs
            
            # Compute Weighted Matrix A = sum(w_i * q_i * q_i^T)
            # Outer product per (m, t): (M, T, 4, 1) * (M, T, 1, 4) -> (M, T, 4, 4)
            Q_outer = Q[..., :, None] * Q[..., None, :] 
            
            # Reshape weights for broadcasting: (M, 1, 1, 1)
            W = normalized_weights[:, None, None, None]
            
            # Sum over M (neighbors) axis -> (T, 4, 4)
            A = np.sum(W * Q_outer, axis=0)
            
            # Find dominant eigenvector for each time step t
            # np.linalg.eigh returns eigenvalues/vectors of Hermitian matrices
            # vals: (T, 4), vecs: (T, 4, 4)
            # Eigenvalues are sorted ascending, so last column is dominant eigenvector
            vals, vecs = np.linalg.eigh(A)
            avg_quat = vecs[..., -1] # (T, 4)
            
            # Ensure temporal continuity (sign flipping ambiguity of eigenvectors)
            # Flip q_t if dot(q_t, q_{t-1}) < 0
            for t in range(1, T):
                if np.dot(avg_quat[t], avg_quat[t-1]) < 0:
                    avg_quat[t] = -avg_quat[t]
                    
            # Inject Root Quat
            if 'body_quat_w' in new_data:
                new_bq = new_data['body_quat_w'].copy()
                new_bq[:, 0, :] = avg_quat
                new_data['body_quat_w'] = new_bq
                
        np.savez(output_path, **new_data)

    except Exception as e:
        print(f"[Warn] Weighted sum failed (likely shape mismatch): {e}")

def batch_elites(elites: Union[List[Elite], Dict[Any, Elite]], output_path: Path) -> bool:
    """Batch elites into a single NPZ file, padding variable-length trajectories to the max length."""
    if not elites:
        print("No elites to batch.")
        return False
    
    if isinstance(elites, dict):
        elite_list = list(elites.values())
    else:
        elite_list = elites
        
    batched_data = {}
    
    # Use first elite as template
    first_data = elite_list[0].data
    keys = list(first_data.keys())
    
    print(f"Batching {len(elite_list)} elites...")

    # Pre-compute minimum EEF channel count across all elites for contact keys.
    # Base-motion elites may have 4 EEFs while pipeline elites have 2; trimming to
    # the minimum shared prefix lets batch_elites preserve the key instead of
    # silently dropping it due to an inner-dimension mismatch.
    _CONTACT_KEYS = {"contact_indicators", "contact_positions"}
    _contact_min_eef: dict[str, int] = {}
    for ck in _CONTACT_KEYS:
        eef_counts = []
        for e in elite_list:
            arr = e.data.get(ck)
            if arr is not None and isinstance(arr, np.ndarray) and arr.ndim >= 2:
                eef_counts.append(arr.shape[-1] if ck == "contact_indicators" else arr.shape[-2])
        if eef_counts:
            _contact_min_eef[ck] = min(eef_counts)

    for key in keys:
        arrays = []
        valid_key = True
        target_shape = None
        time_dim_len = None  # Track the time dimension for padding
        
        for i, e in enumerate(elite_list):
            if key not in e.data:
                valid_key = False
                break
            
            arr = e.data[key]
            
            # Skip if not array
            if not isinstance(arr, (np.ndarray, torch.Tensor)):
                valid_key = False
                break

            # Handle dimensions
            # We expect (T, ...) or (1, T, ...)
            # We want to stack to (N, T, ...)
            
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]

            # Trim contact arrays to the minimum shared EEF count so that
            # base-motion elites (4 EEFs) and pipeline elites (2 EEFs) can
            # be stacked together without dropping the key entirely.
            if key in _contact_min_eef and isinstance(arr, np.ndarray):
                min_eef = _contact_min_eef[key]
                if key == "contact_indicators" and arr.ndim >= 1:
                    arr = arr[..., :min_eef]
                elif key == "contact_positions" and arr.ndim >= 2:
                    arr = arr[..., :min_eef, :]

            if i == 0:
                target_shape = arr.shape
                time_dim_len = arr.shape[0] if arr.ndim >= 1 else None
            else:
                current_shape = arr.shape
                
                if current_shape != target_shape and arr.ndim >= 1:
                    # Variable-length: shapes differ in first dim (time), pad to max
                    if arr.shape[1:] == target_shape[1:]:
                        # Same non-time dimensions, just different time length
                        new_max = max(time_dim_len, arr.shape[0])
                        if new_max > time_dim_len:
                            # Need to extend all previously collected arrays
                            for j in range(len(arrays)):
                                if arrays[j].shape[0] < new_max:
                                    deficit = new_max - arrays[j].shape[0]
                                    pad = np.repeat(arrays[j][-1:], deficit, axis=0)
                                    arrays[j] = np.concatenate([arrays[j], pad], axis=0)
                            time_dim_len = new_max
                            target_shape = (new_max,) + target_shape[1:]
                        # Pad current array if needed
                        if arr.shape[0] < time_dim_len:
                            deficit = time_dim_len - arr.shape[0]
                            pad = np.repeat(arr[-1:], deficit, axis=0)
                            arr = np.concatenate([arr, pad], axis=0)
                    else:
                        # Non-time dimensions differ — skip this key
                        valid_key = False
                        break
            
            arrays.append(arr)
            
        if valid_key and arrays:
            try:
                batched_data[key] = np.stack(arrays, axis=0)
            except Exception as e:
                print(f"Error stacking {key}: {e}")
    
    # Add traj_length metadata if elites have it
    has_traj_lengths = any(getattr(e, 'traj_length', -1) > 0 for e in elite_list)
    if has_traj_lengths:
        lengths = np.array([getattr(e, 'traj_length', -1) for e in elite_list], dtype=np.int64)
        batched_data['traj_lengths'] = lengths
    
    if not batched_data:
        return False
        
    np.savez(output_path, **batched_data)
    print(f"Saved batched elites to {output_path}")
    return True

def compute_batch_metrics(recorded_data: Dict[str, np.ndarray], num_envs: int, task_name: str, evo_task_name: Optional[str] = None, base_object_pos_w: Optional[np.ndarray] = None, traj_lengths: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Compute metrics for all environments in memory to avoid I/O overhead.
    
    Parameters
    ----------
    traj_lengths : np.ndarray, optional
        Per-environment real trajectory lengths (before padding). Shape (num_envs,).
        When provided, rewards are masked so that only timesteps within the real
        trajectory contribute to metrics. If None, all timesteps are used.
    """
    results = []
    
    t_start = time.time()
    print(f"[Timer] Entering compute_batch_metrics...")
    
    # Data extraction
    # Ensure arrays are at least 2D (T, Envs, ...)
    body_pos = recorded_data['body_pos_w'] # (T, Envs, N, 3)
    
    # Handle case where single env might have squeezed dims (though we usually run >1)
    if body_pos.ndim == 3: # (T, N, 3) -> (T, 1, N, 3)
        body_pos = body_pos[:, None, :, :]
        
    T = body_pos.shape[0]
    
    # --- Variable-length time mask ---
    # valid_mask: (T, num_envs) boolean, True where timestep < real length
    if traj_lengths is not None:
        time_idx = np.arange(T)[:, None]  # (T, 1)
        valid_mask = time_idx < traj_lengths[None, :]  # (T, num_envs)
        # Per-env count of valid timesteps
        valid_counts = traj_lengths.astype(np.float64)  # (num_envs,)
    else:
        valid_mask = np.ones((T, num_envs), dtype=bool)
        valid_counts = np.full(num_envs, T, dtype=np.float64)
    
    # Ref body pos
    ref_body_pos = recorded_data.get('ref_body_pos')
    if ref_body_pos is None:
        ref_body_pos = recorded_data.get('ref_body_pos_w')
        
    if ref_body_pos is not None and ref_body_pos.ndim == 3:
        ref_body_pos = ref_body_pos[:, None, :, :]
        
    # Filter body_pos to match ref_body_pos using body names
    metric_body_pos = body_pos
    if 'body_names' in recorded_data:
        try:
            # Load env config to get tracked body names
            # Use the RL task name here (task_name) as it defines the robot config
            env_cfg = load_env_cfg(task_name)
            if env_cfg.commands and "motion" in env_cfg.commands:
                tracked_names = env_cfg.commands["motion"].body_names
                
                # Check if body_pos is already filtered (dim match)
                if body_pos.shape[2] == len(tracked_names):
                    # Already filtered, do nothing
                    metric_body_pos = body_pos
                else:
                    all_names = recorded_data['body_names']
                    
                    # Find indices
                    indices = [list(all_names).index(name) for name in tracked_names if name in all_names]
                    
                    if len(indices) == len(tracked_names):
                        # Slice the body positions
                        metric_body_pos = body_pos[:, :, indices, :]
                        # print(f"Filtered body pos from {body_pos.shape} to {metric_body_pos.shape}")
                    else:
                        print(f"[Warn] Could not find all tracked bodies in recorded body names.")
        except Exception as e:
            print(f"[Warn] Failed to filter body positions: {e}")

    # Inject traj_lengths into recorded_data so metric functions can use it
    if traj_lengths is not None and 'traj_lengths' not in recorded_data:
        recorded_data['traj_lengths'] = traj_lengths

    t_prep = time.time()
    print(f"[Timer] Data prep: {t_prep - t_start:.4f}s")

    # Task Metrics
    # Use the Evo task name here to determine which metrics to calculate
    task_config = get_task_config(evo_task_name)
    task_metrics_values = {}
    
    last_percent = 0.5
    start_idx = int(T * (1.0 - last_percent))
    
    for dim in task_config.dimensions:
        task_metrics_values[dim.name] = dim.metric_fn(recorded_data, start_idx)
    
    t_tm = time.time()
    print(f"[Timer] Task metrics: {t_tm - t_prep:.4f}s")
    
    # Contact
    contact_reward = recorded_data['reward_contact_match']
    if contact_reward.ndim == 1: contact_reward = contact_reward[:, None]
    
    # Masked contact: only count within valid timesteps
    masked_contact = contact_reward * valid_mask
    contact_counts = np.sum(masked_contact > 1e-6, axis=0)
    contact_pcts = (contact_counts / valid_counts) * 100.0
    
    # Bad Termination
    bad_term_reward = recorded_data.get('reward_bad_termination')
    has_bad_term = np.zeros(num_envs, dtype=bool)
    if bad_term_reward is not None:
        if bad_term_reward.ndim == 1: bad_term_reward = bad_term_reward[:, None]
        # Check if any frame within valid range has significant negative reward
        # Skip the first frame as it is always -100
        masked_bad_term = bad_term_reward * valid_mask
        has_bad_term = np.any(masked_bad_term[1:] < -1.0, axis=0)
        
    # Rewards
    def get_reward_mean(key):
        arr = recorded_data.get(key, np.zeros((T, num_envs)))
        if arr.ndim == 1: arr = arr[:, None]
        # Masked mean: only average over valid timesteps
        masked = arr * valid_mask
        return np.sum(masked, axis=0) / valid_counts

    # r_pos = get_reward_mean('reward_motion_body_pos')
    r_vel = get_reward_mean('reward_motion_body_lin_vel')
    # r_root_pos = get_reward_mean('reward_motion_global_root_pos')
    r_root_ori = get_reward_mean('reward_motion_global_root_ori')

    # Compute Body Position tracking Score 
    pos_diff = metric_body_pos - ref_body_pos # (T, Envs, N, 3)
    pos_error_sq = np.sum(pos_diff**2, axis=-1) # (T, Envs, N)
    pos_mse = np.mean(pos_error_sq, axis=-1) # (T, Envs)
    r_pos_t = np.exp(-pos_mse / (0.05**2)) # (T, Envs)
    # Masked mean over time
    r_pos = np.sum(r_pos_t * valid_mask, axis=0) / valid_counts  # (Envs,)

    # Compute root Position tracking Score
    root_pos_diff = pos_diff[:, :, 0, :] # (T, Envs, 3)
    root_pos_error_sq = np.sum(root_pos_diff**2, axis=-1) # (T, Envs)
    r_root_pos_t = np.exp(-root_pos_error_sq / (0.05**2)) # (T, Envs)
    r_root_pos = np.sum(r_root_pos_t * valid_mask, axis=0) / valid_counts  # (Envs,)

    # Compute leg eef tracking Score
    left_leg_eef_pos_diff = pos_diff[:, :, 3, :] # (T, Envs, 3)
    right_leg_eef_pos_diff = pos_diff[:, :, 6, :] # (T, Envs, 3)
    leg_eef_pos_diff = 0.5 * (left_leg_eef_pos_diff + right_leg_eef_pos_diff)
    leg_pos_error_sq = np.sum(leg_eef_pos_diff**2, axis=-1) # (T, Envs)
    r_leg_pos_t = np.exp(-leg_pos_error_sq / (0.04**2)) # (T, Envs)
    r_leg_pos = np.sum(r_leg_pos_t * valid_mask, axis=0) / valid_counts  # (Envs,)
    
    t_rewards = time.time()
    print(f"[Timer] Core rewards: {t_rewards - t_tm:.4f}s")
    
    # Laplacian
    laplacian_scores = np.ones(num_envs) # Default 1.0 (Removed calculation for speed)
    
    t_lap_end = time.time()

    # Assemble results

    # Compute object stability (Z-deviation penalty)
    r_obj_stable = np.ones(num_envs) # Default 1.0
    
    # Compare simulated object Z against reference (diffusion-generated) object Z
    # over the entire valid trajectory range (masked by traj_lengths if provided).
    if base_object_pos_w is not None and 'object_pos_w' in recorded_data:
        try:
            elite_full_pos = recorded_data['object_pos_w']  # (T, Envs, 3) or (T, Envs, N, 3)
            ref_obj_pos = base_object_pos_w                 # (T, Envs, 3) per-env reference

            # Extract Z from simulated object
            if elite_full_pos.ndim == 4:
                elite_obj_z = elite_full_pos[:, :, 0, 2]  # (T, Envs)
            elif elite_full_pos.ndim == 3:
                elite_obj_z = elite_full_pos[:, :, 2]     # (T, Envs)
            else:
                elite_obj_z = None

            # Extract Z from reference (per-env diffusion trajectory)
            if ref_obj_pos.ndim == 4:
                ref_obj_z = ref_obj_pos[:, :, 0, 2]  # (T, Envs)
            elif ref_obj_pos.ndim == 3:
                ref_obj_z = ref_obj_pos[:, :, 2]     # (T, Envs)
            elif ref_obj_pos.ndim == 2:
                # Single reference (T, 3) — broadcast to all envs
                ref_obj_z = ref_obj_pos[:, 2:3] * np.ones((1, num_envs))  # (T, Envs)
            else:
                ref_obj_z = None

            if elite_obj_z is not None and ref_obj_z is not None:
                T_curr = elite_obj_z.shape[0]
                T_ref = ref_obj_z.shape[0]

                # Match lengths
                if T_ref < T_curr:
                    pad = np.broadcast_to(ref_obj_z[-1:, :], (T_curr - T_ref, ref_obj_z.shape[1]))
                    ref_obj_z = np.concatenate([ref_obj_z, pad], axis=0)
                else:
                    ref_obj_z = ref_obj_z[:T_curr]

                PENALTY_SCALE = 0.1

                diff = np.abs(elite_obj_z - ref_obj_z)  # (T, Envs)
                # Mask to valid timesteps only
                diff_masked = diff * valid_mask  # zeros outside valid range
                max_d = np.max(diff_masked, axis=0)  # (Envs,) worst deviation per env
                r_obj_stable = np.exp(-max_d / PENALTY_SCALE)

        except Exception as e:
            print(f"[Warn] Failed to compute Z-stability: {e}")
            import traceback; traceback.print_exc()
            r_obj_stable = np.zeros(num_envs)
    else:
        pass

    t_obj_stab = time.time()
    print(f"[Timer] Object Stability: {t_obj_stab - t_lap_end:.4f}s")

    # Vectorize Loop
    fitness_scores = r_pos * r_vel * (100.0/100.0) * r_root_pos * r_root_ori * r_leg_pos * r_obj_stable
    
    # Apply conditions
    bad_mask = has_bad_term | (contact_pcts < -100)
    fitness_scores[bad_mask] = 0.0

    # Gather results
    # Pre-extract task metrics to avoid dict lookups in loop
    tm_keys = [dim.name for dim in task_config.dimensions]
    tm_values = {k: task_metrics_values[k] for k in tm_keys} # Arrays of (Envs,)

    t_assembly_start = time.time()
    for i in range(num_envs):
        # Collect task specific metrics for this env
        env_task_metrics = {k: float(v[i]) for k, v in tm_values.items()}

        results.append({
            "task_metrics": env_task_metrics,
            "fitness": float(fitness_scores[i]),
            "metrics": {
                "pose_reward": float(r_pos[i]),
                "vel_reward": float(r_vel[i]),
                "root_pos_reward": float(r_root_pos[i]),
                "root_ori_reward": float(r_root_ori[i]),
                "contact_pct": float(contact_pcts[i]),
                "laplacian_score": float(laplacian_scores[i]),
                "leg_pos_reward": float(r_leg_pos[i]),
                "obj_stable_reward": float(r_obj_stable[i])
            }
        })
    print(f"[Timer] Results Assembly Loop: {time.time() - t_assembly_start:.4f}s")
        
    return results


def extract_contacts_from_distance(
  eef_positions: np.ndarray,
  object_position: np.ndarray,
  object_rotation: np.ndarray | None = None,
  object_size: np.ndarray | None = None,
    object_geom_local_position: np.ndarray | None = None,
    object_geom_local_rotation: np.ndarray | None = None,
  threshold: float = 0.05,
) -> dict[str, Any]:
    return _shared_extract_contacts_from_distance(
        eef_positions,
        object_position,
        object_rotation=object_rotation,
        object_size=object_size,
        object_geom_local_position=object_geom_local_position,
        object_geom_local_rotation=object_geom_local_rotation,
        threshold=threshold,
    )


def regenerate_sim_data_multi_gpu(
    input_data: List[Dict[str, Any]],
    devices: list[str],
) -> List[Dict[str, Any]]:
    """Split *input_data* across *devices* and run ``regenerate_sim_data`` on each.

    When ``len(devices) == 1`` this is equivalent to a direct call.  Each chunk
    is completely independent (separate Scene / Simulation), so no inter-device
    communication is needed.

    Note: Warp's CUDA graph capture (``ScopedCapture``) is not thread-safe, so
    chunks must run sequentially.  The benefit of multiple GPUs here is reduced
    per-GPU memory pressure (fewer envs per device).
    """
    from humanoid_dart.dart.multi_device import split_list

    if not input_data:
        return input_data

    chunks = split_list(input_data, len(devices))
    results: list[list[Dict[str, Any]]] = []

    for chunk, dev in zip(chunks, devices):
        if not chunk:
            continue
        chunk_result = regenerate_sim_data(chunk, device=dev)
        if isinstance(chunk_result, list):
            results.append(chunk_result)
        else:
            results.append([chunk_result])

    # Flatten back into a single list preserving original order
    return [item for sublist in results for item in sublist]


def regenerate_sim_data(input_data: Dict[str, Any] | List[Dict[str, Any]], device: str = "cuda:0") -> Dict[str, Any] | List[Dict[str, Any]]:
    """
    Regenerate body positions, velocities, and contacts by running a simulation
    driven by the joint positions in traj_data.
    Supports batch processing if input_data is a list of dicts.
    Handles variable-length trajectories by padding to the maximum length
    and storing 'traj_length' metadata on each trajectory dict.
    """
    print("Regenerating simulation data...")
    
    is_batch = isinstance(input_data, list)
    traj_list = input_data if is_batch else [input_data]
    num_envs = len(traj_list)
    
    if num_envs == 0:
        return input_data

    # Assume all trajectories have same FPS (or at least we use the first one's FPS)
    first_traj = traj_list[0]
    fps_val = first_traj.get("fps", 50.0)
    fps = float(np.asarray(fps_val).flat[0])
    dt = 1.0 / fps

    # --- Variable-length support ---
    # Keep real lengths for metadata, but simulate over the padded array length.
    traj_lengths = [get_trajectory_length(t) for t in traj_list]
    padded_lengths = [t["joint_pos"].shape[0] for t in traj_list]
    T = max(padded_lengths)

    # Pad shorter trajectories by repeating their last frame
    for i, t in enumerate(traj_list):
        current_len = padded_lengths[i]
        if current_len < T:
            for key in list(t.keys()):
                if isinstance(t[key], np.ndarray) and t[key].ndim >= 1 and t[key].shape[0] == current_len:
                    deficit = T - current_len
                    pad = np.repeat(t[key][-1:], deficit, axis=0)
                    t[key] = np.concatenate([t[key], pad], axis=0)
        # Store real length as metadata
        t['traj_length'] = np.array([traj_lengths[i]], dtype=np.int64)
    
    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = dt
    
    has_object = "object_pos_w" in first_traj
    
    if has_object:
        scene = Scene(unitree_g1_flat_tracking_env_cfg_box().scene, device=device)
    else:
        scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
        
    model = scene.compile()
    sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    
    robot: Entity = scene["robot"]
    box: Optional[Entity] = scene.entities.get("box") if hasattr(scene, "entities") else None
    
    # Pre-convert data to tensors
    # joint_pos: List[(T, J)] -> (T, N, J)
    joint_pos_list = [t["joint_pos"] for t in traj_list]
    traj_joint_pos = torch.from_numpy(np.stack(joint_pos_list, axis=1)).to(device).float()
    
    # Recompute joint velocities using gradient
    traj_joint_vel = torch.gradient(traj_joint_pos, spacing=dt, dim=0)[0]
    
    # Root data
    traj_root_pos = None
    traj_root_quat = None
    traj_root_lin_vel = None
    traj_root_ang_vel = None
    
    # Check if body_pos_w exists in input
    if "body_pos_w" in first_traj:
        # body_pos_w: (T, num_bodies, 3) -> we want root (body 0)
        # Stack: (T, N, 3)
        root_pos_list = [t["body_pos_w"][:, 0] for t in traj_list]
        traj_root_pos = torch.from_numpy(np.stack(root_pos_list, axis=1)).to(device).float()
        
        root_quat_list = [t["body_quat_w"][:, 0] for t in traj_list]
        traj_root_quat = torch.from_numpy(np.stack(root_quat_list, axis=1)).to(device).float()
        
        # Recompute root linear velocity
        traj_root_lin_vel = torch.gradient(traj_root_pos, spacing=dt, dim=0)[0]
            
        # Recompute root angular velocity
        q = traj_root_quat
        if q.shape[0] >= 3:
            q_prev, q_next = q[:-2], q[2:]
            q_rel = quat_mul(q_next, quat_conjugate(q_prev))
            omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
            omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        else:
            omega = torch.zeros_like(traj_root_pos)
        traj_root_ang_vel = omega

    # Object data
    traj_obj_pos = None
    traj_obj_quat = None
    traj_obj_lin_vel = None
    traj_obj_ang_vel = None
    if has_object:
        if box is None:
            print("[WARN] regenerate_sim_data: 'box' entity not found in scene but object_pos_w is present. Contact detection will use trajectory object positions only.")
        obj_pos_list = [t["object_pos_w"] for t in traj_list]
        traj_obj_pos = torch.from_numpy(np.stack(obj_pos_list, axis=1)).to(device).float()        
        
        obj_quat_list = [t["object_quat_w"] for t in traj_list]        
        traj_obj_quat = torch.from_numpy(np.stack(obj_quat_list, axis=1)).to(device).float()

        # Object linear velocity via finite differencing
        traj_obj_lin_vel = torch.gradient(traj_obj_pos, spacing=dt, dim=0)[0]

        # Object angular velocity via finite differencing of quaternions
        oq = traj_obj_quat
        if oq.shape[0] >= 3:
            oq_prev, oq_next = oq[:-2], oq[2:]
            oq_rel = quat_mul(oq_next, quat_conjugate(oq_prev))
            obj_omega = axis_angle_from_quat(oq_rel) / (2.0 * dt)
            obj_omega = torch.cat([obj_omega[:1], obj_omega, obj_omega[-1:]], dim=0)
        else:
            obj_omega = torch.zeros_like(traj_obj_pos)
        traj_obj_ang_vel = obj_omega

    # Output containers
    out_body_pos_w = []
    out_body_quat_w = []
    out_body_lin_vel_w = []
    out_body_ang_vel_w = []
    
    scene.reset()
    
    # 3. Simulation Loop
    for t in range(T):
        # Set Robot State
        root_states = robot.data.default_root_state.clone() # (N, 13)
        
        if traj_root_pos is not None:
             root_pos = traj_root_pos[t] # (N, 3)
             root_quat = traj_root_quat[t] # (N, 4)
             root_lin_vel = traj_root_lin_vel[t] # (N, 3)
             root_ang_vel = traj_root_ang_vel[t] # (N, 3)
             
             root_states[:, 0:3] = root_pos
             root_states[:, :2] += scene.env_origins[:, :2]
             root_states[:, 3:7] = root_quat
             root_states[:, 7:10] = root_lin_vel
             # Convert ang vel to local
             root_states[:, 10:] = quat_apply_inverse(root_quat, root_ang_vel)
        
        robot.write_root_state_to_sim(root_states)
        
        # Joint State
        joint_pos = robot.data.default_joint_pos.clone() # (N, J)
        joint_vel = robot.data.default_joint_vel.clone() # (N, J)
        
        jp = traj_joint_pos[t] # (N, J)
        jv = traj_joint_vel[t] # (N, J)
        
        joint_pos[:, :] = jp
        joint_vel[:, :] = jv
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        
        # Step
        sim.forward()
        scene.update(dt)
        
        # Read Back
        out_body_pos_w.append(robot.data.body_link_pos_w.cpu().numpy())
        out_body_quat_w.append(robot.data.body_link_quat_w.cpu().numpy())
        out_body_lin_vel_w.append(robot.data.body_link_lin_vel_w.cpu().numpy())
        out_body_ang_vel_w.append(robot.data.body_link_ang_vel_w.cpu().numpy())

    # 4. Reassemble
    # Stack time: (T, N, num_bodies, 3)
    stack_body_pos_w = np.stack(out_body_pos_w, axis=0)
    stack_body_quat_w = np.stack(out_body_quat_w, axis=0)
    stack_body_lin_vel_w = np.stack(out_body_lin_vel_w, axis=0)
    stack_body_ang_vel_w = np.stack(out_body_ang_vel_w, axis=0)
    
    # Distribute back to traj_list
    env_origins = scene.env_origins.cpu().numpy()
    stack_joint_vel = traj_joint_vel.cpu().numpy()

    # Object velocity arrays (T, N, 3)
    stack_obj_lin_vel = traj_obj_lin_vel.cpu().numpy() if traj_obj_lin_vel is not None else None
    stack_obj_ang_vel = traj_obj_ang_vel.cpu().numpy() if traj_obj_ang_vel is not None else None

    # --- Contact detection (matching victor_npz_to_npz logic) ---
    stack_contact_indicators = None
    stack_contact_positions = None
    if traj_obj_pos is not None:
        desired_num_eefs = _desired_contact_eef_count(traj_list)
        eef_names = list(CONTACT_EEF_BODY_NAMES[:desired_num_eefs])
        eef_body_indices = []
        for name in eef_names:
            if name in robot.body_names:
                eef_body_indices.append(robot.body_names.index(name))
            else:
                eef_body_indices.append(-1)

        # Get object box size from MuJoCo geom.
        # Strategy 1: use the box Entity's indexing (works even when body names are
        # namespaced in the compiled multi-env model).
        object_size = None
        object_geom_local_position = None
        object_geom_local_rotation = None
        if box is not None and hasattr(box, "indexing"):
            try:
                for gid in box.indexing.geom_ids.cpu().numpy():
                    if sim.mj_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                        object_size = sim.mj_model.geom_size[gid].copy()
                        object_geom_local_position = sim.mj_model.geom_pos[gid].copy()
                        object_geom_local_rotation = sim.mj_model.geom_quat[gid].copy()
                        break
            except Exception as _e:
                print(f"[WARN] Contact: box entity geom lookup failed: {_e}")

        # Strategy 2: fall back to mj_name2id (bare names, no namespace prefix)
        if object_size is None:
            object_body_names = ["largebox_link", "box", "object"]
            for bname in object_body_names:
                bid = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, bname)
                if bid >= 0:
                    geom_ids_arr = np.where(sim.mj_model.geom_bodyid == bid)[0]
                    for gid in geom_ids_arr:
                        if sim.mj_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                            object_size = sim.mj_model.geom_size[gid].copy()
                            object_geom_local_position = sim.mj_model.geom_pos[gid].copy()
                            object_geom_local_rotation = sim.mj_model.geom_quat[gid].copy()
                            break
                    if object_size is not None:
                        break

        if object_size is None:
            print("[WARN] Contact: object box geom not found — using sphere threshold fallback. "
                  "Contacts will only register within 5cm of box center (likely wrong).")

        # Diagnostics
        print(f"[Contact] eef_body_indices: {list(zip(eef_names, eef_body_indices))}")
        print(f"[Contact] object_size: {object_size}  (None → sphere threshold fallback)")
        missing = [n for n, idx in zip(eef_names, eef_body_indices) if idx < 0]
        if missing:
            print(f"[WARN] Contact: EEF bodies not found in robot.body_names: {missing}")
            print(f"[WARN] Available body_names sample: {robot.body_names[:10]}")

        contact_threshold = 0.05
        num_eefs = len(eef_names)

        # --- Fully vectorized contact detection (no Python loops over T×N) ---
        # Gather EEF positions: (T, N, num_eefs, 3) from stack_body_pos_w (T, N, num_bodies, 3)
        valid_eef_mask = np.array([idx >= 0 for idx in eef_body_indices])  # (num_eefs,)
        safe_indices = np.array([idx if idx >= 0 else 0 for idx in eef_body_indices])  # (num_eefs,)
        eef_positions = stack_body_pos_w[:, :, safe_indices, :]  # (T, N, num_eefs, 3)

        # obj positions/quats on GPU for quat_apply_inverse
        obj_pos_3d = traj_obj_pos  # (T, N, 3) on device
        obj_quat_4d = traj_obj_quat  # (T, N, 4) on device

        # Move eef positions to GPU tensor for batched quat math
        eef_t = torch.from_numpy(eef_positions).to(device).float()  # (T, N, num_eefs, 3)

        # Expand obj arrays: (T, N, 1, 3) and (T, N, 1, 4)
        obj_pos_exp = obj_pos_3d.unsqueeze(2)   # (T, N, 1, 3)
        obj_quat_exp = obj_quat_4d.unsqueeze(2)  # (T, N, 1, 4)

        effective_obj_pos_exp = obj_pos_exp
        effective_obj_quat_exp = obj_quat_exp
        if object_geom_local_position is not None or object_geom_local_rotation is not None:
            geom_local_position = np.zeros(3, dtype=np.float32)
            if object_geom_local_position is not None:
                geom_local_position = np.asarray(object_geom_local_position, dtype=np.float32)
            geom_local_rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            if object_geom_local_rotation is not None:
                geom_local_rotation = np.asarray(object_geom_local_rotation, dtype=np.float32)

            geom_local_pos_t = torch.from_numpy(geom_local_position).to(device).float()
            geom_local_quat_t = torch.from_numpy(geom_local_rotation).to(device).float()
            geom_local_pos_exp = geom_local_pos_t.view(1, 1, 1, 3).expand(T, num_envs, 1, 3)
            geom_local_quat_exp = geom_local_quat_t.view(1, 1, 1, 4).expand(T, num_envs, 1, 4)

            rotated_geom_offset = quat_apply(
                obj_quat_exp.reshape(-1, 4),
                geom_local_pos_exp.reshape(-1, 3),
            ).reshape(T, num_envs, 1, 3)
            effective_obj_pos_exp = obj_pos_exp + rotated_geom_offset
            effective_obj_quat_exp = quat_mul(
                obj_quat_exp.reshape(-1, 4),
                geom_local_quat_exp.reshape(-1, 4),
            ).reshape(T, num_envs, 1, 4)

        # Relative position in world frame: (T, N, num_eefs, 3)
        rel_pos = eef_t - effective_obj_pos_exp

        if object_size is not None:
            # Transform to object-local frame (vectorized)
            # quat_apply_inverse expects (*, 4) and (*, 3)
            flat_quat = effective_obj_quat_exp.expand(-1, -1, num_eefs, -1).reshape(-1, 4)
            flat_rel = rel_pos.reshape(-1, 3)
            local_pos = quat_apply_inverse(flat_quat, flat_rel).reshape(T, num_envs, num_eefs, 3)

            # Check bounding box
            bounds = torch.from_numpy(object_size + contact_threshold).to(device).float()  # (3,)
            within_bounds = (local_pos.abs() <= bounds).all(dim=-1)  # (T, N, num_eefs) bool

            # Clamp to surface for contact position
            obj_size_t = torch.from_numpy(object_size).to(device).float()
            clamped_local = local_pos.clamp(-obj_size_t, obj_size_t)  # (T, N, num_eefs, 3)

            # Transform clamped back to world
            flat_clamped = clamped_local.reshape(-1, 3)
            contact_world = quat_apply(flat_quat, flat_clamped).reshape(T, num_envs, num_eefs, 3) + effective_obj_pos_exp

            all_ci = within_bounds.cpu().numpy()  # (T, N, num_eefs)
            all_cp_t = contact_world.cpu().numpy().astype(np.float32)

            # Set non-contact positions to NaN
            all_cp = np.where(all_ci[..., None], all_cp_t, np.nan)
        else:
            # Sphere fallback: distance from center < threshold
            dist = rel_pos.norm(dim=-1)  # (T, N, num_eefs)
            all_ci = (dist <= contact_threshold).cpu().numpy()
            obj_pos_cp = effective_obj_pos_exp.expand(-1, -1, num_eefs, -1).cpu().numpy().astype(np.float32)
            all_cp = np.where(all_ci[..., None], obj_pos_cp, np.nan)

        # Mask out invalid EEF indices
        if not valid_eef_mask.all():
            for ei in range(num_eefs):
                if not valid_eef_mask[ei]:
                    all_ci[:, :, ei] = False
                    all_cp[:, :, ei, :] = np.nan

        stack_contact_indicators = all_ci
        stack_contact_positions = all_cp
        total_contacts = all_ci.sum()
        print(f"[Contact] Total True entries: {total_contacts} / {all_ci.size} ({100*total_contacts/all_ci.size:.1f}%)")

    for i in range(num_envs):
        traj_list[i]["body_pos_w"] = stack_body_pos_w[:, i].copy()
        traj_list[i]["body_quat_w"] = stack_body_quat_w[:, i].copy()
        traj_list[i]["body_lin_vel_w"] = stack_body_lin_vel_w[:, i].copy()
        traj_list[i]["body_ang_vel_w"] = stack_body_ang_vel_w[:, i].copy()
        traj_list[i]["joint_vel"] = stack_joint_vel[:, i].copy()
        if stack_obj_lin_vel is not None:
            traj_list[i]["object_lin_vel_w"] = stack_obj_lin_vel[:, i].copy()
        if stack_obj_ang_vel is not None:
            traj_list[i]["object_ang_vel_w"] = stack_obj_ang_vel[:, i].copy()
        if stack_contact_indicators is not None:
            traj_list[i]["contact_indicators"] = stack_contact_indicators[:, i].copy()
        if stack_contact_positions is not None:
            traj_list[i]["contact_positions"] = stack_contact_positions[:, i].copy()
        
    if is_batch:
        return traj_list
    else:
        return traj_list[0]

def clean_batch_rollouts(recorded_data: Dict[str, np.ndarray]) -> None:
    """
    In-place clean up of recorded data discontinuities (e.g. resets at last frame) for the entire batch.
    Faster than doing it per-env in a loop.
    """
    # Keys to check for discontinuity
    keys_to_fix = ['joint_pos', 'body_pos_w']
    
    t_start = time.time()
    for key in keys_to_fix:
        if key in recorded_data:
            arr = recorded_data[key] # (T, Envs, ...)
            T = arr.shape[0]
            if T > 2:
                val_last = arr[-1]
                val_prev = arr[-2]
                val_prev2 = arr[-3]
                
                # Compute valid norms
                # Reshape to (Envs, -1)
                diff_last = np.linalg.norm((val_last - val_prev).reshape(arr.shape[1], -1), axis=1) # (Envs,)
                diff_prev = np.linalg.norm((val_prev - val_prev2).reshape(arr.shape[1], -1), axis=1) # (Envs,)
                
                # Mask where jump is detected
                jump_mask = (diff_last > 0.5) & (diff_last > 5.0 * (diff_prev + 1e-6))
                
                if np.any(jump_mask):
                    arr[-1][jump_mask] = arr[-2][jump_mask]
                    
    print(f"[Timer] clean_batch_rollouts: {time.time() - t_start:.4f}s")


def clean_rollout_and_overwrite(
    original_input: Dict[str, np.ndarray],
    recorded_data: Dict[str, np.ndarray],
    env_idx: int
) -> Dict[str, np.ndarray]:
    """
    Merges recorded simulation data with original input data.
    Assumes recorded_data is already cleaned via `clean_batch_rollouts`.
    """
    input_data = original_input.copy()
    
    # Keys to override from rollout
    override_keys = ['joint_pos', 'joint_vel', 
                     'object_pos_w', 'object_quat_w', 'object_lin_vel_w', 'object_ang_vel_w', 
                     'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w']
    
    for key in override_keys:
        if key in recorded_data:
            input_data[key] = recorded_data[key][:, env_idx].copy()
                
    return input_data


def overwrite_with_rollouts(
    regenerated_list: List[Dict[str, np.ndarray]],
    recorded_data: Dict[str, np.ndarray],
    elite_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Vectorized merge of recorded simulation data for multiple elite environments.
    Returns a single batched dict with shape (N_elites, T, ...) for each key.
    """
    override_keys = ['joint_pos', 'joint_vel',
                     'object_pos_w', 'object_quat_w', 'object_lin_vel_w', 'object_ang_vel_w',
                     'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w',
                     'contact_indicators', 'contact_positions']

    n_elites = len(elite_indices)
    if n_elites == 0:
        return {}

    # Use first elite as key template
    template = regenerated_list[elite_indices[0]]
    all_keys = list(template.keys())

    batched: Dict[str, np.ndarray] = {}

    # --- Override keys: slice entire index set from recorded_data at once ---
    for key in override_keys:
        if key in recorded_data:
            # recorded_data[key] is (T, Envs, ...). Slice all elite envs -> (T, N_elites, ...)
            sliced = recorded_data[key][:, elite_indices]
            # Transpose to (N_elites, T, ...)
            batched[key] = np.moveaxis(sliced, 0, 1).copy()

    # --- Non-override keys: stack from regenerated_list ---
    for key in all_keys:
        if key in batched:
            continue  # already handled above
        val = template[key]
        if not isinstance(val, np.ndarray):
            continue
        try:
            arrays = [regenerated_list[idx][key] for idx in elite_indices]
            batched[key] = np.stack(arrays, axis=0)
        except (KeyError, ValueError):
            pass  # skip non-stackable / missing keys

    return batched
