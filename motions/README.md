# Seed motions

`pick_place.npz` holds the base (seed) demonstrations for the pick-and-place
experiment (`base_motion` in `configs/pick_place.yaml`): 87 trajectories
from two source clips (`sub3_largebox_003`, `sub10_largebox_085`), 70 fps,
363-475 frames. `file_path` / `source_folder_name` record which clip each
trajectory comes from. Another file or a
directory of files can be passed with `--base_motion`.

Each file is a batched motion in the tracking (RL) schema, one entry per seed
trajectory (`B` trajectories, `T` frames, 30 G1 bodies, 29 joints, 2 end effectors):

| key | shape |
|---|---|
| `joint_pos`, `joint_vel` | `(B, T, 29)` |
| `body_pos_w`, `body_lin_vel_w`, `body_ang_vel_w` | `(B, T, 30, 3)` |
| `body_quat_w` | `(B, T, 30, 4)` (w, x, y, z) |
| `object_pos_w`, `object_lin_vel_w`, `object_ang_vel_w` | `(B, T, 3)` |
| `object_quat_w` | `(B, T, 4)` (w, x, y, z) |
| `contact_indicators` | `(B, T, 2)` |
| `contact_positions` | `(B, T, 2, 3)` |
| `fps` | `(B, 1)` |
| `traj_lengths` (optional) | `(B,)`, real lengths when trajectories are padded |

A directory of such `.npz` files is also accepted; the files are merged.

To convert raw kinematic trajectories (root pose, joint angles and object pose)
into this schema, use `python -m humanoid_dart.dart.convert_motion --help`.
