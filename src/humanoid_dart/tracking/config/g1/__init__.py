from mjlab.tasks.registry import register_mjlab_task
from humanoid_dart.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import unitree_g1_flat_tracking_env_cfg_box
from .rl_cfg import unitree_g1_tracking_ppo_runner_cfg_box

# Only the box tasks are registered: stock mjlab already registers the plain
# "Mjlab-Tracking-Flat-Unitree-G1*" ids.

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-Box",
  env_cfg=unitree_g1_flat_tracking_env_cfg_box(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg_box(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg_box(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation",
  env_cfg=unitree_g1_flat_tracking_env_cfg_box(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg_box(
    has_state_estimation=False, play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg_box(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
