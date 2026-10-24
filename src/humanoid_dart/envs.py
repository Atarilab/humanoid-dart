import torch
from mjlab.envs import ManagerBasedRlEnv


class DartRlEnv(ManagerBasedRlEnv):
  """ManagerBasedRlEnv with "reset" events applied after the command reset.

  Stock mjlab applies reset events before the managers are reset. Here they
  run after the command manager reset, so a reset event (e.g. placing the box
  from a per-env target tensor) overrides the pose written by the motion
  command's reference-state initialization.
  """

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    self.curriculum_manager.compute(env_ids=env_ids)
    # Reset the internal buffers of the scene elements.
    self.scene.reset(env_ids)

    # NOTE: This is order sensitive.
    self.extras["log"] = dict()
    # observation manager.
    info = self.observation_manager.reset(env_ids)
    self.extras["log"].update(info)
    # action manager.
    info = self.action_manager.reset(env_ids)
    self.extras["log"].update(info)
    # rewards manager.
    info = self.reward_manager.reset(env_ids)
    self.extras["log"].update(info)
    # curriculum manager.
    info = self.curriculum_manager.reset(env_ids)
    self.extras["log"].update(info)
    # command manager.
    info = self.command_manager.reset(env_ids)
    self.extras["log"].update(info)

    if "reset" in self.event_manager.available_modes:
      env_step_count = self._sim_step_counter // self.cfg.decimation
      self.event_manager.apply(
        mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
      )

    # event manager.
    info = self.event_manager.reset(env_ids)
    self.extras["log"].update(info)
    # termination manager.
    info = self.termination_manager.reset(env_ids)
    self.extras["log"].update(info)
    # reset the episode length buffer.
    self.episode_length_buf[env_ids] = 0
