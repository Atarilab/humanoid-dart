"""Unitree G1 (no hands) and the pick-and-place box.

The robot is stock mjlab's G1 with the rubber hands removed: the hand visual
meshes, the palm sites and the hand collision capsules are deleted, while the
wrist collision capsules (used for box contact) are kept.
"""

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
from mjlab.entity import EntityCfg

G1_ACTION_SCALE = g1_constants.G1_ACTION_SCALE

_REMOVED_GEOMS = ("left_hand_collision", "right_hand_collision")
_REMOVED_SITES = ("left_palm", "right_palm")
_REMOVED_MESHES = ("left_rubber_hand", "right_rubber_hand")


def get_g1_no_hands_spec() -> mujoco.MjSpec:
  spec = g1_constants.get_spec()
  for geom in list(spec.geoms):
    if geom.name in _REMOVED_GEOMS or geom.meshname in _REMOVED_MESHES:
      spec.delete(geom)
  for name in _REMOVED_SITES:
    spec.delete(spec.site(name))
  for name in _REMOVED_MESHES:
    spec.delete(spec.mesh(name))
  return spec


def get_g1_robot_cfg() -> EntityCfg:
  """Get a fresh G1 (no hands) robot configuration instance."""
  cfg = g1_constants.get_g1_robot_cfg()
  cfg.spec_fn = get_g1_no_hands_spec
  return cfg


def get_spec_largebox() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  world = spec.worldbody
  body = world.add_body(name="largebox_link")
  body.pos = np.array([0.4, 0, 0.17], dtype=np.float64)
  body.add_freejoint(name="largebox_freejoint")
  geom = body.add_geom(
    name="largebox_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    friction=(0.6, 0.005, 0.0001),
    size=(0.155, 0.155, 0.17),
    quat=(0.00991298, 0.849052, -0.523456, 0.0707591),
    mass=0.6,
  )
  geom.rgba = np.array([0.2, 0.6, 0.8, 1.0], dtype=np.float32)
  return spec


def get_largebox_cfg() -> EntityCfg:
  """Get a fresh largebox configuration instance."""
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.4, 0, 0.17),
      rot=(1.0, 0.0, 0.0, 0.0),
    ),
    spec_fn=get_spec_largebox,
  )
