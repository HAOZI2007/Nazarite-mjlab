"""XZ1-specific training environment configurations."""

# The directory name is intentionally kept as ``XZ1_env_cfgs`` to match the
# robot-specific task layout requested for this project.
# ruff: noqa: N999

from .him_env_cfg import Nazarite_HIM_Complex_Terrain_XZ1
from .wtw_env_cfg import Nazarite_Velocity_Flat_XZ1_WTW

__all__ = [
  "Nazarite_HIM_Complex_Terrain_XZ1",
  "Nazarite_Velocity_Flat_XZ1_WTW",
]
