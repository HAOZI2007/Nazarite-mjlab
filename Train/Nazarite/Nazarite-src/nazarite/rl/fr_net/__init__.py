"""FR-Net actor and PPO implementations."""

from .frnet_actor import FRNetActor as FRNetActor
from .frnet_ppo import FRNetPPO as FRNetPPO

__all__ = ("FRNetActor", "FRNetPPO")
