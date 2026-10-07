"""WTW-history plus DELTA-token attention policy.

The actor keeps the complete WTW proprioceptive history as its prior input. A
second view of the newest WTW frame is sliced into the DELTA query context, so
the public observation contract remains only ``wtw_proprio`` and ``delta_map``.
"""

from __future__ import annotations

from .direct_action_model import WtwDeltaDirectActionModel


# ``wtw_proprio`` is ordered as ten frames of the six regular terms, followed
# by five behavior frames and one phase frame:
# 10*(3+3+12+12+12+3) + 5*8 + 8 = 498.
WTW_DELTA_CONTEXT_INDICES = tuple(
    list(range(27, 30))
    + list(range(57, 60))
    + list(range(168, 180))
    + list(range(288, 300))
    + list(range(408, 420))
    + list(range(447, 450))
    + list(range(482, 490))
    + list(range(490, 498))
)


class WtwDeltaAttentionModel(WtwDeltaDirectActionModel):
    """Fuse WTW history and DELTA geometric tokens into a full action head."""

    def __init__(self, *args, **kwargs):
        # RslRlModelCfg materializes optional fields as explicit ``None`` before
        # passing them to the model, so ``setdefault`` alone cannot establish the
        # merged WTW context group required by this task.
        if kwargs.get("prior_group") is None:
            kwargs["prior_group"] = "wtw_proprio"
        if kwargs.get("delta_proprio_group") is None:
            kwargs["delta_proprio_group"] = "wtw_proprio"
        kwargs.setdefault("delta_proprio_dim", len(WTW_DELTA_CONTEXT_INDICES))
        kwargs.setdefault("delta_proprio_indices", WTW_DELTA_CONTEXT_INDICES)
        super().__init__(*args, **kwargs)
