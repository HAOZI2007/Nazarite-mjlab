"""Gait diagnostics for the WMP Go2 course task.

These terms are registered as *metrics*, not rewards: they are logged per
episode and are also accumulated per environment so the terrain curriculum can
refuse to promote a robot that is climbing with a crouched, dragging gait.

Why they exist: the 2026-10-18 run climbed to terrain level 4.9 while walking
nose-down with the hind feet skidding on the ground. Every reward term in the
task is worth a fraction of a percent of the velocity term, so the only way to
tell a trot from a crawl was to watch a video. The five numbers below make that
comparison quantitative.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from nazarite.mdp.shared.rewards import feet_gait

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

#: Foot index groups for the ``("FL", "FR", "RL", "RR")`` site order.
FRONT_FOOT_INDICES = (0, 1)
HIND_FOOT_INDICES = (2, 3)


def _accumulate(env, key: str, value: torch.Tensor) -> None:
    """Add ``value`` to a lazily allocated per-environment running sum."""
    accumulator = getattr(env, key, None)
    if accumulator is None or accumulator["count"].shape[0] != env.num_envs:
        accumulator = {
            "sum": torch.zeros(env.num_envs, device=env.device, dtype=torch.float32),
            "count": torch.zeros(env.num_envs, device=env.device, dtype=torch.float32),
        }
        env.__dict__[key] = accumulator
    # Rebound rather than updated in place: accumulators are first allocated
    # inside the training loop's ``torch.inference_mode()`` block, and an in place
    # write from another mode raises "Inplace update to inference tensor".
    accumulator["sum"] = accumulator["sum"] + value.detach().float()
    accumulator["count"] = accumulator["count"] + 1.0


def episode_average(env, key: str, env_ids: torch.Tensor) -> torch.Tensor | None:
    """Return the mean of a running metric for ``env_ids`` and clear it.

    The curriculum calls this at reset time, when the accumulator still holds
    exactly the episode that just ended. Resetting only the finished
    environments keeps the asynchronous resets independent.
    """
    accumulator = getattr(env, key, None)
    if accumulator is None or accumulator["count"].shape[0] != env.num_envs:
        return None
    count = accumulator["count"][env_ids].clamp_min(1.0)
    mean = accumulator["sum"][env_ids] / count
    accumulator["sum"][env_ids] = 0.0
    accumulator["count"][env_ids] = 0.0
    return mean


def gait_diagonal_match(
    env,
    period: float = 0.6,
    offset: list[float] | tuple[float, ...] = (0.0, 0.5, 0.5, 0.0),
    threshold: float = 0.56,
    command_threshold: float = 0.1,
    command_name: str = "twist",
    sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
    """Fraction of feet whose contact state matches the diagonal schedule.

    1.0 is a perfect trot, 0.5 is what random contacts produce, and the
    2026-10-18 run measured 0.40 -- it was actively anti-matching while paying
    almost nothing for it.
    """
    value = feet_gait(
        env,
        period=period,
        offset=list(offset),
        threshold=threshold,
        command_threshold=command_threshold,
        command_name=command_name,
        sensor_name=sensor_name,
    )
    _accumulate(env, "_wmp_gait_match", value)
    return value


def gait_base_clearance(
    env,
    sensor_name: str = "wmp_base_clearance",
) -> torch.Tensor:
    """Base height above the terrain directly underneath, in metres."""
    try:
        heights = env.scene[sensor_name].data.heights
    except (AttributeError, KeyError):
        return torch.zeros(env.num_envs, device=env.device)
    if heights is None:
        return torch.zeros(env.num_envs, device=env.device)
    if heights.ndim > 1:
        heights = heights.reshape(heights.shape[0], -1).min(dim=1).values
    value = heights.clamp(0.0, 2.0)
    _accumulate(env, "_wmp_gait_clearance", value)
    return value


def gait_base_tilt(
    env,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Total base tilt in radians: 0 is level, 0.6 is the observed crawl."""
    asset: Entity = env.scene[asset_cfg.name]
    gravity_b = asset.data.projected_gravity_b
    value = torch.arccos((-gravity_b[:, 2]).clamp(-1.0, 1.0))
    _accumulate(env, "_wmp_gait_tilt", value)
    return value


def gait_hind_swing_ratio(
    env,
    sensor_name: str = "feet_ground_contact",
    hind_indices: tuple[int, ...] = HIND_FOOT_INDICES,
) -> torch.Tensor:
    """Fraction of time the hind feet spend in swing.

    A trot holds each hind foot in swing about half of every cycle; a dragging
    hind leg sits near zero. This is the single number that most directly
    separates the observed crawl from a trot.
    """
    try:
        contact = env.scene[sensor_name].data.found
    except (AttributeError, KeyError):
        return torch.zeros(env.num_envs, device=env.device)
    if contact is None:
        return torch.zeros(env.num_envs, device=env.device)
    # ``found`` counts matched contacts, and a Go2 foot has several collision
    # geoms, so the raw count can exceed one and drive this ratio negative. Only
    # "any contact" matters for a swing ratio.
    hind = (contact > 0).float()[:, list(hind_indices)]
    value = 1.0 - hind.mean(dim=1)
    _accumulate(env, "_wmp_hind_swing", value)
    return value


def gait_front_hind_swing_gap(
    env,
    sensor_name: str = "feet_ground_contact",
    front_indices: tuple[int, ...] = FRONT_FOOT_INDICES,
    hind_indices: tuple[int, ...] = HIND_FOOT_INDICES,
) -> torch.Tensor:
    """Signed difference between front and hind swing ratios.

    Zero for any gait that gives both pairs the same swing budget, including a
    trot. A positive value means the front pair swings more than the hind pair,
    which is exactly the signature of the 2026-10-18 crawl.
    """
    try:
        contact = env.scene[sensor_name].data.found
    except (AttributeError, KeyError):
        return torch.zeros(env.num_envs, device=env.device)
    if contact is None:
        return torch.zeros(env.num_envs, device=env.device)
    swing = 1.0 - contact.float()
    front = swing[:, list(front_indices)].mean(dim=1)
    hind = swing[:, list(hind_indices)].mean(dim=1)
    return front - hind
