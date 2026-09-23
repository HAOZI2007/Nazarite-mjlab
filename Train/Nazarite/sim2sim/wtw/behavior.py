"""Gamepad control for WTW behavior parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..gamepad import GamepadController
from .config import BEHAVIOR_MAX, BEHAVIOR_MIN, BEHAVIOR_NOMINAL
from .observation import WTWObservationBuilder

if TYPE_CHECKING:
    from collections.abc import Callable


class WTWBehaviorController:
    """Adjust safe Trot behavior parameters with a standard gamepad.

    Right-stick Y controls frequency continuously. Button presses adjust the
    remaining scalar parameters in small steps; theta stays fixed at Trot.
    """

    def __init__(
        self,
        builder: WTWObservationBuilder,
        gamepad: GamepadController,
        *,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.builder = builder
        self.gamepad = gamepad
        self.status_callback = status_callback
        self._last_status = ""

    def update(self) -> None:
        behavior = self.builder.behavior
        changed = False

        # Stick up increases frequency. If the receiver does not expose ABS_RY,
        # the nominal frequency remains active and all button controls still work.
        right_stick_y = self.gamepad.normalized_axis("ABS_RY")
        if right_stick_y is not None:
            behavior[3] = np.clip(
                float(BEHAVIOR_NOMINAL[3]) - right_stick_y,
                BEHAVIOR_MIN[3],
                BEHAVIOR_MAX[3],
            )
            changed = True

        changed |= self._step("BTN_TL", 4, -0.005)  # LB: body lower
        changed |= self._step("BTN_TR", 4, 0.005)  # RB: body higher
        changed |= self._step("BTN_TL2", 5, -0.005)  # LT: pitch down
        changed |= self._step("BTN_TR2", 5, 0.005)  # RT: pitch up
        changed |= self._step("BTN_WEST", 6, -0.01)  # X: narrower stance
        changed |= self._step("BTN_EAST", 6, 0.01)  # B: wider stance
        changed |= self._step("BTN_SOUTH", 7, -0.005)  # A: lower swing
        changed |= self._step("BTN_NORTH", 7, 0.005)  # Y: higher swing

        if self.gamepad.consume_button("BTN_SELECT"):
            behavior[:] = BEHAVIOR_NOMINAL
            changed = True

        if changed:
            behavior[:] = np.clip(behavior, BEHAVIOR_MIN, BEHAVIOR_MAX)
            self._report()

    def _step(self, button: str, index: int, amount: float) -> bool:
        if not self.gamepad.consume_button(button):
            return False
        self.builder.behavior[index] += amount
        return True

    def _report(self) -> None:
        values = self.builder.behavior
        status = (
            "[sim2sim] behavior "
            f"freq={values[3]:.2f}Hz "
            f"height={values[4]:+.3f}m "
            f"pitch={values[5]:+.3f}rad "
            f"stance={values[6]:.3f}m "
            f"swing={values[7]:.3f}m"
        )
        if status != self._last_status and self.status_callback is not None:
            self.status_callback(status)
        self._last_status = status
