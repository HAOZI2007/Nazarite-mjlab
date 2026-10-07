"""Linux gamepad input for Nazarite sim2sim.

The Flydigi Vader 4 receiver is normally exposed as an evdev input device on
Linux. This module deliberately talks to that standard interface rather than a
brand-specific driver, so it also works with other XInput/DirectInput pads.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

try:
    from evdev import InputDevice, ecodes, list_devices
except ImportError as exc:  # pragma: no cover - depends on host OS
    raise ImportError(
        "Gamepad support requires evdev. Install the project dependencies "
        "in the Linux sim2sim environment."
    ) from exc


@dataclass(frozen=True)
class GamepadInfo:
    path: str
    name: str
    vendor: int
    product: int


def _device_has_sticks(device: InputDevice) -> bool:
    """Return true for devices advertising the two standard left-stick axes."""
    capabilities = cast(dict[int, list[Any]], device.capabilities(absinfo=False))
    abs_axes = capabilities.get(ecodes.EV_ABS, [])
    return ecodes.ABS_X in abs_axes and ecodes.ABS_Y in abs_axes


def list_gamepads() -> list[GamepadInfo]:
    """Find evdev devices that look like a gamepad or joystick."""
    candidates: list[GamepadInfo] = []
    fallback: list[GamepadInfo] = []
    keywords = ("gamepad", "joystick", "controller", "flydigi", "vader")

    for path in list_devices():
        try:
            device = InputDevice(path)
            if not _device_has_sticks(device):
                continue
            info = GamepadInfo(
                path=device.path,
                name=device.name,
                vendor=int(device.info.vendor),
                product=int(device.info.product),
            )
            fallback.append(info)
            if any(keyword in device.name.lower() for keyword in keywords):
                candidates.append(info)
        except OSError:
            # A USB receiver can disappear between device enumeration and open.
            continue

    return candidates or fallback


def resolve_axis(axis: str | int) -> int:
    """Resolve ``ABS_Y``/``ABS_RX`` or a numeric evdev ABS code."""
    if isinstance(axis, int):
        return axis
    axis = axis.upper()
    if axis.isdigit() or (axis.startswith("-") and axis[1:].isdigit()):
        return int(axis)
    value = getattr(ecodes, axis, None)
    if not isinstance(value, int) or not axis.startswith("ABS_"):
        raise ValueError(f"Unknown absolute axis {axis!r}; use names such as ABS_Y")
    return value


def resolve_button(button: str | int) -> int:
    """Resolve an evdev button name such as ``BTN_TR``."""
    if isinstance(button, int):
        return button
    button = button.upper()
    value = getattr(ecodes, button, None)
    if not isinstance(value, int) or not button.startswith("BTN_"):
        raise ValueError(f"Unknown button {button!r}; use names such as BTN_TR")
    return value


def normalize_axis(value: int, minimum: int, maximum: int, deadzone: float) -> float:
    """Normalize an evdev stick value to [-1, 1], applying a centered deadzone."""
    if maximum <= minimum:
        return 0.0
    center = 0.5 * (minimum + maximum)
    radius = max(center - minimum, maximum - center)
    normalized = float(np.clip((value - center) / radius, -1.0, 1.0))
    magnitude = abs(normalized)
    if magnitude <= deadzone:
        return 0.0
    return float(np.copysign((magnitude - deadzone) / (1.0 - deadzone), normalized))


class GamepadController:
    """Poll a Linux controller and return ``[vx, vy, yaw_rate]`` commands.

    Default mapping for XInput-style Flydigi receiver mode:

    - left stick Y (`ABS_Y`): forward/backward velocity;
    - left stick X (`ABS_X`): lateral velocity;
    - right stick X (`ABS_RX`): yaw rate.

    Axis names can be overridden from ``main.py`` command-line options when a
    different receiver mode exposes a different evdev layout.
    """

    def __init__(
        self,
        device_path: str | None = None,
        *,
        max_vx: float = 1.0,
        max_vy: float = 0.0,
        max_yaw: float = 0.5,
        deadzone: float = 0.08,
        axis_vx: str | int = "ABS_Y",
        axis_vy: str | int = "ABS_X",
        axis_yaw: str | int = "ABS_RX",
    ) -> None:
        if not 0.0 <= deadzone < 1.0:
            raise ValueError("deadzone must be in [0, 1)")
        if device_path is None:
            gamepads = list_gamepads()
            if not gamepads:
                raise RuntimeError(
                    "No gamepad found. Connect the Flydigi receiver, then run "
                    "`python -m sim2sim.main --list-gamepads`."
                )
            device_path = gamepads[0].path

        self.device = InputDevice(str(Path(device_path)))
        if not _device_has_sticks(self.device):
            raise RuntimeError(f"{self.device.path} is not an analog gamepad")
        # ``read()`` is blocking by default. A blocking poll would stop the
        # MuJoCo control loop whenever the user leaves the sticks untouched.
        os.set_blocking(self.device.fd, False)

        self.max_vx = float(max_vx)
        self.max_vy = float(max_vy)
        self.max_yaw = float(max_yaw)
        self.deadzone = float(deadzone)
        self._requested_axes = {
            "vx": axis_vx,
            "vy": axis_vy,
            "yaw": axis_yaw,
        }
        self.axis_vx: int | None = None
        self.axis_vy: int | None = None
        self.axis_yaw: int | None = None
        self._axis_values: dict[int, int] = {}
        self._pressed_buttons: set[int] = set()
        self._axis_ranges = self._read_axis_ranges()
        self._disconnected = False

    @property
    def name(self) -> str:
        return self.device.name

    @property
    def path(self) -> str:
        return self.device.path

    def _read_axis_ranges(self) -> dict[int, tuple[int, int]]:
        capabilities = cast(
            dict[int, list[tuple[int, Any]]], self.device.capabilities(absinfo=True)
        )
        ranges: dict[int, tuple[int, int]] = {}
        for code, abs_info in capabilities.get(ecodes.EV_ABS, []):
            ranges[int(code)] = (int(abs_info.min), int(abs_info.max))
            self._axis_values[int(code)] = int(abs_info.value)
        self.axis_vx = self._select_axis(
            self._requested_axes["vx"], ranges, fallback=("ABS_Y", "ABS_RY")
        )
        self.axis_vy = self._select_axis(
            self._requested_axes["vy"], ranges, fallback=("ABS_X", "ABS_RX")
        )
        # Yaw is optional: some two-stick controllers expose no right-stick X
        # axis. In that case forward/lateral control remains usable and yaw is
        # held at zero instead of preventing startup.
        self.axis_yaw = self._select_axis(
            self._requested_axes["yaw"], ranges, fallback=("ABS_RX", "ABS_RZ", "ABS_Z")
        )
        if self.axis_vx is None or self.axis_vy is None:
            available = sorted(ranges)
            raise RuntimeError(
                f"Gamepad {self.name!r} does not expose usable stick axes. "
                f"Available absolute axes: {available}. "
                "Use --axis-vx/--axis-vy to select available axes."
            )
        return ranges

    @staticmethod
    def _select_axis(
        requested: str | int,
        ranges: dict[int, tuple[int, int]],
        *,
        fallback: tuple[str, ...],
    ) -> int | None:
        requested_code = resolve_axis(requested)
        if requested_code in ranges:
            return requested_code
        for candidate in fallback:
            candidate_code = resolve_axis(candidate)
            if candidate_code in ranges:
                return candidate_code
        return None

    def poll(self) -> np.ndarray:
        """Read all pending events; no event means retain the latest stick state."""
        if self._disconnected:
            return np.zeros(3, dtype=np.float32)

        try:
            for event in self.device.read():
                if event.type == ecodes.EV_ABS and event.code in self._axis_ranges:
                    self._axis_values[event.code] = event.value
                elif event.type == ecodes.EV_KEY and event.value in (1, 2):
                    self._pressed_buttons.add(int(event.code))
        except BlockingIOError:
            pass
        except OSError:
            # Never leave a moving velocity command active after unplugging the
            # receiver. The sim keeps running with a zero command instead.
            self._disconnected = True
            return np.zeros(3, dtype=np.float32)

        vx = -self._normalized(self.axis_vx) * self.max_vx
        vy = self._normalized(self.axis_vy) * self.max_vy
        yaw = self._normalized(self.axis_yaw) * self.max_yaw
        return np.array([vx, vy, yaw], dtype=np.float32)

    def _normalized(self, axis: int | None) -> float:
        if axis is None:
            return 0.0
        minimum, maximum = self._axis_ranges[axis]
        return normalize_axis(self._axis_values[axis], minimum, maximum, self.deadzone)

    def normalized_axis(self, axis: str | int) -> float | None:
        """Read an optional centered axis, returning ``None`` if unavailable."""
        code = resolve_axis(axis)
        if code not in self._axis_ranges:
            return None
        return self._normalized(code)

    @property
    def axis_mapping(self) -> str:
        """Return the resolved evdev axis mapping for startup diagnostics."""
        def name(code: int | None) -> str:
            if code is None:
                return "disabled"
            return str(ecodes.bytype[ecodes.EV_ABS].get(code, code))

        return f"vx={name(self.axis_vx)}, vy={name(self.axis_vy)}, yaw={name(self.axis_yaw)}"

    def consume_button(self, button: str | int) -> bool:
        """Consume one pressed-button event since the previous poll."""
        code = resolve_button(button)
        if code in self._pressed_buttons:
            self._pressed_buttons.remove(code)
            return True
        return False

    def close(self) -> None:
        try:
            self.device.close()
        except OSError:
            pass


def format_gamepads(gamepads: Iterable[GamepadInfo]) -> str:
    lines = ["Detected gamepads:"]
    for gamepad in gamepads:
        lines.append(
            f"  {gamepad.path}: {gamepad.name} "
            f"(vendor=0x{gamepad.vendor:04x}, product=0x{gamepad.product:04x})"
        )
    return "\n".join(lines)
