"""Sensing availability vs driver state.

SENSING_OK means IMU and camera *sources* are delivering fresh data.
It is not ATTENTIVE and must never be logged as a driver_state value.
"""

from __future__ import annotations

import time
from typing import Optional

UNAVAILABLE = frozenset({"DISCONNECTED", "WAITING", "WAITING_CAMERA", "CAMERA_DISCONNECTED", "OFF"})


def source_fresh(
    last_rx_mono: Optional[float],
    timeout_s: float,
    now_mono: Optional[float] = None,
    never_received_status: str = "WAITING",
) -> str:
    """Return OK or DISCONNECTED using monotonic time only."""
    if last_rx_mono is None:
        return never_received_status
    now = time.monotonic() if now_mono is None else now_mono
    if timeout_s <= 0:
        return "OK"
    if (now - last_rx_mono) > timeout_s:
        return "DISCONNECTED"
    return "OK"


def sensing_availability(imu_status: str, camera_status: str, camera_enabled: bool = True) -> str:
    """Gateway-level source health. Not a drowsiness label."""
    imu_bad = imu_status in UNAVAILABLE
    cam_bad = camera_enabled and camera_status in UNAVAILABLE
    if imu_bad or cam_bad:
        parts = []
        if imu_bad:
            parts.append("IMU")
        if cam_bad:
            parts.append("CAMERA")
        return "UNKNOWN (" + ", ".join(parts) + " unavailable)"
    return "SENSING_OK"


def is_driver_state_label(value: str) -> bool:
    return value in {
        "ATTENTIVE",
        "DISTRACTED",
        "DROWSINESS_SUSPECTED",
        "HIGH_RISK",
        "UNKNOWN",
    }
