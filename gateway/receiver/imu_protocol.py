"""IMU JSON validation and serial profiles (SMARTCAP_INTERFACE_SPEC_v1 §7)."""

from __future__ import annotations

from typing import Any, Optional

# Spec §7 required identity/timing/status fields. Feature fields are required
# for a "complete" message used in logging; identity fields are the minimum
# to accept a line as IMU.
IMU_IDENTITY_FIELDS = (
    "type",
    "sequence_id",
    "device_timestamp_ms",
)

IMU_REQUIRED_GATEWAY = (
    "type",
    "sequence_id",
    "device_timestamp_ms",
    "pitch_deg",
    "roll_deg",
    "imu_status",
    "calibration_status",
    "imu_quality",
)

IMU_FEATURE_FIELDS = (
    "pitch_deg",
    "roll_deg",
    "pitch_velocity_dps",
    "head_down_duration_ms",
    "nodding_score",
    "motion_energy",
    "imu_status",
    "calibration_status",
    "imu_quality",
)

ALLOWED_IMU_STATUS = frozenset(
    {"OK", "DEGRADED", "INVALID", "DISCONNECTED", "CALIBRATING", "UNKNOWN"}
)

# Two real firmware sketches exist. Using the wrong baud looks like "broken JSON".
SERIAL_PROFILES = {
    "haptic": {
        "id": "haptic",
        "sketch": "firmware/imu_haptic/src/smartcap_imu_haptic/smartcap_imu_haptic.ino",
        "baud": 230400,
        "sample_hz": 50,
        "output_hz": 25,
        "notes": "Coin-motor PWM on D9. Default for gateway/receiver.",
    },
    "led": {
        "id": "led",
        "sketch": "firmware/imu_haptic/src/smartcap_imu_led/smartcap_imu_led.ino",
        "baud": 115200,
        "sample_hz": 50,
        "output_hz": 10,
        "notes": "Onboard LED pin 13. Do not use 230400 with this sketch.",
    },
}

DEFAULT_PROFILE = "haptic"


def get_serial_profile(name: str) -> dict:
    key = (name or DEFAULT_PROFILE).strip().lower()
    if key not in SERIAL_PROFILES:
        known = ", ".join(sorted(SERIAL_PROFILES))
        raise ValueError(f"Unknown serial profile {name!r}. Known: {known}")
    return dict(SERIAL_PROFILES[key])


def validate_imu_message(message: Any, required=IMU_REQUIRED_GATEWAY) -> tuple[bool, Optional[str]]:
    if not isinstance(message, dict):
        return False, "not an object"
    if message.get("type") != "imu":
        return False, f"unexpected type {message.get('type')!r}"
    missing = [field for field in required if field not in message]
    if missing:
        return False, f"missing fields: {missing}"
    status = message.get("imu_status")
    if status is not None and status not in ALLOWED_IMU_STATUS:
        return False, f"invalid imu_status {status!r}"
    return True, None


def is_imu_usable_for_behaviour(message: dict) -> bool:
    """True when IMU samples may inform behaviour (not health-only heartbeats)."""
    if message.get("calibration_status") != "CALIBRATED":
        return False
    if message.get("imu_status") not in ("OK", "DEGRADED"):
        return False
    return True
