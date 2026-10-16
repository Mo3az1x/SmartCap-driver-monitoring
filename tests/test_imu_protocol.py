import pytest

from imu_protocol import get_serial_profile, validate_imu_message


def test_haptic_and_led_baud_are_different():
    haptic = get_serial_profile("haptic")
    led = get_serial_profile("led")
    assert haptic["baud"] == 230400
    assert led["baud"] == 115200
    assert haptic["output_hz"] == 25
    assert led["output_hz"] == 10


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        get_serial_profile("wifi")


def test_validate_imu_rejects_wrong_type():
    ok, err = validate_imu_message({"type": "camera_frame", "sequence_id": 1, "device_timestamp_ms": 1})
    assert not ok
    assert "type" in err


def test_validate_imu_accepts_spec_minimum():
    ok, err = validate_imu_message({
        "type": "imu",
        "sequence_id": 1,
        "device_timestamp_ms": 10,
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
        "imu_status": "OK",
        "calibration_status": "CALIBRATED",
        "imu_quality": 0.9,
    })
    assert ok, err
