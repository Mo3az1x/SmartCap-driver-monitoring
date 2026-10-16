from rule_fusion import ALLOWED_STATES, fuse


NOW = 1_000_000


def _imu(**kwargs):
    base = {
        "received_timestamp_ms": NOW,
        "imu_status": "OK",
        "calibration_status": "CALIBRATED",
        "imu_quality": 0.9,
        "head_down_duration_ms": 0,
        "nodding_score": 0.0,
    }
    base.update(kwargs)
    return base


def _vis(**kwargs):
    base = {
        "received_timestamp_ms": NOW,
        "vision_status": "OK",
        "vision_quality": 0.9,
        "face_detected": True,
        "eye_state": "OPEN",
        "perclos_30s": 0.05,
        "prolonged_eye_closure": False,
        "prolonged_yawn": False,
        "head_yaw_ratio": 0.0,
    }
    base.update(kwargs)
    return base


def test_no_sources_is_unknown_not_attentive():
    out = fuse(None, None, now_ms=NOW)
    assert out["state"] == "UNKNOWN"
    assert out["type"] == "driver_state"
    assert out["state"] in ALLOWED_STATES


def test_no_face_is_unknown():
    out = fuse(_imu(), _vis(face_detected=False, vision_status="NO_FACE", eye_state="UNKNOWN"), now_ms=NOW)
    assert out["state"] == "UNKNOWN"
    assert "no_face" in out["evidence"]


def test_calibrating_is_unknown():
    out = fuse(_imu(), _vis(vision_status="CALIBRATING", eye_state="CALIBRATING"), now_ms=NOW)
    assert out["state"] == "UNKNOWN"


def test_stale_imu_and_vision_unknown():
    out = fuse(
        _imu(received_timestamp_ms=NOW - 5000),
        _vis(received_timestamp_ms=NOW - 5000),
        now_ms=NOW,
    )
    assert out["state"] == "UNKNOWN"


def test_both_ok_no_cues_attentive():
    out = fuse(_imu(), _vis(), now_ms=NOW)
    assert out["state"] == "ATTENTIVE"
    assert out["risk_level"] == "LOW"


def test_single_source_without_cues_is_not_attentive():
    out = fuse(_imu(), None, now_ms=NOW)
    assert out["state"] == "UNKNOWN"


def test_prolonged_closure_is_drowsiness():
    out = fuse(_imu(), _vis(prolonged_eye_closure=True, eye_state="CLOSED"), now_ms=NOW)
    assert out["state"] == "DROWSINESS_SUSPECTED"


def test_multimodal_high_risk():
    out = fuse(
        _imu(head_down_duration_ms=3500),
        _vis(prolonged_eye_closure=True, eye_state="CLOSED"),
        now_ms=NOW,
    )
    assert out["state"] == "HIGH_RISK"


def test_cap_camera_yaw_is_not_distracted():
    out = fuse(_imu(), _vis(head_yaw_ratio=0.9), now_ms=NOW, camera_placement="cap")
    assert out["state"] == "ATTENTIVE"


def test_cabin_camera_yaw_can_be_distracted():
    out = fuse(_imu(), _vis(head_yaw_ratio=0.9), now_ms=NOW, camera_placement="cabin")
    assert out["state"] == "DISTRACTED"


def test_confidence_is_not_claimed_calibrated():
    out = fuse(_imu(), _vis(), now_ms=NOW)
    assert out["confidence_is_calibrated_probability"] is False
