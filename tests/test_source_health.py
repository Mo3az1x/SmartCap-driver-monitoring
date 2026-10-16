from source_health import is_driver_state_label, sensing_availability, source_fresh


def test_never_received_is_waiting_not_attentive():
    assert source_fresh(None, 0.5) == "WAITING"


def test_timeout_disconnected():
    assert source_fresh(0.0, 0.5, now_mono=1.0) == "DISCONNECTED"
    assert source_fresh(0.8, 0.5, now_mono=1.0) == "OK"


def test_sensing_ok_is_not_a_driver_state():
    label = sensing_availability("OK", "OK", camera_enabled=True)
    assert label == "SENSING_OK"
    assert not is_driver_state_label(label)


def test_missing_camera_is_unknown_sensing():
    label = sensing_availability("OK", "CAMERA_DISCONNECTED", camera_enabled=True)
    assert label.startswith("UNKNOWN")
    assert "CAMERA" in label
    assert "ATTENTIVE" not in label


def test_camera_off_does_not_fail_sensing():
    assert sensing_availability("OK", "OFF", camera_enabled=False) == "SENSING_OK"
