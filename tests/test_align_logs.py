from align_logs import align_records


def test_nearest_match_within_tolerance():
    imu = [
        {"received_timestamp_ms": 1000, "pitch_deg": 1},
        {"received_timestamp_ms": 1080, "pitch_deg": 2},
    ]
    vis = [
        {"received_timestamp_ms": 1070, "ear": 0.2},
    ]
    result = align_records(imu, vis, time_basis="received_timestamp_ms", tolerance_ms=80)
    assert result["summary"]["matched_pairs"] == 1
    assert result["summary"]["unmatched_imu"] == 1
    matched = [r for r in result["rows"] if r["align_status"] == "matched"][0]
    assert matched["imu_pitch_deg"] == 2
    assert matched["match_delta_ms"] == 10
    assert result["summary"]["hardware_clocks_synchronized"] is False
    assert result["summary"]["stale_forward_fill"] is False


def test_unmatched_when_gap_exceeds_tolerance():
    imu = [{"received_timestamp_ms": 1000}]
    vis = [{"received_timestamp_ms": 2000}]
    result = align_records(imu, vis, tolerance_ms=80)
    assert result["summary"]["matched_pairs"] == 0
    assert result["summary"]["unmatched_vision"] == 1
    assert result["summary"]["unmatched_imu"] == 1


def test_imu_sample_is_not_reused():
    imu = [{"received_timestamp_ms": 1000, "sequence_id": 1}]
    vis = [
        {"received_timestamp_ms": 1000, "frame_id": 1},
        {"received_timestamp_ms": 1010, "frame_id": 2},
    ]
    result = align_records(imu, vis, tolerance_ms=80)
    assert result["summary"]["matched_pairs"] == 1
    assert result["summary"]["unmatched_vision"] == 1
