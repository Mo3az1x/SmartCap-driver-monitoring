from mjpeg_client import MjpegStreamParser, classify_metadata
from fixtures.fake_mjpeg import build_part, synthetic_jpeg


def test_split_chunk_headers_are_reassembled():
    jpeg = synthetic_jpeg(seed=3)
    blob = build_part(jpeg, frame_id=42, timestamp_ms=777)
    parser = MjpegStreamParser()
    parts = []
    for i in range(0, len(blob), 17):
        parts.extend(parser.feed(blob[i:i + 17], received_timestamp_ms=5000))
    assert len(parts) == 1
    part = parts[0]
    assert part.device_frame_id == 42
    assert part.device_timestamp_ms == 777
    assert part.metadata_status == "OK"
    assert part.camera_clock_domain == "esp32_millis"
    assert part.received_timestamp_ms == 5000
    assert part.jpeg == jpeg


def test_missing_device_headers_are_classified():
    jpeg = synthetic_jpeg(seed=1)
    blob = build_part(jpeg, frame_id=1, timestamp_ms=1, include_headers=False)
    parser = MjpegStreamParser()
    parts = parser.feed(blob, received_timestamp_ms=9)
    assert parts[0].metadata_status == "MISSING"
    assert parts[0].device_frame_id is None
    assert parts[0].camera_clock_domain == "unknown"


def test_partial_metadata():
    fid, ts, status = classify_metadata({"x-frame-id": "3"})
    assert fid == 3
    assert ts is None
    assert status == "PARTIAL"
