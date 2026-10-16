import json
from pathlib import Path

import run_session
from fixtures.fake_mjpeg import FakeMjpegServer


def test_sim_session_acquisition_only(tmp_path, monkeypatch):
    server = FakeMjpegServer(n_frames=12, split_chunks=64).start()
    try:
        argv = [
            "--sim-imu",
            "--acquisition-only",
            "--duration", "2.5",
            "--camera-url", server.url,
            "--outdir", str(tmp_path),
            "--session-id", "test_sess",
        ]
        assert run_session.main(argv) == 0
    finally:
        server.stop()

    assert (tmp_path / "session.json").exists()
    assert (tmp_path / "session_stop.json").exists()
    config = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    assert config["gateway_consumes_mjpeg"] is False
    assert config["camera_client"] == "vision_worker.py"
    assert config["sim_imu"] is True

    imu_csvs = list(tmp_path.glob("gateway_imu_*.csv"))
    vis_csvs = list(tmp_path.glob("vision_features_*.csv"))
    assert imu_csvs, "IMU CSV missing"
    assert vis_csvs, "vision CSV missing"
    imu_text = imu_csvs[0].read_text(encoding="utf-8")
    vis_text = vis_csvs[0].read_text(encoding="utf-8")
    assert "received_timestamp_ms" in imu_text
    assert "received_timestamp_ms" in vis_text
    assert "device_timestamp_ms" in vis_text
