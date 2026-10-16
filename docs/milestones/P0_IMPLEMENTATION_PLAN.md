# P0 Implementation Plan — Synchronized Multimodal Sensing Foundation

Status: in progress (software). Physical hardware validation is a separate checklist and is **not** claimed here.

Canonical integration path: `gateway/receiver/` (IMU serial + `vision_worker.py`).
Legacy webcam scripts in `edge_ai/vision/src/` remain for lab bring-up; they are **not** the integration feature definitions.

## Preserve

- Existing `vision_worker.py` behaviour (video-path DEMO_MODE, yaw clip, EyeTracker reopen/EMA, auto-recalibrate, urgent JSON export).
- SMARTCAP_INTERFACE_SPEC_v1 field names and units.
- Firmware sketches (no behaviour change this session unless required). Firmware `demoMode` stays selectable; it is **not** a fused AI decision.

## Priority

1. Reproducible environment + serial profiles + entry points.
2. Session launcher: IMU + vision, one MJPEG client.
3. Honest timestamps + offline alignment.
4. Automated hardware-free tests.
5. Minimal rule-based `driver_state` (only after foundation tests pass).
6. Docs + hardware checklist (results left NOT RUN).

## Acceptance (software)

- `pytest tests` passes in the documented environment.
- `run_session.py --sim-imu --acquisition-only` with a fake MJPEG source writes a session directory, IMU CSV, vision CSV, config JSON, and shuts down cleanly.
- Alignment tool matches on an explicit clock domain and reports unmatched rows.
- Fusion emits only allowed `driver_state` values and defaults to `UNKNOWN` when evidence is insufficient.
- Disconnect vs `NO_FACE` remain distinct. `SENSING_OK` is never treated as `ATTENTIVE`.
