#!/usr/bin/env python3
"""Minimal rule-based driver_state baseline (SMARTCAP_INTERFACE_SPEC_v1 §10).

This is NOT a validated drowsiness detector. `confidence` is a heuristic
score in [0, 1], not a calibrated probability.

Wearable-camera yaw is NOT mapped to DISTRACTED by default (spec §9:
orientation is relative to the cap camera, not the road). Set
camera_placement=cabin to enable that optional rule.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


ALLOWED_STATES = ("ATTENTIVE", "DISTRACTED", "DROWSINESS_SUSPECTED", "HIGH_RISK", "UNKNOWN")
ALLOWED_RISK = ("LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN")

DEFAULT_THRESHOLDS = {
    "imu_timeout_ms": 500,
    "vision_timeout_ms": 2000,
    "min_imu_quality": 0.30,
    "min_vision_quality": 0.30,
    "perclos_drowsy": 0.40,
    "perclos_high_risk": 0.60,
    "head_down_drowsy_ms": 1500,
    "head_down_high_risk_ms": 3000,
    "nodding_drowsy": 0.60,
    "yaw_distract": 0.35,
    "packet_loss_unknown": 0.20,
}


@dataclass
class FusionThresholds:
    imu_timeout_ms: int = 500
    vision_timeout_ms: int = 2000
    min_imu_quality: float = 0.30
    min_vision_quality: float = 0.30
    perclos_drowsy: float = 0.40
    perclos_high_risk: float = 0.60
    head_down_drowsy_ms: int = 1500
    head_down_high_risk_ms: int = 3000
    nodding_drowsy: float = 0.60
    yaw_distract: float = 0.35
    packet_loss_unknown: float = 0.20

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "FusionThresholds":
        if not data:
            return cls()
        allowed = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**allowed)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _num(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fresh(sample: Optional[dict], timestamp_key: str, timeout_ms: int, now_ms: int) -> bool:
    if not sample:
        return False
    ts = _num(sample.get(timestamp_key))
    if ts is None:
        return False
    return (now_ms - ts) <= timeout_ms


def fuse(
    imu: Optional[dict],
    vision: Optional[dict],
    *,
    now_ms: Optional[int] = None,
    thresholds: Optional[FusionThresholds] = None,
    camera_placement: str = "cap",
    imu_packet_loss: Optional[float] = None,
) -> dict:
    thr = thresholds or FusionThresholds()
    now = _now_ms() if now_ms is None else now_ms
    evidence: list[str] = []
    rules: list[str] = []

    imu = imu or None
    vision = vision or None

    imu_fresh = _fresh(imu, "received_timestamp_ms", thr.imu_timeout_ms, now)
    vis_ts_key = "received_timestamp_ms" if vision and vision.get("received_timestamp_ms") not in (None, "") else "timestamp_ms"
    vis_fresh = _fresh(vision, vis_ts_key, thr.vision_timeout_ms, now)

    imu_q = _num(imu.get("imu_quality") if imu else None, 0.0) or 0.0
    vis_q = _num(vision.get("vision_quality") if vision else None, 0.0) or 0.0
    imu_status = (imu or {}).get("imu_status") or "DISCONNECTED"
    vis_status = (vision or {}).get("vision_status") or "CAMERA_DISCONNECTED"
    cal = (imu or {}).get("calibration_status")

    imu_health_ok = (
        imu_fresh
        and cal == "CALIBRATED"
        and imu_status in ("OK", "DEGRADED")
        and imu_q >= thr.min_imu_quality
    )
    # Face features are only usable when a face is present and the camera is up.
    vis_camera_up = vis_fresh and vis_status not in {
        "CAMERA_DISCONNECTED",
        "WAITING_CAMERA",
        "DISCONNECTED",
        "WAITING",
    }
    vis_face_ok = (
        vis_camera_up
        and bool(vision.get("face_detected"))
        and vis_status in ("OK", "DEGRADED")
        and vis_q >= thr.min_vision_quality
        and (vision.get("eye_state") not in ("UNKNOWN", "CALIBRATING"))
    )

    if imu_packet_loss is not None and imu_packet_loss > thr.packet_loss_unknown:
        rules.append("packet_loss_high")
        return _state(
            "UNKNOWN", "UNKNOWN", 0.0, 0.15, imu_q, vis_q, evidence + ["packet_loss_high"],
            rules, "DEGRADED", vis_status, imu_status, now,
        )

    if vis_status == "CALIBRATING" or (vision and vision.get("eye_state") == "CALIBRATING"):
        rules.append("vision_calibrating")
        return _state(
            "UNKNOWN", "UNKNOWN", 0.0, 0.2, imu_q, vis_q, ["vision_calibrating"],
            rules, "OK" if vis_camera_up else "DEGRADED", vis_status, imu_status, now,
        )

    if not imu_health_ok and not vis_face_ok:
        if vis_camera_up and vision and not vision.get("face_detected"):
            evidence.append("no_face")
            rules.append("no_face_is_not_attentive")
        if not imu_fresh:
            evidence.append("imu_stale_or_missing")
            rules.append("imu_not_fresh")
        if imu and cal != "CALIBRATED":
            evidence.append("imu_not_calibrated")
            rules.append("imu_not_calibrated")
        if not vis_fresh:
            evidence.append("vision_stale_or_missing")
            rules.append("vision_not_fresh")
        if not evidence:
            evidence.append("insufficient_sources")
            rules.append("no_reliable_source")
        return _state(
            "UNKNOWN", "UNKNOWN", 0.0, 0.2, imu_q, vis_q, evidence, rules,
            "UNKNOWN", vis_status, imu_status, now,
        )

    drowsy_vis = []
    if vis_face_ok:
        if vision.get("prolonged_eye_closure") in (True, "True", "true", 1, "1"):
            drowsy_vis.append("long_eye_closure")
        perclos = _num(vision.get("perclos_30s"), _num(vision.get("perclos"), 0.0)) or 0.0
        if perclos >= thr.perclos_high_risk:
            drowsy_vis.append("perclos_high")
        elif perclos >= thr.perclos_drowsy:
            drowsy_vis.append("perclos_elevated")
        if vision.get("prolonged_yawn") in (True, "True", "true", 1, "1"):
            drowsy_vis.append("prolonged_yawn")

    drowsy_imu = []
    if imu_health_ok:
        head_down = _num(imu.get("head_down_duration_ms"), 0) or 0
        nod = _num(imu.get("nodding_score"), 0) or 0
        if head_down >= thr.head_down_high_risk_ms:
            drowsy_imu.append("head_down_long")
        elif head_down >= thr.head_down_drowsy_ms:
            drowsy_imu.append("head_down")
        if nod >= thr.nodding_drowsy:
            drowsy_imu.append("repeated_nodding")

    evidence.extend(drowsy_vis)
    evidence.extend(drowsy_imu)

    conflict = False
    if vis_face_ok and imu_health_ok:
        eyes_open = vision.get("eye_state") == "OPEN" and not drowsy_vis
        if eyes_open and drowsy_imu:
            conflict = True
            evidence.append("conflict_eyes_open_head_motion")
            rules.append("sensor_conflict_reduce_confidence")

    high = (
        ("long_eye_closure" in drowsy_vis and bool(drowsy_imu))
        or ("perclos_high" in drowsy_vis and bool(drowsy_imu))
        or ("head_down_long" in drowsy_imu and "long_eye_closure" in drowsy_vis)
    )
    if high:
        rules.append("high_risk_multimodal")
        conf = 0.55 if conflict else 0.75
        if not imu_health_ok or not vis_face_ok:
            conf = min(conf, 0.55)
        return _state(
            "HIGH_RISK", "HIGH", 0.85, conf, imu_q, vis_q, evidence, rules,
            "OK", vis_status, imu_status, now,
        )

    if drowsy_vis or drowsy_imu:
        rules.append("drowsiness_cues")
        conf = 0.45 if conflict else 0.65
        if not (imu_health_ok and vis_face_ok):
            conf = min(conf, 0.5)
            rules.append("single_source_weighted")
        return _state(
            "DROWSINESS_SUSPECTED", "MEDIUM", 0.6, conf, imu_q, vis_q, evidence, rules,
            "OK", vis_status, imu_status, now,
        )

    if camera_placement == "cabin" and vis_face_ok:
        yaw = abs(_num(vision.get("head_yaw_ratio"), 0.0) or 0.0)
        if yaw >= thr.yaw_distract:
            rules.append("cabin_yaw_distracted")
            evidence.append("head_turned_cabin_camera")
            return _state(
                "DISTRACTED", "MEDIUM", 0.5, 0.45, imu_q, vis_q, evidence, rules,
                "OK", vis_status, imu_status, now,
            )

    if vis_face_ok and imu_health_ok:
        rules.append("both_sources_ok_no_target_cues")
        evidence.append("no_target_cues")
        return _state(
            "ATTENTIVE", "LOW", 0.1, 0.4, imu_q, vis_q, evidence, rules,
            "OK", vis_status, imu_status, now,
        )

    rules.append("partial_source_no_cues_unknown")
    evidence.append("insufficient_for_attentive")
    return _state(
        "UNKNOWN", "UNKNOWN", 0.0, 0.3, imu_q, vis_q, evidence, rules,
        "DEGRADED", vis_status, imu_status, now,
    )


def _state(
    state, risk_level, risk_score, confidence, camera_quality, imu_quality,
    evidence, rules, system_status, vis_status, imu_status, now_ms,
) -> dict:
    return {
        "type": "driver_state",
        "timestamp_ms": now_ms,
        "state": state,
        "risk_level": risk_level,
        "risk_score": round(float(risk_score), 3),
        "confidence": round(float(confidence), 3),
        "camera_quality": round(float(camera_quality or 0.0), 3),
        "imu_quality": round(float(imu_quality or 0.0), 3),
        "evidence": list(evidence),
        "system_status": system_status,
        "rules_fired": list(rules),
        "confidence_is_calibrated_probability": False,
        "sensing_note": "system_status is diagnostic health, not driver behaviour",
        "vision_status": vis_status,
        "imu_status": imu_status,
    }


def fuse_files(imu_path: Optional[Path], vision_path: Optional[Path], **kwargs) -> dict:
    imu = json.loads(Path(imu_path).read_text(encoding="utf-8")) if imu_path else None
    vision = json.loads(Path(vision_path).read_text(encoding="utf-8")) if vision_path else None
    return fuse(imu, vision, **kwargs)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Rule-based SmartCap driver_state (baseline, not a medical detector)")
    p.add_argument("--imu-json", default=None)
    p.add_argument("--vision-json", default=None)
    p.add_argument("--now-ms", type=int, default=None)
    p.add_argument("--camera-placement", default="cap", choices=("cap", "cabin"))
    p.add_argument("--thresholds-json", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    thr = None
    if args.thresholds_json:
        thr = FusionThresholds.from_dict(json.loads(Path(args.thresholds_json).read_text(encoding="utf-8")))
    result = fuse_files(
        Path(args.imu_json) if args.imu_json else None,
        Path(args.vision_json) if args.vision_json else None,
        now_ms=args.now_ms,
        thresholds=thr,
        camera_placement=args.camera_placement,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
