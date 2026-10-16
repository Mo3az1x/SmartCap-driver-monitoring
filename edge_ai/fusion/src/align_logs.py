#!/usr/bin/env python3
"""Offline alignment of IMU and vision CSVs on an explicit common time basis.

Default time basis: received_timestamp_ms (laptop Unix epoch ms).
This is receipt-time association, NOT hardware clock synchronization.

Unmatched rows are preserved. No forward-fill of stale measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional


ALLOWED_TIME_BASIS = {
    "received_timestamp_ms": {
        "imu": "received_timestamp_ms",
        "vision": "received_timestamp_ms",
        "clock_domain": "unix_epoch_ms",
        "meaning": "Laptop arrival time. Not device-to-laptop latency.",
    },
    "vision_timestamp_ms": {
        "imu": "received_timestamp_ms",
        "vision": "timestamp_ms",
        "clock_domain": "unix_epoch_ms",
        "meaning": "Vision processing/publish wall time vs IMU receipt time (same laptop clock if both logged on one PC).",
    },
}


def _to_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def align_records(
    imu_rows: list[dict],
    vision_rows: list[dict],
    *,
    time_basis: str = "received_timestamp_ms",
    tolerance_ms: int = 80,
) -> dict:
    if time_basis not in ALLOWED_TIME_BASIS:
        raise ValueError(
            f"Unknown time_basis {time_basis!r}. Choose from {sorted(ALLOWED_TIME_BASIS)}"
        )
    spec = ALLOWED_TIME_BASIS[time_basis]
    imu_key, vis_key = spec["imu"], spec["vision"]

    imu_indexed = []
    for i, row in enumerate(imu_rows):
        t = _to_int(row.get(imu_key))
        imu_indexed.append((t, i, row))
    imu_indexed.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0, x[1]))

    used_imu = set()
    aligned = []
    unmatched_vision = []
    deltas = []

    imu_times = [(t, i) for t, i, _ in imu_indexed if t is not None]

    def nearest_imu(t_vis: int) -> Optional[tuple[int, int]]:
        # linear scan is fine for lab logs; keep it obvious
        best = None
        for t_imu, idx in imu_times:
            if idx in used_imu:
                continue
            d = abs(t_imu - t_vis)
            if d > tolerance_ms:
                if best is not None and t_imu > t_vis + tolerance_ms:
                    break
                continue
            if best is None or d < best[0] or (d == best[0] and t_imu <= t_vis):
                best = (d, idx, t_imu)
        if best is None:
            return None
        return best[1], best[2], best[0]

    # Sort IMU times for early break
    imu_times.sort()

    for v_i, vrow in enumerate(vision_rows):
        t_vis = _to_int(vrow.get(vis_key))
        record = {
            "align_status": "unmatched_vision",
            "time_basis": time_basis,
            "clock_domain": spec["clock_domain"],
            "match_delta_ms": "",
            "tolerance_ms": tolerance_ms,
            "vision_row_index": v_i,
            "imu_row_index": "",
        }
        for k, val in vrow.items():
            record[f"vision_{k}"] = val
        if t_vis is None:
            unmatched_vision.append(v_i)
            aligned.append(record)
            continue
        found = None
        best = None
        for t_imu, idx in imu_times:
            if idx in used_imu:
                continue
            d = abs(t_imu - t_vis)
            if d <= tolerance_ms and (best is None or d < best[0]):
                best = (d, idx, t_imu)
        if best is None:
            unmatched_vision.append(v_i)
            aligned.append(record)
            continue
        d, idx, t_imu = best
        used_imu.add(idx)
        imu_row = imu_rows[idx]
        record["align_status"] = "matched"
        record["match_delta_ms"] = d
        record["imu_row_index"] = idx
        for k, val in imu_row.items():
            record[f"imu_{k}"] = val
        deltas.append(d)
        aligned.append(record)

    unmatched_imu = []
    for t, idx, row in imu_indexed:
        if idx in used_imu:
            continue
        unmatched_imu.append(idx)
        record = {
            "align_status": "unmatched_imu",
            "time_basis": time_basis,
            "clock_domain": spec["clock_domain"],
            "match_delta_ms": "",
            "tolerance_ms": tolerance_ms,
            "vision_row_index": "",
            "imu_row_index": idx,
        }
        for k, val in row.items():
            record[f"imu_{k}"] = val
        aligned.append(record)

    n_vis = len(vision_rows)
    n_imu = len(imu_rows)
    n_match = sum(1 for r in aligned if r["align_status"] == "matched")
    summary = {
        "time_basis": time_basis,
        "clock_domain": spec["clock_domain"],
        "time_basis_meaning": spec["meaning"],
        "tolerance_ms": tolerance_ms,
        "imu_rows": n_imu,
        "vision_rows": n_vis,
        "matched_pairs": n_match,
        "unmatched_vision": len(unmatched_vision),
        "unmatched_imu": len(unmatched_imu),
        "vision_match_coverage": (n_match / n_vis) if n_vis else None,
        "imu_match_coverage": (n_match / n_imu) if n_imu else None,
        "match_delta_ms_mean": (sum(deltas) / len(deltas)) if deltas else None,
        "match_delta_ms_max": max(deltas) if deltas else None,
        "stale_forward_fill": False,
        "hardware_clocks_synchronized": False,
    }
    return {"rows": aligned, "summary": summary}


def write_alignment(result: dict, out_csv: Path, out_summary: Path) -> None:
    rows = result["rows"]
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with out_summary.open("w", encoding="utf-8") as fh:
        json.dump(result["summary"], fh, indent=2)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Align SmartCap IMU and vision CSVs (receipt time, not HW sync)")
    p.add_argument("--imu-csv", required=True)
    p.add_argument("--vision-csv", required=True)
    p.add_argument("--time-basis", default="received_timestamp_ms", choices=sorted(ALLOWED_TIME_BASIS))
    p.add_argument("--tolerance-ms", type=int, default=80)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-summary", default=None)
    args = p.parse_args(argv)

    imu_rows = _read_csv(Path(args.imu_csv))
    vision_rows = _read_csv(Path(args.vision_csv))
    result = align_records(
        imu_rows,
        vision_rows,
        time_basis=args.time_basis,
        tolerance_ms=args.tolerance_ms,
    )
    out_csv = Path(args.out_csv)
    out_summary = Path(args.out_summary) if args.out_summary else out_csv.with_suffix(".summary.json")
    write_alignment(result, out_csv, out_summary)
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
