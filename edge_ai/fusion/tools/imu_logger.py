#!/usr/bin/env python3
"""
SmartCap Team 1 - JSONL IMU dataset logger.

Reads one JSON IMU message per line from Arduino Serial,
adds received_timestamp_ms and a manual activity label,
then saves a CSV file.

Examples:
    python imu_logger.py --person P1 --label nod --duration 30
    python imu_logger.py --person P1 --label still --duration 20
    python imu_logger.py --port COM5 --person P2 --label look_down --duration 15
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import serial
import serial.tools.list_ports


LABELS = [
    "normal",
    "look_down",
    "nod",
    "rapid_turn",
    "still",
]


def find_port():
    ports = list(serial.tools.list_ports.comports())

    keywords = (
        "arduino",
        "ch340",
        "usb serial",
        "usb-serial",
        "cp210",
    )

    for port in ports:
        description = (port.description or "").lower()

        if any(keyword in description for keyword in keywords):
            return port.device

    return ports[0].device if ports else None


def is_valid_imu_message(message):
    return (
        isinstance(message, dict)
        and message.get("type") == "imu"
        and "sequence_id" in message
        and "device_timestamp_ms" in message
    )


def main():
    parser = argparse.ArgumentParser(
        description="SmartCap IMU JSONL dataset logger"
    )

    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=230400)
    parser.add_argument(
        "--person",
        required=True,
        help="Participant code, e.g. P1",
    )
    parser.add_argument(
        "--label",
        required=True,
        choices=LABELS,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Recording duration in seconds",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional extra text in filename",
    )
    parser.add_argument(
        "--outdir",
        default="data",
        help="Output directory",
    )

    args = parser.parse_args()

    port = args.port or find_port()

    if not port:
        sys.exit(
            "No serial port found. Use --port COM3 or another COM port."
        )

    os.makedirs(args.outdir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename_parts = [
        args.person,
        args.label,
        args.tag,
        timestamp,
    ]

    filename = "_".join(
        part for part in filename_parts if part
    ) + ".csv"

    output_path = os.path.join(args.outdir, filename)

    print(
        f"Opening {port} at {args.baud} baud.\n"
        "The Arduino may reset after opening the serial port.\n"
        "Stay still in normal posture until calibration completes.\n"
    )

    try:
        connection = serial.Serial(
            port=port,
            baudrate=args.baud,
            timeout=1,
        )
    except serial.SerialException as error:
        sys.exit(f"Could not open {port}: {error}")

    time.sleep(2)
    connection.reset_input_buffer()

    csv_file = None
    writer = None

    rows = 0
    bad_lines = 0
    dropped_samples = 0
    last_sequence_id = None

    started_recording = False
    start_time = None
    open_time = time.time()

    try:
        while True:
            raw_line = connection.readline().decode(
                "utf-8",
                errors="replace",
            ).strip()

            if not raw_line:
                if not started_recording and time.time() - open_time > 45:
                    sys.exit(
                        "No valid IMU data received. "
                        "Check board, COM port, baud rate, and firmware."
                    )
                continue

            if raw_line.startswith("#"):
                print(f"Arduino: {raw_line}")
                continue

            try:
                message = json.loads(raw_line)
            except json.JSONDecodeError:
                bad_lines += 1
                print(f"Ignored non-JSON line: {raw_line[:120]}")
                continue

            if not is_valid_imu_message(message):
                bad_lines += 1
                print(f"Ignored unexpected message: {message}")
                continue

            if message.get("calibration_status") != "CALIBRATED":
                print(
                    "Waiting for calibration... "
                    f"status={message.get('calibration_status')}"
                )
                continue

            if message.get("imu_status") not in ("OK", "DEGRADED"):
                print(
                    "Waiting for valid IMU status... "
                    f"status={message.get('imu_status')}"
                )
                continue

            received_timestamp_ms = int(time.time() * 1000)

            message["received_timestamp_ms"] = received_timestamp_ms
            message["label"] = args.label
            message["participant_id"] = args.person

            if not started_recording:
                fieldnames = list(message.keys())

                csv_file = open(
                    output_path,
                    "w",
                    newline="",
                    encoding="utf-8",
                )

                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=fieldnames,
                )

                writer.writeheader()

                started_recording = True
                start_time = time.time()

                print(
                    f"Recording label='{args.label}' "
                    f"for {args.duration:.1f}s\n"
                    f"Saving to: {output_path}"
                )

            sequence_id = message["sequence_id"]

            if (
                last_sequence_id is not None
                and sequence_id != last_sequence_id + 1
            ):
                dropped_samples += max(
                    0,
                    sequence_id - last_sequence_id - 1,
                )

            last_sequence_id = sequence_id

            writer.writerow(message)
            rows += 1

            if rows % 50 == 0:
                csv_file.flush()

                elapsed = time.time() - start_time

                print(
                    f"rows={rows} | "
                    f"elapsed={elapsed:.1f}s | "
                    f"dropped={dropped_samples} | "
                    f"pitch={message.get('pitch_deg')} | "
                    f"status={message.get('imu_status')}"
                )

            if time.time() - start_time >= args.duration:
                break

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")

    finally:
        if csv_file is not None:
            csv_file.close()

        connection.close()

    if started_recording:
        print(
            f"\nSaved {rows} rows to:\n{output_path}\n"
            f"Malformed lines: {bad_lines}\n"
            f"Dropped samples: {dropped_samples}"
        )
    else:
        print(
            "\nNo CSV was created because no calibrated, valid "
            "IMU messages were received."
        )


if __name__ == "__main__":
    main()