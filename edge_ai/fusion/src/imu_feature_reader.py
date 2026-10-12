import csv
import json
import time
from datetime import datetime
from pathlib import Path

import serial


SERIAL_PORT = "COM5"
BAUD_RATE = 230400
SERIAL_TIMEOUT_SECONDS = 1

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = PROJECT_ROOT / "experiments" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = LOG_DIR / (
    f"imu_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

REQUIRED_FIELDS = [
    "type",
    "sequence_id",
    "device_timestamp_ms",
    "pitch_deg",
    "roll_deg",
    "pitch_velocity_dps",
    "head_down_duration_ms",
    "nodding_score",
    "motion_energy",
    "imu_status",
    "calibration_status",
    "imu_quality",
]


def validate_message(message):
    if message.get("type") != "imu":
        return False, "Unexpected message type."

    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in message
    ]

    if missing_fields:
        return False, f"Missing fields: {missing_fields}"

    return True, None


def print_live_features(message, received_timestamp_ms):
    print(
        f"[{received_timestamp_ms}] "
        f"seq={message['sequence_id']} | "
        f"pitch={message['pitch_deg']:.2f} deg | "
        f"roll={message['roll_deg']:.2f} deg | "
        f"vel={message['pitch_velocity_dps']:.2f} dps | "
        f"down={message['head_down_duration_ms']} ms | "
        f"nod={message['nodding_score']:.3f} | "
        f"energy={message['motion_energy']:.3f} | "
        f"status={message['imu_status']} | "
        f"quality={message['imu_quality']:.2f}"
    )


def main():
    print(f"Opening serial port {SERIAL_PORT} at {BAUD_RATE} baud...")
    print(f"Logging valid IMU messages to: {CSV_PATH}")
    print("Press Ctrl+C to stop.\n")

    try:
        connection = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=SERIAL_TIMEOUT_SECONDS,
        )
    except serial.SerialException as error:
        print(f"Could not open {SERIAL_PORT}: {error}")
        print("Check the COM port and make sure Arduino Serial Monitor is closed.")
        return

    time.sleep(2)
    connection.reset_input_buffer()

    csv_file = None
    writer = None

    try:
        while True:
            raw_line = connection.readline().decode(
                "utf-8",
                errors="replace",
            ).strip()

            if not raw_line:
                continue

            if raw_line.startswith("#"):
                print(f"Arduino: {raw_line}")
                continue

            try:
                message = json.loads(raw_line)
            except json.JSONDecodeError:
                print(f"Invalid JSON ignored: {raw_line}")
                continue

            is_valid, error_message = validate_message(message)

            if not is_valid:
                print(f"Invalid IMU message ignored: {error_message}")
                continue

            received_timestamp_ms = int(time.time() * 1000)
            message["received_timestamp_ms"] = received_timestamp_ms

            if writer is None:
                fieldnames = list(message.keys())

                csv_file = CSV_PATH.open(
                    "w",
                    newline="",
                    encoding="utf-8",
                )

                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=fieldnames,
                )

                writer.writeheader()

            writer.writerow(message)
            csv_file.flush()

            print_live_features(message, received_timestamp_ms)

    except KeyboardInterrupt:
        print("\nStopping IMU reader...")

    finally:
        if csv_file is not None:
            csv_file.close()

        connection.close()
        print(f"Saved log: {CSV_PATH}")


if __name__ == "__main__":
    main()