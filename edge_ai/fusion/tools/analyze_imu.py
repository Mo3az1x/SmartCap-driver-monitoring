#!/usr/bin/env python3
"""
SmartCap Team 1 - IMU filtering comparison and noise analysis.

Examples:
    python analyze.py data/P1_still_*.csv
    python analyze.py data/P1_nod_*.csv --plot
    python analyze.py data/P1_still_motor0.csv data/P1_still_motor2.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "device_timestamp_ms",
    "pitch_deg",
    "ax_g",
    "ay_g",
    "az_g",
    "gx_dps",
    "gy_dps",
    "gz_dps",
]


def lag_ms(reference, signal, sample_rate_hz):
    reference = np.asarray(reference) - np.mean(reference)
    signal = np.asarray(signal) - np.mean(signal)

    correlation = np.correlate(
        signal,
        reference,
        mode="full",
    )

    lags = np.arange(
        -len(reference) + 1,
        len(reference),
    )

    return lags[np.argmax(correlation)] / sample_rate_hz * 1000.0


def get_column_if_present(dataframe, name):
    if name in dataframe.columns:
        return dataframe[name]
    return None


def main():
    file_paths = [
        argument
        for argument in sys.argv[1:]
        if not argument.startswith("--")
    ]

    make_plot = "--plot" in sys.argv

    if not file_paths:
        sys.exit(__doc__)

    summary_rows = []
    first_data = None
    first_path = None

    for file_path in file_paths:
        data = pd.read_csv(file_path)

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in data.columns
        ]

        if missing:
            print(
                f"Skipping {file_path}; missing columns: {missing}"
            )
            continue

        if first_data is None:
            first_data = data
            first_path = file_path

        timestamps = data["device_timestamp_ms"].to_numpy()

        if len(timestamps) < 2:
            print(f"Skipping {file_path}; not enough rows.")
            continue

        sample_rate_hz = 1000.0 / np.median(
            np.diff(timestamps)
        )

        result = {
            "file": Path(file_path).name,
            "rows": len(data),
            "fs_hz": round(sample_rate_hz, 2),
            "pitch_deg_std": data["pitch_deg"].std(),
            "ax_g_std": data["ax_g"].std(),
            "ay_g_std": data["ay_g"].std(),
            "az_g_std": data["az_g"].std(),
            "gx_dps_std": data["gx_dps"].std(),
            "gy_dps_std": data["gy_dps"].std(),
            "gz_dps_std": data["gz_dps"].std(),
            "motion_energy_mean": data["motion_energy"].mean()
            if "motion_energy" in data.columns
            else np.nan,
            "motion_energy_std": data["motion_energy"].std()
            if "motion_energy" in data.columns
            else np.nan,
        }

        pitch_raw = get_column_if_present(
            data,
            "pitch_raw_deg",
        )

        pitch_lpf = get_column_if_present(
            data,
            "pitch_lpf_deg",
        )

        if pitch_raw is not None and pitch_lpf is not None:
            result["pitch_raw_std"] = pitch_raw.std()
            result["pitch_lpf_std"] = pitch_lpf.std()
            result["lpf_lag_ms"] = lag_ms(
                pitch_raw,
                pitch_lpf,
                sample_rate_hz,
            )

        if pitch_raw is not None:
            result["comp_lag_ms"] = lag_ms(
                pitch_raw,
                data["pitch_deg"],
                sample_rate_hz,
            )

        summary_rows.append(result)

    if not summary_rows:
        sys.exit("No valid CSV files could be analyzed.")

    summary = pd.DataFrame(summary_rows).set_index("file")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 100)

    print("\nAnalysis summary:\n")
    print(summary.round(4).T)

    output_path = "analysis_summary.csv"
    summary.round(5).to_csv(output_path)

    print(f"\nSaved summary: {output_path}")

    if make_plot and first_data is not None:
        import matplotlib.pyplot as plt

        timestamps_s = (
            first_data["device_timestamp_ms"]
            - first_data["device_timestamp_ms"].iloc[0]
        ) / 1000.0

        plt.figure(figsize=(11, 5))

        if "pitch_raw_deg" in first_data.columns:
            plt.plot(
                timestamps_s,
                first_data["pitch_raw_deg"],
                label="Raw accelerometer pitch",
                alpha=0.5,
            )

        if "pitch_lpf_deg" in first_data.columns:
            plt.plot(
                timestamps_s,
                first_data["pitch_lpf_deg"],
                label="Low-pass pitch",
            )

        plt.plot(
            timestamps_s,
            first_data["pitch_deg"],
            label="Complementary pitch",
            linewidth=2,
        )

        plt.xlabel("Time (seconds)")
        plt.ylabel("Pitch (degrees)")
        plt.title(f"Filter comparison: {Path(first_path).name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        figure_path = "filter_comparison.png"
        plt.savefig(figure_path, dpi=150)

        print(f"Saved figure: {figure_path}")
        plt.show()


if __name__ == "__main__":
    main()