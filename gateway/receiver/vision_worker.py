#!/usr/bin/env python3
"""
SmartCap vision worker - reads the latest ESP32 camera frame from the gateway
and extracts basic visual driver-state features.

It does NOT detect drowsiness by itself.
It extracts behavioral cues only:
  - Face landmarks
  - EAR (eye aspect ratio)
  - MAR (mouth aspect ratio)
  - Eye closure duration
  - PERCLOS-like score
  - Yawn-like mouth opening

This module is designed to run alongside gateway_receiver.py.
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# -----------------------------
# Utility functions
# -----------------------------

def now_ms():
    return int(time.time() * 1000)


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_ear(landmarks, w, h):
    """
    Eye Aspect Ratio using MediaPipe Face Landmarker indices.
    Left eye: 33, 133, 159, 145, 160, 144
    Right eye: 362, 263, 386, 374, 385, 380
    """
    left = [
        (landmarks[33].x * w, landmarks[33].y * h),
        (landmarks[133].x * w, landmarks[133].y * h),
        (landmarks[159].x * w, landmarks[159].y * h),
        (landmarks[145].x * w, landmarks[145].y * h),
        (landmarks[160].x * w, landmarks[160].y * h),
        (landmarks[144].x * w, landmarks[144].y * h),
    ]
    right = [
        (landmarks[362].x * w, landmarks[362].y * h),
        (landmarks[263].x * w, landmarks[263].y * h),
        (landmarks[386].x * w, landmarks[386].y * h),
        (landmarks[374].x * w, landmarks[374].y * h),
        (landmarks[385].x * w, landmarks[385].y * h),
        (landmarks[380].x * w, landmarks[380].y * h),
    ]

    def ear(points):
        vertical1 = euclidean(points[2], points[3])
        vertical2 = euclidean(points[4], points[5])
        horizontal = euclidean(points[0], points[1])
        if horizontal == 0:
            return 0.0
        return (vertical1 + vertical2) / (2.0 * horizontal)

    return (ear(left) + ear(right)) / 2.0


def compute_mar(landmarks, w, h):
    """
    Mouth Aspect Ratio using MediaPipe Face Landmarker indices.
    Upper lip: 13
    Lower lip: 14
    Mouth corners: 61, 291
    """
    upper = (landmarks[13].x * w, landmarks[13].y * h)
    lower = (landmarks[14].x * w, landmarks[14].y * h)
    left = (landmarks[61].x * w, landmarks[61].y * h)
    right = (landmarks[291].x * w, landmarks[291].y * h)

    vertical = euclidean(upper, lower)
    horizontal = euclidean(left, right)

    if horizontal == 0:
        return 0.0


    return vertical / horizontal


# -----------------------------
# CSV logger
# -----------------------------

class CsvLog:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        self.file = None
        self.writer = None

    def write(self, row):
        if self.file is None:
            self.file = open(self.path, "w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


# -----------------------------
# Vision worker
# -----------------------------

class VisionWorker:
    def __init__(self, args):
        self.args = args

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.closed = False
        self.closed_start_ms = None
        self.closure_duration_s = 0.0

        self.ear_history = []
        self.perclos_window_s = 30.0

        self.yawn_active = False
        self.yawn_start_ms = None

        self.face_detected = False
        self.ear = 0.0
        self.mar = 0.0
        self.eye_state = "NO_FACE"
        self.vision_status = "WAITING"
        self.vision_quality = 0.0

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        self.log_path = outdir / f"vision_features_{stamp}.csv"
        self.log = CsvLog(
            self.log_path,
            [
                "timestamp_ms",
                "face_detected",
                "ear",
                "mar",
                "eye_state",
                "closure_duration_s",
                "perclos_30s",
                "yawn_active",
                "vision_status",
                "vision_quality",
            ],
        )

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        h, w = frame.shape[:2]
        timestamp = now_ms()

        if not results.multi_face_landmarks:
            self.face_detected = False
            self.eye_state = "NO_FACE"
            self.vision_status = "NO_FACE"
            self.vision_quality = 0.0
            self.reset_closure()
            self.write_log(timestamp)
            return frame

        self.face_detected = True
        landmarks = results.multi_face_landmarks[0].landmark

        self.ear = compute_ear(landmarks, w, h)
        self.mar = compute_mar(landmarks, w, h)

        self.update_eye_state(timestamp)
        self.update_perclos(timestamp)
        self.update_yawn(timestamp)

        self.vision_status = "OK"
        self.vision_quality = self.estimate_quality()

        self.write_log(timestamp)

        if self.args.show:
            self.draw_overlay(frame)

        return frame

    def update_eye_state(self, timestamp):
        if self.ear < self.args.ear_closed_threshold:
            if not self.closed:
                self.closed = True
                self.closed_start_ms = timestamp
            self.closure_duration_s = (timestamp - self.closed_start_ms) / 1000.0
            self.eye_state = "CLOSED"
        else:
            if self.closed:
                self.closed = False
                self.closed_start_ms = None
            self.closure_duration_s = 0.0
            self.eye_state = "OPEN"

    def update_perclos(self, timestamp):
        self.ear_history.append((timestamp, self.ear))

        cutoff = timestamp - int(self.perclos_window_s * 1000)
        self.ear_history = [
            (t, e) for t, e in self.ear_history if t >= cutoff
        ]

        if not self.ear_history:
            self.perclos = 0.0
            return

        closed_count = sum(
            1 for _, e in self.ear_history
            if e < self.args.ear_closed_threshold
        )

        self.perclos = closed_count / len(self.ear_history)

    def update_yawn(self, timestamp):
        if self.mar > self.args.mar_yawn_threshold:
            if not self.yawn_active:
                self.yawn_active = True
                self.yawn_start_ms = timestamp
        else:
            if self.yawn_active:
                self.yawn_active = False
                self.yawn_start_ms = None

    def reset_closure(self):
        self.closed = False
        self.closed_start_ms = None
        self.closure_duration_s = 0.0
        self.eye_state = "NO_FACE"

    def estimate_quality(self):
        """
        Simple quality proxy:
        - Face detected
        - EAR and MAR available
        - Reasonable landmark confidence assumed by MediaPipe
        """
        if not self.face_detected:
            return 0.0

        quality = 1.0

        if self.ear <= 0:
            quality -= 0.3

        if self.mar <= 0:
            quality -= 0.2

        return max(0.0, min(1.0, quality))

    def write_log(self, timestamp):
        self.log.write({
            "timestamp_ms": timestamp,
            "face_detected": self.face_detected,
            "ear": round(self.ear, 4),
            "mar": round(self.mar, 4),
            "eye_state": self.eye_state,
            "closure_duration_s": round(self.closure_duration_s, 3),
            "perclos_30s": round(self.perclos, 4),
            "yawn_active": self.yawn_active,
            "vision_status": self.vision_status,
            "vision_quality": round(self.vision_quality, 3),
        })

    def draw_overlay(self, frame):
        text_lines = [
            f"EAR: {self.ear:.3f}",
            f"MAR: {self.mar:.3f}",
            f"Eye: {self.eye_state}",
            f"Closure: {self.closure_duration_s:.2f}s",
            f"PERCLOS: {self.perclos:.3f}",
            f"Yawn: {self.yawn_active}",
        ]

        y = 30
        for line in text_lines:
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            y += 25

        return frame

    def run(self):
        cap = cv2.VideoCapture(self.args.camera_url)

        if not cap.isOpened():
            print(f"Cannot open camera stream: {self.args.camera_url}")
            return

        print(f"Vision worker started: {self.args.camera_url}")
        print(f"Logging vision features to: {self.log_path}")

        while True:
            ok, frame = cap.read()

            if not ok:
                print("Camera stream lost. Reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(self.args.camera_url)
                continue

            frame = self.process_frame(frame)

            if self.args.show:
                cv2.imshow("SmartCap Vision Worker", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        cap.release()
        self.log.close()
        cv2.destroyAllWindows()


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SmartCap vision worker"
    )

    parser.add_argument(
        "--camera-url",
        default="http://192.168.4.1/stream",
        help="ESP32 MJPEG stream URL",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live preview window",
    )

    parser.add_argument(
        "--ear-closed-threshold",
        type=float,
        default=0.20,
        help="EAR threshold below which eye is considered closed",
    )

    parser.add_argument(
        "--mar-yawn-threshold",
        type=float,
        default=0.60,
        help="MAR threshold above which yawn-like mouth opening is considered",
    )

    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parent / "output"),
        help="Output directory for CSV logs",
    )

    args = parser.parse_args()

    worker = VisionWorker(args)
    worker.run()


if __name__ == "__main__":
    main()
