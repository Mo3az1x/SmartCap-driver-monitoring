# SmartCap

A wearable multimodal driver monitoring system with edge AI and prototype vehicle CAN integration.

## Project Goal

SmartCap combines camera-based facial features and IMU-based head-motion features to estimate drowsiness-related and distraction-related driver states.

## Main Components

- ESP32-S3 camera node
- MPU6050 IMU
- Haptic vibration actuator
- Raspberry Pi or Laptop edge gateway
- Temporal AI and sensor fusion
- Prototype CAN integration

## System Flow

```text
Camera + IMU
     ↓
ESP32-S3 / Arduino during development
     ↓
Wi-Fi / USB Serial
     ↓
Laptop or Raspberry Pi
     ↓
Vision + IMU Feature Extraction
     ↓
Temporal Sensor Fusion
     ↓
Risk Assessment
     ↓
Haptic Alert + Prototype CAN Message
```

## Teams

- Team 1: IMU and Haptic
- Team 2: ESP32 Camera and Communication
- Team 3: AI and Computer Vision
- Wave 2: Gateway, CAN, Power, and Dashboard

## Important Documents

- Interface specification: `protocol/SMARTCAP_INTERFACE_SPEC_v1.md`
- Dataset rules: `datasets/README.md`
- Experiment logs: `experiments/logs/`

## Current Milestone

Build a synchronized multimodal sensing prototype:

```text
Camera stream + IMU stream + manual vibration command
```

The first milestone is not yet a complete drowsiness detector.

## Repository Rules

- Follow the interface specification.
- Do not change message fields or units without informing all teams.
- Do not upload participant videos or private raw datasets.
- Use participant codes instead of names.
- Record experiments in `experiments/logs/`.
- Use branches or pull requests for major changes.

- aliaa
- Omar
