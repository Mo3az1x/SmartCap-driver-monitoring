# SmartCap AI - driver state from the camera (laptop version)

AI part of the SmartCap graduation project. It runs on the **laptop webcam and laptop screen**.
The same code is meant to move to the Raspberry Pi and the cap's ESP32-S3 camera later (see "Next steps").
This folder is **separate from the team repo**: nothing in `edge_ai/` or `gateway/` was changed.

## Final result (what is in this project)

The main model is an **image CNN** that looks at a cropped face and classifies it as
`alert`, `microsleep` (closed eyes) or `yawning`. A small ANN is trained next to it for comparison.
Face crop: MediaPipe. Training data: public FL3D dataset (53,331 frames).

| Check | CNN | ANN |
|---|---|---|
| Validation on FL3D (split **by video**, 10,418 images) | 97.27% | 89.01% |
| Parameters | 583,267 | 6,441,475 |
| My own laptop-camera images (`reports/img_test`) | 97.4% | 63.9% |
| Live demo, per frame (`reports/live_eval`) | 97.3% | - |
| Live demo, state after smoothing + rules (UNKNOWN counts as wrong) | 96.3% | - |
| Live pipeline time (face crop + CNN + state), laptop | about 30 ms / frame | - |

The CNN is clearly better than the ANN: it uses about 10x fewer parameters and generalises to a different camera.
The ANN fails on a different camera and lighting (microsleep recall 0% on my images).

### Honest limits (write these in the report)

- All own-image and live tests use **one face and one camera** (the developer's), and the drowsiness is **acted**, not real.
  The numbers show the model works, not that it is ready for any driver in a car.
- The best epoch is picked on the validation set itself, so the 97.27% is slightly optimistic
  (last-epoch validation accuracy is about 95% for the CNN and 87% for the ANN).
- FL3D folders are video ids, not person ids, so the same driver may appear in several videos.
- `ms / image` in `reports/img/img_compare.md` was measured on the GPU with batch size 1; it is dominated by overhead and is
  not a fair speed comparison. Use the live pipeline time (about 30 ms / frame) instead.
- The model was trained and tested on the **laptop camera**. It has not been tested on the cap's ESP32 camera; expect lower accuracy there.
- Research prototype, not a medical or certified safety system.

## Folder contents

| Path | What it is |
|---|---|
| `SmartCap_Images_ANN_CNN.ipynb` | Notebook: prepare data, train ANN and CNN, compare, test on own images |
| `src/train_img.py`, `src/imgmodels.py` | Training code for the ANN and CNN |
| `src/prepare_dataset.py` | Organises FL3D into `dataset_prepared/<class>/*.jpg` (reads the JSON labels, split-by-video names) |
| `src/capture_images.py` | Capture your own labelled face images from the camera |
| `src/test_img.py` | Test both models on your own images |
| `src/live_img.py` | **Live demo** (camera -> face -> CNN -> state), optional `--eval` mode |
| `src/export_img_onnx.py` | Export the CNN to ONNX (`onnxruntime` only, no torch needed) |
| `src/collect.py`, `src/train.py`, `src/train_dl.py`, `src/live.py`, `src/engine.py`, `src/windowing.py`, `src/landmarks.py` | Earlier landmark-based pipeline (EAR/MAR/head pose over 10 s windows -> RandomForest or 1D-CNN+GRU) |
| `models/` | Trained image models: `img_cnn.pt`, `img_cnn.onnx` + `img_cnn.json` (keep together), `img_ann.pt`, MediaPipe `face_landmarker.task` |
| `reports/img`, `reports/img_test`, `reports/live_eval`, `reports/compare_samples.png` | Results for the report |
| `reports/synthetic_pipeline_check/` | Reports from **fake data** that only prove the code runs. Do not quote them. |
| `my_images/` | The developer's own labelled face images (305 images). Remove before sharing publicly. |
| `data/raw/` | One landmark recording (P01), numbers only |
| `tests/test_pipeline.py`, `tools/make_synthetic_data.py` | Camera-free tests on fake data |

**Not included** (too large, and ignored by git): the FL3D images (`dataset_prepared`, about 620 MB), the Python virtual environments,
and the trained landmark-based models (`driver_state_*.onnx/.joblib`). You can recreate them with the steps below.

## Setup (Windows)

```powershell
cd smartcap_ai
python -m venv .venv313
.venv313\Scripts\python.exe -m pip install -r requirements-dl.txt
```

If PowerShell blocks activation scripts, do not activate the venv: call `.venv313\Scripts\python.exe` directly (as below).
For an NVIDIA GPU install the CUDA build of torch from https://pytorch.org/get-started/locally (training also works on CPU, only slower).

## Run the live demo

```powershell
.venv313\Scripts\python.exe src\live_img.py --model models\img_cnn.pt --camera 1
.venv313\Scripts\python.exe src\live_img.py --model models\img_cnn.pt --camera 1 --eval   # measures accuracy, writes reports/live_eval
```

`--camera` accepts a number (0 or 1 = USB/laptop camera; try the other if you get the rear camera), `esp32`, an IP address, or a full URL.
No video or image is stored by the live demo. `UNKNOWN` is a fixed rule (no face, warm-up, or low confidence), never `ATTENTIVE`.

## Retrain the image models

1. Download FL3D: https://www.kaggle.com/datasets/matjazmuc/frame-level-driver-drowsiness-detection-fl3d (or let the notebook use `kagglehub`).
2. Open `SmartCap_Images_ANN_CNN.ipynb` in VS Code, select the `.venv313` kernel, and run the cells in order.
   Or from a terminal:

```powershell
.venv313\Scripts\python.exe src\prepare_dataset.py --src "<path to FL3D>" --dst dataset_prepared
.venv313\Scripts\python.exe src\train_img.py --data dataset_prepared
.venv313\Scripts\python.exe src\capture_images.py --participant P01 --classes alert microsleep yawning --camera 1
.venv313\Scripts\python.exe src\test_img.py --images my_images --models models\img_cnn.pt models\img_ann.pt
```

Capture keys: `1` alert, `2` microsleep, `3` yawning, `0` pause, `q` quit. Record about 100 images per class under different lighting and poses.
Recordings need written consent and only **acted** drowsiness (no sleep deprivation), as the interface spec requires.

## Earlier landmark-based pipeline (optional)

```
Webcam -> MediaPipe (pretrained) -> EAR, MAR, head pose per frame -> 10 s window -> RandomForest or 1D-CNN+GRU -> driver_state
```

```powershell
.venv313\Scripts\python.exe src\collect.py --participant P01 --camera 1
.venv313\Scripts\python.exe src\train.py
.venv313\Scripts\python.exe src\train_dl.py --arch large --epochs 30 --seeds 2
.venv313\Scripts\python.exe src\live.py --model models\driver_state_dl_large.onnx --camera 1
```

On the single-person recording (515 windows) the large deep model and the RandomForest reached about 0.96 and 0.94, but that evaluation
is **not subject-independent** (blocks of the same person), so it is optimistic. Recording more people is what would improve it.
`risk_score = P(drowsy) + 0.4 * P(distracted)` and the `HIGH_RISK` cutoff of 0.85 are starting choices to tune with the team.

## Test without a camera

```powershell
.venv313\Scripts\python.exe tests\test_pipeline.py
```

Uses **fake** data (`tools/make_synthetic_data.py`) only to check that the code runs. Do not put fake data in `data/raw/`.

## Next steps

1. **Cap camera (ESP32-S3):** join the `SmartCap-CAM` Wi-Fi, open `http://192.168.4.1/` once to check the picture, close the tab
   (the ESP32 serves one client), then run `live_img.py --camera esp32 --eval`. Expect lower accuracy; then capture images with the cap camera and fine-tune.
2. **Raspberry Pi:** use `export_img_onnx.py` and `onnxruntime` (`cv2.CAP_DSHOW` is Windows-only; Linux uses the default backend). Measure speed on the Pi CPU, not from the GPU table.
3. **IMU fusion:** add `pitch_deg`, `nodding_score` and `head_down_duration_ms` as extra inputs.
4. **If you move this into the team repo:** its `.gitignore` ignores `*.csv`, `*.onnx`, `*.pt`, `*.tflite`, `*.mp4`, so models, data and reports will be skipped.
   Decide with the team where they live. Also agree with Team 3 about overlap with `edge_ai/vision` and `gateway/receiver/vision_worker.py`.
