# SmartCap AI – laptop version (vision only)

Prototype of the AI part of SmartCap. It runs on the **laptop webcam + laptop screen** first; the same code is meant to move to the Raspberry Pi later.
This folder is **separate from the repo** – nothing in `smartcap-driver-monitoring` was changed.

## الفكرة ببساطة

```
Webcam → MediaPipe (pretrained, نقط الوش) → numbers per frame (EAR, MAR, head pose)
       → window 10 s → 21 features → RandomForest (بيتدرّب على تسجيلاتك) → driver_state
```

- الـ **MediaPipe** موديل جاهز مش بنتدرّبه. اللي بنتدرّبه هو الـ classifier الصغير اللي بيقرر الحالة من سلوك آخر 10 ثواني.
- التصنيف: `attentive` / `distracted` / `drowsy` (تمثيل فقط). الـ `UNKNOWN` **قواعد ثابتة مش موديل**: مفيش وش، أول 10 ثواني، أو الموديل مش واثق → UNKNOWN، عمره ما يبقى ATTENTIVE.
- بيطلّع رسالة `driver_state` زي الـ spec (القسم 10). الـ `imu_quality` = `null` لأن الـ IMU لسه مش مدمج.
- مفيش فيديو ولا صور بتتخزّن – أرقام بس (قسم 14 في الـ spec).

## Setup (Windows)

```powershell
cd smartcap_ai
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1) Record data  (مرة لكل شخص)

```powershell
python src/collect.py --participant P01
```

Keys: `1` attentive · `2` distracted · `3` drowsy (**simulated**) · `0` pause · `q` quit

| Key | Do |
|---|---|
| 1 attentive | بص قدامك كأنك سايق، رمش عادي |
| 2 distracted | بص يمين/شمال/تحت (بعيد عن الطريق) كل كام ثانية |
| 3 drowsy | رمش بطيء، اقفل عينيك 1–3 ثواني، تثاؤب، خلّي راسك تنزل |

- الهدف: **3 دقايق على الأقل لكل label، ولأكتر من شخص** (3+ أشخاص يفضّل). الموديل الحقيقي بيتحدد بالداتا دي.
- اتنوّع: إضاءة مختلفة، بنظارة/بدون، مسافات مختلفة عن الكاميرا.
- قواعد الـ spec: موافقة مكتوبة، تمثيل نعاس بس (ممنوع حرمان من النوم)، تجربة في معمل/عربية واقفة، أكواد مجهولة (P01, P02…).

## 2) Train

```powershell
python src/train.py
```

Writes `models/driver_state_model.joblib` and `reports/train_report.md` (accuracy, confusion matrix, أهم features).
مع شخصين أو أكتر التقييم **subject-independent** (الشخص المُختبَر عمره ما يبقى في التدريب). مع شخص واحد الأرقام متفائلة وهيقولك كده.

## 3) Live demo

```powershell
python src/live.py
python src/live.py --log reports/live_log.jsonl     # بيكتب driver_state JSON كمان
```

أول ~10 ثواني UNKNOWN (warm-up). `q` للخروج.

## 4) Deep learning model (1D-CNN + GRU) – نفس الداتا، موديل تاني

مفيش تسجيل جديد. بعد ما تسجّل (الخطوة 1) شغّل:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dl.txt
.venv\Scripts\python.exe src\train_dl.py
.venv\Scripts\python.exe src\live.py --model models\driver_state_dl.onnx --camera 1
```

- بياخد **التسلسل الخام** لآخر 10 ثواني (100 خطوة × 5 قنوات: EAR, MAR, yaw, pitch, roll) بدل الـ 21 رقم الملخّصين. الـ CNN بيلقط أنماط قصيرة (رمشة، تثاؤب، لفّة راس) والـ GRU بيفهم ترتيبها في الوقت. حوالي 13 ألف parameter.
- بيتدرّب جنب الـ RandomForest على **نفس الـ folds** وبيطلّع المقارنة في `reports/train_dl_report.md`. لو الـ DL مش أحسن، **قولها كده في التقرير** – مع داتا قليلة (كام شخص / كام دقيقة) الـ RandomForest كتير بيكسب. الـ DL بيبدأ يفرق لما الداتا تكبر.
- الناتج `models/driver_state_dl.onnx` ملف واحد + `.json` جنبه (لازم الاتنين مع بعض). بيشتغل بـ `onnxruntime` بس، يعني خفيف على الـ Raspberry Pi، والـ torch بيتحتاج للتدريب بس.
- الـ `live.py` والـ engine نفس القواعد: مفيش وش / warm-up / ثقة قليلة = UNKNOWN.

## 5) الموديل الكبير على Colab (GPU)

افتح `SmartCap_Colab_Training.ipynb` في Colab، اختار `Runtime ▸ Change runtime type ▸ T4 GPU`، وشغّل الخلايا بالترتيب. هتحتاج تـ upload ملفين: `smartcap_ai.zip` و`data_raw.zip` (فولدر `data\raw` بتاعك متضغوط).

- `python src/train_dl.py --arch large --epochs 80 --seeds 5` هو اللي الـ notebook بيشغّله: ResNet-1D + BiGRU + attention (~650 ألف parameter)، augmentation أقوى، و**ensemble من 5 seeds** (بيتوسّط احتمالاتهم). بيطلّع `models/driver_state_dl_large.onnx` + `.json` (لازم الاتنين مع بعض) و`reports/train_dl_large_report.md`.
- على اللابتوب: `python src\live.py --model models\driver_state_dl_large.onnx --camera 1`. الـ inference ~2 ms لكل window على CPU عادي (قياسي هنا)، فمفيش مشكلة على Pi.
- **الأهم:** الموديل الكبير لوحده مش بيضمن دقة أحسن. مع داتا قليلة ممكن يحفظها ويبقى أسوأ من الـ RandomForest على ناس جديدة. الـ report بيقارنهم بصدق على أشخاص مش في التدريب. اللي بيرفع الجودة فعلاً: **ناس أكتر**، إضاءات وزوايا أكتر، وتسجيل أطول.

## 6) كاميرا الكاب (ESP32-S3) بدل كاميرا اللابتوب

1. شغّل الـ ESP32 (firmware `camera_stream`) ووصّل اللابتوب بشبكة الواي فاي **`SmartCap-CAM`** (الباسورد في الـ firmware). اللابتوب هيفقد النت طول ما هو متوصل بيها.
2. افتح `http://192.168.4.1/` في المتصفح وتأكد إن الصورة شغالة، وبعدين **اقفل التاب** (الـ ESP32 بيخدم client واحد بس على `/stream`).
3. سجّل داتا جديدة في فولدر منفصل:

```powershell
.venv313\Scripts\python.exe src\collect.py --participant P01 --camera esp32 --outdir data\raw_cap
.venv313\Scripts\python.exe src\live.py --model models\driver_state_dl_large.onnx --camera esp32
```

- `--camera` بيقبل: رقم (0/1 = كاميرا USB)، أو `esp32`، أو IP (`192.168.4.1`)، أو URL كامل.
- لو بطيء: `http://192.168.4.1/control?var=framesize&val=5` (QVGA) أو `val=8` (VGA) في المتصفح، وبعدين اقفله.
- **الموديل المتدرّب على كاميرا اللابتوب مش هيشتغل كويس على كاميرا الكاب** (زاوية وعدسة وجودة مختلفة). لازم تسجّل داتا جديدة بكاميرا الكاب وتتدرّب عليها (`data_raw.zip` من `data\raw_cap`)، ومتخلطهمش.

## 7) موديلات الصور: ANN و CNN (من VS Code)

افتح `SmartCap_Images_ANN_CNN.ipynb` في VS Code، اختار kernel `.venv313`، وشغّل الخلايا بالترتيب: تنزيل FL3D ← `prepare_dataset.py` ← تقسيم 80/20 ← تدريب ANN وCNN (loss / accuracy / وقت كل epoch) ← مقارنة ← test على صورك (`capture_images.py` + `test_img.py`).
نفس الحاجات متاحة من الـ terminal: `python src/train_img.py --data dataset` و `python src/test_img.py --images my_images --models models/img_cnn.pt models/img_ann.pt`.
الصور بتتحفظ على جهازك بس (`my_images`, `dataset`) ومتترفعش على git.

## Test without a camera

```powershell
python tests/test_pipeline.py
```

بيستخدم داتا **مزيفة** (`tools/make_synthetic_data.py`) عشان يتأكد إن الـ pipeline شغال. الدقة العالية هناك معناها صفر عن السواقين الحقيقيين. **متحطّهاش في `data/raw/`.**

## Honest limits

- Tested here: landmark maths, training (RandomForest and CNN+GRU, ONNX export checked against PyTorch), engine rules (UNKNOWN cases), MediaPipe API call on a blank frame, replay mode.
  **Not tested:** a real webcam with a real face (no camera access in my environment) – first real run is on your laptop.
- Thresholds (`closed_ratio`, `mar_open`, `turn_yaw` in `src/windowing.py`) are starting values; the EAR/head-pitch baseline is rolling over 60 s, so a very long drowsy stretch can pull it down.
- Head pose is a cheap 2D estimate, not a calibrated one. It does **not** mean "looking away from the road" (spec section 9).
- `risk_score = P(drowsy) + 0.4·P(distracted)` and the `HIGH_RISK` cutoff at 0.85 are my starting choices – tune with the team.
- Research prototype, not a medical or certified safety system.

## Next steps

1. **Raspberry Pi:** same code (camera opens with the default backend on Linux, `CAP_DSHOW` is Windows-only). Then export the model to ONNX/TFLite if speed needs it.
2. **IMU fusion:** add `pitch_deg`, `nodding_score`, `head_down_duration_ms` as extra features in the same window.
3. **If you move this into the repo:** `.gitignore` there ignores `*.csv`, `*.joblib` is not ignored but `*.onnx/*.tflite/*.pt` are – decide with the team where data and the trained model live. Also agree with Team 3 about overlap with `edge_ai/vision` and `gateway/receiver/vision_worker.py`.
