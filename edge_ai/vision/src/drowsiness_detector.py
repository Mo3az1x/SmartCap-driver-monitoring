"""Single-file SmartCap web dashboard with laptop webcam and demo vision cues.

Research prototype only. It does not connect to a real vehicle, CAN bus, IMU,
GPS, or validated driver-safety classifier. Camera-derived states are heuristics.

Install with Python 3.12:
    py -3.12 -m pip install flask opencv-python mediapipe==0.10.14
Run from the repository root:
    py -3.12 path\\to\\smartcap_web_dashboard.py --model edge_ai\\vision\\models\\face_landmarker.task
Then open http://127.0.0.1:5000 in your browser.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
from flask import Flask, Response, jsonify, render_template_string


# ---------------------------- Demo tuning values ----------------------------
EAR_CLOSED_THRESHOLD = 0.20
EYES_CLOSED_SECONDS = 1.5
MAR_YAWN_THRESHOLD = 0.55
YAWN_SECONDS = 1.0
HEAD_TURN_RATIO = 0.18
HEAD_TURN_SECONDS = 1.5
CALIBRATION_SECONDS = 3.0
CAMERA_INDEXES = (0, 1, 2)

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH = (13, 14, 78, 308)
NOSE_TIP, LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 1, 33, 263

app = Flask(__name__)
lock = threading.Lock()
runtime = {
    "jpeg": None,
    "state": "STARTING",
    "risk": 0,
    "alerts": [],
    "ear": None,
    "mar": None,
    "head_offset": None,
    "camera": "Starting",
    "calibration": True,
    "updated": None,
    "events": deque(maxlen=30),
    "stop": False,
}


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def eye_ratio(points, ids):
    p = [points[i] for i in ids]
    return (distance(p[1], p[5]) + distance(p[2], p[4])) / (2 * distance(p[0], p[3]) + 1e-6)


def mouth_ratio(points):
    top, bottom, left, right = (points[i] for i in MOUTH)
    return distance(top, bottom) / (distance(left, right) + 1e-6)


def head_measure(points):
    left, right, nose = points[LEFT_EYE_OUTER], points[RIGHT_EYE_OUTER], points[NOSE_TIP]
    midpoint = (left.x + right.x) / 2
    return (nose.x - midpoint) / (abs(right.x - left.x) + 1e-6)


def classify(face_found, eye_seconds, yawn_seconds, turn_seconds):
    if not face_found:
        return "UNKNOWN", 0, ["Driver face not visible. Check camera position and lighting."]
    eyes = eye_seconds >= EYES_CLOSED_SECONDS
    yawn = yawn_seconds >= YAWN_SECONDS
    turned = turn_seconds >= HEAD_TURN_SECONDS
    alerts = []
    if eyes:
        alerts.append("Prolonged eye closure detected")
    if yawn:
        alerts.append("Possible yawn detected")
    if turned:
        alerts.append("Head turned away from calibrated forward position")
    if eyes and (turned or yawn):
        return "CRITICAL", 95, alerts
    if eyes:
        return "DROWSY", 78, alerts
    if turned and yawn:
        return "CRITICAL", 88, alerts
    if turned:
        return "DISTRACTED", 62, alerts
    if yawn:
        return "DROWSY", 48, alerts
    return "NORMAL", 8, ["No sustained warning cues detected"]


def camera_worker(model_path: Path, camera_index: int):
    camera = None
    try:
        for index in dict.fromkeys((camera_index, *CAMERA_INDEXES)):
            for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
                candidate = cv2.VideoCapture(index, backend)
                if candidate.isOpened():
                    ok, frame = candidate.read()
                    if ok and frame is not None:
                        camera = candidate
                        break
                candidate.release()
            if camera is not None:
                break
        if camera is None:
            with lock:
                runtime["camera"] = "Disconnected"
                runtime["state"] = "UNKNOWN"
                runtime["alerts"] = ["Webcam unavailable. Close other camera apps and check Windows permissions."]
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        calibration_values = deque(maxlen=120)
        center = None
        calibration_started = time.monotonic()
        eye_start = yawn_start = turn_start = None
        previous_state = None
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                with lock:
                    if runtime["stop"]:
                        break
                ok, frame = camera.read()
                now = time.monotonic()
                if not ok:
                    time.sleep(0.03)
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, int(now * 1000))
                found = bool(result.face_landmarks)
                ear = mar = head_offset = None
                calibration = center is None
                if found:
                    points = result.face_landmarks[0]
                    ear = (eye_ratio(points, LEFT_EYE) + eye_ratio(points, RIGHT_EYE)) / 2
                    mar = mouth_ratio(points)
                    raw = head_measure(points)
                    if center is None:
                        calibration_values.append(raw)
                        if now - calibration_started >= CALIBRATION_SECONDS and len(calibration_values) >= 10:
                            center = sum(calibration_values) / len(calibration_values)
                    if center is not None:
                        head_offset = raw - center
                    for landmark in points[::3]:
                        cv2.circle(frame, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 1, (73, 224, 221), -1)

                def timer(condition, start):
                    return (start if start is not None else now) if condition else None

                eye_start = timer(found and ear is not None and ear < EAR_CLOSED_THRESHOLD, eye_start)
                yawn_start = timer(found and mar is not None and mar > MAR_YAWN_THRESHOLD, yawn_start)
                turn_start = timer(found and center is not None and head_offset is not None and abs(head_offset) > HEAD_TURN_RATIO, turn_start)
                eye_s = now - eye_start if eye_start is not None else 0
                yawn_s = now - yawn_start if yawn_start is not None else 0
                turn_s = now - turn_start if turn_start is not None else 0
                state, risk, alerts = classify(found, eye_s, yawn_s, turn_s)
                stamp = datetime.now().strftime("%H:%M:%S")
                with lock:
                    runtime.update({
                        "camera": "Connected", "state": state, "risk": risk, "alerts": alerts,
                        "ear": round(ear, 3) if ear is not None else None,
                        "mar": round(mar, 3) if mar is not None else None,
                        "head_offset": round(head_offset, 3) if head_offset is not None else None,
                        "calibration": calibration, "updated": stamp,
                    })
                    if state != previous_state and state not in ("STARTING", "NORMAL", "UNKNOWN"):
                        runtime["events"].appendleft({"time": stamp, "state": state, "risk": risk, "detail": alerts[0] if alerts else state})
                    previous_state = state

                state_color = {"NORMAL": (59, 198, 132), "DISTRACTED": (55, 186, 255), "DROWSY": (53, 151, 255), "CRITICAL": (75, 75, 255), "UNKNOWN": (165, 172, 183)}.get(state, (180, 180, 180))
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 82), (13, 22, 39), -1)
                cv2.putText(frame, f"DRIVER STATE  {state}    RISK  {risk}%", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2, cv2.LINE_AA)
                subtitle = "Look forward for calibration..." if center is None else (alerts[0] if state != "NORMAL" and alerts else "Monitoring active")
                cv2.putText(frame, subtitle[:90], (22, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (229, 236, 245), 1, cv2.LINE_AA)
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    with lock:
                        runtime["jpeg"] = encoded.tobytes()
    except Exception as exc:
        with lock:
            runtime["camera"] = "Error"
            runtime["state"] = "UNKNOWN"
            runtime["alerts"] = [f"Camera/vision startup error: {exc}"]
    finally:
        if camera is not None:
            camera.release()


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmartCap | Fleet Safety</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root{--bg:#080f1c;--panel:#101b2c;--panel2:#142238;--line:#22324a;--muted:#9aa9bd;--text:#edf3fb;--teal:#48d7ca;--blue:#6ea8ff;--green:#51d6a2;--amber:#ffbd62;--red:#ff6879;--radius:17px}*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 75% -20%,#18365a 0,transparent 38%),var(--bg);color:var(--text);font:14px 'DM Sans',sans-serif}.shell{display:grid;grid-template-columns:224px 1fr;min-height:100vh}.side{background:#0b1423;border-right:1px solid var(--line);padding:25px 16px;display:flex;flex-direction:column;gap:32px}.brand{display:flex;gap:12px;align-items:center;padding:2px 9px}.logo{height:38px;width:38px;border-radius:12px;background:linear-gradient(145deg,#58e4d1,#3688d9);display:grid;place-items:center;color:#091421;font-weight:900;font-size:19px}.brand b{font:800 17px Manrope;letter-spacing:.6px}.brand small{display:block;color:var(--muted);font-size:10px;letter-spacing:1.2px;margin-top:2px}.label{color:#74859c;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;padding:0 11px}.nav{display:grid;gap:6px}.nav a{color:#aebbd0;text-decoration:none;padding:12px;border-radius:11px;display:flex;gap:12px;align-items:center}.nav a.active,.nav a:hover{background:#162640;color:white}.ico{width:19px;text-align:center;color:#8ba4c4}.sidefoot{margin-top:auto;background:#111e31;border:1px solid var(--line);border-radius:13px;padding:14px;font-size:12px;color:var(--muted)}.sidefoot b{color:var(--text)}main{padding:28px clamp(18px,3vw,44px);max-width:1680px;width:100%;margin:auto}.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:26px}.crumb{color:var(--muted);font-size:12px}.top h1{font:800 26px Manrope;margin:6px 0}.topright{display:flex;align-items:center;gap:10px}.live{color:var(--green);font-weight:700;font-size:12px;background:#122b2d;border:1px solid #20544d;padding:9px 13px;border-radius:99px}.clock{font-variant-numeric:tabular-nums;color:#aab8ca}.grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:16px}.card{background:linear-gradient(145deg,rgba(20,34,56,.96),rgba(14,25,42,.97));border:1px solid var(--line);border-radius:var(--radius);padding:19px;box-shadow:0 12px 32px #0002;min-width:0}.span-3{grid-column:span 3}.span-4{grid-column:span 4}.span-5{grid-column:span 5}.span-7{grid-column:span 7}.span-8{grid-column:span 8}.span-12{grid-column:span 12}.cardhead{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}.cardhead h2{font:700 14px Manrope;margin:0}.eyebrow{color:var(--muted);font-size:10px;letter-spacing:1.3px;text-transform:uppercase}.subtle{color:var(--muted);font-size:11px}.statecard{display:flex;align-items:center;gap:15px}.state-icon{width:53px;height:53px;border-radius:16px;background:#17312f;display:grid;place-items:center;color:var(--green);font-size:25px}.state-name{font:800 24px Manrope;margin:4px 0}.state-desc{color:var(--muted);font-size:12px}.pill{border:1px solid #2c455f;border-radius:99px;padding:6px 9px;font-size:10px;color:#aebbd0}.metric{font:800 32px Manrope;margin:8px 0 4px}.metric small{font-size:14px;color:var(--muted);font-weight:500}.progress{height:7px;background:#25364d;border-radius:9px;overflow:hidden;margin-top:13px}.progress i{display:block;height:100%;width:8%;background:linear-gradient(90deg,var(--green),var(--teal));border-radius:9px;transition:width .4s}.health-list{display:grid;gap:12px}.health-row{display:flex;align-items:center;justify-content:space-between;padding:10px 0;border-bottom:1px solid #ffffff0b}.health-row:last-child{border:0}.health-name{display:flex;align-items:center;gap:9px}.dot{width:8px;height:8px;border-radius:50%;background:#65748a}.dot.ok{background:var(--green);box-shadow:0 0 10px #51d6a255}.dot.off{background:var(--red)}.status{font-size:11px;color:var(--muted)}.video{width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:12px;background:#050a11;border:1px solid #28374d}.video-note{font-size:10px;color:var(--muted);margin-top:10px;display:flex;justify-content:space-between}.alertlist{display:grid;gap:9px;max-height:260px;overflow:auto}.alert{display:flex;gap:12px;align-items:center;padding:12px;border:1px solid #26334a;border-radius:12px;background:#0d1727}.alert-mark{height:30px;width:30px;flex:none;border-radius:9px;background:#372a20;color:var(--amber);display:grid;place-items:center}.alert b{font-size:12px}.alert small{display:block;color:var(--muted);margin-top:4px}.empty{color:var(--muted);font-size:12px;padding:18px 4px}.cuegrid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.cue{background:#0c1727;border:1px solid #24344b;border-radius:12px;padding:13px}.cue span{font-size:10px;color:var(--muted)}.cue b{display:block;font:700 18px Manrope;margin-top:6px}.footnote{color:#8495aa;font-size:10px;line-height:1.6;margin-top:16px}.tag{font-size:10px;color:#a9c3e4;border:1px solid #304563;padding:5px 8px;border-radius:7px}.bar{height:8px;width:70px;border-radius:6px;background:#26364c;overflow:hidden}.bar i{display:block;height:100%;background:var(--teal)}
@media(max-width:1050px){.shell{grid-template-columns:76px 1fr}.side{padding:20px 10px;align-items:center}.brand{padding:0}.brand>div:last-child,.label,.nav a span:last-child,.sidefoot{display:none}.nav{width:100%}.nav a{justify-content:center}.span-3,.span-4{grid-column:span 6}.span-7,.span-8,.span-5{grid-column:span 12}}
@media(max-width:650px){.shell{display:block}.side{display:none}main{padding:18px 13px}.span-3,.span-4,.span-5,.span-7,.span-8,.span-12{grid-column:span 12}.top h1{font-size:21px}.topright .clock{display:none}.card{padding:15px}.cuegrid{grid-template-columns:1fr 1fr}}
</style></head><body><div class="shell">
<aside class="side"><div class="brand"><div class="logo">S</div><div><b>SMARTCAP</b><small>FLEET SAFETY</small></div></div><div class="label">Workspace</div><nav class="nav"><a class="active" href="#overview"><span class="ico">◫</span><span>Overview</span></a><a href="#monitor"><span class="ico">◉</span><span>Live Monitor</span></a><a href="#alerts"><span class="ico">♧</span><span>Alert History</span></a><a href="#health"><span class="ico">⌁</span><span>System Health</span></a><a href="#about"><span class="ico">ⓘ</span><span>About Demo</span></a></nav><div class="sidefoot"><b>Prototype mode</b><br>Webcam vision active<br><span style="color:#ffbd62">Vehicle sensors not connected</span></div></aside>
<main id="overview"><header class="top"><div><div class="crumb">Fleet Safety / Overview</div><h1>Driver Safety Dashboard</h1><div class="subtle">Live in-cabin monitoring · Local demo session</div></div><div class="topright"><div class="live">● &nbsp; LIVE SESSION</div><div class="clock" id="clock">--:--:--</div></div></header>
<section class="grid">
<article class="card span-4"><div class="cardhead"><h2>Current Driver State</h2><span class="pill">CAMERA AI DEMO</span></div><div class="statecard"><div class="state-icon" id="stateIcon">✓</div><div><div class="eyebrow">Classification</div><div class="state-name" id="state">STARTING</div><div class="state-desc" id="stateDesc">Initializing webcam analysis</div></div></div></article>
<article class="card span-4"><div class="cardhead"><h2>Risk Score</h2><span class="subtle">Heuristic estimate</span></div><div class="metric"><span id="risk">0</span><small>%</small></div><div class="subtle" id="riskLabel">Waiting for camera data</div><div class="progress"><i id="riskBar"></i></div></article>
<article class="card span-4"><div class="cardhead"><h2>Active Alerts</h2><span class="pill">CURRENT</span></div><div class="metric" id="alertCount">0</div><div class="subtle">Current detected warning cues</div><div class="footnote">Events are recorded when the state changes to a warning level.</div></article>
<article class="card span-7" id="monitor"><div class="cardhead"><h2>In-Cabin Camera</h2><span class="tag" id="cameraTag">CONNECTING</span></div><img class="video" src="/video_feed" alt="Live webcam with driver monitoring overlay"><div class="video-note"><span>Local laptop webcam · Press Q in server terminal to stop</span><span id="updated">Awaiting frame</span></div></article>
<article class="card span-5" id="health"><div class="cardhead"><h2>System Health</h2><span class="subtle">Live connection status</span></div><div class="health-list"><div class="health-row"><div class="health-name"><i class="dot" id="camDot"></i><span>Webcam</span></div><b class="status" id="camHealth">Starting</b></div><div class="health-row"><div class="health-name"><i class="dot off"></i><span>IMU Sensor</span></div><b class="status">Not connected</b></div><div class="health-row"><div class="health-name"><i class="dot off"></i><span>Vehicle Gateway / CAN</span></div><b class="status">Not connected</b></div><div class="health-row"><div class="health-name"><i class="dot off"></i><span>GPS Location</span></div><b class="status">Not connected</b></div></div><div class="footnote">Only the laptop webcam is used in this demo. Connect the project hardware and data feed before presenting sensor status as live.</div></article>
<article class="card span-7" id="alerts"><div class="cardhead"><h2>Recent Safety Events</h2><span class="subtle">This session</span></div><div class="alertlist" id="events"><div class="empty">No warning events recorded yet. Monitoring is active.</div></div></article>
<article class="card span-5"><div class="cardhead"><h2>Vision Indicators</h2><span class="tag">EXPERIMENTAL</span></div><div class="cuegrid"><div class="cue"><span>Eye Aspect Ratio</span><b id="ear">—</b></div><div class="cue"><span>Mouth Aspect Ratio</span><b id="mar">—</b></div><div class="cue"><span>Head Offset</span><b id="head">—</b></div></div><div class="footnote" id="calibration">Calibration begins when a face is visible. Look straight at the camera for 3 seconds.</div></article>
<article class="card span-12" id="about"><div class="cardhead"><h2>Prototype Notes</h2><span class="tag">RESEARCH DEMO</span></div><div class="subtle" style="line-height:1.8">DISTRACTED is estimated from sustained head turn; DROWSY uses prolonged eye closure or a possible yawn; CRITICAL combines sustained cues. These thresholds are illustrative and not clinically or safety validated. This dashboard is not a substitute for a validated driver-safety system. Camera frames are processed locally and are not saved by this script.</div></article>
</section></main></div>
<script>
const $=id=>document.getElementById(id);function clock(){ $('clock').textContent=new Date().toLocaleTimeString('en-GB'); }setInterval(clock,1000);clock();
const colors={NORMAL:'#51d6a2',DISTRACTED:'#ffbd62',DROWSY:'#ff9b52',CRITICAL:'#ff6879',UNKNOWN:'#a5acb7',STARTING:'#6ea8ff'};
function render(d){const state=d.state||'UNKNOWN',color=colors[state]||colors.UNKNOWN;$('state').textContent=state;$('state').style.color=color;$('stateIcon').textContent=state==='NORMAL'?'✓':state==='UNKNOWN'?'?':'!';$('stateIcon').style.color=color;$('stateIcon').style.background=color+'20';$('stateDesc').textContent=state==='NORMAL'?'No sustained warning cues detected':state==='UNKNOWN'?'Face not visible or camera unavailable':state==='STARTING'?'Initializing webcam analysis':(d.alerts||[]).join(' · ');$('risk').textContent=d.risk??0;$('risk').style.color=color;$('riskBar').style.width=(d.risk??0)+'%';$('riskBar').style.background=color;$('riskLabel').textContent=state==='NORMAL'?'Low demo risk estimate':state==='UNKNOWN'?'Risk unavailable':state==='STARTING'?'Waiting for camera data':'Warning cues detected';$('alertCount').textContent=(d.alerts||[]).filter(a=>!a.toLowerCase().includes('no sustained')).length;$('cameraTag').textContent=(d.camera||'Starting').toUpperCase();$('camHealth').textContent=d.camera||'Starting';$('camDot').className='dot '+(d.camera==='Connected'?'ok':d.camera==='Disconnected'||d.camera==='Error'?'off':'');$('ear').textContent=d.ear??'—';$('mar').textContent=d.mar??'—';$('head').textContent=d.head_offset??'—';$('updated').textContent=d.updated?'Updated '+d.updated:'Awaiting frame';$('calibration').textContent=d.calibration?'Look straight at the webcam while calibration completes.':(d.camera==='Connected'?'Forward reference calibrated. Press C in camera window is not available in browser mode.':'Calibration will start when camera and face are detected.');const events=d.events||[];$('events').innerHTML=events.length?events.map(e=>`<div class="alert"><div class="alert-mark">!</div><div style="flex:1"><b>${e.state} · ${e.detail}</b><small>Camera vision cue · Risk estimate ${e.risk}%</small></div><span class="subtle">${e.time}</span></div>`).join(''):'<div class="empty">No warning events recorded yet. Monitoring is active.</div>';}
async function poll(){try{const r=await fetch('/api/status',{cache:'no-store'});render(await r.json())}catch(e){$('state').textContent='OFFLINE';$('stateDesc').textContent='Dashboard server is not responding'}}poll();setInterval(poll,800);
</script></body></html>'''


@app.get("/")
def home():
    return render_template_string(PAGE)


@app.get("/api/status")
def status():
    with lock:
        data = {key: value for key, value in runtime.items() if key not in ("jpeg", "stop")}
        data["events"] = list(runtime["events"])
    return jsonify(data)


@app.get("/video_feed")
def video_feed():
    def generate():
        while True:
            with lock:
                frame = runtime["jpeg"]
                stopping = runtime["stop"]
            if stopping:
                break
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.04)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-cache"})


def main():
    parser = argparse.ArgumentParser(description="SmartCap single-file web dashboard")
    parser.add_argument("--model", type=Path, default=None, help="Path to face_landmarker.task")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index; default 0")
    parser.add_argument("--host", default="127.0.0.1", help="Use 127.0.0.1 for local-only access")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    candidates = (
        args.model,
        Path(__file__).resolve().parent / "models" / "face_landmarker.task",
        Path.cwd() / "edge_ai" / "vision" / "models" / "face_landmarker.task",
    )
    model_path = next((p for p in candidates if p is not None and p.exists()), None)
    if model_path is None:
        parser.error("Model not found. Use --model with the full path to edge_ai/vision/models/face_landmarker.task")
    worker = threading.Thread(target=camera_worker, args=(model_path, args.camera), daemon=True)
    worker.start()
    print("SmartCap dashboard: http://127.0.0.1:5000")
    print("Keep this terminal open while the dashboard is running. Press Ctrl+C to stop.")
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
    finally:
        with lock:
            runtime["stop"] = True
        worker.join(timeout=2)


if __name__ == "__main__":
    main()
