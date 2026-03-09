import cv2
import json
import threading
import time
import asyncio
from datetime import datetime
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

# --------------------------------
# FASTAPI SETUP
# --------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/alerts", StaticFiles(directory="alerts"), name="alerts")

# --------------------------------
# WEBSOCKET CLIENTS
# --------------------------------

clients = []

@app.websocket("/ws/alerts")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)

    try:
        while True:
            await ws.receive_text()
    except:
        clients.remove(ws)

async def broadcast(alert):
    for client in clients:
        try:
            await client.send_json(alert)
        except:
            pass

# --------------------------------
# YOLO MODEL
# --------------------------------

print("Loading YOLO model...")

model = YOLO("yolov8m.pt")

print("Model loaded")

OBJECT_MAP = {
    "person": "person",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "vehicle",
    "dog": "animal",
    "cat": "animal",
    "horse": "animal",
    "cow": "animal"
}

# --------------------------------
# ALERT DIRECTORY
# --------------------------------

os.makedirs("alerts", exist_ok=True)

# --------------------------------
# LOAD CAMERAS
# --------------------------------

with open("cameras.json") as f:
    cameras = json.load(f)["cameras"]

print("Loaded cameras:", len(cameras))

last_object_alert = {}
ALERT_COOLDOWN = 120

# --------------------------------
# CAMERA PROCESSING
# --------------------------------

def process_camera(camera):

    name = camera["name"]
    url = camera["rtsp"]

    print(f"Connecting camera: {name}")

    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("❌ Cannot open stream:", name)
        return

    print("✅ Camera started:", name)

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("⚠️ Frame lost, reconnecting:", name)
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(url)
            continue

        frame_count += 1

        # Skip frames for performance
        if frame_count % 5 != 0:
            continue

        frame = cv2.resize(frame, (960, 540))

        # YOLO detection
        results = model(frame, imgsz=1280, conf=0.20)[0]

        detected_objects = []

        for box in results.boxes:

            cls = model.names[int(box.cls)]
            conf = float(box.conf)

            if cls not in OBJECT_MAP:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detected_objects.append((cls, conf))

            # Draw bounding box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            label = f"{cls} {conf:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

        # --------------------------------
        # ALERT LOGIC
        # --------------------------------

        for obj, conf in detected_objects:

            key = f"{name}_{obj}"
            now = time.time()

            if key in last_object_alert:
                if now - last_object_alert[key] < ALERT_COOLDOWN:
                    continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alerts/{name}_{obj}_{ts}.jpg"

            cv2.imwrite(filename, frame)

            alert = {
                "camera": name,
                "object": obj,
                "type": OBJECT_MAP[obj],
                "confidence": round(conf,2),
                "snapshot": filename,
                "time": datetime.now().isoformat()
            }

            print("🚨 ALERT:", alert)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(broadcast(alert))
            loop.close()

            last_object_alert[key] = now

        # debug snapshot every 100 frames
        if frame_count % 100 == 0:
            cv2.imwrite(f"alerts/debug_{name}.jpg", frame)

# --------------------------------
# START CAMERAS
# --------------------------------

def start_cameras():

    for cam in cameras:

        thread = threading.Thread(
            target=process_camera,
            args=(cam,),
            daemon=True
        )

        thread.start()

# --------------------------------
# APP STARTUP
# --------------------------------

@app.on_event("startup")
def startup():

    print("Starting AI engine...")

    start_cameras()