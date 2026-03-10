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

print("Loading YOLOv8m model...")

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

# --------------------------------
# TRACK MEMORY (per camera)
# --------------------------------

alerted_tracks = {}

# --------------------------------
# CAMERA PROCESSING
# --------------------------------

def process_camera(camera):

    name = camera["name"]
    url = camera["rtsp"]

    alerted_tracks[name] = set()

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

        # skip frames for performance
        if frame_count % 3 != 0:
            continue

        # --------------------------------
        # YOLO + BYTE TRACK
        # --------------------------------

        results = model.track(
            frame,
            persist=True,
            imgsz=640,
            conf=0.25,
            tracker="bytetrack.yaml"
        )[0]

        if results.boxes is None:
            continue

        for box in results.boxes:

            cls = model.names[int(box.cls)]

            if cls not in OBJECT_MAP:
                continue

            track_id = int(box.id) if box.id is not None else None

            if track_id is None:
                continue

            key = f"{name}_{track_id}"

            # ignore if already captured
            if key in alerted_tracks[name]:
                continue

            alerted_tracks[name].add(key)

            conf = float(box.conf)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{cls} ID:{track_id}"

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(
                frame,
                label,
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"alerts/{name}_{cls}_ID{track_id}_{ts}.jpg"

            cv2.imwrite(filename, frame)

            alert = {
                "camera": name,
                "object": cls,
                "type": OBJECT_MAP[cls],
                "track_id": track_id,
                "confidence": round(conf,2),
                "snapshot": filename,
                "time": datetime.now().isoformat()
            }

            print("🚨 ALERT:", alert)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(broadcast(alert))
            loop.close()

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