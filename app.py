import cv2
import json
import threading
import time
import asyncio
import os
from datetime import datetime
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ===============================
# CONFIG
# ===============================

FRAME_SKIP = 1
TRACK_EXPIRY = 120
CONFIDENCE = 0.25
IMG_SIZE = 640

CELL_SIZE = 120
LOCATION_EXPIRY = 20

# ===============================
# FASTAPI SETUP
# ===============================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/alerts", StaticFiles(directory="alerts"), name="alerts")

# ===============================
# WEBSOCKET CLIENTS
# ===============================

clients = []

@app.websocket("/ws/alerts")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)

    try:
        while True:
            await ws.receive_text()
    except:
        if ws in clients:
            clients.remove(ws)

async def broadcast(alert):

    dead = []

    for client in clients:

        try:
            await client.send_json(alert)

        except:
            dead.append(client)

    for d in dead:
        if d in clients:
            clients.remove(d)

# ===============================
# LOAD MODEL
# ===============================

print("Loading YOLOv8m...")
model = YOLO("yolov8m.pt")
print("Model loaded")

# ===============================
# OBJECT MAP
# ===============================

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

# ===============================
# STORAGE
# ===============================

os.makedirs("alerts", exist_ok=True)

track_memory = {}
location_memory = {}

# ===============================
# LOAD CAMERAS
# ===============================

with open("cameras.json") as f:
    cameras = json.load(f)["cameras"]

print("Loaded cameras:", len(cameras))

# ===============================
# CAMERA PROCESSING
# ===============================

def process_camera(camera):

    name = camera["name"]
    url = camera["rtsp"]

    print("Connecting camera:", name)

    track_memory[name] = {}
    location_memory[name] = {}

    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Stream failed:", name)
        return

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Reconnecting:", name)

            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(url)

            continue

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue

        results = model.track(
            frame,
            persist=True,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE,
            tracker="botsort.yaml"
        )[0]

        if results.boxes is None:
            continue

        now = time.time()

        # ===============================
        # CLEAN TRACK MEMORY
        # ===============================

        expired_tracks = []

        for tid, ts in track_memory[name].items():

            if now - ts > TRACK_EXPIRY:
                expired_tracks.append(tid)

        for tid in expired_tracks:
            del track_memory[name][tid]

        # ===============================
        # CLEAN LOCATION MEMORY
        # ===============================

        expired_cells = []

        for cell, ts in location_memory[name].items():

            if now - ts > LOCATION_EXPIRY:
                expired_cells.append(cell)

        for cell in expired_cells:
            del location_memory[name][cell]

        # ===============================
        # PROCESS DETECTIONS
        # ===============================

        for box in results.boxes:

            cls = model.names[int(box.cls)]

            if cls not in OBJECT_MAP:
                continue

            track_id = int(box.id) if box.id is not None else None

            if track_id is None:
                continue

            conf = float(box.conf)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cell = (cx // CELL_SIZE, cy // CELL_SIZE)

            # ===============================
            # LOCATION DEDUPLICATION
            # ===============================

            if cell in location_memory[name]:

                location_memory[name][cell] = now
                continue

            location_memory[name][cell] = now

            # ===============================
            # TRACK MEMORY
            # ===============================

            if track_id in track_memory[name]:

                track_memory[name][track_id] = now
                continue

            track_memory[name][track_id] = now

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

            print("ALERT:", alert)

            asyncio.run(broadcast(alert))

# ===============================
# START CAMERAS
# ===============================

def start_cameras():

    for cam in cameras:

        thread = threading.Thread(
            target=process_camera,
            args=(cam,),
            daemon=True
        )

        thread.start()

# ===============================
# APP STARTUP
# ===============================

@app.on_event("startup")
def startup():

    print("Starting Flynet AI engine...")

    start_cameras()