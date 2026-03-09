import cv2
import threading
import time
import json
import os
from datetime import datetime
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

alerts = []
last_alert_time = {}

ALERT_COOLDOWN = 10  # seconds


def load_cameras():
    with open("cameras.json") as f:
        return json.load(f)["cameras"]


def save_snapshot(frame, camera_name):

    if not os.path.exists("alerts"):
        os.makedirs("alerts")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"alerts/{camera_name}_{timestamp}.jpg"

    cv2.imwrite(filename, frame)

    return filename


def process_camera(camera):

    stream = camera["rtsp"]
    name = camera["name"]

    print(f"Starting camera worker: {name}")

    cap = cv2.VideoCapture(stream)

    frame_count = 0
    skip_frames = 10

    while True:

        ret, frame = cap.read()

        if not ret:
            print(f"{name}: reconnecting stream...")
            time.sleep(2)
            cap = cv2.VideoCapture(stream)
            continue

        frame_count += 1

        if frame_count % skip_frames != 0:
            continue

        results = model(frame)

        for r in results:
            for box in r.boxes:

                cls = int(box.cls)
                conf = float(box.conf)
                label = model.names[cls]

                if conf < 0.6:
                    continue

                current_time = time.time()

                key = f"{name}_{label}"

                if key in last_alert_time:
                    if current_time - last_alert_time[key] < ALERT_COOLDOWN:
                        continue

                last_alert_time[key] = current_time

                snapshot = save_snapshot(frame, name)

                alert = {
                    "camera": name,
                    "object": label,
                    "confidence": round(conf, 2),
                    "snapshot": snapshot,
                    "time": datetime.now().isoformat()
                }

                alerts.append(alert)

                print("ALERT:", alert)

        time.sleep(2)


def start_workers():

    cameras = load_cameras()

    for camera in cameras:

        thread = threading.Thread(
            target=process_camera,
            args=(camera,),
            daemon=True
        )

        thread.start()