# =============================================================================
#  Flynet AI Engine  —  app.py                                    v2.4.0
#  Stack : FastAPI + YOLOv8 + InsightFace + EasyOCR + OpenCV
# =============================================================================
#
#  Install:
#    pip install ultralytics fastapi "uvicorn[standard]" opencv-python easyocr
#    pip install insightface onnxruntime numpy python-multipart
#
#  Run:
#    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
#
# ─────────────────────────────────────────────────────────────────────────────
#  Watchlist REST API  — uses only POST + GET (no PATCH/DELETE browser issues)
# ─────────────────────────────────────────────────────────────────────────────
#  POST  /watchlist/add              add new entry (multipart/form-data)
#  GET   /watchlist/list             get all entries
#  POST  /watchlist/delete           delete entry  { "id": "..." }
#  POST  /watchlist/toggle           toggle status { "id": "..." }
#
# ─────────────────────────────────────────────────────────────────────────────
#  Detection REST API
# ─────────────────────────────────────────────────────────────────────────────
#  GET  /health
#  GET  /stats
#  GET  /stats/{camera_name}
#  GET  /people-count
#  GET  /report/daily?date=YYYY-MM-DD
#  WS   /ws/alerts
#  GET  /alerts/<file>
#
# =============================================================================

import cv2
import json
import threading
import time
import asyncio
import os
import collections
import uuid
import numpy as np
from datetime import datetime, date
from contextlib import asynccontextmanager
from typing import Optional

import easyocr
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, Query, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sqlite3

try:
    import insightface
    from insightface.app import FaceAnalysis
    FACE_AVAILABLE = True
except ImportError:
    FACE_AVAILABLE = False
    print("[WARN] insightface not installed — face recognition disabled.")
    print("       Fix: pip install insightface onnxruntime")

# =============================================================================
# ░░  CONFIG
# =============================================================================

FRAME_SKIP      = 2
CONFIDENCE      = 0.40
IMG_SIZE        = 640
ALERT_COOLDOWN  = 15
TRACK_EXPIRY    = 120
PLATE_MIN_LEN   = 4
PLATE_MAX_LEN   = 10
PLATE_OCR_CONF  = 0.40
FACE_SIMILARITY = 0.40
FACE_RELOAD_SEC = 30
WATCHLIST_DIR   = "watchlist"
WATCHLIST_DB    = "watchlist_db.json"
LP_MODEL_PATH   = "license_plate_detector.pt"
ALERTS_DB       = "alerts.db"

# =============================================================================
# ░░  WATCHLIST DATABASE
# =============================================================================

watchlist_entries: list = []
watchlist_lock          = threading.RLock()
face_embeddings: dict   = {}


def _wl_load():
    global watchlist_entries
    if os.path.exists(WATCHLIST_DB):
        try:
            with open(WATCHLIST_DB) as f:
                data = json.load(f)
            for e in data:
                e["id"] = str(e["id"])
            watchlist_entries = data
            print(f"[Watchlist] Loaded {len(watchlist_entries)} entries.")
        except Exception as ex:
            print(f"[Watchlist] Load error: {ex} — starting empty.")
            watchlist_entries = []
    else:
        watchlist_entries = []


def _wl_save():
    try:
        with open(WATCHLIST_DB, "w") as f:
            json.dump(watchlist_entries, f, indent=2)
    except Exception as ex:
        print(f"[Watchlist] Save error: {ex}")


def _rebuild_face_embeddings():
    if face_app is None:
        return
    new_emb = {}
    with watchlist_lock:
        active_persons = [
            e for e in watchlist_entries
            if e["type"] == "person" and e.get("status", True)
        ]
    for entry in active_persons:
        img_path = entry.get("image_file")
        if not img_path or not os.path.exists(img_path):
            continue
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = face_app.get(img)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                new_emb[str(entry["id"])] = {
                    "name":      entry["name"],
                    "embedding": face.normed_embedding,
                }
        except Exception as ex:
            print(f"[Watchlist] Embedding error '{entry['name']}': {ex}")
    with watchlist_lock:
        face_embeddings.clear()
        face_embeddings.update(new_emb)
    print(f"[Watchlist] {len(face_embeddings)} face embedding(s) active.")


def _get_active_plates() -> set:
    with watchlist_lock:
        return {
            "".join(c for c in e["plate"].upper() if c.isalnum())
            for e in watchlist_entries
            if e["type"] == "vehicle"
            and e.get("status", True)
            and e.get("plate", "").strip()
        }


def _periodic_reload():
    while True:
        time.sleep(FACE_RELOAD_SEC)
        _rebuild_face_embeddings()

# =============================================================================
# ░░  ALERTS DATABASE  (SQLite — persists detections across restarts)
# =============================================================================

def _db_init():
    """Create alerts table if it doesn't exist."""
    conn = sqlite3.connect(ALERTS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            event       TEXT,
            camera      TEXT,
            object      TEXT,
            type        TEXT,
            track_id    INTEGER,
            confidence  REAL,
            plate       TEXT,
            face        TEXT,
            watchlist   INTEGER DEFAULT 0,
            snapshot    TEXT,
            time        TEXT
        )
    """)
    conn.commit()
    conn.close()


def _db_save_alert(alert: dict):
    """Save a detection alert to SQLite. Non-blocking — errors are logged only."""
    try:
        conn = sqlite3.connect(ALERTS_DB)
        conn.execute("""
            INSERT INTO alerts
                (event, camera, object, type, track_id, confidence,
                 plate, face, watchlist, snapshot, time)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            alert.get("event"),
            alert.get("camera"),
            alert.get("object"),
            alert.get("type"),
            alert.get("track_id"),
            alert.get("confidence"),
            alert.get("plate"),
            alert.get("face"),
            1 if alert.get("watchlist") else 0,
            alert.get("snapshot"),
            alert.get("time"),
        ))
        conn.commit()
        conn.close()
    except Exception as ex:
        print(f"[DB] Save error: {ex}")


def _db_fetch_alerts(limit: int = 1000, offset: int = 0,
                     camera: str = None, type_: str = None) -> list:
    """Fetch stored alerts, newest first."""
    try:
        conn = sqlite3.connect(ALERTS_DB)
        conn.row_factory = sqlite3.Row
        where_clauses = []
        params = []
        if camera:
            where_clauses.append("camera = ?")
            params.append(camera)
        if type_:
            where_clauses.append("type = ?")
            params.append(type_)
        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        params += [limit, offset]
        rows = conn.execute(
            f"SELECT * FROM alerts {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            params
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as ex:
        print(f"[DB] Fetch error: {ex}")
        return []


def _db_count_alerts() -> dict:
    """Return total counts per type for the stats endpoint."""
    try:
        conn = sqlite3.connect(ALERTS_DB)
        rows = conn.execute(
            "SELECT type, COUNT(*) as cnt FROM alerts GROUP BY type"
        ).fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows}
    except Exception as ex:
        print(f"[DB] Count error: {ex}")
        return {}



# =============================================================================
# ░░  FASTAPI LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("  Flynet AI Engine  v2.4.0")
    print("=" * 60)
    _db_init()
    _wl_load()
    _rebuild_face_embeddings()
    threading.Thread(target=_periodic_reload, daemon=True).start()
    start_cameras()
    yield
    print("Flynet AI Engine stopped.")


app = FastAPI(title="Flynet AI Engine", version="2.4.0", lifespan=lifespan)

# ── CORS: credentials=False is required when allow_origins=["*"] ─────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("alerts",      exist_ok=True)
os.makedirs(WATCHLIST_DIR, exist_ok=True)

app.mount("/alerts",    StaticFiles(directory="alerts"),      name="alerts")
app.mount("/watchlist-images", StaticFiles(directory=WATCHLIST_DIR), name="watchlist_images")

# =============================================================================
# ░░  WEBSOCKET
# =============================================================================

ws_clients: list = []
_event_loop      = None


@app.websocket("/ws/alerts")
async def ws_endpoint(ws: WebSocket):
    global _event_loop
    _event_loop = asyncio.get_event_loop()
    await ws.accept()
    ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except Exception:
        if ws in ws_clients:
            ws_clients.remove(ws)


async def _do_broadcast(payload: dict):
    dead = []
    for client in ws_clients:
        try:
            await client.send_json(payload)
        except Exception:
            dead.append(client)
    for d in dead:
        if d in ws_clients:
            ws_clients.remove(d)


def broadcast(payload: dict):
    # Persist detection alerts to SQLite (not people_count or watchlist_alert events)
    if payload.get("event") == "detection":
        threading.Thread(target=_db_save_alert, args=(payload,), daemon=True).start()
    if _event_loop and _event_loop.is_running():
        asyncio.run_coroutine_threadsafe(_do_broadcast(payload), _event_loop)

# =============================================================================
# ░░  MODEL LOADING
# =============================================================================

print("Loading YOLOv8m …")
yolo = YOLO("yolov8m.pt")

print("Loading EasyOCR …")
ocr = easyocr.Reader(["en"], gpu=False)

lp_yolo = None
if os.path.exists(LP_MODEL_PATH):
    print(f"Loading LP detector → {LP_MODEL_PATH}")
    lp_yolo = YOLO(LP_MODEL_PATH)
else:
    print(f"[INFO] No LP model at '{LP_MODEL_PATH}' — using full-crop OCR fallback.")

face_app = None
if FACE_AVAILABLE:
    try:
        print("Loading InsightFace …")
        face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=0, det_size=(320, 320))
        print("InsightFace loaded.")
    except Exception as ex:
        print(f"[WARN] InsightFace init failed: {ex}")

print("\nAll models ready.\n")

# =============================================================================
# ░░  OBJECT MAP
# =============================================================================

OBJECT_MAP = {
    "person":     "person",
    "car":        "vehicle",
    "truck":      "vehicle",
    "bus":        "vehicle",
    "motorcycle": "vehicle",
    "bicycle":    "vehicle",
    "dog":        "animal",
    "cat":        "animal",
    "horse":      "animal",
    "cow":        "animal",
    "sheep":      "animal",
    "bird":       "animal",
}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

# =============================================================================
# ░░  CAMERAS
# =============================================================================

with open("cameras.json") as f:
    cameras: list = json.load(f)["cameras"]

print(f"Loaded {len(cameras)} cameras.\n")

# =============================================================================
# ░░  ANALYTICS
# =============================================================================

global_stats = {
    "persons": 0, "vehicles": 0, "animals": 0,
    "plates":  0, "faces":    0, "watchlist": 0, "total": 0,
}
camera_stats: dict = {
    cam["name"]: {
        "persons": 0, "vehicles": 0, "animals": 0,
        "plates":  0, "faces":    0, "watchlist": 0, "total": 0,
    }
    for cam in cameras
}
hourly_buckets: dict = collections.defaultdict(
    lambda: collections.defaultdict(lambda: collections.defaultdict(int))
)
people_count: dict = {cam["name"]: {"in": 0, "out": 0} for cam in cameras}
analytics_lock = threading.Lock()


def _record(cam, obj_type, has_plate, is_watchlist):
    with analytics_lock:
        key = obj_type + "s"
        global_stats[key]      = global_stats.get(key, 0) + 1
        global_stats["total"] += 1
        if has_plate:    global_stats["plates"]    += 1
        if is_watchlist: global_stats["watchlist"] += 1
        cs = camera_stats.setdefault(cam, {
            "persons":0,"vehicles":0,"animals":0,
            "plates":0,"faces":0,"watchlist":0,"total":0
        })
        cs[key] = cs.get(key, 0) + 1
        cs["total"] += 1
        if has_plate:    cs["plates"]    = cs.get("plates",    0) + 1
        if is_watchlist: cs["watchlist"] = cs.get("watchlist", 0) + 1
        now = datetime.now()
        d, h = now.strftime("%Y-%m-%d"), now.strftime("%H")
        hourly_buckets[d][h][obj_type] += 1
        if is_watchlist:
            hourly_buckets[d][h]["watchlist"] += 1


def _record_face(cam):
    with analytics_lock:
        global_stats["faces"] = global_stats.get("faces", 0) + 1
        camera_stats.get(cam, {})["faces"] = camera_stats.get(cam, {}).get("faces", 0) + 1


def _record_crossing(cam, direction):
    with analytics_lock:
        people_count.setdefault(cam, {"in": 0, "out": 0})
        people_count[cam][direction] += 1

# =============================================================================
# ░░  WATCHLIST REST API
# ─────────────────────────────────────────────────────────────────────────────
#  All mutating actions use POST only — zero browser method issues.
#  /watchlist/add     POST multipart  → add entry
#  /watchlist/list    GET             → list entries
#  /watchlist/delete  POST JSON       → delete entry by id
#  /watchlist/toggle  POST JSON       → toggle status by id
# =============================================================================

def _entry_with_url(entry: dict) -> dict:
    out = dict(entry)
    img = entry.get("image_file")
    out["image_url"] = ("/watchlist-images/" + os.path.basename(img)) if img else None
    return out


class WatchlistIdBody(BaseModel):
    id: str


# ── ADD ───────────────────────────────────────────────────────────────────────
@app.post("/watchlist/add")
async def api_watchlist_add(
    type:  str                  = Form(...),
    name:  str                  = Form(...),
    plate: str                  = Form(""),
    notes: str                  = Form(""),
    image: Optional[UploadFile] = File(None),
):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name is required.")

    entry_id   = str(uuid.uuid4())
    image_file = None

    if image and image.filename:
        ext        = os.path.splitext(image.filename)[1].lower() or ".jpg"
        image_file = os.path.join(WATCHLIST_DIR, f"{entry_id}{ext}")
        contents   = await image.read()
        with open(image_file, "wb") as f:
            f.write(contents)

    entry = {
        "id":         entry_id,
        "type":       type,
        "name":       name.strip(),
        "plate":      plate.strip().upper() if type == "vehicle" else "",
        "notes":      notes.strip(),
        "image_file": image_file,
        "status":     True,
        "created_at": datetime.now().isoformat(),
    }

    with watchlist_lock:
        watchlist_entries.append(entry)
        _wl_save()

    threading.Thread(target=_rebuild_face_embeddings, daemon=True).start()
    print(f"[Watchlist] Added: {entry['name']} ({entry['type']})")
    return {"success": True, "entry": _entry_with_url(entry)}


# ── LIST ──────────────────────────────────────────────────────────────────────
@app.get("/watchlist/list")
def api_watchlist_list():
    with watchlist_lock:
        entries = list(watchlist_entries)
    return {"success": True, "entries": [_entry_with_url(e) for e in entries]}


# ── DELETE ────────────────────────────────────────────────────────────────────
@app.post("/watchlist/delete")
def api_watchlist_delete(body: WatchlistIdBody):
    entry_id = body.id.strip()

    with watchlist_lock:
        entry = next((e for e in watchlist_entries if e["id"] == entry_id), None)

        if not entry:
            raise HTTPException(
                status_code=404,
                detail=f"Entry '{entry_id}' not found. Total entries: {len(watchlist_entries)}"
            )

        # Remove image file (non-fatal)
        img = entry.get("image_file")
        if img:
            try:
                if os.path.exists(img):
                    os.remove(img)
            except Exception as ex:
                print(f"[Watchlist] Image removal warning: {ex}")

        watchlist_entries[:] = [e for e in watchlist_entries if e["id"] != entry_id]
        face_embeddings.pop(entry_id, None)
        _wl_save()

    print(f"[Watchlist] Deleted: {entry['name']} (id={entry_id})")
    return {"success": True, "deleted": entry_id, "name": entry["name"]}


# ── TOGGLE STATUS ─────────────────────────────────────────────────────────────
@app.post("/watchlist/toggle")
def api_watchlist_toggle(body: WatchlistIdBody):
    entry_id = body.id.strip()

    with watchlist_lock:
        entry = next((e for e in watchlist_entries if e["id"] == entry_id), None)

        if not entry:
            raise HTTPException(
                status_code=404,
                detail=f"Entry '{entry_id}' not found. Total entries: {len(watchlist_entries)}"
            )

        entry["status"] = not entry["status"]
        new_status = entry["status"]
        _wl_save()

    threading.Thread(target=_rebuild_face_embeddings, daemon=True).start()
    print(f"[Watchlist] Toggled: {entry['name']} → {'Active' if new_status else 'Inactive'}")
    return {"success": True, "id": entry_id, "status": new_status}


# =============================================================================
# ░░  ALERTS HISTORY REST API
# =============================================================================
#  GET  /alerts/history                 → latest 1000 alerts (newest first)
#  GET  /alerts/history?limit=N         → N alerts
#  GET  /alerts/history?camera=X        → filter by camera
#  GET  /alerts/history?type=person     → filter by type
#  GET  /alerts/history?offset=N        → pagination

@app.get("/alerts/history")
def api_alerts_history(
    limit:  int            = Query(default=1000, ge=1,  le=5000),
    offset: int            = Query(default=0,    ge=0),
    camera: Optional[str]  = Query(default=None),
    type_:  Optional[str]  = Query(default=None, alias="type"),
):
    """Return stored detection alerts from SQLite, newest first."""
    alerts = _db_fetch_alerts(limit=limit, offset=offset, camera=camera, type_=type_)
    return {"total": len(alerts), "alerts": alerts}


@app.get("/alerts/history/counts")
def api_alerts_counts():
    """Return total detection counts per type from the database."""
    return _db_count_alerts()

# =============================================================================
# ░░  DETECTION REST API
# =============================================================================

@app.get("/health")
def api_health():
    return {
        "status":             "ok",
        "cameras":            len(cameras),
        "face_recognition":   face_app is not None,
        "lp_model":           lp_yolo  is not None,
        "watchlist_persons":  sum(1 for e in watchlist_entries if e["type"] == "person"  and e.get("status")),
        "watchlist_vehicles": sum(1 for e in watchlist_entries if e["type"] == "vehicle" and e.get("status")),
        "watchlist_total":    len(watchlist_entries),
    }


@app.get("/stats")
def api_stats_global():
    with analytics_lock:
        return dict(global_stats)


@app.get("/stats/{camera_name}")
def api_stats_camera(camera_name: str):
    with analytics_lock:
        s = camera_stats.get(camera_name)
    if s is None:
        return {"error": f"Camera '{camera_name}' not found"}
    return {"camera": camera_name, **s}


@app.get("/people-count")
def api_people_count():
    with analytics_lock:
        return dict(people_count)


@app.get("/report/daily")
def api_daily_report(date_str: Optional[str] = Query(default=None, alias="date")):
    if date_str is None:
        date_str = date.today().isoformat()
    with analytics_lock:
        day_data = dict(hourly_buckets.get(date_str, {}))
    types     = ["person", "vehicle", "animal", "watchlist"]
    hours_out = {}
    summary   = {t: 0 for t in types}
    for h in range(24):
        hk            = f"{h:02d}"
        row           = {t: dict(day_data.get(hk, {})).get(t, 0) for t in types}
        hours_out[hk] = row
        for t in types:
            summary[t] += row[t]
    return {"date": date_str, "summary": summary, "hours": hours_out}

# =============================================================================
# ░░  PER-CAMERA STATE
# =============================================================================

track_memory:   dict = {}
alert_cooldown: dict = {}
plate_memory:   dict = {}
face_memory:    dict = {}
line_side:      dict = {}

# =============================================================================
# ░░  DETECTION HELPERS
# =============================================================================

def save_snapshot(frame: np.ndarray, tag: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    fn = f"alerts/{tag}_{ts}.jpg"
    cv2.imwrite(fn, frame)
    return fn


def recognize_plate(vehicle_crop: np.ndarray) -> Optional[str]:
    try:
        roi = vehicle_crop
        if lp_yolo is not None:
            res = lp_yolo(vehicle_crop, imgsz=320, conf=0.40)[0]
            if res.boxes is not None and len(res.boxes):
                px1, py1, px2, py2 = map(int, res.boxes.xyxy[0])
                c = vehicle_crop[py1:py2, px1:px2]
                if c.size > 0:
                    roi = c
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for r in ocr.readtext(gray):
            text, conf = r[1], r[2]
            if conf >= PLATE_OCR_CONF:
                clean = "".join(c for c in text.upper() if c.isalnum())
                if PLATE_MIN_LEN <= len(clean) <= PLATE_MAX_LEN:
                    return clean
    except Exception as ex:
        print(f"[LPR error] {ex}")
    return None


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(1.0 - np.dot(a, b))


def run_face_recognition(frame: np.ndarray, x1, y1, x2, y2) -> tuple:
    if face_app is None:
        return None, False
    try:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, False
        faces = face_app.get(crop)
        if not faces:
            return None, False
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        emb  = face.normed_embedding
        best_name, best_dist = None, FACE_SIMILARITY
        with watchlist_lock:
            for eid, data in face_embeddings.items():
                dist = _cosine_dist(emb, data["embedding"])
                if dist < best_dist:
                    best_dist, best_name = dist, data["name"]
        return (best_name, True) if best_name else ("unknown", False)
    except Exception as ex:
        print(f"[Face error] {ex}")
        return None, False


def _line_side_val(lx1, ly1, lx2, ly2, px, py) -> int:
    v = (lx2-lx1)*(py-ly1) - (ly2-ly1)*(px-lx1)
    return 1 if v > 0 else (-1 if v < 0 else 0)


def check_count_line(cam, tid, cx, cy, count_line) -> Optional[str]:
    side      = _line_side_val(count_line["x1"], count_line["y1"],
                               count_line["x2"], count_line["y2"], cx, cy)
    cam_state = line_side.setdefault(cam, {})
    prev      = cam_state.get(tid, 0)
    if side != 0:
        cam_state[tid] = side
    if prev != 0 and side != 0 and side != prev:
        return "in" if side == count_line.get("in_side", 1) else "out"
    return None

# =============================================================================
# ░░  CAMERA THREAD
# =============================================================================

def process_camera(camera: dict):
    cam_name   = camera["name"]
    cam_url    = camera["rtsp"]
    count_line = camera.get("count_line")

    print(f"[{cam_name}] Connecting …")

    track_memory[cam_name]   = {}
    alert_cooldown[cam_name] = {}
    plate_memory[cam_name]   = {}
    face_memory[cam_name]    = {}
    line_side[cam_name]      = {}

    cap = cv2.VideoCapture(cam_url)
    if not cap.isOpened():
        print(f"[{cam_name}] Stream failed — retrying in 5 s …")
        time.sleep(5)
        cap = cv2.VideoCapture(cam_url)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{cam_name}] Lost stream — reconnecting in 3 s …")
            cap.release()
            time.sleep(3)
            cap = cv2.VideoCapture(cam_url)
            continue

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        if count_line:
            cv2.line(frame,
                     (count_line["x1"], count_line["y1"]),
                     (count_line["x2"], count_line["y2"]),
                     (0, 200, 255), 2)
            with analytics_lock:
                pc = people_count.get(cam_name, {"in": 0, "out": 0})
            cv2.putText(frame, f"IN:{pc['in']}  OUT:{pc['out']}",
                        (count_line["x1"]+6, count_line["y1"]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        results = yolo.track(
            frame, persist=True, imgsz=IMG_SIZE,
            conf=CONFIDENCE, tracker="botsort.yaml"
        )[0]

        if results.boxes is None:
            continue

        now = time.time()

        for tid in [k for k, v in list(track_memory[cam_name].items()) if now - v > TRACK_EXPIRY]:
            track_memory[cam_name].pop(tid, None)
            plate_memory[cam_name].pop(tid, None)
            face_memory[cam_name].pop(tid, None)
            line_side.get(cam_name, {}).pop(tid, None)

        active_plates = _get_active_plates()

        for box in results.boxes:
            cls      = yolo.names[int(box.cls)]
            obj_type = OBJECT_MAP.get(cls)
            if obj_type is None:
                continue

            track_id = int(box.id) if box.id is not None else None
            if track_id is None:
                continue

            conf            = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx              = (x1+x2)//2
            cy              = (y1+y2)//2

            track_memory[cam_name][track_id] = now

            last_alerted = alert_cooldown[cam_name].get(track_id, 0)
            can_alert    = (now - last_alerted) > ALERT_COOLDOWN

            plate_text, plate_in_wl = None, False
            if cls in VEHICLE_CLASSES:
                if track_id in plate_memory[cam_name]:
                    plate_text = plate_memory[cam_name][track_id]
                elif can_alert:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        plate_text = recognize_plate(crop)
                        if plate_text:
                            plate_memory[cam_name][track_id] = plate_text
                if plate_text and plate_text in active_plates:
                    plate_in_wl = True

            face_name, is_watchlist = None, False
            if obj_type == "person":
                if track_id in face_memory[cam_name]:
                    cached      = face_memory[cam_name][track_id]
                    face_name   = cached["name"]
                    is_watchlist = cached["watchlist"]
                elif can_alert:
                    face_name, is_watchlist = run_face_recognition(frame, x1, y1, x2, y2)
                    if face_name:
                        face_memory[cam_name][track_id] = {"name": face_name, "watchlist": is_watchlist}
                        _record_face(cam_name)

            if plate_in_wl:
                is_watchlist = True

            if obj_type == "person" and count_line:
                direction = check_count_line(cam_name, track_id, cx, cy, count_line)
                if direction:
                    _record_crossing(cam_name, direction)
                    with analytics_lock:
                        counts = dict(people_count[cam_name])
                    broadcast({
                        "event": "people_count", "camera": cam_name,
                        "direction": direction, "track_id": track_id,
                        "counts": counts, "time": datetime.now().isoformat(),
                    })

            color = (0, 0, 255) if is_watchlist else (0, 255, 0)
            label = f"{cls} ID:{track_id}"
            if plate_text:
                label += f"  [{plate_text}{'★' if plate_in_wl else ''}]"
            if face_name and face_name != "unknown":
                label += f"  ★{face_name}"
            elif face_name == "unknown":
                label += "  [face:?]"
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if not can_alert:
                continue

            alert_cooldown[cam_name][track_id] = now
            snap = save_snapshot(frame, f"{cam_name}_{cls}_ID{track_id}")
            alert_type = "watchlist" if is_watchlist else ("face_detected" if face_name == "unknown" else obj_type)

            alert = {
                "event": "detection", "camera": cam_name, "object": cls,
                "type": alert_type, "track_id": track_id,
                "confidence": round(conf, 2), "plate": plate_text,
                "face": face_name, "watchlist": is_watchlist,
                "snapshot": snap, "time": datetime.now().isoformat(),
            }

            print(
                f"{'[WATCHLIST]' if is_watchlist else '[ALERT]'} "
                f"{cam_name} | {cls} ID:{track_id} | conf:{conf:.2f}"
                + (f" | plate:{plate_text}" if plate_text else "")
                + (f" | face:{face_name}"   if face_name  else "")
            )

            broadcast(alert)
            _record(cam_name, obj_type, bool(plate_text), is_watchlist)

            if is_watchlist:
                wl_name = face_name if (face_name and face_name != "unknown") else plate_text
                broadcast({
                    "event": "watchlist_alert", "camera": cam_name,
                    "name": wl_name,
                    "type": "face" if obj_type == "person" else "vehicle",
                    "track_id": track_id, "plate": plate_text,
                    "snapshot": snap, "time": datetime.now().isoformat(),
                })

# =============================================================================
# ░░  START CAMERAS
# =============================================================================

def start_cameras():
    for cam in cameras:
        t = threading.Thread(target=process_camera, args=(cam,), daemon=True)
        t.start()
        print(f"  ▶  Thread started: {cam['name']}")
    print()