# =============================================================================
#  Flynet AI Engine  —  app.py                                    v2.5.0
#  Stack : FastAPI + YOLOv8 + InsightFace + EasyOCR + OpenCV
# =============================================================================
#
#  Install:
#    pip install ultralytics fastapi "uvicorn[standard]" opencv-python easyocr
#    pip install insightface onnxruntime numpy python-multipart reportlab
#
#  Run:
#    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
#
# ─────────────────────────────────────────────────────────────────────────────
#  Watchlist REST API
# ─────────────────────────────────────────────────────────────────────────────
#  POST  /watchlist/add
#  GET   /watchlist/list
#  POST  /watchlist/delete
#  POST  /watchlist/toggle
#
# ─────────────────────────────────────────────────────────────────────────────
#  Detection REST API
# ─────────────────────────────────────────────────────────────────────────────
#  GET  /health
#  GET  /stats
#  GET  /stats/{camera_name}
#  GET  /people-count
#  GET  /report/daily?date=YYYY-MM-DD
#  GET  /report/daily/csv?date=YYYY-MM-DD   ← NEW
#  GET  /report/daily/pdf?date=YYYY-MM-DD   ← NEW
#  GET  /alerts/history
#  GET  /alerts/history/counts
#  WS   /ws/alerts
#  GET  /alerts/<file>
#
# =============================================================================

import csv
import io
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
from fastapi.responses import StreamingResponse
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

# ── Motion detection config ────────────────────────────────────────────────────
MOTION_BLUR         = 21       # Gaussian blur kernel size (must be odd)
MOTION_THRESHOLD    = 3000     # Min contour area in px² to count as real motion
MOTION_COOLDOWN     = 30       # Seconds between motion alerts per camera
MOTION_OFF_START    = 22       # Off-hours start (10 PM) — 24h format
MOTION_OFF_END      = 6        # Off-hours end   (6 AM)  — 24h format

# ── Fire detection config ───────────────────────────────────────────────────────
FIRE_MODEL_PATH     = "fire_detector.pt"   # YOLOv8 fire model (optional)
FIRE_CONFIDENCE     = 0.45                 # Min confidence for fire/smoke detection
FIRE_COOLDOWN       = 20                   # Seconds between fire alerts per camera
FIRE_MIN_AREA       = 2000                 # Min contour area px² — above floor markings, catches early fire
FIRE_MIN_BRIGHTNESS = 180                  # Min mean pixel brightness — real fire glows bright
FIRE_FLICKER_FRAMES = 3                    # Consecutive frames with motion needed to confirm fire

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
# ░░  ALERTS DATABASE  (SQLite)
# =============================================================================

def _db_init():
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

def _db_hourly_report(date_str: str) -> dict:
    """
    Build hourly detection breakdown for a given date from SQLite.
    Returns: { "HH": { "person": N, "vehicle": N, "animal": N, "watchlist": N } }
    """
    try:
        conn = sqlite3.connect(ALERTS_DB)
        rows = conn.execute(
            """
            SELECT
                strftime('%H', time) as hour,
                type,
                COUNT(*) as cnt
            FROM alerts
            WHERE date(time) = ?
            GROUP BY hour, type
            """,
            (date_str,)
        ).fetchall()
        conn.close()

        result = {}
        for hour, type_, cnt in rows:
            if hour not in result:
                result[hour] = {}
            result[hour][type_] = cnt

        # Also get people counting from SQLite if stored
        return result
    except Exception as ex:
        print(f"[DB] Hourly report error: {ex}")
        return {}


def _db_people_count(date_str: str) -> dict:
    """Get people IN/OUT counts from SQLite for a given date."""
    try:
        conn = sqlite3.connect(ALERTS_DB)
        # people_count events are stored as type='people_count' — not in alerts table
        # Fall back to in-memory people_count
        conn.close()
    except Exception:
        pass
    return dict(people_count)


# =============================================================================
# ░░  FASTAPI LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("  Flynet AI Engine  v2.5.0")
    print("=" * 60)
    _db_init()
    _wl_load()
    _rebuild_face_embeddings()
    threading.Thread(target=_periodic_reload, daemon=True).start()
    start_cameras()
    yield
    print("Flynet AI Engine stopped.")


app = FastAPI(title="Flynet AI Engine", version="2.5.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("alerts",      exist_ok=True)
os.makedirs(WATCHLIST_DIR, exist_ok=True)

app.mount("/alerts",           StaticFiles(directory="alerts"),      name="alerts")
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

fire_yolo = None
if os.path.exists(FIRE_MODEL_PATH):
    try:
        print(f"Loading Fire detector → {FIRE_MODEL_PATH}")
        fire_yolo = YOLO(FIRE_MODEL_PATH)
        print("Fire model loaded.")
    except Exception as ex:
        print(f"[WARN] Fire model failed to load: {ex}")
else:
    print(f"[INFO] No fire model at '{FIRE_MODEL_PATH}' — using color-based fallback.")

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
# =============================================================================

def _entry_with_url(entry: dict) -> dict:
    out = dict(entry)
    img = entry.get("image_file")
    out["image_url"] = ("/watchlist-images/" + os.path.basename(img)) if img else None
    return out


class WatchlistIdBody(BaseModel):
    id: str


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


@app.get("/watchlist/list")
def api_watchlist_list():
    with watchlist_lock:
        entries = list(watchlist_entries)
    return {"success": True, "entries": [_entry_with_url(e) for e in entries]}


@app.post("/watchlist/delete")
def api_watchlist_delete(body: WatchlistIdBody):
    entry_id = body.id.strip()
    with watchlist_lock:
        entry = next((e for e in watchlist_entries if e["id"] == entry_id), None)
        if not entry:
            raise HTTPException(status_code=404,
                detail=f"Entry '{entry_id}' not found.")
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


@app.post("/watchlist/toggle")
def api_watchlist_toggle(body: WatchlistIdBody):
    entry_id = body.id.strip()
    with watchlist_lock:
        entry = next((e for e in watchlist_entries if e["id"] == entry_id), None)
        if not entry:
            raise HTTPException(status_code=404,
                detail=f"Entry '{entry_id}' not found.")
        entry["status"] = not entry["status"]
        new_status = entry["status"]
        _wl_save()
    threading.Thread(target=_rebuild_face_embeddings, daemon=True).start()
    print(f"[Watchlist] Toggled: {entry['name']} → {'Active' if new_status else 'Inactive'}")
    return {"success": True, "id": entry_id, "status": new_status}

# =============================================================================
# ░░  ALERTS HISTORY REST API
# =============================================================================

@app.get("/alerts/history")
def api_alerts_history(
    limit:  int           = Query(default=1000, ge=1, le=5000),
    offset: int           = Query(default=0,    ge=0),
    camera: Optional[str] = Query(default=None),
    type_:  Optional[str] = Query(default=None, alias="type"),
):
    alerts = _db_fetch_alerts(limit=limit, offset=offset, camera=camera, type_=type_)
    return {"total": len(alerts), "alerts": alerts}


@app.get("/alerts/history/counts")
def api_alerts_counts():
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
# ░░  REPORT EXPORT — CSV
# =============================================================================

@app.get("/report/daily/csv")
def api_daily_report_csv(date_str: Optional[str] = Query(default=None, alias="date")):
    """
    Download daily detection report as CSV.
    ?date=YYYY-MM-DD  (defaults to today)
    """
    if date_str is None:
        date_str = date.today().isoformat()

    # Read from SQLite (persists across restarts) — fall back to memory if empty
    db_data  = _db_hourly_report(date_str)
    with analytics_lock:
        mem_data = dict(hourly_buckets.get(date_str, {}))
    # Merge: prefer DB data, supplement with memory data
    day_data = mem_data.copy()
    for hk, type_counts in db_data.items():
        if hk not in day_data:
            day_data[hk] = {}
        for t, cnt in type_counts.items():
            day_data[hk][t] = max(day_data[hk].get(t, 0), cnt)

    types   = ["person", "vehicle", "animal", "watchlist"]
    rows    = []
    summary = {t: 0 for t in types}

    for h in range(24):
        hk    = f"{h:02d}"
        row   = {t: dict(day_data.get(hk, {})).get(t, 0) for t in types}
        total = sum(row.values())
        rows.append({
            "Hour":           f"{hk}:00",
            "Persons":        row["person"],
            "Vehicles":       row["vehicle"],
            "Animals":        row["animal"],
            "Watchlist Hits": row["watchlist"],
            "Total":          total,
        })
        for t in types:
            summary[t] += row[t]

    # Summary row at bottom
    rows.append({
        "Hour":           "TOTAL",
        "Persons":        summary["person"],
        "Vehicles":       summary["vehicle"],
        "Animals":        summary["animal"],
        "Watchlist Hits": summary["watchlist"],
        "Total":          sum(summary.values()),
    })

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["Hour", "Persons", "Vehicles", "Animals", "Watchlist Hits", "Total"]
    )
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=flynet_report_{date_str}.csv"}
    )


# =============================================================================
# ░░  REPORT EXPORT — PDF
# =============================================================================

@app.get("/report/daily/pdf")
def api_daily_report_pdf(date_str: Optional[str] = Query(default=None, alias="date")):
    """
    Download daily detection report as PDF.
    ?date=YYYY-MM-DD  (defaults to today)
    Requires: pip install reportlab
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="reportlab not installed. Run: pip install reportlab"
        )

    if date_str is None:
        date_str = date.today().isoformat()

    # Read from SQLite (persists across restarts) — fall back to memory if empty
    db_data  = _db_hourly_report(date_str)
    with analytics_lock:
        mem_data = dict(hourly_buckets.get(date_str, {}))
        pc       = dict(people_count)
    # Merge: prefer DB data, supplement with memory data
    day_data = mem_data.copy()
    for hk, type_counts in db_data.items():
        if hk not in day_data:
            day_data[hk] = {}
        for t, cnt in type_counts.items():
            day_data[hk][t] = max(day_data[hk].get(t, 0), cnt)

    types   = ["person", "vehicle", "animal", "watchlist"]
    summary = {t: 0 for t in types}
    h_rows  = []

    for h in range(24):
        hk    = f"{h:02d}"
        row   = {t: dict(day_data.get(hk, {})).get(t, 0) for t in types}
        total = sum(row.values())
        h_rows.append([f"{hk}:00", row["person"], row["vehicle"],
                        row["animal"], row["watchlist"], total])
        for t in types:
            summary[t] += row[t]

    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm,   bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    items  = []

    # ── Title ──────────────────────────────────────────────────────────────────
    items.append(Paragraph("Flynet AI Surveillance", ParagraphStyle(
        "title", parent=styles["Heading1"],
        alignment=TA_CENTER, fontSize=18, spaceAfter=4,
        textColor=colors.HexColor("#1a1a2e")
    )))
    items.append(Paragraph(f"Daily Detection Report — {date_str}", ParagraphStyle(
        "sub", parent=styles["Normal"],
        alignment=TA_CENTER, fontSize=11, spaceAfter=2,
        textColor=colors.HexColor("#555555")
    )))
    items.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ParagraphStyle("gen", parent=styles["Normal"],
                       alignment=TA_CENTER, fontSize=9,
                       textColor=colors.grey, spaceAfter=16)
    ))
    items.append(Spacer(1, 0.3*cm))

    # ── Summary cards ──────────────────────────────────────────────────────────
    summary_data = [
        ["Persons", "Vehicles", "Animals", "Watchlist Hits", "Total"],
        [str(summary["person"]), str(summary["vehicle"]),
         str(summary["animal"]), str(summary["watchlist"]),
         str(sum(summary.values()))],
    ]
    st = Table(summary_data, colWidths=[3.2*cm]*5)
    st.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("FONTNAME",      (0,1), (-1,1),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,1), (-1,1),  16),
        ("BACKGROUND",    (0,1), (-1,1),  colors.HexColor("#f0f4ff")),
        ("TEXTCOLOR",     (0,1), (-1,1),  colors.HexColor("#1e3a5f")),
        ("BOX",           (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("INNERGRID",     (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ]))
    items.append(st)
    items.append(Spacer(1, 0.5*cm))

    # ── People counting ────────────────────────────────────────────────────────
    if pc:
        items.append(Paragraph("People Counting (Entry / Exit)", ParagraphStyle(
            "h2", parent=styles["Heading2"], fontSize=11,
            textColor=colors.HexColor("#1e3a5f"), spaceAfter=4
        )))
        pc_data = [["Camera", "Entry (IN)", "Exit (OUT)", "Net"]]
        for cam, c in pc.items():
            pc_data.append([cam, c.get("in", 0), c.get("out", 0),
                            c.get("in", 0) - c.get("out", 0)])
        pt = Table(pc_data, colWidths=[8*cm, 3*cm, 3*cm, 3*cm])
        pt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#1e3a5f")),
            ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 9),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#f7f9fc")]),
            ("BOX",           (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
            ("INNERGRID",     (0,0), (-1,-1), 0.5, colors.HexColor("#eeeeee")),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        items.append(pt)
        items.append(Spacer(1, 0.5*cm))

    # ── Hourly table ───────────────────────────────────────────────────────────
    items.append(Paragraph("Hourly Breakdown", ParagraphStyle(
        "h2", parent=styles["Heading2"], fontSize=11,
        textColor=colors.HexColor("#1e3a5f"), spaceAfter=4
    )))

    table_data = [["Hour", "Persons", "Vehicles", "Animals", "Watchlist", "Total"]]
    table_data += h_rows
    table_data.append([
        "TOTAL", summary["person"], summary["vehicle"],
        summary["animal"], summary["watchlist"], sum(summary.values())
    ])

    ht = Table(table_data, colWidths=[3*cm, 3*cm, 3*cm, 3*cm, 3.5*cm, 2.5*cm], repeatRows=1)
    ht.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),  (-1,0),  colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",     (0,0),  (-1,0),  colors.white),
        ("FONTNAME",      (0,0),  (-1,0),  "Helvetica-Bold"),
        ("BACKGROUND",    (0,-1), (-1,-1), colors.HexColor("#e8f0fe")),
        ("FONTNAME",      (0,-1), (-1,-1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (0,-1), (-1,-1), colors.HexColor("#1e3a5f")),
        ("FONTSIZE",      (0,0),  (-1,-1), 9),
        ("ALIGN",         (0,0),  (-1,-1), "CENTER"),
        ("VALIGN",        (0,0),  (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),  (-1,-2), [colors.white, colors.HexColor("#f7f9fc")]),
        ("BOX",           (0,0),  (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("INNERGRID",     (0,0),  (-1,-1), 0.5, colors.HexColor("#eeeeee")),
        ("TOPPADDING",    (0,0),  (-1,-1), 5),
        ("BOTTOMPADDING", (0,0),  (-1,-1), 5),
    ]))
    items.append(ht)
    items.append(Spacer(1, 0.5*cm))
    items.append(Paragraph(
        f"Flynet AI Engine · {len(cameras)} camera(s) · Report for {date_str}",
        ParagraphStyle("footer", parent=styles["Normal"],
                       fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    ))

    doc.build(items)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=flynet_report_{date_str}.pdf"}
    )

# =============================================================================
# ░░  PER-CAMERA STATE
# =============================================================================

track_memory:   dict = {}
alert_cooldown: dict = {}
plate_memory:   dict = {}
face_memory:    dict = {}
line_side:      dict = {}
prev_frames:    dict = {}    # [cam_name] = last grayscale frame for motion diff
motion_cooldown:dict = {}    # [cam_name] = last motion alert timestamp
fire_cooldown:  dict = {}    # [cam_name] = last fire alert timestamp

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

def _is_off_hours(camera: dict = None) -> bool:
    """
    Return True if current time is within the off-hours window.
    Per-camera schedule takes priority over global config.
    cameras.json example:
      "off_hours": { "start": 22, "end": 6 }
    If not set, falls back to MOTION_OFF_START / MOTION_OFF_END globals.
    """
    h = datetime.now().hour

    if camera and "off_hours" in camera:
        start = camera["off_hours"].get("start", MOTION_OFF_START)
        end   = camera["off_hours"].get("end",   MOTION_OFF_END)
    else:
        start = MOTION_OFF_START
        end   = MOTION_OFF_END

    if start > end:
        # Overnight range e.g. 22 → 6
        return h >= start or h < end
    else:
        # Same-day range e.g. 0 → 8
        return start <= h < end


def detect_motion(cam_name: str, frame: np.ndarray) -> list:
    """
    Frame-differencing motion detector.
    Returns list of (x, y, w, h) bounding boxes where significant motion found.
    Runs on every frame — very fast, no GPU needed.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (MOTION_BLUR, MOTION_BLUR), 0)

    if prev_frames.get(cam_name) is None:
        prev_frames[cam_name] = gray
        return []

    diff   = cv2.absdiff(prev_frames[cam_name], gray)
    prev_frames[cam_name] = gray

    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh    = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= MOTION_THRESHOLD:
            boxes.append(cv2.boundingRect(cnt))
    return boxes


def detect_fire_color(frame: np.ndarray) -> list:
    """
    Strict color-based fire detection using HSV thresholding.

    Criteria to avoid false positives (floor markings, lights, etc.):
    1. Color must be in fire HSV range (red/orange only — yellow excluded)
    2. Region must be bright enough (real fire glows, paint does not)
    3. Region must be large enough (> FIRE_MIN_AREA px²)
    4. Region must not be too rectangular/uniform (fire is irregular)

    Returns list of (x, y, w, h) bounding boxes.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Tighter fire HSV ranges — red/orange only, high saturation + value
    # Removed yellow (20-40 hue) which was matching floor markings
    mask1 = cv2.inRange(hsv, np.array([0,   150, 150]), np.array([15,  255, 255]))  # deep red/orange
    mask2 = cv2.inRange(hsv, np.array([160, 150, 150]), np.array([180, 255, 255]))  # upper red wrap

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.dilate(mask, None, iterations=3)
    mask = cv2.erode(mask,  None, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < FIRE_MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Brightness check — real fire is bright, floor paint is not
        roi   = frame[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        mean_brightness = float(cv2.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))[0])
        if mean_brightness < FIRE_MIN_BRIGHTNESS:
            continue

        # Shape check — reject very thin/rectangular shapes (floor lines)
        # Fire contours are irregular; lines have extreme aspect ratios
        aspect = w / h if h > 0 else 999
        if aspect > 6 or aspect < 0.15:   # very wide or very tall = line, not fire
            continue

        # Solidity check — fire contours are irregular (low solidity)
        # Solid rectangles (floor paint) have solidity close to 1.0
        hull     = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 1.0
        if solidity > 0.90:               # too solid/uniform = not fire
            continue

        boxes.append((x, y, w, h))

    return boxes


def detect_fire(frame: np.ndarray) -> tuple:
    """Fire detection — tries YOLOv8 fire model first, falls back to color.
    Returns: (boxes, method) — boxes = list of (x, y, w, h)
    """
    if fire_yolo is not None:
        try:
            results = fire_yolo(frame, imgsz=320, conf=FIRE_CONFIDENCE)[0]
            if results.boxes is not None and len(results.boxes):
                boxes = []
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
                return boxes, "model"
        except Exception as ex:
            print(f"[Fire model error] {ex}")

    # Color-based fallback
    boxes = detect_fire_color(frame)
    return boxes, "color"


# =============================================================================
# ░░  CAMERA THREAD
# =============================================================================

def process_camera(camera: dict):
    cam_name   = camera["name"]
    cam_url    = camera["rtsp"]
    count_line = camera.get("count_line")

    print(f"[{cam_name}] Connecting …")

    track_memory[cam_name]    = {}
    alert_cooldown[cam_name]  = {}
    plate_memory[cam_name]    = {}
    face_memory[cam_name]     = {}
    line_side[cam_name]       = {}
    prev_frames[cam_name]     = None
    motion_cooldown[cam_name] = 0
    fire_cooldown[cam_name]   = 0

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

        # ── Motion detection (runs on EVERY frame during off-hours) ───────────
        if _is_off_hours(camera):
            motion_boxes = detect_motion(cam_name, frame)
            if motion_boxes:
                now_ts = time.time()
                if now_ts - motion_cooldown.get(cam_name, 0) > MOTION_COOLDOWN:
                    motion_cooldown[cam_name] = now_ts

                    # Draw motion boxes on frame
                    for (mx, my, mw, mh) in motion_boxes:
                        cv2.rectangle(frame, (mx, my), (mx+mw, my+mh), (0, 165, 255), 2)
                    cv2.putText(frame, "MOTION DETECTED", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

                    snap = save_snapshot(frame, f"{cam_name}_motion")

                    alert = {
                        "event":      "detection",
                        "camera":     cam_name,
                        "object":     "motion",
                        "type":       "motion",
                        "track_id":   None,
                        "confidence": 1.0,
                        "plate":      None,
                        "face":       None,
                        "watchlist":  False,
                        "snapshot":   snap,
                        "time":       datetime.now().isoformat(),
                    }

                    print(f"[MOTION] {cam_name} | off-hours movement detected")
                    broadcast(alert)

                    # Save to SQLite
                    threading.Thread(
                        target=_db_save_alert, args=(alert,), daemon=True
                    ).start()

        # ── Fire detection (runs on EVERY frame — 24/7, no schedule) ────────────
        fire_boxes, fire_method = detect_fire(frame)
        if fire_boxes:
            now_ts = time.time()
            if now_ts - fire_cooldown.get(cam_name, 0) > FIRE_COOLDOWN:
                fire_cooldown[cam_name] = now_ts

                # Draw fire boxes on frame
                for (fx, fy, fw, fh) in fire_boxes:
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 69, 255), 3)
                    cv2.putText(frame, "FIRE", (fx, fy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 69, 255), 2)
                cv2.putText(frame, "! FIRE DETECTED !", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 69, 255), 3)

                snap = save_snapshot(frame, f"{cam_name}_fire")

                alert = {
                    "event":      "detection",
                    "camera":     cam_name,
                    "object":     "fire",
                    "type":       "fire",
                    "track_id":   None,
                    "confidence": 0.95 if fire_method == "model" else 0.70,
                    "plate":      None,
                    "face":       None,
                    "watchlist":  False,
                    "snapshot":   snap,
                    "time":       datetime.now().isoformat(),
                }

                print(f"[FIRE] {cam_name} | fire detected via {fire_method}")
                broadcast(alert)
                threading.Thread(
                    target=_db_save_alert, args=(alert,), daemon=True
                ).start()

                # Record in hourly analytics
                with analytics_lock:
                    d = datetime.now().strftime("%Y-%m-%d")
                    h = datetime.now().strftime("%H")
                    hourly_buckets[d][h]["fire"] += 1

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
                    cached       = face_memory[cam_name][track_id]
                    face_name    = cached["name"]
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
            snap       = save_snapshot(frame, f"{cam_name}_{cls}_ID{track_id}")
            alert_type = "watchlist" if is_watchlist else (
                "face_detected" if face_name == "unknown" else obj_type
            )

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