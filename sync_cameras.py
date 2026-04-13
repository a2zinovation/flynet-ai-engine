#!/usr/bin/env python3
"""
sync_cameras.py  —  Flynet Camera Sync Utility
=================================================
Fetches cameras from the Laravel backend API and updates cameras.json.

Usage:
    python sync_cameras.py

Config (edit below or use environment variables):
    CAMERAS_API_URL   — full URL to the cameras API
    CAMERAS_API_TOKEN — Bearer token for authentication

How it works:
- Fetches ALL cameras from the API
- Only keeps cameras that are BOTH:
    1. is_active == True  (checks multiple field name variants)
    2. Have a valid RTSP / stream URL
- Anything else is REMOVED from cameras.json — no exceptions, no flags
- Preserves count_line and off_hours for cameras that survive the filter
- Updates RTSP stream URLs from the API response

cameras.json after sync = exactly the set of working cameras.

Set DEBUG_API_RESPONSE = True to print raw API fields and diagnose
why cameras are being included or excluded.
"""

import json
import os
import requests

# =============================================================================
#  CONFIG
# =============================================================================
API_URL       = os.getenv("CAMERAS_API_URL",   "https://backend.pinkdreams.store/api/cameras-for-ai-detection")
API_TOKEN     = os.getenv("CAMERAS_API_TOKEN", "")
CAMERAS_JSON  = "cameras.json"
AUTO_RELOAD   = True
AI_ENGINE_URL = os.getenv("AI_ENGINE_URL", "http://localhost:8000")

# Set True to print every raw camera object from the API so you can see
# the exact field names (is_active vs active vs status, etc.)
DEBUG_API_RESPONSE = True

DEFAULT_COUNT_LINE = {"x1": 0, "y1": 360, "x2": 1280, "y2": 520, "in_side": 1}
DEFAULT_OFF_HOURS  = {"start": 20, "end": 6}


# =============================================================================
#  ACTIVE FIELD DETECTION
#  Checks multiple field name variants so it works regardless of what
#  the Laravel API uses: is_active, active, status, enabled, etc.
# =============================================================================
def is_camera_active(cam: dict) -> bool:
    for field in ("is_active", "active", "status", "enabled", "is_enabled"):
        val = cam.get(field)
        if val is None:
            continue
        if val is True:
            return True
        if val is False:
            return False
        if isinstance(val, str):
            return val.lower() in ("1", "true", "active", "online", "enabled", "yes")
        if isinstance(val, int):
            return val == 1

    # No active/status field found at all — warn and include camera
    print(f"  [WARN] No active/status field found for camera id={cam.get('id')} "
          f"name='{cam.get('name')}' — assuming active. "
          f"Fields: {list(cam.keys())}")
    return True


# =============================================================================
#  FETCH FROM API  (handles pagination)
# =============================================================================
def fetch_cameras() -> list:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"

    all_cameras = []
    next_url    = API_URL

    while next_url:
        resp = requests.get(next_url, headers=headers, timeout=15, verify=False)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list):
            all_cameras.extend(data)
            next_url = None
        elif isinstance(data.get("data"), list):
            all_cameras.extend(data["data"])
            next_url = data.get("next_page_url") or data.get("links", {}).get("next")
        elif isinstance(data.get("data", {}).get("data"), list):
            all_cameras.extend(data["data"]["data"])
            next_url = data["data"].get("next_page_url")
        else:
            print(f"[WARN] Unexpected API response shape: {list(data.keys())}")
            break

    return all_cameras


def get_rtsp(cam: dict) -> str:
    """Try every field name the API might use for the stream URL."""
    return (
        cam.get("rtsp_url")     or
        cam.get("rtsp")         or
        cam.get("stream_url")   or
        cam.get("playback_url") or
        cam.get("hls_url")      or
        ""
    )


# =============================================================================
#  LOAD EXISTING cameras.json  ->  {id: camera_dict}
# =============================================================================
def load_existing() -> dict:
    if not os.path.exists(CAMERAS_JSON):
        return {}
    try:
        with open(CAMERAS_JSON) as f:
            data = json.load(f)
        return {cam["id"]: cam for cam in data.get("cameras", [])}
    except Exception as e:
        print(f"[WARN] Could not read {CAMERAS_JSON}: {e}")
        return {}


# =============================================================================
#  MAIN SYNC
# =============================================================================
def sync():
    # ── 1. Fetch from API ─────────────────────────────────────────────────────
    try:
        print(f"Fetching cameras from {API_URL} ...")
        api_cameras = fetch_cameras()
        print(f"API returned {len(api_cameras)} camera(s) total.")
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Could not connect to {API_URL}")
        print("        Aborting — cameras.json unchanged.")
        return
    except Exception as e:
        print(f"[ERROR] Fetch failed: {e}")
        print("        Aborting — cameras.json unchanged.")
        return

    # ── 2. Debug: show raw API fields to diagnose is_active field name ─────────
    if DEBUG_API_RESPONSE and api_cameras:
        print()
        print("-" * 70)
        print("DEBUG - Raw API status fields per camera:")
        print(f"  {'id':>5}  {'name':<22}  {'is_active':<10}  {'active':<7}  "
              f"{'status':<10}  {'enabled':<8}")
        print("  " + "-" * 65)
        for cam in api_cameras:
            print(f"  {str(cam.get('id', '?')):>5}  "
                  f"{str(cam.get('name', '?')):<22}  "
                  f"{str(cam.get('is_active', 'N/A')):<10}  "
                  f"{str(cam.get('active', 'N/A')):<7}  "
                  f"{str(cam.get('status', 'N/A')):<10}  "
                  f"{str(cam.get('enabled', 'N/A')):<8}")
        print("-" * 70)
        print()

    # ── 3. Load existing cameras.json ─────────────────────────────────────────
    existing = load_existing()
    print(f"Existing cameras.json has {len(existing)} camera(s).")
    print()

    final       = []
    added       = []
    url_updated = []
    removed     = []
    rejected    = []
    valid_ids   = set()

    for cam in api_cameras:
        cam_id   = cam["id"]
        cam_name = cam.get("name", f"Camera {cam_id}")

        # ── Filter 1: must be active ──────────────────────────────────────────
        if not is_camera_active(cam):
            print(f"  [SKIP-INACTIVE]  {cam_name} (id={cam_id})")
            rejected.append(f"{cam_name} [inactive]")
            continue

        # ── Filter 2: must have a stream URL ──────────────────────────────────
        rtsp = get_rtsp(cam)
        if not rtsp:
            print(f"  [SKIP-NO-URL]    {cam_name} (id={cam_id})")
            rejected.append(f"{cam_name} [no stream URL]")
            continue

        # ── Passed both filters — include in cameras.json ─────────────────────
        valid_ids.add(cam_id)

        if cam_id in existing:
            entry    = dict(existing[cam_id])
            old_rtsp = entry.get("rtsp", "")
            entry["name"] = cam_name
            entry["rtsp"] = rtsp
            entry.setdefault("count_line", DEFAULT_COUNT_LINE)
            entry.setdefault("off_hours",  DEFAULT_OFF_HOURS)
            final.append(entry)

            if old_rtsp != rtsp:
                print(f"  [URL-UPDATED]    {cam_name} (id={cam_id})")
                url_updated.append(cam_name)
            else:
                print(f"  [OK]             {cam_name} (id={cam_id})")
        else:
            entry = {
                "id":         cam_id,
                "name":       cam_name,
                "rtsp":       rtsp,
                "count_line": DEFAULT_COUNT_LINE,
                "off_hours":  DEFAULT_OFF_HOURS,
            }
            final.append(entry)
            added.append(cam_name)
            print(f"  [NEW]            {cam_name} (id={cam_id})")

    # ── 4. Everything in cameras.json that didn't pass filters is removed ──────
    for cam_id, cam in existing.items():
        if cam_id not in valid_ids:
            print(f"  [REMOVED]        {cam['name']} (id={cam_id})")
            removed.append(cam["name"])

    # ── 5. Write cameras.json ─────────────────────────────────────────────────
    final.sort(key=lambda c: c["id"])
    with open(CAMERAS_JSON, "w") as f:
        json.dump({"cameras": final}, f, indent=2)

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print(f"  cameras.json -> {len(final)} working camera(s)")
    if added:        print(f"  Added      : {', '.join(added)}")
    if url_updated:  print(f"  URL updated: {', '.join(url_updated)}")
    if removed:      print(f"  Removed    : {', '.join(removed)}")
    if rejected:     print(f"  Rejected   : {', '.join(rejected)}")
    print("=" * 55)

    if not final:
        print()
        print("[WARN] cameras.json is EMPTY — no active cameras with a stream URL found.")
        print("       Go to the website and check camera statuses.")

    if added:
        print()
        print("IMPORTANT: Edit cameras.json to set correct count_line coordinates")
        print("           for any newly added cameras.")

    # ── 7. Hot-reload app.py if it's running ──────────────────────────────────
    if AUTO_RELOAD:
        _trigger_reload()


# =============================================================================
#  HOT-RELOAD
# =============================================================================
def _trigger_reload():
    url = f"{AI_ENGINE_URL.rstrip('/')}/cameras/reload"
    print(f"\nTriggering AI engine reload -> {url}")
    try:
        resp = requests.post(url, timeout=10, verify=False)
        resp.raise_for_status()
        data = resp.json()
        print(f"  Reload OK — {data.get('cameras')} camera(s) active")
        if data.get("added"):     print(f"  Started  : {', '.join(data['added'])}")
        if data.get("removed"):   print(f"  Stopped  : {', '.join(data['removed'])}")
        if data.get("restarted"): print(f"  Restarted: {', '.join(data['restarted'])}")
    except requests.exceptions.ConnectionError:
        print(f"  app.py not reachable at {AI_ENGINE_URL} — cameras.json written OK.")
        print("  Restart app.py manually to pick up the changes.")
    except requests.exceptions.HTTPError as e:
        print(f"  [WARN] Reload returned {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        print(f"  [WARN] Reload failed: {e}")


if __name__ == "__main__":
    sync()