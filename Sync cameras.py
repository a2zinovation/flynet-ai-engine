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
- Fetches active cameras from the API
- Merges with existing cameras.json to PRESERVE:
    count_line  (virtual counting line — must be set manually per camera)
    off_hours   (motion detection schedule — must be set manually per camera)
- Adds new cameras found in API
- Removes cameras no longer in API (optional — controlled by REMOVE_MISSING)
- Updates RTSP stream URLs from API response

Run this whenever cameras are added/removed on the website.
Or automate: add to a cron job / Task Scheduler.
"""

import json
import os
import sys
import requests

# =============================================================================
#  CONFIG
# =============================================================================
API_URL        = os.getenv("CAMERAS_API_URL",   "https://backend.pinkdreams.store/api/cameras-for-ai-detection")
API_TOKEN      = os.getenv("CAMERAS_API_TOKEN", "")   # likely not needed — public AI endpoint
CAMERAS_JSON   = "cameras.json"
REMOVE_MISSING = False   # set True to remove cameras deleted from the website

# After writing cameras.json, automatically call the AI engine's hot-reload endpoint.
# Set to False if you want to restart app.py manually instead.
AUTO_RELOAD    = True
AI_ENGINE_URL  = os.getenv("AI_ENGINE_URL", "http://localhost:8000")

# Default count_line and off_hours for NEW cameras (edit per camera after first sync)
DEFAULT_COUNT_LINE = {
    "x1": 0, "y1": 360, "x2": 1280, "y2": 520, "in_side": 1
}
DEFAULT_OFF_HOURS = {"start": 20, "end": 6}

# =============================================================================
#  FETCH FROM API
# =============================================================================
def fetch_cameras():
    headers = {
        "Accept":        "application/json",
        "Content-Type":  "application/json",
    }
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"

    all_cameras = []
    next_url = API_URL

    try:
        while next_url:
            resp = requests.get(next_url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            # Handle response shapes: { data: { data: [...] } } or { data: [...] } or [...]
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

    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Could not connect to {API_URL}")
        print("        Make sure the backend server is running.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] API returned {e.response.status_code}: {e.response.text[:200]}")
        if e.response.status_code == 401:
            print("        Set CAMERAS_API_TOKEN environment variable with a valid token.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


# =============================================================================
#  EXTRACT RTSP URL FROM CAMERA OBJECT
# =============================================================================
def get_rtsp(cam):
    """Try multiple field names the API might use for the stream URL."""
    return (
        cam.get("rtsp_url")    or
        cam.get("rtsp")        or
        cam.get("stream_url")  or
        cam.get("playback_url") or
        cam.get("hls_url")     or
        ""
    )


# =============================================================================
#  LOAD EXISTING cameras.json
# =============================================================================
def load_existing():
    if not os.path.exists(CAMERAS_JSON):
        return {}
    try:
        with open(CAMERAS_JSON) as f:
            data = json.load(f)
        existing = {}
        for cam in data.get("cameras", []):
            existing[cam["id"]] = cam
        return existing
    except Exception as e:
        print(f"[WARN] Could not read {CAMERAS_JSON}: {e}")
        return {}


# =============================================================================
#  MAIN SYNC
# =============================================================================
def sync():
    print(f"Fetching cameras from {API_URL} ...")
    api_cameras = fetch_cameras()
    active = [c for c in api_cameras if c.get("is_active")]
    print(f"Found {len(api_cameras)} cameras, {len(active)} active.")

    existing = load_existing()
    updated  = []
    added    = []
    skipped  = []

    api_ids = set()
    for cam in active:
        cam_id   = cam["id"]
        cam_name = cam.get("name", f"Camera {cam_id}")
        rtsp     = get_rtsp(cam)
        api_ids.add(cam_id)

        if not rtsp:
            print(f"  [SKIP] {cam_name} (id={cam_id}) — no stream URL found in API response")
            skipped.append(cam_name)
            continue

        if cam_id in existing:
            # Update existing entry — preserve count_line and off_hours
            entry = dict(existing[cam_id])
            old_rtsp = entry.get("rtsp", "")
            entry["name"] = cam_name
            entry["rtsp"] = rtsp
            # Keep count_line and off_hours from local config
            if "count_line" not in entry:
                entry["count_line"] = DEFAULT_COUNT_LINE
            if "off_hours" not in entry:
                entry["off_hours"] = DEFAULT_OFF_HOURS
            updated.append(entry)
            if old_rtsp != rtsp:
                print(f"  [UPDATE] {cam_name} (id={cam_id}) — stream URL updated")
            else:
                print(f"  [OK]     {cam_name} (id={cam_id})")
        else:
            # New camera — add with defaults
            entry = {
                "id":         cam_id,
                "name":       cam_name,
                "rtsp":       rtsp,
                "count_line": DEFAULT_COUNT_LINE,
                "off_hours":  DEFAULT_OFF_HOURS,
            }
            updated.append(entry)
            added.append(cam_name)
            print(f"  [NEW]    {cam_name} (id={cam_id}) — added")

    # Handle cameras removed from the website
    removed = []
    if REMOVE_MISSING:
        for cam_id, cam in existing.items():
            if cam_id not in api_ids:
                print(f"  [REMOVE] {cam['name']} (id={cam_id}) — no longer in API")
                removed.append(cam["name"])
    else:
        # Keep cameras not in API (may be temporarily offline)
        for cam_id, cam in existing.items():
            if cam_id not in api_ids:
                updated.append(cam)
                print(f"  [KEEP]   {cam['name']} (id={cam_id}) — not in API, keeping")

    # Sort by id
    updated.sort(key=lambda c: c["id"])

    # Write cameras.json
    with open(CAMERAS_JSON, "w") as f:
        json.dump({"cameras": updated}, f, indent=2)

    print()
    print("=" * 50)
    print(f"cameras.json updated: {len(updated)} cameras")
    if added:
        print(f"  Added:   {', '.join(added)}")
    if removed:
        print(f"  Removed: {', '.join(removed)}")
    if skipped:
        print(f"  Skipped: {', '.join(skipped)} (no stream URL)")
    print()
    print("IMPORTANT: Edit cameras.json to set correct count_line")
    print("           coordinates for any newly added cameras.")

    # ── Hot-reload the running AI engine ──────────────────────────────────────
    if AUTO_RELOAD:
        _trigger_reload()


def _trigger_reload():
    """
    POST to the AI engine's /cameras/reload endpoint so it picks up the
    updated cameras.json immediately without a server restart.
    """
    url = f"{AI_ENGINE_URL.rstrip('/')}/cameras/reload"
    print(f"\nTriggering AI engine reload → {url}")
    try:
        resp = requests.post(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print("AI engine reloaded successfully:")
        print(f"  Total cameras : {data.get('cameras')}")
        if data.get('added'):
            print(f"  Started       : {', '.join(data['added'])}")
        if data.get('removed'):
            print(f"  Stopped       : {', '.join(data['removed'])}")
        if data.get('restarted'):
            print(f"  Restarted     : {', '.join(data['restarted'])}")
    except requests.exceptions.ConnectionError:
        print(f"[INFO] AI engine not reachable at {AI_ENGINE_URL} — cameras.json written OK.")
        print("       Restart app.py manually to pick up the changes.")
    except requests.exceptions.HTTPError as e:
        print(f"[WARN] Reload endpoint returned {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        print(f"[WARN] Reload failed: {e}")


if __name__ == "__main__":
    sync()
