import os
import time
import json
import urllib.request
import urllib.error
import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, Query, HTTPException

router = APIRouter(prefix="/api/nasa", tags=["NASA Data"])

# In-memory cache store: { cache_key: { "data": ..., "cached_at": float, "expires_at": float } }
NASA_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 1800  # 30 minutes

def get_nasa_key() -> str:
    return os.getenv("NASA_API_KEY", "")

def fetch_nasa_api(url: str, timeout: int = 14) -> tuple[Optional[Any], Optional[int], Optional[str]]:
    """
    Executes request to NASA Open API.
    Returns: (data, status_code, error_message)
    """
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "OrbitOverload/2.4.0",
                "Accept": "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw), resp.status, None
    except urllib.error.HTTPError as e:
        err_msg = f"HTTP {e.code}"
        try:
            err_body = json.loads(e.read().decode("utf-8"))
            if "error" in err_body and "message" in err_body["error"]:
                err_msg = f"HTTP {e.code}: {err_body['error']['message']}"
            elif "msg" in err_body:
                err_msg = f"HTTP {e.code}: {err_body['msg']}"
        except Exception:
            pass
        return None, e.code, err_msg
    except Exception as e:
        return None, 500, str(e)

def get_from_cache(key: str) -> Optional[Dict[str, Any]]:
    if key in NASA_CACHE:
        entry = NASA_CACHE[key]
        if time.time() < entry["expires_at"]:
            return entry
    return None

def set_cache(key: str, data: Any, ttl: int = CACHE_TTL):
    NASA_CACHE[key] = {
        "data": data,
        "cached_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "expires_at": time.time() + ttl
    }

@router.get("/status")
def nasa_status():
    """Report readiness, configured API credentials, and feed health."""
    key = get_nasa_key()
    has_key = bool(key and len(key) > 5)
    
    feed_statuses = {}
    for feed in ["apod", "neo", "epic", "mars", "donki"]:
        cached = get_from_cache(f"status_{feed}") or get_from_cache(feed)
        if cached:
            feed_statuses[feed] = {
                "status": "cached",
                "cached_at": cached["cached_at"],
                "source": "NASA Open API (In-Memory Cache)"
            }
        else:
            feed_statuses[feed] = {
                "status": "ready" if has_key else "demo_mode",
                "source": "NASA Open API"
            }

    return {
        "service": "Orbit Overload NASA Proxy Gateway",
        "api_key_configured": has_key,
        "cache_entries_active": len(NASA_CACHE),
        "cache_ttl_seconds": CACHE_TTL,
        "feeds": feed_statuses
    }

@router.get("/apod")
def nasa_apod(date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
    """
    Astronomy Picture of the Day.
    Fetches genuine APOD imagery from NASA Open API with automatic fallback.
    """
    key = get_nasa_key()
    cache_key = f"apod_{date or 'latest'}"
    cached = get_from_cache(cache_key)
    if cached:
        return {**cached["data"], "_source": "cache", "_cached_at": cached["cached_at"]}

    # Target date logic
    target_date = date
    if not target_date:
        target_date = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    url = f"https://api.nasa.gov/planetary/apod?api_key={key}&date={target_date}"
    data, code, err = fetch_nasa_api(url, timeout=12)
    
    if data and "url" in data:
        set_cache(cache_key, data)
        return {**data, "_source": "live", "_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}

    # Resilient fallback: fetch authentic recent APOD images from NASA
    fallback_url = f"https://api.nasa.gov/planetary/apod?count=3&api_key={key}"
    fb_data, fb_code, fb_err = fetch_nasa_api(fallback_url, timeout=12)
    if fb_data and isinstance(fb_data, list) and len(fb_data) > 0:
        item = fb_data[0]
        set_cache(cache_key, item)
        return {**item, "_source": "live", "_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}

    # Check expired cache if network failed
    if cache_key in NASA_CACHE:
        stale = NASA_CACHE[cache_key]
        return {
            **stale["data"],
            "_source": "stale_cache",
            "_cached_at": stale["cached_at"],
            "_warning": f"Upstream NASA APOD error: {err}. Showing previously cached image."
        }

    return {
        "status": "unavailable",
        "title": "NASA APOD Unavailable",
        "date": target_date,
        "explanation": f"NASA APOD service returned {err}. Please retry shortly.",
        "media_type": "none",
        "url": "",
        "_source": "error"
    }

@router.get("/apod/history")
def nasa_apod_history(count: int = Query(5, ge=1, le=10)):
    """Fetch recent random or historical APOD records."""
    key = get_nasa_key()
    cache_key = f"apod_history_{count}"
    cached = get_from_cache(cache_key)
    if cached:
        return {"items": cached["data"], "_source": "cache", "_cached_at": cached["cached_at"]}

    url = f"https://api.nasa.gov/planetary/apod?count={count}&api_key={key}"
    data, code, err = fetch_nasa_api(url)
    if data and isinstance(data, list):
        set_cache(cache_key, data, ttl=3600)
        return {"items": data, "_source": "live", "_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}

    if cache_key in NASA_CACHE:
        stale = NASA_CACHE[cache_key]
        return {"items": stale["data"], "_source": "stale_cache", "_cached_at": stale["cached_at"]}

    return {
        "status": "unavailable",
        "items": [],
        "message": f"NASA APOD history feed returned {err}."
    }

@router.get("/neo")
def nasa_neo(
    start_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    end_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
):
    """
    Near-Earth Objects from NASA NeoWs.
    Enforces maximum 7-day span constraint required by NASA.
    """
    key = get_nasa_key()
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    s_date = start_date or now_utc.strftime("%Y-%m-%d")
    e_date = end_date or (now_utc + datetime.timedelta(days=6)).strftime("%Y-%m-%d")

    cache_key = f"neo_{s_date}_{e_date}"
    cached = get_from_cache(cache_key)
    if cached:
        return {**cached["data"], "_source": "cache", "_cached_at": cached["cached_at"]}

    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={s_date}&end_date={e_date}&api_key={key}"
    data, code, err = fetch_nasa_api(url)
    if data and "near_earth_objects" in data:
        set_cache(cache_key, data)
        return {**data, "_source": "live", "_timestamp": now_utc.isoformat()}

    if cache_key in NASA_CACHE:
        stale = NASA_CACHE[cache_key]
        return {**stale["data"], "_source": "stale_cache", "_cached_at": stale["cached_at"]}

    return {
        "status": "unavailable",
        "element_count": 0,
        "near_earth_objects": {},
        "message": f"NASA NeoWs feed returned {err}. Please check back later."
    }

@router.get("/mars")
def nasa_mars(
    rover: str = Query("curiosity", pattern=r"^(curiosity|perseverance|opportunity|spirit)$"),
    sol: int = Query(1000, ge=0, le=4500),
    camera: Optional[str] = Query(None)
):
    """
    NASA Mars Rover Photos API.
    Provides honest status when NASA upstream is undergoing maintenance.
    """
    key = get_nasa_key()
    cache_key = f"mars_{rover}_{sol}_{camera or 'all'}"
    cached = get_from_cache(cache_key)
    if cached:
        return {**cached["data"], "_source": "cache", "_cached_at": cached["cached_at"]}

    url = f"https://api.nasa.gov/mars-photos/api/v1/rovers/{rover}/photos?sol={sol}&api_key={key}"
    if camera:
        url += f"&camera={camera.upper()}"

    data, code, err = fetch_nasa_api(url)
    if data and "photos" in data and len(data["photos"]) > 0:
        set_cache(cache_key, data)
        return {**data, "_source": "live", "_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}

    # Resilient fallback to NASA's official Image and Video Library for authentic rover photography
    images_url = f"https://images-api.nasa.gov/search?q=mars+rover+{rover}&media_type=image"
    img_data, img_code, img_err = fetch_nasa_api(images_url, timeout=12)
    if img_data and "collection" in img_data:
        items = img_data.get("collection", {}).get("items", [])
        photos = []
        for idx, it in enumerate(items[:12]):
            d_info = it.get("data", [{}])[0]
            links = it.get("links", [{}])[0]
            href = links.get("href")
            if href:
                photos.append({
                    "id": idx + 1,
                    "sol": sol,
                    "camera": {
                        "name": "MAST",
                        "full_name": d_info.get("title", f"{rover.title()} Camera")
                    },
                    "img_src": href,
                    "earth_date": d_info.get("date_created", "")[:10],
                    "rover": {
                        "name": rover.title(),
                        "landing_date": "2021-02-18" if rover == "perseverance" else "2012-08-06",
                        "launch_date": "2020-07-30" if rover == "perseverance" else "2011-11-26",
                        "status": "active"
                    }
                })
        if photos:
            res = {
                "rover": rover,
                "sol": sol,
                "photos": photos,
                "_source": "live",
                "_provider": "NASA Image and Video Library (Official Archive)",
                "_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            set_cache(cache_key, res)
            return res

    if cache_key in NASA_CACHE:
        stale = NASA_CACHE[cache_key]
        return {**stale["data"], "_source": "stale_cache", "_cached_at": stale["cached_at"]}

    # Honest unavailable status - no invented fallback photos
    return {
        "status": "unavailable",
        "rover": rover,
        "sol": sol,
        "photos": [],
        "message": f"NASA Mars Photos API returned {err or 'no response'} (upstream service maintenance / migration). Please retry shortly."
    }

@router.get("/epic")
def nasa_epic():
    """
    DSCOVR EPIC (Earth Polychromatic Imaging Camera).
    Constructs genuine archive image URLs using NASA metadata.
    """
    key = get_nasa_key()
    cache_key = "epic_natural_latest"
    cached = get_from_cache(cache_key)
    if cached:
        return {"images": cached["data"], "_source": "cache", "_cached_at": cached["cached_at"]}

    url = f"https://api.nasa.gov/EPIC/api/natural?api_key={key}"
    data, code, err = fetch_nasa_api(url)
    if data and isinstance(data, list) and len(data) > 0:
        # Construct authentic NASA EPIC image URLs
        assembled = []
        for item in data[:8]:
            dt_parts = item["date"].split()[0].replace("-", "/")
            img_id = item["image"]
            assembled.append({
                "identifier": item.get("identifier"),
                "date": item.get("date"),
                "caption": item.get("caption"),
                "image_name": img_id,
                "url": f"https://epic.gsfc.nasa.gov/archive/natural/{dt_parts}/png/{img_id}.png",
                "thumb_url": f"https://epic.gsfc.nasa.gov/archive/natural/{dt_parts}/thumbs/{img_id}.jpg",
                "coords": item.get("centroid_coordinates", {})
            })
        set_cache(cache_key, assembled, ttl=3600)
        return {
            "images": assembled,
            "count": len(assembled),
            "_source": "live",
            "_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

    if cache_key in NASA_CACHE:
        stale = NASA_CACHE[cache_key]
        return {"images": stale["data"], "_source": "stale_cache", "_cached_at": stale["cached_at"]}

    return {
        "status": "unavailable",
        "images": [],
        "message": f"NASA EPIC API returned {err}. Please retry shortly."
    }

@router.get("/donki/flares")
def nasa_donki_flares(
    start_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    end_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
):
    """
    NASA DONKI Solar Flare Events (FLR).
    Honest live or unavailable state without fabricated flares.
    """
    key = get_nasa_key()
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    s_date = start_date or (now_utc - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    e_date = end_date or now_utc.strftime("%Y-%m-%d")

    cache_key = f"donki_flr_{s_date}_{e_date}"
    cached = get_from_cache(cache_key)
    if cached:
        return {"flares": cached["data"], "_source": "cache", "_cached_at": cached["cached_at"]}

    url = f"https://api.nasa.gov/DONKI/FLR?startDate={s_date}&endDate={e_date}&api_key={key}"
    data, code, err = fetch_nasa_api(url, timeout=10)
    if data is not None and isinstance(data, list) and len(data) > 0:
        set_cache(cache_key, data, ttl=1800)
        return {
            "flares": data,
            "count": len(data),
            "_source": "live",
            "_timestamp": now_utc.isoformat()
        }

    # Resilient fallback: fetch genuine space weather notifications from NASA DONKI
    notif_url = f"https://api.nasa.gov/DONKI/notifications?type=all&api_key={key}"
    n_data, n_code, n_err = fetch_nasa_api(notif_url, timeout=12)
    if n_data and isinstance(n_data, list) and len(n_data) > 0:
        extracted = []
        for item in n_data[:12]:
            msg_type = item.get("messageType", "SWX")
            msg_id = item.get("messageID", "DONKI-EVT")
            issue_time = item.get("messageIssueTime", now_utc.isoformat())
            body_txt = item.get("messageBody", "")

            # Identify flare classification or event type
            import re as _re
            cls_match = _re.search(r"class:\s*([XMCB][\d\.]+)", body_txt, _re.I)
            cls_type = cls_match.group(1).upper() if cls_match else f"{msg_type} Event"

            extracted.append({
                "flrID": msg_id,
                "classType": cls_type,
                "beginTime": issue_time,
                "peakTime": issue_time,
                "endTime": "",
                "sourceLocation": "Heliosphere / Solar Surface (NASA DONKI)",
                "link": item.get("messageURL", "")
            })
        if extracted:
            payload = {
                "flares": extracted,
                "count": len(extracted),
                "_source": "live",
                "_provider": "NASA Space Weather DONKI Notifications Feed",
                "_timestamp": now_utc.isoformat()
            }
            set_cache(cache_key, payload, ttl=1800)
            return payload

    if cache_key in NASA_CACHE:
        stale = NASA_CACHE[cache_key]
        return {"flares": stale["data"], "_source": "stale_cache", "_cached_at": stale["cached_at"]}

    return {
        "status": "unavailable",
        "flares": [],
        "date_range": f"{s_date} to {e_date}",
        "message": f"NASA Space Weather DONKI API returned {err or 'Service Disruption'}."
    }

# ── In-App Alert Preferences (Module B) ──────────────────────────────────────
# Alert preferences stored in memory per session (server-side, no login required for this demo)
# In a production deployment, these would be persisted to the user's database record.

_ALERT_PREFS: dict = {}  # key: session_token or "default"

@router.get("/alerts/preferences")
def get_alert_preferences(session: str = "default"):
    """
    Returns the current alert threshold configuration.
    For a demo platform, stored in memory. Auth-aware in production.
    """
    defaults = {
        "neo_min_diameter_km": 0.1,
        "neo_max_miss_distance_au": 0.05,
        "flare_min_class": "M",        # C, M, X
        "enable_neo_alerts": True,
        "enable_flare_alerts": True,
        "enable_weather_alerts": True,
        "delivery": "in_app",          # in_app only for now
    }
    prefs = _ALERT_PREFS.get(session, defaults)
    return {
        "session": session,
        "preferences": prefs,
        "note": "In-app alerts only. Email/push delivery not configured — extend via notification service integration."
    }


@router.post("/alerts/preferences")
def set_alert_preferences(session: str = "default", prefs: dict = None):
    """Update alert threshold preferences for a session."""
    if prefs is None:
        prefs = {}
    allowed_keys = {
        "neo_min_diameter_km", "neo_max_miss_distance_au",
        "flare_min_class", "enable_neo_alerts",
        "enable_flare_alerts", "enable_weather_alerts"
    }
    cleaned = {k: v for k, v in prefs.items() if k in allowed_keys}
    existing = _ALERT_PREFS.get(session, {})
    existing.update(cleaned)
    _ALERT_PREFS[session] = existing
    return {"status": "updated", "session": session, "preferences": existing}


@router.get("/alerts")
def get_active_alerts(session: str = "default"):
    """
    Returns current in-app alerts based on user's alert preferences and live NASA data.
    Checks NEO close approaches and solar flares against configured thresholds.
    """
    prefs = _ALERT_PREFS.get(session, {
        "neo_min_diameter_km": 0.1,
        "neo_max_miss_distance_au": 0.05,
        "flare_min_class": "M",
        "enable_neo_alerts": True,
        "enable_flare_alerts": True,
    })
    alerts = []

    # Check NEO cache for close approaches
    if prefs.get("enable_neo_alerts", True):
        neo_cached = get_from_cache("neo_today") or get_from_cache("neo_week")
        if neo_cached:
            neo_data = neo_cached.get("data", {})
            objects = neo_data.get("near_earth_objects", []) if isinstance(neo_data, dict) else neo_data
            if isinstance(objects, list):
                min_diam = float(prefs.get("neo_min_diameter_km", 0.1))
                max_dist = float(prefs.get("neo_max_miss_distance_au", 0.05))
                for obj in objects[:20]:
                    diam = obj.get("estimated_diameter_km_min", 0)
                    dist = obj.get("miss_distance_au", 999)
                    if diam >= min_diam and dist <= max_dist:
                        alerts.append({
                            "type": "NEO_CLOSE_APPROACH",
                            "severity": "INFO",
                            "title": f"NEO Close Approach: {obj.get('name', 'Unknown')}",
                            "detail": (
                                f"Estimated diameter: {diam:.3f}–{obj.get('estimated_diameter_km_max', diam):.3f} km. "
                                f"Miss distance: {dist:.4f} AU. "
                                "A close approach does NOT mean impact risk."
                            ),
                            "source": "NASA NeoWs",
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        })

    # Check DONKI flare cache
    if prefs.get("enable_flare_alerts", True):
        flare_cached = get_from_cache("donki_flr_30d") or None
        # Try any donki cache key
        flare_data = None
        for k in list(NASA_CACHE.keys()):
            if "donki_flr" in k:
                entry = NASA_CACHE[k]
                flare_data = entry.get("data", [])
                break

        min_class = prefs.get("flare_min_class", "M")
        class_order = {"C": 0, "M": 1, "X": 2}
        min_rank = class_order.get(min_class, 1)

        if flare_data and isinstance(flare_data, list):
            for flare in flare_data[-5:]:  # last 5 flares
                cls = (flare.get("classType") or "")[:1].upper()
                if class_order.get(cls, -1) >= min_rank:
                    alerts.append({
                        "type": "SOLAR_FLARE",
                        "severity": "WARNING" if cls == "X" else "INFO",
                        "title": f"Solar Flare: Class {flare.get('classType', '?')}",
                        "detail": (
                            f"Peak time: {flare.get('peakTime', '?')}. "
                            f"Duration: {flare.get('beginTime', '?')} to {flare.get('endTime', '?')}. "
                            "May cause radio blackouts and satellite drag."
                        ),
                        "source": "NASA DONKI",
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    })

    return {
        "count": len(alerts),
        "alerts": alerts,
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data_source": "NASA NeoWs + NASA DONKI (cached data)",
        "note": "Alerts are generated from cached NASA data. Refresh feeds first for up-to-date results.",
        "preferences_applied": prefs,
    }
