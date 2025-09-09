# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import traceback
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import unicodedata as ud
import re
import requests
import urllib.parse

from flask import (Flask, Response, redirect, render_template, request, session, url_for, abort)
from flask_session import Session

# =========================
# Config Import
# =========================
from recommend.config import (
    PATH_TMF,
    KAKAO_API_KEY,
    PATH_KAKAO_IMAGE_CACHE,
    KAKAO_JS_KEY,
)

# Import recommendation engine modules
import recommend.run_walk as run_walk_module
import recommend.run_transit as run_transit_module

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.secret_key = "dev-secret-key"

# ---------- Server-side Session Configuration ----------
app.config.update(
    SESSION_TYPE="filesystem",
    SESSION_FILE_DIR=str((BASE_DIR / "_fs_sessions").resolve()),
    SESSION_PERMANENT=False,
    SESSION_USE_SIGNER=True,
    SESSION_COOKIE_NAME="srv_session",
)
Path(app.config["SESSION_FILE_DIR"]).mkdir(parents=True, exist_ok=True)
Session(app)

# ---------- Common Utilities ----------
MAX_MSGS = 30

def _trim_msgs():
    session["messages"] = session.get("messages", [])[-MAX_MSGS:]

def _json(payload: Dict[str, Any], status: int = 200) -> Response:
    return app.response_class(
        response=json.dumps(payload, ensure_ascii=False, allow_nan=False),
        status=status,
        mimetype="application/json",
    )

def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    clean = df.replace({np.nan: None})
    recs = clean.to_dict(orient="records")
    for r in recs:
        for k, v in list(r.items()):
            if isinstance(v, np.generic):
                r[k] = v.item()
    return recs

def _nfc(s: str) -> str:
    return ud.normalize("NFC", str(s or "")).strip()

def _init_session_if_needed():
    session.setdefault("state", "지역")
    session.setdefault("messages", [])

# ─────────────────────────────────────────
# CSV Loading & Caching
# ─────────────────────────────────────────
_PLACES_CACHE = {"df": None, "mtime": None, "path": None}

def _read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try: return pd.read_csv(path, encoding=enc)
        except Exception: pass
    return pd.read_csv(path)

def _pick_column(df: pd.DataFrame, *names: str) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low: return low[n.lower()]
    for c in df.columns:
        cl = c.lower()
        for n in names:
            if n.lower() in cl: return c
    return None

def _load_places_df() -> pd.DataFrame:
    path = PATH_TMF
    p = Path(path)
    mtime = p.stat().st_mtime if p.exists() else None

    if _PLACES_CACHE["df"] is not None and _PLACES_CACHE["mtime"] == mtime:
        return _PLACES_CACHE["df"].copy()

    df = _read_csv_robust(path).copy()
    req = {
        "title": _pick_column(df, "title", "명칭", "place", "name"),
        "addr1": _pick_column(df, "addr1", "주소"),
        "cat1": _pick_column(df, "cat1", "대분류", "category1"),
        "tour_score": _pick_column(df, "tour_score", "관광지수", "tour-score"),
        "review_score": _pick_column(df, "review_score", "인기도지수", "review-score"),
    }
    if miss := [k for k, v in req.items() if v is None]:
        raise KeyError(f"Missing required CSV columns: {miss} / Found: {list(df.columns)}")

    opt = {
        "cat3": _pick_column(df, "cat3", "소분류", "category3"),
        "firstimage": _pick_column(df, "firstimage", "image", "img1", "thumbnail"),
    }
    rename_map = {v: k for k, v in req.items() if v}
    for k, v in opt.items():
        if v: rename_map[v] = k
    df = df.rename(columns=rename_map)

    for c in ("cat3", "firstimage"):
        if c not in df.columns: df[c] = ""
    for c in ("tour_score", "review_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("title", "addr1", "cat1", "cat3", "firstimage"):
        df[c] = df[c].astype(str).fillna("")

    df = df.drop_duplicates(subset=["title", "addr1"], keep="first").reset_index(drop=True)
    _PLACES_CACHE.update({"df": df.copy(), "mtime": mtime, "path": path})
    return df

def _sort_key_from_param(s: str) -> tuple[str, str]:
    s = (s or "").strip().lower()
    return ("review_score", "인기도 지수") if s in {"popular", "review", "review_score", "인기도"} else ("tour_score", "관광 지수")

# ─────────────────────────────────────────
# Kakao Image/Local API + Caching
# ─────────────────────────────────────────
_IMAGE_CACHE: dict[str, dict] | None = None
_SESSION: requests.Session | None = None
DEFAULT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def _load_image_cache() -> dict:
    global _IMAGE_CACHE
    if _IMAGE_CACHE is not None: return _IMAGE_CACHE
    p = Path(PATH_KAKAO_IMAGE_CACHE)
    if p.exists():
        try: _IMAGE_CACHE = json.loads(p.read_text(encoding="utf-8"))
        except Exception: _IMAGE_CACHE = {}
    else: _IMAGE_CACHE = {}
    return _IMAGE_CACHE

def _save_image_cache():
    try:
        p = Path(PATH_KAKAO_IMAGE_CACHE)
        p.write_text(json.dumps(_IMAGE_CACHE or {}, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception: pass

def _ensure_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({"User-Agent": DEFAULT_UA})
    if KAKAO_API_KEY and "Authorization" not in _SESSION.headers:
        _SESSION.headers.update({"Authorization": f"KakaoAK {KAKAO_API_KEY}"})

def _addr_region_tokens(addr1: str) -> List[str]:
    cand = re.findall(r"\b[\w가-힣]+(?:시|군|구)\b", _nfc(addr1)) or re.split(r"[,\s]+", _nfc(addr1))
    return [w for w in cand if w][:3]

def _kakao_image_search(query: str, size: int = 4) -> List[str]:
    if not KAKAO_API_KEY: return []
    _ensure_session()
    try:
        params = {"query": query, "sort": "accuracy", "page": 1, "size": max(1, min(10, int(size)))}
        r = _SESSION.get("https://dapi.kakao.com/v2/search/image", params=params, timeout=4)
        if not r.ok: return []
        docs = r.json().get("documents", []) or []
        urls = [d.get("image_url") for d in docs if str(d.get("image_url") or "").startswith("http")]
        return [u for u in urls if len(u) < 2000]
    except Exception: return []

# ✅ FASTER VERSION of image fetching
def _images_for_place(title: str, addr1: str, max_n: int = 4) -> List[str]:
    """
    Directly fetches images from Kakao API without server-side validation.
    This is much faster. The browser will handle any broken image links.
    """
    cache = _load_image_cache()
    key = f"{_nfc(title)}|{_nfc(addr1)}"
    
    # Check cache first
    if key in cache:
        return cache[key].get("urls", [])[:max_n]

    # If not in cache, call the API
    q = " ".join([_nfc(title), *_addr_region_tokens(addr1)])
    urls = _kakao_image_search(q, size=max_n)
    
    # Save to cache and return
    cache[key] = {"q": q, "urls": urls, "ts": int(datetime.now().timestamp())}
    _save_image_cache()
    
    return urls

def _kakao_geocode(query: str, addr1: str = "") -> Dict[str, Any] | None:
    if not KAKAO_API_KEY: return None
    _ensure_session()
    try:
        if addr1:
            r = _SESSION.get("https://dapi.kakao.com/v2/local/search/address.json", params={"query": addr1}, timeout=4)
            if r.ok and r.json().get("documents"):
                d = r.json()["documents"][0]
                return {"name": query, "x": float(d["x"]), "y": float(d["y"]), "source": "address"}
        
        q_kw = " ".join([_nfc(query), *_addr_region_tokens(addr1)])
        r = _SESSION.get("https://dapi.kakao.com/v2/local/search/keyword.json", params={"query": q_kw, "size": 1}, timeout=4)
        if r.ok and r.json().get("documents"):
            d = r.json()["documents"][0]
            return {"name": d.get("place_name") or query, "x": float(d["x"]), "y": float(d["y"]), "source": "keyword"}
    except Exception: pass
    return None

# ─────────────────────────────────────────
# Main Routes
# ─────────────────────────────────────────
@app.get("/")
def home():
    return render_template("home.html")

@app.get("/chat")
def index():
    _init_session_if_needed()
    kakao_js_key = KAKAO_JS_KEY
    return render_template("index.html", kakao_js_key=kakao_js_key)

@app.post("/chat")
def chat():
    # This route handles the step-by-step chat form submissions
    # ... (existing form handling logic) ...
    return redirect(url_for("index"))

@app.get("/reset_chat")
def reset_chat():
    session.clear()
    return redirect(url_for("index"))

@app.get("/go_back")
def go_back():
    """Navigates to the previous state in the chat conversation."""
    _init_session_if_needed()
    current_state = session.get("state")

    # Defines the conversation flow and what data to clear when going back
    state_flow = {
        "점수": {"prev": "지역", "clear": ["region"]},
        "테마": {"prev": "점수", "clear": ["score_label"]},
        "기간": {"prev": "테마", "clear": ["cats"]},
        "이동수단": {"prev": "기간", "clear": ["days", "start_date", "end_date"]},
        "실행중": {"prev": "이동수단", "clear": ["transport_mode", "pending_job"]},
        "완료": {"prev": "이동수단", "clear": ["transport_mode", "itinerary", "columns"]},
    }

    if current_state in state_flow:
        config = state_flow[current_state]
        prev_state = config["prev"]
        
        for key in config["clear"]:
            session.pop(key, None)
            
        messages = session.get("messages", [])
        if messages and len(messages) >= 2:
             session["messages"] = messages[:-2]

        session["state"] = prev_state
    else:
        session["state"] = "지역"
        session["messages"] = []

    return redirect(url_for("index"))


# ─────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────
@app.get("/api/places")
def api_places():
    try:
        df = _load_places_df()
        title_q = (request.args.get("title") or "").strip()
        addr1_q = (request.args.get("addr1") or "").strip()

        def process_view_to_items(view_df: pd.DataFrame) -> List[Dict]:
            items_list = []
            for _, r in view_df.iterrows():
                title, addr1 = _nfc(r["title"]), _nfc(r["addr1"])
                csv_imgs = [u for u in [str(r.get("firstimage") or "")] if u]
                kakao_imgs = _images_for_place(title, addr1, max_n=4)
                all_images = list(dict.fromkeys(csv_imgs + kakao_imgs))
                
                items_list.append({
                    "rank": int(r.get("rank", 0)), "title": title, "addr1": addr1,
                    "cat1":  str(r.get("cat1", "")), "cat3":  str(r.get("cat3", "")),
                    "images": all_images[:4],
                    "tour_score":   r.get("tour_score") if pd.notna(r.get("tour_score")) else None,
                    "review_score": r.get("review_score") if pd.notna(r.get("review_score")) else None,
                })
            return items_list

        if title_q and addr1_q:
            mask = (df["title"].apply(_nfc) == _nfc(title_q)) & (df["addr1"].apply(_nfc) == _nfc(addr1_q))
            view = df[mask]
            return _json({"ok": True, "items": process_view_to_items(view)})

        else:
            sort = request.args.get("sort", "review")
            page = max(1, int(request.args.get("page", 1)))
            per_page = max(1, min(100, int(request.args.get("per_page", 40))))
            score_col, score_label = _sort_key_from_param(sort)

            df_sorted = df.sort_values(by=[score_col], ascending=[False], na_position="last").reset_index(drop=True)
            total, total_pages = len(df_sorted), max(1, math.ceil(len(df_sorted) / per_page))
            page = min(page, total_pages)
            start, end = (page - 1) * per_page, page * per_page

            view = df_sorted.iloc[start:end].copy()
            view["rank"] = range(start + 1, start + 1 + len(view))
            
            return _json({
                "ok": True, "sort_label": score_label, "sort_col": score_col, "total": total, 
                "page": page, "per_page": per_page, "total_pages": total_pages, 
                "items": process_view_to_items(view),
            })
    except Exception as e:
        return _json({"ok": False, "error": str(e), "trace": traceback.format_exc(limit=4)}, 500)

@app.get("/api/geocode")
def api_geocode():
    title = (request.args.get("title") or "").strip()
    addr = (request.args.get("addr") or "").strip()
    if not title and not addr: 
        return _json({"ok": False, "error": "Query parameter 'title' or 'addr' is required."}, 400)
    
    coords = _kakao_geocode(title or addr, addr1=addr)
    if not coords:
        return _json({"ok": False, "error": "Geocoding failed. Location not found."})
    return _json({"ok": True, "result": coords})

@app.get("/img-proxy")
def img_proxy():
    url = request.args.get("u")
    if not url or not url.startswith("http"): return abort(400)
    try:
        _ensure_session()
        r = _SESSION.get(url, stream=True, timeout=8, headers={"Referer": ""})
        r.raise_for_status()
        headers = {"Content-Type": r.headers.get("Content-Type", "image/jpeg"), "Cache-Control": "public, max-age=86400"}
        return Response(r.iter_content(chunk_size=8192), status=r.status_code, headers=headers)
    except requests.exceptions.RequestException:
        return abort(502)

# ... other routes like /do_generate remain ...

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)