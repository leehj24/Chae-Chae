# app.py

import json, time, traceback, re, os, threading, math, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import unicodedata as ud
import requests
from werkzeug.utils import secure_filename
from flask import (Flask, Response, redirect, render_template, request, session, url_for, abort, send_from_directory)
from flask_session import Session

from recommend.config import (PATH_TMF, KAKAO_API_KEY, PATH_KAKAO_IMAGE_CACHE, KAKAO_JS_KEY)
import recommend.run_walk as run_walk_module
import recommend.run_transit as run_transit_module
# [삭제] filter.utils는 더 이상 사용하지 않습니다.
# from filter.utils import get_filter_options

from filter.cache_builder import update_cache_if_needed

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
app.secret_key = "dev-secret-key"

# --- (기존 설정 부분은 동일) ---
UPLOAD_FOLDER = str(BASE_DIR / "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
app.config.update(
    SESSION_TYPE="filesystem",
    SESSION_FILE_DIR=str((BASE_DIR / "_fs_sessions").resolve()),
    SESSION_PERMANENT=False,
    SESSION_USE_SIGNER=True,
    SESSION_COOKIE_NAME="srv_session",
)
Path(app.config["SESSION_FILE_DIR"]).mkdir(parents=True, exist_ok=True)
Session(app)
BOT_PROMPTS = {
    "지역": "안녕하세요! 😊<br /><b>어떤 지역</b>으로 여행 가실 건가요?",
    "점수": "어떤 기준으로 추천할까요? <b>관광지수 vs 인기도지수</b><br />하나만 선택해 주세요.",
    "테마": "좋아요! 이제 <b>원하는 테마를 최대 3개</b>까지 골라주세요.",
    "기간": "<b>여행 기간</b>을 선택해 주세요. 시작~종료 날짜를 고르면 <em>총 일수</em>가 자동 계산돼요.",
    "이동수단": "마지막으로, <b>어떤 이동수단</b>으로 맞출까요?",
    "실행중": "<div class='spinner'></div>모든 정보를 확인했어요.<br>이제 최적의 여행 경로를 만들고 있어요. 잠시만 기다려 주세요!",
}
sido_map = {
    '서울': '서울특별시', '서울특별시': '서울특별시', '서울시': '서울특별시', '부산': '부산광역시', '부산광역시': '부산광역시',
    '대구': '대구광역시', '대구광역시': '대구광역시', '인천': '인천광역시', '인천광역시': '인천광역시',
    '광주': '광주광역시', '광주광역시': '광주광역시', '대전': '대전광역시', '대전광역시': '대전광역시',
    '울산': '울산광역시', '울산광역시': '울산광역시', '울산시': '울산광역시', '세종': '세종특별자치시', '세종특별자치시': '세종특별자치시',
    '경기': '경기도', '경기도': '경기도', '강원': '강원', '강원도': '강원', '강원특별자치도': '강원',
    '충남': '충청남도', '충청남도': '충청남도', '충북': '충청북도', '충청북도': '충청북도',
    '전남': '전라남도', '전라남도': '전라남도', '전북': '전라북도', '전라북도': '전라북도', '전북특별자치도': '전라북도',
    '경남': '경상남도', '경상남도': '경상남도', '경북': '경상북도', '경상북도': '경상북도',
    '제주': '제주', '제주도': '제주', '제주특별자치도': '제주',
}
MAX_MSGS = 30

PATH_USER_REVIEWS = str(BASE_DIR / "_user_reviews.json")
_USER_REVIEWS_CACHE = {"data": None, "mtime": None}
_USER_REVIEWS_LOCK = threading.Lock()

def _trim_msgs():
    session["messages"] = session.get("messages", [])[-MAX_MSGS:]
def _json(payload: Dict[str, Any], status: int = 200) -> Response:
    return app.response_class(response=json.dumps(payload, ensure_ascii=False, allow_nan=False), status=status, mimetype="application/json")
def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    clean = df.replace({np.nan: None})
    recs = clean.to_dict(orient="records")
    for r in recs:
        for k, v in list(r.items()):
            if isinstance(v, np.generic): r[k] = v.item()
    return recs
def _nfc(s: str) -> str:
    return ud.normalize("NFC", str(s or "")).strip()
def _init_session_if_needed():
    if "state" not in session: session["state"] = "지역"
    if "messages" not in session or not session["messages"]: session["messages"] = [{"sender": "bot", "html": BOT_PROMPTS["지역"]}]
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())

# ▼▼▼ [변경] 메모리 문제 해결을 위한 핵심 수정 ▼▼▼

# [1. 전역 변수 선언]
# 앱 전체에서 공유할 데이터프레임과 필터 옵션을 저장할 변수를 만듭니다.
PLACES_DF = None
FILTER_OPTIONS = None

def _read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try: return pd.read_csv(path, encoding=enc)
        except Exception: pass
    raise IOError(f"Failed to read CSV file with common encodings: {path}")

def _pick_column(df: pd.DataFrame, *names: str) -> str | None:
    low = {c.lower(): c for c in df.columns};
    for n in names:
        if n.lower() in low: return low[n.lower()]
    for c in df.columns:
        cl = c.lower()
        for n in names:
            if n.lower() in cl: return c
    return None

# [2. 데이터 로딩 함수]
# 기존 _load_places_df 함수를 앱 시작 시 데이터를 로드하고 전처리하는 역할로 변경합니다.
def load_places_data() -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    앱 시작 시 한번만 호출되어 CSV 파일을 읽고,
    전처리된 데이터프레임과 필터 옵션을 반환합니다.
    """
    print("🚀 앱 시작! 관광지 데이터를 메모리에 로드합니다...")
    df = _read_csv_robust(PATH_TMF).copy()
    
    req = {
        "title": _pick_column(df, "title", "명칭", "place", "name"),
        "addr1": _pick_column(df, "addr1", "주소"),
        "cat1": _pick_column(df, "cat1", "대분류", "category1"),
        "tour_score": _pick_column(df, "tour_score", "관광지수", "tour-score"),
        "review_score": _pick_column(df, "review_score", "인기도지수", "review-score"),
        "mapx": _pick_column(df, "mapx", "x", "lon", "longitude"),
        "mapy": _pick_column(df, "mapy", "y", "lat", "latitude"),
    }

    if miss := [k for k, v in req.items() if v is None]:
        raise KeyError(f"Missing required CSV columns: {miss} / Found: {list(df.columns)}")

    opt = {"cat3": _pick_column(df, "cat3", "소분류", "category3"),"firstimage": _pick_column(df, "firstimage", "image", "img1", "thumbnail"),}
    rename_map = {v: k for k, v in req.items() if v}
    for k, v in opt.items():
        if v: rename_map[v] = k
    
    df = df.rename(columns=rename_map)
    for c in ("cat3", "firstimage"):
        if c not in df.columns: df[c] = ""
    for c in ("tour_score", "review_score", "mapx", "mapy"):
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in ("title", "addr1", "cat1", "cat3", "firstimage"):
        df[c] = df[c].astype(str).fillna("")

    df = df.drop_duplicates(subset=["title", "addr1"], keep="first").reset_index(drop=True)

    # 필터 옵션 생성 로직 추가
    sidos = sorted(df['addr1'].str.split().str[0].dropna().unique().tolist())
    cat1s = sorted(df['cat1'].dropna().unique().tolist())
    
    all_cat3s = set()
    df['cat3'].str.split(',').dropna().apply(lambda tags: all_cat3s.update(t.strip() for t in tags if t.strip()))
    cat3s = sorted(list(all_cat3s))
    
    filter_opts = {"sidos": sidos, "cat1s": cat1s, "cat3s": cat3s}
    
    print(f"✅ 데이터 로드 완료! 총 {len(df):,}개의 장소, 필터 옵션 생성 완료.")
    return df, filter_opts

# [3. 앱 시작 시 데이터 로드 실행]
# Flask 앱이 실행될 때 위 함수를 호출하여 전역 변수에 데이터를 할당합니다.
PLACES_DF, FILTER_OPTIONS = load_places_data()

# ▲▲▲ 변경점 끝 ▲▲▲


def _sort_key_from_param(s: str) -> tuple[str, str]:
    s = (s or "").strip().lower()
    return ("review_score", "인기도 지수") if s in {"popular", "review", "review_score", "인기도"} else ("tour_score", "관광 지수")
PATH_USER_UPLOADS = str(BASE_DIR / "_user_uploads.json")
_USER_UPLOADS_CACHE = {"data": None, "mtime": None}
_USER_UPLOADS_LOCK = threading.Lock()
def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- (기존 _load_user_reviews, _save_user_reviews 등 다른 헬퍼 함수들은 그대로 유지) ---
def _load_user_reviews():
    with _USER_REVIEWS_LOCK:
        p = Path(PATH_USER_REVIEWS)
        if not p.exists(): return {}
        try:
            mtime = p.stat().st_mtime
            if _USER_REVIEWS_CACHE["data"] is not None and _USER_REVIEWS_CACHE["mtime"] == mtime:
                return _USER_REVIEWS_CACHE["data"]
            data = json.loads(p.read_text(encoding="utf-8"))
            _USER_REVIEWS_CACHE["data"] = data
            _USER_REVIEWS_CACHE["mtime"] = mtime
            return data
        except (json.JSONDecodeError, IOError):
            return {}

def _save_user_reviews(data):
    with _USER_REVIEWS_LOCK:
        try:
            p = Path(PATH_USER_REVIEWS)
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            _USER_REVIEWS_CACHE["data"] = None
            _USER_REVIEWS_CACHE["mtime"] = None
        except Exception as e:
            print(f"❌ 에러: 사용자 후기 파일 저장 실패 - {e}")

def _load_user_uploads():
    with _USER_UPLOADS_LOCK:
        p = Path(PATH_USER_UPLOADS)
        if not p.exists(): return {}
        try:
            mtime = p.stat().st_mtime
            if _USER_UPLOADS_CACHE["data"] is not None and _USER_UPLOADS_CACHE["mtime"] == mtime: return _USER_UPLOADS_CACHE["data"]
            data = json.loads(p.read_text(encoding="utf-8"))
            _USER_UPLOADS_CACHE["data"] = data
            _USER_UPLOADS_CACHE["mtime"] = mtime
            return data
        except (json.JSONDecodeError, IOError): return {}
def _save_user_uploads(data):
    with _USER_UPLOADS_LOCK:
        try:
            p = Path(PATH_USER_UPLOADS)
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            _USER_UPLOADS_CACHE["data"] = None
            _USER_UPLOADS_CACHE["mtime"] = None
        except Exception as e:
            print(f"❌ 에러: 사용자 업로드 파일 저장 실패 - {e}")
_IMAGE_CACHE: dict[str, dict] | None = None
_SESSION: requests.Session | None = None
DEFAULT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
def _load_image_cache() -> dict:
    global _IMAGE_CACHE
    if _IMAGE_CACHE is not None: return _IMAGE_CACHE
    p = Path(PATH_KAKAO_IMAGE_CACHE)
    if not p.exists(): _IMAGE_CACHE = {}; return _IMAGE_CACHE
    try: _IMAGE_CACHE = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError): print(f"⚠️ 경고: '{PATH_KAKAO_IMAGE_CACHE}' 파일을 읽는 데 실패했습니다. 빈 캐시로 시작합니다."); _IMAGE_CACHE = {}
    return _IMAGE_CACHE
def _save_image_cache():
    global _IMAGE_CACHE
    if _IMAGE_CACHE is None: return
    try: p = Path(PATH_KAKAO_IMAGE_CACHE); p.write_text(json.dumps(_IMAGE_CACHE, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e: print(f"❌ 에러: 이미지 캐시 파일 저장 실패 - {e}")
def _ensure_session():
    global _SESSION
    if _SESSION is None: _SESSION = requests.Session(); _SESSION.headers.update({"User-Agent": DEFAULT_UA})
    if KAKAO_API_KEY and "Authorization" not in _SESSION.headers: _SESSION.headers.update({"Authorization": f"KakaoAK {KAKAO_API_KEY}"})
def _addr_region_tokens(addr1: str) -> List[str]:
    cand = re.findall(r"\b[\w가-힣]+(?:시|군|구)\b", _nfc(addr1)) or re.split(r"[,\s]+", _nfc(addr1)); return [w for w in cand if w][:3]
def _kakao_image_search(query: str, size: int = 4) -> List[str]:
    if not KAKAO_API_KEY: return []
    _ensure_session()
    try:
        params = {"query": query, "sort": "accuracy", "page": 1, "size": max(1, min(10, int(size)))}; r = _SESSION.get("https://dapi.kakao.com/v2/search/image", params=params, timeout=4)
        if not r.ok: return []
        docs = r.json().get("documents", []) or []; urls = [d.get("image_url") for d in docs if str(d.get("image_url") or "").startswith("http")]
        return [u for u in urls if len(u) < 2000]
    except Exception: return []
def _images_for_place(title: str, addr1: str, max_n: int = 4) -> List[str]:
    cache = _load_image_cache(); key = f"{_nfc(title)}|{_nfc(addr1)}"
    if key in cache: return cache[key].get("urls", [])[:max_n]
    return []
def _fetch_and_cache_images_live(title: str, addr1: str) -> list[str]:
    key = f"{_nfc(title)}|{_nfc(addr1)}"; query = " ".join([title, *_addr_region_tokens(addr1)]); urls = _kakao_image_search(query, size=4); cache = _load_image_cache(); cache[key] = {"q": query, "urls": urls, "ts": int(datetime.now().timestamp())}; _save_image_cache(); return urls

def _get_all_images_for_place(title: str, addr1: str, firstimage_url: str | None, max_n: int = 4, include_user_uploads: bool = False, auto_fetch_if_needed: bool = False) -> List[str]:
    key = f"{_nfc(title)}|{_nfc(addr1)}"
    csv_imgs: list[str] = []
    u = str(firstimage_url or '').strip()
    if u and u.lower().startswith('http'):
        csv_imgs.append(u)
    kakao_imgs = _images_for_place(title, addr1, max_n=4)
    if not kakao_imgs and auto_fetch_if_needed:
        kakao_imgs = _fetch_and_cache_images_live(title, addr1)
    user_imgs: list[str] = []
    if include_user_uploads:
        uploads_db = _load_user_uploads()
        user_uploads = uploads_db.get(key, [])
        user_imgs = [url_for('uploaded_file', filename=f) if not str(f).startswith('http') else str(f) for f in user_uploads]
    ordered: list[str] = []
    if csv_imgs:
        ordered.append(csv_imgs[0])
    ordered.extend(kakao_imgs)
    ordered.extend(user_imgs)
    ordered = [u for u in ordered if isinstance(u, str) and u.strip()]
    ordered = list(dict.fromkeys(ordered))[:max_n]
    return ordered

# --- (이하 다른 헬퍼 함수들도 그대로 유지) ---
def _kakao_geocode_coords(query: str, addr1: str = "") -> Optional[Tuple[float, float]]:
    if not KAKAO_API_KEY: return None
    _ensure_session()
    try:
        if addr1:
            r = _SESSION.get("https://dapi.kakao.com/v2/local/search/address.json", params={"query": addr1}, timeout=4)
            if r.ok and r.json().get("documents"): d = r.json()["documents"][0]; return float(d["y"]), float(d["x"])
        q_kw = " ".join([_nfc(query), *_addr_region_tokens(addr1)]); r = _SESSION.get("https://dapi.kakao.com/v2/local/search/keyword.json", params={"query": q_kw, "size": 1}, timeout=4)
        if r.ok and r.json().get("documents"): d = r.json()["documents"][0]; return float(d["y"]), float(d["x"])
    except Exception: pass
    return None
def _nearest_subway(lat, lon) -> Tuple[str, str]:
    _ensure_session()
    try:
        params = {"category_group_code": "SW8", "x": lon, "y": lat, "radius": 900, "size": 1, "sort": "distance"}; r = _SESSION.get("https://dapi.kakao.com/v2/local/search/category.json", params=params, timeout=4)
        if r.ok and (docs := r.json().get("documents")):
            d = docs[0]; name = _nfc(d.get("place_name")); raw = " ".join([name, _nfc(d.get("category_name", "")), _nfc(d.get("address_name", "")), _nfc(d.get("road_address_name", ""))]); m = re.search(r"(\d+)\s*호선", raw); return name, f"{m.group(1)}호선" if m else ""
    except Exception: pass
    return "", ""
def _nearest_bus(lat, lon) -> str:
    _ensure_session()
    try:
        for r in [900, 1200, 1500]:
            for kw in ["버스정류장", "정류장", "버스"]:
                params = {"query": kw, "x": lon, "y": lat, "radius": r, "size": 10, "sort": "distance"}; resp = _SESSION.get("https://dapi.kakao.com/v2/local/search/keyword.json", params=params, timeout=4)
                if resp.ok:
                    docs = sorted(resp.json().get("documents", []), key=lambda d: int(float(d.get("distance", "1e9"))))
                    for d in docs:
                        nm = _nfc(d.get("place_name"))
                        if any(k in nm for k in ["정류", "버스", "정류장", "정류소"]): return nm
                    if docs: return _nfc(docs[0].get("place_name"))
    except Exception: pass
    return ""
def _get_kakao_place_url(title: str, x: str, y: str) -> Optional[str]:
    if not KAKAO_API_KEY: return None
    _ensure_session()
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": title, "x": x, "y": y, "radius": 200, "sort": "accuracy"}
    try:
        res = _SESSION.get("https://dapi.kakao.com/v2/local/search/keyword.json", headers=headers, params=params, timeout=3)
        if not res.ok: return None
        docs = res.json().get("documents", [])
        if not docs: return None
        clean_title = title.replace('_', ' ')
        for place in docs:
            if clean_title in place.get("place_name", ""):
                return place.get("place_url")
        return docs[0].get("place_url")
    except requests.exceptions.RequestException:
        return None

def start_self_pinging():
    def self_ping_task():
        ping_url = os.environ.get("RENDER_EXTERNAL_URL")
        if not ping_url: print("⚠️ self-ping: RENDER_EXTERNAL_URL 환경 변수가 없어 셀프 핑을 건너뜁니다."); return
        interval_seconds = 600; print(f"🚀 self-ping: 셀프 핑 스레드를 시작합니다. 대상: {ping_url}, 주기: {interval_seconds}초")
        while True:
            try:
                time.sleep(interval_seconds); print(f"⏰ self-ping: 서버가 잠들지 않도록 스스로를 깨웁니다... (-> {ping_url})"); requests.get(ping_url, timeout=10)
            except requests.exceptions.RequestException as e: print(f"❌ self-ping: 셀프 핑 실패: {e}")
            except Exception as e: print(f"❌ self-ping: 알 수 없는 오류 발생: {e}")
    ping_thread = threading.Thread(target=self_ping_task, daemon=True); ping_thread.start()

# --- (라우트 핸들러들) ---
@app.get("/")
def home():
    return render_template("home.html")
@app.get("/chat")
def index():
    _init_session_if_needed()
    return render_template("index.html", kakao_js_key=KAKAO_JS_KEY)
# ... (기존 /chat, /do_generate 등 다른 라우트들은 그대로 유지) ...
@app.post("/chat")
def chat():
    _init_session_if_needed(); state = session.get("state"); messages = session.get("messages", [])
    if state == "지역":
        region = request.form.get("region", "").strip()
        if region: session["region"] = region; messages.append({"sender": "user", "text": region}); messages.append({"sender": "bot", "html": BOT_PROMPTS["점수"]}); session["state"] = "점수"
    elif state == "점수":
        score = request.form.get("score", "").strip()
        if score in {"관광지수", "인기도지수"}: session["score_label"] = score; messages.append({"sender": "user", "text": score}); messages.append({"sender": "bot", "html": BOT_PROMPTS["테마"]}); session["state"] = "테마"
    elif state == "테마":
        themes_str = request.form.get("themes", "").strip()
        if themes_str:
            themes = [t.strip() for t in themes_str.split(",") if t.strip()]; session["cats"] = themes; messages.append({"sender": "user", "text": ", ".join(themes)}); messages.append({"sender": "bot", "html": BOT_PROMPTS["기간"]}); session["state"] = "기간"
    elif state == "기간":
        start_date_str = request.form.get("start_date"); end_date_str = request.form.get("end_date")
        try:
            start = datetime.strptime(start_date_str, "%Y-%m-%d").date(); end = datetime.strptime(end_date_str, "%Y-%m-%d").date(); days = (end - start).days + 1
            if 1 <= days <= 100: session["days"] = days; user_text = f"{start_date_str} ~ {end_date_str} (총 {days}일)"; messages.append({"sender": "user", "text": user_text}); messages.append({"sender": "bot", "html": BOT_PROMPTS["이동수단"]}); session["state"] = "이동수단"
        except (ValueError, TypeError): pass
    elif state == "이동수단":
        transport = request.form.get("transport", "").strip()
        if transport in {"walk", "transit"}: session["transport_mode"] = transport; transport_text = "걷기" if transport == "walk" else "대중교통"; messages.append({"sender": "user", "text": transport_text}); messages.append({"sender": "bot", "html": BOT_PROMPTS["실행중"]}); session["state"] = "실행중"
    session["messages"] = messages; _trim_msgs(); return redirect(url_for("index"))
@app.post("/do_generate")
def do_generate():
    try:
        params = {"region": session.get("region"),"score_label": session.get("score_label"),"cats": session.get("cats"),"days": session.get("days"),"transport_mode": session.get("transport_mode"),}
        if not all(params.values()): raise ValueError("필수 입력값이 누락되었습니다.")
        engine = run_walk_module if params["transport_mode"] == "walk" else run_transit_module; itinerary_df = engine.run(**params); session["itinerary"] = _df_to_records(itinerary_df); session["state"] = "완료"; messages = session.get("messages", []); completion_html = "완료! 추천 일정을 아래에 표시했어요."
        if messages and messages[-1].get("sender") == "bot" and "spinner" in messages[-1].get("html", ""): messages[-1]["html"] = completion_html
        else: messages.append({"sender": "bot", "html": completion_html})
        session["messages"] = messages; return _json({"ok": True})
    except Exception as e:
        trace = traceback.format_exc(limit=4); print(f"Generation Error: {e}\n{trace}"); session["state"] = "오류"; session["messages"].append({"sender": "bot", "html": f"<strong>오류 발생:</strong><br><pre>{e}</pre>"}); return _json({"ok": False, "error": str(e)}, 500)
@app.get("/reset_chat")
def reset_chat():
    session.clear(); return redirect(url_for("index"))
@app.get("/go_back")
def go_back():
    _init_session_if_needed(); current_state = session.get("state"); state_flow = {"점수": {"prev": "지역"},"테마": {"prev": "점수"},"기간": {"prev": "테마"},"이동수단": {"prev": "기간"},"실행중": {"prev": "이동수단"},"완료": {"prev": "이동수단"},"오류": {"prev": "이동수단"},}
    if current_state in state_flow:
        messages = session.get("messages", [])
        if len(messages) >= 2: session["messages"] = messages[:-2]
        session["state"] = state_flow[current_state]["prev"]
    else: session.clear()
    return redirect(url_for("index"))

# ▼▼▼ [변경] /api/filter-options 라우트 수정 ▼▼▼
@app.get("/api/filter-options")
def api_filter_options():
    # 전역 변수에 저장된 필터 옵션을 바로 반환합니다.
    try:
        return _json({"ok": True, "options": FILTER_OPTIONS})
    except Exception as e:
        traceback.print_exc()
        return _json({"ok": False, "error": str(e)}, 500)

@app.get("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
@app.post("/api/upload-image")
def upload_image():
    title = request.form.get('title'); addr1 = request.form.get('addr1')
    if 'file' not in request.files or not title or not addr1: return _json({"ok": False, "error": "필수 정보가 누락되었습니다."}, 400)
    file = request.files['file']
    if file.filename == '' or not _allowed_file(file.filename): return _json({"ok": False, "error": "허용되지 않는 파일 형식입니다."}, 400)
    key = f"{_nfc(title)}|{_nfc(addr1)}"
    uploads = _load_user_uploads(); current_images = uploads.get(key, [])
    
    # [주의] firstimage_url을 찾기 위해 PLACES_DF를 조회해야 합니다.
    # 이 부분은 성능에 큰 영향이 없으므로 그대로 둡니다.
    place_row = PLACES_DF[(PLACES_DF['title'] == title) & (PLACES_DF['addr1'] == addr1)]
    firstimage_url = place_row.iloc[0]['firstimage'] if not place_row.empty else None
    
    all_images_before_upload = _get_all_images_for_place(title, addr1, firstimage_url, include_user_uploads=True, auto_fetch_if_needed=False)
    if len(all_images_before_upload) >= 4: return _json({"ok": False, "error": "이미지를 최대 4개까지 등록할 수 있습니다."}, 400)
    ext = file.filename.rsplit('.', 1)[1].lower(); filename = secure_filename(f"{uuid.uuid4()}.{ext}"); file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    current_images.append(filename); uploads[key] = current_images; _save_user_uploads(uploads)
    all_images_after_upload = _get_all_images_for_place(title, addr1, firstimage_url, include_user_uploads=True, auto_fetch_if_needed=False); 
    return _json({"ok": True, "images": all_images_after_upload})

# ▼▼▼ [변경] /api/places 라우트 수정 ▼▼▼
@app.get("/api/places")
def api_places():
    try:
        # 더 이상 파일을 읽지 않고, 메모리에 있는 PLACES_DF를 사용합니다.
        sido = request.args.get("sido"); cat1 = request.args.get("cat1"); cat3 = request.args.get("cat3"); query = request.args.get("q"); 
        
        # 필터링을 위해 매번 전체 데이터프레임을 복사하는 대신, 필터링 조건을 순차적으로 적용합니다.
        # pandas는 이런 연산을 매우 효율적으로 처리합니다.
        filtered_df = PLACES_DF
        
        if sido and sido != 'all':
            sido_val = sido_map.get(sido, sido)
            if '강원' in sido_val: sido_prefix = '강원'
            elif '제주' in sido_val: sido_prefix = '제주'
            else: sido_prefix = sido_val
            filtered_df = filtered_df[filtered_df['addr1'].str.startswith(sido_prefix, na=False)]
        
        if cat1 and cat1 != 'all': filtered_df = filtered_df[filtered_df['cat1'] == cat1]
        if cat3 and cat3 != 'all': filtered_df = filtered_df[filtered_df['cat3'].str.contains(cat3, na=False)]
        if query: 
            query_nfc = _nfc(query).lower()
            filtered_df = filtered_df[filtered_df['title'].str.lower().str.contains(query_nfc, na=False)]

        sort = request.args.get("sort", "review"); order = request.args.get("order", "desc"); score_col, score_label = _sort_key_from_param(sort); sort_ascending = (order == 'asc'); 
        df_sorted = filtered_df.sort_values(by=[score_col], ascending=sort_ascending, na_position="last")
        
        page = max(1, int(request.args.get("page", 1))); per_page = max(1, min(100, int(request.args.get("per_page", 40)))); 
        total = len(df_sorted); 
        total_pages = max(1, math.ceil(total / per_page)); 
        page = min(page, total_pages); 
        start, end = (page - 1) * per_page, page * per_page
        
        view = df_sorted.iloc[start:end].copy() # 최종 페이징 결과만 복사
        view["rank"] = range(start + 1, start + 1 + len(view))

        def process_view_to_items(view_df: pd.DataFrame) -> List[Dict]:
            items_list = []
            for _, r in view_df.iterrows():
                title, addr1 = _nfc(r["title"]), _nfc(r["addr1"])
                firstimage_url = r.get("firstimage")
                all_images = _get_all_images_for_place(
                    title, addr1, firstimage_url, max_n=4, include_user_uploads=True, auto_fetch_if_needed=True
                )
                items_list.append({
                    "rank": int(r.get("rank", 0)), "title": title, "addr1": addr1,
                    "cat1":  str(r.get("cat1", "")), "cat3":  str(r.get("cat3", "")),
                    "images": all_images,
                    "tour_score":   r.get("tour_score") if pd.notna(r.get("tour_score")) else None,
                    "review_score": r.get("review_score") if pd.notna(r.get("review_score")) else None,
                    "mapx": r.get("mapx") if pd.notna(r.get("mapx")) else None,
                    "mapy": r.get("mapy") if pd.notna(r.get("mapy")) else None,
                })
            return items_list

        return _json({"ok": True, "sort_label": score_label, "sort_col": score_col, "total": total, "page": page, "per_page": per_page, "total_pages": total_pages, "items": process_view_to_items(view),})
    except Exception as e: 
        print("❌ API Error in /api/places:"); traceback.print_exc(); 
        return _json({"ok": False, "error": str(e)}, 500)
# ▲▲▲ 변경점 끝 ▲▲▲

# --- (이하 나머지 라우트들은 그대로 유지) ---
@app.get("/api/place-media")
def api_place_media():
    title = _nfc(request.args.get("title", "")); addr1 = _nfc(request.args.get("addr1", ""))
    if not title or not addr1: return _json({"ok": False, "error": "title and addr1 are required."}, 400)
    
    # [주의] 여기도 firstimage_url을 찾아야 합니다.
    place_row = PLACES_DF[(PLACES_DF['title'] == title) & (PLACES_DF['addr1'] == addr1)]
    firstimage_url = place_row.iloc[0]['firstimage'] if not place_row.empty else None

    images = _get_all_images_for_place(title, addr1, firstimage_url, max_n=4, include_user_uploads=True, auto_fetch_if_needed=True); 
    coords = _kakao_geocode_coords(title, addr1); 
    payload: Dict[str, Any] = {"ok": True, "images": images}
    if coords: payload["coords"] = {"y": coords[0], "x": coords[1]}
    return _json(payload)

@app.get("/api/place-details")
def api_place_details():
    _init_session_if_needed()
    title = _nfc(request.args.get("title", ""))
    addr1 = _nfc(request.args.get("addr1", ""))
    mapx = request.args.get("mapx", "")
    mapy = request.args.get("mapy", "")
    if not title or not addr1:
        return _json({"ok": False, "error": "title, addr1이 필요합니다."}, 400)
    kakao_url = _get_kakao_place_url(title, mapx, mapy)
    key = f"{title}|{addr1}"
    reviews_db = _load_user_reviews()
    place_reviews = reviews_db.get(key, {})
    ratings = place_reviews.get("ratings", {})
    reviews = place_reviews.get("reviews", {})
    avg_rating = 0
    total_ratings = 0
    if ratings:
        total_ratings = len(ratings)
        avg_rating = sum(ratings.values()) / total_ratings if total_ratings > 0 else 0
    my_rating = ratings.get(session.get('user_id'))
    my_review_data = next((r for r in reviews.values() if r.get('user_id') == session.get('user_id')), None)
    my_review_text = my_review_data.get('text') if my_review_data else None
    return _json({
        "ok": True,
        "kakao_url": kakao_url,
        "avg_rating": avg_rating,
        "total_ratings": total_ratings,
        "my_rating": my_rating,
        "my_review_text": my_review_text,
    })

@app.get("/api/get-reviews")
def get_reviews():
    title = _nfc(request.args.get("title", ""))
    addr1 = _nfc(request.args.get("addr1", ""))
    if not title or not addr1:
        return _json({"ok": False, "error": "필수 정보가 누락되었습니다."}, 400)
    key = f"{title}|{addr1}"
    reviews_db = _load_user_reviews()
    place_reviews_data = reviews_db.get(key, {}).get("reviews", {})
    reviews_list = list(place_reviews_data.values())
    return _json({"ok": True, "reviews": reviews_list})

@app.post("/api/submit-review")
def api_submit_review():
    _init_session_if_needed()
    data = request.json
    title = _nfc(data.get("title", ""))
    addr1 = _nfc(data.get("addr1", ""))
    rating = data.get("rating")
    review_text = data.get("review_text", "").strip()
    if not title or not addr1:
        return _json({"ok": False, "error": "필수 정보가 누락되었습니다."}, 400)
    key = f"{title}|{addr1}"
    user_id = session.get('user_id')
    reviews_db = _load_user_reviews()
    if key not in reviews_db:
        reviews_db[key] = {"ratings": {}, "reviews": {}}
    if rating is not None:
        try:
            rating_val = int(rating)
            if not (1 <= rating_val <= 5): raise ValueError()
            reviews_db[key].setdefault("ratings", {})[user_id] = rating_val
        except (ValueError, TypeError):
            return _json({"ok": False, "error": "별점은 1-5 사이의 정수여야 합니다."}, 400)
    if review_text:
        existing_review_id = next((rid for rid, r in reviews_db[key].get("reviews", {}).items() if r.get('user_id') == user_id), None)
        review_data = {
            "user_id": user_id,
            "text": review_text,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        review_id = existing_review_id or str(uuid.uuid4())
        reviews_db[key].setdefault("reviews", {})[review_id] = review_data
    _save_user_reviews(reviews_db)
    return _json({"ok": True, "message": "후기가 저장되었습니다."})

@app.get("/api/geocode")
def api_geocode():
    title = (request.args.get("title") or "").strip(); addr = (request.args.get("addr") or "").strip()
    if not title and not addr: return _json({"ok": False, "error": "Query parameter 'title' or 'addr' is required."}, 400)
    coords = _kakao_geocode_coords(title or addr, addr1=addr)
    if not coords: return _json({"ok": False, "error": "Geocoding failed. Location not found."})
    return _json({"ok": True, "result": {"name": title or addr, "y": coords[0], "x": coords[1]}})
@app.get("/api/nearest-transit")
def api_nearest_transit():
    addr = (request.args.get("addr") or "").strip()
    if not addr: return _json({"ok": False, "error": "Query parameter 'addr' is required."}, 400)
    coords = _kakao_geocode_coords(addr, addr1=addr)
    if not coords: return _json({"ok": False, "error": f"Geocoding failed for address: {addr}"})
    lat, lon = coords
    subway_station, subway_line = _nearest_subway(lat, lon); bus_station = _nearest_bus(lat, lon)
    return _json({"ok": True, "result": {"addr": addr, "lat": lat, "lon": lon, "subway_station": subway_station, "subway_line": subway_line, "bus_station": bus_station,}})
@app.get("/img-proxy")
def img_proxy():
    url = request.args.get("u")
    if not url or not url.startswith("http"): return abort(400)
    try:
        _ensure_session(); r = _SESSION.get(url, stream=True, timeout=15, headers={"Referer": ""}); r.raise_for_status()
        headers = {"Content-Type": r.headers.get("Content-Type", "image/jpeg"),"Cache-Control": "public, max-age=86400"}
        return Response(r.iter_content(chunk_size=8192), status=r.status_code, headers=headers)
    except requests.exceptions.RequestException: return abort(502)

if __name__ == "__main__":
    cache_thread = threading.Thread(target=update_cache_if_needed, daemon=True)
    cache_thread.start()
    start_self_pinging()
    app.run(host="0.0.0.0", port=5000, debug=True)