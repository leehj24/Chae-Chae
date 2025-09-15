# app.py (성능 및 메모리 최적화 최종본)

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

from recommend.config import *
import recommend.run_walk as run_walk_module
import recommend.run_transit as run_transit_module

# --- Flask 앱 설정 ---
BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-for-testing")
UPLOAD_FOLDER = str(BASE_DIR / "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
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

# --- 상수 및 초기 설정 ---
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
MAX_MSGS = 15
PATH_USER_REVIEWS = str(BASE_DIR / "_user_reviews.json")
_USER_REVIEWS_CACHE = {"data": None, "mtime": None}
_USER_REVIEWS_LOCK = threading.Lock()
PATH_USER_UPLOADS = str(BASE_DIR / "_user_uploads.json")
_USER_UPLOADS_CACHE = {"data": None, "mtime": None}
_USER_UPLOADS_LOCK = threading.Lock()
_IMAGE_CACHE: dict[str, dict] | None = None
_SESSION: requests.Session | None = None
DEFAULT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

PLACES_DF = None
FILTER_OPTIONS = None

# --- 데이터 로딩 및 최적화 ---

def _read_csv_robust(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc, usecols=usecols)
        except Exception:
            pass
    raise IOError(f"Failed to read CSV file with common encodings: {path}")

def _pick_column(df_cols: List[str], *names: str) -> str | None:
    low_cols = {c.lower().strip(): c for c in df_cols}
    for n in names:
        if n.lower() in low_cols:
            return low_cols[n.lower()]
    # Fallback for partial matches
    for c in df_cols:
        cl = c.lower().strip()
        for n in names:
            if n.lower() in cl:
                return c
    return None

def load_places_data() -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    [수정] 메모리 효율성을 극대화한 데이터 로딩 함수.
    1. 필요한 컬럼만 정확히 지정하여 로드.
    2. 문자열 컬럼은 'category' 타입으로 변환하여 메모리 절약.
    3. 숫자형 컬럼은 'float32'로 변환.
    """
    print("🚀 앱 시작! 관광지 데이터를 메모리에 로드하고 최적화합니다...")

    required_cols_map = {
        "title": ["title", "명칭", "장소명"],
        "addr1": ["addr1", "주소", "소재지"],
        "cat1": ["cat1", "대분류"],
        "cat3": ["cat3", "소분류"],
        "tour_score": ["tour_score", "관광지수"],
        "review_score": ["review_score", "인기도지수"],
        "mapx": ["mapx", "x", "lon", "경도"],
        "mapy": ["mapy", "y", "lat", "위도"],
        "firstimage": ["firstimage", "대표이미지", "이미지"],
    }
    
    try:
        temp_df_cols = pd.read_csv(PATH_TMF, encoding='utf-8', nrows=0).columns.tolist()
    except Exception:
        temp_df_cols = pd.read_csv(PATH_TMF, encoding='cp949', nrows=0).columns.tolist()

    cols_to_load = {}
    final_col_names = []
    for key, candidates in required_cols_map.items():
        found_col = _pick_column(temp_df_cols, *candidates)
        if found_col:
            cols_to_load[key] = found_col
            final_col_names.append(found_col)
        elif key not in ["cat3", "firstimage"]:
             raise KeyError(f"필수 컬럼 '{key}'에 해당하는 컬럼을 CSV에서 찾을 수 없습니다: {candidates}")

    df = _read_csv_robust(PATH_TMF, usecols=list(set(final_col_names)))
    
    rename_map = {v: k for k, v in cols_to_load.items()}
    df = df.rename(columns=rename_map)

    for c in ("cat3", "firstimage"):
        if c not in df.columns: df[c] = ""

    df['title'] = df['title'].astype(str).str.strip()
    df['addr1'] = df['addr1'].astype(str).str.strip()
    df = df.dropna(subset=['title', 'addr1'])
    df = df[df['title'] != '']
    df = df.drop_duplicates(subset=["title", "addr1"], keep="first").reset_index(drop=True)

    print("✅ CSV 로드 완료. 이제 데이터 타입을 최적화합니다...")
    
    # 주소에서 시도(sido) 정보 추출 후 category 타입으로 변환
    df['sido'] = df['addr1'].astype(str).str.split().str[0].astype('category')
    
    for col in df.columns:
        if 'score' in col or col in ['mapx', 'mapy']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        elif df[col].dtype == 'object':
            df[col] = df[col].fillna("").astype(str)
            # 유니크한 값이 적은 컬럼을 'category'로 변환
            if col in ['cat1']:
                num_unique_values = df[col].nunique()
                if num_unique_values / len(df) < 0.5:
                    df[col] = df[col].astype('category')

    print("✅ 데이터 타입 최적화 완료!")
    df.info(memory_usage='deep')
    
    sidos = sorted([s for s in df['sido'].cat.categories if s])
    cat1s = sorted([c for c in df['cat1'].cat.categories if c])
    
    all_cat3s = set()
    df['cat3'].astype(str).str.split(r'[,/|]').dropna().apply(
        lambda tags: all_cat3s.update(t.strip() for t in tags if t.strip())
    )
    cat3s = sorted(list(all_cat3s))
    
    filter_opts = {"sidos": sidos, "cat1s": cat1s, "cat3s": cat3s}
    
    print(f"✅ 데이터 로드 및 최적화 최종 완료! 총 {len(df):,}개의 장소.")
    return df, filter_opts


PLACES_DF, FILTER_OPTIONS = load_places_data()


# --- 유틸리티 함수들 ---

def _sort_key_from_param(s: str) -> tuple[str, str]:
    s = (s or "").strip().lower()
    return ("review_score", "인기도 지수") if s in {"popular", "review", "review_score", "인기도"} else ("tour_score", "관광 지수")

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

def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        user_imgs = [url_for('uploaded_file', filename=f) for f in user_uploads]

    # 중복 제거 및 순서 보장 (CSV 대표이미지 -> 카카오 이미지 -> 사용자 업로드)
    ordered: list[str] = []
    seen = set()
    for img_list in [csv_imgs, kakao_imgs, user_imgs]:
        for img_url in img_list:
            if img_url and img_url not in seen:
                seen.add(img_url)
                ordered.append(img_url)
    return ordered[:max_n]

def _kakao_geocode_coords(query: str, addr1: str = "") -> Optional[Tuple[float, float]]:
    if not KAKAO_API_KEY: return None
    _ensure_session()
    try:
        # 주소 우선 검색
        if addr1:
            r = _SESSION.get("https://dapi.kakao.com/v2/local/search/address.json", params={"query": addr1}, timeout=4)
            if r.ok and r.json().get("documents"):
                d = r.json()["documents"][0]
                return float(d["y"]), float(d["x"])
        # 키워드 검색
        q_kw = " ".join([_nfc(query), *_addr_region_tokens(addr1)])
        r = _SESSION.get("https://dapi.kakao.com/v2/local/search/keyword.json", params={"query": q_kw, "size": 1}, timeout=4)
        if r.ok and r.json().get("documents"):
            d = r.json()["documents"][0]
            return float(d["y"]), float(d["x"])
    except Exception: pass
    return None

def _get_kakao_place_url(title: str, x: str, y: str) -> Optional[str]:
    if not KAKAO_API_KEY or not x or not y: return None
    _ensure_session()
    params = {"query": title, "x": x, "y": y, "radius": 500, "sort": "accuracy", "size": 5}
    try:
        res = _SESSION.get("https://dapi.kakao.com/v2/local/search/keyword.json", params=params, timeout=3)
        if not res.ok: return None
        docs = res.json().get("documents", [])
        if not docs: return None
        # 이름이 가장 유사한 장소의 URL 반환
        clean_title = re.sub(r'[\(\)\[\]\s]', '', title)
        for place in docs:
            place_name = re.sub(r'[\(\)\[\]\s]', '', place.get("place_name", ""))
            if clean_title in place_name or place_name in clean_title:
                return place.get("place_url")
        return docs[0].get("place_url") # 유사한 장소가 없으면 첫 번째 결과 반환
    except requests.exceptions.RequestException:
        return None

def start_self_pinging():
    def self_ping_task():
        ping_url = os.environ.get("RENDER_EXTERNAL_URL")
        if not ping_url: print("⚠️ self-ping: RENDER_EXTERNAL_URL 환경 변수가 없어 셀프 핑을 건너뜁니다."); return
        interval_seconds = 600; print(f"🚀 self-ping: 셀프 핑 스레드를 시작합니다. 대상: {ping_url}, 주기: {interval_seconds}초")
        while True:
            try:
                time.sleep(interval_seconds)
                print(f"⏰ self-ping: 서버가 잠들지 않도록 스스로를 깨웁니다... (-> {ping_url})")
                requests.get(ping_url, timeout=10)
            except requests.exceptions.RequestException as e: print(f"❌ self-ping: 셀프 핑 실패: {e}")
            except Exception as e: print(f"❌ self-ping: 알 수 없는 오류 발생: {e}")
    ping_thread = threading.Thread(target=self_ping_task, daemon=True); ping_thread.start()


# --- 라우트(Routes) ---

@app.get("/")
def home():
    return render_template("home.html")

@app.get("/chat")
def index():
    _init_session_if_needed()
    return render_template("index.html", kakao_js_key=KAKAO_JS_KEY)

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
        engine = run_walk_module if params["transport_mode"] == "walk" else run_transit_module; itinerary_df = engine.run(**params); session["itinerary"] = _df_to_records(itinerary_df); session["state"] = "완료"; messages = session.get("messages", [])
        completion_html = "완료! 추천 일정을 아래에 표시했어요."
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
    _init_session_if_needed(); current_state = session.get("state"); state_flow = {"점수": "지역", "테마": "점수", "기간": "테마", "이동수단": "기간", "실행중": "이동수단", "완료": "이동수단", "오류": "이동수단"}
    prev_state = state_flow.get(current_state)
    if prev_state:
        messages = session.get("messages", [])
        if len(messages) >= 2: session["messages"] = messages[:-2]
        session["state"] = prev_state
    else: session.clear()
    return redirect(url_for("index"))

@app.get("/api/filter-options")
def api_filter_options():
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
    
    place_rows = PLACES_DF[(PLACES_DF['title'] == title) & (PLACES_DF['addr1'] == addr1)]
    firstimage_url = place_rows.iloc[0]['firstimage'] if not place_rows.empty else None
    
    all_images_before_upload = _get_all_images_for_place(title, addr1, firstimage_url, include_user_uploads=True)
    if len(all_images_before_upload) >= 4: return _json({"ok": False, "error": "이미지를 최대 4개까지 등록할 수 있습니다."}, 400)
    
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = secure_filename(f"{uuid.uuid4()}.{ext}")
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    uploads = _load_user_uploads()
    uploads.setdefault(key, []).append(filename)
    _save_user_uploads(uploads)
    
    all_images_after_upload = _get_all_images_for_place(title, addr1, firstimage_url, include_user_uploads=True)
    return _json({"ok": True, "images": all_images_after_upload})

@app.get("/api/places")
def api_places():
    """
    [수정] 메인 그리드 데이터 API. 이미지 로딩 로직을 제거하여 응답 속도를 대폭 향상시킵니다.
    - 동기적인 이미지 검색(_get_all_images_for_place)을 호출하지 않습니다.
    - CSV에 있는 대표이미지(firstimage) URL 하나만 반환합니다.
    """
    try:
        sido = request.args.get("sido"); cat1 = request.args.get("cat1"); cat3 = request.args.get("cat3"); query = request.args.get("q")
        
        # [핵심] .copy()를 제거하여 메모리 복사를 방지합니다.
        filtered_df = PLACES_DF
        
        if sido and sido != 'all':
            filtered_df = filtered_df[filtered_df['sido'] == sido]
        if cat1 and cat1 != 'all': filtered_df = filtered_df[filtered_df['cat1'] == cat1]
        if cat3 and cat3 != 'all': filtered_df = filtered_df[filtered_df['cat3'].astype(str).str.contains(cat3, na=False)]
        if query: 
            query_nfc = _nfc(query).lower()
            filtered_df = filtered_df[filtered_df['title'].astype(str).str.lower().str.contains(query_nfc, na=False)]

        sort = request.args.get("sort", "review"); order = request.args.get("order", "desc"); score_col, score_label = _sort_key_from_param(sort); sort_ascending = (order == 'asc')
        df_sorted = filtered_df.sort_values(by=[score_col], ascending=sort_ascending, na_position="last")
        
        page = max(1, int(request.args.get("page", 1))); per_page = max(1, min(100, int(request.args.get("per_page", 40))))
        total = len(df_sorted)
        total_pages = max(1, math.ceil(total / per_page))
        page = min(page, total_pages)
        start, end = (page - 1) * per_page, page * per_page
        
        view = df_sorted.iloc[start:end].copy()
        view["rank"] = range(start + 1, start + 1 + len(view))

        cols_to_return = [
            "rank", "title", "addr1", "cat1", "cat3", "firstimage",
            "tour_score", "review_score", "mapx", "mapy"
        ]
        existing_cols = [c for c in cols_to_return if c in view.columns]
        items_list = _df_to_records(view[existing_cols])

        return _json({
            "ok": True, "sort_label": score_label, "sort_col": score_col, 
            "total": total, "page": page, "per_page": per_page, 
            "total_pages": total_pages, "items": items_list,
        })
    except Exception as e: 
        print("❌ API Error in /api/places:"); traceback.print_exc(); 
        return _json({"ok": False, "error": str(e)}, 500)

@app.get("/api/place-media")
def api_place_media():
    """
    [추가] 특정 장소의 모든 이미지(CSV, Kakao, 사용자 업로드)를 가져오는 전용 API.
    프론트엔드에서 카드가 화면에 보일 때 이 API를 호출하여 이미지를 지연 로딩합니다.
    """
    title = _nfc(request.args.get("title", "")); addr1 = _nfc(request.args.get("addr1", ""))
    if not title or not addr1: return _json({"ok": False, "error": "title and addr1 are required."}, 400)
    
    # DataFrame에서 해당 장소를 찾아 'firstimage' URL을 가져옵니다.
    # astype('object')를 사용하여 category 타입 비교 경고를 피합니다.
    place_mask = (PLACES_DF['title'].astype('object') == title) & (PLACES_DF['addr1'].astype('object') == addr1)
    place_rows = PLACES_DF[place_mask]

    firstimage_url = place_rows.iloc[0]['firstimage'] if not place_rows.empty and 'firstimage' in place_rows.columns else None

    # 모든 이미지 소스를 결합하여 반환합니다. Kakao 이미지가 없으면 실시간으로 가져옵니다.
    images = _get_all_images_for_place(title, addr1, firstimage_url, max_n=4, include_user_uploads=True, auto_fetch_if_needed=True)
    coords = _kakao_geocode_coords(title, addr1)
    
    payload: Dict[str, Any] = {"ok": True, "images": images}
    if coords: payload["coords"] = {"y": coords[0], "x": coords[1]}
    return _json(payload)

@app.get("/api/place-details")
def api_place_details():
    _init_session_if_needed()
    title = _nfc(request.args.get("title", ""))
    addr1 = _nfc(request.args.get("addr1", ""))
    mapx = str(request.args.get("mapx", ""))
    mapy = str(request.args.get("mapy", ""))
    if not title or not addr1:
        return _json({"ok": False, "error": "title, addr1이 필요합니다."}, 400)
    
    kakao_url = _get_kakao_place_url(title, mapx, mapy)
    if kakao_url and kakao_url.startswith("http://"):
        kakao_url = kakao_url.replace("http://", "https://", 1)
        
    key = f"{title}|{addr1}"
    reviews_db = _load_user_reviews()
    place_reviews = reviews_db.get(key, {})
    ratings = place_reviews.get("ratings", {})
    reviews = place_reviews.get("reviews", {})
    
    avg_rating = sum(ratings.values()) / len(ratings) if ratings else 0
    total_ratings = len(ratings)
    
    my_rating = ratings.get(session.get('user_id'))
    my_review_data = next((r for r in reviews.values() if r.get('user_id') == session.get('user_id')), None)
    my_review_text = my_review_data.get('text') if my_review_data else None
    
    return _json({
        "ok": True, "kakao_url": kakao_url, "avg_rating": avg_rating,
        "total_ratings": total_ratings, "my_rating": my_rating, "my_review_text": my_review_text,
    })

@app.get("/api/get-reviews")
def get_reviews():
    title = _nfc(request.args.get("title", "")); addr1 = _nfc(request.args.get("addr1", ""))
    if not title or not addr1: return _json({"ok": False, "error": "필수 정보가 누락되었습니다."}, 400)
    
    key = f"{title}|{addr1}"
    reviews_db = _load_user_reviews()
    place_reviews_data = reviews_db.get(key, {}).get("reviews", {})
    return _json({"ok": True, "reviews": list(place_reviews_data.values())})

@app.post("/api/submit-review")
def api_submit_review():
    _init_session_if_needed()
    data = request.json
    title = _nfc(data.get("title", "")); addr1 = _nfc(data.get("addr1", ""))
    rating = data.get("rating"); review_text = (data.get("review_text") or "").strip()
    if not title or not addr1: return _json({"ok": False, "error": "필수 정보가 누락되었습니다."}, 400)

    key = f"{title}|{addr1}"
    user_id = session.get('user_id')
    reviews_db = _load_user_reviews()
    reviews_db.setdefault(key, {"ratings": {}, "reviews": {}})
    
    if rating is not None:
        try:
            rating_val = int(rating)
            if not (0 <= rating_val <= 5): raise ValueError()
            if rating_val == 0 and user_id in reviews_db[key].get("ratings", {}): # 별점 0은 삭제
                del reviews_db[key]["ratings"][user_id]
            elif rating_val > 0:
                 reviews_db[key].setdefault("ratings", {})[user_id] = rating_val
        except (ValueError, TypeError):
            return _json({"ok": False, "error": "별점은 0-5 사이의 정수여야 합니다."}, 400)

    if review_text:
        review_id = next((rid for rid, r in reviews_db[key].get("reviews", {}).items() if r.get('user_id') == user_id), str(uuid.uuid4()))
        reviews_db[key].setdefault("reviews", {})[review_id] = {
            "user_id": user_id, "text": review_text, "timestamp": datetime.now(timezone.utc).isoformat()
        }

    _save_user_reviews(reviews_db)
    return _json({"ok": True, "message": "후기가 저장되었습니다."})

@app.get("/img-proxy")
def img_proxy():
    url = request.args.get("u")
    if not url or not url.startswith("http"): return abort(400)
    try:
        _ensure_session()
        r = _SESSION.get(url, stream=True, timeout=10, headers={"Referer": ""})
        r.raise_for_status()
        headers = { "Content-Type": r.headers.get("Content-Type", "image/jpeg"), "Cache-Control": "public, max-age=604800" } # 7일 캐시
        return Response(r.iter_content(chunk_size=8192), status=r.status_code, headers=headers)
    except requests.exceptions.RequestException:
        return abort(502)


if __name__ == "__main__":
    start_self_pinging()
    # 디버그 모드는 메모리를 더 많이 사용하므로 운영 환경에서는 False로 설정하는 것이 좋습니다.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)