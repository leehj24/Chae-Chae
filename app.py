# app.py (KakaoTalk 메시지 보내기 적용 최종본)

import json
import math
import os
import re
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import unicodedata as ud
from flask import (Flask, Response, abort, flash, redirect, render_template,
                   request, send_from_directory, session, url_for)
from flask_session import Session
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from urllib.parse import urlencode

from recommend.config import *
import recommend.kakaotalk as kakaotalk
import recommend.run_transit as run_transit_module
import recommend.run_walk as run_walk_module
import recommend.naver_calendar as naver_calendar

# --- Flask 앱 설정 ---
BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-for-testing")
UPLOAD_FOLDER = str(BASE_DIR / "uploads")
# ▼▼▼ [추가] 공유 일정 파일을 저장할 폴더 경로 ▼▼▼
PATH_SHARED_ITINERARIES = str(BASE_DIR / "_shared_itineraries")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(PATH_SHARED_ITINERARIES).mkdir(exist_ok=True) # 폴더 생성
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
app.config.update(
    SESSION_TYPE="filesystem",
    SESSION_FILE_DIR=str((BASE_DIR / "_fs_sessions").resolve()),
    SESSION_PERMANENT=False,
    SESSION_USE_SIGNER=True,
    SESSION_COOKIE_NAME="srv_session",
    # ▼ Render/HTTPS에서 세션이 콜백까지 붙도록 명시
    PREFERRED_URL_SCHEME="https",
    SESSION_COOKIE_SECURE=True,       # HTTPS만
    SESSION_COOKIE_SAMESITE="Lax",    # OAuth 콜백 안전
    SESSION_COOKIE_HTTPONLY=True,
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
CONGESTION_DF = None
CONGESTION_COORDS_DF = None

# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ▼▼▼ [수정/추가] 혼잡도 최종등급 로딩/조회 로직 ▼▼▼
# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

def _nfc(s: str) -> str:
    return ud.normalize("NFC", str(s or "")).strip()

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    dLat, dLon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    a = math.sin(dLat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def load_congestion_final_data(path: str) -> Optional[pd.DataFrame]:
    global CONGESTION_COORDS_DF
    p = Path(path)
    if not p.exists():
        print(f"⚠️ 경고: 전국 혼잡도 파일을 찾을 수 없습니다: '{path}'")
        return None
    try:
        print("🚀 전국 혼잡도 최종 등급 데이터를 로드합니다...")
        df = pd.read_csv(p, encoding='utf-8')

        required_cols = ['시도', '시군구', '읍면동', '시간대', 'final_level', 'lat', 'lon']
        if not all(col in df.columns for col in required_cols):
            print(f"⛔️ {p.name} 파일에 필수 컬럼({', '.join(required_cols)})이 없어 근접 검색 기능을 비활성화합니다.")
            CONGESTION_COORDS_DF = None
        else:
            for col in ['시군구', '읍면동']:
                df[col] = df[col].apply(_nfc)
            
            coord_cols = ['시도', '시군구', '읍면동', 'lat', 'lon']
            CONGESTION_COORDS_DF = df[coord_cols].drop_duplicates().reset_index(drop=True)
            print(f"✅ 혼잡도 근접 검색을 위한 {len(CONGESTION_COORDS_DF):,}개 읍면동 좌표 데이터 생성 완료!")

        df.set_index(['시군구', '읍면동', '시간대'], inplace=True)
        df.sort_index(inplace=True)
        print(f"✅ 전국 혼잡도 데이터 로드 완료! 총 {len(df):,}개 행.")
        return df
    except Exception as e:
        print(f"⛔️ {p.name} 파일 로드 실패: {e}")
        traceback.print_exc()
        return None

def parse_address_for_key(address: str) -> tuple[str | None, str | None, str | None]:
    address = _nfc(address)
    parts = address.split()
    if not parts: return None, None, None
    
    sido_key = next((s for s in sido_map if parts[0].startswith(s)), None)
    if not sido_key: return None, None, None
    
    norm = [p.strip('()') for p in parts[1:]]
    sigungu = next((p for p in norm if p.endswith(('시', '군', '구'))), None)
    eupmyeondong = next((p for p in norm if p.endswith(('읍', '면', '동', '가')) and p != sigungu), None)
    
    return sido_key, sigungu, eupmyeondong

def get_congestion_level(sigungu: str, eupmyeondong: Optional[str], hour: int, df: pd.DataFrame) -> Optional[str]:
    if df is None: return None
    time_str = f"{hour:02d}시"
    try:
        if sigungu and eupmyeondong:
            level = df.loc[(sigungu, eupmyeondong, time_str), 'final_level']
            return level if isinstance(level, str) else level.iloc[0]
        if sigungu:
            rows = df.loc[sigungu]
            if not rows.empty:
                level = rows[rows.index.get_level_values('시간대') == time_str]['final_level']
                if not level.empty:
                    return level.iloc[0]
        return None
    except (KeyError, IndexError, TypeError):
        return None

def map_congestion_to_class(level: str) -> str:
    return {'매우 붐빔': 'very-high', '붐빔': 'high', '보통': 'medium', '여유': 'low'}.get(level, 'unknown')

def add_congestion_to_schedule(schedule_df: pd.DataFrame, congestion_df: pd.DataFrame) -> pd.DataFrame:
    if congestion_df is None:
        schedule_df['congestion_level'], schedule_df['congestion_class'] = '정보 없음', 'unknown'
        return schedule_df
        
    levels, classes = [], []
    for _, row in schedule_df.iterrows():
        if row.get('title') == '이동':
            levels.append(None)
            classes.append(None)
            continue
            
        addr, time_str = row.get('addr1'), row.get('start_time')
        if not all(isinstance(i, str) and i for i in [addr, time_str]):
            levels.append('정보 없음')
            classes.append('unknown')
            continue

        _, sigungu, eupmyeondong = parse_address_for_key(addr)
        level_text = '정보 없음'
        if sigungu:
            try:
                hour = int(time_str.split(':')[0])
                level_text = get_congestion_level(sigungu, eupmyeondong, hour, congestion_df) or '정보 없음'
            except (ValueError, TypeError):
                pass
        
        levels.append(level_text)
        classes.append(map_congestion_to_class(level_text))
        
    schedule_df['congestion_level'] = levels
    schedule_df['congestion_class'] = classes
    return schedule_df

# ░░░░░░ 혼잡도 관련 끝 ░░░░░░

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
        if n.lower() in low_cols: return low_cols[n.lower()]
    for c in df_cols:
        cl = c.lower().strip()
        for n in names:
            if n.lower() in cl: return c
    return None

def load_places_data() -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    print("🚀 앱 시작! 관광지 데이터를 메모리에 로드하고 최적화합니다...")
    required_cols_map = {
        "title": ["title"], "addr1": ["addr1"], "cat1": ["cat1"], "cat3": ["cat3"],
        "tour_score": ["tour_score"], "review_score": ["review_score"],
        "mapx": ["mapx"], "mapy": ["mapy"], "firstimage": ["firstimage"],
    }
    temp_df_cols = pd.read_csv(PATH_TMF, encoding='utf-8', nrows=0).columns.tolist()

    cols_to_load, final_col_names = {}, []
    for key, candidates in required_cols_map.items():
        found_col = _pick_column(temp_df_cols, *candidates)
        if found_col:
            cols_to_load[key] = found_col
            final_col_names.append(found_col)
        elif key not in ["cat3", "firstimage"]:
            raise KeyError(f"필수 컬럼 '{key}'을 찾을 수 없습니다: {candidates}")

    df = _read_csv_robust(PATH_TMF, usecols=list(set(final_col_names)))
    df = df.rename(columns={v: k for k, v in cols_to_load.items()})
    for c in ("cat3", "firstimage"):
        if c not in df.columns: df[c] = ""

    df['title'], df['addr1'] = df['title'].astype(str).str.strip(), df['addr1'].astype(str).str.strip()
    df = df.dropna(subset=['title', 'addr1'])
    df = df[df['title'] != '']
    df = df.drop_duplicates(subset=["title", "addr1"], keep="first").reset_index(drop=True)

    print("✅ CSV 로드 완료. 데이터 타입을 최적화합니다...")
    df['sido'] = df['addr1'].astype(str).str.split().str[0].astype('category')

    for col in df.columns:
        if 'score' in col or col in ['mapx', 'mapy']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        elif df[col].dtype == 'object':
            df[col] = df[col].fillna("").astype(str)
            if col in ['cat1'] and df[col].nunique() / len(df) < 0.5:
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
CONGESTION_DF = load_congestion_final_data(PATH_CONGESTION_FINAL)

def _sort_key_from_param(s: str) -> tuple[str, str]:
    s = (s or "").strip().lower()
    return ("review_score", "인기도 지수") if s in {"popular", "review", "review_score", "인기도"} else ("tour_score", "관광 지수")

def _trim_msgs():
    session["messages"] = session.get("messages", [])[-MAX_MSGS:]

def _json(payload: Dict[str, Any], status: int = 200) -> Response:
    return app.response_class(
        response=json.dumps(payload, ensure_ascii=False, allow_nan=False),
        status=status,
        mimetype="application/json"
    )

def _external_base_url() -> str:
    """
    콜백을 항상 동일 호스트로 고정.
    1) PUBLIC_BASE_URL (권장: https://your.domain)
    2) RENDER_EXTERNAL_URL (예: https://xxx.onrender.com)
    3) 최후수단: X-Forwarded-Proto/Host 기반 추론
    """
    base = os.environ.get("PUBLIC_BASE_URL") or os.environ.get("RENDER_EXTERNAL_URL")
    if base:
        return base.rstrip("/")
    scheme = request.headers.get("X-Forwarded-Proto", request.scheme)
    host = request.headers.get("X-Forwarded-Host", request.host)
    return f"{scheme}://{host}"

def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    clean = df.replace({np.nan: None})
    recs = clean.to_dict(orient="records")
    for r in recs:
        for k, v in list(r.items()):
            if isinstance(v, np.generic): r[k] = v.item()
    return recs

def _init_session_if_needed():
    if "state" not in session: session["state"] = "지역"
    if "messages" not in session or not session["messages"]:
        session["messages"] = [{"sender": "bot", "html": BOT_PROMPTS["지역"]}]
    if 'user_id' not in session: session['user_id'] = str(uuid.uuid4())

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
            _USER_REVIEWS_CACHE["data"], _USER_REVIEWS_CACHE["mtime"] = data, mtime
            return data
        except (json.JSONDecodeError, IOError):
            return {}

def _save_user_reviews(data):
    with _USER_REVIEWS_LOCK:
        try:
            Path(PATH_USER_REVIEWS).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            _USER_REVIEWS_CACHE["data"], _USER_REVIEWS_CACHE["mtime"] = None, None
        except Exception as e:
            print(f"❌ 에러: 사용자 후기 파일 저장 실패 - {e}")

def _load_user_uploads():
    with _USER_UPLOADS_LOCK:
        p = Path(PATH_USER_UPLOADS)
        if not p.exists(): return {}
        try:
            mtime = p.stat().st_mtime
            if _USER_UPLOADS_CACHE["data"] is not None and _USER_UPLOADS_CACHE["mtime"] == mtime:
                return _USER_UPLOADS_CACHE["data"]
            data = json.loads(p.read_text(encoding="utf-8"))
            _USER_UPLOADS_CACHE["data"], _USER_UPLOADS_CACHE["mtime"] = data, mtime
            return data
        except (json.JSONDecodeError, IOError):
            return {}

def _save_user_uploads(data):
    with _USER_UPLOADS_LOCK:
        try:
            Path(PATH_USER_UPLOADS).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            _USER_UPLOADS_CACHE["data"], _USER_UPLOADS_CACHE["mtime"] = None, None
        except Exception as e:
            print(f"❌ 에러: 사용자 업로드 파일 저장 실패 - {e}")

def _load_image_cache() -> dict:
    global _IMAGE_CACHE
    if _IMAGE_CACHE is not None: return _IMAGE_CACHE
    p = Path(PATH_KAKAO_IMAGE_CACHE)
    if not p.exists(): _IMAGE_CACHE = {}; return _IMAGE_CACHE
    try:
        _IMAGE_CACHE = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        _IMAGE_CACHE = {}
    return _IMAGE_CACHE

def _save_image_cache():
    if _IMAGE_CACHE is None: return
    try:
        Path(PATH_KAKAO_IMAGE_CACHE).write_text(json.dumps(_IMAGE_CACHE, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"❌ 에러: 이미지 캐시 파일 저장 실패 - {e}")

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
    except Exception:
        return []

def _images_for_place(title: str, addr1: str, max_n: int = 4) -> List[str]:
    cache = _load_image_cache()
    key = f"{_nfc(title)}|{_nfc(addr1)}"
    if key in cache:
        return cache[key].get("urls", [])[:max_n]
    return []

def _fetch_and_cache_images_live(title: str, addr1: str) -> list[str]:
    key, query = f"{_nfc(title)}|{_nfc(addr1)}", " ".join([title, *_addr_region_tokens(addr1)])
    urls, cache = _kakao_image_search(query, size=4), _load_image_cache()
    cache[key] = {"q": query, "urls": urls, "ts": int(datetime.now().timestamp())}
    _save_image_cache()
    return urls

def _get_all_images_for_place(title: str, addr1: str, firstimage_url: str | None, max_n: int = 4, include_user_uploads: bool = False, auto_fetch_if_needed: bool = False) -> List[str]:
    u = str(firstimage_url or '').strip()
    csv_imgs = [u] if u and u.lower().startswith('http') else []
    
    kakao_imgs = _images_for_place(title, addr1, max_n=4)
    if not kakao_imgs and auto_fetch_if_needed:
        kakao_imgs = _fetch_and_cache_images_live(title, addr1)
        
    user_imgs = []
    if include_user_uploads:
        upload_entries = _load_user_uploads().get(f"{_nfc(title)}|{_nfc(addr1)}", [])
        user_imgs = [url_for('uploaded_file', filename=entry.get('filename')) for entry in upload_entries if entry.get('filename')]

    ordered, seen = [], set()
    for img_url in sum([csv_imgs, kakao_imgs, user_imgs], []):
        if img_url and img_url not in seen:
            seen.add(img_url)
            ordered.append(img_url)
    return ordered[:max_n]

@app.post("/api/upload-image")
def upload_image():
    _init_session_if_needed()
    title, addr1 = request.form.get('title'), request.form.get('addr1')
    if 'file' not in request.files or not title or not addr1:
        return _json({"ok": False, "error": "필수 정보가 누락되었습니다."}, 400)
    file = request.files['file']
    if file.filename == '' or not _allowed_file(file.filename):
        return _json({"ok": False, "error": "허용되지 않는 파일 형식입니다."}, 400)

    key = f"{_nfc(title)}|{_nfc(addr1)}"
    
    user_id = session.get('user_id')
    uploads = _load_user_uploads()
    place_uploads = uploads.get(key, [])

    if any(entry.get('user_id') == user_id for entry in place_uploads):
        return _json({"ok": False, "error": "이미 이 장소에 사진을 업로드하셨습니다."}, 400)

    place_rows = PLACES_DF[(PLACES_DF['title'] == title) & (PLACES_DF['addr1'] == addr1)]
    firstimage_url = place_rows.iloc[0]['firstimage'] if not place_rows.empty else None
    
    current_images_count = len(_get_all_images_for_place(title, addr1, firstimage_url, include_user_uploads=True))
    if current_images_count >= 4:
        return _json({"ok": False, "error": "이미지를 최대 4개까지 등록할 수 있습니다."}, 400)

    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = secure_filename(f"{uuid.uuid4()}.{ext}")
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    new_entry = {"user_id": user_id, "filename": filename}
    uploads.setdefault(key, []).append(new_entry)
    _save_user_uploads(uploads)

    all_images = _get_all_images_for_place(title, addr1, firstimage_url, include_user_uploads=True)
    return _json({"ok": True, "images": all_images})


def _kakao_geocode_coords(query: str, addr1: str = "") -> Optional[Tuple[float, float]]:
    if not KAKAO_API_KEY: return None
    _ensure_session()
    try:
        if addr1:
            r = _SESSION.get("https://dapi.kakao.com/v2/local/search/address.json", params={"query": addr1}, timeout=4)
            if r.ok and r.json().get("documents"):
                d = r.json()["documents"][0]
                return float(d["y"]), float(d["x"])
        q_kw = " ".join([_nfc(query), *_addr_region_tokens(addr1)])
        r = _SESSION.get("https://dapi.kakao.com/v2/local/search/keyword.json", params={"query": q_kw, "size": 1}, timeout=4)
        if r.ok and r.json().get("documents"):
            d = r.json()["documents"][0]
            return float(d["y"]), float(d["x"])
    except Exception:
        pass
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
        clean_title = re.sub(r'[\(\)\[\]\s]', '', title)
        for place in docs:
            place_name = re.sub(r'[\(\)\[\]\s]', '', place.get("place_name", ""))
            if clean_title in place_name or place_name in clean_title:
                return place.get("place_url")
        return docs[0].get("place_url")
    except requests.exceptions.RequestException:
        return None

def start_self_pinging():
    def self_ping_task():
        ping_url = os.environ.get("RENDER_EXTERNAL_URL")
        if not ping_url:
            print("⚠️ self-ping: RENDER_EXTERNAL_URL 환경 변수가 없어 셀프 핑을 건너뜁니다.")
            return
        interval_seconds = 600
        print(f"🚀 self-ping: 셀프 핑 스레드를 시작합니다. 대상: {ping_url}, 주기: {interval_seconds}초")
        while True:
            try:
                time.sleep(interval_seconds)
                print(f"⏰ self-ping: 서버가 잠들지 않도록 스스로를 깨웁니다... (-> {ping_url})")
                requests.get(ping_url, timeout=10)
            except Exception as e:
                print(f"❌ self-ping: 오류 발생: {e}")
    threading.Thread(target=self_ping_task, daemon=True).start()

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
    _init_session_if_needed()
    state = session.get("state")
    messages = session.get("messages", [])

    if state == "지역":
        region = request.form.get("region", "").strip()
        if region:
            session["region"] = region
            messages.append({"sender": "user", "text": region})
            messages.append({"sender": "bot", "html": BOT_PROMPTS["점수"]})
            session["state"] = "점수"
    elif state == "점수":
        score = request.form.get("score", "").strip()
        if score in {"관광지수", "인기도지수"}:
            session["score_label"] = score
            messages.append({"sender": "user", "text": score})
            messages.append({"sender": "bot", "html": BOT_PROMPTS["테마"]})
            session["state"] = "테마"
    elif state == "테마":
        themes_str = request.form.get("themes", "").strip()
        if themes_str:
            themes = [t.strip() for t in themes_str.split(",") if t.strip()]
            session["cats"] = themes
            messages.append({"sender": "user", "text": ", ".join(themes)})
            messages.append({"sender": "bot", "html": BOT_PROMPTS["기간"]})
            session["state"] = "기간"
    elif state == "기간":
        start_date_str, end_date_str = request.form.get("start_date"), request.form.get("end_date")
        try:
            start = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
            days = (end - start).days + 1
            if 1 <= days <= 100:
                session["start_date"], session["end_date"], session["days"] = start_date_str, end_date_str, days
                user_text = f"{start_date_str} ~ {end_date_str} (총 {days}일)"
                messages.append({"sender": "user", "text": user_text})
                messages.append({"sender": "bot", "html": BOT_PROMPTS["이동수단"]})
                session["state"] = "이동수단"
        except (ValueError, TypeError):
            pass
    elif state == "이동수단":
        transport = request.form.get("transport", "").strip()
        if transport in {"walk", "transit"}:
            session["transport_mode"] = transport
            transport_text = "걷기" if transport == "walk" else "대중교통"
            messages.append({"sender": "user", "text": transport_text})
            messages.append({"sender": "bot", "html": BOT_PROMPTS["실행중"]})
            session["state"] = "실행중"
    
    session["messages"] = messages
    _trim_msgs()
    return redirect(url_for("index"))

@app.post("/do_generate")
def do_generate():
    try:
        _init_session_if_needed() # 세션 초기화 추가
        params = {
            "region": session.get("region"), "score_label": session.get("score_label"),
            "cats": session.get("cats"), "days": session.get("days"),
            "transport_mode": session.get("transport_mode")
        }
        if not all(params.values()):
            raise ValueError("필수 입력값이 누락되었습니다.")

        engine = run_walk_module if params["transport_mode"] == "walk" else run_transit_module
        itinerary_df = engine.run(**params)

        if itinerary_df.empty:
            itinerary_records = []
        else:
            itinerary_df = add_congestion_to_schedule(itinerary_df, CONGESTION_DF)
            itinerary_records = _df_to_records(itinerary_df)
            
            # ▼▼▼ [추가] 추천 결과에 사용자의 업로드 여부를 확인하는 로직 ▼▼▼
            user_id = session.get('user_id')
            user_uploads = _load_user_uploads()

            for item in itinerary_records:
                if item.get("title") != "이동":
                    # firstimage 정보 추가 (기존 로직)
                    title = item.get("title", "")
                    addr1 = item.get("addr1", "")
                    place_mask = (PLACES_DF['title'].astype('object') == title) & (PLACES_DF['addr1'].astype('object') == addr1)
                    place_rows = PLACES_DF[place_mask]
                    item["firstimage"] = place_rows.iloc[0]['firstimage'] if not place_rows.empty and 'firstimage' in place_rows.columns else ""
                    
                    # 업로드 여부 정보 추가 (새 로직)
                    key = f"{_nfc(title)}|{_nfc(addr1)}"
                    place_uploads = user_uploads.get(key, [])
                    item['user_has_uploaded'] = any(entry.get('user_id') == user_id for entry in place_uploads)
            # ▲▲▲ [추가 완료] ▲▲▲
        
        # ▼▼▼ [추가] 일정 저장 및 공유 ID 생성 로직 ▼▼▼
        share_id = str(uuid.uuid4())
        share_path = Path(PATH_SHARED_ITINERARIES) / f"{share_id}.json"
        
        share_data = {
            "itinerary": itinerary_records,
            "region": session.get("region"),
            "score_label": session.get("score_label"),
            "cats": session.get("cats"),
            "days": session.get("days"),
            "transport_mode": session.get("transport_mode"),
            "start_date": session.get("start_date"),
            "end_date": session.get("end_date")
        }
        with open(share_path, 'w', encoding='utf-8') as f:
            json.dump(share_data, f, ensure_ascii=False)
        
        session["share_id"] = share_id
        # ▲▲▲ [추가 완료] ▲▲▲

        session["itinerary"] = itinerary_records
        session["state"] = "완료"
        messages = session.get("messages", [])
        completion_html = "완료! 추천 일정을 아래에 표시했어요."
        if messages and "spinner" in messages[-1].get("html", ""):
            messages[-1]["html"] = completion_html
        else:
            messages.append({"sender": "bot", "html": completion_html})
        session["messages"] = messages
        return _json({"ok": True})
    except Exception as e:
        trace = traceback.format_exc(limit=4)
        print(f"Generation Error: {e}\n{trace}")
        session["state"] = "오류"
        session["messages"].append({"sender": "bot", "html": f"<strong>오류 발생:</strong><br><pre>{e}</pre>"})
        return _json({"ok": False, "error": str(e)}, 500)

# --- API Routes ---
@app.get("/reset_chat")
def reset_chat():
    session.clear()
    return redirect(url_for("index"))

@app.get("/go_back")
def go_back():
    _init_session_if_needed()
    current_state = session.get("state")
    state_flow = {
        "점수": "지역", "테마": "점수", "기간": "테마", "이동수단": "기간",
        "실행중": "이동수단", "완료": "이동수단", "오류": "이동수단"
    }
    prev_state = state_flow.get(current_state)
    if prev_state:
        messages = session.get("messages", [])
        if len(messages) >= 2:
            session["messages"] = messages[:-2]
        session["state"] = prev_state
    else:
        session.clear()
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

@app.get("/api/places")
def api_places():
    try:
        # ▼▼▼ [추가] 세션을 초기화하여 user_id를 가져옵니다. ▼▼▼
        _init_session_if_needed()
        
        sido, cat1, cat3, query = request.args.get("sido"), request.args.get("cat1"), request.args.get("cat3"), request.args.get("q")
        filtered_df = PLACES_DF
        if sido and sido != 'all':
            filtered_df = filtered_df[filtered_df['sido'] == sido]
        if cat1 and cat1 != 'all':
            filtered_df = filtered_df[filtered_df['cat1'] == cat1]
        if cat3 and cat3 != 'all':
            filtered_df = filtered_df[filtered_df['cat3'].astype(str).str.contains(cat3, na=False)]
        if query:
            filtered_df = filtered_df[filtered_df['title'].astype(str).str.lower().str.contains(_nfc(query).lower(), na=False)]

        sort, order = request.args.get("sort", "review"), request.args.get("order", "desc")
        score_col, score_label = _sort_key_from_param(sort)
        df_sorted = filtered_df.sort_values(by=[score_col], ascending=(order == 'asc'), na_position="last")

        page, per_page = max(1, int(request.args.get("page", 1))), max(1, min(100, int(request.args.get("per_page", 40))))
        total = len(df_sorted)
        total_pages = max(1, math.ceil(total / per_page))
        page = min(page, total_pages)
        start, end = (page - 1) * per_page, page * per_page

        view = df_sorted.iloc[start:end].copy()
        view["rank"] = range(start + 1, start + 1 + len(view))

        # ▼▼▼ [추가] 사용자의 업로드 여부를 확인하는 로직 ▼▼▼
        user_id = session.get('user_id')
        user_uploads = _load_user_uploads()
        
        def check_upload(row):
            key = f"{_nfc(row['title'])}|{_nfc(row['addr1'])}"
            place_uploads = user_uploads.get(key, [])
            return any(entry.get('user_id') == user_id for entry in place_uploads)

        view['user_has_uploaded'] = view.apply(check_upload, axis=1)
        # ▲▲▲ [추가 완료] ▲▲▲

        # ▼▼▼ [수정] 반환하는 컬럼 목록에 'user_has_uploaded' 추가 ▼▼▼
        cols_to_return = ["rank", "title", "addr1", "cat1", "cat3", "firstimage", "tour_score", "review_score", "mapx", "mapy", "user_has_uploaded"]
        items_list = _df_to_records(view[[c for c in cols_to_return if c in view.columns]])
        
        return _json({
            "ok": True, "sort_label": score_label, "sort_col": score_col,
            "total": total, "page": page, "per_page": per_page,
            "total_pages": total_pages, "items": items_list
        })
    except Exception as e:
        print("❌ API Error in /api/places:")
        traceback.print_exc()
        return _json({"ok": False, "error": str(e)}, 500)

@app.get("/api/place-media")
def api_place_media():
    title, addr1 = _nfc(request.args.get("title", "")), _nfc(request.args.get("addr1", ""))
    if not title or not addr1:
        return _json({"ok": False, "error": "title and addr1 are required."}, 400)

    place_mask = (PLACES_DF['title'].astype('object') == title) & (PLACES_DF['addr1'].astype('object') == addr1)
    place_rows = PLACES_DF[place_mask]
    firstimage_url = place_rows.iloc[0]['firstimage'] if not place_rows.empty and 'firstimage' in place_rows.columns else None

    images = _get_all_images_for_place(title, addr1, firstimage_url, max_n=4, include_user_uploads=True, auto_fetch_if_needed=True)
    
    coords = _kakao_geocode_coords(title, addr1)
    
    payload: Dict[str, Any] = {"ok": True, "images": images}
    if coords:
        payload["coords"] = {"y": coords[0], "x": coords[1]}
    else:
        payload["coords"] = None
        
    return _json(payload)

@app.get("/api/place-details")
def api_place_details():
    _init_session_if_needed()
    title, addr1, mapx, mapy = _nfc(request.args.get("title", "")), _nfc(request.args.get("addr1", "")), str(request.args.get("mapx", "")), str(request.args.get("mapy", ""))
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
        "total_ratings": total_ratings, "my_rating": my_rating, "my_review_text": my_review_text
    })

@app.get("/api/get-reviews")
def get_reviews():
    title, addr1 = _nfc(request.args.get("title", "")), _nfc(request.args.get("addr1", ""))
    if not title or not addr1:
        return _json({"ok": False, "error": "필수 정보가 누락되었습니다."}, 400)
    key = f"{title}|{addr1}"
    reviews_db = _load_user_reviews()
    place_reviews_data = reviews_db.get(key, {}).get("reviews", {})
    return _json({"ok": True, "reviews": list(place_reviews_data.values())})

@app.post("/api/submit-review")
def api_submit_review():
    _init_session_if_needed()
    data = request.json
    title, addr1, rating = _nfc(data.get("title", "")), _nfc(data.get("addr1", "")), data.get("rating")
    review_text = (data.get("review_text") or "").strip()
    if not title or not addr1:
        return _json({"ok": False, "error": "필수 정보가 누락되었습니다."}, 400)

    key, user_id = f"{title}|{addr1}", session.get('user_id')
    reviews_db = _load_user_reviews()
    reviews_db.setdefault(key, {"ratings": {}, "reviews": {}})

    if rating is not None:
        try:
            rating_val = int(rating)
            if not (0 <= rating_val <= 5): raise ValueError()
            if rating_val == 0 and user_id in reviews_db[key].get("ratings", {}):
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
        headers = {
            "Content-Type": r.headers.get("Content-Type", "image/jpeg"),
            "Cache-Control": "public, max-age=604800"
        }
        return Response(r.iter_content(chunk_size=8192), status=r.status_code, headers=headers)
    except requests.exceptions.RequestException:
        return abort(502)

@app.get("/api/congestion")
def get_congestion():
    addr1 = request.args.get('addr1', type=str)
    time_str = request.args.get('time', type=str)

    if CONGESTION_DF is None or not addr1 or not time_str:
        return _json({"level": "정보없음"})
    try:
        hour = int(time_str.split(':')[0])
    except (ValueError, IndexError):
        return _json({"level": "정보없음"})

    _, sigungu, eupmyeondong = parse_address_for_key(addr1)
    if sigungu:
        level = get_congestion_level(sigungu, eupmyeondong, hour, CONGESTION_DF)
        if level:
            return _json({"level": level})

    if CONGESTION_COORDS_DF is None or CONGESTION_COORDS_DF.empty:
        return _json({"level": "정보없음"})

    target_coords = _kakao_geocode_coords(query="", addr1=addr1)
    if not target_coords:
        return _json({"level": "정보없음"})
    target_lat, target_lon = target_coords

    coords_df = CONGESTION_COORDS_DF.copy()
    coords_df['distance_km'] = coords_df.apply(
        lambda row: haversine(target_lat, target_lon, row['lat'], row['lon']),
        axis=1
    )
    
    sorted_coords = coords_df.sort_values('distance_km')

    for radius in [5, 10, 15, 20]:
        nearby_places = sorted_coords[sorted_coords['distance_km'] <= radius]
        if not nearby_places.empty:
            closest = nearby_places.iloc[0]
            level = get_congestion_level(
                sigungu=closest['시군구'],
                eupmyeondong=closest['읍면동'],
                hour=hour,
                df=CONGESTION_DF
            )
            if level:
                return _json({"level": level})

    return _json({"level": "정보없음"})

@app.post("/api/add-naver-calendar")
def add_naver_calendar_api():
    payload = request.get_json(silent=True) or {}
    schedules = payload.get("schedules") or payload  # 단건/복수 모두 허용

    if not isinstance(schedules, list):
        schedules = [schedules]

    if not schedules or not isinstance(schedules[0], dict) or "title" not in schedules[0]:
        return _json({"status": "bad_request", "message": "유효한 일정 데이터가 필요합니다."}, 400)

    access_token = session.get("naver_access_token")
    if not access_token:
        # 인증 전: 추가할 일정 전부를 임시로 세션에 저장
        session["temp_schedule_data"] = schedules
        return _json({"status": "auth_required"})

    # 이미 인증되어 있으면 즉시 추가
    success = 0
    for sc in schedules:
        ok, body = naver_calendar.add_schedule(access_token, sc)
        if ok:
            success += 1
        else:
            print(f"[Naver Calendar] create failed: {body}")

    if success == len(schedules):
        return _json({"status": "success", "message": f"{success}개 일정 추가 완료"})
    elif success > 0:
        return _json({"status": "partial_success", "message": f"{len(schedules)}개 중 {success}개 추가"})
    else:
        # 토큰 만료 등
        session.pop("naver_access_token", None)
        return _json({"status": "auth_required", "message": "토큰 만료 또는 오류 – 재인증 필요"})

@app.get("/naver/auth")
def naver_auth_start():
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        flash("서버에 네이버 API 키가 설정되어 있지 않습니다.", "error")
        return redirect(url_for("index"))

    state = str(uuid.uuid4())
    session["naver_auth_state"] = state

    # ▼ 콜백 URL 고정 (도메인 섞임 방지)
    base_url = _external_base_url()  # e.g. https://your.app
    redirect_uri = f"{base_url}{url_for('naver_auth_callback', _external=False)}"
    session["naver_redirect_uri"] = redirect_uri  # 디버깅용

    params = {
        "response_type": "code",
        "client_id": NAVER_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "state": state,
    }

    # 로그로 즉시 진단 가능
    print("[NAVER AUTH] base_url=", base_url)
    print("[NAVER AUTH] redirect_uri=", redirect_uri)
    print("[NAVER AUTH] state(session)=", state)

    # 안전 인코딩
    req = requests.Request("GET", "https://nid.naver.com/oauth2.0/authorize", params=params)
    auth_url = req.prepare().url
    return redirect(auth_url)

@app.get("/naver/callback")
def naver_auth_callback():
    code = request.args.get("code")
    state = request.args.get("state")
    sess_state = session.get("naver_auth_state")

    # 디버깅 로그
    print("[NAVER CALLBACK] recv_state=", state)
    print("[NAVER CALLBACK] sess_state=", sess_state)
    print("[NAVER CALLBACK] host_url=", request.host_url)
    print("[NAVER CALLBACK] saved_redirect_uri=", session.get("naver_redirect_uri"))

    if not state or state != sess_state:
        flash("비정상적인 접근입니다.", "error")
        return redirect(url_for("index"))

    token_info = naver_calendar.get_access_token(code, state)
    if not token_info or "access_token" not in token_info:
        flash("네이버 인증에 실패했습니다.", "error")
        return redirect(url_for("index"))

    session["naver_access_token"] = token_info["access_token"]
    # 일회성 state는 바로 제거
    session.pop("naver_auth_state", None)

    schedules = session.get("temp_schedule_data")
    if schedules:
        success = 0
        for sc in schedules:
            ok, body = naver_calendar.add_schedule(token_info["access_token"], sc)
            if ok:
                success += 1
            else:
                print(f"[Naver Calendar] create failed: {body}")
        session.pop("temp_schedule_data", None)

        if success > 0:
            flash(f"네이버 일정 {success}개 추가 완료", "success")
        else:
            flash("일정 추가에 실패했습니다. 다시 시도하세요.", "error")
    else:
        flash("네이버 로그인 완료", "success")

    return redirect(url_for("index"))

# === KakaoTalk 연동 라우트 =========================================
@app.get("/kakaotalk/auth")
def kakao_auth():
    if not KAKAO_API_KEY:
        flash("카카오톡 API 키가 설정되지 않았습니다.")
        return redirect(url_for('index'))
    redirect_uri = url_for('kakao_oauth_callback', _external=True)
    auth_url = (
        f"https://kauth.kakao.com/oauth/authorize?"
        f"response_type=code&client_id={KAKAO_API_KEY}&redirect_uri={redirect_uri}"
        f"&scope=talk_message"
    )
    return redirect(auth_url)

@app.get("/kakaotalk/callback")
def kakao_oauth_callback():
    code = request.args.get('code')
    error_description = request.args.get('error_description')
    if not code:
        flash(f"카카오톡 연동에 실패했습니다: {error_description or '사용자가 동의하지 않았습니다.'}", "error")
        return redirect(url_for('index'))

    access_token = kakaotalk.get_access_token(code)
    if not access_token:
        flash("카카오톡 서버 인증에 실패했습니다. 잠시 후 다시 시도해주세요.", "error")
        return redirect(url_for('index'))

    itinerary = session.get("itinerary")
    share_id = session.get("share_id")

    if not itinerary or not share_id:
        flash("전송할 일정 정보가 없습니다. 새로운 추천을 받아주세요.", "error")
        return redirect(url_for('index'))
    
    # ▼▼▼ [수정] 외부 서버 환경에서 전체 URL을 더 안정적으로 생성하는 로직으로 변경 ▼▼▼
    # 기존: share_url = url_for('view_shared_itinerary', share_id=share_id, _external=True)

    # 1. Render.com 같은 서비스에 배포된 경우, 환경 변수에서 기본 URL을 가져옵니다.
    base_url = os.environ.get("RENDER_EXTERNAL_URL")

    # 2. 로컬 환경에서 테스트하는 경우를 위한 대비책
    if not base_url:
        # url_for를 사용해 현재 요청의 기본 URL(http://127.0.0.1:5000 등)을 알아냅니다.
        base_url = url_for('home', _external=True)
        # 만약 URL이 '/'로 끝나면 제거합니다. (예: http://test.com/ -> http://test.com)
        if base_url.endswith('/'):
            base_url = base_url[:-1]

    # 3. 기본 URL과 경로, share_id를 직접 조합하여 최종 URL을 만듭니다.
    share_url = f"{base_url}/share/{share_id}"
    
    print(f"🚀 생성된 공유 페이지 URL (수정된 방식): {share_url}")
    # ▲▲▲ [수정 완료] ▲▲▲

    success = kakaotalk.send_message_to_me(access_token, itinerary, share_url)
    if success:
        flash("카카오톡 메시지를 성공적으로 보냈습니다!", "success")
        print("✅ 카카오톡 메시지 전송 성공!")
    else:
        flash("카카오톡 메시지 전송 중 일부 실패했습니다.", "error")
        print("❌ 카카오톡 메시지 전송 중 일부 실패")

    return redirect(url_for('index'))

# ▼▼▼ [추가] 공유된 일정을 보여주는 새로운 페이지 라우트 ▼▼▼
@app.get("/share/<share_id>")
def view_shared_itinerary(share_id):
    # 보안을 위해 파일 이름으로 사용되기 전에 share_id를 정리합니다.
    safe_share_id = secure_filename(share_id)
    share_path = Path(PATH_SHARED_ITINERARIES) / f"{safe_share_id}.json"
    
    if not share_path.exists():
        abort(404) # 파일이 없으면 404 에러를 반환합니다.
    
    try:
        with open(share_path, 'r', encoding='utf-8') as f:
            share_data = json.load(f)
    except (IOError, json.JSONDecodeError):
        abort(500) # 파일 읽기 오류 시 500 에러를 반환합니다.

    return render_template("share.html", 
                           itinerary=share_data.get("itinerary", []),
                           share_info=share_data,
                           kakao_js_key=KAKAO_JS_KEY)
# ▲▲▲ [추가 완료] ▲▲▲

# ======================================================================

if __name__ == "__main__":
    # 배포환경 keep-alive용 셀프 핑 동작
    def _self_ping():
        url = os.environ.get("RENDER_EXTERNAL_URL")
        if not url:
            return
        while True:
            try:
                requests.get(url, timeout=5)
            except Exception:
                pass
            time.sleep(60 * 5)

    threading.Thread(target=_self_ping, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)