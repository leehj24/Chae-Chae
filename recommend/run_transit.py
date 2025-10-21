# recommend/run_transit.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import re
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

import pandas as pd
import numpy as np
import unicodedata as ud

from recommend.config import (
    PATH_TMF,         # 관광지/POI CSV 경로
    KAKAO_API_KEY,    # 카카오 키워드검색 API 키 (없으면 지오코딩 생략)
    FAST_MODE,        # 빠른 모드 (지오코딩은 무시하고 항상 시도하도록 아래에서 오버라이드)
)

# ─────────────────────────────────────────────────────────────────────
# 정책/상수
# ─────────────────────────────────────────────────────────────────────
MAX_SPECIFIC_MATCH_PER_DAY = 3       # 하루(day_label) 정밀 교통 매핑 상한
DAILY_VISIT_TARGET       = 6         # 기본 목표 (하루 방문지 수)
DAILY_VISIT_HARD_MIN     = 6         # ★ 하드 하한: 최소 6개
DEFAULT_STAY_MIN         = 60        # 방문 체류(분)
DEFAULT_TRANSIT_KM_H     = 18.0      # 대중교통 평균속도(km/h)
PADDING_MIN              = 10        # 이동/대기 여유
MIN_MOVE_MIN             = 8         # 최소 이동시간(분)

# 반경 정책: 기본 3km, 부족하면 10km (days>=8 또는 cats 비었으면 10km부터 시작)
RADIUS_BASE_KM_DEFAULT   = 3.0
RADIUS_EXPAND_KM_DEFAULT = 10.0

# 1km 초과 시 양쪽 모두 대중교통 메타 필수
DIST_STRICT_KM           = 1.0

# 음식 규칙
MEAL_CAT = "음식"
MEAL_MAIN_KEYWORDS = {"한식", "중식", "일식", "서양식", "이색음식점"}
CAFE_KEYWORDS      = {"카페", "전통찻집"}

LUNCH_START, LUNCH_END   = 11 * 60, 13 * 60
DINNER_START, DINNER_END = 17 * 60, 20 * 60
NIGHT_AFTER              = 20 * 60

DEFAULT_WEIGHTS = [0.6, 0.3, 0.1]
HOP_COUNT_PER_DAY   = 2
FORCED_HOP_MINUTES  = 60
TIME_BUDGET = 8.0

# 시작 장소 최소 점수
STARTER_MIN_SCORE_PRIMARY = 80.0
STARTER_MIN_SCORE_SECOND  = 70.0

# ─────────────────────────────────────────────────────────────────────
# 권역/시도 표준화
# ─────────────────────────────────────────────────────────────────────
SIDO_MAP: Dict[str, str] = {
    # 서울/수도권
    '서울':'서울특별시','서울시':'서울특별시','서울특별자치시':'서울특별시','서울특별시':'서울특별시',
    '경기':'경기도','경기도':'경기도',
    '인천':'인천광역시','인천시':'인천광역시','인천광역시':'인천광역시',
    # 5대 광역
    '부산':'부산광역시','부산시':'부산광역시','부산광역시':'부산광역시',
    '대구':'대구광역시','대구시':'대구광역시','대구광역시':'대구광역시',
    '광주':'광주광역시','광주시':'광주광역시','광주광역시':'광주광역시',
    '대전':'대전광역시','대전시':'대전광역시','대전광역시':'대전광역시',
    '울산':'울산광역시','울산시':'울산광역시','울산광역시':'울산광역시',
    # 세종
    '세종':'세종특별자치시','세종시':'세종특별자치시','세종특별자치시':'세종특별자치시',
    # 강원
    '강원':'강원도','강원도':'강원도','강원특별자치도':'강원도',
    # 충청
    '충북':'충청북도','충청북도':'충청북도',
    '충남':'충청남도','충청남도':'충청남도',
    # 전라
    '전북':'전라북도','전라북도':'전라북도','전북특별자치도':'전라북도',
    '전남':'전라남도','전라남도':'전라남도',
    # 경상
    '경북':'경상북도','경상북도':'경상북도',
    '경남':'경상남도','경상남도':'경상남도',
    # 제주
    '제주':'제주특별자치도','제주도':'제주특별자치도','제주특별자치도':'제주특별자치도',
    '제주시':'제주특별자치도','서귀포시':'제주특별자치도',
}

MACRO_TO_SIDO: Dict[str, List[str]] = {
    '수도권': ['서울특별시', '경기도', '인천광역시'],
    '제주권': ['제주특별자치도'],
    '강원권': ['강원도'],
    '충청권': ['충청북도', '충청남도', '세종특별자치시', '대전광역시'],
    '호남권': ['전라북도', '전라남도', '광주광역시'],
    '영남권': ['경상북도', '경상남도', '부산광역시', '대구광역시', '울산광역시'],
}

_SIDO_TOKEN_RE = re.compile(
    r"(서울특별시|서울시|서울|부산광역시|부산시|부산|대구광역시|대구시|대구|인천광역시|인천시|인천|"
    r"광주광역시|광주시|광주|대전광역시|대전시|대전|울산광역시|울산시|울산|세종특별자치시|세종시|세종|"
    r"경기도|경기|강원특별자치도|강원도|강원|충청북도|충북|충청남도|충남|전북특별자치도|전라북도|전북|"
    r"전라남도|전남|경상북도|경북|경상남도|경남|제주특별자치도|제주도|제주시|서귀포시|제주)"
)

# ─────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────
def _nfc(s: Optional[str]) -> str:
    return ud.normalize("NFC", str(s)).strip() if s is not None else ""

def _float(x, default=np.nan):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default

# (스칼라) 하버사인
def _haversine(lat1, lon1, lat2, lon2) -> float:
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan
    lat1r, lon1r = math.radians(float(lat1)), math.radians(float(lon1))
    lat2r, lon2r = math.radians(float(lat2)), math.radians(float(lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = math.sin(dlat/2)**2 + math.cos(lat1r)*math.cos(lat2r)*math.sin(dlon/2)**2
    return 2 * 6371.0088 * math.asin(math.sqrt(a))

# (벡터) 하버사인: Series/ndarray ↔ 스칼라 중심점 (★ Series→float 오류 회피)
def _haversine_vec(lat_series, lon_series, lat0, lon0) -> np.ndarray:
    y = pd.to_numeric(lat_series, errors="coerce").to_numpy()
    x = pd.to_numeric(lon_series, errors="coerce").to_numpy()
    out = np.full_like(y, np.nan, dtype=float)
    if pd.isna(lat0) or pd.isna(lon0):
        return out
    mask = ~np.isnan(y) & ~np.isnan(x)
    if not np.any(mask):
        return out
    y1r = np.radians(y[mask]); x1r = np.radians(x[mask])
    y0r = math.radians(float(lat0)); x0r = math.radians(float(lon0))
    dlat = y0r - y1r
    dlon = x0r - x1r
    a = np.sin(dlat/2)**2 + np.cos(y1r)*np.cos(y0r)*np.sin(dlon/2)**2
    out[mask] = 2 * 6371.0088 * np.arcsin(np.sqrt(a))
    return out

def _estimate_transit_minutes(d_km: float) -> int:
    if pd.isna(d_km):
        return MIN_MOVE_MIN
    mins = max(MIN_MOVE_MIN, int(round((d_km / max(1e-6, DEFAULT_TRANSIT_KM_H)) * 60 + PADDING_MIN)))
    return mins

def _parse_hhmm(s: str) -> datetime:
    return datetime.strptime(s, "%H:%M")

def _fmt_hhmm(dtobj: datetime) -> str:
    return dtobj.strftime("%H:%M")

def _minutes_of_day(hhmm: str) -> int:
    dt = _parse_hhmm(hhmm); return dt.hour * 60 + dt.minute

def _time_slots_per_day(start_hhmm: str, end_hhmm: str, count: int) -> List[str]:
    t0 = _parse_hhmm(start_hhmm); t1 = _parse_hhmm(end_hhmm)
    total = (t1 - t0).total_seconds() / 60.0
    if count <= 0 or total <= 0: return []
    step = total / count
    out, cur = [], t0
    for _ in range(count):
        out.append(_fmt_hhmm(cur))
        cur = cur + timedelta(minutes=step)
    return out

def _between_time(day_start: str, day_end: str, s: str, e: str) -> bool:
    sdt = _parse_hhmm(s); edt = _parse_hhmm(e)
    ds  = _parse_hhmm(day_start); de = _parse_hhmm(day_end)
    return (ds <= sdt < de) and (ds < edt <= de) and (sdt < edt)

# ─────────────────────────────────────────────────────────────────────
# 권역 유틸
# ─────────────────────────────────────────────────────────────────────
def normalize_sido(name: str) -> Optional[str]:
    if not name: return None
    key = name.strip()
    return SIDO_MAP.get(key, SIDO_MAP.get(key.replace(' ', ''), None))

def extract_sido_from_addr(addr1: str) -> Optional[str]:
    if not addr1: return None
    m = _SIDO_TOKEN_RE.search(addr1)
    if not m: return None
    raw = m.group(1)
    return normalize_sido(raw)

def macro_from_sido(sido_std: str) -> Optional[str]:
    if not sido_std: return None
    for macro, sidos in MACRO_TO_SIDO.items():
        if sido_std in sidos:
            return macro
    return None

def attach_sido_macro_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sidos, macros = [], []
    for a in df['addr1'].astype(str):
        s = extract_sido_from_addr(a)
        sidos.append(s or '')
        macros.append(macro_from_sido(s) if s else '')
    df['__sido_norm'] = sidos
    df['__macro'] = macros
    return df

def infer_macro_from_text(region_text: str) -> Optional[str]:
    t = (region_text or '').strip()
    if any(k in t for k in ['수도권','서울','경기','인천']): return '수도권'
    if '제주' in t: return '제주권'
    if '강원' in t: return '강원권'
    if any(k in t for k in ['충청','세종','대전']): return '충청권'
    if any(k in t for k in ['전라','광주']): return '호남권'
    if any(k in t for k in ['경상','부산','대구','울산']): return '영남권'
    return None

# ─────────────────────────────────────────────────────────────────────
# 데이터 로드/표준화/필터
# ─────────────────────────────────────────────────────────────────────
def _read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ("utf-8","utf-8-sig","cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def _load_places() -> pd.DataFrame:
    df = _read_csv_robust(PATH_TMF)

    low = {c.lower(): c for c in df.columns}
    def col(*names, default=None):
        for n in names:
            if n in low: return low[n]
        return default

    title = col("title","name", default=None)
    addr1 = col("addr1","address","addr", default=None)
    cat1  = col("cat1", default=None)
    cat2  = col("cat2", default=None)
    cat3  = col("cat3", default=None)
    lat   = col("mapy","lat","latitude", default=None)
    lon   = col("mapx","lon","lng","longitude", default=None)
    score = col("인기도지수","관광지수","score", default=None)

    c_sub_st = col("closest_subway_station")
    c_sub_ln = col("closest_subway_line")
    c_bus_st = col("closest_bus_station")

    df["title"] = df[title].astype(str) if title else ""
    df["addr1"] = df[addr1].astype(str) if addr1 else ""
    df["cat1"]  = df[cat1] if cat1 else ""
    df["cat2"]  = df[cat2] if cat2 else ""
    df["cat3"]  = df[cat3] if cat3 else ""
    df["mapy"]  = pd.to_numeric(df[lat], errors="coerce") if lat else np.nan
    df["mapx"]  = pd.to_numeric(df[lon], errors="coerce") if lon else np.nan
    df["score"] = pd.to_numeric(df[score], errors="coerce").fillna(0.0) if score else 0.0

    df["closest_subway_station"] = df[c_sub_st].astype(str) if c_sub_st else ""
    df["closest_subway_line"]    = df[c_sub_ln].astype(str) if c_sub_ln else ""
    df["closest_bus_station"]    = df[c_bus_st].astype(str) if c_bus_st else ""

    df = df[[
        "title","addr1","cat1","cat2","cat3","mapx","mapy","score",
        "closest_subway_station","closest_subway_line","closest_bus_station"
    ]].copy()

    # 시·도/권역 붙이기
    df = attach_sido_macro_columns(df)
    return df

# ─────────────────────────────────────────────────────────────────────
# 중심좌표(지오코딩) + 반경 필터
# ─────────────────────────────────────────────────────────────────────
def _geocode_region_kakao(region: str, time_left: float) -> Optional[Tuple[float, float]]:
    """
    ✅ FAST_MODE와 무관하게, 시간이 남고 KAKAO_API_KEY가 있으면 항상 먼저 시도
    """
    if time_left < 1.5 or not KAKAO_API_KEY:
        return None
    import requests
    region = _nfc(region)
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": region, "size": 1}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=2.5)
        if r.status_code != 200:
            return None
        docs = r.json().get("documents", [])
        if not docs:
            return None
        return float(docs[0]["y"]), float(docs[0]["x"])  # (lat, lon)
    except Exception:
        return None

def _center_of_region(df: pd.DataFrame, region: str, time_left: float) -> Tuple[float, float]:
    # 1) 카카오 지오코딩 우선 (FAST_MODE 무시)
    pos = _geocode_region_kakao(region, time_left)
    if pos:
        return pos
    # 2) 텍스트 매칭된 후보의 중간값(백업)
    mask_addr = df["addr1"].astype(str).str.contains(region, na=False)
    sub = df[mask_addr].copy() if mask_addr.sum() >= 1 else df.copy()
    cy = float(pd.to_numeric(sub["mapy"], errors="coerce").median())
    cx = float(pd.to_numeric(sub["mapx"], errors="coerce").median())
    return cy, cx

def _filter_by_radius_km(df: pd.DataFrame, center_lat: float, center_lon: float, r_km: float) -> pd.DataFrame:
    df = df.copy()
    df["__dist0"] = _haversine_vec(df["mapy"], df["mapx"], center_lat, center_lon)
    return df[df["__dist0"] <= float(r_km)].copy()

def _pick_radius_km(days: int, cats: List[str]) -> Tuple[float, float]:
    """
    기본 3km, 확장 10km.
    - days >= 8 또는 cats 비었으면: 10km부터 시작
    """
    base = RADIUS_BASE_KM_DEFAULT
    expand = RADIUS_EXPAND_KM_DEFAULT
    if (days >= 8) or (not cats):
        base = RADIUS_EXPAND_KM_DEFAULT
    return base, expand

def _filter_candidates_by_radius_then_expand(df: pd.DataFrame, center_lat: float, center_lon: float,
                                             base_km: float, expand_km: float, need_n: int) -> pd.DataFrame:
    df_base = _filter_by_radius_km(df, center_lat, center_lon, base_km)
    if len(df_base) >= need_n:
        return df_base
    df_expand = _filter_by_radius_km(df, center_lat, center_lon, expand_km)
    return df_expand

# ─────────────────────────────────────────────────────────────────────
# 카테고리/음식 큐
# ─────────────────────────────────────────────────────────────────────
def _build_theme_queues(selected_df: pd.DataFrame, cats_norm: List[str]) -> Dict[str, List[int]]:
    queues: Dict[str, List[int]] = {c: [] for c in cats_norm}
    seen = set()
    for i, s in selected_df.iterrows():
        key = (_nfc(s.get("title", "")), _nfc(s.get("addr1", "")))
        if key in seen: continue
        text_cats = f"{s.get('cat1','')} {s.get('cat2','')} {s.get('cat3','')}"
        for c in cats_norm:
            if c and c in text_cats:
                queues[c].append(i)
                seen.add(key)
                break
    return queues

def _build_food_queues(selected_df: pd.DataFrame) -> Dict[str, List[int]]:
    meal_main, cafe = [], []
    for i, s in selected_df.iterrows():
        if str(s.get("cat1","")) != MEAL_CAT: continue
        c2 = str(s.get("cat2","")); c3 = str(s.get("cat3",""))
        bag = {t.strip() for t in (c2 + "," + c3).split(",") if t.strip()}
        if bag & CAFE_KEYWORDS: cafe.append(i)
        elif bag & MEAL_MAIN_KEYWORDS: meal_main.append(i)
        else: meal_main.append(i)
    return {"meal_main": meal_main, "cafe": cafe}

def _allocate_quota_for_day(cats_norm: List[str], want: int) -> Dict[str, int]:
    L = len(cats_norm)
    if L <= 0 or want <= 0: return {}
    weights = DEFAULT_WEIGHTS[:L]
    if len(weights) < L:
        tail = [max(0.0, (1.0 - sum(weights)) / (L - len(weights)))] * (L - len(weights))
        weights = weights + tail
    base = [1] * L
    remain = max(0, want - sum(base))
    quota_float = [w * remain for w in weights]
    quota_add = [int(q) for q in quota_float]
    diff = remain - sum(quota_add)
    fracs = sorted([(i, quota_float[i] - quota_add[i]) for i in range(L)], key=lambda x: x[1], reverse=True)
    for i, _ in fracs:
        if diff <= 0: break
        quota_add[i] += 1; diff -= 1
    final = [base[i] + quota_add[i] for i in range(L)]
    return {cats_norm[i]: final[i] for i in range(L)}

# ─────────────────────────────────────────────────────────────────────
# 대중교통 메타/포맷
# ─────────────────────────────────────────────────────────────────────
def _clean_line_str(line_raw: str) -> str:
    s = _nfc(line_raw)
    if not s or s.lower() == "nan": return ""
    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1]
        parts = [p.strip().strip("'").strip('"') for p in inner.split(',') if p.strip()]
        return parts[0] if parts else ""
    if ',' in s: return _nfc(s.split(',')[0])
    if '/' in s: return _nfc(s.split('/')[0])
    return s

def _format_subway_label(station: str, line: str) -> str:
    st = _nfc(station)
    ln = _clean_line_str(line)
    if not st: return ""
    return f"{st} {ln}" if ln else st

def _get_str(row: dict, key: str) -> str:
    v = (row or {}).get(key, "")
    s = _nfc(v)
    return "" if s.lower() == "nan" else s

def _ensure_stop_suffix(name: str) -> str:
    """버스 정류장 이름 뒤에 '정류장'이 없으면 붙여준다."""
    s = _nfc(name)
    if not s:
        return s
    if "정류장" in s:
        return s
    return f"{s} 정류장"

def _has_transit_meta(row: dict) -> bool:
    return bool(_get_str(row, "closest_subway_station") or _get_str(row, "closest_bus_station"))

def _specific_mapping_from_metadata(a: dict, b: dict) -> Optional[Tuple[str, str]]:
    # 지하철 우선
    a_sub, a_line = _get_str(a, "closest_subway_station"), _get_str(a, "closest_subway_line")
    b_sub, b_line = _get_str(b, "closest_subway_station"), _get_str(b, "closest_subway_line")
    if a_sub and b_sub:
        lab_a = _format_subway_label(a_sub, a_line)
        lab_b = _format_subway_label(b_sub, b_line)
        if lab_a == lab_b:
            return None  # 같은 역/호선이면 타이틀 폴백
        return f"{lab_a} 승차", f"{lab_b} 하차"

    # 버스 정류장
    a_bus, b_bus = _get_str(a, "closest_bus_station"), _get_str(b, "closest_bus_station")
    if a_bus and b_bus:
        lab_a = _ensure_stop_suffix(a_bus)
        lab_b = _ensure_stop_suffix(b_bus)
        if lab_a == lab_b:
            return None
        return f"{lab_a} 승차", f"{lab_b} 하차"

    return None

def _fallback_titles(from_title: str, to_title: str) -> Tuple[str, str]:
    return _nfc(from_title), _nfc(to_title)

# ─────────────────────────────────────────────────────────────────────
# 지역 블록 + 강제 점프
# ─────────────────────────────────────────────────────────────────────
def _area_key_from_addr(addr1: str) -> str:
    a = _nfc(addr1)
    if not a: return ""
    parts = a.split()
    if len(parts) >= 2: return f"{parts[0]} {parts[1]}"
    return parts[0]

def _group_by_area_and_inject_hops(day_visits: List[dict], k: int) -> List[dict]:
    if not day_visits: return day_visits
    enriched = []
    for v in day_visits:
        area = _area_key_from_addr(v.get("addr1","")) or _nfc(v.get("title","")).split()[0] if v.get("title") else ""
        vy, vx = _float(v.get("mapy")), _float(v.get("mapx"))
        enriched.append({**v, "_area": area, "_y": vy, "_x": vx})

    from collections import defaultdict
    buckets = defaultdict(list)
    for v in enriched:
        buckets[v["_area"]].append(v)

    def _centroid(vs):
        ys = [vv["_y"] for vv in vs if not pd.isna(vv["_y"])]
        xs = [vv["_x"] for vv in vs if not pd.isna(vv["_x"])]
        if ys and xs: return float(np.mean(ys)), float(np.mean(xs))
        return np.nan, np.nan

    areas = list(buckets.keys())
    if not areas: return day_visits
    areas_sorted = sorted(areas, key=lambda a: (-len(buckets[a]), a))
    start_area = areas_sorted[0]
    y0, x0 = _centroid(buckets[start_area])

    def _dist_from_start(a):
        y1, x1 = _centroid(buckets[a])
        if any(pd.isna([y0, x0, y1, x1])): return 1e9
        return _haversine(y0, x0, y1, x1)

    rest = sorted([a for a in areas if a != start_area], key=_dist_from_start)
    plan = [start_area] + rest
    plan = plan[:max(1, min(int(k), 3))]

    out: List[dict] = []
    first_block = True
    for area in plan:
        block = buckets[area]
        if first_block:
            out.extend(block); first_block = False
        else:
            if block:
                block[0] = {**block[0], "_force_move_min": FORCED_HOP_MINUTES}
                out.extend(block)

    unused = [a for a in areas if a not in plan]
    for area in unused:
        block = buckets[area]
        if not block: continue
        block[0] = {**block[0], "_force_move_min": FORCED_HOP_MINUTES}
        out.extend(block)

    for v in out:
        v.pop("_area", None); v.pop("_y", None); v.pop("_x", None)
    return out

# ─────────────────────────────────────────────────────────────────────
# 1km 초과 → 양쪽 모두 대중교통 메타 필수 (시퀀스 강제)
# ─────────────────────────────────────────────────────────────────────
def _enforce_distance_transit_rule(visits: List[dict]) -> List[dict]:
    if not visits: return visits
    pool = visits[:]
    seq: List[dict] = [pool.pop(0)]
    while pool:
        added = False
        for i in range(len(pool)):
            cand = pool[i]
            prev = seq[-1]
            d_km = _haversine(_float(prev.get("mapy")), _float(prev.get("mapx")),
                              _float(cand.get("mapy")), _float(cand.get("mapx")))
            if pd.isna(d_km) or d_km <= DIST_STRICT_KM:
                seq.append(cand); pool.pop(i); added = True; break
            # 1km 초과 → 양쪽 모두 메타 필요
            if _has_transit_meta(prev) and _has_transit_meta(cand):
                seq.append(cand); pool.pop(i); added = True; break
        if not added:
            # 남은 후보가 모두 불가 → 종료(보강은 상위 로직에서 처리)
            break
    return seq

# ─────────────────────────────────────────────────────────────────────
# 시작 장소 = 반경 우선 + 점수 하한 적용 + 가까움/점수 혼합 랭킹
# ─────────────────────────────────────────────────────────────────────
def _pick_start_place(df: pd.DataFrame, center_lat: float, center_lon: float,
                      min_score_primary: float = STARTER_MIN_SCORE_PRIMARY,
                      base_km: float = RADIUS_BASE_KM_DEFAULT,
                      expand_km: float = RADIUS_EXPAND_KM_DEFAULT) -> Optional[dict]:
    if df.empty: return None

    # 반경 우선(3km → 10km)
    cand = _filter_by_radius_km(df, center_lat, center_lon, base_km)
    if cand.empty:
        cand = _filter_by_radius_km(df, center_lat, center_lon, expand_km)
    if cand.empty:
        cand = df.copy()

    def _rank_pick(pool: pd.DataFrame) -> Optional[dict]:
        work = pool.copy()
        work["__dist0"] = _haversine_vec(work["mapy"], work["mapx"], center_lat, center_lon)
        s = pd.to_numeric(work["score"], errors="coerce").fillna(0.0)
        smin, smax = float(s.min()), float(s.max())
        work["s_norm"] = (s - smin) / (smax - smin) if smax > smin else 0.0
        d = pd.to_numeric(work["__dist0"], errors="coerce").fillna(0.0)
        dmax = float(d.max()) if len(d) else 1.0
        work["d_norm"] = 1.0 - (d / dmax if dmax > 0 else 0.0)
        work["rank_score"] = 0.6 * work["s_norm"] + 0.4 * work["d_norm"]
        r0 = work.sort_values(["rank_score","score"], ascending=[False, False]).head(1)
        return r0.iloc[0].to_dict() if not r0.empty else None

    # 1) score >= min_score_primary(기본 80)
    hi = cand[pd.to_numeric(cand["score"], errors="coerce") >= min_score_primary]
    pick = _rank_pick(hi) if not hi.empty else None
    if pick: return pick

    # 2) score >= 70 완화
    mid = cand[pd.to_numeric(cand["score"], errors="coerce") >= STARTER_MIN_SCORE_SECOND]
    pick = _rank_pick(mid) if not mid.empty else None
    if pick: return pick

    # 3) 제한 없이
    return _rank_pick(cand)

# ─────────────────────────────────────────────────────────────────────
# 하루 선발(시작장소 고정 + 음식시간대 + 가중치) → 1km/메타 시퀀싱 강제
# ─────────────────────────────────────────────────────────────────────
def _build_theme_queues_and_pick(
    df: pd.DataFrame, cats_norm: List[str], start_time: str, end_time: str,
    want: int, center_lat: float, center_lon: float
) -> List[dict]:
    want = max(1, int(want))
    if df.empty: return []

    # 점수 우선 정렬
    df_sorted = df.sort_values(["score"], ascending=False).reset_index(drop=True)
    theme_queues = _build_theme_queues(df_sorted, cats_norm)
    meal_enabled = (MEAL_CAT in cats_norm)
    food_queues = _build_food_queues(df_sorted) if meal_enabled else {"meal_main": [], "cafe": []}

    used = set()
    def _take(idx: int) -> Optional[pd.Series]:
        if idx < 0 or idx >= len(df_sorted): return None
        rec = df_sorted.loc[idx]
        key = (_nfc(rec.get("title","")), _nfc(rec.get("addr1","")))
        if key in used: return None
        used.add(key); return rec

    def _pop(q: List[int]) -> Optional[pd.Series]:
        while q:
            i = q.pop(0)
            rec = _take(i)
            if rec is not None: return rec
        return None

    quota = _allocate_quota_for_day(cats_norm, want)
    for c in list(quota.keys()):
        if not theme_queues.get(c): quota[c] = 0

    slots = _time_slots_per_day(start_time, end_time, want)
    picks: List[pd.Series] = []

    # 시작 장소: 중심에 가깝고 점수 높은 곳(반경/점수 하한 포함)
    starter = _pick_start_place(
        df_sorted, center_lat, center_lon,
        min_score_primary=STARTER_MIN_SCORE_PRIMARY,
        base_km=RADIUS_BASE_KM_DEFAULT,
        expand_km=RADIUS_EXPAND_KM_DEFAULT
    )
    if starter is not None:
        picks.append(pd.Series(starter))
        used.add((_nfc(starter.get("title","")), _nfc(starter.get("addr1",""))))
        # 카테고리 1개 소모 처리
        for c in cats_norm:
            text_cats = f"{starter.get('cat1','')} {starter.get('cat2','')} {starter.get('cat3','')}"
            if c and (c in text_cats) and quota.get(c, 0) > 0:
                quota[c] -= 1
                break

    cat_cursor, L = 0, max(1, len(cats_norm))
    for slot_hm in slots:
        if len(picks) >= want: break
        cur_min = _minutes_of_day(slot_hm)
        chosen: Optional[pd.Series] = None

        # 음식 시간대
        if meal_enabled and quota.get(MEAL_CAT, 0) > 0:
            if LUNCH_START <= cur_min < LUNCH_END or DINNER_START <= cur_min < DINNER_END:
                chosen = _pop(food_queues["meal_main"])
                if chosen is not None: quota[MEAL_CAT] -= 1
            elif cur_min >= NIGHT_AFTER:
                chosen = _pop(food_queues["cafe"])
                if chosen is not None: quota[MEAL_CAT] -= 1

        # 라운드로빈
        if chosen is None:
            for _ in range(L):
                c = cats_norm[cat_cursor % L]; cat_cursor += 1
                if quota.get(c, 0) <= 0: continue
                if c == MEAL_CAT: continue
                temp = _pop(theme_queues.get(c, []))
                if temp is not None:
                    quota[c] -= 1; chosen = temp; break

        # 폴백
        if chosen is None:
            for c in cats_norm:
                if c == MEAL_CAT: continue
                if quota.get(c, 0) <= 0: continue
                temp = _pop(theme_queues.get(c, []))
                if temp is not None:
                    quota[c] -= 1; chosen = temp; break

        if chosen is None:
            for i in range(len(df_sorted)):
                temp = _take(i)
                if temp is not None: chosen = temp; break

        if chosen is not None:
            picks.append(chosen)

    # 최소 한 번 카테고리 보강
    def _appeared(cat: str) -> bool:
        for r in picks:
            text = f"{r.get('cat1','')} {r.get('cat2','')} {r.get('cat3','')}"
            if cat and (cat in text): return True
        return False
    for c in cats_norm:
        if not theme_queues.get(c): continue
        if not _appeared(c):
            forced = None
            if c == MEAL_CAT:
                forced = _pop(food_queues["meal_main"]) or _pop(food_queues["cafe"])
            else:
                forced = _pop(theme_queues.get(c, []))
            if forced is not None:
                picks.append(forced)
                if len(picks) > want: picks = picks[:want]

    # 1km/대중교통 규칙으로 시퀀스 재정렬
    seq = [r.to_dict() for r in picks]
    seq = _enforce_distance_transit_rule(seq)
    return seq[:want]

# ─────────────────────────────────────────────────────────────────────
# 이동 행 생성
# ─────────────────────────────────────────────────────────────────────
def _build_day_rows(day: int, visits: List[dict], start_time: str, end_time: str) -> List[dict]:
    day_label = f"{day}일차"
    rows: List[dict] = []
    cur_time = _parse_hhmm(start_time)
    specific_used = 0

    for i, v in enumerate(visits):
        st = cur_time
        en = st + timedelta(minutes=DEFAULT_STAY_MIN)
        if not _between_time(start_time, end_time, _fmt_hhmm(st), _fmt_hhmm(en)): break

        rows.append({
            "day_label": day_label, "day": day,
            "start_time": _fmt_hhmm(st), "end_time": _fmt_hhmm(en),
            "title": _nfc(v.get("title","")), "addr1": _nfc(v.get("addr1","")),
            "cat1": _nfc(v.get("cat1","")), "cat2": _nfc(v.get("cat2","")), "cat3": _nfc(v.get("cat3","")),
            "출발지": "", "교통편1": "", "교통편2": "", "도착지": "",
            "final_score": float(_float(v.get("score"), 0.0)),
            "distance_from_prev_km": np.nan, "move_min": 0, "stay_min": DEFAULT_STAY_MIN,
            "mapx": _float(v.get("mapx")), "mapy": _float(v.get("mapy"))
        })
        cur_time = en

        if i + 1 < len(visits):
            a, b = v, visits[i+1]
            d_km = _haversine(_float(a.get("mapy")), _float(a.get("mapx")),
                              _float(b.get("mapy")), _float(b.get("mapx")))
            forced = int(b.get("_force_move_min", 0)) if isinstance(b, dict) else 0
            move_min = forced if forced > 0 else _estimate_transit_minutes(d_km)

            t_specific = _specific_mapping_from_metadata(a, b)
            if t_specific and specific_used < MAX_SPECIFIC_MATCH_PER_DAY:
                t1, t2 = t_specific
                specific_used += 1
            else:
                t1, t2 = _fallback_titles(_nfc(a.get("title","")), _nfc(b.get("title","")))

            st_m = cur_time
            en_m = st_m + timedelta(minutes=move_min)
            if not _between_time(start_time, end_time, _fmt_hhmm(st_m), _fmt_hhmm(en_m)): break

            rows.append({
                "day_label": day_label, "day": day,
                "start_time": _fmt_hhmm(st_m), "end_time": _fmt_hhmm(en_m),
                "title": "이동", "addr1": "",
                "cat1":"", "cat2":"", "cat3":"",
                "출발지": _nfc(a.get("addr1") or a.get("title","")),
                "교통편1": t1, "교통편2": t2,
                "도착지": _nfc(b.get("addr1") or b.get("title","")),
                "final_score": np.nan,
                "distance_from_prev_km": round(float(d_km),2) if not pd.isna(d_km) else np.nan,
                "move_min": int(move_min), "stay_min": 0,
                "mapx": np.nan, "mapy": np.nan
            })
            cur_time = en_m
    return rows

# ─────────────────────────────────────────────────────────────────────
# 풀 확장(요청 순서 준수: 반경 → 권역 → 기타)
# ─────────────────────────────────────────────────────────────────────
def _widen_pool(df_all: pd.DataFrame, region: str, used_global: set, need_n: int,
                center_lat: float, center_lon: float, days: int, cats: List[str]) -> pd.DataFrame:
    """요청대로 반경 우선(3km → 부족 시 10km) → 그 다음 권역 → 그 다음 전체 반경/전체 상위."""
    def _mask_not_used(df):
        if not used_global: return df
        def _key_row_series(r: pd.Series) -> tuple:
            return (_nfc(r.get("title","")), _nfc(r.get("addr1","")))
        mask = ~df.apply(lambda r: _key_row_series(r) in used_global, axis=1)
        return df[mask]

    base_km, expand_km = _pick_radius_km(days, cats)

    # 1) (반경 3/10km) 후보 확보
    df_r = _filter_candidates_by_radius_then_expand(df_all, center_lat, center_lon, base_km, expand_km, need_n)
    df_r = _mask_not_used(df_r)

    # 2) 권역 일치 필터 (있다면)
    macro = infer_macro_from_text(region)
    if macro:
        df_r_macro = df_r[df_r['__macro'] == macro].copy()
    else:
        df_r_macro = df_r.copy()

    if len(df_r_macro) >= need_n:
        return df_r_macro

    # 3) 권역만 (반경 이탈 허용)
    df_macro_only = df_all[df_all['__macro'] == macro].copy() if macro else df_all.copy()
    df_macro_only = _mask_not_used(df_macro_only)
    if len(df_macro_only) >= need_n:
        return df_macro_only

    # 4) 전체에서 (반경 3/10km)
    df_all_r = _filter_candidates_by_radius_then_expand(df_all, center_lat, center_lon, base_km, expand_km, need_n)
    df_all_r = _mask_not_used(df_all_r)
    if len(df_all_r) >= need_n:
        return df_all_r

    # 5) 전체 상위 점수
    return _mask_not_used(df_all.sort_values("score", ascending=False))

# ─────────────────────────────────────────────────────────────────────
# 엔트리
# ─────────────────────────────────────────────────────────────────────
def run(
    region: str,
    transport_mode: str,
    score_label: str,
    days: int,
    cats: List[str],
    start_time: str = "09:00",
    end_time: str   = "21:00",
) -> pd.DataFrame:
    """
    대중교통 모드 (요청사항 모두 반영):
      - 카카오 지오코딩으로 중심좌표 산출(없으면 데이터 중앙값)
      - ★ 반경 3km를 먼저 적용(부족/특정 조건은 10km) → 그 다음 권역(수도권/영남권/…) 필터 적용
      - 시작지=3km 내 + score ≥ 80 우선(없으면 70, 그래도 없으면 완화) + 가까움/점수 혼합 랭킹
      - CATS 가중치 + 음식 시간대 규칙
      - 방문 사이 ‘이동’ 행 삽입, 정밀 매핑 하루 3회(동일 승하차면 타이틀 폴백)
      - 지하철 라벨: 대괄호/따옴표 제거, 다중 노선은 1개만, 역명 동일 시 라벨 생략
      - 버스 라벨: ‘OOO정류장 승차/하차’ (이미 ‘정류장’ 있으면 중복 없이)
      - ★ 1km 초과 이동은 양쪽 모두 대중교통 메타 없으면 그 조합 폐기
      - 하루 최소 6곳 보장(풀 자동 확장), 여행 전체 중복 방문 방지
    """
    assert transport_mode == "transit"
    cats = [c for c in map(_nfc, cats or []) if c]
    want_per_day = max(DAILY_VISIT_HARD_MIN, DAILY_VISIT_TARGET)

    t0 = time.time()
    def left():
        return TIME_BUDGET - (time.time() - t0)

    # 전체 후보 로드
    df_all = _load_places()

    # 중심 좌표(카카오 → 백업)
    center_lat, center_lon = _center_of_region(df_all, region, left())

    # 전역 재방문 방지
    used_global = set()
    def _key_row_dict(r: dict) -> tuple:
        return (_nfc(r.get("title","")), _nfc(r.get("addr1","")))
    def _key_row_series(r: pd.Series) -> tuple:
        return (_nfc(r.get("title","")), _nfc(r.get("addr1","")))

    all_rows: List[dict] = []
    total_days = max(1, int(days))

    for d in range(1, total_days + 1):
        need_today = want_per_day

        # 풀 확보(반경 → 권역 → 기타)
        df_day_pool = _widen_pool(df_all, region, used_global, need_today, center_lat, center_lon, days, cats)
        if df_day_pool.empty: break

        # 하루 선발 (시작지: 반경/점수 하한 반영, 1km/메타 시퀀싱)
        day_visits = _build_theme_queues_and_pick(
            df_day_pool, cats, start_time, end_time, want=need_today,
            center_lat=center_lat, center_lon=center_lon
        )

        # 부족 시 상위 미사용 보충
        if len(day_visits) < need_today:
            used_today_keys = { (_nfc(v.get("title","")), _nfc(v.get("addr1",""))) for v in day_visits }
            for _, r in df_day_pool.sort_values("score", ascending=False).iterrows():
                key = (_nfc(r.get("title","")), _nfc(r.get("addr1","")))
                if key in used_today_keys or key in used_global: continue
                day_visits.append(r.to_dict()); used_today_keys.add(key)
                if len(day_visits) >= need_today: break

        if not day_visits: continue

        # 보충 후에도 1km/메타 규칙 재강제
        day_visits = _enforce_distance_transit_rule(day_visits)

        # 지역 블록 + 강제 점프
        day_visits = _group_by_area_and_inject_hops(day_visits, k=HOP_COUNT_PER_DAY)

        # 스케줄 행 생성
        rows = _build_day_rows(d, day_visits, start_time, end_time)
        all_rows.extend(rows)

        # 사용 처리 (중복 방문 금지)
        for r in rows:
            if r.get("title") and r.get("title") != "이동":
                used_global.add(_key_row_dict(r))

        # 원본 풀에서도 제거
        if used_global:
            mask2 = ~df_all.apply(lambda r: _key_row_series(r) in used_global, axis=1)
            df_all = df_all[mask2].copy()
            if df_all.empty and d < total_days:
                break

    if not all_rows:
        return pd.DataFrame(columns=[
            "day_label","day","start_time","end_time","title","addr1",
            "cat1","cat2","cat3","출발지","교통편1","교통편2","도착지",
            "final_score","distance_from_prev_km","move_min","stay_min","mapx","mapy"
        ])

    df = pd.DataFrame(all_rows)
    df["__ord"] = (df["title"] != "이동").astype(int)
    df = df.sort_values(["day","start_time","__ord"]).drop(columns="__ord").reset_index(drop=True)
    return df
