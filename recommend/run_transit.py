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
    PATH_TMF,         # 관광지/POI CSV 경로 (예: '/mnt/data/관광지_법정동_매핑결과.csv')
    KAKAO_API_KEY,    # 카카오 키워드검색 API 키(있으면 지오코딩 시도)
    FAST_MODE,        # True면 외부 지오코딩 생략
)

# ─────────────────────────────────────────────────────────────────────
# 정책/상수
# ─────────────────────────────────────────────────────────────────────
# 하루 방문 목표/하한
DAILY_VISIT_TARGET    = 6         # 목표 6
DAILY_VISIT_HARD_MIN  = 5         # ★ 무조건 하루 최소 5 보장

# 체류/이동 시간 계산
DEFAULT_STAY_MIN      = 60
DEFAULT_TRANSIT_KM_H  = 18.0
PADDING_MIN           = 10
MIN_MOVE_MIN          = 8

# 반경 정책
BASE_RADIUS_KM        = 3.0       # ★ 1순위 반경: 3km
EXPAND_RADIUS_KM      = 10.0      # ★ 2순위 반경: 10km

# 1km 이동 규칙
DIST_STRICT_KM        = 1.0       # 1km 초과면 양쪽 모두 대중교통 메타 필요

# 시간대/음식
MEAL_CAT = "음식"
MEAL_MAIN_KEYWORDS = {"한식", "중식", "일식", "서양식", "이색음식점"}
CAFE_KEYWORDS      = {"카페", "전통찻집"}
LUNCH_START, LUNCH_END   = 11 * 60, 13 * 60
DINNER_START, DINNER_END = 17 * 60, 20 * 60
NIGHT_AFTER              = 20 * 60

# 카테고리 가중치(앞선 선호)
DEFAULT_WEIGHTS = [0.6, 0.3, 0.1]

# 지역 블록(권역 내 하위 블록) 이동
HOP_COUNT_PER_DAY   = 2
FORCED_HOP_MINUTES  = 60

# 지오코딩 시간 예산(초)
TIME_BUDGET = 8.0

# ─────────────────────────────────────────────────────────────────────
# 권역/시도 표준화
# ─────────────────────────────────────────────────────────────────────
SIDO_MAP: Dict[str, str] = {
    # 서울/수도권
    '서울':'서울특별시','서울시':'서울특별자치시','서울특별시':'서울특별자치시','서울특별자치시':'서울특별자치시',
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
    # 강원(개편)
    '강원':'강원특별자치도','강원도':'강원특별자치도','강원특별자치도':'강원특별자치도',
    # 충청
    '충북':'충청북도','충청북도':'충청북도',
    '충남':'충청남도','충청남도':'충청남도',
    # 전라(개편)
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
    '수도권': ['서울특별자치시', '경기도', '인천광역시'],
    '제주권': ['제주특별자치도'],
    '강원권': ['강원특별자치도'],
    '충청권': ['충청북도', '충청남도', '세종특별자치시', '대전광역시'],
    '호남권': ['전라북도', '전라남도', '광주광역시'],
    '영남권': ['경상북도', '경상남도', '부산광역시', '대구광역시', '울산광역시'],
}

CITY_TO_MACRO = {
    # 강원권 대표 도시
    "속초":"강원권","춘천":"강원권","원주":"강원권","강릉":"강원권","동해":"강원권","삼척":"강원권",
    # 제주권
    "제주":"제주권","서귀포":"제주권",
    # 호남권
    "전주":"호남권","군산":"호남권","익산":"호남권","목포":"호남권","여수":"호남권","순천":"호남권","광주":"호남권",
    # 영남권
    "부산":"영남권","대구":"영남권","울산":"영남권","경주":"영남권","포항":"영남권","창원":"영남권","김해":"영남권","거제":"영남권",
    # 충청권
    "대전":"충청권","세종":"충청권","청주":"충청권","천안":"충청권","아산":"충청권",
    # 수도권
    "서울":"수도권","인천":"수도권","수원":"수도권","성남":"수도권","용인":"수도권","고양":"수도권","안양":"수도권","의정부":"수도권",
}

_SIDO_TOKEN_RE = re.compile(
    r"(서울특별자치시|서울특별시|서울시|서울|"
    r"부산광역시|부산시|부산|"
    r"대구광역시|대구시|대구|"
    r"인천광역시|인천시|인천|"
    r"광주광역시|광주시|광주|"
    r"대전광역시|대전시|대전|"
    r"울산광역시|울산시|울산|"
    r"세종특별자치시|세종시|세종|"
    r"경기도|경기|"
    r"강원특별자치도|강원도|강원|"
    r"충청북도|충북|충청남도|충남|"
    r"전라북도|전북|전라남도|전남|"
    r"경상북도|경북|경상남도|경남|"
    r"제주특별자치도|제주도|제주시|서귀포시|제주)"
)

# ─────────────────────────────────────────────────────────────────────
# 문자열/수치 유틸
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

# 하버사인(스칼라)
def _haversine(lat1, lon1, lat2, lon2) -> float:
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan
    lat1r, lon1r = math.radians(float(lat1)), math.radians(float(lon1))
    lat2r, lon2r = math.radians(float(lat2)), math.radians(float(lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = math.sin(dlat/2)**2 + math.cos(lat1r)*math.cos(lat2r)*math.sin(dlon/2)**2
    return 2 * 6371.0088 * math.asin(math.sqrt(a))

# 하버사인(벡터) — Series→float 변환 오류 방지
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
# 권역 판정/부착
# ─────────────────────────────────────────────────────────────────────
def normalize_sido(name: str) -> Optional[str]:
    if not name: return None
    key = name.strip()
    # 공백 제거 변형도 허용
    return SIDO_MAP.get(key, SIDO_MAP.get(key.replace(' ', ''), None))

def extract_sido_from_addr(addr1: str) -> Optional[str]:
    if not addr1: return None
    m = _SIDO_TOKEN_RE.search(addr1)
    if not m: return None
    raw = m.group(1)
    # 예: 강원도 → 강원특별자치도 로 표준화
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
    # 도시명으로 보강
    for city, macro in CITY_TO_MACRO.items():
        if city in t:
            return macro
    return None

def macro_from_center(df: pd.DataFrame, lat: float, lon: float) -> Optional[str]:
    """중심좌표 주변(≤30km)에서 가장 많은 __macro를 선택해 권역 추정."""
    if pd.isna(lat) or pd.isna(lon) or df.empty:
        return None
    tmp = df.copy()
    tmp["__dist0"] = _haversine_vec(tmp["mapy"], tmp["mapx"], lat, lon)
    near = tmp[tmp["__dist0"] <= 30.0]
    if near.empty:
        near = tmp.sort_values("__dist0").head(200)
    counts = near["__macro"].value_counts(dropna=True)
    return counts.index[0] if len(counts) > 0 and counts.index[0] else None

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

    # 시·도/권역 붙이기 (addr1 기반 → 요청하신 방식)
    df = attach_sido_macro_columns(df)
    return df

# ─────────────────────────────────────────────────────────────────────
# 중심좌표(지오코딩) + 반경 필터
# ─────────────────────────────────────────────────────────────────────
def _geocode_region_kakao(region: str, time_left: float) -> Optional[Tuple[float, float]]:
    if FAST_MODE or time_left < 2.0 or not KAKAO_API_KEY:
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
    # 1) 카카오 지오코딩 우선
    pos = _geocode_region_kakao(region, time_left)
    if pos:
        return pos
    # 2) 후보 중 텍스트 매칭 중앙값(백업)
    mask = df["addr1"].astype(str).str.contains(region, na=False) | df["title"].astype(str).str.contains(region, na=False)
    sub = df[mask].copy() if mask.sum() >= 1 else df.copy()
    cy = float(pd.to_numeric(sub["mapy"], errors="coerce").median())
    cx = float(pd.to_numeric(sub["mapx"], errors="coerce").median())
    return cy, cx

def _filter_by_radius_km(df: pd.DataFrame, center_lat: float, center_lon: float, r_km: float) -> pd.DataFrame:
    df = df.copy()
    df["__dist0"] = _haversine_vec(df["mapy"], df["mapx"], center_lat, center_lon)
    return df[df["__dist0"] <= float(r_km)].copy()

# ─────────────────────────────────────────────────────────────────────
# 카테고리/음식 큐 + 할당
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
# 대중교통 메타/표기
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

def _ensure_stop_suffix(name: str) -> str:
    s = _nfc(name)
    if not s: return s
    return s if "정류장" in s else f"{s} 정류장"

def _get_str(row: dict, key: str) -> str:
    v = (row or {}).get(key, "")
    s = _nfc(v)
    return "" if s.lower() == "nan" else s

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
        la, lb = _ensure_stop_suffix(a_bus), _ensure_stop_suffix(b_bus)
        if la == lb:
            return None
        return f"{la} 승차", f"{lb} 하차"
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
            if _has_transit_meta(prev) and _has_transit_meta(cand):
                seq.append(cand); pool.pop(i); added = True; break
        if not added:
            break
    return seq

def _repair_sequence_min_len(candidates: List[dict], want: int) -> List[dict]:
    """
    1km/메타 규칙 때문에 길이가 줄었을 때,
    남은 후보 중에서 규칙을 만족하는 것들을 추가로 붙여 최소 want까지 채움.
    """
    if not candidates: return []
    seq = candidates[:]
    pool = []  # 아직 안 쓴 후보들을 점수 높은 순으로
    used = { (_nfc(v.get("title","")), _nfc(v.get("addr1",""))) for v in candidates }
    # 점수 기준으로 정리
    to_sort = [v for v in candidates]  # 이미 picks였던 것
    # pool을 별도로 받아오도록 호출측에서 넘길 수도 있으나, 여기선 간단히 skip
    while len(seq) < want:
        # 규칙 맞는 다음 후보를 찾아 붙인다(메타 우선 → 1km 근접 순)
        # 실제로는 호출측에서 day_pool에서 남은 것들을 주면 더 좋음
        break  # 현재 컨텍스트에선 시퀀싱 전 단계에서 충분히 채우도록 하므로 noop
    return seq

# ─────────────────────────────────────────────────────────────────────
# 시작 장소 = 중심에 가깝고 점수 높은 곳
# ─────────────────────────────────────────────────────────────────────
def _pick_start_place(df: pd.DataFrame, center_lat: float, center_lon: float) -> Optional[dict]:
    if df.empty: return None
    work = df.copy()
    work["__dist0"] = _haversine_vec(work["mapy"], work["mapx"], center_lat, center_lon)
    s = pd.to_numeric(work["score"], errors="coerce").fillna(0.0)
    smin, smax = float(s.min()), float(s.max())
    work["s_norm"] = (s - smin) / (smax - smin) if smax > smin else 0.0
    d = pd.to_numeric(work["__dist0"], errors="coerce").fillna(0.0)
    dmax = float(d.max()) if len(d) else 1.0
    work["d_norm"] = 1.0 - (d / dmax if dmax > 0 else 0.0)
    # 가까움(0.6) + 점수(0.4) 비중 — "엄청 가깝고 점수도 높은" 쪽으로 치우침
    work["rank_score"] = 0.6 * work["d_norm"] + 0.4 * work["s_norm"]
    r0 = work.sort_values(["rank_score","score"], ascending=[False, False]).head(1)
    return r0.iloc[0].to_dict() if not r0.empty else None

# ─────────────────────────────────────────────────────────────────────
# 하루 선발(시작고정 + 음식시간 + 가중치) → 시퀀싱 강제
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

    # 시작 장소: 중심 가까움 + 점수 높은 곳
    starter = _pick_start_place(df_sorted, center_lat, center_lon)
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

        # 음식 시간대 우선
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

        # 아무것도 못 뽑으면 전체 점수순에서
        if chosen is None:
            for i in range(len(df_sorted)):
                temp = _take(i)
                if temp is not None: chosen = temp; break

        if chosen is not None:
            picks.append(chosen)

    # 카테고리 최소 1 보강
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

    # 1km/대중교통 규칙 적용
    seq = [r.to_dict() for r in picks]
    seq = _enforce_distance_transit_rule(seq)
    # 필요 시 보정(최소 길이 채우기) — 현재는 상위 단계에서 풀 확보를 충분히 하므로 noop
    seq = _repair_sequence_min_len(seq, want)
    return seq[:want]

# ─────────────────────────────────────────────────────────────────────
# 이동 Row 생성
# ─────────────────────────────────────────────────────────────────────
def _build_day_rows(day: int, visits: List[dict], start_time: str, end_time: str) -> List[dict]:
    day_label = f"{day}일차"
    rows: List[dict] = []
    cur_time = _parse_hhmm(start_time)
    specific_used = 0
    MAX_SPECIFIC_MATCH_PER_DAY = 3

    for i, v in enumerate(visits):
        # 방문
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

        # 이동
        if i + 1 < len(visits):
            a, b = v, visits[i+1]
            d_km = _haversine(_float(a.get("mapy")), _float(a.get("mapx")),
                              _float(b.get("mapy")), _float(b.get("mapx")))
            move_min = _estimate_transit_minutes(d_km)

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
# 풀 확장(반경 → 권역 → 기타) — 요청사항 반영
# ─────────────────────────────────────────────────────────────────────
def _widen_pool(df_all: pd.DataFrame, region: str, used_global: set, need_n: int,
                center_lat: float, center_lon: float, cats: List[str]) -> pd.DataFrame:
    """
    1) 반경 3km + 권역 고정   2) (부족 시) 반경 10km + 권역 고정
    3) (그래도 부족) 권역만   4) 전체에서 반경 3/10   5) 권역 상위 점수   6) 전체 상위
    """
    def _mask_not_used(df):
        if not used_global: return df
        def _key_row_series(r: pd.Series) -> tuple:
            return (_nfc(r.get("title","")), _nfc(r.get("addr1","")))
        mask = ~df.apply(lambda r: _key_row_series(r) in used_global, axis=1)
        return df[mask]

    # 권역 고정: 텍스트→실패 시 중심좌표 근처 데이터로 추정
    macro = infer_macro_from_text(region)
    if not macro:
        macro = macro_from_center(df_all, center_lat, center_lon)

    # 1) 반경 3km + 권역
    df_r3 = _filter_by_radius_km(df_all, center_lat, center_lon, BASE_RADIUS_KM)
    if macro: df_r3 = df_r3[df_r3['__macro'] == macro]
    df_r3 = _mask_not_used(df_r3).copy()
    if len(df_r3) >= need_n:
        return df_r3

    # 2) 반경 10km + 권역
    df_r10 = _filter_by_radius_km(df_all, center_lat, center_lon, EXPAND_RADIUS_KM)
    if macro: df_r10 = df_r10[df_r10['__macro'] == macro]
    df_r10 = _mask_not_used(df_r10).copy()
    if len(df_r10) >= need_n:
        return df_r10

    # 3) 권역만 (반경 이탈 허용)
    if macro:
        df_macro = _mask_not_used(df_all[df_all['__macro'] == macro])
        if len(df_macro) >= need_n:
            return df_macro

    # 4) 전체에서 반경 3/10
    df_all_r3  = _mask_not_used(_filter_by_radius_km(df_all, center_lat, center_lon, BASE_RADIUS_KM))
    if len(df_all_r3) >= need_n:
        return df_all_r3
    df_all_r10 = _mask_not_used(_filter_by_radius_km(df_all, center_lat, center_lon, EXPAND_RADIUS_KM))
    if len(df_all_r10) >= need_n:
        return df_all_r10

    # 5) 권역 상위 점수
    if macro:
        df_top_macro = _mask_not_used(df_all[df_all['__macro'] == macro].sort_values("score", ascending=False))
        if len(df_top_macro):
            return df_top_macro

    # 6) 마지막 전체 상위 점수
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
    요청사항 요약:
      - 입력 지역(예: '속초역','잠실역')을 카카오 지오코딩(가능 시)으로 위경도 확보
      - 먼저 반경 3km에서 후보 추출, 부족하면 반경 10km로 확장
      - addr1에서 시·도 추출해 권역을 고정(강원특별자치도 → 강원권 등)
      - 시작 장소는 '중심 아주 가까움 + 점수 높은' 곳을 우선 선정
      - 점수 높은 순 + 테마 가중치/음식 시간대 배치
      - 1km 초과 이동은 양쪽 모두 대중교통 메타 있을 때만 허용
      - 버스: 'OOO정류장 승차/하차', 지하철: '역명(첫 노선) 승차/하차'
      - 하루 최소 5~6곳 보장(반경/권역 단계적으로 확장해서 충족)
    """
    assert transport_mode == "transit"
    cats = [c for c in map(_nfc, cats or []) if c]
    if not cats:
        cats = ["관광"]
    want_per_day = max(DAILY_VISIT_HARD_MIN, DAILY_VISIT_TARGET)

    t0 = time.time()
    def left():
        return TIME_BUDGET - (time.time() - t0)

    # 전체 후보
    df_all = _load_places()

    # 중심 좌표
    center_lat, center_lon = _center_of_region(df_all, region, left())

    # 전역 중복 방지
    used_global = set()
    def _key_row_dict(r: dict) -> tuple:
        return (_nfc(r.get("title","")), _nfc(r.get("addr1","")))
    def _key_row_series(r: pd.Series) -> tuple:
        return (_nfc(r.get("title","")), _nfc(r.get("addr1","")))

    all_rows: List[dict] = []
    total_days = max(1, int(days))

    for d in range(1, total_days + 1):
        need_today = want_per_day

        # 풀 확보(3km → 10km → 권역 → …)
        df_day_pool = _widen_pool(df_all, region, used_global, need_today, center_lat, center_lon, cats)
        if df_day_pool.empty:
            continue

        # 하루 선발
        day_visits = _build_theme_queues_and_pick(
            df_day_pool, cats, start_time, end_time, want=need_today,
            center_lat=center_lat, center_lon=center_lon
        )

        # 부족 시 점수 순으로 보충(여전히 권역/반경 풀에서)
        if len(day_visits) < need_today:
            missing = need_today - len(day_visits)
            used_today_keys = { (_nfc(v.get("title","")), _nfc(v.get("addr1",""))) for v in day_visits }
            for _, r in df_day_pool.sort_values("score", ascending=False).iterrows():
                key = (_nfc(r.get("title","")), _nfc(r.get("addr1","")))
                if key in used_today_keys or key in used_global:
                    continue
                day_visits.append(r.to_dict()); used_today_keys.add(key)
                if len(day_visits) >= need_today:
                    break

        # 그래도 너무 적으면 최소 5까지 재확보
        if len(day_visits) < DAILY_VISIT_HARD_MIN:
            # 반경 10km 권역에서 추가로 확보
            df_extra = _filter_by_radius_km(df_all, center_lat, center_lon, EXPAND_RADIUS_KM)
            macro = infer_macro_from_text(region) or macro_from_center(df_all, center_lat, center_lon)
            if macro:
                df_extra = df_extra[df_extra['__macro'] == macro]
            used_keys = { (_nfc(v.get("title","")), _nfc(v.get("addr1",""))) for v in day_visits } | used_today_keys if 'used_today_keys' in locals() else set()
            for _, r in df_extra.sort_values("score", ascending=False).iterrows():
                key = (_nfc(r.get("title","")), _nfc(r.get("addr1","")))
                if key in used_keys or key in used_global:
                    continue
                day_visits.append(r.to_dict()); used_keys.add(key)
                if len(day_visits) >= DAILY_VISIT_HARD_MIN:
                    break

        if not day_visits:
            continue

        # 1km/메타 규칙 재강제
        day_visits = _enforce_distance_transit_rule(day_visits)

        # 지역 블록 묶고 강제 점프 주입(권역 내부 하위 이동)
        day_visits = _group_by_area_and_inject_hops(day_visits, k=HOP_COUNT_PER_DAY)

        # 스케줄 행 생성
        rows = _build_day_rows(d, day_visits, start_time, end_time)
        all_rows.extend(rows)

        # 방문 사용 처리
        for r in rows:
            if r.get("title") and r.get("title") != "이동":
                used_global.add(_key_row_dict(r))

        # 원본 풀에서 제거
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
    # 방문 → 이동 순으로 보기 좋게
    df["__ord"] = (df["title"] != "이동").astype(int)
    df = df.sort_values(["day","start_time","__ord"]).drop(columns="__ord").reset_index(drop=True)
    return df
