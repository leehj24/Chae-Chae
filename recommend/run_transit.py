# recommend/run_transit.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

import pandas as pd
import numpy as np
import unicodedata as ud

from recommend.config import (
    PATH_TMF,  # 관광지/POI CSV 경로 (예: '/mnt/data/관광지_법정동_매핑결과.csv')
)

# ─────────────────────────────────────────────────────────────────────────────
# 정책/상수
# ─────────────────────────────────────────────────────────────────────────────
MAX_SPECIFIC_MATCH_PER_DAY = 3   # 하루(day_label) 정밀 교통 매핑 상한
DAILY_VISIT_TARGET       = 6     # 기본 목표 (하루 방문지 수)
DAILY_VISIT_HARD_MIN     = 6     # ★ 하드 하한: 최소 6개 보장
DEFAULT_STAY_MIN         = 60    # 방문 체류시간(분)
DEFAULT_TRANSIT_KM_H     = 18.0  # 대중교통 평균속도(km/h) 가정
PADDING_MIN              = 10    # 이동/대기 여유
MIN_MOVE_MIN             = 8     # 최소 이동시간(분)
DIST_STRICT_KM           = 1.0   # ★ 1km 초과시 양쪽 모두 대중교통 메타 필수

# 음식 규칙
MEAL_CAT = "음식"
MEAL_MAIN_KEYWORDS = {"한식", "중식", "일식", "서양식", "이색음식점"}
CAFE_KEYWORDS      = {"카페", "전통찻집"}

LUNCH_START, LUNCH_END   = 11 * 60, 13 * 60  # 11:00~13:00
DINNER_START, DINNER_END = 17 * 60, 20 * 60  # 17:00~20:00
NIGHT_AFTER              = 20 * 60           # 20:00 이후 카페

# CATS 가중치: 입력 순서대로 [높음, 중간, 낮음] 분배
DEFAULT_WEIGHTS = [0.6, 0.3, 0.1]

# 지역 점프(하루) 설정: 하루 1~3회 정도 지역 블록 전환 시 강제 60분 이동
HOP_COUNT_PER_DAY   = 2
FORCED_HOP_MINUTES  = 60

# ─────────────────────────────────────────────────────────────────────────────
# 시·도 표준화 + 광역권
# ─────────────────────────────────────────────────────────────────────────────
SIDO_MAP: Dict[str, str] = {
    # 서울
    '서울': '서울특별시', '서울시': '서울특별시', '서울특별시': '서울특별시',
    # 부산
    '부산': '부산광역시', '부산시': '부산광역시', '부산광역시': '부산광역시',
    # 대구
    '대구': '대구광역시', '대구시': '대구광역시', '대구광역시': '대구광역시',
    # 인천
    '인천': '인천광역시', '인천시': '인천광역시', '인천광역시': '인천광역시',
    # 광주
    '광주': '광주광역시', '광주시': '광주광역시', '광주광역시': '광주광역시',
    # 대전
    '대전': '대전광역시', '대전시': '대전광역시', '대전광역시': '대전광역시',
    # 울산
    '울산': '울산광역시', '울산시': '울산광역시', '울산광역시': '울산광역시',
    # 세종
    '세종': '세종특별자치시', '세종시': '세종특별자치시', '세종특별자치시': '세종특별자치시',
    # 경기
    '경기': '경기도', '경기도': '경기도',
    # 강원
    '강원': '강원도', '강원도': '강원도', '강원특별자치도': '강원도',
    # 충청
    '충남': '충청남도', '충청남도': '충청남도',
    '충북': '충청북도', '충청북도': '충청북도',
    # 전라
    '전남': '전라남도', '전라남도': '전라남도',
    '전북': '전라북도', '전라북도': '전라북도', '전북특별자치도': '전라북도',
    # 경상
    '경남': '경상남도', '경상남도': '경상남도',
    '경북': '경상북도', '경상북도': '경상북도',
    # 제주
    '제주': '제주특별자치도', '제주시': '제주특별자치도', '서귀포시': '제주특별자치도',
    '제주도': '제주특별자치도', '제주특별자치도': '제주특별자치도',
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

# ─────────────────────────────────────────────────────────────────────────────
# 문자열/시간/수치 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _nfc(s: Optional[str]) -> str:
    return ud.normalize("NFC", str(s)).strip() if s is not None else ""

def _float(x, default=np.nan):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default

def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _haversine(lat1, lon1, lat2, lon2) -> float:
    """스칼라용 하버사인 거리(km)"""
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan
    lat1r, lon1r = math.radians(float(lat1)), math.radians(float(lon1))
    lat2r, lon2r = math.radians(float(lat2)), math.radians(float(lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = math.sin(dlat/2)**2 + math.cos(lat1r)*math.cos(lat2r)*math.sin(dlon/2)**2
    return 2 * 6371.0088 * math.asin(math.sqrt(a))

def _haversine_km_array(lat1, lon1, lat2, lon2) -> np.ndarray:
    """배열/Series용 하버사인(km) — lat2/lon2 스칼라 또는 배열 브로드캐스트 허용"""
    lat1 = _to_float_series(pd.Series(lat1)).to_numpy()
    lon1 = _to_float_series(pd.Series(lon1)).to_numpy()
    if isinstance(lat2, (np.ndarray, pd.Series)):
        lat2 = _to_float_series(pd.Series(lat2)).to_numpy()
        lon2 = _to_float_series(pd.Series(lon2)).to_numpy()
    else:
        lat2 = float(lat2)
        lon2 = float(lon2)
    lat1r = np.radians(lat1); lon1r = np.radians(lon1)
    lat2r = np.radians(lat2); lon2r = np.radians(lon2)
    dlat = lat2r - lat1r; dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    d = 6371.0088 * c
    if isinstance(lat2, float):  # lat2/lon2 스칼라일 때 NaN 마스킹
        mask_nan = np.isnan(lat1) | np.isnan(lon1)
    else:
        mask_nan = np.isnan(lat1) | np.isnan(lon1) | np.isnan(lat2r) | np.isnan(lon2r)
    d[mask_nan] = np.nan
    return d

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

# ─────────────────────────────────────────────────────────────────────────────
# 시·도/권역 유틸
# ─────────────────────────────────────────────────────────────────────────────
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
    if any(k in t for k in ['수도권', '서울', '경기', '인천']): return '수도권'
    if '제주' in t: return '제주권'
    if '강원' in t: return '강원권'
    if any(k in t for k in ['충청', '세종', '대전']): return '충청권'
    if any(k in t for k in ['전라', '광주']): return '호남권'
    if any(k in t for k in ['경상', '부산', '대구', '울산']): return '영남권'
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드/필터
# ─────────────────────────────────────────────────────────────────────────────
def _load_places() -> pd.DataFrame:
    df = pd.read_csv(PATH_TMF, encoding="utf-8")

    low = {c.lower(): c for c in df.columns}
    def col(*names, default=None):
        for n in names:
            if n in low: return low[n]
        return default

    title = col("title", "name", default=None)
    addr1 = col("addr1", "address", "addr", default=None)
    cat1  = col("cat1", default=None)
    cat2  = col("cat2", default=None)
    cat3  = col("cat3", default=None)
    lat   = col("mapy", "lat", "latitude", default=None)
    lon   = col("mapx", "lon", "lng", "longitude", default=None)
    score = col("인기도지수", "관광지수", "score", default=None)

    c_sub_st = col("closest_subway_station")
    c_sub_ln = col("closest_subway_line")
    c_bus_st = col("closest_bus_station")

    if title is None: df["title"] = ""
    else:             df["title"] = df[title].astype(str)
    if addr1 is None: df["addr1"] = ""
    else:             df["addr1"] = df[addr1].astype(str)

    df["cat1"] = df[cat1] if cat1 else ""
    df["cat2"] = df[cat2] if cat2 else ""
    df["cat3"] = df[cat3] if cat3 else ""

    df["mapy"] = pd.to_numeric(df[lat], errors="coerce") if lat else np.nan
    df["mapx"] = pd.to_numeric(df[lon], errors="coerce") if lon else np.nan

    df["score"] = pd.to_numeric(df[score], errors="coerce") if score else 0.0
    df["score"] = df["score"].fillna(0.0)

    df["closest_subway_station"] = df[c_sub_st].astype(str) if c_sub_st else ""
    df["closest_subway_line"]    = df[c_sub_ln].astype(str) if c_sub_ln else ""
    df["closest_bus_station"]    = df[c_bus_st].astype(str) if c_bus_st else ""

    df = df[[
        "title","addr1","cat1","cat2","cat3","mapx","mapy","score",
        "closest_subway_station","closest_subway_line","closest_bus_station"
    ]].copy()

    # 시·도/광역권 컬럼 부착
    df = attach_sido_macro_columns(df)
    return df

def _region_filter_macro(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """권역(수도권 등) → 키워드 필터(가능하면) → 비면 권역 풀 그대로"""
    macro = infer_macro_from_text(region)
    work = df.copy()
    if macro:
        work = work[work['__macro'] == macro].copy()

    r = _nfc(region)
    if not r:
        return work

    addr1 = work["addr1"].astype(str)
    title = work["title"].astype(str)
    cat1  = work["cat1"].astype(str)
    mask = (addr1.str.contains(r, na=False) |
            title.str.contains(r, na=False) |
            cat1.str.contains(r, na=False))
    sub = work[mask].copy()
    return sub if not sub.empty else work

# ─────────────────────────────────────────────────────────────────────────────
# 카테고리/음식 큐
# ─────────────────────────────────────────────────────────────────────────────
def _build_theme_queues(selected_df: pd.DataFrame, cats_norm: List[str]) -> Dict[str, List[int]]:
    queues: Dict[str, List[int]] = {c: [] for c in cats_norm}
    seen = set()
    for i, s in selected_df.iterrows():
        key = (_nfc(s.get("title","")), _nfc(s.get("addr1","")))
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

# ─────────────────────────────────────────────────────────────────────────────
# 대중교통 메타 처리 & 포맷
# ─────────────────────────────────────────────────────────────────────────────
def _clean_line_str(line_raw: str) -> str:
    """
    "['2호선', '8호선']" → "2호선" (첫 요소만)
    "부산 도시철도 2호선" 같은 전체 명칭도 그대로 허용.
    """
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

def _has_transit_meta(row: dict) -> bool:
    # 지하철 또는 버스 정보 중 하나라도 있으면 '메타 있음'
    has_sub = bool(_get_str(row, "closest_subway_station"))
    has_bus = bool(_get_str(row, "closest_bus_station"))
    return has_sub or has_bus

def _specific_mapping_from_metadata(a: dict, b: dict) -> Optional[Tuple[str, str]]:
    # 지하철 우선
    a_sub, a_line = _get_str(a, "closest_subway_station"), _get_str(a, "closest_subway_line")
    b_sub, b_line = _get_str(b, "closest_subway_station"), _get_str(b, "closest_subway_line")
    if a_sub and b_sub:
        a_lbl = _format_subway_label(a_sub, a_line)
        b_lbl = _format_subway_label(b_sub, b_line)
        # 승차/하차 라벨이 완전히 동일하면 매핑 사용하지 않음 → title 폴백 유도
        if a_lbl == b_lbl:
            return None
        return (f"{a_lbl} 승차", f"{b_lbl} 하차")
    # 버스
    a_bus, b_bus = _get_str(a, "closest_bus_station"), _get_str(b, "closest_bus_station")
    if a_bus and b_bus:
        if a_bus == b_bus:
            return None
        return (f"{a_bus} 승차", f"{b_bus} 하차")
    return None

def _fallback_titles(from_title: str, to_title: str) -> Tuple[str, str]:
    return _nfc(from_title), _nfc(to_title)

# ─────────────────────────────────────────────────────────────────────────────
# 지역 블록 구성 + 강제 점프 플래그
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# “1km 초과 → 양쪽 모두 대중교통 메타 필수” 강제 시퀀싱
# ─────────────────────────────────────────────────────────────────────────────
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
            break  # 남은 후보가 모두 규칙 위반 → 종료
    return seq

# ─────────────────────────────────────────────────────────────────────────────
# 하루 선발: CATS 가중치/최소1 + 음식 시간대 + 시작장소 선정
# ─────────────────────────────────────────────────────────────────────────────
def _pick_start_place(df: pd.DataFrame, region: str) -> Optional[dict]:
    if df.empty: return None
    r = _nfc(region)
    sub = df[(df["addr1"].astype(str).str.contains(r, na=False)) |
             (df["title"].astype(str).str.contains(r, na=False))]
    if sub.empty: sub = df.copy()

    # 기준점(선택 지역 근처의 중앙값 좌표)
    cy = float(pd.to_numeric(sub["mapy"], errors="coerce").median())
    cx = float(pd.to_numeric(sub["mapx"], errors="coerce").median())

    work = df.copy()
    work["mapy_f"] = pd.to_numeric(work["mapy"], errors="coerce")
    work["mapx_f"] = pd.to_numeric(work["mapx"], errors="coerce")

    # 🔧 Series 전체에 대해 벡터 하버사인 사용
    work["dist0"] = _haversine_km_array(work["mapy_f"], work["mapx_f"], cy, cx)

    # 점수/거리 정규화
    s = pd.to_numeric(work["score"], errors="coerce").fillna(0.0)
    smin, smax = float(s.min()), float(s.max())
    work["s_norm"] = (s - smin) / (smax - smin) if smax > smin else 0.0

    d = pd.to_numeric(work["dist0"], errors="coerce")
    dmax = float(d.max()) if len(d) else 1.0
    work["d_norm"] = 1.0 - (d / dmax if dmax > 0 else 0.0)  # 가까울수록 크게

    work["rank_score"] = 0.6 * work["s_norm"] + 0.4 * work["d_norm"]
    r0 = work.sort_values(["rank_score", "score"], ascending=[False, False]).head(1)
    return r0.iloc[0].to_dict() if not r0.empty else None

def _pick_for_day_with_food_and_cats(
    df: pd.DataFrame, cats_norm: List[str], start_time: str, end_time: str, want: int, region: str
) -> List[dict]:
    want = max(1, int(want))
    if df.empty: return []

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

    # ★ 시작 장소 우선 선정 (해당 지역에 가깝고 점수 높은 곳)
    starter = _pick_start_place(df_sorted, region)
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

        # 라운드로빈(음식은 지정시간 외 미배치)
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

    # 카테고리 최소 1개 보강
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

    # ★ 1km-대중교통 규칙 강제 시퀀싱
    seq = [r.to_dict() for r in picks]
    seq = _enforce_distance_transit_rule(seq)
    return seq[:want]

# ─────────────────────────────────────────────────────────────────────────────
# 스케줄링(방문→이동 행 생성)
# ─────────────────────────────────────────────────────────────────────────────
def _build_day_rows(day: int, visits: List[dict], start_time: str, end_time: str) -> List[dict]:
    day_label = f"{day}일차"
    rows: List[dict] = []
    cur_time = _parse_hhmm(start_time)
    specific_used = 0

    for i, v in enumerate(visits):
        # 방문 row
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
            # 지역 점프 플래그 우선
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

# ─────────────────────────────────────────────────────────────────────────────
# 풀 확장(하루 최소 6 보장)
# ─────────────────────────────────────────────────────────────────────────────
def _widen_pool(df_all: pd.DataFrame, region: str, used_global: set, need_n: int) -> pd.DataFrame:
    region = _nfc(region)
    df_all = df_all.copy()

    def _mask_not_used(df):
        if not used_global: return df
        def _key_row_series(r: pd.Series) -> tuple:
            return (_nfc(r.get("title","")), _nfc(r.get("addr1","")))
        mask = ~df.apply(lambda r: _key_row_series(r) in used_global, axis=1)
        return df[mask]

    # 1) 권역 + 키워드
    df_region = _region_filter_macro(df_all, region)
    df_region = _mask_not_used(df_region)
    if len(df_region) >= need_n: return df_region

    # 2) 접두어 확장
    prefix = region.split()[0] if " " in region else region[:2]
    addr1 = df_all["addr1"].astype(str)
    df_wide = df_all[addr1.str.contains(prefix, na=False)].copy()
    df_wide = _mask_not_used(df_wide)
    if len(df_wide) >= need_n: return df_wide

    # 3) 전체 상위 점수
    df_top = _mask_not_used(df_all.sort_values("score", ascending=False))
    return df_top

# ─────────────────────────────────────────────────────────────────────────────
# 엔트리 함수
# ─────────────────────────────────────────────────────────────────────────────
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
    대중교통 모드:
      - CATS 순서 가중치(앞이 큼) + 각 카테고리 최소 1개(가능한 경우)
      - 점심/저녁 메인식사, 20시 이후 카페 우선
      - 방문 사이 ‘이동’ 행 무조건 삽입
      - 정밀 교통 매핑(지하철/버스) 하루 3회 상한
        · 대괄호/따옴표 제거, 다중 노선 → 1개
        · 승차/하차 라벨 동일 시 매핑 쓰지 않고 title 폴백
      - ★ 1km 초과 이동이면 인접 양쪽 모두 대중교통 메타 필수(없으면 그 조합 폐기)
      - 시작 장소: 입력 지역과 가장 가깝고 점수도 높은 곳(복합 랭크)
      - 권역(수도권/제주/강원/충청/호남/영남) 우선 필터
      - 하루 최소 6곳 보장(풀 자동 확장)
      - 여행 전체 재방문 금지
      - 지역 블록 전환 시 첫 진입에 60분 강제 이동(하루 1~3회)
    """
    cats = [c for c in map(_nfc, cats or []) if c]
    if not cats: cats = ["관광"]

    want_per_day = max(DAILY_VISIT_HARD_MIN, DAILY_VISIT_TARGET)

    # 전체 후보
    df_all = _load_places()

    # 전역 재방문 금지 세트
    used_global = set()
    def _key_row_dict(r: dict) -> tuple:
        return (_nfc(r.get("title","")), _nfc(r.get("addr1","")))
    def _key_row_series(r: pd.Series) -> tuple:
        return (_nfc(r.get("title","")), _nfc(r.get("addr1","")))

    all_rows: List[dict] = []
    total_days = max(1, int(days))

    for d in range(1, total_days + 1):
        need_today = want_per_day

        # 오늘 풀: 권역/키워드 + 부족 시 확장
        df_day_pool = _widen_pool(df_all, region, used_global, need_today)
        if df_day_pool.empty: break

        # 하루 선발
        day_visits = _pick_for_day_with_food_and_cats(
            df_day_pool, cats, start_time, end_time, want=need_today, region=region
        )

        # 부족하면 상위 미사용으로 보충(최소 6 보장 시도)
        if len(day_visits) < need_today:
            missing = need_today - len(day_visits)
            used_today_keys = { (_nfc(v.get("title","")), _nfc(v.get("addr1",""))) for v in day_visits }
            for _, r in df_day_pool.sort_values("score", ascending=False).iterrows():
                key = (_nfc(r.get("title","")), _nfc(r.get("addr1","")))
                if key in used_today_keys or key in used_global: continue
                day_visits.append(r.to_dict()); used_today_keys.add(key)
                if len(day_visits) >= need_today: break

        # 최종 안전망: 1km/교통메타 규칙 다시 강제
        if day_visits:
            day_visits = _enforce_distance_transit_rule(day_visits)
        if not day_visits:
            continue

        # 지역 블록 구성 + 블록 전환 시 60분 강제 이동
        day_visits = _group_by_area_and_inject_hops(day_visits, k=HOP_COUNT_PER_DAY)

        # 스케줄 생성
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
    df["__ord"] = (df["title"] != "이동").astype(int)
    df = df.sort_values(["day","start_time","__ord"]).drop(columns="__ord").reset_index(drop=True)
    return df
