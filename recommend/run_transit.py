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
    PATH_TMF,  # 관광지/POI CSV 경로
    # FAST_MODE 등 다른 설정이 있으면 자유롭게 사용 가능
)

# ─────────────────────────────────────────────────────────────────────────────
# 정책/상수
# ─────────────────────────────────────────────────────────────────────────────
# 하루(day_label 기준) “정밀 교통 매핑(역/정류장/호선 등 메타데이터 사용)” 최대 횟수
MAX_SPECIFIC_MATCH_PER_DAY = 3

# 일정/이동 추정 기본값
DAILY_VISIT_TARGET     = 6         # 기본 목표 (하루 추천 방문지 개수)
DAILY_VISIT_HARD_MIN   = 6         # ★ 하드 하한: 최소 6개 이상 보장
DEFAULT_STAY_MIN       = 60        # 기본 체류 시간(분)
DEFAULT_TRANSIT_KM_H   = 18.0      # 대중교통 평균 이동 속도(km/h) 가정
PADDING_MIN            = 10        # 이동/대기 여유
MIN_MOVE_MIN           = 8         # 최소 이동시간(분)

# 음식 규칙
MEAL_CAT = "음식"
MEAL_MAIN_KEYWORDS = {"한식", "중식", "일식", "서양식", "이색음식점"}
CAFE_KEYWORDS      = {"카페", "전통찻집"}

LUNCH_START, LUNCH_END     = 11 * 60, 13 * 60  # 11:00~13:00
DINNER_START, DINNER_END   = 17 * 60, 20 * 60  # 17:00~20:00
NIGHT_AFTER                = 20 * 60           # 20:00 이후 카페 우선

# CATS 가중치: 입력 순서대로 [높음, 중간, 낮음] 분배
DEFAULT_WEIGHTS = [0.6, 0.3, 0.1]

# 지역 점프(하루) 설정: 하루에 1~3회 정도 지역을 바꿔서 60분쯤 이동
HOP_COUNT_PER_DAY   = 2          # (권장: 1~3) 하루 지역 블록 전환 횟수
FORCED_HOP_MINUTES  = 60         # 지역 간 점프 시 강제 이동시간(분)

# ─────────────────────────────────────────────────────────────────────────────
# 유틸
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

def _haversine(lat1, lon1, lat2, lon2) -> float:
    """두 좌표 간 거리(km)"""
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan
    lat1r, lon1r = math.radians(float(lat1)), math.radians(float(lon1))
    lat2r, lon2r = math.radians(float(lat2)), math.radians(float(lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = math.sin(dlat/2)**2 + math.cos(lat1r)*math.cos(lat2r)*math.sin(dlon/2)**2
    return 2 * 6371.0088 * math.asin(math.sqrt(a))

def _estimate_transit_minutes(d_km: float) -> int:
    """대중교통 이동시간(추정)"""
    if pd.isna(d_km):
        return MIN_MOVE_MIN
    mins = max(MIN_MOVE_MIN, int(round((d_km / max(1e-6, DEFAULT_TRANSIT_KM_H)) * 60 + PADDING_MIN)))
    return mins

def _parse_hhmm(s: str) -> datetime:
    return datetime.strptime(s, "%H:%M")

def _fmt_hhmm(dtobj: datetime) -> str:
    return dtobj.strftime("%H:%M")

def _minutes_of_day(hhmm: str) -> int:
    dt = _parse_hhmm(hhmm)
    return dt.hour * 60 + dt.minute

def _time_slots_per_day(start_hhmm: str, end_hhmm: str, count: int) -> List[str]:
    t0 = _parse_hhmm(start_hhmm)
    t1 = _parse_hhmm(end_hhmm)
    total = (t1 - t0).total_seconds() / 60.0
    if count <= 0 or total <= 0:
        return []
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
# 데이터 적응 레이어
# ─────────────────────────────────────────────────────────────────────────────
def _load_places() -> pd.DataFrame:
    """관광지 CSV 로드. 컬럼명 다양성 대응 + 교통 메타데이터 컬럼 포함."""
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

    # 정밀 매핑용 메타데이터 컬럼
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

    df["mapy"] = df[lat] if lat else np.nan
    df["mapx"] = df[lon] if lon else np.nan

    if score:
        df["score"] = pd.to_numeric(df[score], errors="coerce")
    else:
        df["score"] = 0.0

    # 메타데이터 컬럼(없으면 빈 문자열)
    df["closest_subway_station"] = df[c_sub_st].astype(str) if c_sub_st else ""
    df["closest_subway_line"]    = df[c_sub_ln].astype(str) if c_sub_ln else ""
    df["closest_bus_station"]    = df[c_bus_st].astype(str) if c_bus_st else ""

    cols = [
        "title","addr1","cat1","cat2","cat3","mapx","mapy","score",
        "closest_subway_station","closest_subway_line","closest_bus_station"
    ]
    return df[cols].copy()

def _region_filter(df: pd.DataFrame, region: str) -> pd.DataFrame:
    region = _nfc(region)
    if not region:
        return df.copy()

    addr1 = df["addr1"].astype(str)
    title = df["title"].astype(str)
    cat1  = df["cat1"].astype(str)

    mask = (
        addr1.str.contains(region, na=False)
        | title.str.contains(region, na=False)
        | cat1.str.contains(region, na=False)
    )
    f = df[mask].copy()
    if f.empty:
        f = df.copy().head(1000)
    return f.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 카테고리/음식 큐
# ─────────────────────────────────────────────────────────────────────────────
def _build_theme_queues(selected_df: pd.DataFrame, cats_norm: List[str]) -> Dict[str, List[int]]:
    """cats_norm에 포함된 각 테마별 인덱스 큐 생성 (title/addr1 중복 1회)."""
    queues: Dict[str, List[int]] = {c: [] for c in cats_norm}
    seen = set()
    for i, s in selected_df.iterrows():
        key = (_nfc(s.get("title", "")), _nfc(s.get("addr1", "")))
        if key in seen:
            continue
        text_cats = f"{s.get('cat1','')} {s.get('cat2','')} {s.get('cat3','')}"
        for c in cats_norm:
            if c and c in text_cats:
                queues[c].append(i)
                seen.add(key)
                break
    return queues

def _build_food_queues(selected_df: pd.DataFrame) -> Dict[str, List[int]]:
    """음식(cat1='음식') 중 메인식사/카페 분리."""
    meal_main, cafe = [], []
    for i, s in selected_df.iterrows():
        if str(s.get("cat1","")) != MEAL_CAT:
            continue
        c2 = str(s.get("cat2",""))
        c3 = str(s.get("cat3",""))
        bag = {t.strip() for t in (c2 + "," + c3).split(",") if t.strip()}
        if bag & CAFE_KEYWORDS:
            cafe.append(i)
        elif bag & MEAL_MAIN_KEYWORDS:
            meal_main.append(i)
        else:
            meal_main.append(i)  # 태그 없으면 일반 식사로 간주
    return {"meal_main": meal_main, "cafe": cafe}

def _allocate_quota_for_day(cats_norm: List[str], want: int) -> Dict[str, int]:
    """
    입력 CATS 순서대로 가중치 분배 + 각 카테고리 최소 1개(가능한 경우).
    """
    L = len(cats_norm)
    if L <= 0 or want <= 0:
        return {}
    weights = DEFAULT_WEIGHTS[:L]
    if len(weights) < L:
        tail = [max(0.0, (1.0 - sum(weights)) / (L - len(weights)))] * (L - len(weights))
        weights = weights + tail

    base = [1] * L  # 각 카테고리 최소 1개
    remain = max(0, want - sum(base))
    quota_float = [w * remain for w in weights]
    quota_add = [int(q) for q in quota_float]
    diff = remain - sum(quota_add)
    fracs = sorted([(i, quota_float[i] - quota_add[i]) for i in range(L)], key=lambda x: x[1], reverse=True)
    for i, _ in fracs:
        if diff <= 0:
            break
        quota_add[i] += 1
        diff -= 1
    final = [base[i] + quota_add[i] for i in range(L)]
    return {cats_norm[i]: final[i] for i in range(L)}

# ─────────────────────────────────────────────────────────────────────────────
# 교통편 매핑(메타데이터 기반) & 폴백
# ─────────────────────────────────────────────────────────────────────────────
def _get_str(row: dict, key: str) -> str:
    return _nfc((row or {}).get(key, ""))

def _specific_mapping_from_metadata(a: dict, b: dict) -> Optional[Tuple[str, str]]:
    """
    장소 메타데이터에 있는 가장 가까운 역/노선/버스정류장 정보를 사용해
    '정밀 매핑' 문구(교통편1/교통편2)를 만든다.
    - 우선순위: 지하철(양쪽 다 역 정보) > 버스(양쪽 다 정류장)
    - 반환값: (t1, t2)  예) "홍대입구역(2호선) 승차", "을지로입구역(2호선) 하차"
    - 둘 중 하나라도 정보가 없으면 None
    """
    # 1) 지하철 우선
    a_sub = _get_str(a, "closest_subway_station")
    a_line = _get_str(a, "closest_subway_line")
    b_sub = _get_str(b, "closest_subway_station")
    b_line = _get_str(b, "closest_subway_line")

    if a_sub and b_sub:
        def _fmt_sub(st, ln):
            return f"{st}({ln})" if ln else f"{st}"
        t1 = f"{_fmt_sub(a_sub, a_line)} 승차"
        t2 = f"{_fmt_sub(b_sub, b_line)} 하차"
        return t1, t2

    # 2) 버스 정류장
    a_bus = _get_str(a, "closest_bus_station")
    b_bus = _get_str(b, "closest_bus_station")
    if a_bus and b_bus:
        t1 = f"{a_bus} 승차"
        t2 = f"{b_bus} 하차"
        return t1, t2

    # 3) 정밀 매핑 불가
    return None

def _fallback_titles(from_title: str, to_title: str) -> Tuple[str, str]:
    """교통편 미매핑/하루 3회 초과 시: 교통편1=출발지 title, 교통편2=도착지 title"""
    return _nfc(from_title), _nfc(to_title)

# ─────────────────────────────────────────────────────────────────────────────
# 지역 블록(하루) 편성 + 강제 점프 플래그 주입
# ─────────────────────────────────────────────────────────────────────────────
def _area_key_from_addr(addr1: str) -> str:
    """주소에서 대략적인 지역 키('시/도 + 구/군')."""
    a = _nfc(addr1)
    if not a:
        return ""
    parts = a.split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return parts[0]

def _group_by_area_and_inject_hops(day_visits: List[dict], k: int) -> List[dict]:
    """
    하루 방문을 지역 블록으로 묶고, 블록 전환 시 다음 블록의 첫 방문에
    _force_move_min=FORCED_HOP_MINUTES 플래그를 심는다.
    """
    if not day_visits:
        return day_visits

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
        if ys and xs:
            return float(np.mean(ys)), float(np.mean(xs))
        return np.nan, np.nan

    areas = list(buckets.keys())
    if not areas:
        return day_visits

    # 첫 블록: 가장 많은 후보 지역
    areas_sorted = sorted(areas, key=lambda a: (-len(buckets[a]), a))
    start_area = areas_sorted[0]
    y0, x0 = _centroid(buckets[start_area])

    def _dist_from_start(a):
        y1, x1 = _centroid(buckets[a])
        if any(pd.isna([y0, x0, y1, x1])):
            return 1e9
        return _haversine(y0, x0, y1, x1)

    rest = sorted([a for a in areas if a != start_area], key=_dist_from_start)

    plan = [start_area] + rest
    plan = plan[:max(1, min(int(k), 3))]

    out: List[dict] = []
    first_block = True
    for area in plan:
        block = buckets[area]
        if first_block:
            out.extend(block)
            first_block = False
        else:
            if block:
                block[0] = {**block[0], "_force_move_min": FORCED_HOP_MINUTES}
                out.extend(block)

    # 계획 외 지역도 뒤에 이어 붙이되, 진입 시 강제 점프
    unused = [a for a in areas if a not in plan]
    for area in unused:
        block = buckets[area]
        if not block:
            continue
        block[0] = {**block[0], "_force_move_min": FORCED_HOP_MINUTES}
        out.extend(block)

    for v in out:
        v.pop("_area", None); v.pop("_y", None); v.pop("_x", None)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 선발(하루 단위): CATS 가중치/최소1 + 음식 시간대
# ─────────────────────────────────────────────────────────────────────────────
def _pick_for_day_with_food_and_cats(df: pd.DataFrame, cats_norm: List[str],
                                     start_time: str, end_time: str,
                                     want: int) -> List[dict]:
    """
    - CATS 순서 가중치(앞이 큼) + 각 카테고리 최소 1개(가능한 경우)
    - 점심/저녁엔 '음식-메인', 20시 이후엔 '카페' 우선
    - 못 채우면 점수 상위로 폴백
    """
    want = max(1, int(want))
    if df.empty:
        return []

    df_sorted = df.sort_values(["score"], ascending=False).reset_index(drop=True)
    theme_queues = _build_theme_queues(df_sorted, cats_norm)
    meal_enabled = (MEAL_CAT in cats_norm)
    food_queues = _build_food_queues(df_sorted) if meal_enabled else {"meal_main": [], "cafe": []}

    used = set()
    def _take_from_index(idx: int) -> Optional[pd.Series]:
        if idx < 0 or idx >= len(df_sorted):
            return None
        rec = df_sorted.loc[idx]
        key = (_nfc(rec.get("title","")), _nfc(rec.get("addr1","")))
        if key in used:
            return None
        used.add(key)
        return rec

    def _pop_from_queue(q: List[int]) -> Optional[pd.Series]:
        while q:
            i = q.pop(0)
            rec = _take_from_index(i)
            if rec is not None:
                return rec
        return None

    quota = _allocate_quota_for_day(cats_norm, want)
    for c in list(quota.keys()):
        if not theme_queues.get(c):
            quota[c] = 0

    slots = _time_slots_per_day(start_time, end_time, want)
    picks: List[pd.Series] = []

    cat_cursor, L = 0, max(1, len(cats_norm))
    for slot_hm in slots:
        if len(picks) >= want:
            break
        cur_min = _minutes_of_day(slot_hm)
        chosen: Optional[pd.Series] = None

        # (A) 음식 시간대 규칙
        if meal_enabled and quota.get(MEAL_CAT, 0) > 0:
            if LUNCH_START <= cur_min < LUNCH_END or DINNER_START <= cur_min < DINNER_END:
                chosen = _pop_from_queue(food_queues["meal_main"])
                if chosen is not None:
                    quota[MEAL_CAT] -= 1
            elif cur_min >= NIGHT_AFTER:
                chosen = _pop_from_queue(food_queues["cafe"])
                if chosen is not None:
                    quota[MEAL_CAT] -= 1

        # (B) 라운드로빈(음식은 지정시간 외 미배치)
        if chosen is None:
            for _ in range(L):
                c = cats_norm[cat_cursor % L]; cat_cursor += 1
                if quota.get(c, 0) <= 0:
                    continue
                if c == MEAL_CAT:
                    continue
                q = theme_queues.get(c, [])
                temp = _pop_from_queue(q)
                theme_queues[c] = q
                if temp is not None:
                    quota[c] -= 1
                    chosen = temp
                    break

        # (C) 폴백(남은 큐 → 전체 상위)
        if chosen is None:
            for c in cats_norm:
                if quota.get(c, 0) <= 0:
                    continue
                if c == MEAL_CAT:
                    continue
                temp = _pop_from_queue(theme_queues.get(c, []))
                if temp is not None:
                    quota[c] -= 1
                    chosen = temp
                    break
        if chosen is None:
            for i in range(len(df_sorted)):
                temp = _take_from_index(i)
                if temp is not None:
                    chosen = temp
                    break

        if chosen is not None:
            picks.append(chosen)

    # 카테고리 보강(후보는 있었는데 시간대/분배로 누락된 경우 1개 강제)
    def _category_appeared(cat: str) -> bool:
        for r in picks:
            text_cats = f"{r.get('cat1','')} {r.get('cat2','')} {r.get('cat3','')}"
            if cat and (cat in text_cats):
                return True
        return False

    for c in cats_norm:
        if not theme_queues.get(c):
            continue
        if not _category_appeared(c):
            if c == MEAL_CAT:
                forced = _pop_from_queue(food_queues["meal_main"]) or _pop_from_queue(food_queues["cafe"])
            else:
                forced = _pop_from_queue(theme_queues.get(c, []))
            if forced is not None:
                picks.append(forced)
                if len(picks) > want:
                    picks = picks[:want]

    return [r.to_dict() for r in picks[:want]]

# ─────────────────────────────────────────────────────────────────────────────
# 스케줄링(방문→이동 행 생성)
# ─────────────────────────────────────────────────────────────────────────────
def _build_day_rows(
    day: int,
    visits: List[dict],
    start_time: str,
    end_time: str,
) -> List[dict]:
    """
    - 방문지 사이에 ‘이동’ 행 무조건 삽입
    - day_label 기준으로 하루당 정밀 교통 매핑 3회만 허용
    - 정밀 매핑은 메타데이터(closest_* 열)에서만 생성
    - 나머지 이동 행은 교통편1/2를 출발/도착 title로 강제 세팅
    - 지역 블록 전환 시 목적지에 달린 _force_move_min을 이동시간으로 사용
    """
    day_label = f"{day}일차"
    rows: List[dict] = []

    cur_time = _parse_hhmm(start_time)
    specific_used = 0  # 오늘(day_label) 정밀 매핑 사용 횟수

    for i, v in enumerate(visits):
        # 방문 row
        stay_min = DEFAULT_STAY_MIN
        st = cur_time
        en = st + timedelta(minutes=stay_min)
        if not _between_time(start_time, end_time, _fmt_hhmm(st), _fmt_hhmm(en)):
            break

        rows.append({
            "day_label": day_label, "day": day,
            "start_time": _fmt_hhmm(st), "end_time": _fmt_hhmm(en),
            "title": _nfc(v.get("title","")), "addr1": _nfc(v.get("addr1","")),
            "cat1": _nfc(v.get("cat1","")), "cat2": _nfc(v.get("cat2","")), "cat3": _nfc(v.get("cat3","")),

            # 이동행에서 사용되는 필드(방문행에선 빈칸)
            "출발지": "", "교통편1": "", "교통편2": "", "도착지": "",

            "final_score": float(_float(v.get("score"), 0.0)),
            "distance_from_prev_km": np.nan, "move_min": 0, "stay_min": stay_min,
            "mapx": _float(v.get("mapx")), "mapy": _float(v.get("mapy"))
        })
        cur_time = en

        # 다음 방문이 있다면 이동 row 생성
        if i + 1 < len(visits):
            a = v
            b = visits[i+1]
            d_km = _haversine(_float(a.get("mapy")), _float(a.get("mapx")),
                              _float(b.get("mapy")), _float(b.get("mapx")))

            # 지역 점프 강제 시간 우선
            forced = int(b.get("_force_move_min", 0)) if isinstance(b, dict) else 0
            move_min = forced if forced > 0 else _estimate_transit_minutes(d_km)

            # 메타데이터 기반 정밀 매핑(하루 3회 제한)
            t_specific = _specific_mapping_from_metadata(a, b)
            if t_specific and specific_used < MAX_SPECIFIC_MATCH_PER_DAY:
                t1, t2 = t_specific
                specific_used += 1
            else:
                # 정밀 매핑 불가 또는 횟수 초과 → 규칙대로 title 폴백
                t1, t2 = _fallback_titles(_nfc(a.get("title","")), _nfc(b.get("title","")))

            st_m = cur_time
            en_m = st_m + timedelta(minutes=move_min)
            if not _between_time(start_time, end_time, _fmt_hhmm(st_m), _fmt_hhmm(en_m)):
                break

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
# 풀 확장 유틸: 하루 최소 6개 보장을 위해 후보를 넓혀서 확보
# ─────────────────────────────────────────────────────────────────────────────
def _widen_pool(df_all: pd.DataFrame, region: str, used_global: set, need_n: int) -> pd.DataFrame:
    """
    지역 필터 결과가 부족할 때 후보 풀을 넓혀서 최소 need_n개 이상 되도록 반환.
    1) region 정확 매칭
    2) 상위 행정구/접두어(예: '성수' → '서울', '서울 성동구' 패턴 등) 확장
    3) 전체에서 상위 score 순 추출
    모든 단계에서 used_global(이미 방문한 곳)는 제외.
    """
    region = _nfc(region)
    df_all = df_all.copy()

    def _mask_not_used(df):
        if not used_global:
            return df
        def _key_row_series(r: pd.Series) -> tuple:
            return (_nfc(r.get("title","")), _nfc(r.get("addr1","")))
        mask = ~df.apply(lambda r: _key_row_series(r) in used_global, axis=1)
        return df[mask]

    # 1) 정확 지역 필터
    df_region = _region_filter(df_all, region)
    df_region = _mask_not_used(df_region)
    if len(df_region) >= need_n:
        return df_region

    # 2) 상위/접두어 확장
    #    - '성수' → addr1에서 '서울' 같은 상위 토큰을 추정 (간단히 addr1의 첫 토큰들 활용)
    #    - 너무 공격적이면 과포함될 수 있지만, 보장 우선
    prefix = region.split()[0] if " " in region else region[:2]
    addr1 = df_all["addr1"].astype(str)
    df_wide = df_all[addr1.str.contains(prefix, na=False)].copy()
    df_wide = _mask_not_used(df_wide)
    if len(df_wide) >= need_n:
        return df_wide

    # 3) 전체 상위 점수
    df_top = df_all.sort_values("score", ascending=False)
    df_top = _mask_not_used(df_top)
    if len(df_top) >= need_n:
        return df_top

    # 그래도 부족하면 가능한 만큼만 반환
    return df_top

# ─────────────────────────────────────────────────────────────────────────────
# 엔트리 함수 (전역 재방문 금지 포함 + 하루 6개 보장)
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
    대중교통 모드 스케줄 생성
    - 입력 CATS 순서대로 가중치(앞이 높음) + 각 카테고리 최소 1개(가능한 경우)
    - 점심/저녁엔 '음식-메인', 20시 이후엔 '카페' 우선 배치
    - 방문 사이 ‘이동’ 행 무조건 삽입
    - day_label 기준 하루당 정밀 교통 매핑 3회만 허용 (메타데이터 열 사용)
      · 매핑 불가/초과 시 교통편1/2를 출발/도착 title로 채움
    - 하루 1~3회 지역 블록 점프(강제 60분 이동) 지원
    - ★ 여행 전체에서 장소 재방문 금지 (title+addr1 기준)
    - ★ 하루당 최소 6곳 이상 무조건 보장(풀 자동 확장)
    """
    cats = [c for c in map(_nfc, cats or []) if c]  # 빈값/공백 제거, 순서 유지
    if not cats:
        cats = ["관광"]

    want_per_day = max(DAILY_VISIT_HARD_MIN, DAILY_VISIT_TARGET)  # ★ 최소 6 보장

    # 전체 후보 로드
    df_all = _load_places()
    if "score" not in df_all.columns:
        df_all["score"] = 0.0
    df_all["score"] = pd.to_numeric(df_all["score"], errors="coerce").fillna(0.0)

    # 전역 재방문 금지용 세트
    used_global = set()
    def _key_row_dict(r: dict) -> tuple:
        return (_nfc(r.get("title","")), _nfc(r.get("addr1","")))
    def _key_row_series(r: pd.Series) -> tuple:
        return (_nfc(r.get("title","")), _nfc(r.get("addr1","")))

    all_rows: List[dict] = []
    total_days = max(1, int(days))

    for d in range(1, total_days + 1):
        # 이 날에 필요한 최소 수
        need_today = want_per_day

        # 1) 오늘 사용할 기본 풀 확보(지역 기준) + 부족 시 확장
        df_day_pool = _widen_pool(df_all, region, used_global, need_today)

        # 2) 그래도 '미사용' 후보가 need_today 미만이면, 전체에서 추가 보충
        if len(df_day_pool) < need_today:
            extra = df_all.copy()
            if used_global:
                mask_ex = ~extra.apply(lambda r: _key_row_series(r) in used_global, axis=1)
                extra = extra[mask_ex]
            df_day_pool = pd.concat([df_day_pool, extra], ignore_index=True)
            # 중복 제거
            df_day_pool = df_day_pool.drop_duplicates(subset=["title","addr1"])

        if df_day_pool.empty:
            # 물리적으로 후보가 전혀 없으면 종료
            break

        # 3) 하루분 방문 선발(음식/가중치/최소1 보장)
        day_visits = _pick_for_day_with_food_and_cats(
            df_day_pool, cats, start_time, end_time, want=need_today
        )

        # 4) 선발 결과가 6개 미만이면, 점수 상위 미사용 후보로 보충해서 ★무조건 6개 이상★
        if len(day_visits) < need_today:
            missing = need_today - len(day_visits)
            used_today_keys = { (_nfc(v.get("title","")), _nfc(v.get("addr1",""))) for v in day_visits }
            # df_day_pool에서 미사용 상위 추출
            can_add = []
            for _, r in df_day_pool.sort_values("score", ascending=False).iterrows():
                key = (_nfc(r.get("title","")), _nfc(r.get("addr1","")))
                if key in used_today_keys or key in used_global:
                    continue
                can_add.append(r.to_dict())
                used_today_keys.add(key)
                if len(can_add) >= missing:
                    break
            day_visits.extend(can_add)

        # 최종 안전망: 그래도 부족하면 전체에서 마지막 보충
        if len(day_visits) < need_today:
            missing = need_today - len(day_visits)
            used_today_keys = { (_nfc(v.get("title","")), _nfc(v.get("addr1",""))) for v in day_visits }
            for _, r in df_all.sort_values("score", ascending=False).iterrows():
                key = (_nfc(r.get("title","")), _nfc(r.get("addr1","")))
                if key in used_today_keys or key in used_global:
                    continue
                day_visits.append(r.to_dict())
                used_today_keys.add(key)
                if len(day_visits) >= need_today:
                    break

        # (진짜로) 후보 자체가 6 미만인 경우만 제외하고, 이제 최소 6 보장됨
        if not day_visits:
            continue

        # 5) 지역 블록 구성 + 블록 전환 시 60분 강제 이동
        day_visits = _group_by_area_and_inject_hops(day_visits, k=HOP_COUNT_PER_DAY)

        # 6) 스케줄(방문→이동) 생성
        rows = _build_day_rows(d, day_visits, start_time, end_time)
        all_rows.extend(rows)

        # 7) 오늘 간 '방문' 장소를 전역 used에 추가
        for r in rows:
            if r.get("title") and r.get("title") != "이동":
                used_global.add(_key_row_dict(r))

        # 8) 원본 풀에서도 제거(성능/안전)
        if used_global:
            mask2 = ~df_all.apply(lambda r: _key_row_series(r) in used_global, axis=1)
            df_all = df_all[mask2].copy()
            if df_all.empty and d < total_days:
                # 남은 날이 있어도 더 이상 후보가 없으면 중단
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
