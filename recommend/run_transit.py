# recommend/run_transit.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import math
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

import requests
import pandas as pd
import numpy as np
import unicodedata as ud

from recommend.config import *  # PATH_TMF, KAKAO_API_KEY, FAST_MODE, TRANSIT_RADIUS_KM, TRANSIT_TOP_N

# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _nfc(s: str) -> str:
    return ud.normalize("NFC", str(s or "")).strip()

def _check_hhmm(s: str):
    datetime.strptime(s, "%H:%M")

def _parse_hhmm(hhmm: str) -> datetime:
    return datetime.strptime(hhmm, "%H:%M")

def _to_dt(base: datetime, hhmm: str) -> datetime:
    h, m = map(int, hhmm.split(":"))
    return base.replace(hour=h, minute=m, second=0, microsecond=0)

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _haversine_np(lat1, lon1, lat_arr, lon_arr):
    lat1r, lon1r = np.radians(float(lat1)), np.radians(float(lon1))
    lat2r, lon2r = np.radians(lat_arr.astype(float)), np.radians(lon_arr.astype(float))
    dlat = lat2r - lat1r; dlon = lon2r - lon1r
    a = np.sin(dlat/2)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2)**2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))

# ─────────────────────────────────────────────────────────────────────────────
# 시간/성능 가드
# ─────────────────────────────────────────────────────────────────────────────
TIME_BUDGET_SEC = 9.0
HARD_ABORT_SEC  = 9.5
QUICK_FILL_LEFT_SEC = 0.6  # 매우 촉박하면 폴백

# 이동/선정 상수
WALK_SKIP_KM          = 0.30
ANCHOR_HOP_MIN_KM     = 5.0
DAILY_TRANSIT_MIN_MIN = 60
BASE_SPEED_KMH        = 18.0
ADD_FIXED_MIN         = 8.0

# 음식/카페 규칙
MEAL_CAT = "음식"
MEAL_MAIN_KEYWORDS = {"한식","중식","일식","서양식","이색음식점"}
CAFE_KEYWORDS      = {"카페","전통찻집"}
LUNCH_START, LUNCH_END   = 11*60, 13*60
DINNER_START, DINNER_END = 17*60, 20*60
NIGHT_AFTER              = 20*60

# 테마 동의어/부분일치
THEME_SYNONYM = {
    "문화": {"문화","인문(문화/예술/역사)","역사","예술","박물관","전시"},
    "쇼핑": {"쇼핑","시장","상점가","아울렛","면세점","전통시장","상가"},
    "자연": {"자연","호수","해변","바다","산","공원","숲","계곡"},
    "레포츠": {"레포츠","액티비티","레저","스포츠","서핑","자전거"},
    "음식": {"음식","맛집","식당","한식","중식","일식","서양식","이색음식점"},
    "카페": {"카페","전통찻집"},
    "문화/예술": {"문화","예술","인문(문화/예술/역사)"},
}

SUBWAY_REGIONS = ("서울","경기","인천","부산","대구","대전","광주","울산")

def _estimate_transit_minutes(d_km: float, rel: str) -> int:
    try:
        d = float(d_km)
        if not np.isfinite(d) or d <= 0:
            d = 0.2
    except Exception:
        d = 0.2
    base_min = (d / BASE_SPEED_KMH) * 60.0 + ADD_FIXED_MIN
    if rel == "walk_hint":
        return max(10, int(round(base_min * 0.6)))
    elif rel == "subway_hint":
        return max(25, int(round(base_min * 0.9 + 8)))
    elif rel == "bus_hint":
        return max(30, int(round(base_min * 1.15 + 6)))
    else:
        return max(20, int(round(base_min)))

# ─────────────────────────────────────────────────────────────────────────────
# 메인 엔트리
# ─────────────────────────────────────────────────────────────────────────────
def run(
    region: str,
    transport_mode: str,
    score_label: str,
    days: int,
    cats: List[str],
    start_time: str = "08:00",
    end_time: str   = "22:30",
    **_,
) -> pd.DataFrame:

    t0 = time.monotonic()
    elapsed = lambda: time.monotonic() - t0
    left    = lambda: HARD_ABORT_SEC - elapsed()
    has     = lambda s: left() > s

    region = _nfc(region)
    if transport_mode != "transit":
        raise ValueError("transport_mode는 'transit' 이어야 합니다.")
    if score_label not in {"인기도지수","관광지수"}:
        raise ValueError("score_label은 '인기도지수' 또는 '관광지수' 중 선택.")
    days = max(1, int(days))
    cats = [c for c in map(_nfc, (cats or [])) if c][:3]
    if not cats:
        raise ValueError("테마 최소 1개 필요.")
    _check_hhmm(start_time); _check_hhmm(end_time)

    df_all = _standardize_cols(_read_csv_robust(PATH_TMF))

    coords = _geocode_region_kakao(region) if (KAKAO_API_KEY and has(2.0)) else None
    if coords:
        center_lat, center_lon = coords
    else:
        mask_addr = df_all["addr1"].str.contains(region, na=False)
        sub = df_all[mask_addr].copy() if mask_addr.sum() >= 1 else df_all.copy()
        center_lat = float(sub["lat"].median())
        center_lon = float(sub["lon"].median())

    radius_km = TRANSIT_RADIUS_KM if FAST_MODE else 20
    df_all["distance_km"] = _haversine_np(center_lat, center_lon, df_all["lat"], df_all["lon"])
    df_all = df_all[df_all["distance_km"] <= radius_km].copy()
    if df_all.empty:
        return _empty_df()

    score_col = {"관광지수": "tour_score", "인기도지수": "review_score"}[score_label]
    df_all["sort_score"] = _blend_score(df_all.get("tour_score"), df_all.get("review_score"), score_label)
    df_all = df_all.sort_values([score_col, "distance_km"], ascending=[False, True]) \
                   .drop_duplicates(subset=["title","addr1"], keep="first").reset_index(drop=True)

    df_theme = _filter_by_themes(df_all, cats)
    if df_theme.empty:
        df_theme = df_all.copy()

    anchors = _select_anchors(df_theme, center_lat, center_lon, max_anchors=3)

    DAY_VISIT_MIN, DAY_VISIT_MAX = 4, 6
    per_anchor_visits = 2

    rows_all: List[dict] = []
    used_keys: set = set()

    for day in range(1, days+1):
        if left() < QUICK_FILL_LEFT_SEC:
            _quick_fill_remaining_days(rows_all, df_theme, day, days, start_time, end_time)
            break

        midnight0 = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        day_start = _to_dt(midnight0 + timedelta(days=day-1), start_time)
        day_end   = _to_dt(midnight0 + timedelta(days=day-1), end_time)
        cur_time  = day_start
        day_label = f"{day}일차"

        target_visits = min(DAY_VISIT_MAX, max(DAY_VISIT_MIN, per_anchor_visits * len(anchors)))
        quota = _allocate_quota_weighted(cats, target_visits)
        meal_enabled = ("음식" in cats)

        transit_used_today = False
        prev_visit_row: Optional[pd.Series] = None

        for ai, anch in enumerate(anchors):
            if not has(0.2):
                break

            local = _around_anchor(df_theme, anch["lat"], anch["lon"], within_km=2.5)
            picks = _pick_visits_from_pool(local, cats, quota, per_anchor_visits,
                                           used_keys, cur_time, day_end, meal_enabled)
            for rec in picks:
                st = cur_time
                et = st + timedelta(minutes=_stay_minutes(rec.get("cat1","")))
                if et > day_end: break
                rows_all.append(_visit_row(day_label, day, st, et, rec))
                cur_time = et
                prev_visit_row = rec

            if ai < len(anchors)-1 and prev_visit_row is not None and has(0.2):
                nxt = anchors[ai+1]
                d_km = _haversine(prev_visit_row["lat"], prev_visit_row["lon"], nxt["lat"], nxt["lon"])

                rel, t1, t2 = _relation_and_text_with_names(
                    region, prev_visit_row, pd.Series(nxt), d_km, df_all
                )

                move_min = _estimate_transit_minutes(d_km, rel)
                if (not transit_used_today) and (rel in {"subway_hint","bus_hint"}):
                    move_min = max(move_min, DAILY_TRANSIT_MIN_MIN)

                m_end = cur_time + timedelta(minutes=move_min)
                if m_end <= day_end:
                    rows_all.append(_move_row(day_label, day, cur_time, m_end,
                                              prev_visit_row, nxt, d_km, t1, t2, move_min))
                    cur_time = m_end
                    if rel in {"subway_hint","bus_hint"}:
                        transit_used_today = True

        if has(0.15) and sum(quota.values()) > 0 and len(anchors) >= 1:
            tail = anchors[-1]
            pool = _around_anchor(df_theme, tail["lat"], tail["lon"], within_km=3.0)
            picks = _pick_visits_from_pool(pool, cats, quota, 2, used_keys, cur_time, day_end, meal_enabled)
            for rec in picks:
                st = cur_time
                et = st + timedelta(minutes=_stay_minutes(rec.get("cat1","")))
                if et > day_end: break
                rows_all.append(_visit_row(day_label, day, st, et, rec))
                cur_time = et

        if not transit_used_today:
            vs = [r for r in rows_all if r["day"] == day and r["title"] != "이동"]
            if len(vs) >= 2:
                a, b = vs[-2], vs[-1]
                d_km = _haversine(a["mapy"], a["mapx"], b["mapy"], b["mapx"])
                prev_stub = pd.Series({"title": a["title"], "addr1": a["addr1"], "lat": a["mapy"], "lon": a["mapx"]})
                next_stub = pd.Series({"title": b["title"], "addr1": b["addr1"], "lat": b["mapy"], "lon": b["mapx"]})
                rel, t1, t2 = _relation_and_text_with_names(region, prev_stub, next_stub, d_km, df_all)
                st_dt = _parse_hhmm(vs[-1]["end_time"])
                m_end = st_dt + timedelta(minutes=DAILY_TRANSIT_MIN_MIN)
                if m_end <= day_end:
                    rows_all.append({
                        "day_label": day_label, "day": day,
                        "start_time": st_dt.strftime("%H:%M"),
                        "end_time": m_end.strftime("%H:%M"),
                        "title": "이동", "addr1": "", "cat1": "", "cat2": "", "cat3": "",
                        "출발지": a["addr1"], "교통편1": t1, "교통편2": t2, "도착지": b["addr1"],
                        "final_score": np.nan, "distance_from_prev_km": round(d_km, 2),
                        "move_min": DAILY_TRANSIT_MIN_MIN, "stay_min": 0,
                        "mapx": np.nan, "mapy": np.nan,
                    })

        day_rows_idx = [i for i, r in enumerate(rows_all) if r["day"] == day]
        if day_rows_idx:
            last_idx = day_rows_idx[-1]
            if rows_all[last_idx]["title"] == "이동":
                last_move = rows_all[last_idx]
                cur_time2 = _parse_hhmm(last_move["end_time"])
                if cur_time2 < day_end and anchors:
                    tail = anchors[-1]
                    pool = _around_anchor(df_theme, tail["lat"], tail["lon"], within_km=3.0)
                    pick = _first_fitting_visit(pool, used_keys, cur_time2, day_end)
                    if pick is not None:
                        rows_all.append(_visit_row(day_label, day, cur_time2,
                                                   cur_time2 + timedelta(minutes=_stay_minutes(pick.get("cat1",""))),
                                                   pick))
                    else:
                        rows_all.pop()
                else:
                    rows_all.pop()

    cols = ["day_label","day","start_time","end_time","title","addr1",
            "cat1","cat2","cat3","출발지","교통편1","교통편2","도착지",
            "final_score","distance_from_prev_km","move_min","stay_min","mapx","mapy"]
    return pd.DataFrame(rows_all, columns=cols)

# ─────────────────────────────────────────────────────────────────────────────
# 폴백(남은 시간이 거의 없을 때 빠르게 남은 날 채움)
# ─────────────────────────────────────────────────────────────────────────────
def _quick_fill_remaining_days(rows_all, df_theme, cur_day, total_days, start_time, end_time):
    TARGET_MIN_PER_DAY, TARGET_MAX_PER_DAY = 3, 4
    midnight0 = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    pool = df_theme.sort_values(["sort_score"], ascending=False).drop_duplicates(
        subset=["title","addr1"], keep="first"
    ).reset_index(drop=True)

    used = {(_nfc(r["title"]), _nfc(r["addr1"])) for r in rows_all if r.get("title")}
    i = 0
    for day in range(cur_day, total_days+1):
        day_label = f"{day}일차"
        day_start = _to_dt(midnight0 + timedelta(days=day-1), start_time)
        day_end   = _to_dt(midnight0 + timedelta(days=day-1), end_time)
        cur_time  = day_start

        visits = []
        want = TARGET_MAX_PER_DAY
        while len(visits) < want and i < len(pool):
            r = pool.iloc[i]; i += 1
            key = (_nfc(r["title"]), _nfc(r["addr1"]))
            if key in used: continue
            st = cur_time
            et = st + timedelta(minutes=_stay_minutes(r.get("cat1","")))
            if et > day_end: break
            rows_all.append(_visit_row(day_label, day, st, et, r))
            used.add(key)
            cur_time = et
            visits.append(r)

        if len(visits) >= 2:
            a, b = visits[-2], visits[-1]
            st = _parse_hhmm(rows_all[-1]["end_time"])
            m_end = st + timedelta(minutes=60)
            if m_end <= day_end:
                rows_all.append({
                    "day_label": day_label, "day": day,
                    "start_time": st.strftime("%H:%M"), "end_time": m_end.strftime("%H:%M"),
                    "title": "이동", "addr1": "", "cat1": "", "cat2": "", "cat3": "",
                    "출발지": _nfc(a.get("addr1") or a.get("title")),
                    "교통편1": "대중교통(간단)", "교통편2": "대중교통(간단)",
                    "도착지": _nfc(b.get("addr1") or b.get("title")),
                    "final_score": np.nan, "distance_from_prev_km": np.nan,
                    "move_min": 60, "stay_min": 0, "mapx": np.nan, "mapy": np.nan,
                })
                j = i
                while j < len(pool):
                    r = pool.iloc[j]; j += 1
                    key = (_nfc(r["title"]), _nfc(r["addr1"]))
                    if key in used: continue
                    st2 = m_end
                    et2 = st2 + timedelta(minutes=_stay_minutes(r.get("cat1","")))
                    if et2 <= day_end:
                        rows_all.append(_visit_row(day_label, day, st2, et2, r))
                        used.add(key)
                    break
                i = j

# ─────────────────────────────────────────────────────────────────────────────
# 방문/이동 행 생성
# ─────────────────────────────────────────────────────────────────────────────
def _stay_minutes(cat1: str) -> int:
    c = _nfc(cat1)
    if c == _nfc("음식"):   return 75
    if c == _nfc("자연"):   return 90
    if c == _nfc("레포츠"): return 120
    return 90

def _visit_row(day_label: str, day: int, st: datetime, et: datetime, rec: pd.Series) -> dict:
    return {
        "day_label": day_label, "day": day,
        "start_time": st.strftime("%H:%M"), "end_time": et.strftime("%H:%M"),
        "title": _nfc(rec.get("title","")), "addr1": _nfc(rec.get("addr1","")),
        "cat1": _nfc(rec.get("cat1","")), "cat2": _nfc(rec.get("cat2","")), "cat3": _nfc(rec.get("cat3","")),
        "출발지": "", "교통편1": "", "교통편2": "", "도착지": "",
        "final_score": float(pd.to_numeric(rec.get("sort_score", 0.0), errors="coerce") or 0.0),
        "distance_from_prev_km": 0.0, "move_min": 0, "stay_min": int(_stay_minutes(rec.get("cat1",""))),
        "mapx": float(rec.get("lon", np.nan)), "mapy": float(rec.get("lat", np.nan)),
    }

def _move_row(day_label: str, day: int, st: datetime, et: datetime, prev_visit: pd.Series, nxt_anchor: dict,
              d_km: float, t1: str, t2: str, move_min: int) -> dict:
    return {
        "day_label": day_label, "day": day,
        "start_time": st.strftime("%H:%M"), "end_time": et.strftime("%H:%M"),
        "title": "이동", "addr1": "", "cat1": "", "cat2": "", "cat3": "",
        "출발지": _nfc(prev_visit.get("addr1") or prev_visit.get("title")),
        "교통편1": t1, "교통편2": t2, "도착지": _nfc(nxt_anchor.get("addr1") or nxt_anchor.get("title")),
        "final_score": np.nan, "distance_from_prev_km": round(float(d_km), 2),
        "move_min": int(move_min), "stay_min": 0, "mapx": np.nan, "mapy": np.nan,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 이동 수단/정류장·역 이름 산출
# (1) 위경도 반경 탐색 → (2) 동서남북 오프셋 → (3) addr1 단계적 축약+반경탐색 → (4) 전역 폴백 → (5) 휴리스틱
# ─────────────────────────────────────────────────────────────────────────────
def _relation_and_text_with_names(region: str, prev_row: pd.Series, nxt_row: pd.Series,
                                  d_km: float, df_all: pd.DataFrame):
    plat, plon = float(prev_row["lat"]), float(prev_row["lon"])
    nlat, nlon = float(nxt_row["lat"]), float(nxt_row["lon"])

    # 1) 카카오 키가 있으면 실제 주변 POI 탐색
    if KAKAO_API_KEY:
        sub1 = _nearest_subway_api(plat, plon)
        sub2 = _nearest_subway_api(nlat, nlon)
        if sub1[0] and sub2[0] and _nfc(sub1[0]) != _nfc(sub2[0]):
            return "subway_hint", f"지하철 {_line_station_text(sub1[1], sub1[0])} 승차", f"{_line_station_text(sub2[1], sub2[0])} 하차"
        bus1 = _nearest_bus_api(plat, plon)
        bus2 = _nearest_bus_api(nlat, nlon)
        if bus1 and bus2 and _nfc(bus1) != _nfc(bus2):
            return "bus_hint", f"버스 {bus1} 승차", f"{bus2} 하차"

    # 2) CSV 근접(반경 확장 + 오프셋)
    sub1_local = _nearest_subway_local(plat, plon, df_all)
    sub2_local = _nearest_subway_local(nlat, nlon, df_all)
    if sub1_local and sub2_local and _nfc(sub1_local) != _nfc(sub2_local) and _in_subway_region(region):
        return "subway_hint", f"지하철 {sub1_local} 승차", f"{sub2_local} 하차"

    bus1_local = _nearest_bus_local(plat, plon, df_all)
    bus2_local = _nearest_bus_local(nlat, nlon, df_all)
    if bus1_local and bus2_local and _nfc(bus1_local) != _nfc(bus2_local):
        return "bus_hint", f"버스 {bus1_local} 승차", f"{bus2_local} 하차"

    # 3) addr1 축약 + 반경탐색(100→200→300→400→500m) — 마지막 단어 0/1/2개 삭제까지
    sub1_txt = _nearest_transit_by_addr(prev_row.get("addr1",""), df_all, prefer="subway", lat=plat, lon=plon)
    sub2_txt = _nearest_transit_by_addr(nxt_row.get("addr1",""), df_all, prefer="subway", lat=nlat, lon=nlon)
    if sub1_txt and sub2_txt and _nfc(sub1_txt) != _nfc(sub2_txt) and _in_subway_region(region):
        return "subway_hint", f"지하철 {sub1_txt} 승차", f"{sub2_txt} 하차"

    bus1_txt = _nearest_transit_by_addr(prev_row.get("addr1",""), df_all, prefer="bus", lat=plat, lon=plon)
    bus2_txt = _nearest_transit_by_addr(nxt_row.get("addr1",""), df_all, prefer="bus", lat=nlat, lon=nlon)
    if bus1_txt and bus2_txt and _nfc(bus1_txt) != _nfc(bus2_txt):
        return "bus_hint", f"버스 {bus1_txt} 승차", f"{bus2_txt} 하차"

    # 4) 전역 폴백: 데이터셋 전체에서 '실제 POI' 최단 후보 강제 매핑
    sub1_glob = _nearest_any_transit_global("subway", plat, plon, df_all)
    sub2_glob = _nearest_any_transit_global("subway", nlat, nlon, df_all)
    if sub1_glob and sub2_glob and _nfc(sub1_glob) != _nfc(sub2_glob) and _in_subway_region(region):
        return "subway_hint", f"지하철 {sub1_glob} 승차", f"{sub2_glob} 하차"

    bus1_glob = _nearest_any_transit_global("bus", plat, plon, df_all)
    bus2_glob = _nearest_any_transit_global("bus", nlat, nlon, df_all)
    if bus1_glob and bus2_glob and _nfc(bus1_glob) != _nfc(bus2_glob):
        return "bus_hint", f"버스 {bus1_glob} 승차", f"{bus2_glob} 하차"

    # 4-1) 한쪽 모드가 비었으면 다른 모드로라도 강제 매핑
    if _in_subway_region(region):
        if not (sub1_glob and sub2_glob):
            if bus1_glob and bus2_glob:
                return "bus_hint", f"버스 {bus1_glob} 승차", f"{bus2_glob} 하차"
    else:
        if not (bus1_glob and bus2_glob):
            if sub1_glob and sub2_glob:
                return "subway_hint", f"지하철 {sub1_glob} 승차", f"{sub2_glob} 하차"

    # 5) 최후 휴리스틱(거의 오지 않음)
    if _in_subway_region(region) and 4.0 <= d_km <= 20.0:
        return "subway_hint", "지하철 승차", "지하철 하차"
    if d_km >= ANCHOR_HOP_MIN_KM:
        return "bus_hint", "버스 승차", "버스 하차"
    if d_km < WALK_SKIP_KM:
        return "walk_hint", "", ""
    return "bus_hint", "버스 승차", "버스 하차"

def _in_subway_region(region: str) -> bool:
    r = _nfc(region)
    return any(k in r for k in SUBWAY_REGIONS)

# ── API 기반 근접 ──
def _nearest_subway_api(lat, lon) -> Tuple[str,str]:
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params  = {"category_group_code":"SW8","x": lon,"y": lat,"radius": 900,"size": 1,"sort":"distance"}
    try:
        r = requests.get("https://dapi.kakao.com/v2/local/search/category.json", headers=headers, params=params, timeout=2.5)
        if r.ok and (docs := r.json().get("documents")):
            d = docs[0]
            name = _nfc(d.get("place_name",""))
            raw  = " ".join([name,_nfc(d.get("category_name","")),_nfc(d.get("address_name","")),_nfc(d.get("road_address_name",""))])
            m = re.search(r"(\d+)\s*호선", raw)
            line = f"{m.group(1)}호선" if m else ""
            return name, line
    except Exception:
        pass
    return "",""

def _nearest_bus_api(lat, lon) -> str:
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    for r_km in (700, 900, 1200):
        try:
            r = requests.get("https://dapi.kakao.com/v2/local/search/keyword.json",
                             headers=headers, params={"query":"버스정류장","x":lon,"y":lat,"radius":int(r_km),"size":1,"sort":"distance"},
                             timeout=2.5)
            if r.ok and (docs := r.json().get("documents")):
                return _nfc(docs[0].get("place_name",""))
        except Exception:
            continue
    return ""

# ── CSV 기반 근접(반경 확장 + 동서남북 오프셋) ──
_OFFSETS_DEG = [0.0, 0.0010, -0.0010, 0.0016, -0.0016]  # ≈0m, 110m, -110m, 180m, -180m
_RADII_KM    = [0.10, 0.20, 0.30, 0.50]

def _nearest_subway_local(lat: float, lon: float, df: pd.DataFrame) -> str:
    cand = df.copy()
    mask = cand["title"].str.contains("역", na=False) | cand["cat3"].str.contains("지하철|전철", na=False)
    cand = cand[mask].copy()
    if cand.empty: return ""
    for dy in _OFFSETS_DEG:
        for dx in _OFFSETS_DEG:
            lat0, lon0 = lat + dy, lon + dx
            cand["_d"] = _haversine_np(lat0, lon0, cand["lat"], cand["lon"])
            for r in _RADII_KM:
                sub = cand[cand["_d"] <= r].sort_values(["_d","sort_score"], ascending=[True,False]).head(1)
                if not sub.empty:
                    name = _nfc(sub.iloc[0]["title"])
                    return name if name.endswith("역") else (name + "역")
    return ""

def _nearest_bus_local(lat: float, lon: float, df: pd.DataFrame) -> str:
    cand = df.copy()
    mask = cand["title"].str.contains("정류장|버스", na=False) | cand["cat3"].str.contains("버스", na=False)
    cand = cand[mask].copy()
    if cand.empty: return ""
    for dy in _OFFSETS_DEG:
        for dx in _OFFSETS_DEG:
            lat0, lon0 = lat + dy, lon + dx
            cand["_d"] = _haversine_np(lat0, lon0, cand["lat"], cand["lon"])
            for r in _RADII_KM:
                sub = cand[cand["_d"] <= r].sort_values(["_d","sort_score"], ascending=[True,False]).head(1)
                if not sub.empty:
                    name = _nfc(sub.iloc[0]["title"])
                    return name if name.endswith("정류장") else (name + " 정류장")
    return ""

# ── addr1 축약 + 반경탐색(요청사항 강화) ──
def _nearest_transit_by_addr(addr1: str, df: pd.DataFrame, prefer: str, lat: float, lon: float) -> str:
    """
    addr1을 기준으로:
      - 원문(prefix 100%) → 뒤 단어 1개 제거 → 2개 제거 (총 3회)
      - 각 시도마다 반경 100→200→300→400→500m로 확장하며 가장 가까운 후보 선택
      - 지하철/버스 타입 필터링
    """
    a = _nfc(addr1)
    if not a:
        return ""
    tokens = [t for t in a.split() if t]

    # 시도: 0(원문), 1(뒤 1개 삭제), 2(뒤 2개 삭제)
    tries = []
    if len(tokens) >= 1: tries.append(" ".join(tokens))
    if len(tokens) >= 2: tries.append(" ".join(tokens[:-1]))
    if len(tokens) >= 3: tries.append(" ".join(tokens[:-2]))

    # 반경 리스트(m→km)
    radii_km = [0.10, 0.20, 0.30, 0.40, 0.50]

    for prefix in tries:
        if prefer == "subway":
            mask = (df["addr1"].str.contains(prefix, na=False)) & \
                   (df["title"].str.contains("역", na=False) | df["cat3"].str.contains("지하철|전철", na=False))
        else:
            mask = (df["addr1"].str.contains(prefix, na=False)) & \
                   (df["title"].str.contains("정류장|버스", na=False) | df["cat3"].str.contains("버스", na=False))
        cand = df[mask].copy()
        if cand.empty:
            continue

        # 각 반경별로 가장 가까운 후보를 즉시 반환
        for r in radii_km:
            cand["_d"] = _haversine_np(lat, lon, cand["lat"], cand["lon"])
            sub = cand[cand["_d"] <= r].sort_values(["_d","sort_score"], ascending=[True,False]).head(1)
            if not sub.empty:
                name = _nfc(sub.iloc[0]["title"])
                if prefer == "subway":
                    return name if name.endswith("역") else (name + "역")
                else:
                    return name if name.endswith("정류장") else (name + " 정류장")

        # 반경 내 없으면, 전체 cand 중 가장 가까운 것(마지막 보정)
        cand["_d"] = _haversine_np(lat, lon, cand["lat"], cand["lon"])
        cand = cand.sort_values(["_d","sort_score"], ascending=[True,False]).head(1)
        if not cand.empty:
            name = _nfc(cand.iloc[0]["title"])
            if prefer == "subway":
                return name if name.endswith("역") else (name + "역")
            else:
                return name if name.endswith("정류장") else (name + " 정류장")

    return ""

# ── 전역 폴백(실제 POI) ──
def _nearest_any_transit_global(kind: str, lat: float, lon: float, df: pd.DataFrame) -> str:
    """
    데이터셋 전체에서 가장 가까운 '실제' 대중교통 POI(역/정류장)를 반환.
    kind: 'subway' | 'bus'
    """
    cand = df.copy()
    if kind == "subway":
        mask = cand["title"].str.contains("역", na=False) | cand["cat3"].str.contains("지하철|전철", na=False)
    else:
        mask = cand["title"].str.contains("정류장|버스", na=False) | cand["cat3"].str.contains("버스", na=False)
    cand = cand[mask].copy()
    if cand.empty:
        return ""
    cand["_d"] = _haversine_np(lat, lon, cand["lat"], cand["lon"])
    best = cand.sort_values(["_d","sort_score"], ascending=[True,False]).head(1)
    if best.empty:
        return ""
    nm = _nfc(best.iloc[0]["title"])
    if kind == "subway":
        return nm if nm.endswith("역") else (nm + "역")
    else:
        return nm if nm.endswith("정류장") else (nm + " 정류장")

def _nearest_either_global(lat: float, lon: float, df: pd.DataFrame) -> Tuple[str, str]:
    sub = _nearest_any_transit_global("subway", lat, lon, df)
    bus = _nearest_any_transit_global("bus",    lat, lon, df)
    return sub, bus

# ─────────────────────────────────────────────────────────────────────────────
# 선택/클러스터/배치
# ─────────────────────────────────────────────────────────────────────────────
def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    need = {
        "title": cols.get("title") or "title",
        "addr1": cols.get("addr1") or "addr1",
        "cat1":  cols.get("cat1")  or "cat1",
        "cat2":  cols.get("cat2")  or "cat2",
        "cat3":  cols.get("cat3")  or "cat3",
        "mapx":  cols.get("mapx")  or cols.get("lon") or cols.get("longitude") or "mapx",
        "mapy":  cols.get("mapy")  or cols.get("lat")  or cols.get("latitude")  or "mapy",
        "review_score": cols.get("review_score") or "review_score",
        "tour_score":   cols.get("tour_score")   or "tour_score",
    }
    std = df.rename(columns={
        need["title"]: "title", need["addr1"]: "addr1",
        need["cat1"]: "cat1", need["cat2"]: "cat2", need["cat3"]: "cat3",
        need["mapx"]: "lon", need["mapy"]: "lat",
        need["review_score"]: "review_score", need["tour_score"]: "tour_score",
    }).copy()

    for c in ("lon","lat","review_score","tour_score"):
        std[c] = pd.to_numeric(std.get(c), errors="coerce")
    std["title"] = std["title"].astype(str)
    std["addr1"] = std["addr1"].astype(str)
    for c in ("cat1","cat2","cat3"):
        if c not in std.columns: std[c] = ""
        std[c] = std[c].fillna("").astype(str)

    std["cat3"] = std["cat3"].str.replace(r"[;/·|]", ",", regex=True)
    std["cat3"] = std["cat3"].str.replace(r"\s*,\s*", ",", regex=True)
    std = std.dropna(subset=["lon","lat"]).copy()
    return std

def _filter_by_themes(df: pd.DataFrame, cats: List[str]) -> pd.DataFrame:
    cats = [_nfc(c) for c in (cats or [])]
    if not cats: return df.copy()
    def _match(r, theme):
        text = _nfc(f"{r.get('cat1','')} {r.get('cat2','')} {r.get('cat3','')}")
        bag = THEME_SYNONYM.get(theme, {theme})
        return any(_nfc(k) in text for k in bag)
    mask = df.apply(lambda r: any(_match(r, c) for c in cats), axis=1)
    return df[mask].copy()

def _select_anchors(df: pd.DataFrame, lat: float, lon: float, max_anchors: int = 3) -> List[Dict]:
    rows = df.copy()
    rows["_d0"] = _haversine_np(lat, lon, rows["lat"], rows["lon"])
    anchors: List[Dict] = []
    first = rows.sort_values(["_d0","sort_score"], ascending=[True,False]).head(1)
    if first.empty: return []
    a0 = first.iloc[0]
    anchors.append({"title": a0["title"], "addr1": a0["addr1"], "lat": float(a0["lat"]), "lon": float(a0["lon"])})
    cur_lat, cur_lon = float(a0["lat"]), float(a0["lon"])
    remain = rows.drop(index=first.index).copy()
    while len(anchors) < max_anchors and not remain.empty:
        remain["_dprev"] = _haversine_np(cur_lat, cur_lon, remain["lat"], remain["lon"])
        far = remain[remain["_dprev"] >= ANCHOR_HOP_MIN_KM]
        if far.empty: break
        nxt = far.sort_values(["_dprev","sort_score"], ascending=[True,False]).head(1).iloc[0]
        anchors.append({"title": nxt["title"], "addr1": nxt["addr1"], "lat": float(nxt["lat"]), "lon": float(nxt["lon"])})
        cur_lat, cur_lon = float(nxt["lat"]), float(nxt["lon"])
        remain = remain.drop(index=[nxt.name])
    return anchors

def _around_anchor(df: pd.DataFrame, lat: float, lon: float, within_km: float = 2.5) -> pd.DataFrame:
    sub = df.copy()
    sub["_d_anchor"] = _haversine_np(lat, lon, sub["lat"], sub["lon"])
    return sub[sub["_d_anchor"] <= within_km].sort_values(["sort_score","_d_anchor"], ascending=[False,True]).copy()

def _allocate_quota_weighted(cats: List[str], want: int) -> Dict[str, int]:
    C = [_nfc(c) for c in cats]; L = len(C)
    if want <= 0 or L == 0: return {c: 0 for c in C}
    if L == 1: weights = [1.0]
    elif L == 2: weights = [0.7, 0.3]
    else: weights = [0.6, 0.3, 0.1][:L]
    base = [max(0, int(round(w * want))) for w in weights]
    diff = want - sum(base)
    i = 0
    while diff > 0: base[i % L] += 1; i += 1; diff -= 1
    i = L - 1
    while diff < 0:
        if base[i] > 0: base[i] -= 1; diff += 1
        i = (i - 1) % L
    quota = {C[j]: base[j] for j in range(L)}
    # 각 테마 최소 1개 보장
    needs = [c for c in C if quota.get(c, 0) == 0]
    for c in needs:
        for k in C:
            if k != c and quota.get(k, 0) > 1:
                quota[k] -= 1; quota[c] = 1; break
    return quota

def _pick_visits_from_pool(
    pool: pd.DataFrame, cats: List[str], quota: Dict[str,int], want: int,
    used_keys: set, cur_time: datetime, day_end: datetime, meal_enabled: bool
) -> List[pd.Series]:
    if pool.empty or want <= 0: return []
    out: List[pd.Series] = []

    def _rec_ok(rec: pd.Series) -> bool:
        key = (_nfc(rec.get("title","")), _nfc(rec.get("addr1","")))
        return key not in used_keys

    def _pop_by_theme(theme: str) -> Optional[pd.Series]:
        bag = THEME_SYNONYM.get(theme, {theme})
        mask = pool.apply(lambda r: _nfc(r["cat1"]) in bag or any(_nfc(k) in _nfc(f"{r['cat2']} {r['cat3']}") for k in bag), axis=1)
        cand = pool[mask].copy()
        if cand.empty: return None
        cand = cand.sort_values(["sort_score","_d_anchor"], ascending=[False,True])
        for _, row in cand.iterrows():
            if _rec_ok(row): return row
        return None

    def _pop_food(is_main: bool) -> Optional[pd.Series]:
        cand = pool[pool["cat1"].map(_nfc) == _nfc(MEAL_CAT)].copy()
        if cand.empty: return None
        def ok(row):
            tags = {t.strip() for t in (str(row.get("cat2","")) + "," + str(row.get("cat3",""))).split(",") if t.strip()}
            is_cafe = bool(tags & CAFE_KEYWORDS)
            is_main_tag = bool(tags & MEAL_MAIN_KEYWORDS) or not is_cafe
            return (is_main and is_main_tag) or ((not is_main) and is_cafe)
        cand["ok"] = cand.apply(ok, axis=1)
        cand = cand[cand["ok"]].drop(columns=["ok"])
        if cand.empty:
            cand = pool[pool["cat1"].map(_nfc) == _nfc(MEAL_CAT)].copy()
        cand = cand.sort_values(["sort_score","_d_anchor"], ascending=[False,True])
        for _, row in cand.iterrows():
            if _rec_ok(row): return row
        return None

    cur_min = cur_time.hour*60 + cur_time.minute

    # 음식 시간대 우선
    if meal_enabled and quota.get(_nfc(MEAL_CAT),0) > 0 and len(out) < want:
        if (LUNCH_START <= cur_min < LUNCH_END) or (DINNER_START <= cur_min < DINNER_END):
            r = _pop_food(is_main=True)
            if r is not None:
                out.append(r); used_keys.add((_nfc(r["title"]), _nfc(r["addr1"]))); quota[_nfc(MEAL_CAT)] -= 1
        elif cur_min >= NIGHT_AFTER:
            r = _pop_food(is_main=False)
            if r is not None:
                out.append(r); used_keys.add((_nfc(r["title"]), _nfc(r["addr1"]))); quota[_nfc(MEAL_CAT)] -= 1

    # 테마 라운드로빈
    order = [_nfc(c) for c in cats]
    i = 0; safety = 0
    while len(out) < want and safety < 200:
        safety += 1
        theme = order[i % len(order)]; i += 1
        if quota.get(theme,0) <= 0: continue
        if theme == _nfc(MEAL_CAT):  # 음식은 시간대 외 미배치
            continue
        r = _pop_by_theme(theme)
        if r is not None:
            out.append(r); used_keys.add((_nfc(r["title"]), _nfc(r["addr1"]))); quota[theme] -= 1

    # 부족분 아무거나(음식 규칙 준수)
    j = 0
    while len(out) < want and j < len(pool):
        r = pool.iloc[j]; j += 1
        if not _rec_ok(r): continue
        if _nfc(r.get("cat1","")) == _nfc(MEAL_CAT):
            if (LUNCH_START <= cur_min < LUNCH_END) or (DINNER_START <= cur_min < DINNER_END):
                pass
            elif cur_min >= NIGHT_AFTER:
                tags = {t.strip() for t in (str(r.get("cat2","")) + "," + str(r.get("cat3",""))).split(",") if t.strip()}
                if not (tags & CAFE_KEYWORDS): continue
            else:
                continue
        out.append(r); used_keys.add((_nfc(r["title"]), _nfc(r["addr1"])))
    return out[:want]

def _first_fitting_visit(pool: pd.DataFrame, used_keys: set, st: datetime, day_end: datetime) -> Optional[pd.Series]:
    if pool.empty: return None
    for _, r in pool.iterrows():
        key = (_nfc(r.get("title","")), _nfc(r.get("addr1","")))
        if key in used_keys: continue
        et = st + timedelta(minutes=_stay_minutes(r.get("cat1","")))
        if et <= day_end:
            used_keys.add(key)
            return r
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 점수/지오코딩/빈 DF
# ─────────────────────────────────────────────────────────────────────────────
def _blend_score(tour_s, review_s, score_label: str) -> pd.Series:
    def _minmax(s):
        s = pd.to_numeric(s, errors="coerce").fillna(0)
        mn, mx = float(np.nanmin(s)), float(np.nanmax(s))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return pd.Series([0.0]*len(s), index=s.index)
        return (s - mn) / (mx - mn)
    ts = _minmax(tour_s); rs = _minmax(review_s)
    return 0.70*ts + 0.30*rs if score_label == "관광지수" else 0.35*ts + 0.65*rs

def _read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ("utf-8","utf-8-sig","cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def _geocode_region_kakao(region: str) -> Optional[Tuple[float,float]]:
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    try:
        r = requests.get("https://dapi.kakao.com/v2/local/search/keyword.json",
                         headers=headers, params={"query": region, "size": 1}, timeout=2.5)
        if r.status_code != 200: return None
        docs = r.json().get("documents", [])
        if not docs: return None
        return float(docs[0]["y"]), float(docs[0]["x"])
    except Exception:
        return None

def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "day_label","day","start_time","end_time","title","addr1",
        "cat1","cat2","cat3","출발지","교통편1","교통편2","도착지",
        "final_score","distance_from_prev_km","move_min","stay_min","mapx","mapy"
    ])

def _line_station_text(line, name):
    return (f"{line} {name}").strip() if line else name
