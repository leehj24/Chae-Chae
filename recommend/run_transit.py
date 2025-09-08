# run_transit.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re, math, requests
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

import pandas as pd
import numpy as np
import unicodedata as ud
from concurrent.futures import ThreadPoolExecutor, as_completed

# ⚠️ 모든 경로/API 키는 config에서만!
# - PATH_TMF      : 관광 POI 원본 CSV 경로
# - KAKAO_API_KEY : 카카오 로컬 API 키
# - ODSAY_API_KEY : (선택) ODsay 대중교통 API 키 (없어도 동작)
from recommend.config import *  # noqa: F403,F401  (프로젝트 컨벤션 유지)

from dotenv import load_dotenv
load_dotenv()
import os

# 환경변수 있으면 우선, 없으면 config 값 유지
_KAKAO_ENV = os.environ.get("KAKAO_API_KEY")
if _KAKAO_ENV:
    KAKAO_API_KEY = _KAKAO_ENV  # noqa: F405  (config의 것을 덮되, env 없으면 그대로)

_ODSAY_ENV = os.environ.get("ODSAY_API_KEY")
if _ODSAY_ENV:
    ODSAY_API_KEY = _ODSAY_ENV  # noqa: F405


# ========================
# Public API
# ========================
def run(
    region: str,
    transport_mode: str,             # 반드시 'transit' 로 호출
    score_label: str,                # '인기도지수' | '관광지수'
    days: int,
    cats: List[str],
    start_time: str = "08:00",       # 필요 시 UI에서 받되, 미입력 시 기본
    end_time: str   = "22:30",
    **_,                              # ← 알 수 없는 키워드(return_df 등) 무시
) -> pd.DataFrame:
    """
    UI 입력만 사용해 대중교통 중심 일정표(DataFrame)를 반환합니다.
    - CSV 저장 없음
    - PATH_TMF/KAKAO_API_KEY/ODsay 키는 config에서 로드
    """

    # ----- 입력 검증 -----
    region = (region or "").strip()
    if not region:
        raise ValueError("여행 지역을 입력하세요.")
    if transport_mode != "transit":
        raise ValueError("교통 모드는 'transit' 여야 합니다. (걷기는 run_walk.py 호출)")
    if score_label not in {"인기도지수", "관광지수"}:
        raise ValueError("점수 기준은 '인기도지수' 또는 '관광지수' 중 선택하세요.")
    if not isinstance(days, int) or days <= 0:
        raise ValueError("여행 일수를 1 이상으로 지정하세요.")
    days = max(1, min(100, int(days)))

    cats = list(dict.fromkeys([_nfc(c) for c in (cats or [])]))
    if not cats:
        raise ValueError("테마를 최소 1개 이상 선택하세요. (예: 음식, 자연, 레포츠)")
    if len(cats) > 3:
        cats = cats[:3]

    _check_hhmm(start_time)
    _check_hhmm(end_time)

    # ----- 지역 지오코딩 -----
    coords = _geocode_region_kakao(region)
    if not coords:
        raise ValueError(f"카카오 지오코딩 실패: '{region}'")
    center_lat, center_lon = coords

    # ----- 원본 CSV 로드(열 표준화) -----
    tmf = _read_csv_robust(PATH_TMF)  # type: ignore[name-defined]
    cols_lower = {c.lower(): c for c in tmf.columns}
    need = {
        "title": cols_lower.get("title"),
        "addr1": cols_lower.get("addr1"),
        "cat1":  cols_lower.get("cat1") or _first_contains(tmf.columns, "cat1"),
        "mapx":  cols_lower.get("mapx") or cols_lower.get("lon") or cols_lower.get("longitude") or cols_lower.get("x"),
        "mapy":  cols_lower.get("mapy") or cols_lower.get("lat") or cols_lower.get("latitude") or cols_lower.get("y"),
        "tour_score": cols_lower.get("tour_score"),
        "review_score": cols_lower.get("review_score"),
    }
    miss = [k for k, v in need.items() if v is None]
    if miss:
        raise KeyError(f"필수 컬럼 누락: {miss} / 실제컬럼: {list(tmf.columns)}")

    df = tmf.rename(columns={
        need["title"]: "title",
        need["addr1"]: "addr1",
        need["cat1"]:  "cat1",
        need["mapx"]:  "lon",     # 경도
        need["mapy"]:  "lat",     # 위도
        need["tour_score"]: "tour_score",
        need["review_score"]: "review_score",
    }).copy()

    # 타입 정리
    for c in ("lat", "lon", "tour_score", "review_score"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()

    # ----- 반경/거리 필터 -----
    radius_km = TRANSIT_RADIUS_KM if FAST_MODE else 20
    df["distance_km"] = _haversine_np(center_lat, center_lon, df["lat"], df["lon"])
    df = df[df["distance_km"] <= radius_km].copy()
    if df.empty:
        raise RuntimeError("선택한 지역 주변(반경)에서 후보가 없습니다. 지역/반경/데이터를 확인하세요.")

    # ----- 정렬 기준 -----
    score_map = {"관광지수": "tour_score", "인기도지수": "review_score"}
    score_col = score_map[score_label]
    df = df.sort_values([score_col, "distance_km"], ascending=[False, True]).copy()
    df = df.drop_duplicates(subset=["title", "addr1"], keep="first").reset_index(drop=True)

    # ----- 선택 카테고리 우선 + 결핍 보충 -----
    FALLBACK_N = 5
    if cats:
        df_sorted = df[df["cat1"].isin(cats)].copy()
    else:
        df_sorted = df.copy()

    pool = df.copy()
    for cat in cats:
        if (df_sorted["cat1"] == cat).sum() == 0:
            remain = pool.loc[~pool.index.isin(df_sorted.index)]
            fb = remain.sort_values(["distance_km", score_col], ascending=[True, False]).head(FALLBACK_N)
            if not fb.empty:
                df_sorted = pd.concat([df_sorted, fb], ignore_index=True)

    # 메타/정규화
    df_sorted["cat1_norm"] = df_sorted["cat1"].map(_nfc)
    df_sorted["cat3_norm"] = df_sorted["cat3"].astype(str).map(_nfc)
    df_sorted["final_score"] = pd.to_numeric(
        df_sorted["review_score"].fillna(df_sorted["tour_score"]), errors="coerce"
    ).fillna(0.0)
    df_sorted = df_sorted.sort_values([score_col, "distance_km"], ascending=[False, True]).reset_index(drop=True)

    # ----- 대중교통 힌트(가까운 지하철/버스 정류장) 병렬 조회 -----
    pois = _enrich_transit_hints_fast(df_sorted, KAKAO_API_KEY)  # type: ignore[name-defined]

    # ----- 간단 라우팅(탐욕)
    route = _greedy_route(pois, float(pois.iloc[0]["lat"]), float(pois.iloc[0]["lon"]))

    # ======================================
    # 스케줄링 (대중교통 + 이동행 삽입)
    # ======================================
    MEAL_CAT = "음식"
    MEAL_CUISINE_TAGS = {"서양식", "이색음식점", "일식", "중식", "한식"}
    DAY_VISIT_MIN, DAY_VISIT_MAX = 5, 6
    WALK_SKIP_KM = 0.30
    BASE_SPEED_KMH, ADD_FIXED_MIN = 18.0, 8.0  # 대중교통 혼합 추정

    def estimate_transit_minutes(d_km: float, rel: str) -> int:
        base = (float(d_km) / BASE_SPEED_KMH) * 60.0 + ADD_FIXED_MIN
        if rel == "same_subway_station": return max(3, int(round(base - 10)))
        if rel == "same_subway_line":    return max(4, int(round(base - 6)))
        if rel == "same_bus_station":    return max(5, int(round(base - 5)))
        return int(round(base))

    # ODsay가 있으면 실제값 우선
    def transit_minutes_via_api_or_est(p: pd.Series, n: pd.Series, rel: str, d_km: float) -> int:
        try:
            if rel in {"subway_hint", "bus_hint"} and ODSAY_API_KEY:  # type: ignore[name-defined]
                tt = _odsay_total_minutes(p["lon"], p["lat"], n["lon"], n["lat"], ODSAY_API_KEY)  # type: ignore[name-defined]
                if isinstance(tt, int) and tt > 0:
                    return tt
        except Exception:
            pass
        return estimate_transit_minutes(d_km, rel)

    def relation_and_text(prev_row, nxt_row, d_km: float):
        ps_raw, pl = _nfc(prev_row.get("closest_subway_station", "")), _nfc(prev_row.get("closest_subway_line", ""))
        ns_raw, nl = _nfc(nxt_row.get("closest_subway_station", "")), _nfc(nxt_row.get("closest_subway_line", ""))
        pb, nb = _nfc(prev_row.get("closest_bus_station", "")), _nfc(nxt_row.get("closest_bus_station", ""))
        ps, ns = _norm_station(ps_raw), _norm_station(ns_raw)

        if ps and ns and ps == ns:  return "same_subway_station", "", ""
        if pb and nb and pb == nb:  return "same_bus_station", "", ""
        if pl and nl and ps and ns and (pl == nl): return "same_subway_line", "", ""
        if d_km < WALK_SKIP_KM:     return "walk_hint", "", ""

        if ps_raw and ns_raw and ps != ns:
            t1 = f"지하철 { _line_station_text(pl, ps_raw) } 승차".strip()
            t2 = f"{ _line_station_text(nl, ns_raw) } 하차".strip()
            return "subway_hint", t1, t2
        if pb and nb and pb != nb:
            return "bus_hint", f"버스 {pb} 승차", f"{nb} 하차"
        return "walk_hint", "", ""

    def possible_transit_hint(prev_row, nxt_row):
        ps_raw, pl = _nfc(prev_row.get("closest_subway_station", "")), _nfc(prev_row.get("closest_subway_line", ""))
        ns_raw, nl = _nfc(nxt_row.get("closest_subway_station", "")), _nfc(nxt_row.get("closest_subway_line", ""))
        pb, nb = _nfc(prev_row.get("closest_bus_station", "")), _nfc(nxt_row.get("closest_bus_station", ""))
        ps, ns = _norm_station(ps_raw), _norm_station(ns_raw)
        if ps_raw and ns_raw and ps and ns and ps != ns:
            t1 = f"지하철 { _line_station_text(pl, ps_raw) } 승차".strip()
            t2 = f"{ _line_station_text(nl, ns_raw) } 하차".strip()
            return "subway_hint", t1, t2
        if pb and nb and pb != nb:
            return "bus_hint", f"버스 {pb} 승차", f"{nb} 하차"
        return None, "", ""

    def stay_minutes(cat1: str) -> int:
        c = _nfc(cat1)
        if c == "음식": return 75
        if c == "자연": return 90
        if c == "레포츠": return 120
        return 90

    # 하루 목표 방문 수 분배
    visit_counts = _split_visits(len(route), days, DAY_VISIT_MIN, DAY_VISIT_MAX)

    # 스케줄링 루프
    rows_all: List[dict] = []
    pos = 0
    midnight0 = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    used_titles_global: set = set()
    meal_enabled_flag = ("음식" in set(cats))

    for d in range(1, days + 1):
        base  = midnight0 + timedelta(days=d - 1)
        label = f"{d}일"
        want  = visit_counts[d - 1] if d - 1 < len(visit_counts) else 0
        if want <= 0:
            break

        # 하루 풀 (비음식 최소 보장 보정)
        pool = _ensure_variety_pool(route, pos, used_titles_global, want)
        if pool.empty:
            break

        # 시간 경계
        day_start = _to_dt(base, start_time)
        day_end   = _to_dt(base, end_time)
        lunch_s, lunch_e   = base.replace(hour=11, minute=0), base.replace(hour=13, minute=0)
        dinner_s, dinner_e = base.replace(hour=17, minute=0), base.replace(hour=20, minute=0)

        # 쿼터(3,2,1)
        quota = _allocate_day_quota(cats, want)
        cur_time = day_start
        used_idx = set()
        prev_row = None
        transit_used_today = False

        # 내부 헬퍼
        def already_used_globally(r: pd.Series) -> bool:
            return (_nfc(r.get("title", "")), _nfc(r.get("addr1", ""))) in used_titles_global

        def pick_best(sub: pd.DataFrame, ref_lat: float, ref_lon: float):
            if sub.empty:
                return None
            dkm = np.sqrt((sub["lat"] - ref_lat) ** 2 + (sub["lon"] - ref_lon) ** 2) * 111.0
            pen = dkm.apply(lambda x: (x / BASE_SPEED_KMH) * 60.0) / 60.0
            sc  = sub.get("final_score", pd.Series([0] * len(sub))).fillna(0) - 0.1 * pen
            return sc.sort_values(ascending=False).index[0]

        def place_visit(idx: int):
            nonlocal cur_time, prev_row, transit_used_today
            n = pool.loc[idx]
            if already_used_globally(n):
                return False

            # 이동행 삽입
            if prev_row is not None:
                d_km = _haversine(prev_row["lat"], prev_row["lon"], n["lat"], n["lon"])
                rel, t1, t2 = relation_and_text(prev_row, n, d_km)

                # 아직 오늘 교통 미사용이면, 교통 유발 후보 우선 시도
                if (rel in {"same_subway_station", "same_bus_station", "same_subway_line", "walk_hint"}) and (not transit_used_today):
                    best_j, best_sc = None, 1e9
                    for j, r2 in pool.iterrows():
                        if j in used_idx: continue
                        if already_used_globally(r2): continue
                        rel2, _, _ = possible_transit_hint(prev_row, r2)
                        if rel2 in {"subway_hint", "bus_hint"} and quota.get(_nfc(r2.get("cat1", "")), 0) > 0:
                            sc = _step_cost_from(prev_row["lat"], prev_row["lon"], r2)
                            if sc < best_sc:
                                best_sc, best_j = sc, j
                    if best_j is not None:
                        idx = best_j
                        n = pool.loc[idx]
                        d_km = _haversine(prev_row["lat"], prev_row["lon"], n["lat"], n["lon"])
                        rel, t1, t2 = relation_and_text(prev_row, n, d_km)

                if rel not in {"same_subway_station", "same_bus_station", "same_subway_line", "walk_hint"}:
                    move_min = transit_minutes_via_api_or_est(prev_row, n, rel, d_km)
                    m_end = cur_time + timedelta(minutes=move_min)
                    if m_end > day_end:
                        return False
                    rows_all.append({
                        "day_label": label, "day": d,
                        "start_time": cur_time.strftime("%H:%M"), "end_time": m_end.strftime("%H:%M"),
                        "title": "이동", "addr1": "", "cat1": "", "cat2": "", "cat3": "",
                        "출발지": _nfc(prev_row.get("addr1") or prev_row.get("title")),
                        "교통편1": t1, "교통편2": t2,
                        "도착지": _nfc(n.get("addr1") or n.get("title")),
                        "final_score": np.nan, "distance_from_prev_km": round(d_km, 2),
                        "move_min": int(move_min), "stay_min": 0
                    })
                    cur_time = m_end
                    if rel in {"subway_hint", "bus_hint"}:
                        transit_used_today = True

            # 방문행
            smin = stay_minutes(n.get("cat1", ""))
            v_end = cur_time + timedelta(minutes=smin)
            if v_end > day_end:
                return False
            rows_all.append({
                "day_label": label, "day": d,
                "start_time": cur_time.strftime("%H:%M"), "end_time": v_end.strftime("%H:%M"),
                "title": _nfc(n["title"]), "addr1": _nfc(n["addr1"]),
                "cat1": _nfc(n["cat1"]), "cat2": _nfc(n["cat2"]), "cat3": _nfc(n["cat3"]),
                "출발지": "", "교통편1": "", "교통편2": "", "도착지": "",
                "final_score": float(n.get("final_score", np.nan)),
                "distance_from_prev_km": np.nan, "move_min": 0, "stay_min": smin
            })
            cur_time = v_end
            used_idx.add(idx)
            prev_row = n
            used_titles_global.add((_nfc(n.get("title", "")), _nfc(n.get("addr1", ""))))
            return True

        # ① 오전: 음식 제외 카테고리 씨드
        non_food = [c for c in cats if c != MEAL_CAT]
        for c in non_food:
            if quota.get(c, 0) <= 0 or cur_time >= lunch_s:
                continue
            idx = pick_best(pool[(pool["cat1"].map(_nfc) == c) & (~pool.index.isin(used_idx))],
                            float(pool.iloc[0]["lat"]), float(pool.iloc[0]["lon"]))
            if idx is None:
                continue
            if place_visit(idx):
                quota[c] -= 1

        # ② 오전 나머지(음식 제외)
        while cur_time < lunch_s and sum(quota.values()) > 0:
            choices = [c for c in cats if quota.get(c, 0) > 0 and c != MEAL_CAT]
            if not choices:
                break
            choices.sort(key=lambda x: -quota.get(x, 0))
            placed = False
            for c in choices:
                idx = pick_best(pool[(pool["cat1"].map(_nfc) == c) & (~pool.index.isin(used_idx))],
                                float(pool.iloc[0]["lat"]), float(pool.iloc[0]["lon"]))
                if idx is None:
                    continue
                if place_visit(idx):
                    quota[c] -= 1
                    placed = True
                    break
            if not placed:
                break

        # ③ 점심(음식 1곳)
        if meal_enabled_flag and quota.get(MEAL_CAT, 0) > 0:
            cur_time = max(cur_time, lunch_s)
            if cur_time < lunch_e:
                sub = pool[(pool["cat1"].map(_nfc) == MEAL_CAT) & (~pool.index.isin(used_idx))]
                # 카페/전통찻집 제외 + 대표 버킷 1종
                sub = sub[~sub["cat3"].astype(str).map(lambda s: "카페" in _nfc(s) or "전통찻집" in _nfc(s))]
                idx = sub.index[0] if not sub.empty else None
                if idx is not None and place_visit(idx):
                    quota[MEAL_CAT] -= 1

        # ④ 오후(음식 제외)
        while cur_time < dinner_s and sum(quota.values()) > 0:
            choices = [c for c in cats if quota.get(c, 0) > 0 and c != MEAL_CAT]
            if not choices:
                break
            choices.sort(key=lambda x: -quota.get(x, 0))
            placed = False
            for c in choices:
                sub = pool[(pool["cat1"].map(_nfc) == c) & (~pool.index.isin(used_idx))]
                if sub.empty:
                    continue
                idx = sub.index[0]
                if place_visit(idx):
                    quota[c] -= 1
                    placed = True
                    break
            if not placed:
                break

        # ⑤ 저녁(음식 1곳)
        if meal_enabled_flag and quota.get(MEAL_CAT, 0) > 0:
            cur_time = max(cur_time, dinner_s)
            if cur_time < dinner_e:
                sub = pool[(pool["cat1"].map(_nfc) == MEAL_CAT) & (~pool.index.isin(used_idx))]
                sub = sub[~sub["cat3"].astype(str).map(lambda s: "카페" in _nfc(s) or "전통찻집" in _nfc(s))]
                idx = sub.index[0] if not sub.empty else None
                if idx is not None and place_visit(idx):
                    quota[MEAL_CAT] -= 1

        # ⑥ 저녁 이후(~end) 남은 쿼터
        while cur_time < day_end and sum(quota.values()) > 0:
            choices = [c for c in cats if quota.get(c, 0) > 0]
            if not choices:
                break
            choices.sort(key=lambda x: -quota.get(x, 0))
            placed = False
            for c in choices:
                sub = pool[(pool["cat1"].map(_nfc) == c) & (~pool.index.isin(used_idx))]
                if c == MEAL_CAT:
                    # 20시 이후는 (한/중/일/양/이색) 버킷 제외
                    sub = sub[~sub["cat3"].astype(str).map(lambda s: _contains_any(s, MEAL_CUISINE_TAGS))]
                if sub.empty:
                    continue
                idx = sub.index[0]
                if place_visit(idx):
                    quota[c] -= 1
                    placed = True
                    break
            if not placed:
                break

        # 하루 종료 후 사용 인덱스만큼 전역 이동
        used = max(1, len(used_idx))
        pos += used

        # 마지막 방문 체류를 end_time까지 연장
        day_slice = [r for r in rows_all if r["day"] == d and r["title"] != "이동"]
        if day_slice:
            last_end = _to_dt(base, day_slice[-1]["end_time"])
            if last_end < day_end:
                add_min = int((day_end - last_end).total_seconds() // 60)
                if add_min > 0:
                    day_slice[-1]["end_time"] = end_time
                    day_slice[-1]["stay_min"] = int(day_slice[-1].get("stay_min", 0)) + add_min

    # 결과 DF (CSV 저장 없이 반환)
    itinerary = pd.DataFrame(rows_all, columns=[
        "day_label", "day", "start_time", "end_time", "title", "addr1", "cat1", "cat2", "cat3",
        "출발지", "교통편1", "교통편2", "도착지",
        "final_score", "distance_from_prev_km", "move_min", "stay_min"
    ])

    return itinerary


# ========================
# Helpers
# ========================
def _nfc(s: str) -> str:
    return ud.normalize("NFC", str(s or "")).strip()

def _first_contains(cols, key):
    key = key.lower()
    for c in cols:
        if key in c.lower():
            return c
    return None

def _read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def _haversine_np(lat1, lon1, lat_arr, lon_arr):
    lat1r = np.radians(float(lat1))
    lon1r = np.radians(float(lon1))
    lat2r = np.radians(pd.to_numeric(lat_arr, errors="coerce").astype(float))
    lon2r = np.radians(pd.to_numeric(lon_arr, errors="coerce").astype(float))
    dphi = lat2r - lat1r
    dlmb = lon2r - lon1r
    a = np.sin(dphi/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlmb/2.0)**2
    return 2.0 * 6371.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _geocode_region_kakao(region_name: str) -> Optional[Tuple[float, float]]:
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}  # type: ignore[name-defined]
    params  = {"query": region_name}
    resp    = requests.get(url, headers=headers, params=params, timeout=5)
    docs    = resp.json().get("documents", [])
    if not docs:
        return None
    first = docs[0]
    return float(first["y"]), float(first["x"])  # (lat, lon)

def _enrich_transit_hints_fast(df: pd.DataFrame, kakao_key: str) -> pd.DataFrame:
    GRID_DEG, SUBWAY_RADIUS_M, BUS_RADIUS_M = 0.0025, 900, 900
    session = requests.Session()
    session.headers.update({"Authorization": f"KakaoAK {kakao_key}"})

    def tile_key(lat: float, lon: float) -> str:
        return f"{round(lat/GRID_DEG)*GRID_DEG:.6f}|{round(lon/GRID_DEG)*GRID_DEG:.6f}"

    def _get(url, params):
        try:
            r = session.get(url, params=params, timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            return None
        return None

    def nearest_subway(lat, lon):
        js = _get("https://dapi.kakao.com/v2/local/search/category.json",
                  {"category_group_code": "SW8", "x": lon, "y": lat, "radius": SUBWAY_RADIUS_M, "size": 1, "sort": "distance"})
        name, line = "", ""
        if js and js.get("documents"):
            d = js["documents"][0]
            name = _nfc(d.get("place_name"))
            raw = " ".join([name, _nfc(d.get("category_name", "")), _nfc(d.get("address_name", "")), _nfc(d.get("road_address_name", ""))])
            m = re.search(r"(\d+)\s*호선", raw)
            line = f"{m.group(1)}호선" if m else ""
        return name, line

    def _nearest_bus_once(lat, lon, radius, keyword):
        js = _get("https://dapi.kakao.com/v2/local/search/keyword.json",
                  {"query": keyword, "x": lon, "y": lat, "radius": radius, "size": 10, "sort": "distance"})
        if not (js and js.get("documents")): return ""
        docs = sorted(js["documents"], key=lambda d: int(float(d.get("distance", "1e9"))))
        for d in docs:
            nm = _nfc(d.get("place_name"))
            if ("정류" in nm) or ("버스" in nm) or ("정류장" in nm) or ("정류소" in nm):
                return nm
        return _nfc(docs[0].get("place_name"))

    def nearest_bus(lat, lon):
        for r in [BUS_RADIUS_M, max(700, BUS_RADIUS_M + 200), 1200, 1500]:
            for kw in ["버스정류장", "정류장", "버스"]:
                nm = _nearest_bus_once(lat, lon, r, kw)
                if nm: return nm
        return ""

    df = df.copy()
    df["tile_key"] = [tile_key(a, b) for a, b in zip(df["lat"], df["lon"])]
    tiles = df["tile_key"].unique().tolist()

    subway_map: Dict[str, tuple] = {}
    bus_map: Dict[str, str] = {}

    def job_sub(t):
        r = df.loc[df["tile_key"] == t].iloc[0]
        return t, nearest_subway(float(r["lat"]), float(r["lon"]))
    def job_bus(t):
        r = df.loc[df["tile_key"] == t].iloc[0]
        return t, nearest_bus(float(r["lat"]), float(r["lon"]))

    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(job_sub, t) for t in tiles] + [ex.submit(job_bus, t) for t in tiles]
        for fu in as_completed(futs):
            try:
                t, val = fu.result()
                if isinstance(val, tuple): subway_map[t] = val
                else: bus_map[t] = val
            except Exception:
                pass

    df["closest_subway_station"] = df["tile_key"].map(lambda k: subway_map.get(k, ("", ""))[0])
    df["closest_subway_line"]    = df["tile_key"].map(lambda k: subway_map.get(k, ("", ""))[1])
    df["closest_bus_station"]    = df["tile_key"].map(lambda k: bus_map.get(k, ""))
    return df.drop(columns=["tile_key"])

def _greedy_route(rows: pd.DataFrame, start_lat: float, start_lon: float) -> pd.DataFrame:
    def step_cost_from(lat, lon, nxt_row):
        d_km = _haversine(lat, lon, nxt_row["lat"], nxt_row["lon"])
        s = nxt_row.get("final_score", np.nan)
        bonus = min(0.3, float(s) / 100.0) if np.isfinite(s) else 0.0
        return max(0.0, d_km - bonus)
    used = set(); order = []
    cur_lat, cur_lon = start_lat, start_lon
    while len(used) < len(rows):
        cand = [i for i in range(len(rows)) if i not in used]
        j = min(cand, key=lambda k: step_cost_from(cur_lat, cur_lon, rows.loc[k]))
        order.append(j); used.add(j)
        cur_lat, cur_lon = float(rows.loc[j, "lat"]), float(rows.loc[j, "lon"])
    return rows.iloc[order].reset_index(drop=True)

def _split_visits(total: int, days: int, vmin: int, vmax: int) -> list:
    counts, rem, rem_days = [], total, days
    for _ in range(days):
        if rem_days <= 0: counts.append(0); continue
        target = math.ceil(rem / rem_days)
        target = max(vmin if rem >= vmin else rem, min(target, vmax))
        target = min(target, rem)
        counts.append(target); rem -= target; rem_days -= 1
    return counts

def _allocate_day_quota(cats: List[str], visit_target: int) -> dict:
    base_w = [3, 2, 1] + [1] * max(0, len(cats) - 3)
    w = base_w[:len(cats)]
    S = sum(w)
    ideal = [visit_target * wi / S for wi in w]
    floor_cnt = [int(math.floor(x)) for x in ideal]
    if visit_target >= len(cats):
        for i in range(len(cats)):
            if floor_cnt[i] == 0:
                floor_cnt[i] = 1
    cur = sum(floor_cnt)
    rem = max(0, visit_target - cur)
    residuals = [(ideal[i] - floor_cnt[i], i) for i in range(len(cats))]
    residuals.sort(reverse=True)
    k = 0
    while rem > 0 and k < len(residuals):
        i = residuals[k][1]
        floor_cnt[i] += 1
        rem -= 1
        k += 1
    return {cats[i]: floor_cnt[i] for i in range(len(cats))}

def _ensure_variety_pool(route_df: pd.DataFrame, start_idx: int, used_global: set, want: int) -> pd.DataFrame:
    NON_MEAL_MIN_PER_DAY = 3
    MEAL_CAT = "음식"
    def used_filter(row):
        key = (_nfc(row.get("title", "")), _nfc(row.get("addr1", "")))
        return key in used_global
    chunk_size = max(want * 6, want)
    chunk = route_df.iloc[start_idx:start_idx + chunk_size].copy()
    if chunk.empty: return chunk
    chunk = chunk.loc[~chunk.apply(used_filter, axis=1)].copy()
    is_meal = chunk["cat1"].map(lambda s: _nfc(s) == MEAL_CAT)
    non_meal_cnt = int((~is_meal).sum())
    if non_meal_cnt < NON_MEAL_MIN_PER_DAY:
        rest = route_df.iloc[start_idx + chunk_size:].copy()
        if not rest.empty:
            rest = rest.loc[~rest.apply(used_filter, axis=1)]
            rest_nm = rest.loc[rest["cat1"].map(lambda s: _nfc(s) != MEAL_CAT)]
            add_n = NON_MEAL_MIN_PER_DAY - non_meal_cnt
            if add_n > 0 and not rest_nm.empty:
                chunk = pd.concat([chunk, rest_nm.head(add_n)], ignore_index=True)
    return chunk

def _step_cost_from(lat, lon, nxt_row):
    d_km = _haversine(lat, lon, nxt_row["lat"], nxt_row["lon"])
    s = nxt_row.get("final_score", np.nan)
    bonus = min(0.3, float(s) / 100.0) if np.isfinite(s) else 0.0
    return max(0.0, d_km - bonus)

def _norm_station(s: str) -> str:
    t = _nfc(s)
    t = re.sub(r"\(.*?\)", "", t)
    t = t.replace("역", "")
    t = re.sub(r"\s+", "", t)
    return t

def _line_station_text(line: str, station: str) -> str:
    line, station = _nfc(line), _nfc(station)
    if line and (line in station): return station
    return f"{line} {station}".strip() if line else station

def _odsay_total_minutes(plon, plat, nlon, nlat, api_key: str) -> Optional[int]:
    url = "https://api.odsay.com/v1/api/searchPubTransPathT"
    params = {"SX": f"{plon:.6f}", "SY": f"{plat:.6f}", "EX": f"{nlon:.6f}", "EY": f"{nlat:.6f}",
              "apiKey": api_key, "lang": 0, "OPT": 0}
    r = requests.get(url, params=params, timeout=5)
    js = r.json()
    if js and "result" in js and "path" in js["result"] and js["result"]["path"]:
        info = js["result"]["path"][0].get("info", {})
        tt = info.get("totalTime")
        if isinstance(tt, (int, float)) and tt > 0:
            return int(tt)
    return None

def _to_dt(base: datetime, hhmm: str) -> datetime:
    h, m = map(int, hhmm.split(":"))
    return base.replace(hour=h, minute=m, second=0, microsecond=0)

def _contains_any(s: str, keys: set[str]) -> bool:
    t = _nfc(s)
    return any(k in t for k in keys)

def _check_hhmm(s: str):
    if not re.fullmatch(r"\d{2}:\d{2}", s or ""):
        raise ValueError("시간 형식은 HH:MM 이어야 합니다.")
