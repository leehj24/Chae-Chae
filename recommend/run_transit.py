# recommend/run_transit.py
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
from recommend.config import * # noqa: F403,F401

from dotenv import load_dotenv
load_dotenv()
import os

# 환경변수 있으면 우선, 없으면 config 값 유지
_KAKAO_ENV = os.environ.get("KAKAO_API_KEY")
if _KAKAO_ENV:
    KAKAO_API_KEY = _KAKAO_ENV

_ODSAY_ENV = os.environ.get("ODSAY_API_KEY")
if _ODSAY_ENV:
    ODSAY_API_KEY = _ODSAY_ENV


# ========================
# Public API
# ========================
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
    # ... (기존 입력 검증 및 데이터 로드 부분은 동일) ...
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
    tmf = _read_csv_robust(PATH_TMF)
    cols_lower = {c.lower(): c for c in tmf.columns}
    need = {
        "title": cols_lower.get("title"), "addr1": cols_lower.get("addr1"),
        "cat1":  cols_lower.get("cat1") or _first_contains(tmf.columns, "cat1"),
        "mapx":  cols_lower.get("mapx") or cols_lower.get("lon") or cols_lower.get("longitude") or cols_lower.get("x"),
        "mapy":  cols_lower.get("mapy") or cols_lower.get("lat") or cols_lower.get("latitude") or cols_lower.get("y"),
        "tour_score": cols_lower.get("tour_score"), "review_score": cols_lower.get("review_score"),
    }
    miss = [k for k, v in need.items() if v is None]
    if miss:
        raise KeyError(f"필수 컬럼 누락: {miss} / 실제컬럼: {list(tmf.columns)}")

    df = tmf.rename(columns={
        need["title"]: "title", need["addr1"]: "addr1", need["cat1"]:  "cat1",
        need["mapx"]:  "lon", need["mapy"]:  "lat",
        need["tour_score"]: "tour_score", need["review_score"]: "review_score",
    }).copy()
    for c in ("lat", "lon", "tour_score", "review_score"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()
    if "cat2" not in df.columns: df["cat2"] = ""
    if "cat3" not in df.columns: df["cat3"] = ""

    # ----- 필터링 및 정렬 (기존과 동일) -----
    radius_km = TRANSIT_RADIUS_KM if FAST_MODE else 20
    df["distance_km"] = _haversine_np(center_lat, center_lon, df["lat"], df["lon"])
    df = df[df["distance_km"] <= radius_km].copy()
    if df.empty:
        raise RuntimeError("선택한 지역 주변(반경)에서 후보가 없습니다.")
    score_map = {"관광지수": "tour_score", "인기도지수": "review_score"}
    score_col = score_map[score_label]
    df = df.sort_values([score_col, "distance_km"], ascending=[False, True]).copy()
    df = df.drop_duplicates(subset=["title", "addr1"], keep="first").reset_index(drop=True)
    
    # ... (카테고리 보충 로직 등 기존과 동일) ...
    FALLBACK_N = 5
    df_sorted = df[df["cat1"].isin(cats)].copy() if cats else df.copy()
    pool = df.copy()
    for cat in cats:
        if (df_sorted["cat1"] == cat).sum() == 0:
            remain = pool.loc[~pool.index.isin(df_sorted.index)]
            fb = remain.sort_values(["distance_km", score_col], ascending=[True, False]).head(FALLBACK_N)
            if not fb.empty:
                df_sorted = pd.concat([df_sorted, fb], ignore_index=True)
    
    df_sorted["cat1_norm"] = df_sorted["cat1"].map(_nfc)
    df_sorted["cat3_norm"] = df_sorted["cat3"].astype(str).map(_nfc)
    df_sorted["final_score"] = pd.to_numeric(df_sorted["review_score"].fillna(df_sorted["tour_score"]), errors="coerce").fillna(0.0)
    df_sorted = df_sorted.sort_values([score_col, "distance_km"], ascending=[False, True]).reset_index(drop=True)
    
    # ----- 대중교통 힌트 병렬 조회 (Top N) -----
    top_n = TRANSIT_TOP_N if FAST_MODE else len(df_sorted)
    top_pois = df_sorted.head(top_n).copy()
    remaining_pois = df_sorted.tail(len(df_sorted) - top_n).copy()
    enriched_top = _enrich_transit_hints_fast(top_pois, KAKAO_API_KEY)
    for col in ["closest_subway_station", "closest_subway_line", "closest_bus_station"]:
        if col not in remaining_pois.columns:
            remaining_pois[col] = np.nan
    pois_with_hints = pd.concat([enriched_top, remaining_pois], ignore_index=True)
    
    route = _greedy_route(pois_with_hints, float(pois_with_hints.iloc[0]["lat"]), float(pois_with_hints.iloc[0]["lon"]))

    # ======================================
    # 스케줄링
    # ======================================
    MEAL_CAT = "음식"
    MEAL_CUISINE_TAGS = {"서양식", "이색음식점", "일식", "중식", "한식"}
    DAY_VISIT_MIN, DAY_VISIT_MAX = 2, 6
    WALK_SKIP_KM = 0.30
    BASE_SPEED_KMH, ADD_FIXED_MIN = 18.0, 8.0

    def estimate_transit_minutes(d_km: float, rel: str) -> int:
        base = (float(d_km) / BASE_SPEED_KMH) * 60.0 + ADD_FIXED_MIN
        if rel == "same_subway_station": return max(3, int(round(base - 10)))
        if rel == "same_subway_line":    return max(4, int(round(base - 6)))
        if rel == "same_bus_station":    return max(5, int(round(base - 5)))
        return int(round(base))

    def transit_minutes_via_api_or_est(p: pd.Series, n: pd.Series, rel: str, d_km: float, odsay_cache: dict) -> int:
        """캐시에 ODsay 조회 결과가 있으면 사용하고, 없으면 예측값을 사용합니다."""
        key = (p.name, n.name)
        cached_time = odsay_cache.get(key)
        
        if isinstance(cached_time, int) and cached_time > 0:
            return cached_time
            
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
        
    def stay_minutes(cat1: str) -> int:
        c = _nfc(cat1); return {"음식": 75, "자연": 90, "레포츠": 120}.get(c, 90)

    visit_counts = _split_visits(len(route), days, DAY_VISIT_MIN, DAY_VISIT_MAX)
    rows_all: List[dict] = []
    pos = 0
    midnight0 = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    used_titles_global: set = set()
    meal_enabled_flag = ("음식" in set(cats))

    for d in range(1, days + 1):
        base  = midnight0 + timedelta(days=d - 1)
        label = f"{d}일"
        want  = visit_counts[d - 1] if d - 1 < len(visit_counts) else 0
        if want <= 0: break

        pool = _ensure_variety_pool(route, pos, used_titles_global, want)
        if pool.empty: break

        day_start, day_end = _to_dt(base, start_time), _to_dt(base, end_time)
        lunch_s, lunch_e   = base.replace(hour=11, minute=0), base.replace(hour=13, minute=0)
        dinner_s, dinner_e = base.replace(hour=17, minute=0), base.replace(hour=20, minute=0)

        quota = _allocate_day_quota(cats, want)
        cur_time = day_start
        used_idx = set()
        prev_row = None
        transit_used_today = False
        odsay_cache = {}

        def already_used_globally(r: pd.Series) -> bool:
            return (_nfc(r.get("title", "")), _nfc(r.get("addr1", ""))) in used_titles_global

        def pick_best(sub: pd.DataFrame, ref_lat: float, ref_lon: float):
            if sub.empty: return None
            dkm = np.sqrt((sub["lat"] - ref_lat) ** 2 + (sub["lon"] - ref_lon) ** 2) * 111.0
            pen = dkm.apply(lambda x: (x / BASE_SPEED_KMH) * 60.0) / 60.0
            sc  = sub.get("final_score", pd.Series([0] * len(sub))).fillna(0) - 0.1 * pen
            return sc.sort_values(ascending=False).index[0]

        def place_visit(idx: int):
            nonlocal cur_time, prev_row, transit_used_today
            n = pool.loc[idx]
            if already_used_globally(n): return False

            if prev_row is not None:
                d_km = _haversine(prev_row["lat"], prev_row["lon"], n["lat"], n["lon"])
                rel, t1, t2 = relation_and_text(prev_row, n, d_km)

                is_last_visit_of_day = (len(used_idx) == want - 1)
                if not transit_used_today and is_last_visit_of_day and rel not in {"subway_hint", "bus_hint"}:
                    transit_ready = pool.loc[(~pool.index.isin(used_idx | {idx})) & (pool['closest_subway_station'].notna() | pool['closest_bus_station'].notna())]
                    best_transit_j, min_dist = None, float('inf')
                    for j, r2 in transit_ready.iterrows():
                        if already_used_globally(r2): continue
                        temp_d_km = _haversine(prev_row["lat"], prev_row["lon"], r2["lat"], r2["lon"])
                        temp_rel, _, _ = relation_and_text(prev_row, r2, temp_d_km)
                        if temp_rel in {"subway_hint", "bus_hint"} and temp_d_km < min_dist:
                            min_dist, best_transit_j = temp_d_km, j
                    
                    if best_transit_j is not None:
                        idx = best_transit_j
                        n = pool.loc[idx]
                        d_km = _haversine(prev_row["lat"], prev_row["lon"], n["lat"], n["lon"])
                        rel, t1, t2 = relation_and_text(prev_row, n, d_km)

                if rel not in {"same_subway_station", "same_bus_station", "walk_hint"}:
                    move_min = transit_minutes_via_api_or_est(prev_row, n, rel, d_km, odsay_cache)
                    m_end = cur_time + timedelta(minutes=move_min)
                    if m_end > day_end: return False
                    rows_all.append({"day_label": label, "day": d, "start_time": cur_time.strftime("%H:%M"), "end_time": m_end.strftime("%H:%M"),
                        "title": "이동", "addr1": "", "cat1": "", "cat2": "", "cat3": "",
                        "출발지": _nfc(prev_row.get("addr1") or prev_row.get("title")), "교통편1": t1, "교통편2": t2,
                        "도착지": _nfc(n.get("addr1") or n.get("title")), "final_score": np.nan,
                        "distance_from_prev_km": round(d_km, 2), "move_min": int(move_min), "stay_min": 0})
                    cur_time = m_end
                    if rel in {"subway_hint", "bus_hint"}: transit_used_today = True

            smin = stay_minutes(n.get("cat1", ""))
            v_end = cur_time + timedelta(minutes=smin)


            if v_end > day_end:
                if rows_all and rows_all[-1]["title"] == "이동":
                    last_move = rows_all.pop()
                    cur_time -= timedelta(minutes=last_move["move_min"])
                return False
                
            # ▼▼▼ [수정] mapx, mapy 좌표 추가 ▼▼▼
            rows_all.append({"day_label": label, "day": d, "start_time": cur_time.strftime("%H:%M"), "end_time": v_end.strftime("%H:%M"),
                "title": _nfc(n["title"]), "addr1": _nfc(n["addr1"]), "cat1": _nfc(n["cat1"]), "cat2": _nfc(n["cat2"]), "cat3": _nfc(n["cat3"]),
                "출발지": "", "교통편1": "", "교통편2": "", "도착지": "", "final_score": float(n.get("final_score", np.nan)),
                "distance_from_prev_km": np.nan, "move_min": 0, "stay_min": smin,
                "mapx": n.get("lon"), "mapy": n.get("lat") # lon이 mapx, lat이 mapy
            })
            # ▼▼▼ [FIX] Corrected syntax error by splitting the assignment ▼▼▼
            cur_time = v_end
            used_idx.add(idx)
            prev_row = n
            # ▲▲▲ [FIX] ▲▲▲

            used_titles_global.add((_nfc(n.get("title", "")), _nfc(n.get("addr1", ""))))
            return True

        # --- 스케줄링 메인 루프 (기존과 유사, ODsay 캐시 채우는 부분만 추가) ---
        def schedule_period(cats_to_schedule, is_meal_period=False, meal_sub_filter=None):
            nonlocal cur_time
            if prev_row is not None and ODSAY_API_KEY:
                candidates = pool.loc[~pool.index.isin(used_idx)]
                if not candidates.empty:
                    newly_fetched = _batch_fetch_odsay_times(prev_row, candidates, ODSAY_API_KEY)
                    odsay_cache.update(newly_fetched)
            
            choices = [c for c in cats_to_schedule if quota.get(c, 0) > 0]
            if not choices: return False
            choices.sort(key=lambda x: -quota.get(x, 0))
            
            for c in choices:
                sub = pool[(pool["cat1_norm"] == c) & (~pool.index.isin(used_idx))]
                if is_meal_period and meal_sub_filter is not None: sub = sub[meal_sub_filter(sub)]
                if sub.empty: continue
                
                ref_lat = prev_row["lat"] if prev_row is not None else float(pool.iloc[0]["lat"])
                ref_lon = prev_row["lon"] if prev_row is not None else float(pool.iloc[0]["lon"])
                idx = pick_best(sub, ref_lat, ref_lon)
                if idx is not None and place_visit(idx):
                    quota[c] -= 1
                    return True
            return False

        # ① 오전
        while cur_time < lunch_s and sum(q for c, q in quota.items() if c != MEAL_CAT) > 0:
            if not schedule_period([c for c in cats if c != MEAL_CAT]): break
        # ② 점심
        if meal_enabled_flag and quota.get(MEAL_CAT, 0) > 0 and cur_time < lunch_e:
            cur_time = max(cur_time, lunch_s)
            is_cafe = lambda s: "카페" in _nfc(s) or "전통찻집" in _nfc(s)
            schedule_period([MEAL_CAT], True, lambda df: ~df["cat3_norm"].apply(is_cafe))
        # ③ 오후
        while cur_time < dinner_s and sum(q for c, q in quota.items() if c != MEAL_CAT) > 0:
            if not schedule_period([c for c in cats if c != MEAL_CAT]): break
        # ④ 저녁
        if meal_enabled_flag and quota.get(MEAL_CAT, 0) > 0 and cur_time < dinner_e:
            cur_time = max(cur_time, dinner_s)
            is_cafe = lambda s: "카페" in _nfc(s) or "전통찻집" in _nfc(s)
            schedule_period([MEAL_CAT], True, lambda df: ~df["cat3_norm"].apply(is_cafe))
        # ⑤ 저녁 이후
        while cur_time < day_end and sum(quota.values()) > 0:
            is_main_meal = lambda s: _contains_any(s, MEAL_CUISINE_TAGS)
            if not schedule_period(cats, False, lambda df: ~df["cat3_norm"].apply(is_main_meal)): break
            
        pos += max(1, len(used_idx))

    return pd.DataFrame(rows_all, columns=["day_label", "day", "start_time", "end_time", "title", "addr1",
        "cat1", "cat2", "cat3", "출발지", "교통편1", "교통편2", "도착지", "final_score",
        "distance_from_prev_km", "move_min", "stay_min","mapx", "mapy"])


# ========================
# Helpers
# ========================
def _batch_fetch_odsay_times(from_poi: pd.Series, to_pois_df: pd.DataFrame, api_key: str) -> dict:
    """한 출발지에서 여러 도착지까지의 ODsay 소요 시간을 병렬로 조회합니다."""
    results = {}
    if not api_key or to_pois_df.empty:
        return results

    from_lon, from_lat = from_poi["lon"], from_poi["lat"]

    def fetch_one(to_poi_row: pd.Series):
        to_lon, to_lat = to_poi_row["lon"], to_poi_row["lat"]
        minutes = _odsay_total_minutes(from_lon, from_lat, to_lon, to_lat, api_key)
        return (from_poi.name, to_poi_row.name), minutes

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_one, row) for _, row in to_pois_df.iterrows()]
        for future in as_completed(futures):
            try:
                key, minutes = future.result()
                if minutes is not None:
                    results[key] = minutes
            except Exception:
                pass # 실패 시 그냥 넘어감 (예측값 사용)
    return results

def _nfc(s: str) -> str:
    return ud.normalize("NFC", str(s or "")).strip()

def _first_contains(cols, key):
    key = key.lower(); return next((c for c in cols if key in c.lower()), None)

def _read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try: return pd.read_csv(path, encoding=enc)
        except Exception: pass
    return pd.read_csv(path)

def _haversine_np(lat1, lon1, lat_arr, lon_arr):
    lat1r, lon1r = np.radians(float(lat1)), np.radians(float(lon1))
    lat2r, lon2r = np.radians(pd.to_numeric(lat_arr, errors="coerce")), np.radians(pd.to_numeric(lon_arr, errors="coerce"))
    dphi, dlmb = lat2r - lat1r, lon2r - lon1r
    a = np.sin(dphi/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlmb/2.0)**2
    return 2.0 * 6371.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    # ▼▼▼ [FIX] Corrected a typo from float(1) to float(lat1) ▼▼▼
    dphi, dlmb = math.radians(float(lat2) - float(lat1)), math.radians(float(lon2) - float(lon1))
    # ▲▲▲ [FIX] ▲▲▲
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _geocode_region_kakao(region_name: str) -> Optional[Tuple[float, float]]:
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params  = {"query": region_name}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        docs = resp.json().get("documents", [])
        if docs: return float(docs[0]["y"]), float(docs[0]["x"])
    except requests.exceptions.RequestException: pass
    return None

def _get_transit_info_on_demand(row: pd.Series, kakao_key: str) -> pd.Series:
    lat, lon = row["lat"], row["lon"]
    sub_name, sub_line = _nearest_subway(lat, lon, kakao_key)
    bus_name = _nearest_bus(lat, lon, kakao_key)
    row["closest_subway_station"], row["closest_subway_line"], row["closest_bus_station"] = sub_name, sub_line, bus_name
    return row

def _nearest_subway(lat, lon, kakao_key: str):
    headers, params = {"Authorization": f"KakaoAK {kakao_key}"}, {"category_group_code": "SW8", "x": lon, "y": lat, "radius": 900, "size": 1, "sort": "distance"}
    try:
        r = requests.get("https://dapi.kakao.com/v2/local/search/category.json", headers=headers, params=params, timeout=4)
        if r.ok and (docs := r.json().get("documents")):
            d = docs[0]
            name = _nfc(d.get("place_name"))
            raw = " ".join([name, _nfc(d.get("category_name", "")), _nfc(d.get("address_name", "")), _nfc(d.get("road_address_name", ""))])
            m = re.search(r"(\d+)\s*호선", raw)
            return name, f"{m.group(1)}호선" if m else ""
    except Exception: pass
    return "", ""

def _nearest_bus(lat, lon, kakao_key: str):
    headers = {"Authorization": f"KakaoAK {kakao_key}"}
    for r_km in [700, 900, 1200]:
        for kw in ["버스정류장", "버스"]:
            params = {"query": kw, "x": lon, "y": lat, "radius": r_km, "size": 1, "sort": "distance"}
            try:
                r = requests.get("https://dapi.kakao.com/v2/local/search/keyword.json", headers=headers, params=params, timeout=4)
                if r.ok and (docs := r.json().get("documents")): return _nfc(docs[0].get("place_name", ""))
            except Exception: continue
    return ""

def _enrich_transit_hints_fast(df: pd.DataFrame, kakao_key: str) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"Authorization": f"KakaoAK {kakao_key}"})
    df["tile_key"] = [f"{round(lat/0.0025)*0.0025:.6f}|{round(lon/0.0025)*0.0025:.6f}" for lat, lon in zip(df["lat"], df["lon"])]
    subway_map, bus_map = {}, {}
    
    def job(tile, is_subway):
        r = df.loc[df["tile_key"] == tile].iloc[0]
        return tile, (_nearest_subway(r["lat"], r["lon"], kakao_key) if is_subway else _nearest_bus(r["lat"], r["lon"], kakao_key))
        
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(job, t, True) for t in df["tile_key"].unique()] + [ex.submit(job, t, False) for t in df["tile_key"].unique()]
        for fu in as_completed(futs):
            try:
                t, val = fu.result()
                if isinstance(val, tuple): subway_map[t] = val
                else: bus_map[t] = val
            except Exception: pass
            
    df["closest_subway_station"] = df["tile_key"].map(lambda k: subway_map.get(k, ("", ""))[0])
    df["closest_subway_line"]    = df["tile_key"].map(lambda k: subway_map.get(k, ("", ""))[1])
    df["closest_bus_station"]    = df["tile_key"].map(lambda k: bus_map.get(k, ""))
    return df.drop(columns=["tile_key"])

def _greedy_route(rows: pd.DataFrame, start_lat: float, start_lon: float) -> pd.DataFrame:
    order, used = [], set()
    cur_lat, cur_lon = start_lat, start_lon
    available = list(rows.index)
    while len(used) < len(rows):
        cand_indices = [i for i in available if i not in used]
        if not cand_indices: break
        
        def cost_func(k):
            nxt = rows.loc[k]
            d_km = _haversine(cur_lat, cur_lon, nxt["lat"], nxt["lon"])
            s = nxt.get("final_score", 0)
            return max(0.0, d_km - (min(0.3, float(s) / 100.0) if pd.notna(s) else 0.0))

        best_idx = min(cand_indices, key=cost_func)
        order.append(best_idx)
        used.add(best_idx)
        next_stop = rows.loc[best_idx]
        cur_lat, cur_lon = float(next_stop["lat"]), float(next_stop["lon"])
    return rows.loc[order].reset_index(drop=True)

def _split_visits(total: int, days: int, vmin: int, vmax: int) -> list:
    counts, rem, rem_days = [], total, days
    for _ in range(days):
        if rem_days <= 0: counts.append(0); continue
        target = max(vmin if rem >= vmin else rem, min(math.ceil(rem / rem_days), vmax))
        target = min(target, rem)
        counts.append(target); rem -= target; rem_days -= 1
    return counts

def _allocate_day_quota(cats: List[str], visit_target: int) -> dict:
    w = [3, 2, 1] + [1] * max(0, len(cats) - 3)
    ideal = [visit_target * wi / sum(w) for wi in w[:len(cats)]]
    floor_cnt = [max(1, int(i)) if visit_target >= len(cats) else int(i) for i in ideal]
    rem = visit_target - sum(floor_cnt)
    residuals = sorted([(ideal[i] - floor_cnt[i], i) for i in range(len(cats))], reverse=True)
    for _, i in residuals[:rem]: floor_cnt[i] += 1
    return {cats[i]: floor_cnt[i] for i in range(len(cats))}

def _ensure_variety_pool(route_df, start_idx, used_global, want):
    chunk_size = max(want * 6, want)
    chunk = route_df.iloc[start_idx:start_idx + chunk_size].copy()
    chunk = chunk.loc[~chunk.apply(lambda r: (_nfc(r.get("title")), _nfc(r.get("addr1"))) in used_global, axis=1)]
    if (chunk["cat1_norm"] != "음식").sum() < 1:
        rest = route_df.iloc[start_idx + chunk_size:].copy()
        rest = rest.loc[~rest.apply(lambda r: (_nfc(r.get("title")), _nfc(r.get("addr1"))) in used_global, axis=1)]
        if not rest.empty:
            chunk = pd.concat([chunk, rest[rest["cat1_norm"] != "음식"].head(1)], ignore_index=True)
    return chunk

def _norm_station(s: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"\(.*?\)", "", _nfc(s)).replace("역", ""))

def _line_station_text(line: str, station: str) -> str:
    line, station = _nfc(line), _nfc(station)
    return station if line and line in station else f"{line} {station}".strip()

def _odsay_total_minutes(plon, plat, nlon, nlat, api_key: str) -> Optional[int]:
    url = "https://api.odsay.com/v1/api/searchPubTransPathT"
    params = {"SX": f"{plon:.6f}", "SY": f"{plat:.6f}", "EX": f"{nlon:.6f}", "EY": f"{nlat:.6f}", "apiKey": api_key, "lang": 0, "OPT": 0}
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.ok and (path := r.json().get("result", {}).get("path", [])):
            tt = path[0].get("info", {}).get("totalTime")
            if isinstance(tt, (int, float)) and tt > 0: return int(tt)
    except requests.exceptions.RequestException: pass
    return None

def _to_dt(base: datetime, hhmm: str) -> datetime:
    h, m = map(int, hhmm.split(":")); return base.replace(hour=h, minute=m, second=0, microsecond=0)

def _contains_any(s: str, keys: set[str]) -> bool:
    return any(k in _nfc(s) for k in keys)

def _check_hhmm(s: str):
    if not re.fullmatch(r"\d{2}:\d{2}", s or ""): raise ValueError("시간 형식은 HH:MM 이어야 합니다.")