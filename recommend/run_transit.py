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

from recommend.config import *

from dotenv import load_dotenv
load_dotenv()
import os

_KAKAO_ENV = os.environ.get("KAKAO_API_KEY")
if _KAKAO_ENV: KAKAO_API_KEY = _KAKAO_ENV
_ODSAY_ENV = os.environ.get("ODSAY_API_KEY")
if _ODSAY_ENV: ODSAY_API_KEY = _ODSAY_ENV

# 전역 상수 설정
MEAL_CAT = "음식"
DAY_VISIT_MIN, DAY_VISIT_MAX = 3, 5
BASE_WEIGHTS = [3, 2, 1]


def build_day_quota(cats: List[str], day_total_slots: int, base_weights: List[int]) -> dict:
    u = list(dict.fromkeys(cats)); quotas = {c:0 for c in u}
    for i, w in enumerate(base_weights):
        if i < len(u): quotas[u[i]] += w
    remain = day_total_slots - sum(quotas.values()); i = 0
    while remain > 0 and u:
        quotas[u[i % len(u)]] += 1; remain -= 1; i += 1
    return quotas

# ========================
# Public API
# ========================
def run(
    region: str, transport_mode: str, score_label: str, days: int, cats: List[str],
    start_time: str = "08:00", end_time: str = "22:30", **_,
) -> pd.DataFrame:
    # --- 데이터 로드 및 전처리 ---
    if not region: raise ValueError("여행 지역을 입력하세요.")
    cats = list(dict.fromkeys([_nfc(c) for c in (cats or [])]))
    if not cats: raise ValueError("테마를 최소 1개 이상 선택하세요.")

    coords = _geocode_region_kakao(region)
    if not coords: raise ValueError(f"카카오 지오코딩 실패: '{region}'")
    center_lat, center_lon = coords
    
    df = _load_and_prepare_data(PATH_TMF, score_label, center_lat, center_lon, cats)
    if df.empty: return pd.DataFrame()

    # --- 경로 탐색 및 교통정보 조회 (최적화) ---
    preliminary_route = _greedy_route(df, center_lat, center_lon)
    top_n_for_transit_search = TRANSIT_TOP_N if FAST_MODE else 100
    pois_for_search = preliminary_route.head(top_n_for_transit_search).copy()
    pois_with_hints = _enrich_transit_hints_fast(pois_for_search, KAKAO_API_KEY)
    
    final_candidates = pd.concat([
        pois_with_hints,
        preliminary_route.loc[~preliminary_route.index.isin(pois_with_hints.index)]
    ]).drop_duplicates(subset=["title", "addr1"]).reset_index(drop=True)
    
    route = _greedy_route(final_candidates, center_lat, center_lon)

    # ======================================
    # 스케줄링 로직
    # ======================================
    visit_counts = _split_visits(len(route), days, DAY_VISIT_MIN, DAY_VISIT_MAX)
    rows_all: List[dict] = []
    pos = 0
    midnight0 = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for d in range(1, days + 1):
        base  = midnight0 + timedelta(days=d - 1)
        day_label = f"{d}일"
        want  = visit_counts[d - 1] if d - 1 < len(visit_counts) else 0
        if want <= 0: continue

        day_pool = _ensure_variety_pool(route, pos, {r['title'] for r in rows_all if 'title' in r}, want)
        if day_pool.empty: continue

        day_start = _to_dt(base, start_time)
        day_end   = _to_dt(base, end_time)
        
        quota = build_day_quota(cats, want, BASE_WEIGHTS)
        
        cur_time = day_start
        day_schedule: List[pd.Series] = []
        
        while len(day_schedule) < want and cur_time < day_end:
            ref_lat = day_schedule[-1]["lat"] if day_schedule else center_lat
            ref_lon = day_schedule[-1]["lon"] if day_schedule else center_lon

            available_pool = day_pool.loc[~day_pool.index.isin([p.name for p in day_schedule])].copy()
            valid_cats = {c for c, q in quota.items() if q > 0}
            available_pool = available_pool[available_pool["cat1_norm"].isin(valid_cats)]
            if available_pool.empty: break

            pref_pool, fallback_pool = _get_preferred_pool(available_pool, cur_time, base, quota)
            
            use_transit_now = (len(day_schedule) > 0 and not any(p.get('used_transit', False) for p in day_schedule))
            
            next_place = _select_next_place(pref_pool, fallback_pool, day_schedule[-1] if day_schedule else None, use_transit_now, ref_lat, ref_lon)
            if next_place is None: break

            move_info = _calculate_move_info(day_schedule[-1] if day_schedule else None, next_place)
            
            st = cur_time + timedelta(minutes=move_info["move_min"])
            et = st + timedelta(minutes=stay_minutes(next_place["cat1"]))

            if et > day_end: break

            if move_info["move_min"] > 0 and day_schedule:
                rows_all.append(_create_move_row(day_label, d, cur_time, st, day_schedule[-1], next_place, move_info))
            
            rows_all.append(_create_visit_row(day_label, d, st, et, next_place, stay_minutes(next_place["cat1"])))

            cur_time = et
            next_place_copy = next_place.copy()
            next_place_copy['used_transit'] = move_info['is_transit'] or (day_schedule[-1].get('used_transit', False) if day_schedule else False)
            day_schedule.append(next_place_copy)

            quota[next_place["cat1_norm"]] = quota.get(next_place["cat1_norm"], 1) - 1
        
        pos += len(day_schedule)
        
    return pd.DataFrame(rows_all)


# ========================
# Helper 함수 영역
# ========================
def _load_and_prepare_data(path, score_label, center_lat, center_lon, cats):
    df = _read_csv_robust(path)
    cols_lower = {c.lower(): c for c in df.columns}
    need = {"title": "title", "addr1": "addr1", "cat1": "cat1", "mapx": "lon", "mapy": "lat", "tour_score": "tour_score", "review_score": "review_score"}
    rename_map = {cols_lower.get(k): v for k, v in need.items() if cols_lower.get(k)}
    if len(rename_map) < len(need): raise KeyError(f"필수 컬럼 부족")
    
    df = df.rename(columns=rename_map)
    for c in ["cat2", "cat3"]:
        if c not in df.columns: df[c] = ""
    for c in ("lat", "lon", "tour_score", "review_score"): df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()
    df["distance_km"] = _haversine_np(center_lat, center_lon, df["lat"], df["lon"])
    df = df[df["distance_km"] <= (TRANSIT_RADIUS_KM if FAST_MODE else 20)].copy()
    if df.empty: return pd.DataFrame()
    
    score_col = {"관광지수": "tour_score", "인기도지수": "review_score"}[score_label]
    df = df.sort_values([score_col, "distance_km"], ascending=[False, True]).drop_duplicates(subset=["title", "addr1"]).reset_index(drop=True)
    
    df_sorted_themed = df[df["cat1"].isin(cats)].copy()
    if df_sorted_themed.empty:
        df_sorted = df.copy()
    else:
        for cat in cats:
            if not df_sorted_themed["cat1"].isin([cat]).any():
                fb = df[~df.index.isin(df_sorted_themed.index)].sort_values(["distance_km", score_col], ascending=[True, False]).head(5)
                if not fb.empty: df_sorted_themed = pd.concat([df_sorted_themed, fb], ignore_index=True)
        df_sorted = df_sorted_themed
            
    df_sorted["cat1_norm"] = df_sorted["cat1"].map(_nfc)
    df_sorted["cat3_norm"] = df_sorted["cat3"].astype(str).map(_nfc)
    df_sorted["final_score"] = pd.to_numeric(df_sorted["review_score"].fillna(df_sorted["tour_score"]), errors="coerce").fillna(0.0)
    return df_sorted.sort_values([score_col, "distance_km"], ascending=[False, True]).reset_index(drop=True)

# ▼▼▼ 수정된 부분: 누락되었던 stay_minutes 함수를 여기에 추가 ▼▼▼
def stay_minutes(cat1: str) -> int:
    """카테고리별 체류 시간을 반환합니다."""
    c = _nfc(cat1)
    if c == "음식": return 75
    if c == "자연": return 90
    if c == "레포츠": return 120
    return 90
# ▲▲▲ 수정된 부분 ▲▲▲

def _get_preferred_pool(pool, cur_time, base_date, quota):
    is_lunch = (cur_time >= base_date.replace(hour=11) and cur_time < base_date.replace(hour=13))
    is_dinner = (cur_time >= base_date.replace(hour=17) and cur_time < base_date.replace(hour=20))
    if (is_lunch or is_dinner) and quota.get(MEAL_CAT, 0) > 0:
        pref = pool[pool["cat1_norm"] == MEAL_CAT]; pref = pref[~pref["cat3_norm"].str.contains("카페|전통찻집", na=False)]; fallback = pool[pool["cat1_norm"] != MEAL_CAT]
        return pref, fallback
    else:
        pref = pool[pool["cat1_norm"] != MEAL_CAT]; fallback = pool[pool["cat1_norm"] == MEAL_CAT]
        return pref, fallback

def _select_next_place(pref, fallback, prev_row, force_transit, ref_lat, ref_lon):
    def find_best(pool):
        if pool.empty: return None
        costs = pool.apply(lambda r: _step_cost_from(ref_lat, ref_lon, r), axis=1)
        return pool.loc[costs.idxmin()] if not costs.empty else None
    if force_transit and prev_row is not None:
        transit_pref = pref[pref.apply(lambda r: _relation_and_text(prev_row, r)[0] in {"subway_hint", "bus_hint"}, axis=1)]
        if not transit_pref.empty: return find_best(transit_pref)
    best_pref = find_best(pref)
    if best_pref is not None: return best_pref
    return find_best(fallback)

def _calculate_move_info(prev, next):
    if prev is None: return {"move_min": 0, "is_transit": False, "d_km": 0, "t1": "", "t2": ""}
    rel, t1, t2, d_km = _relation_and_text(prev, next); is_transit = rel in {"subway_hint", "bus_hint"}
    move_min = 0
    if is_transit: move_min = _odsay_total_minutes(prev["lon"], prev["lat"], next["lon"], next["lat"], ODSAY_API_KEY) or int((d_km / 18.0) * 60 + 8)
    else: move_min = int((d_km / 4.5) * 60)
    return {"move_min": move_min, "is_transit": is_transit, "d_km": d_km, "t1": t1, "t2": t2}

def _create_move_row(label, day, st, et, prev, next, move_info): return {"day_label": label, "day": day, "start_time": st.strftime("%H:%M"), "end_time": et.strftime("%H:%M"), "title": "이동", "addr1": "", "출발지": _nfc(prev.get("title")), "교통편1": move_info["t1"], "교통편2": move_info["t2"], "도착지": _nfc(next.get("title")), "distance_from_prev_km": round(move_info["d_km"], 2), "move_min": move_info["move_min"], "stay_min": 0}
def _create_visit_row(label, day, st, et, place, stay_min): return {"day_label": label, "day": day, "start_time": st.strftime("%H:%M"), "end_time": et.strftime("%H:%M"), "title": _nfc(place["title"]), "addr1": _nfc(place["addr1"]), "cat1": _nfc(place["cat1"]), "cat2": _nfc(place["cat2"]), "cat3": _nfc(place["cat3"]), "final_score": float(place.get("final_score", np.nan)), "stay_min": stay_min}
def _relation_and_text(prev_row, nxt_row):
    d_km = _haversine(prev_row["lat"], prev_row["lon"], nxt_row["lat"], nxt_row["lon"]); ps_raw, pl = _nfc(prev_row.get("closest_subway_station", "")), _nfc(prev_row.get("closest_subway_line", "")); ns_raw, nl = _nfc(nxt_row.get("closest_subway_station", "")), _nfc(nxt_row.get("closest_subway_line", "")); pb, nb = _nfc(prev_row.get("closest_bus_station", "")), _nfc(prev_row.get("closest_bus_station", "")); ps, ns = _norm_station(ps_raw), _norm_station(ns_raw)
    if ps and ns and ps == ns: return "same_subway_station", "", "", d_km
    if pb and nb and pb == nb: return "same_bus_station", "", "", d_km
    if pl and nl and ps and ns and (pl == nl): return "same_subway_line", "", "", d_km
    if d_km < 0.3: return "walk_hint", "", "", d_km
    if ps_raw and ns_raw and ps != ns: t1 = f"지하철 { _line_station_text(pl, ps_raw) }".strip(); t2 = f"{ _line_station_text(nl, ns_raw) } 하차".strip(); return "subway_hint", t1, t2, d_km
    if pb and nb and pb != nb: return "bus_hint", f"버스 {pb} 승차", f"{nb} 하차", d_km
    return "walk_hint", "", "", d_km
def _nfc(s: str) -> str: return ud.normalize("NFC", str(s or "")).strip()
def _read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try: return pd.read_csv(path, encoding=enc)
        except Exception: pass
    return pd.read_csv(path)
def _haversine_np(lat1, lon1, lat_arr, lon_arr): lat1r, lon1r = np.radians(float(lat1)), np.radians(float(lon1)); lat2r, lon2r = np.radians(pd.to_numeric(lat_arr, errors="coerce").astype(float)), np.radians(pd.to_numeric(lon_arr, errors="coerce").astype(float)); dphi, dlmb = lat2r - lat1r, lon2r - lon1r; a = np.sin(dphi/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlmb/2.0)**2; return 2.0 * 6371.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
def _haversine(lat1, lon1, lat2, lon2) -> float: R = 6371.0; phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2)); dphi, dlmb = math.radians(float(lat2) - float(lat1)), math.radians(float(lon2) - float(lon1)); a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2)**2; return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
def _geocode_region_kakao(region_name: str) -> Optional[Tuple[float, float]]:
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"; headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}; params  = {"query": region_name}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        docs = resp.json().get("documents", [])
        if docs: return float(docs[0]["y"]), float(docs[0]["x"])
    except Exception: return None
def _enrich_transit_hints_fast(df: pd.DataFrame, kakao_key: str) -> pd.DataFrame:
    if df.empty: return df
    session = requests.Session(); session.headers.update({"Authorization": f"KakaoAK {kakao_key}"})
    def _get(url, params):
        try:
            r = session.get(url, params=params, timeout=5)
            if r.status_code == 200: return r.json()
        except Exception: return None
    def nearest_subway(lat, lon):
        js = _get("https://dapi.kakao.com/v2/local/search/category.json", {"category_group_code": "SW8", "x": lon, "y": lat, "radius": 900, "size": 1, "sort": "distance"})
        if js and js.get("documents"): d = js["documents"][0]; name = _nfc(d.get("place_name")); raw = " ".join([name, _nfc(d.get("category_name", "")), _nfc(d.get("address_name", "")), _nfc(d.get("road_address_name", ""))]); m = re.search(r"(\d+)\s*호선", raw); return name, f"{m.group(1)}호선" if m else ""
        return "", ""
    def nearest_bus(lat, lon):
        for r in [900, 1200, 1500]:
            for kw in ["버스정류장", "정류장", "버스"]:
                js = _get("https://dapi.kakao.com/v2/local/search/keyword.json", {"query": kw, "x": lon, "y": lat, "radius": r, "size": 10, "sort": "distance"})
                if js and js.get("documents"):
                    docs = sorted(js["documents"], key=lambda d: int(float(d.get("distance", "1e9"))))
                    for d in docs: nm = _nfc(d.get("place_name"));
                    if any(k in nm for k in ["정류", "버스", "정류장", "정류소"]): return nm
                    return _nfc(docs[0].get("place_name"))
        return ""
    df["tile_key"] = [f"{round(a/0.0025)*0.0025:.6f}|{round(b/0.0025)*0.0025:.6f}" for a, b in zip(df["lat"], df["lon"])]
    tiles = df["tile_key"].unique().tolist(); subway_map, bus_map = {}, {}
    def job(t, func): return t, func(float(df.loc[df["tile_key"] == t].iloc[0]["lat"]), float(df.loc[df["tile_key"] == t].iloc[0]["lon"]))
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(job, t, nearest_subway) for t in tiles] + [ex.submit(job, t, nearest_bus) for t in tiles]
        for fu in as_completed(futs):
            try:
                t, val = fu.result();
                if isinstance(val, tuple): subway_map[t] = val
                else: bus_map[t] = val
            except Exception: pass
    df["closest_subway_station"] = df["tile_key"].map(lambda k: subway_map.get(k, ("", ""))[0]); df["closest_subway_line"] = df["tile_key"].map(lambda k: subway_map.get(k, ("", ""))[1]); df["closest_bus_station"] = df["tile_key"].map(lambda k: bus_map.get(k, ""))
    return df.drop(columns=["tile_key"])
def _greedy_route(rows: pd.DataFrame, start_lat: float, start_lon: float) -> pd.DataFrame:
    if rows.empty: return pd.DataFrame()
    rows = rows.reset_index(drop=True); used_indices, ordered_indices = set(), []; cur_lat, cur_lon = start_lat, start_lon
    initial_costs = rows.apply(lambda row: _step_cost_from(cur_lat, cur_lon, row, is_initial=True), axis=1)
    if initial_costs.empty: return pd.DataFrame()
    first_idx = initial_costs.idxmin(); ordered_indices.append(first_idx); used_indices.add(first_idx); cur_lat, cur_lon = float(rows.loc[first_idx, "lat"]), float(rows.loc[first_idx, "lon"])
    while len(used_indices) < len(rows):
        candidate_indices = [i for i in rows.index if i not in used_indices]
        if not candidate_indices: break
        costs = {k: _step_cost_from(cur_lat, cur_lon, rows.loc[k]) for k in candidate_indices}; next_idx = min(costs, key=costs.get)
        ordered_indices.append(next_idx); used_indices.add(next_idx); cur_lat, cur_lon = float(rows.loc[next_idx, "lat"]), float(rows.loc[next_idx, "lon"])
    return rows.loc[ordered_indices].reset_index(drop=True)
def _split_visits(total: int, days: int, vmin: int, vmax: int) -> list:
    counts, rem, rem_days = [], total, days
    for _ in range(days):
        if rem_days <= 0: counts.append(0); continue
        target = math.ceil(rem / rem_days) if rem_days > 0 else 0
        target = max(vmin if rem >= vmin else rem, min(target, vmax)); target = min(target, rem)
        counts.append(target); rem -= target; rem_days -= 1
    return counts
def _ensure_variety_pool(route_df: pd.DataFrame, start_idx: int, used_titles: set, want: int) -> pd.DataFrame:
    def used_filter(row): return _nfc(row.get("title", "")) in used_titles
    chunk_size = max(want * 8, want + 15)
    chunk = route_df.iloc[start_idx:start_idx + chunk_size].copy()
    if chunk.empty: return chunk
    chunk = chunk.loc[~chunk.apply(used_filter, axis=1)].copy()
    is_meal = chunk["cat1_norm"] == MEAL_CAT
    non_meal_cnt = int((~is_meal).sum())
    if non_meal_cnt < 3 and len(chunk) < want:
        rest = route_df.iloc[start_idx + chunk_size:].copy()
        if not rest.empty:
            rest = rest.loc[~rest.apply(used_filter, axis=1)]
            rest_nm = rest.loc[rest["cat1_norm"] != MEAL_CAT]
            if not rest_nm.empty: chunk = pd.concat([chunk, rest_nm.head(3 - non_meal_cnt)], ignore_index=True)
    return chunk
def _step_cost_from(lat, lon, nxt_row, is_initial=False):
    d_km = _haversine(lat, lon, nxt_row["lat"], nxt_row["lon"]); s = nxt_row.get("final_score", 0.0)
    score_weight = 2.0 if is_initial else 1.0
    cost = d_km - (s * score_weight)
    return cost
def _norm_station(s: str) -> str: t = _nfc(s); t = re.sub(r"\(.*?\)", "", t); t = t.replace("역", ""); t = re.sub(r"\s+", "", t); return t
def _line_station_text(line: str, station: str) -> str:
    line, station = _nfc(line), _nfc(station);
    if line and (line in station): return station
    return f"{line} {station}".strip() if line else station
def _odsay_total_minutes(plon, plat, nlon, nlat, api_key: str) -> Optional[int]:
    if not api_key: return None
    url = "https://api.odsay.com/v1/api/searchPubTransPathT"; params = {"SX": f"{plon:.6f}", "SY": f"{plat:.6f}", "EX": f"{nlon:.6f}", "EY": f"{nlat:.6f}", "apiKey": api_key, "lang": 0, "OPT": 0}
    try:
        r = requests.get(url, params=params, timeout=5)
        js = r.json()
        if js and "result" in js and "path" in js["result"] and js["result"]["path"]:
            tt = js["result"]["path"][0].get("info", {}).get("totalTime")
            if isinstance(tt, (int, float)) and tt > 0: return int(tt)
    except Exception: pass
    return None
def _to_dt(base: datetime, hhmm: str) -> datetime: h, m = map(int, hhmm.split(":")); return base.replace(hour=h, minute=m, second=0, microsecond=0)
def _check_hhmm(s: str):
    if not re.fullmatch(r"\d{2}:\d{2}", s or ""): raise ValueError("시간 형식은 HH:MM 이어야 합니다.")