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

from recommend.config import *
from recommend import run_walk as run_walk_module 

from dotenv import load_dotenv
load_dotenv()
import os

_KAKAO_ENV = os.environ.get("KAKAO_API_KEY")
if _KAKAO_ENV: KAKAO_API_KEY = _KAKAO_ENV
_ODSAY_ENV = os.environ.get("ODSAY_API_KEY")
if _ODSAY_ENV: ODSAY_API_KEY = _ODSAY_ENV

MEAL_CAT = "음식"
DAY_VISIT_MIN, DAY_VISIT_MAX = 2, 5
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
    if df.empty: return pd.DataFrame(columns=["day_label", "day", "start_time", "end_time", "title", "addr1", "cat1", "cat2", "cat3", "final_score", "stay_min", "출발지", "도착지", "교통편1", "교통편2", "distance_from_prev_km", "move_min"])

    # --- 교통정보 사전 조회 (최적화) ---
    top_n_for_transit_search = TRANSIT_TOP_N if FAST_MODE else 100
    pois_for_search = df.head(top_n_for_transit_search).copy()
    pois_with_hints = _enrich_transit_hints_fast(pois_for_search, KAKAO_API_KEY)
    
    final_candidates = pd.concat([
        pois_with_hints,
        df.loc[~df.index.isin(pois_with_hints.index)]
    ]).drop_duplicates(subset=["title", "addr1"]).reset_index(drop=True)

    transit_cols = ["closest_subway_station", "closest_subway_line", "closest_bus_station"]
    for col in transit_cols:
        if col in final_candidates.columns:
            final_candidates[col] = final_candidates[col].fillna("")
    
    route = final_candidates

    # ======================================
    # 스케줄링 로직 (v3: Transit-First Seeding)
    # ======================================
    visit_counts = _split_visits(len(route), days, DAY_VISIT_MIN, DAY_VISIT_MAX)
    rows_all: List[dict] = []
    used_indices = set()
    midnight0 = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for d in range(1, days + 1):
        base  = midnight0 + timedelta(days=d - 1)
        day_label = f"{d}일차"
        want  = visit_counts[d - 1] if d - 1 < len(visit_counts) else 0
        if want < DAY_VISIT_MIN: continue

        day_pool = route[~route.index.isin(used_indices)].copy()
        if len(day_pool) < DAY_VISIT_MIN: continue

        day_start = _to_dt(base, start_time)
        day_end   = _to_dt(base, end_time)
        
        cur_time = day_start
        day_schedule: List[pd.Series] = []
        day_rows_final: List[dict] = []
        
        place_a = None
        for idx, row in day_pool.iterrows():
            if idx not in used_indices:
                place_a = row
                break
        if place_a is None: continue

        partners = day_pool[day_pool.index != place_a.name]
        transit_partners = partners[partners.apply(lambda r: _relation_and_text(place_a, r)[0] in {"subway_hint", "bus_hint"}, axis=1)].copy()
        
        if transit_partners.empty:
            print(f"INFO: {day_label} - {place_a['title']}에서 출발하는 적절한 대중교통 파트너를 찾지 못해 해당일 계획을 건너뜁니다.")
            continue
            
        transit_partners['cost'] = transit_partners.apply(lambda r: _step_cost_from(place_a['lat'], place_a['lon'], r), axis=1)
        place_b = transit_partners.loc[transit_partners['cost'].idxmin()]
        
        t_stay_a = stay_minutes(place_a["cat1"])
        move_info = _calculate_move_info(place_a, place_b)
        t_move_ab = move_info["move_min"]
        t_stay_b = stay_minutes(place_b["cat1"])
        
        st_a = cur_time
        et_a = st_a + timedelta(minutes=t_stay_a)
        st_b = et_a + timedelta(minutes=t_move_ab)
        et_b = st_b + timedelta(minutes=t_stay_b)
        
        if et_b > day_end:
            print(f"INFO: {day_label} - 핵심 경로({place_a['title']}->{place_b['title']})만으로도 시간이 초과되어 계획을 건너뜁니다.")
            continue
        
        day_schedule.extend([place_a, place_b])
        day_rows_final.append(_create_visit_row(day_label, d, st_a, et_a, place_a, t_stay_a))
        day_rows_final.append(_create_move_row(day_label, d, et_a, st_b, place_a, place_b, move_info))
        day_rows_final.append(_create_visit_row(day_label, d, st_b, et_b, place_b, t_stay_b))
        cur_time = et_b

        quota = build_day_quota(cats, want, BASE_WEIGHTS)
        quota[place_a["cat1_norm"]] = quota.get(place_a["cat1_norm"], 1) - 1
        quota[place_b["cat1_norm"]] = quota.get(place_b["cat1_norm"], 1) - 1

        while len(day_schedule) < want and cur_time < day_end:
            prev_place = day_schedule[-1]
            ref_lat, ref_lon = prev_place["lat"], prev_place["lon"]
            current_scheduled_indices = {p.name for p in day_schedule}
            available_pool = day_pool.loc[~day_pool.index.isin(current_scheduled_indices)].copy()
            valid_cats = {c for c, q in quota.items() if q > 0}
            available_pool = available_pool[available_pool["cat1_norm"].isin(valid_cats)]
            if available_pool.empty: break
            pref_pool, fallback_pool = _get_preferred_pool(available_pool, cur_time, base, quota)
            next_place = _select_next_place(pref_pool, fallback_pool, prev_place, False, ref_lat, ref_lon)
            if next_place is None: break
            move_info = _calculate_move_info(prev_place, next_place)
            st = cur_time + timedelta(minutes=move_info["move_min"])
            et = st + timedelta(minutes=stay_minutes(next_place["cat1"]))
            if et > day_end: break
            if move_info["move_min"] > 0:
                day_rows_final.append(_create_move_row(day_label, d, cur_time, st, prev_place, next_place, move_info))
            day_rows_final.append(_create_visit_row(day_label, d, st, et, next_place, stay_minutes(next_place["cat1"])))
            cur_time = et
            day_schedule.append(next_place)
            quota[next_place["cat1_norm"]] = quota.get(next_place["cat1_norm"], 1) - 1

        rows_all.extend(day_rows_final)
        used_indices.update(p.name for p in day_schedule)

    final_df = pd.DataFrame(rows_all)
    if final_df.empty and days > 0:
        print("WARNING: 규칙을 만족하는 대중교통 일정을 생성하지 못했습니다. 도보 기준으로 다시 추천합니다.")
        return run_walk_module.run(region=region, transport_mode='walk', score_label=score_label, days=days, cats=cats)
    return final_df

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
    if df_sorted_themed.empty: df_sorted = df.copy()
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

def stay_minutes(cat1: str) -> int:
    c = _nfc(cat1)
    if c == "음식": return 75
    if c == "자연": return 90
    if c == "레포츠": return 120
    if c == "쇼핑": return 90
    if c == "인문(문화/예술/역사)": return 100
    return 90

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
        candidates = pool.to_dict('records')
        costs = [_step_cost_from(ref_lat, ref_lon, r) for r in candidates]
        if not costs: return None
        best_idx_in_pool = np.argmin(costs)
        return pool.iloc[best_idx_in_pool]
    if force_transit and prev_row is not None:
        transit_pref = pref[pref.apply(lambda r: _relation_and_text(prev_row, r)[0] in {"subway_hint", "bus_hint"}, axis=1)]
        if not transit_pref.empty: return find_best(transit_pref)
    best_pref = find_best(pref)
    if best_pref is not None: return best_pref
    return find_best(fallback)

def _calculate_move_info(prev, next):
    if prev is None: return {"move_min": 0, "is_transit": False, "d_km": 0, "t1": "", "t2": ""}
    rel, t1, t2, d_km = _relation_and_text(prev, next)
    is_transit = rel in {"subway_hint", "bus_hint"}
    move_min = 0
    if is_transit: move_min = int((d_km / 18.0) * 60 + 8)
    else: move_min = int((d_km / 4.5) * 60)
    return {"move_min": max(5, move_min), "is_transit": is_transit, "d_km": d_km, "t1": t1, "t2": t2}

def _create_move_row(label, day, st, et, prev, next, move_info):
    from_title = _nfc(prev.get("title")); to_title = _nfc(next.get("title"))
    return {"day_label": label, "day": day, "start_time": st.strftime("%H:%M"), "end_time": et.strftime("%H:%M"), "title": "이동", "addr1": "", "출발지": from_title, "도착지": to_title, "교통편1": move_info["t1"], "교통편2": move_info["t2"], "distance_from_prev_km": round(move_info["d_km"], 2), "move_min": move_info["move_min"], "stay_min": 0}

def _create_visit_row(label, day, st, et, place, stay_min): return {"day_label": label, "day": day, "start_time": st.strftime("%H:%M"), "end_time": et.strftime("%H:%M"), "title": _nfc(place["title"]), "addr1": _nfc(place["addr1"]), "cat1": _nfc(place["cat1"]), "cat2": _nfc(place["cat2"]), "cat3": _nfc(place["cat3"]), "final_score": float(place.get("final_score", np.nan)), "stay_min": stay_min}

def _relation_and_text(prev_row, nxt_row):
    d_km = _haversine(prev_row["lat"], prev_row["lon"], nxt_row["lat"], nxt_row["lon"])
    ps_name, ps_line = _nfc(prev_row.get("closest_subway_station", "")), _nfc(prev_row.get("closest_subway_line", ""))
    ns_name, ns_line = _nfc(nxt_row.get("closest_subway_station", "")), _nfc(nxt_row.get("closest_subway_line", ""))
    pb_name, nb_name = _nfc(prev_row.get("closest_bus_station", "")), _nfc(nxt_row.get("closest_bus_station", ""))
    ps_norm, ns_norm = _norm_station(ps_name), _norm_station(ns_name)
    if ps_name and ns_name and ps_norm == ns_norm: return "same_subway_station", "", "", d_km
    if pb_name and nb_name and pb_name == nb_name: return "same_bus_station", "", "", d_km
    if ps_name and ns_name:
        t1 = f"지하철 {_line_station_text(ps_name, ps_line)} 승차"
        t2 = f"{_line_station_text(ns_name, ns_line)} 하차"
        return "subway_hint", t1, t2, d_km
    if pb_name and nb_name:
        t1 = f"버스 {pb_name} 승차"
        t2 = f"{nb_name} 하차"
        return "bus_hint", t1, t2, d_km
    return "walk_hint", "도보 이동", "", d_km

def _nfc(s: str) -> str: return ud.normalize("NFC", str(s or "")).strip()

def _read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try: return pd.read_csv(path, encoding=enc)
        except Exception: pass
    return pd.read_csv(path)

def _haversine_np(lat1, lon1, lat_arr, lon_arr): lat1r, lon1r = np.radians(float(lat1)), np.radians(float(lon1)); lat2r, lon2r = np.radians(pd.to_numeric(lat_arr, errors="coerce").astype(float)), np.radians(pd.to_numeric(lon_arr, errors="coerce").astype(float)); dphi, dlmb = lat2r - lat1r, lon2r - lon1r; a = np.sin(dphi/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlmb/2.0)**2; return 2.0 * 6371.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

def _haversine(lat1, lon1, lat2, lon2) -> float: R = 6371.0; phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2)); dphi, dlmb = math.radians(float(lat2) - float(lat1)), math.radians(float(lon2) - float(lon1)); a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2)**2; return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _geocode_region_kakao(region_name: str) -> Optional[Tuple[float, float]]:
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    queries_to_try = [region_name]
    if not region_name.endswith("역"): queries_to_try.append(f"{region_name}역")
    for query in queries_to_try:
        params = {"query": query}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            docs = data.get("documents", [])
            if docs:
                first = docs[0]
                print(f"INFO: 지오코딩 성공: '{query}' -> '{first.get('place_name')}'")
                return float(first["y"]), float(first["x"])
        except requests.exceptions.RequestException as e: print(f"ERROR: 카카오 API 요청 실패 ({query}): {e}")
        except Exception as e: print(f"ERROR: 지오코딩 중 알 수 없는 오류 ({query}): {e}")
    return None

# ▼▼▼ [최종 수정] 교통 정보 처리 로직 ▼▼▼
def _parse_kakao_subway_doc(doc: Dict) -> Tuple[str, str]:
    """
    카카오 API의 지하철역 응답 하나를 파싱하여 역 이름과 정리된 대표 노선명 1개를 반환합니다.
    """
    name = _nfc(doc.get("place_name", ""))
    category = _nfc(doc.get("category_name", ""))
    
    # 1. place_name에서 'X호선'이 명확하게 있으면 그걸 최우선으로 사용
    # 예: "시청역 1호선" -> ("시청역", "1호선")
    match = re.search(r'(.+?역)\s*(.+?호선)', name)
    if match:
        station, line = match.groups()
        return _nfc(station), _nfc(line)

    # 2. category_name에서 파싱
    # `>`로 분리된 마지막 부분이 가장 구체적인 정보이므로 그것을 사용
    cat_parts = [p.strip() for p in category.split('>')]
    full_text = name + " " + cat_parts[-1]
    
    # 모든 종류의 호선 정보 추출 (e.g., "1호선", "대전 도시철도 1호선")
    lines_found = re.findall(r'([^,>\s]+호선)', full_text)
    if not lines_found:
        return name, ""

    # 가장 대표적인 노선 1개만 선택 (숫자로 시작하는 것을 우선)
    lines_found.sort(key=lambda x: (not x[0].isdigit(), x))
    return name, lines_found[0]


def _enrich_transit_hints_fast(df: pd.DataFrame, kakao_key: str) -> pd.DataFrame:
    if df.empty: return df
    session = requests.Session(); session.headers.update({"Authorization": f"KakaoAK {kakao_key}"})
    def _get(url, params):
        try:
            r = session.get(url, params=params, timeout=5)
            if r.status_code == 200: return r.json()
        except Exception: return None

    def get_transit_info_for_tile(lat, lon):
        subway_name, subway_line, bus_name = "", "", ""
        sw_js = _get("https://dapi.kakao.com/v2/local/search/category.json", {"category_group_code": "SW8", "x": lon, "y": lat, "radius": 1000, "size": 1, "sort": "distance"})
        if sw_js and sw_js.get("documents"):
            subway_name, subway_line = _parse_kakao_subway_doc(sw_js["documents"][0])
        
        bus_js = _get("https://dapi.kakao.com/v2/local/search/category.json", {"category_group_code": "BS0", "x": lon, "y": lat, "radius": 1000, "size": 1, "sort": "distance"})
        if bus_js and bus_js.get("documents"):
            bus_name = _nfc(bus_js["documents"][0].get("place_name"))
        return subway_name, subway_line, bus_name

    df["tile_key"] = [f"{round(a/0.005)*0.005:.5f}|{round(b/0.005)*0.005:.5f}" for a, b in zip(df["lat"], df["lon"])]
    unique_tiles = df["tile_key"].unique().tolist()
    tile_transit_map = {}

    with ThreadPoolExecutor(max_workers=12) as ex:
        future_to_tile = {ex.submit(get_transit_info_for_tile, float(df.loc[df["tile_key"] == t].iloc[0]["lat"]), float(df.loc[df["tile_key"] == t].iloc[0]["lon"])): t for t in unique_tiles}
        for future in as_completed(future_to_tile):
            tile = future_to_tile[future]
            try:
                tile_transit_map[tile] = future.result()
            except Exception as exc:
                print(f'{tile} generated an exception: {exc}')

    df["closest_subway_station"] = df["tile_key"].map(lambda k: tile_transit_map.get(k, ("", "", ""))[0])
    df["closest_subway_line"] = df["tile_key"].map(lambda k: tile_transit_map.get(k, ("", "", ""))[1])
    df["closest_bus_station"] = df["tile_key"].map(lambda k: tile_transit_map.get(k, ("", "", ""))[2])
    return df.drop(columns=["tile_key"])
# ▲▲▲ [최종 수정] 교통 정보 처리 로직 끝 ▲▲▲

def _split_visits(total: int, days: int, vmin: int, vmax: int) -> list:
    counts, rem, rem_days = [], total, days
    for _ in range(days):
        if rem_days <= 0: counts.append(0); continue
        target = math.ceil(rem / rem_days) if rem_days > 0 else 0
        target = max(vmin if rem >= vmin else rem, min(target, vmax)); target = min(target, rem)
        counts.append(target); rem -= target; rem_days -= 1
    return counts

def _step_cost_from(lat, lon, nxt_row, is_initial=False):
    d_km = _haversine(lat, lon, nxt_row.get("lat"), nxt_row.get("lon"))
    s = nxt_row.get("final_score", 0.0)
    score_weight = 2.0 if is_initial else 1.0
    cost = d_km - (s * score_weight)
    return cost

def _norm_station(s: str) -> str: t = _nfc(s); t = re.sub(r"\(.*?\)", "", t); t = t.replace("역", ""); t = re.sub(r"\s+", "", t); return t

def _line_station_text(station: str, line: str) -> str:
    """
    [수정된 함수]
    역 이름과 노선 정보를 받아 가장 깔끔한 형태의 "역이름 노선" 문자열 1개를 반환합니다.
    - 예: ("시청역", "1호선, 2호선") -> "시청역 1호선"
    """
    station, line = _nfc(station), _nfc(line)
    if not station:
        return ""

    # line 변수에서 모든 'X호선' 또는 '경강선' 같은 노선명을 추출
    all_lines = re.findall(r'([\w\d]+선)', str(line))

    # 만약 line 변수에서 노선을 못 찾으면, station 이름 자체에서 찾아봄
    if not all_lines:
        match = re.search(r'(\S+호선)', station)
        if match:
            return station # "시청역 1호선" 처럼 이미 완벽한 형태면 그대로 반환
        return station   # 노선 정보가 없으면 역 이름만 반환

    # 추출된 노선 중 대표 노선 1개 선택 (숫자 > 가나다 순)
    all_lines.sort(key=lambda x: (not x[0].isdigit(), x))
    best_line = all_lines[0]
    
    # 역 이름에 이미 대표 노선이 포함되어 있으면 그대로 반환 (중복 방지)
    if best_line in station:
        return station
    
    # "역이름"과 "대표노선"을 합쳐서 최종 텍스트 생성
    return f"{station} {best_line}".strip()

def _to_dt(base: datetime, hhmm: str) -> datetime: h, m = map(int, hhmm.split(":")); return base.replace(hour=h, minute=m, second=0, microsecond=0)