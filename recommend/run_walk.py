# -*- coding: utf-8 -*-
from __future__ import annotations

import json, math, requests
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
from pathlib import Path

import pandas as pd
import numpy as np
import unicodedata as ud

# ⚠️ PATH_TMF, KAKAO_API_KEY, ROOT, FAST_MODE 는 config에서만 관리
from recommend.config import *  # PATH_TMF, KAKAO_API_KEY, ROOT, FAST_MODE  # noqa

# =========================
# Globals & Caches
# =========================
_TMF_CACHE: Optional[pd.DataFrame] = None             # ### NEW: CSV 메모리 캐시
_GEOCODE_MEMO: dict[str, tuple[float, float]] = {}    # ### NEW: 지오코딩 메모리 캐시
_GEOCODE_CACHE_FILE = Path(ROOT) / "_cache_geocode.json"  # ### NEW: 파일 캐시

def _load_geocode_cache() -> dict:
    try:
        return json.loads(Path(_GEOCODE_CACHE_FILE).read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_geocode_cache(d: dict) -> None:
    try:
        Path(_GEOCODE_CACHE_FILE).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

# =========================
# Public API
# =========================
def run(
    region: str,
    transport_mode: str,             # 'walk' | 'transit' (여기선 반경 계산에만 사용)
    score_label: str,                # '인기도지수' | '관광지수'
    days: int,                       # UI 슬라이더 값
    cats: List[str],                 # 예: ["음식","자연","레포츠"] (선호 순서)
) -> pd.DataFrame:
    """
    도보 중심 일정표 DataFrame을 반환.
    """
    # ----- 입력 검증 -----
    region = (region or "").strip()
    if not region:
        raise ValueError("여행 지역을 입력하세요.")
    if transport_mode not in {"walk", "transit"}:
        raise ValueError("이동수단은 'walk' 또는 'transit' 이어야 합니다.")
    if score_label not in {"인기도지수", "관광지수"}:
        raise ValueError("점수 기준은 '인기도지수' 또는 '관광지수' 중 선택하세요.")
    if not isinstance(days, int) or days <= 0:
        raise ValueError("여행 일수를 1 이상으로 지정하세요.")
    days = max(1, min(100, int(days)))  # 상한 보호
    for i in cats:
        if cats == "문화":
            cats = "인문(문화/예술/역사)"
            
    cats = list(dict.fromkeys([_nfc(c) for c in (cats or [])]))
    if not cats:
        raise ValueError("테마를 최소 1개 이상 선택하세요. (예: 음식, 자연, 레포츠)")
    if len(cats) > 3:
        cats = cats[:3]

    # ----- 지역 지오코딩 (캐시/폴백/짧은 타임아웃) -----
    center_lat, center_lon = _geocode_region_cached_or_fallback(region)

    # ----- 원본 CSV 로드(1회 캐시) + 필요한 열만 사용 -----
    df = _load_tmf_cached()

    # 필수 컬럼 확인 및 표준화
    cols_lower = {c.lower(): c for c in df.columns}
    need = {
        "title": cols_lower.get("title"),
        "addr1": cols_lower.get("addr1"),
        "cat1":  cols_lower.get("cat1") or _first_contains(df.columns, "cat1"),
        "mapx":  cols_lower.get("mapx") or cols_lower.get("lon") or cols_lower.get("longitude") or cols_lower.get("x"),
        "mapy":  cols_lower.get("mapy") or cols_lower.get("lat") or cols_lower.get("latitude") or cols_lower.get("y"),
        "tour_score": cols_lower.get("tour_score"),
        "review_score": cols_lower.get("review_score"),
    }
    miss = [k for k, v in need.items() if v is None]
    if miss:
        raise KeyError(f"필수 컬럼 누락: {miss} / 실제컬럼: {list(df.columns)}")

    df = df.rename(columns={
        need["title"]: "title",
        need["addr1"]: "addr1",
        need["cat1"]:  "cat1",
        need["mapx"]:  "lon",      # mapx = 경도
        need["mapy"]:  "lat",      # mapy = 위도
        need["tour_score"]: "tour_score",
        need["review_score"]: "review_score",
    }).copy()

    # 타입 정리
    for c in ("lat", "lon", "tour_score", "review_score"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()
    if "cat2" not in df.columns: df["cat2"] = ""
    if "cat3" not in df.columns: df["cat3"] = ""

    # ----- 1차 Bounding Box → 2차 벡터화 하버사인 -----
    radius_km = 5 if transport_mode == "walk" else 20
    df = _bbox_prefilter(df, center_lat, center_lon, radius_km)              # ### NEW: bbox
    if df.empty:
        raise RuntimeError("선택한 지역 주변(반경)에서 후보가 없습니다. 지역/반경/데이터를 확인하세요.")

    dist_km = _haversine_np(center_lat, center_lon, df["lat"].values, df["lon"].values)  # ### NEW: vectorized
    df = df.assign(distance_km=dist_km)
    df = df[df["distance_km"] <= radius_km].copy()
    if df.empty:
        raise RuntimeError("선택한 지역 주변(반경)에서 후보가 없습니다. 지역/반경/데이터를 확인하세요.")

    # ----- 정렬/슬라이싱 (FAST_MODE 가속) -----
    score_map = {"관광지수": "tour_score", "인기도지수": "review_score"}
    score_col = score_map[score_label]

    # 우선 점수 NaN → 0 처리
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)

    # FAST_MODE: 전체 상위 TOPK → 2차 정렬
    if 'FAST_MODE' in globals() and FAST_MODE:
        TOPK_ALL = 300   # 전체 상한
        if len(df) > TOPK_ALL:
            keep_idx = df[score_col].nlargest(TOPK_ALL).index
            df = df.loc[keep_idx]
    df = df.sort_values([score_col, "distance_km"], ascending=[False, True]).copy()
    df = df.drop_duplicates(subset=["title", "addr1"], keep="first").reset_index(drop=True)

    # ----- 선택 카테고리 우선 + 결핍 보충 -----
    FALLBACK_N = 5
    df_sorted = df[df["cat1"].isin(cats)].copy() if cats else df.copy()
    pool = df.copy()
    for cat in cats:
        if (df_sorted["cat1"] == cat).sum() == 0:
            remain = pool.loc[~pool.index.isin(df_sorted.index)]
            fb = remain.sort_values(["distance_km", score_col], ascending=[True, False]).head(FALLBACK_N)
            if not fb.empty:
                df_sorted = pd.concat([df_sorted, fb], ignore_index=True)

    # 메타/정규화 (문자 정규화는 1회만)
    df_sorted["cat1_norm"] = df_sorted["cat1"].map(_nfc)
    df_sorted["cat3_norm"] = df_sorted["cat3"].astype(str).map(_nfc)
    df_sorted["final_score"] = pd.to_numeric(
        df_sorted["review_score"].fillna(df_sorted["tour_score"]), errors="coerce"
    ).fillna(0.0)
    df_sorted = df_sorted.sort_values([score_col, "distance_km"], ascending=[False, True]).reset_index(drop=True)

    # ======================================
    # 스케줄링 (도보 기준) — 구조는 유지, 내부 선택만 경량화
    # ======================================
    MEAL_CAT = "음식"
    DAY_TOTAL_SLOTS = 5
    BASE_WEIGHTS = [3, 2, 1]
    BLOCKED_CAFE_KEYS = {"카페", "전통찻집"}
    MEAL_CUISINE_TAGS = {"서양식", "이색음식점", "일식", "중식", "한식"}

    selected_pool = df_sorted[df_sorted["cat1_norm"].isin(cats)].copy()
    if selected_pool.empty:
        selected_pool = df_sorted.copy()

    # 선택 풀 사이즈 제한(FAST_MODE)
    if 'FAST_MODE' in globals() and FAST_MODE:
        PER_CAT_TOP = 120
        take_frames = []
        for c in cats:
            sub = selected_pool[selected_pool["cat1_norm"] == c]
            if len(sub) > PER_CAT_TOP:
                sub = sub.nlargest(PER_CAT_TOP, columns=[score_col])
            take_frames.append(sub)
        if take_frames:
            selected_pool = pd.concat(take_frames + [selected_pool[~selected_pool["cat1_norm"].isin(cats)]], ignore_index=True).drop_duplicates()

    # 중심점 재설정(도보 탐색 안정화)
    center_lat = float(selected_pool["lat"].mean())
    center_lon = float(selected_pool["lon"].mean())

    def travel_minutes(d_km: float) -> float:
        speed_kmh = 4.5  # 도보
        return max(5.0, (float(d_km) / speed_kmh) * 60.0)

    def stay_minutes(cat: str) -> float:
        c = _nfc(cat)
        if len(cats) >= 1 and c == cats[0]: return 75.0
        if len(cats) >= 2 and c == cats[1]: return 90.0
        if len(cats) >= 3 and c == cats[2]: return 120.0
        return 90.0

    def is_blocked_cafe_tag(tag: str) -> bool:
        s = _nfc(tag)
        return any(k in s for k in BLOCKED_CAFE_KEYS)

    def pick_best(sub: pd.DataFrame, cur_lat: float, cur_lon: float):
        if sub.empty: return None, None
        # 벡터화 거리 + 패널티
        dkm = np.sqrt((sub["lat"].values - cur_lat) ** 2 + (sub["lon"].values - cur_lon) ** 2) * 111.0
        pen = (dkm / 4.5) + 10.0  # 이동시간(분) 근사 + 고정비
        sc = sub["final_score"].fillna(0).values - 0.1 * (pen / 60.0)
        idx_local = int(np.argmax(sc))
        return sub.iloc[idx_local].name, float(dkm[idx_local])

    def build_day_quota() -> dict:
        u = list(dict.fromkeys(cats))
        quotas = {c: 0 for c in u}
        for i, w in enumerate(BASE_WEIGHTS):
            if i < len(u):
                quotas[u[i]] += w
        remain = DAY_TOTAL_SLOTS - sum(quotas.values())
        i = 0
        while remain > 0 and u:
            quotas[u[i % len(u)]] += 1
            remain -= 1
            i += 1
        return quotas

    # 본 스케줄링
    itins = []
    cur_lat, cur_lon = center_lat, center_lon
    remain_pool = selected_pool.copy()

    today0 = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    meal_enabled = (MEAL_CAT in set(cats))

    # used 마킹(반복 drop 최소화)
    used_idx = set()

    for d in range(1, days + 1):
        base = today0 + timedelta(days=d - 1)
        day_start = base.replace(hour=9, minute=0)
        lunch_s, lunch_e = base.replace(hour=11, minute=0), base.replace(hour=13, minute=0)
        dinner_s, dinner_e = base.replace(hour=17, minute=0), base.replace(hour=20, minute=0)
        day_end = base.replace(hour=22, minute=30)

        quotas = build_day_quota()
        cur_time = day_start
        day_rows: List[dict] = []

        def _move_row(day_i: int, st: datetime, et: datetime, from_title: str, to_title: str) -> dict:
            return {
                "day": day_i, "start_time": st.strftime("%H:%M"), "end_time": et.strftime("%H:%M"),
                "title": "이동", "addr1": "", "cat1": "", "cat2": "", "cat3": "",
                "final_score": np.nan, "distance_from_prev_km": np.nan,
                "move_min": int((et - st).total_seconds() / 60), "stay_min": 0,
                "출발지": from_title, "도착지": to_title,
                "교통편1": from_title, "교통편2": to_title,
            }

        def _visit_row(day_i, st: datetime, et: datetime, row: pd.Series, cat: str, dkm: float, t_mv: float, t_st: float) -> dict:
            return {
                "day": day_i, "start_time": st.time().strftime("%H:%M"), "end_time": et.time().strftime("%H:%M"),
                "title": _nfc(row.get("title","")), "addr1": _nfc(row.get("addr1","")),
                "cat1": _nfc(row.get("cat1","")), "cat2": _nfc(row.get("cat2","")), "cat3": _nfc(row.get("cat3","")),
                "final_score": float(row.get("review_score") if pd.notna(row.get("review_score")) else row.get("tour_score") or 0),
                "distance_from_prev_km": round(float(dkm), 2) if pd.notna(dkm) else np.nan,
                "move_min": int(round(t_mv)), "stay_min": int(round(t_st)),
                "출발지": "", "도착지": "", "교통편1": "", "교통편2": "",
                "mapx": row.get("lon"), "mapy": row.get("lat")
            }

        def _select_and_place(cand_df: pd.DataFrame, cat_name: str, time_guard_end: datetime, block_cafe: bool = False) -> bool:
            nonlocal cur_time, cur_lat, cur_lon
            sub = cand_df[~cand_df.index.isin(used_idx)]
            if block_cafe:
                sub = sub[~sub["cat3_norm"].apply(is_blocked_cafe_tag)]
            if sub.empty: return False

            idx, dkm = pick_best(sub, cur_lat, cur_lon)
            if idx is None: return False
            row = sub.loc[idx]

            t_mv = travel_minutes(dkm) + 10.0
            t_st = stay_minutes(cat_name)

            move_start_time = cur_time
            move_end_time = cur_time + timedelta(minutes=t_mv)
            if move_end_time > time_guard_end: return False

            if day_rows:
                prev_title = day_rows[-1]["title"]
                next_title = _nfc(row.get("title", ""))
                day_rows.append(_move_row(d, move_start_time, move_end_time, prev_title, next_title))

            st = move_end_time
            et = st + timedelta(minutes=t_st)
            if et > time_guard_end:
                if day_rows and day_rows[-1]['title'] == '이동':
                    day_rows.pop()
                return False

            day_rows.append(_visit_row(d, st, et, row, cat_name, dkm, t_mv, t_st))
            used_idx.add(idx)
            cur_time = et
            cur_lat, cur_lon = float(row["lat"]), float(row["lon"])
            return True

        # 오전: 음식 제외 선점
        non_food = [c for c in cats if c != MEAL_CAT]
        for c in non_food:
            if quotas.get(c, 0) <= 0 or cur_time >= lunch_s: continue
            cand = remain_pool[(remain_pool["cat1_norm"] == c)]
            if _select_and_place(cand, c, lunch_s):
                quotas[c] -= 1

        # 오전 나머지(음식 제외)
        while cur_time < lunch_s and sum(quotas.values()) > 0:
            choices = [c for c in cats if quotas.get(c, 0) > 0 and c != MEAL_CAT]
            if not choices: break
            choices.sort(key=lambda x: -quotas.get(x, 0))
            placed = False
            for c in choices:
                cand = remain_pool[(remain_pool["cat1_norm"] == c)]
                if _select_and_place(cand, c, lunch_s):
                    quotas[c] -= 1
                    placed = True
                    break
            if not placed: break

        # 점심(음식 1곳)
        if meal_enabled and quotas.get(MEAL_CAT, 0) > 0:
            cur_time = max(cur_time, lunch_s)
            if cur_time < lunch_e:
                cand = remain_pool[(remain_pool["cat1_norm"] == MEAL_CAT)]
                if _select_and_place(cand, MEAL_CAT, lunch_e, block_cafe=True):
                    quotas[MEAL_CAT] -= 1

        # 오후(음식 제외)
        while cur_time < dinner_s and sum(quotas.values()) > 0:
            choices = [c for c in cats if quotas.get(c, 0) > 0 and c != MEAL_CAT]
            if not choices: break
            choices.sort(key=lambda x: -quotas.get(x, 0))
            placed = False
            for c in choices:
                cand = remain_pool[(remain_pool["cat1_norm"] == c)]
                if _select_and_place(cand, c, dinner_s):
                    quotas[c] -= 1
                    placed = True
                    break
            if not placed: break

        # 저녁(음식 1곳)
        if meal_enabled and quotas.get(MEAL_CAT, 0) > 0:
            cur_time = max(cur_time, dinner_s)
            if cur_time < dinner_e:
                cand = remain_pool[(remain_pool["cat1_norm"] == MEAL_CAT)]
                if _select_and_place(cand, MEAL_CAT, dinner_e, block_cafe=True):
                    quotas[MEAL_CAT] -= 1

        # 저녁 이후(~22:30)
        while cur_time < day_end and sum(quotas.values()) > 0:
            choices = [c for c in cats if quotas.get(c, 0) > 0]
            if not choices: break
            choices.sort(key=lambda x: -quotas.get(x, 0))
            placed = False
            for c in choices:
                cand = remain_pool[(remain_pool["cat1_norm"] == c)]
                # 20시 이후엔 메인식사 버킷 회피
                if c == MEAL_CAT:
                    cand = cand[~cand["cat3_norm"].apply(lambda s: _contains_any(s, MEAL_CUISINE_TAGS))]
                if _select_and_place(cand, c, day_end):
                    quotas[c] -= 1
                    placed = True
                    break
            if not placed: break

        if day_rows:
            day_df = pd.DataFrame(day_rows).sort_values("start_time").reset_index(drop=True)
            itins.append(day_df)

    itinerary = pd.concat(itins, ignore_index=True) if itins else pd.DataFrame(
        columns=["day","start_time","end_time","title","addr1","cat1","cat2","cat3",
                 "final_score","distance_from_prev_km","move_min","stay_min",
                 "출발지","도착지","교통편1","교통편2","mapx","mapy"]
    )
    return itinerary

_UI_THEME_TO_CAT1 = {
    "문화": "인문(문화/예술/역사)"
}

# =========================
# Helpers (Vectorized & Cached)
# =========================
def _nfc(s: str) -> str:
    return ud.normalize("NFC", str(s or "")).strip()

def _map_ui_cats_to_cat1(cats):
    """UI에서 넘어온 테마 리스트를 데이터셋의 cat1 값으로 매핑·정규화·중복제거."""
    out, seen = [], set()
    for c in (cats or []):
        c = _nfc(c)
        c = _UI_THEME_TO_CAT1.get(c, c)  # 문화 → 인문(문화/예술/역사)
        if c and c not in seen:
            out.append(c); seen.add(c)
    return out[:3]  # 상위 3개만 사용

def _first_contains(cols, key):
    key = key.lower()
    for c in cols:
        if key in c.lower():
            return c
    return None

def _load_tmf_cached() -> pd.DataFrame:
    global _TMF_CACHE
    if _TMF_CACHE is not None:
        return _TMF_CACHE
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            _TMF_CACHE = pd.read_csv(PATH_TMF, encoding=enc)
            break
        except Exception:
            continue
    if _TMF_CACHE is None:
        _TMF_CACHE = pd.read_csv(PATH_TMF)  # 마지막 시도
    return _TMF_CACHE

def _bbox_prefilter(df: pd.DataFrame, lat0: float, lon0: float, r_km: float) -> pd.DataFrame:
    # 위도/경도 단순 박스 마스크 (초저비용)
    dlat = r_km / 111.0
    dlon = r_km / (111.0 * max(0.1, math.cos(math.radians(lat0))))
    return df[(df["lat"].between(lat0 - dlat, lat0 + dlat)) &
              (df["lon"].between(lon0 - dlon, lon0 + dlon))].copy()

def _haversine_np(lat1, lon1, lat_arr, lon_arr):
    lat1r, lon1r = np.radians(float(lat1)), np.radians(float(lon1))
    lat2r, lon2r = np.radians(pd.to_numeric(lat_arr, errors="coerce")), np.radians(pd.to_numeric(lon_arr, errors="coerce"))
    dphi, dlmb = lat2r - lat1r, lon2r - lon1r
    a = np.sin(dphi/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlmb/2.0)**2
    return 2.0 * 6371.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

def _contains_any(s: str, keys: set[str]) -> bool:
    t = _nfc(s)
    return any(k in t for k in keys)

def _geocode_region_cached_or_fallback(region_name: str) -> Tuple[float, float]:
    key = _nfc(region_name)
    # 1) 메모리 캐시
    if key in _GEOCODE_MEMO:
        return _GEOCODE_MEMO[key]
    # 2) 파일 캐시
    fc = _load_geocode_cache()
    if key in fc:
        _GEOCODE_MEMO[key] = tuple(fc[key])
        return _GEOCODE_MEMO[key]
    # 3) 네트워크(짧은 타임아웃)
    if KAKAO_API_KEY:
        try:
            url = "https://dapi.kakao.com/v2/local/search/keyword.json"
            headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
            params = {"query": key}
            resp = requests.get(url, headers=headers, params=params, timeout=1.8)
            if resp.ok:
                docs = resp.json().get("documents", [])
                if docs:
                    lat, lon = float(docs[0]["y"]), float(docs[0]["x"])
                    _GEOCODE_MEMO[key] = (lat, lon)
                    fc[key] = [lat, lon]
                    _save_geocode_cache(fc)
                    return lat, lon
        except requests.exceptions.RequestException:
            pass
    # 4) 오프라인 폴백: CSV에서 주소 부분일치 평균 좌표 → 최후에는 전체 평균
    df = _load_tmf_cached().copy()
    col_addr = _first_contains(df.columns, "addr1") or "addr1"
    col_lat = _first_contains(df.columns, "mapy") or _first_contains(df.columns, "lat") or "lat"
    col_lon = _first_contains(df.columns, "mapx") or _first_contains(df.columns, "lon") or "lon"
    try:
        mask = df[col_addr].astype(str).str.contains(key, na=False)
        sub = df.loc[mask]
        if len(sub) >= 3:
            lat, lon = float(pd.to_numeric(sub[col_lat], errors="coerce").mean()), float(pd.to_numeric(sub[col_lon], errors="coerce").mean())
        else:
            lat, lon = float(pd.to_numeric(df[col_lat], errors="coerce").mean()), float(pd.to_numeric(df[col_lon], errors="coerce").mean())
    except Exception:
        lat, lon = 37.5665, 126.9780  # 서울 시청 근처(최후 폴백)
    _GEOCODE_MEMO[key] = (lat, lon)
    return lat, lon
