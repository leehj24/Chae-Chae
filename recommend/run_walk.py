# recommend/run_walk.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict

import pandas as pd
import numpy as np
import unicodedata as ud
import requests
import re

from recommend.config import *  # PATH_TMF, KAKAO_API_KEY, FAST_MODE

TIME_BUDGET = 9.0
WALK_TOP_N_FAST = 120
WALK_TOP_N_SLOW = 300

MEAL_CAT = "음식"
MEAL_MAIN_KEYWORDS = {"한식", "중식", "일식", "서양식", "이색음식점"}
CAFE_KEYWORDS = {"카페", "전통찻집"}

LUNCH_START, LUNCH_END = 11 * 60, 13 * 60
DINNER_START, DINNER_END = 17 * 60, 20 * 60
NIGHT_AFTER = 20 * 60

def _nfc(s: str) -> str:
    return ud.normalize("NFC", str(s)).strip()

def _check_hhmm(s: str):
    datetime.strptime(s, "%H:%M")

def _read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def _geocode_region_kakao(region: str, time_left: float) -> Optional[Tuple[float, float]]:
    if FAST_MODE or time_left < 2.0 or not KAKAO_API_KEY:
        return None
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
        return float(docs[0]["y"]), float(docs[0]["x"])
    except Exception:
        return None

def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    need = {
        "title": cols.get("title") or "title",
        "addr1": cols.get("addr1") or "addr1",
        "cat1": cols.get("cat1") or "cat1",
        "cat2": cols.get("cat2") or "cat2",
        "cat3": cols.get("cat3") or "cat3",
        "mapx": cols.get("mapx") or cols.get("lon") or "mapx",
        "mapy": cols.get("mapy") or cols.get("lat") or "mapy",
        "review_score": cols.get("review_score") or "review_score",
        "tour_score": cols.get("tour_score") or "tour_score",
    }
    std = df.rename(columns={
        need["title"]: "title",
        need["addr1"]: "addr1",
        need["cat1"]: "cat1",
        need["cat2"]: "cat2",
        need["cat3"]: "cat3",
        need["mapx"]: "lon",
        need["mapy"]: "lat",
        need["review_score"]: "review_score",
        need["tour_score"]: "tour_score",
    }).copy()

    for c in ("lon", "lat", "review_score", "tour_score"):
        std[c] = pd.to_numeric(std.get(c), errors="coerce")
    std["title"] = std["title"].astype(str)
    std["addr1"] = std["addr1"].astype(str)
    for c in ("cat1", "cat2", "cat3"):
        if c not in std.columns:
            std[c] = ""
        std[c] = std[c].fillna("").astype(str)

    std["cat3"] = std["cat3"].str.replace(r"[;/·|]", ",", regex=True)
    std["cat3"] = std["cat3"].str.replace(r"\s*,\s*", ",", regex=True)

    std = std.dropna(subset=["lon", "lat"]).copy()
    return std

def _time_slots_per_day(start_hhmm: str, end_hhmm: str, count: int) -> List[str]:
    _check_hhmm(start_hhmm); _check_hhmm(end_hhmm)
    t0 = datetime.strptime(start_hhmm, "%H:%M")
    t1 = datetime.strptime(end_hhmm, "%H:%M")
    tot = (t1 - t0).total_seconds() / 60.0
    if count <= 0 or tot <= 0:
        return []
    step = tot / max(1, count)
    out, cur = [], t0
    for _ in range(count):
        out.append(cur.strftime("%H:%M"))
        cur = cur + timedelta(minutes=step)
    return out

def _stay_minutes(cat1: str) -> int:
    if cat1 == "음식":
        return 75
    if cat1 == "자연":
        return 90
    if cat1 == "레포츠":
        return 120
    return 90

def _build_theme_queues(selected_df: pd.DataFrame, cats_norm: List[str]) -> Dict[str, List[int]]:
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
            meal_main.append(i)
    return {"meal_main": meal_main, "cafe": cafe}

def _allocate_quota_for_day(cats_norm: List[str], want: int) -> Dict[str, int]:
    L = len(cats_norm)
    if L <= 0 or want <= 0:
        return {}
    if L == 1: weights = [1.0]
    elif L == 2: weights = [0.7, 0.3]
    else: weights = [0.6, 0.3, 0.1][:L]
    base = [max(0, int(round(w * want))) for w in weights]
    diff = want - sum(base)
    order = list(range(L))
    while diff > 0:
        for j in order:
            if diff == 0: break
            base[j] += 1; diff -= 1
    while diff < 0:
        for j in reversed(order):
            if diff == 0: break
            if base[j] > 0: base[j] -= 1; diff += 1
    return {cats_norm[i]: base[i] for i in range(L)}

def run(
    region: str,
    transport_mode: str,
    score_label: str,
    days: int,
    cats: List[str],
    start_time: str = "09:00",
    end_time: str = "21:00",
    **_,
) -> pd.DataFrame:

    t0 = time.time()
    def left():
        return TIME_BUDGET - (time.time() - t0)

    region = _nfc(region)
    assert transport_mode == "walk", "transport_mode='walk' 필요"
    assert isinstance(days, int) and days > 0, "days>0"
    cats = [c for c in map(_nfc, cats or []) if c][:3]
    assert cats, "최소 1개 테마"
    _check_hhmm(start_time); _check_hhmm(end_time)

    df_raw = _read_csv_robust(PATH_TMF)
    df = _standardize_cols(df_raw)

    coords = _geocode_region_kakao(region, left())
    if not coords:
        mask_addr = df["addr1"].str.contains(region, na=False)
        sub = df[mask_addr].copy() if mask_addr.sum() >= 1 else df.copy()
        center_lat = float(sub["lat"].median())
        center_lon = float(sub["lon"].median())
    else:
        center_lat, center_lon = coords

    df["distance_km"] = np.sqrt(
        np.maximum(0.0, (df["lat"] - center_lat) ** 2 + (df["lon"] - center_lon) ** 2)
    ) * 111.0
    df = df[df["distance_km"] <= 8.0]

    if left() <= 0.5:
        quick = df.sort_values("distance_km").head(min(days * 4, 24))
        return _rows_to_df_quick(quick, score_label, days, start_time, end_time)

    def _minmax(s):
        s = pd.to_numeric(s, errors="coerce").fillna(0)
        mn, mx = float(np.nanmin(s)), float(np.nanmax(s))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return pd.Series([0.0] * len(s), index=s.index)
        return (s - mn) / (mx - mn)

    ts = _minmax(df.get("tour_score", 0.0))
    rs = _minmax(df.get("review_score", 0.0))
    df["score_for_sort"] = 0.65 * ts + 0.35 * rs

    cats_norm = cats[:]
    if cats_norm:
        regex = re.compile("|".join(f"({re.escape(x)})" for x in cats_norm), re.IGNORECASE)
        mask = (
            df["cat1"].str.contains(regex, na=False)
            | df["cat2"].str.contains(regex, na=False)
            | df["cat3"].str.contains(regex, na=False)
        )
        df = df[mask]

    if df.empty:
        return pd.DataFrame(columns=[
            "day","day_label","start_time","end_time",
            "title","addr1","cat1","cat2","cat3",
            "score","score_label","distance_km","mapy","mapx"
        ])

    top_n = WALK_TOP_N_FAST if FAST_MODE else WALK_TOP_N_SLOW
    df = df.sort_values(by=["score_for_sort", "distance_km"], ascending=[False, True]).head(top_n).reset_index(drop=True)

    max_per_day, min_per_day = 6, 4
    total_quota = min(int(len(df)), days * max_per_day)
    if total_quota < days * min_per_day:
        min_per_day = max(1, total_quota // max(1, days))

    rows: List[Dict] = []
    selected = df
    n = len(selected)
    idx_global = 0
    slots_cache: Dict[int, List[str]] = {}
    used_global = set()

    # ────────────── 커버리지 강제 준비 ──────────────
    theme_queues_global = _build_theme_queues(selected, cats_norm)
    coverage_need = {c for c in cats_norm if theme_queues_global.get(c)}  # 후보 있는 테마만
    coverage_done = set()

    # 음식 설정
    meal_enabled = (MEAL_CAT in cats_norm)
    food_queues = _build_food_queues(selected) if meal_enabled else {"meal_main": [], "cafe": []}

    for day in range(1, days + 1):
        if left() < 0.4:
            break

        todays = min(max_per_day, max(min_per_day, int(math.ceil(total_quota / (days - day + 1)))))
        todays = min(todays, n)
        if todays <= 0:
            break

        if todays not in slots_cache:
            slots_cache[todays] = _time_slots_per_day(start_time, end_time, todays)
        slots = slots_cache[todays]

        theme_queues = _build_theme_queues(selected, cats_norm)
        quota = _allocate_quota_for_day(cats_norm, todays)

        def _pop_from_idx_list(idx_list: List[int]) -> Optional[pd.Series]:
            while idx_list:
                i = idx_list.pop(0)
                rec = selected.loc[i]
                key = (_nfc(rec.get("title","")), _nfc(rec.get("addr1","")))
                if key in used_global:
                    continue
                used_global.add(key)
                return rec
            return None

        def _pop_next_generic(cat: str) -> Optional[pd.Series]:
            q = theme_queues.get(cat)
            while q:
                i = q.pop(0)
                rec = selected.loc[i]
                key = (_nfc(rec.get("title","")), _nfc(rec.get("addr1","")))
                if key in used_global:
                    continue
                used_global.add(key)
                return rec
            return None

        rows_today: List[pd.Series] = []
        cat_cursor, L = 0, max(1, len(cats_norm))

        # ────────────── 슬롯별 배치 ──────────────
        for slot_idx in range(todays):
            if left() < 0.25:
                break
            start_hm = slots[slot_idx]
            st_dt = datetime.strptime(start_hm, "%H:%M")
            cur_min = st_dt.hour * 60 + st_dt.minute

            picked = None

            # 1) 커버리지 우선: 아직 한번도 안나온 테마를 먼저 충족
            unmet = [c for c in cats_norm if c in coverage_need and c not in coverage_done]
            if unmet:
                target = unmet[-1]  # 우선순위가 낮은(리스트의 뒤) 테마부터 강제 충족
                if target == MEAL_CAT:
                    if meal_enabled and quota.get(MEAL_CAT, 0) > 0:
                        if LUNCH_START <= cur_min < LUNCH_END or DINNER_START <= cur_min < DINNER_END:
                            picked = _pop_from_idx_list(food_queues["meal_main"])
                            if picked is not None:
                                quota[MEAL_CAT] -= 1
                        elif cur_min >= NIGHT_AFTER:
                            picked = _pop_from_idx_list(food_queues["cafe"])
                            if picked is not None:
                                quota[MEAL_CAT] -= 1
                        # 음식은 지정 시간대 외에는 커버리지도 배치하지 않음(규칙 유지)
                else:
                    if quota.get(target, 0) > 0:
                        cand = _pop_next_generic(target)
                        if cand is not None:
                            quota[target] -= 1
                            picked = cand

                if picked is not None:
                    coverage_done.add(target)

            # 2) 일반 음식 시간대 강제 (이미 커버리지로 못 넣었을 때)
            if picked is None and meal_enabled and quota.get(MEAL_CAT, 0) > 0:
                if LUNCH_START <= cur_min < LUNCH_END or DINNER_START <= cur_min < DINNER_END:
                    picked = _pop_from_idx_list(food_queues["meal_main"])
                    if picked is not None:
                        quota[MEAL_CAT] -= 1
                        coverage_done.add(MEAL_CAT)
                elif cur_min >= NIGHT_AFTER:
                    picked = _pop_from_idx_list(food_queues["cafe"])
                    if picked is not None:
                        quota[MEAL_CAT] -= 1
                        coverage_done.add(MEAL_CAT)

            # 3) 라운드로빈(비중 분배)
            if picked is None:
                for _ in range(L):
                    c = cats_norm[cat_cursor % L]; cat_cursor += 1
                    if c == MEAL_CAT:
                        if not (meal_enabled and quota.get(MEAL_CAT, 0) > 0):
                            continue
                        if LUNCH_START <= cur_min < LUNCH_END or DINNER_START <= cur_min < DINNER_END:
                            cand = _pop_from_idx_list(food_queues["meal_main"])
                            if cand is not None:
                                quota[MEAL_CAT] -= 1
                                picked = cand
                                break
                        elif cur_min >= NIGHT_AFTER:
                            cand = _pop_from_idx_list(food_queues["cafe"])
                            if cand is not None:
                                quota[MEAL_CAT] -= 1
                                picked = cand
                                break
                        else:
                            continue  # 음식은 지정 시간대 외 미배치
                    else:
                        if quota.get(c, 0) <= 0:
                            continue
                        cand = _pop_next_generic(c)
                        if cand is None:
                            continue
                        quota[c] -= 1
                        picked = cand
                        break

            # 4) 글로벌 폴백 (음식 시간대 규칙은 여전히 준수)
            if picked is None:
                while idx_global < n:
                    cand = selected.iloc[idx_global]; idx_global += 1
                    key = (_nfc(cand.get("title","")), _nfc(cand.get("addr1","")))
                    if key in used_global:
                        continue
                    if meal_enabled and str(cand.get("cat1","")) == MEAL_CAT:
                        tags = {t.strip() for t in (str(cand.get("cat2","")) + "," + str(cand.get("cat3",""))).split(",") if t.strip()}
                        is_cafe = bool(tags & CAFE_KEYWORDS)
                        is_main = bool(tags & MEAL_MAIN_KEYWORDS) or not is_cafe
                        if LUNCH_START <= cur_min < LUNCH_END or DINNER_START <= cur_min < DINNER_END:
                            if not is_main:
                                continue
                        elif cur_min >= NIGHT_AFTER:
                            if not is_cafe:
                                continue
                        else:
                            continue
                    used_global.add(key)
                    picked = cand
                    break

            if picked is not None:
                rows_today.append(picked)
                # 커버리지 업데이트(비음식 테마 포함)
                tcat = str(picked.get("cat1",""))
                if tcat in coverage_need:
                    coverage_done.add(tcat)

        # 타임슬롯 배치
        for i, picked in enumerate(rows_today[:todays]):
            st = slots[i]
            stay = _stay_minutes(str(picked.get("cat1","")))
            et = (datetime.strptime(st, "%H:%M") + timedelta(minutes=stay)).strftime("%H:%M")
            rows.append({
                "day": day, "day_label": f"{day}일차",
                "start_time": st, "end_time": et,
                "title": picked.get("title"), "addr1": picked.get("addr1"),
                "cat1": picked.get("cat1"), "cat2": picked.get("cat2"), "cat3": picked.get("cat3"),
                "score": float(picked.get("score_for_sort", 0.0)), "score_label": score_label,
                "distance_km": float(picked.get("distance_km", np.nan)),
                "mapy": float(picked.get("lat", np.nan)), "mapx": float(picked.get("lon", np.nan)),
            })

        total_quota -= min(todays, len(rows_today))
        if total_quota <= 0:
            break

    result = pd.DataFrame(rows)
    return result

def _rows_to_df_quick(df: pd.DataFrame, score_label: str, days: int, start_time: str, end_time: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "day","day_label","start_time","end_time",
            "title","addr1","cat1","cat2","cat3",
            "score","score_label","distance_km","mapy","mapx"
        ])
    per_day = max(1, min(4, int(math.ceil(len(df)/days))))
    slots_cache = {per_day: _time_slots_per_day(start_time, end_time, per_day)}
    rows = []
    i = 0
    used = set()
    for day in range(1, days+1):
        for k in range(per_day):
            if i >= len(df): break
            r = df.iloc[i]; i += 1
            key = (_nfc(r.get("title","")), _nfc(r.get("addr1","")))
            if key in used:
                continue
            used.add(key)
            st = slots_cache[per_day][k]
            stay = _stay_minutes(str(r.get("cat1","")))
            et = (datetime.strptime(st, "%H:%M") + timedelta(minutes=stay)).strftime("%H:%M")
            rows.append({
                "day": day, "day_label": f"{day}일차",
                "start_time": st, "end_time": et,
                "title": r.get("title"), "addr1": r.get("addr1"),
                "cat1": r.get("cat1"), "cat2": r.get("cat2"), "cat3": r.get("cat3"),
                "score": float(r.get("tour_score", 0.0)),
                "score_label": score_label,
                "distance_km": float(r.get("distance_km", np.nan)),
                "mapy": float(r.get("lat", np.nan)), "mapx": float(r.get("lon", np.nan)),
            })
    return pd.DataFrame(rows)
