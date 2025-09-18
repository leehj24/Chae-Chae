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

# 프로젝트 공통 설정: PATH_TMF, KAKAO_API_KEY, FAST_MODE 등
from recommend.config import *  # noqa: F401,F403

# ─────────────────────────────────────
# SLA: 항상 10초 이내 반환을 보장하기 위한 예산
# ─────────────────────────────────────
TIME_BUDGET = 9.0  # 초(앱 오버헤드 감안, 여유 1초 남김)
WALK_TOP_N_FAST = 120
WALK_TOP_N_SLOW = 300

# ─────────────────────────────────────
# 유틸
# ─────────────────────────────────────
def _nfc(s: str) -> str:
    return ud.normalize("NFC", str(s)).strip()

def _check_hhmm(s: str):
    datetime.strptime(s, "%H:%M")

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    vals = [lat1, lon1, lat2, lon2]
    if any(pd.isna(v) for v in vals):
        return float("inf")
    R = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def _read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def _geocode_region_kakao(region: str, time_left: float) -> Optional[Tuple[float, float]]:
    """FAST_MODE면 네트워크 호출 금지. 남은 시간이 2.0s 미만이면 스킵."""
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
        return float(docs[0]["y"]), float(docs[0]["x"])  # lat, lon
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

# ─────────────────────────────────────
# 테마 믹스(쿼터 + 라운드로빈)
# ─────────────────────────────────────
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

# ─────────────────────────────────────
# 메인
# ─────────────────────────────────────
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

    # 입력 검증
    region = _nfc(region)
    assert transport_mode == "walk", "transport_mode='walk' 필요"
    assert isinstance(days, int) and days > 0, "days>0"
    cats = [c for c in map(_nfc, cats or []) if c][:3]
    assert cats, "최소 1개 테마"
    _check_hhmm(start_time); _check_hhmm(end_time)

    # 데이터 로드/표준화
    df_raw = _read_csv_robust(PATH_TMF)  # 경로는 config에서 관리 :contentReference[oaicite:1]{index=1}
    df = _standardize_cols(df_raw)

    # 지역 중심좌표
    coords = _geocode_region_kakao(region, left())
    if not coords:
        # 주소 포함 여부로 1차 필터
        mask_addr = df["addr1"].str.contains(region, na=False)
        if mask_addr.sum() >= 1:
            sub = df[mask_addr].copy()
        else:
            sub = df.copy()
        center_lat = float(sub["lat"].median())
        center_lon = float(sub["lon"].median())
    else:
        center_lat, center_lon = coords

    # 반경/거리 계산(벡터)
    df["distance_km"] = np.sqrt(
        np.maximum(
            0.0,
            ((df["lat"] - center_lat) ** 2 + (df["lon"] - center_lon) ** 2)
        )
    ) * 111.0  # 대략적 환산(하버사인보다 빠름)
    # 걷기 반경
    radius_km = 8.0
    df = df[df["distance_km"] <= radius_km]

    if left() <= 0.5:
        # 시간 초과 임박 → 상위 몇 개만 골라 즉시 반환
        quick = df.sort_values("distance_km").head(min(days * 4, 24))
        return _rows_to_df_quick(quick, score_label, days, start_time, end_time)

    # 점수(정규화 + 가중합)
    def _minmax(s):
        s = pd.to_numeric(s, errors="coerce").fillna(0)
        mn, mx = float(np.nanmin(s)), float(np.nanmax(s))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return pd.Series([0.0] * len(s), index=s.index)
        return (s - mn) / (mx - mn)

    ts = _minmax(df.get("tour_score", 0.0))
    rs = _minmax(df.get("review_score", 0.0))
    df["score_for_sort"] = 0.65 * ts + 0.35 * rs

    # 테마 OR 필터
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

    # 1차 정렬(점수 desc, 거리 asc) + 상한 적용
    top_n = WALK_TOP_N_FAST if FAST_MODE else WALK_TOP_N_SLOW  # :contentReference[oaicite:2]{index=2}
    df = df.sort_values(by=["score_for_sort", "distance_km"], ascending=[False, True]).head(top_n).reset_index(drop=True)

    # 하루 목표 개수 산정
    max_per_day, min_per_day = 6, 4
    total_quota = min(int(len(df)), days * max_per_day)
    if total_quota < days * min_per_day:
        min_per_day = max(1, total_quota // max(1, days))

    rows: List[Dict] = []
    selected = df
    n = len(selected)
    idx_global = 0
    slots_cache: Dict[int, List[str]] = {}

    for day in range(1, days + 1):
        if left() < 0.4:  # 시간 임계 → 현재까지 결과로 종료
            break

        todays = min(max_per_day, max(min_per_day, int(math.ceil(total_quota / (days - day + 1)))))
        todays = min(todays, n)
        if todays <= 0: break

        if todays not in slots_cache:
            slots_cache[todays] = _time_slots_per_day(start_time, end_time, todays)
        slots = slots_cache[todays]

        # 테마 큐 + 쿼터
        theme_queues = _build_theme_queues(selected, cats_norm)
        quota = _allocate_quota_for_day(cats_norm, todays)

        used_keys = set()
        def _pop_next_from(cat: str):
            q = theme_queues.get(cat)
            while q:
                i = q.pop(0)
                rec = selected.loc[i]
                key = (_nfc(rec.get("title","")), _nfc(rec.get("addr1","")))
                if key not in used_keys:
                    used_keys.add(key); return rec
            return None

        taken, cat_cursor, L = 0, 0, max(1, len(cats_norm))
        while taken < todays:
            if left() < 0.25:
                break  # 남은 슬롯은 폴백으로 채우거나 루프 종료

            picked = None
            for _ in range(L):
                c = cats_norm[cat_cursor % L]; cat_cursor += 1
                if quota.get(c, 0) <= 0: continue
                rec = _pop_next_from(c)
                if rec is None: continue
                quota[c] -= 1; picked = rec; break

            if picked is None:
                while idx_global < n and taken < todays:
                    rec = selected.iloc[idx_global]; idx_global += 1
                    key = (_nfc(rec.get("title","")), _nfc(rec.get("addr1","")))
                    if key in used_keys: continue
                    picked = rec; break
                if picked is None: break

            start_hm = slots[taken]
            end_hm = (datetime.strptime(start_hm, "%H:%M") + timedelta(minutes=90)).strftime("%H:%M")
            rows.append({
                "day": day, "day_label": f"{day}일차",
                "start_time": start_hm, "end_time": end_hm,
                "title": picked.get("title"), "addr1": picked.get("addr1"),
                "cat1": picked.get("cat1"), "cat2": picked.get("cat2"), "cat3": picked.get("cat3"),
                "score": float(picked.get("score_for_sort", 0.0)), "score_label": score_label,
                "distance_km": float(picked.get("distance_km", np.nan)),
                "mapy": float(picked.get("lat", np.nan)), "mapx": float(picked.get("lon", np.nan)),
            })
            taken += 1

        total_quota -= taken
        if total_quota <= 0: break

    result = pd.DataFrame(rows)
    return result

# ─────────────────────────────────────
# 긴급 폴백(시간 임박시 즉시 반환용)
# ─────────────────────────────────────
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
    for day in range(1, days+1):
        for k in range(per_day):
            if i >= len(df): break
            r = df.iloc[i]; i += 1
            st = slots_cache[per_day][k]
            et = (datetime.strptime(st, "%H:%M") + timedelta(minutes=90)).strftime("%H:%M")
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
