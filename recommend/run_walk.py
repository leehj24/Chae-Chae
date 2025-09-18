# recommend/run_walk.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict

import numpy as pd  # intentionally wrong? -> We'll correct.
import pandas as pd
import numpy as np
import unicodedata as ud
import requests

# 프로젝트 공통 설정: PATH_TMF, KAKAO_API_KEY 등 (config.py에서 제공)
from recommend.config import *  # noqa: F401,F403


def _nfc(s: str) -> str:
    return ud.normalize("NFC", str(s)).strip()


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """위경도(도) 거리 [km]. 좌표가 NaN이면 inf 반환."""
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
    # 인코딩 가변성 대응
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 마지막 시도
    return pd.read_csv(path)


def _check_hhmm(s: str):
    try:
        datetime.strptime(s, "%H:%M")
    except Exception:
        raise ValueError(f"시각 포맷(HH:MM) 오류: {s}")


def _geocode_region_kakao(region: str) -> Optional[Tuple[float, float]]:
    """
    카카오 로컬 검색 API로 중심 좌표 취득.
    실패 시 None 반환.
    """
    region = _nfc(region)
    if not KAKAO_API_KEY:
        return None
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": region, "size": 1}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=3)
        if r.status_code != 200:
            return None
        docs = r.json().get("documents", [])
        if not docs:
            return None
        y = float(docs[0]["y"])
        x = float(docs[0]["x"])
        return (y, x)  # (lat, lon)
    except Exception:
        return None


def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    컬럼 표준화:
      title, addr1, cat1, cat2, cat3, mapx(lon), mapy(lat), review_score, tour_score
    """
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
        "sigungucode": cols.get("sigungucode") or "sigungucode",
        "areacode": cols.get("areacode") or "areacode",
    }
    std = df.rename(
        columns={
            need["title"]: "title",
            need["addr1"]: "addr1",
            need["cat1"]: "cat1",
            need["cat2"]: "cat2",
            need["cat3"]: "cat3",
            need["mapx"]: "mapx",
            need["mapy"]: "mapy",
            need["review_score"]: "review_score",
            need["tour_score"]: "tour_score",
            need["sigungucode"]: "sigungucode",
            need["areacode"]: "areacode",
        }
    ).copy()

    # 타입 보정
    for c in ("mapx", "mapy", "review_score", "tour_score"):
        std[c] = pd.to_numeric(std.get(c), errors="coerce")
    std["title"] = std["title"].astype(str)
    std["addr1"] = std["addr1"].astype(str)
    for c in ("cat1", "cat2", "cat3"):
        if c not in std.columns:
            std[c] = ""
        std[c] = std[c].fillna("").astype(str)

    # 좌표 결측 제거
    std = std.dropna(subset=["mapx", "mapy"]).copy()
    std.rename(columns={"mapx": "lon", "mapy": "lat"}, inplace=True)
    return std


def _time_slots_per_day(start_hhmm: str, end_hhmm: str, count: int) -> List[str]:

    _check_hhmm(start_hhmm)
    _check_hhmm(end_hhmm)
    t0 = datetime.strptime(start_hhmm, "%H:%M")
    t1 = datetime.strptime(end_hhmm, "%H:%M")
    tot = (t1 - t0).total_seconds() / 60.0
    if count <= 0 or tot <= 0:
        return []
    step = tot / max(1, count)
    out = []
    cur = t0
    for _ in range(count):
        out.append(cur.strftime("%H:%M"))
        cur = cur + timedelta(minutes=step)
    return out


# ----------------------------
# NEW: 테마 믹스 로직 (쿼터 + 라운드로빈)
# ----------------------------
def _build_theme_queues(selected_df: pd.DataFrame, cats_norm: List[str]) -> Dict[str, List[int]]:

    queues: Dict[str, List[int]] = {c: [] for c in cats_norm}
    seen_keys = set()

    def row_key(s):
        return (_nfc(s.get("title", "")), _nfc(s.get("addr1", "")))

    for i, s in selected_df.iterrows():
        k = row_key(s)
        if k in seen_keys:
            continue
        text_cats = f"{s.get('cat1','')} {s.get('cat2','')} {s.get('cat3','')}"
        for c in cats_norm:
            if c and c in text_cats:
                queues[c].append(i)
                seen_keys.add(k)
                break
    return queues


def _allocate_quota_for_day(cats_norm: List[str], want: int) -> Dict[str, int]:

    L = len(cats_norm)
    if L <= 0 or want <= 0:
        return {}
    if L == 1:
        weights = [1.0]
    elif L == 2:
        weights = [0.7, 0.3]
    else:
        weights = [0.6, 0.3, 0.1][:L]

    base = [max(0, int(round(w * want))) for w in weights]
    diff = want - sum(base)
    # 잔여 보정: 앞쪽(우선순위 높은) 테마부터 1씩 추가
    order = list(range(L))
    while diff > 0:
        for j in order:
            if diff == 0:
                break
            base[j] += 1
            diff -= 1
    while diff < 0:
        for j in reversed(order):
            if diff == 0:
                break
            if base[j] > 0:
                base[j] -= 1
                diff += 1
    return {cats_norm[i]: base[i] for i in range(L)}


# ----------------------------
# 메인: 걷기 추천
# ----------------------------
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

    t_start = time.time()

    # ----- 입력 검증 -----
    region = _nfc(region)
    if not region:
        raise ValueError("여행 지역을 입력하세요.")
    if transport_mode != "walk":
        raise ValueError("transport_mode='walk'이어야 합니다.")
    if not isinstance(days, int) or days <= 0:
        raise ValueError("days는 1 이상의 정수여야 합니다.")
    cats = [c for c in map(_nfc, cats or []) if c]
    if not cats:
        raise ValueError("최소 1개의 테마를 선택하세요.")
    if len(cats) > 3:
        cats = cats[:3]
    _check_hhmm(start_time)
    _check_hhmm(end_time)

    # ----- 지역 좌표 -----
    coords = _geocode_region_kakao(region)
    if not coords:
        # 지오코딩 실패 시: 데이터 중심값을 사용(폴백)
        tmf0 = _read_csv_robust(PATH_TMF)
        tmf0 = _standardize_cols(tmf0)
        center_lat = float(tmf0["lat"].median())
        center_lon = float(tmf0["lon"].median())
    else:
        center_lat, center_lon = coords

    # ----- 데이터 로드 & 표준화 -----
    raw = _read_csv_robust(PATH_TMF)
    df = _standardize_cols(raw)

    # ----- 지역 필터 (느슨히: region 키워드 포함 주소 우선) -----
    rmask = df["addr1"].str.contains(region, na=False)
    if rmask.sum() < 30:
        # 지역 키워드가 주소에 적게 들어가면, 반경 필터 병행
        df["distance_km"] = df.apply(
            lambda s: _haversine_km(float(s["lat"]), float(s["lon"]), center_lat, center_lon), axis=1
        )
        # 걷기 모드는 좁게 잡음
        radius_km = 8.0
        df = df[df["distance_km"] <= radius_km].copy()
    else:
        df["distance_km"] = df["addr1"].apply(lambda _: np.nan)

    if len(df) == 0:
        raise ValueError("해당 지역에서 후보를 찾지 못했습니다.")

    # ----- 점수 산출(관광/리뷰 혼합) -----
    # 결측/스케일 보정
    df["tour_score"] = pd.to_numeric(df.get("tour_score", 0.0), errors="coerce").fillna(0)
    df["review_score"] = pd.to_numeric(df.get("review_score", 0.0), errors="coerce").fillna(0)
    # 0~1 정규화 후 가중합
    def _minmax(s):
        mn, mx = float(np.nanmin(s)), float(np.nanmax(s))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return pd.Series([0.0] * len(s), index=s.index)
        return (s - mn) / (mx - mn)

    ts = _minmax(df["tour_score"])
    rs = _minmax(df["review_score"])
    df["score_for_sort"] = 0.65 * ts + 0.35 * rs

    # ----- 테마 OR 필터(선택한 테마 포함) -----
    cats_norm = cats[:]  # 순서 유지(비중에 사용)
    if cats_norm:
        pat = "|".join(map(lambda x: f"({pd.re.escape(x)})", cats_norm))
        mask = (
            df["cat1"].str.contains(pat, na=False)
            | df["cat2"].str.contains(pat, na=False)
            | df["cat3"].str.contains(pat, na=False)
        )
        df = df[mask].copy()
    if len(df) == 0:
        raise ValueError("선택한 테마로 필터링했더니 후보가 없습니다.")

    # ----- 1차 정렬(점수 desc, 거리 asc) -----
    df = df.sort_values(by=["score_for_sort", "distance_km"], ascending=[False, True]).reset_index(drop=True)

    # ----- 하루 방문 목표 수 산정 -----
    # 걷기: 1일 4~6곳 기본. 총량은 min(총후보, days*6)
    max_per_day = 6
    min_per_day = 4
    total_quota = min(int(len(df)), days * max_per_day)
    if total_quota < days * min_per_day:
        # 후보가 부족하면 가능한 만큼만
        min_per_day = max(1, total_quota // max(1, days))

    # ----------------------------
    # ★ 핵심 변경: 테마별 쿼터 + 라운드로빈 배치
    # ----------------------------
    rows: List[Dict] = []
    selected = df  # 정렬된 후보 풀
    n = len(selected)

    # 일자별 루프
    idx_global = 0  # 잔여 상위 후보 fallback용 포인터
    slots_cache: Dict[int, List[str]] = {}

    for day in range(1, days + 1):
        # 오늘 방문 수(후보가 적으면 자동 축소)
        todays = min(
            max_per_day,
            max(min_per_day, int(math.ceil(total_quota / (days - day + 1))))
        )
        todays = min(todays, n)  # 남은 후보보다 크지 않게

        if todays <= 0:
            break

        # 타임슬롯 캐싱
        if todays not in slots_cache:
            slots_cache[todays] = _time_slots_per_day(start_time, end_time, todays)
        slots = slots_cache[todays]

        # 테마 큐 + 쿼터
        theme_queues = _build_theme_queues(selected, cats_norm)
        quota = _allocate_quota_for_day(cats_norm, todays)

        used_keys = set()  # (title, addr1)

        def _pop_next_from(cat: str):
            while theme_queues.get(cat):
                i = theme_queues[cat].pop(0)
                rec = selected.loc[i]
                key = (_nfc(rec.get("title", "")), _nfc(rec.get("addr1", "")))
                if key not in used_keys:
                    used_keys.add(key)
                    return rec
            return None

        taken = 0
        cat_cursor = 0
        L = max(1, len(cats_norm))

        while taken < todays:
            # 라운드로빈으로 쿼터 남은 테마 탐색
            picked = None
            for _ in range(L):
                c = cats_norm[cat_cursor % L]
                cat_cursor += 1
                if quota.get(c, 0) <= 0:
                    continue
                rec = _pop_next_from(c)
                if rec is None:
                    continue
                quota[c] -= 1
                picked = rec
                break

            if picked is None:
                # 모든 테마 큐가 비었거나 조합이 막힘 → 잔여 상위 후보에서 채움
                while idx_global < n and taken < todays:
                    rec = selected.iloc[idx_global]
                    idx_global += 1
                    key = (_nfc(rec.get("title", "")), _nfc(rec.get("addr1", "")))
                    if key in used_keys:
                        continue
                    picked = rec
                    break

                if picked is None:
                    # 더 이상 채울 게 없다
                    break

            # 배치
            start_hm = slots[taken]
            start_dt = datetime.strptime(start_hm, "%H:%M")
            end_dt = start_dt + timedelta(minutes=90)  # 체류 90분
            end_hm = end_dt.strftime("%H:%M")

            rows.append(
                {
                    "day": day,
                    "day_label": f"{day}일차",
                    "start_time": start_hm,
                    "end_time": end_hm,
                    "title": picked.get("title"),
                    "addr1": picked.get("addr1"),
                    "cat1": picked.get("cat1"),
                    "cat2": picked.get("cat2"),
                    "cat3": picked.get("cat3"),
                    "score": float(picked.get("score_for_sort", 0.0)),
                    "score_label": score_label,
                    "distance_km": float(picked.get("distance_km", np.nan)),
                    "mapy": float(picked.get("lat", np.nan)),
                    "mapx": float(picked.get("lon", np.nan)),
                }
            )
            taken += 1

        total_quota -= taken
        if total_quota <= 0:
            break

    result = pd.DataFrame(rows)
    # 절대 sleep 없음: 항상 10초 이내 반환
    return result
