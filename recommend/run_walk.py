# recommend/run_walk.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import math
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import unicodedata as ud
import requests

# 프로젝트 공통 설정: PATH_TMF, KAKAO_API_KEY 등 (config.py에서 제공)
from recommend.config import *  # noqa: F401,F403


# ---------------------------
# 유틸
# ---------------------------
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
    p = math.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (math.sin(dlat / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def _read_csv_robust(path: str) -> pd.DataFrame:
    encs = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last = None
    for enc in encs:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last = e
    raise RuntimeError(f"CSV 읽기 실패: {path} / {last}")

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
    }
    out = df.copy()
    for _, v in need.items():
        if v not in out.columns:
            out[v] = np.nan

    out["mapx"] = pd.to_numeric(out["mapx"], errors="coerce")
    out["mapy"] = pd.to_numeric(out["mapy"], errors="coerce")
    out["review_score"] = pd.to_numeric(out["review_score"], errors="coerce")
    out["tour_score"] = pd.to_numeric(out["tour_score"], errors="coerce")

    for c in ["cat1", "cat2", "cat3", "addr1", "title"]:
        if c in out.columns:
            out[c] = out[c].astype(str)

    return out

def _geocode_region_kakao(region: str, timeout_s: float) -> Optional[Tuple[float, float]]:
    """카카오 REST 지오코딩. 실패/타임아웃 시 None."""
    key = globals().get("KAKAO_API_KEY") or ""
    if not key or timeout_s <= 0.15:
        return None
    try:
        url = "https://dapi.kakao.com/v2/local/search/keyword.json"
        headers = {"Authorization": f"KakaoAK {key.strip()}"}
        params = {"query": region, "size": 1}
        resp = requests.get(url, headers=headers, params=params, timeout=timeout_s)
        if resp.status_code != 200:
            return None
        data = resp.json()
        docs = data.get("documents") or []
        if not docs:
            return None
        y = _safe_float(docs[0].get("y"))
        x = _safe_float(docs[0].get("x"))
        if pd.isna(y) or pd.isna(x):
            return None
        return float(y), float(x)
    except Exception:
        return None

def _fallback_center_by_addr(df: pd.DataFrame, region: str) -> Tuple[float, float]:
    """
    주소 내 region 문자열 매칭 평균 좌표.
    매칭이 없으면 전체 평균, 그래도 NaN이면 서울시청.
    """
    addr_col = "addr1" if "addr1" in df.columns else df.columns[0]
    mask = df[addr_col].astype(str).str.contains(region, na=False)
    sub = df.loc[mask]
    lat_col = "mapy" if "mapy" in df.columns else "lat"
    lon_col = "mapx" if "mapx" in df.columns else "lon"
    if not sub.empty:
        lat = pd.to_numeric(sub[lat_col], errors="coerce").mean()
        lon = pd.to_numeric(sub[lon_col], errors="coerce").mean()
    else:
        lat = pd.to_numeric(df[lat_col], errors="coerce").mean()
        lon = pd.to_numeric(df[lon_col], errors="coerce").mean()
    if pd.isna(lat) or pd.isna(lon):
        lat, lon = 37.5665, 126.9780  # 서울시청
    return float(lat), float(lon)

def _normalize_cats(cats: List[str]) -> List[str]:
    """오타/동의어 포함 정규화. '문화' 입력 시 인문(문화/예술/역사) 추가."""
    if not isinstance(cats, (list, tuple)):
        cats = [cats]
    cats = [_nfc(c) for c in cats if str(c).strip()]
    fixed = []
    for c in cats:
        c2 = c.replace("쇼팡", "쇼핑")
        if c2 == "문화":
            fixed.append("문화")
            fixed.append("인문(문화/예술/역사)")
        else:
            fixed.append(c2)
    seen = set()
    out = []
    for c in fixed:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def _choose_score_col(score_label: str) -> str:
    score_label = _nfc(score_label)
    if score_label == "인기도지수":
        return "review_score"
    elif score_label == "관광지수":
        return "tour_score"
    else:
        return "review_score"

def _time_slots_per_day(start_hm: str, end_hm: str, n: int) -> List[str]:
    def to_min(hm: str) -> int:
        h, m = map(int, hm.split(":"))
        return h * 60 + m

    def to_hm(m: int) -> str:
        h = m // 60
        mm = m % 60
        return f"{h:02d}:{mm:02d}"

    s = to_min(start_hm)
    e = to_min(end_hm)
    if e <= s:
        e = s + 12 * 60  # 최소 12시간 보장
    if n <= 1:
        mid = (s + e) // 2
        return [to_hm(mid)]
    step = max(1, (e - s) // n)
    return [to_hm(int(s + step * (i + 0.5))) for i in range(n)]

def _visits_today(remain: int, days_left: int) -> int:
    """
    6 -> 3 -> 1 단계.
    - 미래 각 날 최소 1곳을 예약(reserve)한 뒤, 오늘 쓸 수 있는 여유(spare)로 결정.
    """
    reserve_for_future = max(0, (days_left - 1) * 1)
    spare = max(0, remain - reserve_for_future)
    if spare >= 6:
        return 6
    elif spare >= 3:
        return 3
    else:
        return 1


# ---------------------------
# 메인 엔트리
# ---------------------------
def run(
    region: str,
    transport_mode: str,
    score_label: str,
    days: int,
    cats: List[str],
    start_time: str = "09:00",
    end_time: str = "21:00",
    *,
    max_run_seconds: float = 10.0,  # 어떤 조건이든 "최대" 10초 이내
) -> pd.DataFrame:
    """
    조건 요약:
      - 어떤 지역/테마/일수 조합이라도 DataFrame 생성
      - 점수 기준: '인기도지수'→review_score, '관광지수'→tour_score (결측 상호보완)
      - 지역 근처 우선: walk 6km→12km→무제한 / transit 15km→30km→무제한
      - 1일 방문 수: 가능하면 6곳, 부족하면 3곳, 더 부족하면 1곳(매일 최소 1곳 보장)
      - 전체 실행시간: 무조건 max_run_seconds(기본 10초) 이하
    """
    t0 = time.time()
    deadline = t0 + float(max_run_seconds)
    def time_left() -> float:
        return deadline - time.time()

    # 0) 입력 정리
    region = _nfc(region)
    if transport_mode not in {"walk", "transit"}:
        transport_mode = "walk"
    days = max(1, int(days))
    cats_norm = _normalize_cats(cats)
    score_col = _choose_score_col(score_label)

    # 1) 데이터 로드/표준화
    tmf = _read_csv_robust(PATH_TMF)
    tmf = _standardize_cols(tmf)

    # 2) 중심 좌표 (카카오 → 실패 시 주소기반 평균 → 서울시청)
    #    타임가드: 남은 시간이 1.0s 미만이면 외부 호출 생략
    geo_timeout = min(0.7, max(0.2, time_left() - 0.3))
    coords = _geocode_region_kakao(region, timeout_s=geo_timeout) if time_left() > 1.0 else None
    if coords is None:
        coords = _fallback_center_by_addr(tmf, region)
    center_lat, center_lon = coords

    # 3) 카테고리 필터
    if cats_norm:
        mask_cat = pd.Series(False, index=tmf.index)
        for c in cats_norm:
            match = tmf["cat1"].astype(str).str.contains(c, na=False)
            if "cat2" in tmf.columns:
                match |= tmf["cat2"].astype(str).str.contains(c, na=False)
            if "cat3" in tmf.columns:
                match |= tmf["cat3"].astype(str).str.contains(c, na=False)
            mask_cat |= match
    else:
        mask_cat = pd.Series(True, index=tmf.index)

    # 4) 거리 계산(타임가드)
    #    시간이 매우 부족하면, 우선 주소 포함 빠른 프리필터 → 이후 최소량만 거리계산
    need_count_for_plan = days * 6 if days * 6 > 0 else 6
    if time_left() < 1.0:
        # 시간 촉박: region 포함만 우선, 없으면 전체
        fast_pool = tmf[tmf["addr1"].astype(str).str.contains(region, na=False)]
        if fast_pool.empty:
            fast_pool = tmf
        tmf = fast_pool

    tmf["distance_km"] = [
        _haversine_km(center_lat, center_lon, _safe_float(y), _safe_float(x))
        for y, x in zip(tmf["mapy"].values, tmf["mapx"].values)
    ]

    # 5) 정렬 점수(요청 점수 우선, 결측 상호 보완)
    tmf["score_for_sort"] = (
        pd.to_numeric(tmf[score_col], errors="coerce")
        .fillna(pd.to_numeric(tmf["review_score"], errors="coerce"))
        .fillna(pd.to_numeric(tmf["tour_score"], errors="coerce"))
        .fillna(0.0)
    )

    # 6) 반경 확장 + 빠른 상위 추림(nlargest)로 정렬 비용 최소화
    base_radius = 6.0 if transport_mode == "walk" else 15.0
    radii = [base_radius, base_radius * 2, float("inf")]

    selected = None
    for r in radii:
        if time_left() <= 0.15:
            break  # 시간 초과 직전이면 즉시 탈출
        if math.isfinite(r):
            cand = tmf[(tmf["distance_km"] <= r) & (mask_cat)]
        else:
            cand = tmf[mask_cat]
        if cand.empty:
            continue

        # 필요 수량의 2~3배만 점수 상위 선추림 → 최종 정렬
        k = min(len(cand), max(need_count_for_plan * 3, 50))
        cand_top = cand.nlargest(k, columns="score_for_sort")
        cand_sorted = cand_top.sort_values(["score_for_sort", "distance_km"], ascending=[False, True])
        selected = cand_sorted

        if len(selected) >= need_count_for_plan:
            break

    if selected is None or selected.empty:
        # 최후: 전체에서 점수 상위만 빠르게
        k = min(len(tmf), max(need_count_for_plan * 3, 50))
        top = tmf.nlargest(k, columns="score_for_sort")
        selected = top.sort_values(["score_for_sort", "distance_km"], ascending=[False, True])

    # 7) 스케줄 배분: 6 -> 3 -> 1 단계, 매일 최소 1곳 보장
    rows: List[dict] = []
    idx = 0
    n = len(selected)
    slots_cache: dict[int, List[str]] = {}

    for day in range(1, days + 1):
        if time_left() <= 0.05 and rows:
            # 시간이 완전히 모자라면 남은 날은 1곳씩 빠르게 채움(가장 가까운 상위)
            for d2 in range(day, days + 1):
                fb = selected.iloc[min(idx, n - 1)] if n > 0 and idx < n else tmf.sort_values(
                    ["distance_km", "score_for_sort"], ascending=[True, False]
                ).head(1).iloc[0]
                st = _time_slots_per_day(start_time, end_time, 1)[0]
                rows.append({
                    "day": d2, "visit_order": 1, "time": st,
                    "title": fb.get("title"), "addr1": fb.get("addr1"),
                    "cat1": fb.get("cat1"), "cat2": fb.get("cat2"), "cat3": fb.get("cat3"),
                    "score": float(fb.get("score_for_sort", 0.0)),
                    "score_label": score_label,
                    "distance_km": float(fb.get("distance_km", np.nan)),
                    "lat": float(fb.get("mapy", np.nan)),
                    "lon": float(fb.get("mapx", np.nan)),
                })
                idx = min(idx + 1, n)
            break

        days_left = days - day + 1
        remain = n - idx
        todays = _visits_today(remain, days_left) if remain > 0 else 1

        if todays not in slots_cache:
            slots_cache[todays] = _time_slots_per_day(start_time, end_time, todays)
        slots = slots_cache[todays]

        taken = 0
        while taken < todays and idx < n:
            if time_left() <= 0.02:
                break
            rec = selected.iloc[idx]; idx += 1
            rows.append({
                "day": day,
                # "visit_order": taken + 1,
                "time": slots[taken],
                "title": rec.get("title"),
                "addr1": rec.get("addr1"),
                "cat1": rec.get("cat1"),
                "cat2": rec.get("cat2"),
                "cat3": rec.get("cat3"),
                "score": float(rec.get("score_for_sort", 0.0)),
                "score_label": score_label,
                "distance_km": float(rec.get("distance_km", np.nan)),
                "lat": float(rec.get("mapy", np.nan)),
                "lon": float(rec.get("mapx", np.nan)),
            })
            taken += 1

        if taken == 0:
            # 후보가 바닥나도/시간 촉박해도 매일 최소 1곳 강제
            fb = selected.iloc[min(idx, n - 1)] if n > 0 and idx < n else tmf.sort_values(
                ["distance_km", "score_for_sort"], ascending=[True, False]
            ).head(1).iloc[0]
            rows.append({
                "day": day,
                "time": _time_slots_per_day(start_time, end_time, 1)[0],
                "title": fb.get("title"),
                "addr1": fb.get("addr1"),
                "cat1": fb.get("cat1"),
                "cat2": fb.get("cat2"),
                "cat3": fb.get("cat3"),
                "score": float(fb.get("score_for_sort", 0.0)),
                "score_label": score_label,
                "distance_km": float(fb.get("distance_km", np.nan)),
                "lat": float(fb.get("mapy", np.nan)),
                "lon": float(fb.get("mapx", np.nan)),
            })
            idx = min(idx + 1, n)

    result = pd.DataFrame(rows)
    # 절대 sleep 없음: 항상 10초 이내 반환
    return result
