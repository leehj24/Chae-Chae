# run_walk.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import re, math, requests
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
import unicodedata as ud

# ⚠️ PATH_TMF, KAKAO_API_KEY 는 config에서만 관리
from recommend.config import *
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

# ------------------------
# Public API
# ------------------------
def run(
    region: str,
    transport_mode: str,             # 'walk' | 'transit' (여기선 반경 계산에만 사용)
    score_label: str,                # '인기도지수' | '관광지수'
    days: int,                       # UI 슬라이더 값
    cats: List[str],                 # 예: ["음식","자연","레포츠"] (선호 순서)
) -> pd.DataFrame:
    """
    UI 값만 써서 도보 중심 일정표 DataFrame을 반환.
    - 파일 저장 없음
    - PATH_TMF/KAKAO_API_KEY 는 config에서 로드
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
    days = max(1, min(100, int(days)))  # 슬라이더 상한 보호

    cats = list(dict.fromkeys([_nfc(c) for c in (cats or [])]))  # 중복 제거 + 정규화
    if not cats:
        raise ValueError("테마를 최소 1개 이상 선택하세요. (예: 음식, 자연, 레포츠)")
    if len(cats) > 3:
        cats = cats[:3]  # 상위 3개만 사용

    # ----- 지역 지오코딩 -----
    coords = _geocode_region_kakao(region)
    if not coords:
        raise ValueError(f"카카오 지오코딩 실패: '{region}'")
    center_lat, center_lon = coords

    # ----- 원본 CSV 로드(열 표준화) -----
    tmf = _read_csv_robust(PATH_TMF)
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
    miss = [k for k,v in need.items() if v is None]
    if miss:
        raise KeyError(f"필수 컬럼 누락: {miss} / 실제컬럼: {list(tmf.columns)}")

    df = tmf.rename(columns={
        need["title"]: "title",
        need["addr1"]: "addr1",
        need["cat1"]:  "cat1",
        need["mapx"]:  "lon",      # mapx = 경도
        need["mapy"]:  "lat",      # mapy = 위도
        need["tour_score"]: "tour_score",
        need["review_score"]: "review_score",
    }).copy()

    # 타입 정리
    for c in ("lat","lon","tour_score","review_score"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["lat","lon"]).copy()

    # 🔐 일부 CSV에 cat2/cat3가 없을 수 있으므로 기본 컬럼 보강
    if "cat2" not in df.columns: df["cat2"] = ""
    if "cat3" not in df.columns: df["cat3"] = ""

    # ----- 반경/거리 필터 -----
    radius_km = 5 if transport_mode == "walk" else 20
    df["distance_km"] = df.apply(lambda r: _haversine(center_lat, center_lon, r["lat"], r["lon"]), axis=1)
    df = df[df["distance_km"] <= radius_km].copy()
    if df.empty:
        raise RuntimeError("선택한 지역 주변(반경)에서 후보가 없습니다. 지역/반경/데이터를 확인하세요.")

    # ----- 정렬 기준 -----
    score_map = {"관광지수":"tour_score", "인기도지수":"review_score"}
    score_col = score_map[score_label]
    df = df.sort_values([score_col, "distance_km"], ascending=[False, True]).copy()
    df = df.drop_duplicates(subset=["title","addr1"], keep="first").reset_index(drop=True)

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

    # ======================================
    # 스케줄링 (도보 기준)
    # ======================================
    MEAL_CAT = "음식"
    DAY_TOTAL_SLOTS = 5
    BASE_WEIGHTS = [3,2,1]  # cats 선호 3:2:1
    BLOCKED_CAFE_KEYS = {"카페","전통찻집"}
    MEAL_CUISINE_TAGS = {"서양식","이색음식점","일식","중식","한식"}

    selected_pool = df_sorted[df_sorted["cat1_norm"].isin(cats)].copy()
    if selected_pool.empty:
        selected_pool = df_sorted.copy()

    center_lat = float(selected_pool["lat"].mean())
    center_lon = float(selected_pool["lon"].mean())

    # ===== 유틸 =====
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
        dkm = np.sqrt((sub["lat"]-cur_lat)**2 + (sub["lon"]-cur_lon)**2) * 111.0
        pen = (dkm.apply(lambda x: travel_minutes(x))) / 60.0
        rank = sub["final_score"].fillna(0) - 0.1 * pen
        idx = rank.sort_values(ascending=False).index[0]
        return idx, float(dkm.loc[idx])

    def build_day_quota() -> dict:
        u = list(dict.fromkeys(cats))
        quotas = {c:0 for c in u}
        for i,w in enumerate(BASE_WEIGHTS):
            if i < len(u):
                quotas[u[i]] += w
        remain = DAY_TOTAL_SLOTS - sum(quotas.values())
        i = 0
        while remain > 0 and u:
            quotas[u[i % len(u)]] += 1
            remain -= 1
            i += 1
        return quotas

    def _bucket_ok(day_rows: List[dict], new_row: pd.Series, meal_keys: set[str]) -> bool:
        def _bucket_from_tag(s: str) -> Optional[str]:
            t = _nfc(s)
            for k in meal_keys:
                if k in t:
                    return k
            return None
        cand_bucket = _bucket_from_tag(str(new_row.get("cat3_norm", new_row.get("cat3",""))))
        if cand_bucket is None:
            return True
        for r in day_rows:
            if _nfc(r.get("cat1","")) != "음식":
                continue
            if _bucket_from_tag(str(r.get("cat3",""))) == cand_bucket:
                return False
        return True

    # 본 스케줄링
    itins = []
    cur_lat, cur_lon = center_lat, center_lon
    remain_pool = selected_pool.copy()

    today0 = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    meal_enabled = (MEAL_CAT in set(cats))

    for d in range(1, days+1):
        base = today0 + timedelta(days=d-1)
        day_start  = base.replace(hour=8,  minute=0)
        lunch_s, lunch_e = base.replace(hour=11, minute=0), base.replace(hour=13, minute=0)
        dinner_s, dinner_e = base.replace(hour=17, minute=0), base.replace(hour=20, minute=0)
        day_end = base.replace(hour=22, minute=30)

        quotas = build_day_quota()
        cur_time = day_start
        day_rows: List[dict] = []

        # 오전: 음식 제외 선점
        non_food = [c for c in cats if c != MEAL_CAT]
        for c in non_food:
            if quotas.get(c,0) <= 0 or cur_time >= lunch_s: continue
            idx, dkm = pick_best(remain_pool[remain_pool["cat1_norm"]==c], cur_lat, cur_lon)
            if idx is None: continue
            
            row = remain_pool.loc[idx]
            t_mv = travel_minutes(dkm) + 10.0
            t_st = stay_minutes(c)
            
            move_start_time = cur_time
            move_end_time = cur_time + timedelta(minutes=t_mv)

            if move_end_time > lunch_s: continue

            if day_rows:
                prev_title = day_rows[-1]["title"]
                next_title = _nfc(row.get("title", ""))
                day_rows.append(_move_row(d, move_start_time, move_end_time, prev_title, next_title))
            
            st = move_end_time
            et = st + timedelta(minutes=t_st)
            
            if et > lunch_s:
                if day_rows and day_rows[-1]['title'] == '이동':
                    day_rows.pop()
                continue
            
            day_rows.append(_visit_row(d, st, et, row, c, dkm, t_mv, t_st))
            quotas[c] -= 1
            cur_time = et
            cur_lat, cur_lon = float(row["lat"]), float(row["lon"])
            remain_pool = remain_pool.drop(index=idx)

        # 오전 나머지(음식 제외)
        while cur_time < lunch_s and sum(quotas.values())>0:
            choices = [c for c in cats if quotas.get(c,0)>0 and c != MEAL_CAT]
            if not choices: break
            choices.sort(key=lambda x: -quotas.get(x,0))
            placed=False
            for c in choices:
                idx, dkm = pick_best(remain_pool[remain_pool["cat1_norm"]==c], cur_lat, cur_lon)
                if idx is None: continue
                
                row = remain_pool.loc[idx]
                t_mv = travel_minutes(dkm) + 10.0
                t_st = stay_minutes(c)
                
                move_start_time = cur_time
                move_end_time = cur_time + timedelta(minutes=t_mv)
                if move_end_time > lunch_s: continue

                if day_rows:
                    prev_title = day_rows[-1]["title"]
                    next_title = _nfc(row.get("title", ""))
                    day_rows.append(_move_row(d, move_start_time, move_end_time, prev_title, next_title))

                st = move_end_time
                et = st + timedelta(minutes=t_st)

                if et > lunch_s:
                    if day_rows and day_rows[-1]['title'] == '이동':
                        day_rows.pop()
                    continue

                day_rows.append(_visit_row(d, st, et, row, c, dkm, t_mv, t_st))
                quotas[c] -= 1
                cur_time = et
                cur_lat, cur_lon = float(row["lat"]), float(row["lon"])
                remain_pool = remain_pool.drop(index=idx)
                placed=True
                break
            if not placed: break

        # 점심 (음식 1곳)
        if meal_enabled and quotas.get(MEAL_CAT,0)>0:
            cur_time = max(cur_time, lunch_s)
            if cur_time < lunch_e:
                sub = remain_pool[(remain_pool["cat1_norm"]==MEAL_CAT)]
                sub = sub[~sub["cat3_norm"].apply(is_blocked_cafe_tag)]
                if not sub.empty:
                    idx, dkm = pick_best(sub, cur_lat, cur_lon)
                    if idx is not None:
                        row = remain_pool.loc[idx]
                        if _bucket_ok(day_rows, row, MEAL_CUISINE_TAGS):
                            t_mv = travel_minutes(dkm) + 10.0
                            t_st = stay_minutes(MEAL_CAT)
                            
                            move_start_time = cur_time
                            move_end_time = cur_time + timedelta(minutes=t_mv)

                            if day_rows:
                                prev_title = day_rows[-1]["title"]
                                next_title = _nfc(row.get("title", ""))
                                day_rows.append(_move_row(d, move_start_time, move_end_time, prev_title, next_title))
                            
                            st = move_end_time
                            et = st + timedelta(minutes=t_st)
                            
                            if et <= lunch_e:
                                day_rows.append(_visit_row(d, st, et, row, MEAL_CAT, dkm, t_mv, t_st))
                                quotas[MEAL_CAT] -= 1
                                cur_time = et
                                cur_lat, cur_lon = float(row["lat"]), float(row["lon"])
                                remain_pool = remain_pool.drop(index=idx)
                            elif day_rows and day_rows[-1]['title'] == '이동':
                                day_rows.pop()

        # 오후(음식 제외)
        while cur_time < dinner_s and sum(quotas.values())>0:
            choices = [c for c in cats if quotas.get(c,0)>0 and c != MEAL_CAT]
            if not choices: break
            choices.sort(key=lambda x: -quotas.get(x,0))
            placed=False
            for c in choices:
                idx, dkm = pick_best(remain_pool[remain_pool["cat1_norm"]==c], cur_lat, cur_lon)
                if idx is None: continue
                row = remain_pool.loc[idx]
                t_mv = travel_minutes(dkm) + 10.0
                t_st = stay_minutes(c)

                move_start_time = cur_time
                move_end_time = cur_time + timedelta(minutes=t_mv)
                if move_end_time > dinner_s: continue

                if day_rows:
                    prev_title = day_rows[-1]["title"]
                    next_title = _nfc(row.get("title", ""))
                    day_rows.append(_move_row(d, move_start_time, move_end_time, prev_title, next_title))

                st = move_end_time
                et = st + timedelta(minutes=t_st)

                if et > dinner_s:
                    if day_rows and day_rows[-1]['title'] == '이동':
                        day_rows.pop()
                    continue

                day_rows.append(_visit_row(d, st, et, row, c, dkm, t_mv, t_st))
                quotas[c] -= 1
                cur_time = et
                cur_lat, cur_lon = float(row["lat"]), float(row["lon"])
                remain_pool = remain_pool.drop(index=idx)
                placed=True
                break
            if not placed: break

        # 저녁 (음식 1곳)
        if meal_enabled and quotas.get(MEAL_CAT,0)>0:
            cur_time = max(cur_time, dinner_s)
            if cur_time < dinner_e:
                sub = remain_pool[(remain_pool["cat1_norm"]==MEAL_CAT)]
                sub = sub[~sub["cat3_norm"].apply(is_blocked_cafe_tag)]
                if not sub.empty:
                    idx, dkm = pick_best(sub, cur_lat, cur_lon)
                    if idx is not None:
                        row = remain_pool.loc[idx]
                        if _bucket_ok(day_rows, row, MEAL_CUISINE_TAGS):
                            t_mv = travel_minutes(dkm) + 10.0
                            t_st = stay_minutes(MEAL_CAT)

                            move_start_time = cur_time
                            move_end_time = cur_time + timedelta(minutes=t_mv)

                            if day_rows:
                                prev_title = day_rows[-1]["title"]
                                next_title = _nfc(row.get("title", ""))
                                day_rows.append(_move_row(d, move_start_time, move_end_time, prev_title, next_title))

                            st = move_end_time
                            et = st + timedelta(minutes=t_st)

                            if et <= dinner_e:
                                day_rows.append(_visit_row(d, st, et, row, MEAL_CAT, dkm, t_mv, t_st))
                                quotas[MEAL_CAT] -= 1
                                cur_time = et
                                cur_lat, cur_lon = float(row["lat"]), float(row["lon"])
                                remain_pool = remain_pool.drop(index=idx)
                            elif day_rows and day_rows[-1]['title'] == '이동':
                                day_rows.pop()

        # 저녁 이후(~22:30) 남은 쿼터 (20시 이후 버킷류 금지)
        while cur_time < day_end and sum(quotas.values())>0:
            choices = [c for c in cats if quotas.get(c,0)>0]
            if not choices: break
            choices.sort(key=lambda x: -quotas.get(x,0))
            placed=False
            for c in choices:
                sub = remain_pool[remain_pool["cat1_norm"]==c]
                if c == MEAL_CAT:
                    sub = sub[~sub["cat3_norm"].apply(lambda s: _contains_any(s, MEAL_CUISINE_TAGS))]
                idx, dkm = pick_best(sub, cur_lat, cur_lon)
                if idx is None: continue
                row = remain_pool.loc[idx]
                t_mv = travel_minutes(dkm) + 10.0
                t_st = stay_minutes(c)

                move_start_time = cur_time
                move_end_time = cur_time + timedelta(minutes=t_mv)
                if move_end_time > day_end: continue

                if day_rows:
                    prev_title = day_rows[-1]["title"]
                    next_title = _nfc(row.get("title", ""))
                    day_rows.append(_move_row(d, move_start_time, move_end_time, prev_title, next_title))

                st = move_end_time
                et = st + timedelta(minutes=t_st)
                
                if et > day_end:
                    if day_rows and day_rows[-1]['title'] == '이동':
                        day_rows.pop()
                    continue

                day_rows.append(_visit_row(d, st, et, row, c, dkm, t_mv, t_st))
                quotas[c] -= 1
                cur_time = et
                cur_lat, cur_lon = float(row["lat"]), float(row["lon"])
                remain_pool = remain_pool.drop(index=idx)
                placed=True
                break
            if not placed: break

        # 하루 마감(빈칸이면 건너뜀)
        day_df = pd.DataFrame(day_rows)
        if not day_df.empty:
            day_df = day_df.sort_values("start_time").reset_index(drop=True)
        itins.append(day_df)

    # 합치기 — ✅ 최종 산출은 이 DataFrame
    # ▼▼▼ 수정된 부분 ▼▼▼
    itinerary = pd.concat(itins, ignore_index=True) if itins else pd.DataFrame(
        columns=["day","start_time","end_time","title","addr1","cat1","cat2","cat3",
                 "final_score","distance_from_prev_km","move_min","stay_min",
                 "출발지", "도착지", "교통편1", "교통편2"]
    )
    # ▲▲▲ 수정된 부분 ▲▲▲
    return itinerary

# ------------------------
# Helpers
# ------------------------
def _nfc(s: str) -> str:
    return ud.normalize("NFC", str(s or "")).strip()

def _first_contains(cols, key):
    key = key.lower()
    for c in cols:
        if key in c.lower():
            return c
    return None

def _read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ("utf-8","utf-8-sig","cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2)-float(lat1))
    dlmb = math.radians(float(lon2)-float(lon1))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def _geocode_region_kakao(region_name: str) -> Optional[Tuple[float,float]]:
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params  = {"query": region_name}
    resp    = requests.get(url, headers=headers, params=params)
    docs    = resp.json().get("documents", [])
    if not docs:
        return None
    first = docs[0]
    return float(first["y"]), float(first["x"])  # (lat, lon)

def _contains_any(s: str, keys: set[str]) -> bool:
    t = _nfc(s)
    return any(k in t for k in keys)

# ▼▼▼ 수정된 부분 (새로운 함수 추가) ▼▼▼
def _move_row(day_i: int, st: datetime, et: datetime, from_title: str, to_title: str) -> dict:
    """이동 정보를 담는 딕셔너리를 생성합니다."""
    return {
        "day": day_i,
        "start_time": st.strftime("%H:%M"),
        "end_time": et.strftime("%H:%M"),
        "title": "이동",
        "addr1": "", "cat1": "", "cat2": "", "cat3": "",
        "final_score": np.nan,
        "distance_from_prev_km": np.nan,
        "move_min": int((et - st).total_seconds() / 60),
        "stay_min": 0,
        "출발지": from_title,
        "도착지": to_title,
        "교통편1": "도보",
        "교통편2": "",
    }
# ▲▲▲ 수정된 부분 ▲▲▲

def _visit_row(day_i, st: datetime, et: datetime, row: pd.Series, cat: str, dkm: float, t_mv: float, t_st: float) -> dict:
    return {
        "day": day_i,
        "start_time": st.time().strftime("%H:%M"),
        "end_time": et.time().strftime("%H:%M"),
        "title": _nfc(row.get("title","")),
        "addr1": _nfc(row.get("addr1","")),
        "cat1": _nfc(row.get("cat1","")),
        "cat2": _nfc(row.get("cat2","")),
        "cat3": _nfc(row.get("cat3","")),
        "final_score": float(row.get("review_score") if pd.notna(row.get("review_score")) else row.get("tour_score") or 0),
        "distance_from_prev_km": round(float(dkm), 2) if pd.notna(dkm) else np.nan,
        "move_min": int(round(t_mv)),
        "stay_min": int(round(t_st)),
        # ▼▼▼ 수정된 부분 (일관성을 위한 키 추가) ▼▼▼
        "출발지": "",
        "도착지": "",
        "교통편1": "",
        "교통편2": "",
        # ▲▲▲ 수정된 부분 ▲▲▲
    }