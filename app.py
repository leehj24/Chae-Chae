# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import traceback
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from flask import (
    Flask,
    Response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_session import Session

# 추천 엔진 모듈
from recommend import run_walk as run_walk_module
from recommend import run_transit as run_transit_module
from recommend.config import PATH_TMF  # 홈 카드용 CSV 절대경로

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.secret_key = "dev-secret-key"  # 개발용

# ---------- 서버사이드 세션 ----------
app.config.update(
    SESSION_TYPE="filesystem",
    SESSION_FILE_DIR=str((BASE_DIR / "_fs_sessions").resolve()),
    SESSION_PERMANENT=False,
    SESSION_USE_SIGNER=True,
    SESSION_COOKIE_NAME="srv_session",
)
Path(app.config["SESSION_FILE_DIR"]).mkdir(parents=True, exist_ok=True)
Session(app)

# ---------- 공통 유틸 ----------
MAX_MSGS = 30  # 세션 비대화 방지

def _trim_msgs():
    session["messages"] = session.get("messages", [])[-MAX_MSGS:]

def _json(payload: Dict[str, Any], status: int = 200) -> Response:
    # NaN이 JSON으로 나가는 것을 막아 SyntaxError 방지
    return app.response_class(
        response=json.dumps(payload, ensure_ascii=False, allow_nan=False),
        status=status,
        mimetype="application/json",
    )

def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    clean = df.replace({np.nan: None})
    recs = clean.to_dict(orient="records")
    for r in recs:
        for k, v in list(r.items()):
            if isinstance(v, np.generic):
                r[k] = v.item()
    return recs

def _parse_themes(raw: str) -> List[str]:
    if not raw:
        return []
    s = str(raw).strip()
    try:
        if s.startswith("[") and s.endswith("]"):
            arr = json.loads(s)
            return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        pass
    return [t.strip() for t in s.split(",") if t.strip()]

def _parse_date(s: str) -> date | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

def _drop_inline_spinner_messages():
    msgs = session.get("messages", [])
    if not msgs:
        return
    filtered = []
    for m in msgs:
        html = m.get("html")
        if html and "inline-spinner" in html:
            continue
        filtered.append(m)
    session["messages"] = filtered[-MAX_MSGS:]

def _push_bot_text(text: str):
    session["messages"].append({"sender": "bot", "text": text})
    _trim_msgs()

def _push_bot_html(html: str):
    session["messages"].append({"sender": "bot", "html": html})
    _trim_msgs()

def _push_user(text: str):
    session["messages"].append({"sender": "user", "text": text})
    _trim_msgs()

# ─────────────────────────────────────────
# 홈(메인) 카드용: CSV 로딩 & 캐시 + 정규화
# ─────────────────────────────────────────
_PLACES_CACHE = {"df": None, "mtime": None, "path": None}

def _read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def _pick_column(df: pd.DataFrame, *names: str) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    for c in df.columns:
        cl = c.lower()
        for n in names:
            if n.lower() in cl:
                return c
    return None

def _load_places_df() -> pd.DataFrame:
    path = PATH_TMF
    p = Path(path)
    mtime = p.stat().st_mtime if p.exists() else None

    # 캐시 히트
    if _PLACES_CACHE["df"] is not None and _PLACES_CACHE["mtime"] == mtime:
        return _PLACES_CACHE["df"].copy()

    df = _read_csv_robust(path).copy()

    # 필수 컬럼 매핑
    req = {
        "title":        _pick_column(df, "title", "명칭", "place", "name"),
        "addr1":        _pick_column(df, "addr1", "주소"),
        "cat1":         _pick_column(df, "cat1", "대분류", "category1"),
        "tour_score":   _pick_column(df, "tour_score", "관광지수", "tour-score"),
        "review_score": _pick_column(df, "review_score", "인기도지수", "review-score"),
    }
    miss = [k for k, v in req.items() if v is None]
    if miss:
        raise KeyError(f"CSV 필수 컬럼 누락: {miss} / 실제: {list(df.columns)}")

    # 선택 컬럼
    opt = {
        "cat3":        _pick_column(df, "cat3", "소분류", "category3"),
        "firstimage":  _pick_column(df, "firstimage", "image", "img1", "thumbnail"),
        "firstimage2": _pick_column(df, "firstimage2", "image2", "img2"),
    }

    # 리네임
    rename_map = {
        req["title"]: "title",
        req["addr1"]: "addr1",
        req["cat1"]: "cat1",
        req["tour_score"]: "tour_score",
        req["review_score"]: "review_score",
    }
    if opt["cat3"]:        rename_map[opt["cat3"]] = "cat3"
    if opt["firstimage"]:  rename_map[opt["firstimage"]] = "firstimage"
    if opt["firstimage2"]: rename_map[opt["firstimage2"]] = "firstimage2"

    df = df.rename(columns=rename_map)

    # 기본값 보강
    for c in ("cat3", "firstimage", "firstimage2"):
        if c not in df.columns:
            df[c] = ""

    # 타입 정리
    for c in ("tour_score", "review_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("title", "addr1", "cat1", "cat3", "firstimage", "firstimage2"):
        df[c] = df[c].astype(str)

    # 중복 제거
    df = df.drop_duplicates(subset=["title", "addr1"], keep="first").reset_index(drop=True)

    _PLACES_CACHE.update({"df": df.copy(), "mtime": mtime, "path": path})
    return df

def _sort_key_from_param(s: str) -> tuple[str, str]:
    s = (s or "").strip().lower()
    if s in {"popular", "review", "review_score", "인기도"}:
        return "review_score", "인기도 지수"
    return "tour_score", "관광 지수"

# ── ‘질문 말풍선’ 고정 노출용 헬퍼들 ─────────────────────────
def _persist_region_prompt_once():
    html = "안녕하세요! 😊<br /><b>어떤 지역</b>으로 여행 가실 건가요? 아래 입력창에 지역명을 적어주세요."
    msgs = session.get("messages", [])
    if not any(m.get("sender") == "bot" and m.get("html") == html for m in msgs[-10:]):
        _push_bot_html(html)

def _persist_theme_prompt_once():
    html = "좋아요! 이제 <b>원하는 테마를 최대 3개</b>까지 골라주세요. (다시 누르면 해제돼요)"
    msgs = session.get("messages", [])
    if not any(m.get("sender") == "bot" and m.get("html") == html for m in msgs[-10:]):
        _push_bot_html(html)

def _persist_score_prompt_once():
    html = "어떤 기준으로 추천할까요? (<b>관광지수 vs 인기도지수</b>) 하나만 선택해 주세요."
    msgs = session.get("messages", [])
    if not any(m.get("sender") == "bot" and m.get("html") == html for m in msgs[-10:]):
        _push_bot_html(html)

def _persist_transport_prompt_once():
    html = "마지막으로, <b>어떤 이동수단</b>으로 맞출까요? (걷기 vs 대중교통) 하나만 선택해 주세요."
    msgs = session.get("messages", [])
    if not any(m.get("sender") == "bot" and m.get("html") == html for m in msgs[-10:]):
        _push_bot_html(html)

def _persist_date_prompt_once():
    html = "<b>여행 기간</b>을 선택해 주세요. 시작~종료 날짜를 고르면 <em>총 일수</em>가 자동 계산돼요."
    msgs = session.get("messages", [])
    if not any(m.get("sender") == "bot" and m.get("html") == html for m in msgs[-10:]):
        _push_bot_html(html)

# ── 뒤로가기 전용 헬퍼들 ────────────────────────────────────
def _state_prompt_html(state: str) -> str:
    mapping = {
        "지역": "안녕하세요! 😊<br /><b>어떤 지역</b>으로 여행 가실 건가요? 아래 입력창에 지역명을 적어주세요.",
        "점수": "어떤 기준으로 추천할까요? (<b>관광지수 vs 인기도지수</b>) 하나만 선택해 주세요.",
        "테마": "좋아요! 이제 <b>원하는 테마를 최대 3개</b>까지 골라주세요. (다시 누르면 해제돼요)",
        "기간": "<b>여행 기간</b>을 선택해 주세요. 시작~종료 날짜를 고르면 <em>총 일수</em>가 자동 계산돼요.",
        "이동수단": "마지막으로, <b>어떤 이동수단</b>으로 맞출까요? (걷기 vs 대중교통) 하나만 선택해 주세요.",
    }
    return mapping.get(state, "")

def _prune_messages_to_state(target_state: str) -> None:
    msgs = session.get("messages", [])
    prompt_html = _state_prompt_html(target_state)

    if target_state == "지역":
        session["messages"] = []
        _trim_msgs()
        return

    cut_idx = None
    if prompt_html:
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if m.get("sender") == "bot" and m.get("html") == prompt_html:
                cut_idx = i
                break

    kept = msgs if cut_idx is None else msgs[:cut_idx]
    if prompt_html:
        kept = [m for m in kept if not (m.get("sender") == "bot" and m.get("html") == prompt_html)]

    session["messages"] = kept[-MAX_MSGS:]

def _clear_values_from_state(target_state: str) -> None:
    if target_state == "지역":
        session.update(
            region=None, score_label=None, cats=[],
            start_date=None, end_date=None, days=2,
            transport_mode=None, itinerary=[], columns=[],
            pending_job=False
        )
    elif target_state == "점수":
        session.update(
            score_label=None, cats=[],
            start_date=None, end_date=None, days=2,
            transport_mode=None, itinerary=[], columns=[],
            pending_job=False
        )
    elif target_state == "테마":
        session.update(
            cats=[], start_date=None, end_date=None, days=2,
            transport_mode=None, itinerary=[], columns=[],
            pending_job=False
        )
    elif target_state == "기간":
        session.update(
            start_date=None, end_date=None, days=2,
            transport_mode=None, itinerary=[], columns=[],
            pending_job=False
        )
    elif target_state == "이동수단":
        session.update(
            transport_mode=None, itinerary=[], columns=[],
            pending_job=False
        )
    else:
        session["pending_job"] = False
        _drop_inline_spinner_messages()

def _init_session_if_needed():
    # '지역' → '점수' → '테마' → '기간' → '이동수단' → '실행중' → '완료'
    session.setdefault("state", "지역")
    session.setdefault("messages", [])
    session.setdefault("region", None)
    session.setdefault("score_label", None)      # '관광지수' | '인기도지수'
    session.setdefault("cats", [])
    session.setdefault("days", 2)
    session.setdefault("start_date", None)
    session.setdefault("end_date", None)
    session.setdefault("transport_mode", None)   # 'walk' | 'transit'
    session.setdefault("itinerary", [])
    session.setdefault("columns", [])
    session.setdefault("pending_job", False)

# -----------------------
# 헬스체크
# -----------------------
@app.get("/health")
def health() -> Response:
    return _json({"ok": True})

# -----------------------
# 홈(메인) — 템플릿 렌더
# -----------------------
@app.get("/")
def home() -> Response:
    return render_template("home.html")

# -----------------------
# 홈(메인) — 카드 JSON API
# -----------------------
@app.get("/api/places")
def api_places() -> Response:
    """
    /api/places?sort=review|tour&page=1&per_page=40
    응답: { ok, sort_label, total, page, per_page, total_pages, items:[...] }

    정렬 규칙:
      1) 드롭다운으로 고른 컬럼(review_score|tour_score) 내림차순(높은 점수 먼저)
      2) 동점(특히 0점)은 title 오름차순(가나다)
      3) NaN/null 점수는 가장 뒤 (그 안에서도 title 오름차순)
    """
    try:
        df = _load_places_df()

        sort = request.args.get("sort", "review")
        page = max(1, int(request.args.get("page", 1)))
        per_page = max(1, int(request.args.get("per_page", 40)))

        score_col, score_label = _sort_key_from_param(sort)

        # 내림차순 + 동점 제목 ㄱㄴㄷ + NaN은 맨 뒤
        df = df.sort_values(
            by=[score_col, "title"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)

        total = len(df)
        total_pages = max(1, math.ceil(total / per_page))
        page = min(page, total_pages)
        start = (page - 1) * per_page
        end = start + per_page

        view = df.iloc[start:end].copy()
        view["rank"] = range(start + 1, start + 1 + len(view))  # 전역 랭크
        view["score_value"] = view[score_col].fillna(0).round(0).astype(int)  # 배지용 정수

        def _float_or_none(x):
            return None if pd.isna(x) else float(x)

        items: List[Dict[str, Any]] = []
        for _, r in view.iterrows():
            images = []
            for col in ("firstimage", "firstimage2"):
                v = str(r.get(col, "")).strip()
                if v and v.lower() not in {"nan", "none", "null"}:
                    images.append(v)

            items.append({
                "rank": int(r["rank"]),
                "title": str(r["title"]),
                "addr1": str(r["addr1"]),
                "cat1":  str(r["cat1"]),
                "cat3":  str(r.get("cat3", "") or ""),
                "firstimage":  str(r.get("firstimage", "") or ""),
                "firstimage2": str(r.get("firstimage2", "") or ""),
                "images": images,
                "tour_score":   _float_or_none(r.get("tour_score")),
                "review_score": _float_or_none(r.get("review_score")),
                "score": int(r["score_value"]),  # 현재 정렬 기준 점수의 정수 배지
            })

        return _json({
            "ok": True,
            "sort_label": score_label,
            "sort_col": score_col,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "items": items,
        })
    except Exception as e:
        tb = traceback.format_exc(limit=6)
        return _json({"ok": False, "error": str(e), "trace": tb}, 500)

# -----------------------
# (채팅) 인덱스 페이지: /chat 이동
# -----------------------
@app.get("/chat")
def index() -> Response:
    _init_session_if_needed()
    if session.get("state") != "실행중":
        _drop_inline_spinner_messages()
    return render_template("index.html")

@app.get("/reset_chat")
def reset_chat():
    session.clear()
    return redirect(url_for("index"))

# -----------------------
# 뒤로 가기
# -----------------------
@app.get("/go_back")
def go_back():
    _init_session_if_needed()
    state = session.get("state", "지역")

    prev_state = {
        "점수": "지역",
        "테마": "점수",
        "기간": "테마",
        "이동수단": "기간",
        "실행중": "이동수단",
        "완료": "이동수단",
    }.get(state, "지역")

    if state == "실행중":
        session["pending_job"] = False
        _drop_inline_spinner_messages()

    _prune_messages_to_state(prev_state)
    _clear_values_from_state(prev_state)
    session["state"] = prev_state
    return redirect(url_for("index"))

# -----------------------
# 채팅(폼 + JSON API)
# -----------------------
@app.post("/chat")
def chat() -> Response:
    _init_session_if_needed()
    if request.is_json:
        try:
            data = request.get_json(force=True, silent=False) or {}
            region = (data.get("region") or "").strip()
            mode = (data.get("transport_mode") or "").strip()
            score_label = (data.get("score_label") or "관광지수").strip()
            days = int(data.get("days") or 1)
            cats = data.get("cats") or []
            if mode not in {"walk", "transit"}:
                return _json({"ok": False, "error": "transport_mode must be 'walk' or 'transit'."}, 400)
            if score_label not in {"관광지수", "인기도지수"}:
                return _json({"ok": False, "error": "score_label must be '관광지수' or '인기도지수'."}, 400)

            if mode == "walk":
                df = run_walk_module.run(region=region, transport_mode="walk", score_label=score_label, days=days, cats=cats)
            else:
                start_time = (data.get("start_time") or "08:00").strip()
                end_time = (data.get("end_time") or "22:30").strip()
                df = run_transit_module.run(
                    region=region, transport_mode="transit", score_label=score_label,
                    days=days, cats=cats, start_time=start_time, end_time=end_time
                )
            rows = _df_to_records(df)
            return _json({"ok": True, "count": len(rows), "columns": list(df.columns), "itinerary": rows})
        except Exception as e:
            tb = traceback.format_exc(limit=6)
            return _json({"ok": False, "error": str(e), "trace": tb}, 500)

    state = session.get("state", "지역")
    try:
        if state == "지역":
            region = (request.form.get("region") or request.form.get("message") or "").strip()
            if not region:
                return redirect(url_for("index"))
            _persist_region_prompt_once()
            _push_user(region)
            session["region"] = region
            session["state"] = "점수"
            return redirect(url_for("index"))

        elif state == "점수":
            val = (request.form.get("score") or request.form.get("message") or "").strip()
            if val not in {"관광지수", "인기도지수"}:
                if val:
                    _push_user(val)
                _push_bot_text("점수 기준은 ‘관광지수’ 또는 ‘인기도지수’ 중 하나를 선택해 주세요.")
                return redirect(url_for("index"))
            _persist_score_prompt_once()
            _push_user(val)
            session["score_label"] = val
            session["state"] = "테마"
            return redirect(url_for("index"))

        elif state == "테마":
            themes_raw = (request.form.get("themes") or request.form.get("message") or "").strip()
            if not themes_raw:
                return redirect(url_for("index"))
            _persist_theme_prompt_once()
            cats = list(dict.fromkeys(_parse_themes(themes_raw)))[:3]
            _push_user(", ".join(cats) if cats else themes_raw)
            if not cats:
                _push_bot_text("테마를 최소 1개 선택해 주세요. (예: 음식, 자연, 레포츠)")
                return redirect(url_for("index"))
            session["cats"] = cats
            session["state"] = "기간"
            return redirect(url_for("index"))

        elif state == "기간":
            _persist_date_prompt_once()
            s_raw = (request.form.get("start_date") or "").strip()
            e_raw = (request.form.get("end_date") or "").strip()
            try:
                s = datetime.strptime(s_raw, "%Y-%m-%d").date()
                e = datetime.strptime(e_raw, "%Y-%m-%d").date()
            except Exception:
                _push_bot_text("날짜 형식이 올바르지 않아요. YYYY-MM-DD로 다시 선택해 주세요.")
                return redirect(url_for("index"))
            if e < s:
                _push_bot_text("종료 날짜가 시작보다 빠릅니다. 다시 선택해 주세요.")
                return redirect(url_for("index"))
            days = (e - s).days + 1
            if days < 1 or days > 100:
                _push_bot_text("여행 기간은 1~100일만 지원해요. 날짜를 다시 선택해 주세요.")
                return redirect(url_for("index"))
            session.update(start_date=s.isoformat(), end_date=e.isoformat(), days=days)
            _push_user(f"{s.isoformat()} ~ {e.isoformat()} (총 {days}일)")
            session["state"] = "이동수단"
            return redirect(url_for("index"))

        elif state == "이동수단":
            transport = (request.form.get("transport") or request.form.get("message") or "").strip()
            if transport not in {"walk", "transit"}:
                if transport:
                    _push_user(transport)
                _push_bot_text("이동수단은 ‘걷기(walk)’ 또는 ‘대중교통(transit)’ 중에서 골라주세요.")
                return redirect(url_for("index"))
            _persist_transport_prompt_once()
            _push_user("걷기" if transport == "walk" else "대중교통")
            session["transport_mode"] = transport
            _drop_inline_spinner_messages()
            session.update(state="실행중", pending_job=True)
            _push_bot_html("<span class='inline-spinner'></span> 일정을 생성하는 중이에요… 잠시만요!")
            return redirect(url_for("index"))

        else:
            msg = (request.form.get("message") or "").strip()
            if msg:
                _push_user(msg)
                if msg in {"다시", "처음", "reset", "restart"}:
                    return redirect(url_for("reset_chat"))
                _push_bot_text("이미 일정이 만들어졌어요. ‘다시’라고 입력하면 처음부터 다시 시작합니다.")
            return redirect(url_for("index"))

    except Exception:
        tb = traceback.format_exc(limit=6)
        _push_bot_text("오류가 발생했어요. ‘다시’라고 입력해 처음부터 다시 시도해 주세요.")
        _push_bot_html(f"<pre class='trace'>{tb}</pre>")
        session["state"] = "완료"
        return redirect(url_for("index"))

# -----------------------
# 비동기 실행 트리거
# -----------------------
@app.post("/do_generate")
def do_generate() -> Response:
    _init_session_if_needed()

    if session.get("state") != "실행중" or not session.get("pending_job"):
        return _json({"ok": False, "error": "no_pending_job"}, 400)

    try:
        region = session.get("region") or ""
        cats   = session.get("cats") or []
        days   = int(session.get("days") or 2)
        score  = session.get("score_label") or "관광지수"
        mode   = session.get("transport_mode") or "transit"

        if mode == "transit":
            df = run_transit_module.run(
                region=region, transport_mode="transit", score_label=score,
                days=days, cats=cats, start_time="08:00", end_time="22:30"
            )
        else:
            df = run_walk_module.run(
                region=region, transport_mode="walk", score_label=score,
                days=days, cats=cats
            )

        session["itinerary"] = _df_to_records(df)
        session["columns"]   = list(df.columns)
        session["state"]     = "완료"
        session["pending_job"] = False

        _drop_inline_spinner_messages()
        _push_bot_text("완료! 추천 일정을 아래에 표시했어요.")
        return _json({"ok": True})

    except Exception:
        tb = traceback.format_exc(limit=6)
        session["pending_job"] = False
        session["state"] = "완료"
        _drop_inline_spinner_messages()
        _push_bot_text("오류가 발생했어요. ‘다시’라고 입력해 처음부터 다시 시도해 주세요.")
        _push_bot_html(f"<pre class='trace'>{tb}</pre>")
        return _json({"ok": False, "error": "failed"})

# -----------------------
# 로컬 실행
# -----------------------
if __name__ == "__main__":
    print("[INFO] templates/home.html =", (BASE_DIR / "templates" / "home.html").exists())
    print("[INFO] templates/index.html =", (BASE_DIR / "templates" / "index.html").exists())
    print("[INFO] static dir =", BASE_DIR / "static")
    app.run(host="0.0.0.0", port=5000, debug=True)
# <-- 여기서 파일 끝! 아래에 HTML 들어가 있으면 전부 삭제
