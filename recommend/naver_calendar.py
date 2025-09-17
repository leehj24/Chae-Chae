# recommend/naver_calendar.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Tuple, Optional

import requests

from recommend.config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET

# OAuth / API Endpoints
TOKEN_API_URL = "https://nid.naver.com/oauth2.0/token"
CREATE_API_URL = "https://openapi.naver.com/calendar/createSchedule.json"

# ─────────────────────────────────────────────────────────────────────
# Public: OAuth 토큰 교환
# ─────────────────────────────────────────────────────────────────────
def get_access_token(code: str, state: str) -> Optional[dict]:
    """
    Authorization Code -> Access Token 교환.
    반환: {"access_token": "...", "refresh_token": "...", ...} 또는 None
    """
    data = {
        "grant_type": "authorization_code",
        "client_id": NAVER_CLIENT_ID,
        "client_secret": NAVER_CLIENT_SECRET,
        "code": code,
        "state": state,
    }
    try:
        r = requests.post(TOKEN_API_URL, data=data, timeout=8)
        if not r.ok:
            print(f"❌ Naver token error: {r.text}")
            return None
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Naver token exception: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────
# Public: 일정 생성
# ─────────────────────────────────────────────────────────────────────
def add_schedule(
    access_token: str,
    schedule: dict,
    calendar_id: str = "defaultCalendarId",
) -> Tuple[bool, str]:
    """
    네이버 캘린더 일정 생성.
    Args:
        access_token: 네이버 OAuth 액세스 토큰
        schedule: {
            "title": str,
            "startTime": "YYYY-MM-DDTHH:MM[:SS]",
            "endTime":   "YYYY-MM-DDTHH:MM[:SS]",
            "location": str,           # optional
            "description": str,        # optional
        }
        calendar_id: 기본값은 'defaultCalendarId'
    Returns:
        (ok: bool, body: str)
        ok 은 HTTP 200 그리고 응답 JSON result == "success" 일 때만 True
        body 는 원문 응답 문자열
    """
    ical = _build_vcalendar(
        summary=schedule.get("title", "제목 없음"),
        start_iso=schedule["startTime"],
        end_iso=schedule["endTime"],
        location=schedule.get("location", ""),
        description=schedule.get("description", ""),
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
    }
    data = {
        "calendarId": calendar_id,
        "scheduleIcalString": ical,
    }

    try:
        r = requests.post(CREATE_API_URL, headers=headers, data=data, timeout=10)
        body = r.text
        if not r.ok:
            # HTTP 오류면 바로 실패
            return False, body

        # JSON 파싱해 result 확인(네이버는 성공 시 result: "success")
        try:
            j = r.json()
            ok = (j.get("result") == "success")
            return (ok, body) if ok else (False, body)
        except Exception:
            # JSON 아님 → 실패 처리
            return False, body
    except requests.exceptions.RequestException as e:
        return False, str(e)

# ─────────────────────────────────────────────────────────────────────
# ICS Builder
# ─────────────────────────────────────────────────────────────────────
def _build_vcalendar(
    summary: str,
    start_iso: str,
    end_iso: str,
    location: str = "",
    description: str = "",
    tzid: str = "Asia/Seoul",
) -> str:
    """
    네이버 createSchedule.json 에 전달할 단일 VEVENT 포함 VCALENDAR 문자열 생성.
    - DTSTART/DTEND 에 TZID 포함(로컬 시간대 지정)
    - CRLF(\r\n) 사용
    - 최소한의 속성만 사용(네이버 호환성 우선)
    """
    uid = _gen_uid()
    dtstamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    dtstart_local = _fmt_ics_datetime(start_iso)  # YYYYMMDDTHHMMSS
    dtend_local = _fmt_ics_datetime(end_iso)

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//ChaeChae//Itinerary//KR",
        "CALSCALE:GREGORIAN",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{dtstamp}",
        f"DTSTART;TZID={tzid}:{dtstart_local}",
        f"DTEND;TZID={tzid}:{dtend_local}",
        f"SUMMARY:{_esc_ics(summary)}",
        f"LOCATION:{_esc_ics(location)}" if location else "LOCATION:",
        f"DESCRIPTION:{_esc_ics(description)}" if description else "DESCRIPTION:",
        "END:VEVENT",
        "END:VCALENDAR",
    ]

    # 75 옥텟 폴딩(간단 구현) 및 CRLF 보장
    folded = []
    for line in lines:
        folded.extend(_fold_ics_line(line))
    return "\r\n".join(folded)

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _gen_uid() -> str:
    # RFC5545 명시적 제약은 없지만, 전역 유일성 확보를 위해 host-like suffix 부여
    return f"{uuid.uuid4().hex}@chaechae.local"

def _fmt_ics_datetime(iso_str: str) -> str:
    """
    'YYYY-MM-DDTHH:MM' or 'YYYY-MM-DDTHH:MM:SS' → 'YYYYMMDDTHHMMSS'
    seconds 미지정 시 00 가정.
    """
    s = iso_str.strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?", s)
    if not m:
        # fallback: 날짜만 들어오는 경우 YYYYMMDD
        m2 = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
        if m2:
            y, M, d = m2.groups()
            return f"{y}{M}{d}T000000"
        # 완전 실패 시 현재시간
        return datetime.now().strftime("%Y%m%dT%H%M%S")

    y, M, d, h, m, sec = m.groups()
    sec = sec if sec is not None else "00"
    return f"{y}{M}{d}T{h}{m}{sec}"

def _esc_ics(text: str) -> str:
    """
    ICS 속성값 이스케이프: 백슬래시, 콤마, 세미콜론, 줄바꿈
    - \  → \\
    - ,  → \,
    - ;  → \;
    - \n → \\n
    """
    t = str(text or "")
    t = t.replace("\\", "\\\\")
    t = t.replace(",", r"\,")
    t = t.replace(";", r"\;")
    t = t.replace("\r\n", r"\n").replace("\n", r"\n").replace("\r", r"\n")
    return t

def _fold_ics_line(line: str, limit: int = 75) -> list[str]:
    """
    ICS line folding (RFC 5545).
    단순 문자 길이 기준으로 폴딩(옥텟 기준과 약간의 차이는 있을 수 있으나 실사용 문제 없음).
    첫 줄 이후는 한 칸(space) 들여쓰기.
    """
    if len(line) <= limit:
        return [line]
    out = [line[:limit]]
    s = line[limit:]
    while s:
        out.append(" " + s[:limit])
        s = s[limit:]
    return out
