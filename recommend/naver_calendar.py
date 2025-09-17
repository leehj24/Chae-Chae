# recommend/naver_calendar.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import uuid, requests
from datetime import datetime
from recommend.config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET

TOKEN_API_URL = "https://nid.naver.com/oauth2.0/token"
CREATE_API_URL = "https://openapi.naver.com/calendar/createSchedule.json"

def get_access_token(code: str, state: str) -> dict | None:
    params = {
        "grant_type": "authorization_code",
        "client_id": NAVER_CLIENT_ID,
        "client_secret": NAVER_CLIENT_SECRET,
        "code": code,
        "state": state,
    }
    try:
        r = requests.get(TOKEN_API_URL, params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Naver OAuth token error: {e}")
        return None

def _fmt_ics(dt_iso: str) -> str:
    # 입력: "YYYY-MM-DDTHH:MM:SS"
    # 출력: "YYYYMMDDTHHMMSS"
    dt = datetime.fromisoformat(dt_iso)
    return dt.strftime("%Y%m%dT%H%M%S")

def _esc_ics(s: str | None) -> str:
    s = (s or "").replace("\\", "\\\\").replace("\n", "\\n").replace(",", "\\,").replace(";", "\\;")
    return s

def _build_vcalendar(summary: str, start_iso: str, end_iso: str,
                     location: str = "", description: str = "") -> str:
    uid = str(uuid.uuid4())
    now_utc_stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dtstart = _fmt_ics(start_iso)
    dtend   = _fmt_ics(end_iso)

    # TZ 지정이 필요한 경우 아래 줄처럼 TZID를 붙여도 된다. (네이버가 허용)
    # DTSTART;TZID=Asia/Seoul:YYYYMMDDTHHMMSS
    # 여기서는 로컬시간 그대로 사용.
    ics = "\r\n".join([
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//ChaeChae//Itinerary//KR",
        "CALSCALE:GREGORIAN",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{now_utc_stamp}",
        f"DTSTART:{dtstart}",
        f"DTEND:{dtend}",
        f"SUMMARY:{_esc_ics(summary)}",
        f"LOCATION:{_esc_ics(location)}",
        f"DESCRIPTION:{_esc_ics(description)}",
        "END:VEVENT",
        "END:VCALENDAR",
    ])
    return ics

def add_schedule(access_token: str, schedule: dict, calendar_id: str = "defaultCalendarId") -> tuple[bool, str]:
    """
    schedule dict 예:
    {
      "title": "[채채] 장소명",
      "startTime": "2025-09-20T10:00:00",
      "endTime":   "2025-09-20T11:30:00",
      "location": "주소",
      "description": "추천 여행 일정: ..."
    }
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
        ok = r.ok
        body = r.text
        if not ok:
            print(f"❌ Naver createSchedule error: {body}")
        return ok, body
    except requests.exceptions.RequestException as e:
        return False, str(e)
