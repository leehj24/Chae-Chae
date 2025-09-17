# recommend/naver_calendar.py

import requests
import json
import uuid
from urllib.parse import urlencode
from recommend.config import *
from datetime import datetime, timezone

# 네이버 API 엔드포인트
TOKEN_API_URL = "https://nid.naver.com/oauth2.0/token"
CALENDAR_API_URL = "https://openapi.naver.com/v1/calendar/schedules"

def get_access_token(code: str, state: str) -> dict | None:
    """인증 코드로 액세스 토큰을 발급받습니다."""
    params = {
        "grant_type": "authorization_code",
        "client_id": NAVER_CLIENT_ID,
        "client_secret": NAVER_CLIENT_SECRET,
        "code": code,
        "state": state,
    }
    try:
        res = requests.get(TOKEN_API_URL, params=params)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ 네이버 토큰 발급 실패: {e}")
        return None

def add_schedule(access_token: str, schedule_data: dict) -> bool:
    """액세스 토큰을 이용해 네이버 캘린더에 일정을 추가합니다."""
    
    headers = {
        "Authorization": f"Bearer {access_token}",
    }
    
    try:
        start_dt = datetime.fromisoformat(schedule_data.get("startTime"))
        end_dt = datetime.fromisoformat(schedule_data.get("endTime"))
        start_time_utc = start_dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time_utc = end_dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    except (ValueError, TypeError):
        print(f"❌ 시간 형식 변환 실패: startTime={schedule_data.get('startTime')}, endTime={schedule_data.get('endTime')}")
        return False

    # [수정] API 명세에 맞게 startTime과 endTime을 {"time": "..."} 객체 형태로 변경
    schedule_json_content = {
        "categoryId": "0",
        "title": schedule_data.get("title", ""),
        "startTime": {"time": start_time_utc},
        "endTime": {"time": end_time_utc},
        "location": schedule_data.get("location", ""),
        "content": schedule_data.get("description", ""),
        "isAllDay": False,
        "isRepeat": False,
    }
    
    # API 명세에 맞게 최상위 레벨에 uid, calendarId, schedule(json 문자열)을 배치
    form_data = {
        'calendarId': 'primary',
        'uid': str(uuid.uuid4()),
        'schedule': json.dumps(schedule_json_content, ensure_ascii=False)
    }

    try:
        res = requests.post(CALENDAR_API_URL, headers=headers, data=form_data)
        res.raise_for_status()
        
        response_json = res.json()
        if response_json.get("result") == "success" and response_json.get("scheduleId"):
            print(f"✅ 네이버 캘린더 일정 추가 성공: {response_json}")
            return True
        else:
            print(f"❌ 네이버 서버 응답 오류: {response_json}")
            return False

    except requests.exceptions.RequestException as e:
        error_text = res.text if 'res' in locals() else str(e)
        print(f"❌ 네이버 캘린더 API 호출 실패: {error_text}")
        return False