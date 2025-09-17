# recommend/naver_calendar.py
import requests
import json
import uuid
from urllib.parse import urlencode, quote_plus
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
    
    # 1. 네이버 캘린더 API의 공식 요구사항에 맞게 페이로드 구조를 최종 수정합니다.
    #    - 시간 값을 RFC3339 UTC 형식(YYYY-MM-DDTHH:MM:SSZ)으로 변환합니다.
    #    - 필수 값인 'categoryId', 'isAllDay', 'isRepeat' 등을 추가합니다.
    try:
        # 시간 문자열을 datetime 객체로 변환
        start_dt = datetime.fromisoformat(schedule_data.get("startTime"))
        end_dt = datetime.fromisoformat(schedule_data.get("endTime"))

        # UTC 시간으로 변환 후 'Z'를 붙여 RFC3339 형식으로 포맷팅
        start_time_utc = start_dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time_utc = end_dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    except (ValueError, TypeError):
        print(f"❌ 시간 형식 변환 실패: startTime={schedule_data.get('startTime')}, endTime={schedule_data.get('endTime')}")
        return False

    schedule_payload = {
        "calendarId": "primary",
        "categoryId": "0", # "일반" 카테고리
        "uid": str(uuid.uuid4()),
        "title": schedule_data.get("title"),
        "startTime": start_time_utc,
        "endTime": end_time_utc,
        "location": schedule_data.get("location"),
        "content": schedule_data.get("description"),
        "isAllDay": False,
        "isRepeat": False,
    }
    
    form_data = {
        'schedule': quote_plus(json.dumps(schedule_payload, ensure_ascii=False))
    }

    try:
        res = requests.post(CALENDAR_API_URL, headers=headers, data=form_data)
        res.raise_for_status()
        
        response_json = res.json()
        if response_json.get("result") == "success" and response_json.get("scheduleId"):
            print(f"✅ 네이버 캘린더 일정 추가 최종 성공: {response_json}")
            return True
        else:
            print(f"❌ 응답은 성공했으나, 일정 추가 실패: {response_json}")
            return False

    except requests.exceptions.RequestException as e:
        error_text = res.text if 'res' in locals() else str(e)
        print(f"❌ 네이버 캘린더 API 호출 실패: {error_text}")
        return False
