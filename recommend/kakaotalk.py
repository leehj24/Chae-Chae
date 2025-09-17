# recommend/kakaotalk.py
# -*- coding: utf-8 -*-
import requests
import json
import time
from flask import url_for
from recommend.config import KAKAO_API_KEY
from itertools import groupby

def get_access_token(code: str) -> str | None:
    """Exchanges an authorization code for an access token."""
    token_url = 'https://kauth.kakao.com/oauth/token'
    redirect_uri = url_for('kakao_oauth_callback', _external=True)
    
    data = {
        'grant_type': 'authorization_code',
        'client_id': KAKAO_API_KEY,
        'redirect_uri': redirect_uri,
        'code': code,
    }
    try:
        response = requests.post(token_url, data=data, timeout=5)
        response.raise_for_status()
        return response.json().get('access_token')
    except requests.exceptions.RequestException as e:
        print(f"❌ Kakao Token Error: {e}")
        return None

def send_message_to_me(access_token: str, itinerary: list, chat_url: str) -> bool:
    """
    Sends a single message with the full itinerary.
    If the itinerary is too long, it sends a summary message with a link.
    """
    send_url = 'https://kapi.kakao.com/v2/api/talk/memo/default/send'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8',
    }
    
    # 전체 일정에 대한 텍스트를 하나의 문자열로 생성합니다.
    message_parts = []
    itinerary_sorted = sorted(itinerary, key=lambda x: x['day'])
    for day, items_of_day_iter in groupby(itinerary_sorted, key=lambda x: x['day']):
        day_items = list(items_of_day_iter)
        day_label = day_items[0].get('day_label', f"{day}일차")
        message_parts.append(f"\n✈️ {day_label} 추천 경로")

        for item in day_items:
            start_time = item.get('start_time', '')
            title = item.get('title', '알 수 없는 활동')
            
            if title == "이동":
                departure = item.get('출발지', '')
                arrival = item.get('도착지', '')
                if departure and arrival:
                     message_parts.append(f"• {start_time}~ | 🚶 이동: {departure} → {arrival}")
                else:
                     message_parts.append(f"• {start_time}~ | 🚶 이동")
            else:
                message_parts.append(f"• {start_time}~ | {title}")

    full_text = "\n".join(message_parts).strip()

    # 카카오톡 텍스트 길이 제한(200자)을 확인하여 메시지 내용을 결정합니다.
    MAX_TEXT_LENGTH = 195  # 여유 있게 195자로 설정
    if len(full_text) > MAX_TEXT_LENGTH:
        display_text = "🗓️ 추천 여행 일정이 도착했어요!\n아래 버튼을 눌러 전체 경로를 확인해 보세요."
    else:
        # 일정이 길지 않으면 전체 내용을 그대로 보여줍니다.
        display_text = full_text if full_text else "추천 일정이 생성되었습니다. 자세한 내용은 웹에서 확인해주세요."

    template_object = {
        "object_type": "text",
        "text": display_text,
        "link": {
            "web_url": chat_url,
            "mobile_web_url": chat_url
        },
        "button_title": "일정 확인하기"
    }
    
    payload = {'template_object': json.dumps(template_object, ensure_ascii=False)}
    
    try:
        response = requests.post(send_url, headers=headers, data=payload, timeout=5)
        if response.status_code != 200 or response.json().get("result_code", 0) != 0:
            print(f"❌ Kakao Send Message Error: {response.text}")
            return False
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Kakao Send Message Exception: {e}")
        return False