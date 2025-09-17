# recommend/kakaotalk.py
# -*- coding: utf-8 -*-
import requests
import json
from flask import url_for
from recommend.config import KAKAO_API_KEY
from itertools import groupby

def get_access_token(code: str) -> str | None:
    """Exchanges an authorization code for an access token."""
    token_url = 'https://kauth.kakao.com/oauth/token'
    # _external=True ensures the full URL is generated, which is required by Kakao.
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
    Sends a single message with a preview of the itinerary.
    The message text is truncated if it's too long, and a button is always included to see the full view.
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

    # 카카오톡 텍스트 길이 제한에 맞춰 미리보기를 생성합니다.
    MAX_TEXT_LENGTH = 190  # 200자 제한보다 여유있게 설정
    if len(full_text) > MAX_TEXT_LENGTH:
        # 긴 텍스트는 잘라서 "..."을 붙여줍니다.
        display_text = full_text[:MAX_TEXT_LENGTH] + "..."
    elif not full_text:
        display_text = "추천 일정이 생성되었습니다. 자세한 내용은 웹에서 확인해주세요."
    else:
        # 짧은 텍스트는 전체 내용을 보여줍니다.
        display_text = full_text

    # 항상 버튼과 함께 텍스트 템플릿을 사용합니다.
    template_object = {
        "object_type": "text",
        "text": display_text,
        "link": {
            "web_url": chat_url,
            "mobile_web_url": chat_url
        },
        "button_title": "전체 일정 확인하기"
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