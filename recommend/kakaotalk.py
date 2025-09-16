# recommend/kakaotalk.py
# -*- coding: utf-8 -*-
import requests
import json
import time
from flask import url_for
from recommend.config import KAKAO_API_KEY
from itertools import groupby
import math

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

def _create_text_for_one_day(day_items: list, chunk_num: int = 1, total_chunks: int = 1) -> str:
    """Helper function to create a text message for a single day's itinerary chunk."""
    if not day_items:
        return ""
    
    day_label = day_items[0].get('day_label', f"{day_items[0]['day']}일차")
    
    if total_chunks > 1:
        day_label += f" ({chunk_num}/{total_chunks})"
        
    message_parts = [f"✈️ {day_label} 추천 경로"]

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
    
    full_text = "\n".join(message_parts)
    if len(full_text) > 197:
        return full_text[:197] + "..."
    return full_text

def send_message_to_me(access_token: str, itinerary: list, chat_url: str) -> bool:
    """
    Sends multiple messages using the default template (link is required).
    Long itineraries are split into multiple messages.
    """
    send_url = 'https://kapi.kakao.com/v2/api/talk/memo/default/send'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8',
    }
    
    all_success = True
    itinerary_sorted = sorted(itinerary, key=lambda x: x['day'])
    
    for day, items_of_day_iter in groupby(itinerary_sorted, key=lambda x: x['day']):
        day_items = list(items_of_day_iter)
        
        MAX_ITEMS_PER_MESSAGE = 5 
        total_items = len(day_items)
        total_chunks = math.ceil(total_items / MAX_ITEMS_PER_MESSAGE)
        
        item_chunks = [
            day_items[i:i + MAX_ITEMS_PER_MESSAGE]
            for i in range(0, total_items, MAX_ITEMS_PER_MESSAGE)
        ]

        for i, chunk in enumerate(item_chunks):
            daily_text = _create_text_for_one_day(chunk, chunk_num=i + 1, total_chunks=total_chunks)
            if not daily_text:
                continue

            template_object = {
                "object_type": "text",
                "text": daily_text,
                "link": {
                    "web_url": chat_url,
                    "mobile_web_url": chat_url
                }
            }
            
            payload = {'template_object': json.dumps(template_object, ensure_ascii=False)}
            
            try:
                response = requests.post(send_url, headers=headers, data=payload, timeout=5)
                if response.status_code != 200 or response.json().get("result_code", 0) != 0:
                    all_success = False
                    print(f"❌ Kakao Send Message Error (Day {day}, Chunk {i+1}): {response.text}")
                time.sleep(0.3)
            except requests.exceptions.RequestException as e:
                all_success = False
                print(f"❌ Kakao Send Message Exception (Day {day}, Chunk {i+1}): {e}")

    return all_success