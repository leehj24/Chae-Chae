import os
import re
import requests
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("dotenv 라이브러리가 설치되지 않았습니다.")

# --- 설정 ---
KAKAO_API_KEY = os.environ.get("KAKAO_API_KEY")
_SESSION = None

def _ensure_session():
    """요청 세션을 만들고 헤더를 설정하는 함수"""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
    if KAKAO_API_KEY and "Authorization" not in _SESSION.headers:
        _SESSION.headers.update({"Authorization": f"KakaoAK {KAKAO_API_KEY}"})

def get_kakao_place_url_by_title(title: str, original_addr1: str) -> Optional[str]:
    """
    장소 이름으로 검색하고, 결과 목록에서 원래 주소와 가장 유사한 장소를 찾아 URL을 반환합니다.
    """
    if not KAKAO_API_KEY:
        print("⛔️ KAKAO_API_KEY가 설정되지 않았습니다.")
        return None
    
    _ensure_session()
    
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    params = {"query": title, "size": 5} # 넉넉하게 5개 결과 요청
    
    print(f"🔍 '{title}'(으)로 카카오맵 API에 검색을 요청합니다...")

    try:
        response = _SESSION.get(url, params=params, timeout=3)
        if not response.ok:
            print(f"❌ API 요청 실패: {response.status_code}")
            return None

        docs = response.json().get("documents", [])
        if not docs:
            print("❌ API 검색 결과가 없습니다.")
            return None

        print(f"✅ {len(docs)}개의 검색 결과를 찾았습니다. 이제 주소를 비교합니다.")

        # 주소의 핵심 부분만 추출 (숫자, 띄어쓰기 제거)
        def simplify_address(addr):
            return re.sub(r'[\d\s-]', '', addr)

        simplified_original_addr = simplify_address(original_addr1)

        for place in docs:
            api_addr = place.get("address_name", "")
            simplified_api_addr = simplify_address(api_addr)
            
            # API 결과 주소에 원래 주소의 핵심 부분이 포함되어 있는지 확인
            if simplified_original_addr in simplified_api_addr:
                print(f"✅ 일치하는 장소를 찾았습니다: {place.get('place_name')} ({api_addr})")
                return place.get("place_url")

        # 만약 정확히 일치하는 주소가 없다면, 첫 번째 결과를 반환 (차선책)
        print("⚠️ 정확히 일치하는 주소가 없어 첫 번째 결과를 반환합니다.")
        return docs[0].get("place_url")

    except requests.exceptions.RequestException as e:
        print(f"❌ API 검색 중 오류 발생: {e}")
        
    return None

# --- 예시 실행 ---
if __name__ == "__main__":
    # 1. 예시 데이터 (CSV 파일에 있는 정보)
    example_title = "모담"
    example_addr1 = "경기 고양시 일산서구 호수로 817"

    # 2. 함수 호출
    place_url = get_kakao_place_url_by_title(example_title, example_addr1)

    # 3. 결과 출력
    print("-" * 30)
    if place_url:
        print(f"🎉 최종 성공! '{example_title}'의 카카오맵 URL을 찾았습니다:")
        print(f"   URL: {place_url}")
    else:
        print(f"❌ 최종 실패. '{example_title}'의 URL을 찾지 못했습니다.")