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
    """요청 세션을 만듭니다."""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()

def get_kakao_place_by_coords(title: str, x: str, y: str) -> Optional[str]:
    """주어진 좌표와 장소 이름으로 검색하여 가장 정확한 장소의 URL을 반환합니다."""
    if not KAKAO_API_KEY:
        print("⛔️ KAKAO_API_KEY가 설정되지 않았습니다.")
        return None
        
    _ensure_session()
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    params = {
        "query": title,
        "x": x,
        "y": y,
        "radius": 200 # 오차를 고려해 반경을 200m로 설정
    }
    
    print(f"🔍 좌표 기반 검색: '{title}' (경도={x}, 위도={y})")
    
    try:
        response = _SESSION.get(url, headers=headers, params=params, timeout=3)
        if not response.ok:
            print(f"❌ 좌표 기반 검색 실패: {response.status_code}")
            return None
        
        docs = response.json().get("documents", [])
        if not docs:
            print("❌ 해당 좌표 근처에서 장소를 찾지 못했습니다.")
            return None
        
        # 검색 결과 중 이름이 가장 비슷한 장소를 선택
        best_match = docs[0]
        clean_title = title.replace('_', ' ')
        for place in docs:
            if clean_title in place.get("place_name", ""):
                best_match = place
                break
        
        print(f"✅ 장소 찾기 성공: '{best_match.get('place_name')}'")
        return best_match.get("place_url")

    except requests.exceptions.RequestException as e:
        print(f"❌ 좌표 기반 검색 중 오류 발생: {e}")
        return None

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    # 사용자의 원본 데이터 파일에 있는 값을 시뮬레이션
    example_title = "장독대김치찜김치찌개_용산점"
    example_mapx = "126.9647566449"
    example_mapy = "37.5298249324"

    # 데이터의 좌표를 직접 사용하여 장소 URL 검색
    place_url = get_kakao_place_by_coords(example_title, example_mapx, example_mapy)

    print("-" * 30)
    if place_url:
        print(f"🎉 최종 성공! '{example_title}'의 URL을 찾았습니다:")
        print(f"  URL: {place_url}")
    else:
        print(f"❌ 최종 실패. '{example_title}'의 URL을 찾지 못했습니다.")