# precache_images.py

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import requests
import unicodedata as ud
from tqdm import tqdm  # 진행률 표시를 위한 라이브러리

# --- 설정 로드 ---
# app.py와 동일한 설정값을 사용합니다.
from recommend.config import KAKAO_API_KEY, PATH_TMF, PATH_KAKAO_IMAGE_CACHE

# --- app.py에서 가져온 헬퍼 함수들 ---

def _nfc(s: str) -> str:
    """문자열을 NFC 형식으로 정규화하고 양쪽 공백을 제거합니다."""
    return ud.normalize("NFC", str(s or "")).strip()

def _read_csv_robust(path: str) -> pd.DataFrame:
    """다양한 인코딩으로 CSV 파일을 안전하게 읽습니다."""
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    # 모든 인코딩 실패 시 예외 발생
    raise IOError(f"Failed to read CSV file with common encodings: {path}")

def _addr_region_tokens(addr1: str) -> List[str]:
    """주소에서 지역 토큰(시, 군, 구 등)을 추출합니다."""
    import re
    cand = re.findall(r"\b[\w가-힣]+(?:시|군|구)\b", _nfc(addr1)) or re.split(r"[,\s]+", _nfc(addr1))
    return [w for w in cand if w][:3]

# --- API 호출 및 캐싱 로직 ---

# API 호출을 위한 세션 객체 (연결 재사용으로 성능 향상)
_SESSION = requests.Session()
if KAKAO_API_KEY:
    _SESSION.headers.update({"Authorization": f"KakaoAK {KAKAO_API_KEY}"})

def kakao_image_search(query: str, size: int = 4) -> List[str]:
    """주어진 쿼리로 카카오 이미지 검색 API를 호출합니다."""
    if not KAKAO_API_KEY:
        print("⚠️ KAKAO_API_KEY가 설정되지 않았습니다. API 호출을 건너뜁니다.")
        return []
    try:
        params = {"query": query, "sort": "accuracy", "page": 1, "size": max(1, min(10, int(size)))}
        r = _SESSION.get("https://dapi.kakao.com/v2/search/image", params=params, timeout=5)
        
        if not r.ok:
            print(f"❌ API 오류: {r.status_code} - {r.text}")
            return []
            
        docs = r.json().get("documents", []) or []
        urls = [d.get("image_url") for d in docs if str(d.get("image_url") or "").startswith("http")]
        return [u for u in urls if len(u) < 2000]
    except requests.exceptions.RequestException as e:
        print(f"❌ 네트워크 오류 발생: {e}")
        return []

def run_precache():
    """
    메인 캐싱 스크립트.
    CSV를 읽고 기존 캐시와 비교하여 없는 항목만 API를 통해 가져와 저장합니다.
    """
    print("--- 이미지 캐시 사전 생성 스크립트를 시작합니다 ---")
    
    if not KAKAO_API_KEY:
        print("⛔️ 카카오 API 키가 없습니다. 스크립트를 종료합니다.")
        return

    # 1. 원본 CSV 데이터 로드
    try:
        df = _read_csv_robust(PATH_TMF)
        # app.py와 동일하게 필수 컬럼만 선택
        df = df.rename(columns={
            "title": "title",
            "addr1": "addr1",
        })
        df = df[["title", "addr1"]].copy()
        df.dropna(subset=["title", "addr1"], inplace=True)
        df.drop_duplicates(subset=["title", "addr1"], inplace=True)
        print(f"✅ 원본 CSV 로드 완료. 고유 장소 {len(df):,}개 발견.")
    except Exception as e:
        print(f"⛔️ CSV 파일('{PATH_TMF}') 로드 실패: {e}")
        return

    # 2. 기존 캐시 파일 로드
    cache_path = Path(PATH_KAKAO_IMAGE_CACHE)
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"✅ 기존 캐시('{cache_path.name}') 로드 완료. {len(cache):,}개 항목 존재.")
        except json.JSONDecodeError:
            print(f"⚠️ 기존 캐시 파일이 손상되었습니다. 새 캐시를 생성합니다.")
    else:
        print("ℹ️ 기존 캐시 파일이 없습니다. 새로 생성합니다.")

    # 3. 새롭게 추가할 항목 찾기 및 API 호출
    new_items_count = 0
    save_interval = 50  # 50개 항목마다 저장하여 안정성 확보

    # tqdm을 사용하여 진행률 바 표시
    pbar = tqdm(df.itertuples(), total=len(df), desc="🔍 캐시 항목 확인 중")
    
    for row in pbar:
        title = _nfc(row.title)
        addr1 = _nfc(row.addr1)
        key = f"{title}|{addr1}"

        if key in cache:
            continue  # 이미 캐시에 있으면 건너뛰기

        # 캐시에 없는 경우, API 호출
        pbar.set_description(f"🚚 '{title[:10]}...' 이미지 가져오는 중")
        
        query = " ".join([title, *_addr_region_tokens(addr1)])
        urls = kakao_image_search(query, size=4)
        
        cache[key] = {
            "q": query,
            "urls": urls,
            "ts": int(datetime.now().timestamp())
        }
        new_items_count += 1

        # API 과호출 방지를 위한 약간의 딜레이
        time.sleep(0.1)

        # 주기적으로 저장
        if new_items_count > 0 and new_items_count % save_interval == 0:
            pbar.set_description(f"💾 {new_items_count}개 추가 후 저장 중...")
            try:
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:
                print(f"\n❌ 캐시 파일 저장 중 오류 발생: {e}")

    # 4. 최종 저장
    print("\n--- 작업 완료 ---")
    if new_items_count > 0:
        print(f"✅ 총 {new_items_count}개의 새로운 장소 이미지를 캐시에 추가했습니다.")
        try:
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✅ 최종 캐시 파일('{cache_path.name}') 저장 완료! (총 {len(cache):,}개 항목)")
        except Exception as e:
            print(f"❌ 최종 캐시 파일 저장 실패: {e}")
    else:
        print("✨ 모든 장소가 이미 캐시되어 있습니다. 추가할 항목이 없습니다.")

if __name__ == "__main__":
    # tqdm 라이브러리가 없다면 설치 안내
    try:
        from tqdm import tqdm
    except ImportError:
        print("="*50)
        print("이 스크립트를 실행하려면 'tqdm' 라이브러리가 필요합니다.")
        print("터미널에서 아래 명령어를 실행하여 설치해주세요.")
        print("pip install tqdm")
        print("="*50)
    else:
        run_precache()