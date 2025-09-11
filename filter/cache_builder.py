# cache_builder.py
import time
import re
from pathlib import Path
from typing import List
from datetime import datetime

import requests
import pandas as pd
import unicodedata as ud
from tqdm import tqdm

from recommend.config import (
    PATH_TMF,
    KAKAO_API_KEY,
    PATH_KAKAO_IMAGE_CACHE,
)

_IMAGE_CACHE: dict[str, dict] | None = None
_SESSION: requests.Session | None = None
DEFAULT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def _nfc(s: str) -> str:
    return ud.normalize("NFC", str(s or "")).strip()

def _load_image_cache() -> dict:
    global _IMAGE_CACHE
    if _IMAGE_CACHE is not None:
        return _IMAGE_CACHE
    p = Path(PATH_KAKAO_IMAGE_CACHE)
    if not p.exists():
        _IMAGE_CACHE = {}
        return _IMAGE_CACHE
    try:
        import json
        _IMAGE_CACHE = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        _IMAGE_CACHE = {}
    return _IMAGE_CACHE

def _save_image_cache():
    global _IMAGE_CACHE
    if _IMAGE_CACHE is None: return
    try:
        import json
        p = Path(PATH_KAKAO_IMAGE_CACHE)
        p.write_text(json.dumps(_IMAGE_CACHE, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"❌ 에러: 이미지 캐시 파일 저장 실패 - {e}")

def _ensure_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({"User-Agent": DEFAULT_UA})
    if KAKAO_API_KEY and "Authorization" not in _SESSION.headers:
        _SESSION.headers.update({"Authorization": f"KakaoAK {KAKAO_API_KEY}"})

def _addr_region_tokens(addr1: str) -> List[str]:
    cand = re.findall(r"\b[\w가-힣]+(?:시|군|구)\b", _nfc(addr1)) or re.split(r"[,\s]+", _nfc(addr1))
    return [w for w in cand if w][:3]

def _kakao_image_search(query: str, size: int = 4) -> List[str]:
    if not KAKAO_API_KEY: return []
    _ensure_session()
    try:
        params = {"query": query, "sort": "accuracy", "page": 1, "size": max(1, min(10, int(size)))}
        r = _SESSION.get("https://dapi.kakao.com/v2/search/image", params=params, timeout=4)
        if not r.ok: return []
        docs = r.json().get("documents", []) or []
        urls = [d.get("image_url") for d in docs if str(d.get("image_url") or "").startswith("http")]
        return [u for u in urls if len(u) < 2000]
    except Exception:
        return []

def _fetch_and_cache_images_live(title: str, addr1: str) -> list[str]:
    key = f"{_nfc(title)}|{_nfc(addr1)}"
    query = " ".join([title, *_addr_region_tokens(addr1)])
    urls = _kakao_image_search(query, size=4)
    cache = _load_image_cache()
    cache[key] = { "q": query, "urls": urls, "ts": int(datetime.now().timestamp()) }
    return urls

# ▼▼▼ [수정] 이 파일을 직접 실행하지 않고 app.py에서 호출할 함수로 변경 ▼▼▼
def update_cache_if_needed():
    """
    백그라운드에서 실행될 캐시 업데이트 함수.
    """
    print("--- 🖼️  백그라운드 이미지 캐시 업데이트 확인 시작 ---")
    if not KAKAO_API_KEY:
        print("⛔️ KAKAO_API_KEY가 없어 캐싱을 건너뜁니다.")
        return

    try:
        df = pd.read_csv(PATH_TMF, usecols=["title", "addr1"], encoding='utf-8')
        df.dropna(inplace=True)
    except Exception as e:
        print(f"⛔️ CSV 파일('{PATH_TMF}') 로드 실패: {e}")
        return

    cache = _load_image_cache()
    print(f"✅ 현재 캐시 크기: {len(cache):,}개")

    new_items_to_fetch = []
    
    # CSV의 모든 항목을 순회하며 캐시에 없는 항목 찾기
    for _, row in df.iterrows():
        title, addr1 = _nfc(row.get("title")), _nfc(row.get("addr1"))
        if not title or not addr1:
            continue
        key = f"{title}|{addr1}"
        if key not in cache:
            new_items_to_fetch.append({"title": title, "addr1": addr1})

    if not new_items_to_fetch:
        print("✨ 새로운 항목 없음. 캐시가 최신 상태입니다.")
        return

    print(f"🚚 총 {len(new_items_to_fetch):,}개의 새로운 장소 이미지를 가져옵니다...")
    
    save_interval = 50
    for i, item in enumerate(new_items_to_fetch):
        title, addr1 = item["title"], item["addr1"]
        print(f"  -> [{i+1}/{len(new_items_to_fetch)}] '{title}' 이미지 검색 중...")
        
        _fetch_and_cache_images_live(title, addr1)
        time.sleep(0.1) # API 속도 제한 준수를 위해 약간의 딜레이 추가

        if (i + 1) % save_interval == 0:
            _save_image_cache()
            print(f"💾 중간 저장 완료 ({i+1}개 처리)")

    _save_image_cache()
    print(f"\n✅ 캐시 업데이트 완료! {len(new_items_to_fetch)}개 항목 추가. 최종 캐시 크기: {len(_load_image_cache()):,}개.")

# 직접 실행했을 때의 동작은 그대로 유지 (테스트용)
if __name__ == "__main__":
    update_cache_if_needed()