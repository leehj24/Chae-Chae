# cache_builder.py
import time
import re
from pathlib import Path
from typing import List, Dict
from datetime import datetime

import requests
import pandas as pd
import unicodedata as ud
from tqdm import tqdm

# app.py와 동일한 설정 및 헬퍼 함수들을 가져옵니다.
from recommend.config import (
    PATH_TMF,
    KAKAO_API_KEY,
    PATH_KAKAO_IMAGE_CACHE,
)

# ----------------------------------------------------
# app.py에서 가져온 헬퍼 함수들 (API 호출 및 데이터 처리용)
# ----------------------------------------------------
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

# ----------------------------------------------------
# 메인 캐시 빌더 함수
# ----------------------------------------------------
def build_cache():
    print("--- 🖼️  이미지 캐시 빌드를 시작합니다 ---")
    if not KAKAO_API_KEY:
        print("⛔️ KAKAO_API_KEY가 설정되지 않아 캐싱을 중단합니다. .env 파일을 확인해주세요.")
        return

    try:
        df = pd.read_csv(PATH_TMF, encoding='utf-8')
        df = df[["title", "addr1"]].copy()
        print(f"✅ 원본 CSV 로드 완료. 고유 장소 {len(df):,}개.")
    except Exception as e:
        print(f"⛔️ CSV 파일('{PATH_TMF}') 로드 실패: {e}")
        return

    cache = _load_image_cache()
    print(f"✅ 기존 캐시 로드 완료. {len(cache):,}개 항목 존재.")

    new_items_to_fetch = []
    for _, row in df.iterrows():
        title, addr1 = _nfc(row["title"]), _nfc(row["addr1"])
        key = f"{title}|{addr1}"
        if key not in cache:
            new_items_to_fetch.append({"key": key, "title": title, "addr1": addr1})

    if not new_items_to_fetch:
        print("✨ 모든 장소의 이미지가 이미 캐시되어 있습니다. 동기화 완료!")
        return

    print(f"🚚 총 {len(new_items_to_fetch):,}개의 새로운 장소 이미지를 가져옵니다...")
    
    save_interval = 50
    with tqdm(total=len(new_items_to_fetch), desc="이미지 검색 중") as pbar:
        for i, item in enumerate(new_items_to_fetch):
            key, title, addr1 = item["key"], item["title"], item["addr1"]
            pbar.set_description(f"'{title[:10]}...' 검색")
            
            _fetch_and_cache_images_live(title, addr1)
            time.sleep(0.05) 

            if (i + 1) % save_interval == 0:
                _save_image_cache()
                pbar.set_postfix_str("💾 중간 저장 완료")
            
            pbar.update(1)

    _save_image_cache()
    print(f"\n✅ 캐시 빌드 완료! {len(new_items_to_fetch)}개 항목 추가. 최종 캐시 크기: {len(_load_image_cache()):,}개.")

if __name__ == "__main__":
    build_cache()