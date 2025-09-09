# recommend/config.py
from pathlib import Path
import os

# .env 로드 (있으면)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# 프로젝트 루트: .../UI_MASTER
ROOT = Path(__file__).resolve().parents[1]

# ✅ CSV 절대경로 (OS 상관없이 안전)
PATH_TMF = str(ROOT / "관광지_법정동_매핑결과.csv")

# (선택) Flask 세션키
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")

# 속도 우선 모드 (추천 엔진용)
FAST_MODE = True
TRANSIT_TOP_N = 60
TRANSIT_RADIUS_KM = 15
USE_ENRICH_TRANSIT_HINTS = True
USE_ODSay = True

# ░░ Kakao API 키 ░░
# REST API 키 (서버 전용 — 절대 프론트 노출 금지)
KAKAO_API_KEY = os.environ.get("KAKAO_API_KEY", "")

# ✅ JavaScript 키 (프론트에서 Kakao Maps JS SDK 로드)
KAKAO_JS_KEY = os.environ.get("KAKAO_JS_KEY", "")

# Kakao 이미지 URL 캐시 파일 경로 (이미지 파일 X, URL만 JSON으로 저장)
PATH_KAKAO_IMAGE_CACHE = str(ROOT / "_cache_kakao_images.json")
