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
FAST_MODE = True # True: 속도 우선, False: 정확도 우선
TRANSIT_TOP_N = 60 # 대중교통 모드에서 상위 N개 관광지만 사용 (속도 향상용)
TRANSIT_RADIUS_KM = 15 # 대중교통 반경 (km)
USE_ENRICH_TRANSIT_HINTS = True # 대중교통 모드에서 힌트 데이터 사용 여부
USE_ODSay = True # True: 실제대중교통경로시간 Flase: 직선거리*1.4로 시간계산 

# ░░ Kakao API 키 ░░
# REST API 키 (서버 전용 — 절대 프론트 노출 금지)
KAKAO_API_KEY = os.environ.get("KAKAO_API_KEY", "") # 대중교통 매핑용 카카오 api
#대중교통 시간계산 api
ODSAY_API_KEY = os.environ.get("ODSAY_API_KEY", "")

###############################################
#UI용
# ✅ JavaScript 키 (프론트에서 Kakao Maps JS SDK 로드)_사진로드
KAKAO_JS_KEY = os.environ.get("KAKAO_JS_KEY", "")
# Kakao 이미지 URL 캐시 파일 경로 (이미지 파일 X, URL만 JSON으로 저장)
PATH_KAKAO_IMAGE_CACHE = str(ROOT / "_cache_kakao_images.json")
