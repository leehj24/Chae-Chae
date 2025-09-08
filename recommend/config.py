from pathlib import Path
import os

# 프로젝트 루트: .../UI_MASTER
ROOT = Path(__file__).resolve().parents[1]

# ✅ CSV 절대경로 (OS 상관없이 안전)
PATH_TMF = str(ROOT / "관광지_법정동_매핑결과.csv")

# (선택) Flask 세션키를 여기서도 관리하고 싶으면
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")

# 속도 우선 모드 (True면 훨씬 빠름)
FAST_MODE = True    # 배포 시 기본 True 권장, 필요할 때만 False
TRANSIT_TOP_N = 60  # 후보 상위 N개만 상세 처리
TRANSIT_RADIUS_KM = 15  # 대중교통 반경 축소
USE_ENRICH_TRANSIT_HINTS = True  # 정류장/노선 힌트 조회 스킵
USE_ODSay = True  # ODsay API 조회 끔
