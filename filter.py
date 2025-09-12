sido_map = {
    '서울': '서울특별시', '서울시': '서울특별시', '서울특별시': '서울특별시',
    '부산': '부산광역시', '부산시': '부산광역시', '부산광역시': '부산광역시',
    '대구': '대구광역시', '대구시': '대구광역시', '대구광역시': '대구광역시',
    '인천': '인천광역시', '인천시': '인천광역시', '인천광역시': '인천광역시',
    '광주': '광주광역시', '광주시': '광주광역시', '광주광역시': '광주광역시',
    '대전': '대전광역시', '대전시': '대전광역시', '대전광역시': '대전광역시',
    '울산': '울산광역시', '울산시': '울산광역시', '울산광역시': '울산광역시',
    '세종': '세종특별자치시', '세종시': '세종특별자치시', '세종특별자치시': '세종특별자치시',
    '경기': '경기도', '경기도': '경기도',
    '강원': '강원특별자치도', '강원도': '강원특별자치도',
    '충남': '충청남도', '충청남도': '충청남도',
    '충북': '충청북도', '충청북도': '충청북도',
    '전남': '전라남도', '전라남도': '전라남도',
    '전북': '전라북도', '전라북도': '전라북도', '전북특별자치도': '전라북도',
    '경남': '경상남도', '경상남도': '경상남도',
    '경북': '경상북도', '경상북도': '경상북도',
    '제주': '제주특별자치도', '제주도': '제주특별자치도', '제주특별자치도': '제주특별자치도'
}


# 2. 특별히 수동으로 변환할 주소 3개에 대한 정답 맵
manual_address_map = {
    "119-11, Sansuhwa-ro, Hwacheon-gun, Gangwon-do": "강원특별자치도 화천군 산수화로 119-11",
    "13-29, Jukjeongseowon-gil, Yeongam-gun, Jeollanam-do": "전라남도 영암군 죽정서원길 13-29",
    "18, Hoegi-ro 29-gil, Dongdaemun-gu, Seoul": "서울특별시 동대문구 회기로29길 18"
}


def convert_address(addr):
    # 주소가 문자열이 아니면 그냥 반환
    if not isinstance(addr, str):
        return addr
    
    # 1. 먼저, 수동으로 변환할 주소인지 확인
    if addr in manual_address_map:
        return manual_address_map[addr] # 맞으면 정답으로 바로 바꿔서 반환
    
    # 2. 아니라면, 원래의 sido_map을 이용해 시/도 이름만 변환
    for key in sorted(sido_map.keys(), key=len, reverse=True):
        if key in addr and sido_map[key] not in addr:
            addr = addr.replace(key, sido_map[key], 1)
            break
            
    return addr

import pandas as pd


# ---[메인 코드]---
try:
    # CSV 파일 불러오기
    df = pd.read_csv('관광지_법정동_매핑결과.csv')

    # 새로 만든 변환 함수를 addr1 열에 적용
    df['addr1'] = df['addr1'].apply(convert_address)

    # 수정된 CSV 저장 (한글 깨짐 방지 encoding 추가)
    df.to_csv('관광지_법정동_매핑결과.csv', index=False, encoding='utf-8-sig')

    print("주소 변환이 완료되었습니다! '관광지_법정동_매핑결과.csv' 파일을 확인해주세요. 👍")

except FileNotFoundError:
    print("오류: '관광지_법정동_매핑결과.csv' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"알 수 없는 오류가 발생했습니다: {e}")