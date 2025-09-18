# -*- coding: utf-8 -*-
import pandas as pd
import re


# Paths
places_path = "관광지_법정동_매핑결과.csv"
pop_path = "전국_인기도지수.csv"
out_path = "관광지_법정동_매핑결과.csv"

# Load
places = pd.read_csv(places_path)
pop = pd.read_csv(pop_path)

# Normalize columns
places.columns = [c.strip() for c in places.columns]
pop.columns = [c.strip() for c in pop.columns]

# sido map
sido_map = {
    '서울': '서울특별시', '서울특별시': '서울특별시', '서울시': '서울특별시',
    '부산': '부산광역시', '부산광역시': '부산광역시',
    '대구': '대구광역시', '대구광역시': '대구광역시',
    '인천': '인천광역시', '인천광역시': '인천광역시',
    '광주': '광주광역시', '광주광역시': '광주광역시',
    '대전': '대전광역시', '대전광역시': '대전광역시',
    '울산': '울산광역시', '울산광역시': '울산광역시', '울산시': '울산광역시',
    '세종': '세종특별자치시', '세종특별자치시': '세종특별자치시',
    '경기': '경기도', '경기도': '경기도',
    '강원': '강원', '강원도': '강원', '강원특별자치도': '강원',
    '충남': '충청남도', '충청남도': '충청남도',
    '충북': '충청북도', '충청북도': '충청북도',
    '전남': '전라남도', '전라남도': '전라남도',
    '전북': '전라북도', '전라북도': '전라북도', '전북특별자치도': '전라북도',
    '경남': '경상남도', '경상남도': '경상남도',
    '경북': '경상북도', '경상북도': '경상북도',
    '제주': '제주', '제주도': '제주', '제주특별자치도': '제주',
}

# Helpers
region_pattern = re.compile(
    r'(서울특별시|서울시|서울|부산광역시|부산|대구광역시|대구|인천광역시|인천|광주광역시|광주|대전광역시|대전|울산광역시|울산시|울산|세종특별자치시|세종|'
    r'경기도|경기|강원특별자치도|강원도|강원|충청남도|충남|충청북도|충북|전라남도|전남|전라북도|전북특별자치도|전북|경상남도|경남|경상북도|경북|제주특별자치도|제주도|제주)'
)
def extract_sido(addr: str) -> str:
    if not isinstance(addr, str) or not addr.strip():
        return ''
    m = region_pattern.search(addr)
    return m.group(1) if m else addr.split()[0]

def normalize_sido(x: str) -> str:
    x = (x or '').strip()
    return sido_map.get(x, x)

# Prepare normalized keys
places['norm_sido'] = places['addr1'].apply(extract_sido).apply(normalize_sido)
pop['norm_region'] = pop['region'].astype(str).str.strip().apply(normalize_sido)

# Region-only strict match
merged = places.merge(
    pop[['store_name','norm_region','review_score']].rename(columns={'review_score':'pop_review_score'}),
    left_on=['title','norm_sido'],
    right_on=['store_name','norm_region'],
    how='left'
)

# Drop old review_score if present, and write into review_socre (exact spelling as requested)
if 'review_score' in merged.columns:
    merged = merged.drop(columns=['review_score'])

merged['review_socre'] = merged['pop_review_score']

# Remove helper columns from output
drop_cols = ['store_name','norm_region','pop_review_score']
output_cols = [c for c in merged.columns if c not in drop_cols]

merged[output_cols].to_csv(out_path, index=False, encoding='utf-8-sig')

# Show quick coverage
coverage = pd.DataFrame({
    '총_관광지_수': [len(places)],
    'region기준_매칭_성공(재작성 review_socre)': [merged['review_socre'].notna().sum()],
    'region기준_매칭_실패': [merged['review_socre'].isna().sum()],
})


out_path
