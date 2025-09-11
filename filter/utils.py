# filter/utils.py
import pandas as pd
import unicodedata as ud
from pathlib import Path
import re  # <--- [수정] re 모듈 임포트
import numpy as np # <--- [수정] numpy 모듈 임포트

# --- 설정 ---
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

_places_df = None

def _nfc(s: str) -> str:
    return ud.normalize("NFC", str(s or "")).strip()

def _load_data():
    global _places_df
    if _places_df is not None:
        return _places_df
    
    ROOT = Path(__file__).resolve().parents[1]
    PATH_TMF = str(ROOT / "관광지_법정동_매핑결과.csv")
    
    df = pd.read_csv(PATH_TMF, encoding='utf-8')
    df.columns = [c.lower() for c in df.columns]
    
    def extract_sido(addr):
        addr_nfc = _nfc(addr)
        for key, value in sido_map.items():
            if addr_nfc.startswith(key):
                if '강원' in value: return '강원도'
                if '제주' in value: return '제주특별자치도'
                return value
        return None
        
    df['sido'] = df['addr1'].apply(extract_sido)
    
    df['cat3_list'] = df['cat3'].astype(str).apply(
        lambda x: sorted([_nfc(tag) for tag in re.split(r'[,/|·]', x) if _nfc(tag)])
    )
    
    _places_df = df.copy()
    return _places_df

def get_filter_options():
    df = _load_data()
    sidos = sorted(df['sido'].dropna().unique().tolist())
    cat1s = sorted(df['cat1'].dropna().unique().tolist())
    
    all_cat3s = set()
    for cat3_list in df['cat3_list']:
        all_cat3s.update(cat3_list)
    cat3s = sorted([c for c in list(all_cat3s) if c and c.lower() != 'nan'])
    
    return {"sidos": sidos, "cat1s": cat1s, "cat3s": cat3s}

def get_filtered_places(sido=None, cat1=None, cat3=None, query=None):
    df = _load_data()
    filtered_df = df.copy()
    
    if sido and sido != 'all':
        sido_val = sido_map.get(sido, sido)
        if '강원' in sido_val:
            filtered_df = filtered_df[filtered_df['sido'] == '강원도']
        elif '제주' in sido_val:
            filtered_df = filtered_df[filtered_df['sido'] == '제주특별자치도']
        else:
            filtered_df = filtered_df[filtered_df['sido'] == sido_val]

    if cat1 and cat1 != 'all':
        filtered_df = filtered_df[filtered_df['cat1'] == cat1]
        
    if cat3 and cat3 != 'all':
        filtered_df = filtered_df[filtered_df['cat3_list'].apply(lambda x: cat3 in x)]

    if query:
        query_nfc = _nfc(query).lower()
        filtered_df = filtered_df[filtered_df['title'].str.lower().contains(query_nfc, na=False)]
        
    return filtered_df.replace({pd.NA: None, np.nan: None}).to_dict(orient='records')