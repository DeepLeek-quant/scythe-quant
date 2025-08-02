from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import requests
import zipfile
import tqdm
import shutil
import io
import re
import os

from .data import DataKit

# utils
def roc_to_ad(date_str:str):
    if (pd.isna(date_str)) or (date_str in [' ', '']): 
        return np.nan
    else:
        s = str(int(date_str.replace('/', ''))).zfill(7)
        return f"{int(s[:3])+1911}{s[3:]}"


# 證期局_受理申報案件
def download_and_unzip(url, extract_to='證期局_受理申報案件', exclude_file_type='.ods'):
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.makedirs(extract_to, exist_ok=True)
    resp = requests.get(url)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        for info in tqdm.tqdm(z.infolist(), desc="Extracting files"):
            try:
                filename = info.filename.encode('cp437').decode('big5')
            except Exception:
                filename = info.filename
            if not info.is_dir() and (exclude_file_type is None or not filename.endswith(exclude_file_type)):
                with z.open(info) as source, open(os.path.join(extract_to, filename), "wb") as target:
                    target.write(source.read())

def get_historical_data_year_path(path):
    xlsx_files = [f for f in os.listdir(path) if f.endswith('.xlsx') or f.endswith('.xls')]
    year_path = {}
    for file in xlsx_files:
        
        if file.startswith('~$'):
            continue
        file_name = file.replace('case', '').replace('.xlsx', '').replace('.xls', '').replace('-', '').replace('申報案件彙總表 ', '').split('(')[0]
        if len(file_name) <= 3:
            year = int(file_name)+1911
        elif len(file_name) == 5:
            year = int(file_name[:3])+1911
        elif len(file_name) >= 6:
            year = int(file_name[:3])+1911
        else:
            year = int(file_name)
        if year >=2005:
            year_path[year] = os.path.join(path, file)

    return dict(sorted(year_path.items()))

def process_corp_submit(path:str, year:int) -> pd.DataFrame:
    replace_value_dict = {
        '公司型態': {
            1: "上市",
            2: "上櫃",
            3: "未上市(櫃)",
            4: "補辦公開發行",
            5: "興櫃",
            6: "外國企業"
        },
        '結案型態': {
            1: "生效",
            2: "退件",
            3: "核減",
            4: "自行撤回",
            6: "撤銷",
            "": "未結案"
        }
    }
    rename_columns_dict = {
        '證券代號':'stock_id', 
        '申報事項':'案件類別', 
        '結案類型':'結案型態', 
        '申報生效日期':'生效日期', 
        '金額_元':'金額',
        '撤銷_廢止日期':'廢止_撤銷日期',
        '廢止日期':'廢止_撤銷日期',
    }
    
    ## pre condition
    if year in [2007, 2008, 2009]:
        header_row = 15
    elif year < 2015:
        header_row = 14
    else:
        header_row = 2

    df = pd.read_excel(path, header=None)
    header = df.iloc[header_row].replace('\n', '', regex=True).replace(['/'], '_', regex=True)
    header = header.rename(None)
    df.columns = [re.sub(r'\(([^)]+)\)', r'_\1', col).replace('　', '').replace(' ', '') if isinstance(col, str) else col for col in header]
    try:
        return df\
            .loc[header_row+1:]\
            .rename(columns=rename_columns_dict)\
            .pipe(lambda df: df.assign(**{col: (lambda x, col=col, mapping=mapping: x[col].dropna().astype(int).map(mapping).reindex(x.index))(df) for col, mapping in replace_value_dict.items()}) if year <= 2014 else df)\
            .assign(**{
                'stock_id': lambda df: df['stock_id'].astype(str).str.zfill(4),
                '生效日期': lambda df: df['生效日期'].fillna(df['結案日期_申請制']) if year <= 2006 else df['生效日期'],
                '金額': lambda df: df['金額'].mul(1000) if year <= 2014 else df['金額'],
            })\
            .assign(**{
                col: lambda df, col=col: pd.to_datetime(df[col].apply(roc_to_ad), format='%Y%m%d')
                for col in [
                    '收文日期', 
                    '生效日期', 
                    *[c for c in ['廢止_撤銷日期', '退件日期'] if c in df.columns],
                ]
            })\
            .query('公司型態!="未上市(櫃)" and 公司型態!="公發" and stock_id not in ["合計", "00合計"]')\
            .dropna(how='all', axis=1)\
            .reset_index(drop=True)
    except Exception as e:
        print(e, year, path)

def get_corp_submit() -> pd.DataFrame:
    source_rul = 'https://www.sfb.gov.tw/ch/home.jsp?id=1016&parentpath=0,6,52'
    sorted_years = sorted([int(x)+1911 for x in pd.read_html(source_rul)[0].loc[0, '下載附件名稱'].replace('年度申報案件', '').split('  ') if x.strip()], reverse=True)

    latest_data_url = 'https://www.fsc.gov.tw/userfiles/file/114730%E7%94%B3%E5%A0%B1%E6%A1%88%E4%BB%B6%E5%BD%99%E7%B8%BD%E8%A1%A8.xlsx'
    last_year_data_url = 'https://www.fsc.gov.tw/userfiles/file/11311473%E7%94%B3%E5%A0%B1%E6%A1%88%E4%BB%B6%E5%BD%99%E7%B8%BD%E8%A1%A8.xlsx'
    hist_data_zip_url = 'https://www.fsc.gov.tw/userfiles/file/%E7%94%B3%E5%A0%B1%E6%A1%88%E4%BB%B6case89-112.zip'
    hist_data_path = '證期局_受理申報案件'
    
    # download historical data
    download_and_unzip(hist_data_zip_url, extract_to=hist_data_path)
    
    # process data with year
    year_path = get_historical_data_year_path(hist_data_path)
    year_path[sorted_years[1]] = last_year_data_url
    year_path[sorted_years[0]] = latest_data_url

    data = pd.concat([process_corp_submit(path, year) for year, path in tqdm.tqdm(year_path.items(), desc="Concating files")], ignore_index=True).sort_values(by=['收文日期', '生效日期'])

    # finally, remove the historical data
    if os.path.exists(hist_data_path):
        shutil.rmtree(hist_data_path)

    return data

def get_cb_submit_from_corp_submit() -> pd.DataFrame:
    return get_corp_submit()\
    .query('案件類別.str.contains("轉換", na=False) and \
        結案型態 == "生效" and \
        not 案件類別.str.contains("私募|海外", na=False) and \
        公司型態.str.contains("上櫃|上市", na=False)'
    )\
    [['stock_id', '承銷商', '案件類別', '金額', '收文日期', '生效日期']]\
    .dropna(subset=['收文日期', '生效日期'])\
    .sort_values(['收文日期', '生效日期'])\
    .reset_index(drop=True)

# 公開資訊觀測站_重大訊息
def get_mops_material_info_single(stock_id:str, year:int) -> pd.DataFrame:
    if isinstance(year, str):
        year = int(year)
    roc_year = year - 1911
    # MOPS query URL
    url = 'https://mopsov.twse.com.tw/mops/web/t05st01'
    
    # Prepare form data for POST request
    form_data = {
        'encodeURIComponent': '1',
        'step': '1',
        'firstin': '1',
        'off': '1',
        'keyword4': '',
        'code1': '',
        'TYPEK2': '',
        'checkbtn': '',
        'queryName': 'co_id',
        'inpuType': 'co_id',
        'TYPEK': 'all',
        'isnew': 'false',
        'co_id': stock_id,
        'year': str(roc_year)
    }
    
    # Send POST request
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    response = session.post(url, data=form_data)
    response.encoding = 'utf-8'
    
    if response.status_code == 200:
        try:
            tables = pd.read_html(response.text)
            return tables[10]
        except Exception as e:
            print(f"Error parsing HTML tables: {e}")
            return None
    else:
        print(f"Error: HTTP {response.status_code}")
        return None

def get_mops_material_info_bulk(stock_ids:list[str], start_year:int=2005, end_year:int=None) -> pd.DataFrame:
    if isinstance(stock_ids, str):
        stock_ids = [stock_ids]
    if end_year is None:
        end_year = pd.Timestamp.today().year
    info_df = pd.DataFrame()
    tasks = [(s, y) for s in stock_ids for y in range(start_year, end_year+1)]

    results = []
    with ThreadPoolExecutor(max_workers=100) as executor:
        for info in tqdm.tqdm(executor.map(lambda args: get_mops_material_info_single(*args), tasks), total=len(tasks), desc="Fetching MOPS info"):
            if info is not None:
                results.append(info)
    info_df = pd.concat(results, ignore_index=True)

    return info_df\
        .rename(columns={
            '公司代號': 'stock_id',
            '公司名稱': 'stock_name',
            '發言日期': 'release_date',
            '發言時間': 'release_time',
            '主旨': 'title',
        })\
        .drop(columns=['Unnamed: 5'])\
        .assign(release_date= lambda df: pd.to_datetime(df['release_date'].apply(roc_to_ad), format='%Y%m%d'))\
        .sort_values(['release_date', 'release_time', 'stock_id'])

def get_mops_material_info(stock_id:str, year:int) -> pd.DataFrame:
    stock_ids = None