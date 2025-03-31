from typing import Union, Literal
import pyarrow.parquet as pq
import pyarrow as pa
import datetime as dt
import pandas as pd
import numpy as np
import logging
import tejapi
import time
import os

# stock industry tag
from bs4 import BeautifulSoup

# finmind
import requests
import geocoder

# # threading
# from concurrent.futures import ThreadPoolExecutor
# from requests.adapters import HTTPAdapter, Retry

# config
from .config import config

from warnings import filterwarnings, simplefilter
filterwarnings('ignore', category=FutureWarning, module='tejapi.get')
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.set_option('future.no_silent_downcasting', True)


class DataBankInfra:
    """資料庫基礎建設

    Args:
        databank_path (str): 資料庫路徑
        start_date (str): 資料起始日期，格式為 YYYY-MM-DD
        databank_map (dict): 資料庫結構對照表，包含以下資料集:
        
            - rf_rate: 無風險利率
            - broker: 券商相關資料
            - raw_data: 原始資料
                - fundamental: 基本面資料
                - technical: 技術面資料
                - chip: 籌碼面資料
            - processed_data: 處理後資料
                - fundamental: 基本面資料
                - technical: 技術面資料
                - chip: 籌碼面資料

    Examples:
        ```python
        # 初始化資料庫基礎建設
        db = DataBankInfra()

        # 讀取資料集
        df = db.read_dataset('monthly_rev')

        # 寫入資料集
        db.write_dataset('monthly_rev', df)

        # 讀取特定時間區間的資料
        df = db._read_dataset_max(
            dataset='stock_trading_data',
            start='2020-01-01',
            end='2020-12-31'
        )
        ```

    Note:
        - 需要在 config 中設定 databank_path
        - 資料庫結構需符合 databank_map 定義
        - 支援 parquet 格式讀寫
        - 提供基礎的資料讀寫功能
    """
    def __init__(self):
        self.databank_path = '/Users/jianrui/Desktop/Research/Quant/databank'
        self.data_type = 'parquet'
        self.databank_map = {
            'datasets':{
                'tej_data':{
                    'monthly_rev',
                    'self_disclosed',
                    'fin_data',
                    'dividend_policy',
                    'capital_formation',
                    'stock_basic_info',
                    'stock_trading_data',
                    'stock_trading_notes',
                    'mkt_calendar',
                    'trading_activity',
                    'stock_custody',
                },
                'processed_data':{
                    'fin_data_chng',
                    'fin_ratio_diff',
                    'fin_data_lag',
                    'monthly_rev_chng',
                    'monthly_rev_reach_ath',
                    'monthly_rev_ath_distance',
                    'monthly_rev_lag',
                    'stock_momentum',
                    'shareholding_pct',
                    'shareholding_pct_diff',
                    'inst_investor_ratio_diff',
                    'inst_investor_money_chng',
                    'factor_model',
                    'stock_sector',
                    'exp_returns',
                },
                'public_data':{
                    'stock_industry_tag',
                    'rf_rate',
                }
            },
            'data':{},
        }

        # init
        self._create_databank()

    def _create_databank(self, current_map=None, current_path=None):
        """建立資料庫目錄結構

        根據 databank_map 定義的結構建立對應的資料夾。

        Examples:
            ```python
            db = DataBank()
            db.create_databank()
            ```
        """
        current_map = self.databank_map if current_map is None else current_map
        current_path = self.databank_path if current_path is None else current_path

        for key, value in current_map.items():
            path = os.path.join(current_path, key)
            os.makedirs(path, exist_ok=True)

            if isinstance(value, dict):
                self._create_databank(value, path)

    def _get_dataset_path_from_map(self, current_map, current_path, dataset:str):
        for key, value in current_map.items():
            path = os.path.join(current_path, key)
            if isinstance(value, dict):
                result = self._get_dataset_path_from_map(value, path, dataset)
                if result:
                    return result
            elif isinstance(value, set) and dataset in value:
                return os.path.join(path, f"{dataset}.{self.data_type}")
        return None

    def _get_dataset_path(self, dataset:str):
        # First try to find existing file
        for root, dirs, files in os.walk(os.path.join(self.databank_path, 'datasets')):
            for file in files:
                if (file.endswith('.parquet')) and (dataset == file.replace('.parquet', '')):
                    return os.path.join(root, file)
        map_path = self._get_dataset_path_from_map(self.databank_map, self.databank_path, dataset)
        if map_path:
            return map_path
        else:
            raise ValueError(f'Dataset {dataset} not found in either databank or databank_map')
        
    def list_datasets(self):
        """列出所有可用的資料集

        遍歷資料庫目錄結構，找出所有 .parquet 檔案，並返回資料集名稱與路徑的映射。

        Returns:
            dict: 資料集名稱與其對應路徑的字典

        Examples:
            ```python
            db = DataBank()
            datasets = db.list_datasets()
            ```
        """
        # datasets = {}
        datasets = []
        for root, dirs, files in os.walk(os.path.join(self.databank_path, 'datasets')):
            for file in files:
                if file.endswith('.parquet'):
                    # full_path = os.path.join(root, file)
                    # dataset_name = os.path.splitext(file)[0]
                    # datasets[dataset_name] = full_path
                    datasets.append(file.replace('.parquet', ''))
        return datasets
    
    def list_columns(self, dataset:str, keyword:str=None):
        """取得資料集中所有欄位名稱

        Args:
            dataset (str): 資料集名稱

        Returns:
            list: 包含所有欄位名稱的列表

        Examples:
            ```python
            # 取得資料集中所有欄位名稱
            db = DataBank()
            db.list_columns('stock_trading_data')
            ```

            output:
            ```
            ['收盤價', '開盤價', '最高價', '最低價', ...]
            ```
        """
        
        data_path = self._get_dataset_path(dataset)
        if (keyword is not None) and (os.path.exists(data_path)):
            return [col for col in pq.ParquetFile(data_path).schema.names if keyword in col]
        elif not os.path.exists(data_path):
            return []
        else:
            return pq.ParquetFile(data_path).schema.names

    def find_dataset(self, column:str):
        """根據欄位名稱找出對應的資料集名稱

        Args:
            column (str): 欄位名稱

        Examples:
            例如，找出包含 '收盤價' 欄位的資料集：
            ```py
            find_dataset(column='收盤價')
            ```
            output
            ```
            'stock_trading_data'
            ```
        """
        column_datasets = {}
        for data_name in self.list_datasets():
            if data_name != 'exp_returns':
                try:
                    for col in self.list_columns(data_name):
                        if col not in column_datasets:
                            column_datasets[col] = []
                        column_datasets[col].append(data_name)
                except FileNotFoundError:
                    continue
        if len(column_datasets[column])>1:
            raise ValueError(f'Column {column} found in multiple datasets: {column_datasets[column]}')
        elif len(column_datasets[column])==0:
            raise ValueError(f'Column {column} not found in any dataset')
        else:
            return column_datasets[column][0]
    
    # db read & write
    def read_dataset(self, dataset:str, filter_date:str='date', start:Union[int, str]=None, end:Union[int, str]=None, columns:list=None, filters:list=None) -> pd.DataFrame:
        """讀取資料集

        Args:
            dataset (str): 資料集名稱
            filter_date (str, optional): 用於日期過濾的欄位名稱，預設為 'date'
            start (Union[int, str], optional): 起始日期，可以是年份(int)或日期字串(str)
            end (Union[int, str], optional): 結束日期，可以是年份(int)或日期字串(str) 
            columns (list, optional): 要讀取的欄位清單
            filters (list, optional): 額外的過濾條件，格式為 [(column, op, value),...]

        Returns:
            pd.DataFrame: 包含請求資料的 DataFrame

        Examples:
            讀取 2020 年的股票交易資料:
            ```python
            db = DataBank()
            df = db.read_dataset(
                dataset='stock_trading_data',
                start=2020,
                end=2020
            )
            ```

            讀取特定日期區間的特定欄位:
            ```python
            df = db.read_dataset(
                dataset='stock_trading_data',
                start='2020-01-01',
                end='2020-12-31',
                columns=['股票代號', '收盤價']
            )
            ```

            使用自訂過濾條件:
            ```python
            df = db.read_dataset(
                dataset='stock_trading_data',
                filters=[('收盤價', '>', 100)]
            )
            ```
        """
        # Process date filters
        filters = filters or []
        if start:
            start_date = dt.datetime(start, 1, 1) if isinstance(start, int) else pd.to_datetime(start)
            filters.append((filter_date, '>=', start_date))
        if end:
            end_date = dt.datetime(end, 12, 31) if isinstance(end, int) else pd.to_datetime(end)
            filters.append((filter_date, '<=', end_date))

        # Read data
        data_path = self._get_dataset_path(dataset)
        if not os.path.exists(data_path):
            logging.warning(f'{dataset}.{self.data_type} not exist')
            raise FileNotFoundError(f'{dataset}.{self.data_type} not exist')
        
        df = pq.read_table(source=data_path, columns=columns, filters=filters or None).to_pandas()
        logging.info(f'Exported {dataset} as DataFrame from {self.data_type}')
        return df

    def write_dataset(self, dataset:str, df:pd.DataFrame):
        """將 DataFrame 寫入資料庫

        Args:
            dataset (str): 資料集名稱
            df (pd.DataFrame): 要寫入的 DataFrame

        Examples:
            ```python
            # 將 DataFrame 寫入資料庫
            db = DataBank()
            db.write_dataset('monthly_rev', df)
            ```
        """

        # Sort value
        sort_columns = [col for col in ['date', 'stock_id'] if col in df.columns]
        if sort_columns:
            df = df.sort_values(by=sort_columns, ascending=True).reset_index(drop=True)

        # Write to pqt
        table = pa.Table.from_pandas(df)
        data_path = self._get_dataset_path(dataset)
        pq.write_table(table, data_path)
        logging.info(f'Saved {dataset} as parquet from DataFrame')


class TEJHandler(DataBankInfra):
    """TEJ API 資料處理器

    處理 TEJ API 資料的下載、更新與儲存。

    Args:
        tej_token (str): TEJ API 金鑰
        tej_datasets (dict): TEJ 資料集對照表，包含中文名稱與 API ID
        bulk_tej_data_temp (pd.DataFrame): 暫存批量下載的資料

    Examples:
        ```python
        # 初始化 TEJ 處理器
        tej = TEJHandler()

        # 下載月營收資料
        df = tej.get_tej_data(
            dataset='monthly_rev',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )

        # 更新資料集
        tej.update_tej_dataset('stock_trading_data')
        ```

    Note:
        - 需要在 config 中設定 tej_token
        - API 使用量有每日上限，請注意使用量
    """

    def __init__(self):
        super().__init__()
        self.__tej_token = config.data_config.get('tej_token')
        tejapi.ApiConfig.api_key = self.__tej_token
        tejapi.ApiConfig.ignoretz = True
        self.start_date = '2005-01-01'
        self.tej_datasets = { 
            'monthly_rev':{'ch':'月營收', 'id':'TWN/APISALE'},
            'self_disclosed':{'ch':'公司自結數', 'id':'TWN/AFESTM1'},
            'fin_data':{'ch':'會計師簽證財務資料', 'id':'TWN/AINVFQ1'},
            'dividend_policy':{'ch':'股利政策', 'id':'TWN/APIMT1'},
            'capital_formation':{'ch':'資本形成', 'id':'TWN/APISTK1'},
            'stock_basic_info':{'ch':'證券屬性資料表', 'id':'TWN/APISTOCK'},
            'stock_trading_data':{'ch':'股價交易資訊', 'id':'TWN/APIPRCD'},
            'stock_trading_notes':{'ch':'股票日交易註記資訊', 'id':'TWN/APISTKATTR'},
            'mkt_calendar':{'ch':'交易日期表', 'id':'TWN/TRADEDAY_TWSE'},
            'trading_activity':{'ch':'三大法人、融資券、當沖', 'id':'TWN/APISHRACT'},
            'stock_custody':{'ch':'集保庫存', 'id':'TWN/APISHRACTW'},
        }
        
        # temp storage
        self.bulk_tej_data_temp = pd.DataFrame()

    def reach_tej_limit(self) -> bool:
        """檢查 TEJ API 使用量是否達到上限

        Returns:
            bool: 是否達到上限
                - True: 達到上限
                - False: 未達上限

        Examples:
            ```python
            # 檢查 API 使用量
            tej = TEJHandler()
            tej.reach_tej_limit()  # 回傳: False
            ```

        Note:
            當使用量達到 95% 時，會將 self.almost_reach_limit 設為 True
        """
        api_info = tejapi.ApiConfig.info()
        today_req, req_limit = api_info.get('todayReqCount'), api_info.get('reqDayLimit')
        today_row, row_limit = api_info.get('todayRows'), api_info.get('rowsDayLimit')
        req_usage = today_req / req_limit
        row_usage = today_row / row_limit

        # current status
        logging.info(f'Req usage: {req_usage:.2%} ({today_req: ,}/{req_limit: ,}) ; Row usage: {row_usage:.2%} ({today_row: ,}/{row_limit: ,})')

        # almost reach limit:
        self.almost_reach_limit = ((row_usage >= 0.95) or (req_usage >= 0.95))
        
        # reach limit
        reach_limit = ((row_usage >= 1) or (req_usage >= 1))
        if reach_limit:
            logging.warning('TEJ API limit reached')
        return reach_limit

    def get_tej_data(self, dataset:str, start_date:str=None, end_date:str=None, stock_id:str=None)-> pd.DataFrame:
        """從 TEJ API 取得資料

        Args:
            dataset (str): 資料集名稱
            start_date (str, optional): 起始日期，格式為 'YYYY-MM-DD'
            end_date (str, optional): 結束日期，格式為 'YYYY-MM-DD'
            stock_id (str, optional): 股票代號

        Returns:
            pd.DataFrame: 從 TEJ API 取得的資料

        Examples:
            ```python
            # 取得單一股票的月營收資料
            tej = TEJHandler()
            df = tej.get_tej_data(
                dataset='monthly_rev',
                start_date='2020-01-01',
                end_date='2020-12-31',
                stock_id='2330'
            )
            ```

            ```python
            # 取得所有股票的月營收資料
            df = tej.get_tej_data(
                dataset='monthly_rev',
                start_date='2020-01-01',
                end_date='2020-12-31'
            )
            ```

        Note:
            - 若達到 API 使用量上限，將回傳 None
            - 部分資料集(如 mkt_calendar, stock_basic_info)不需要日期區間
        """
        if self.reach_tej_limit():
            return None
        
        tej_dataset = self.tej_datasets.get(dataset).get('id')
        mdate = {
            'gte': start_date,
            'lte': end_date,
        } if dataset not in ['mkt_calendar', 'stock_basic_info'] else None
        
        #fast get
        df = tejapi.fastget(
            datatable_code = tej_dataset, 
            coid=stock_id,
            mdate=mdate, 
            chinese_column_name=True,
            paginate=True, 
        )
        df = self._process_tej_data(dataset, df)
        if 'date' in df.columns:
            logging.info(f'Fetched {dataset} from {df.date.min().strftime("%Y-%m-%d")} to {df.date.max().strftime("%Y-%m-%d")} from TEJ', )
        else:
            logging.info(f'Fetched {dataset} from TEJ')
        return df

    def get_tej_data_bulk(self, dataset:str, start_date:str=None, end_date:str=None, stock_id:str=None)-> pd.DataFrame:
        """批量取得 TEJ 資料

        當需要取得大量資料時，為避免一次取得太多資料導致 API 使用量超過上限，
        將資料分批取得，每次取得一年的資料。

        Args:
            dataset (str): 資料集名稱
            start_date (str): 起始日期，格式為 YYYY-MM-DD
            end_date (str, optional): 結束日期，格式為 YYYY-MM-DD，預設為今天
            stock_id (str, optional): 股票代號，預設為 None，表示取得所有股票資料

        Returns:
            pd.DataFrame: TEJ 資料

        Examples:
            ```python
            # 取得 2005 年至今的所有股票交易資料
            tej = TEJHandler()
            df = tej.get_tej_data_bulk(
                dataset='stock_trading_data',
                start_date='2005-01-01'
            )
            ```

        Note:
            - 若達到 API 使用量上限，將回傳已取得的資料
            - 每次取得一年的資料，直到取得所有資料或達到 API 使用量上限
        """
        start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date = dt.date.today() if end_date is None else dt.datetime.strptime(end_date, "%Y-%m-%d").date()

        self.bulk_tej_data_temp = pd.DataFrame()
        current_start_date = start_date
        while current_start_date < end_date:
            current_end_date = current_start_date + dt.timedelta(days=365) # every year
            if current_end_date > end_date:
                current_end_date = end_date
            
            # get data
            df = self.get_tej_data(
                dataset, 
                start_date=current_start_date.strftime('%Y-%m-%d'),
                end_date=current_end_date.strftime('%Y-%m-%d'),
                stock_id=stock_id
                )
            self.bulk_tej_data_temp = pd.concat([self.bulk_tej_data_temp, df], ignore_index=True).reset_index(drop=True)
            
            # save if almost reach limit
            if self.almost_reach_limit:
                self.reach_tej_limit()
                break
            
            # continue
            current_start_date = current_end_date + dt.timedelta(days=1)
        
        return self.bulk_tej_data_temp

    def _process_tej_data(self, dataset:str, df:pd.DataFrame):
        """處理 TEJ 資料

        Args:
            dataset (str): 資料集名稱
            df (pd.DataFrame): 要處理的 DataFrame

        Returns:
            pd.DataFrame: 處理後的 DataFrame

        Note:
            - 重新命名欄位
            - 調整資料型態
            - 處理日期欄位
            - 排序資料
            - 加入交易日期
        """
        # leave only 期間別 =='Q' & 序號=='001'
        if dataset in ['self_disclosed', 'fin_data']:
            df = df[
                (df['期間別']=='Q')&
                (df['序號']=='001')
            ]
        
        # rename columns
        rename_columns = {
            'monthly_rev':{'公司':'stock_id', '年月':'date', '營收發布日':'release_date'},
            'self_disclosed':{'公司':'stock_id', '年/月':'date', '編表日':'release_date'},
            'fin_data':{'公司':'stock_id', '年/月':'date', '編表日':'release_date'},
            'dividend_policy':{'公司':'stock_id'},
            'capital_formation':{'公司':'stock_id'},
            'stock_basic_info':{'公司簡稱':'stock_id'},
            'stock_trading_data':{'證券名稱':'stock_id', '資料日':'date'},
            'stock_trading_notes':{'證券名稱':'stock_id', '資料日':'date'},
            'mkt_calendar':{'日期':'date'},
            'trading_activity':{'證券名稱':'stock_id', '資料日':'date'},
            'stock_custody':{'證券名稱':'stock_id', '資料日':'date', '公告日(集保庫存)':'release_date_集保庫存' ,'公告日(集保股權)':'release_date_集保股權'},
        }
        df = df.rename(columns=rename_columns.get(dataset))

        # rename 自結
        if dataset == 'self_disclosed':
            df = df.rename(columns={col: f'{col}_自' for col in df.columns if col not in ['date', 'stock_id', 'release_date', '期間別', '序號', '季別', '合併(Y/N)', '幣別', '產業別', 't_date', 'insert_time']})

        # reorder
        front_cols=['date', 'stock_id', 'release_date']
        df=df[
            [col for col in front_cols if col in df.columns] + \
            [col for col in df.columns if col not in front_cols]
            ]

        # dtype
        if 'date' in df.columns:
            df = df.assign(date=lambda x: pd.to_datetime(x['date']))
        if 'release_date' in df.columns:
            df = df.assign(release_date=lambda x: pd.to_datetime(x['release_date']))
        if 'stock_id' in df.columns:
            df = df.assign(stock_id=lambda x: x['stock_id'].astype(str))
        
        # fillna for monthly rev release date
        if dataset=='monthly_rev':
            if df['release_date'].isna().any():
                df['release_date'] = df['release_date'].fillna(df['date'] + pd.DateOffset(days=9, months=1))
        
        # stock calendar
        if dataset == 'mkt_calendar':
            df = df[
                (df['date'] >= self.start_date) &
                (df['date'].dt.year <= dt.date.today().year+1)
                ]

        # sort value
        sort_columns = [col for col in ['date', 'stock_id'] if col in df.columns]
        if sort_columns:
            df = df.sort_values(by=sort_columns, ascending=True).reset_index(drop=True)
        
        # add t_date
        if (dataset != 'mkt_calendar') & ('date' in df.columns):
            df = self._add_trade_date(df)

        return df

    def _add_trade_date(self, df:pd.DataFrame) -> pd.DataFrame:
        """將資料加入交易日期

        將資料的日期或發布日期對應到下一個交易日，用於計算資訊揭露後的市場反應

        Args:
            df (pd.DataFrame): 要加入交易日期的 DataFrame

        Returns:
            pd.DataFrame: 加入交易日期後的 DataFrame，新增欄位:
                - t_date: 下一個交易日期

        Examples:
            ```python
            # 將月營收資料加入交易日期
            db = DataBank()
            df = db.read_dataset('monthly_rev')
            df = db._add_trade_date(df)
            ```

        Note:
            - 若資料有 release_date 欄位，則以 release_date 為基準找下一個交易日
            - 若無 release_date 欄位，則以 date 欄位為基準找下一個交易日
            - 若資料已有 t_date 欄位，則會先刪除再重新計算
        """
        
        # import trade date
        t_date = self.read_dataset('mkt_calendar', columns=['date'], filters=[('休市原因中文說明(人工建置)','=','')]).rename(columns={'date':'t_date'})
        date_col = 'release_date' if 'release_date' in df.columns else 'date'
        
        df = df\
            .drop(columns='t_date', errors='ignore')\
            .assign(next_day=lambda x: (x[date_col] + pd.DateOffset(days=1)).astype('datetime64[ns]'))\
            .sort_values('next_day')\
            .reset_index(drop=True)
        df = pd.merge_asof(
            df, 
            t_date, 
            left_on='next_day', 
            right_on='t_date', 
            direction='forward'
        ).drop('next_day', axis=1)

        # re-order
        if df.columns[-2:].tolist() == ['insert_time', 't_date']:
            cols = df.columns.tolist()
            cols.insert(cols.index('insert_time'), cols.pop(-1))
            df = df[cols]

        return df

    def _concat_new_trading_data(self, old_data, new_data):
        """合併新舊交易資料並調整除權息係數

        將新的交易資料與舊資料合併，並重新計算除權息調整係數。新資料的調整係數會乘上舊資料的調整係數，
        以確保歷史資料的連續性。

        Args:
            old_data (pd.DataFrame): 舊的交易資料，包含 date、stock_id、調整係數、調整係數(除權) 等欄位
            new_data (pd.DataFrame): 新的交易資料，包含 date、stock_id、調整係數、調整係數(除權) 等欄位

        Returns:
            pd.DataFrame: 合併後的交易資料，包含以下欄位:
                - date: 交易日期
                - stock_id: 股票代號
                - 調整係數: 更新後的除權息調整係數
                - 調整係數(除權): 更新後的除權調整係數
                - 其他原有欄位

        Examples:
            ```python
            # 合併新舊交易資料
            old_data = db.read_dataset('stock_trading_data')
            new_data = db.get_tej_data('stock_trading_data', start_date='2023-01-01')
            merged_data = db._concat_new_trading_data(old_data, new_data)
            ```

        Note:
            - 新資料中不包含舊資料的最後一天，避免重複計算調整係數
            - 調整係數採用累乘方式計算，確保歷史資料的連續性
            - 若股票代號在新資料中不存在，則調整係數維持不變
        """

        # pord new 調整係數 (do not pord today)
        new_data = new_data[new_data.date != old_data.date.max()]
        adj_params=new_data\
            .groupby('stock_id')\
            .agg({
                '調整係數': 'prod',
                '調整係數(除權)': 'prod',})\
            .reset_index().rename(columns={
                '調整係數': '調整係數_new',
                '調整係數(除權)': '調整係數(除權)_new',})

        # merge with old data
        old_data = old_data.merge(adj_params, on='stock_id', how='left').fillna({
            '調整係數_new': 1,
            '調整係數(除權)_new': 1
        })
        
        old_data['調整係數'] = old_data['調整係數'] * old_data['調整係數_new']
        old_data['調整係數(除權)'] = old_data['調整係數(除權)'] * old_data['調整係數(除權)_new']

        old_data = old_data.drop(columns=['調整係數_new', '調整係數(除權)_new'])
        
        old_data = pd.concat([old_data, new_data], ignore_index=True)
        old_data = old_data\
            .drop_duplicates(subset=['date', 'stock_id'], keep='last')\
            .reset_index(drop=True)

        return old_data
        
    def update_tej_dataset(self, dataset:str, end_date:str=None, back_period:int=2):
        """更新TEJ資料集

        Args:
            dataset (str): 要更新的資料集名稱
            end_date (str, optional): 更新資料的結束日期. Defaults to None.
            back_period (int, optional): 回溯期間. Defaults to 2.
        Examples:
            ```python
            # 更新股票交易資料
            db.update_tej_dataset('stock_trading_data')

            # 更新到指定日期的財務資料
            db.update_tej_dataset('fin_data', end_date='2023-12-31')
            ```

        Note:
            - 對於交易相關資料(stock_trading_data等)，若更新區間超過365天，會使用批量下載模式
            - 股票交易資料會自動處理除權息調整係數
            - 所有資料會加入insert_time欄位記錄寫入時間
            - 重複資料會保留最新一筆
        """
        
        # check if dataset is in tej_datasets
        if dataset not in self.tej_datasets.keys():
            raise ValueError(f'{dataset} is not in tej_datasets')

        # mkt calendar
        if dataset == 'mkt_calendar':
            # first check last date
            if self.read_dataset('mkt_calendar', columns=['date'])['date'].max().year <= dt.datetime.today().year:
                data = self.get_tej_data('mkt_calendar')
                data['insert_time'] = pd.Timestamp.now()
                self.write_dataset(dataset, data)
        
        # stock basic info
        elif dataset == 'stock_basic_info':
            # first check last insert_time
            if self.read_dataset('stock_basic_info', columns=['insert_time'])['insert_time'].min() <= (pd.Timestamp.now() - pd.Timedelta(weeks=1)):
                data = self.get_tej_data('stock_basic_info')
                data['insert_time'] = pd.Timestamp.now()
                self.write_dataset(dataset, data)
        
        # others
        else:
            # old data
            old_data = self.read_dataset(dataset)

            # date
            request_dates = {
                'monthly_rev':'date',
                'self_disclosed':'date',
                'fin_data':'date',
                'dividend_policy':'today',
                'capital_formation':'today',
                'stock_trading_data':'date',
                'stock_trading_notes':'date',
                'trading_activity':'date',
                'stock_custody':'date',
            }
            if (not old_data.empty) & (request_dates.get(dataset) == 'date'):
                start_date = pd.Series(old_data['date'].unique()).nlargest(back_period).min().strftime('%Y-%m-%d')
            elif (not old_data.empty) & (request_dates.get(dataset) == 'today'):
                start_date = (dt.date.today() - dt.timedelta(days=back_period)).strftime('%Y-%m-%d')
            else:
                start_date = self.start_date
            
            # get data, process data
            if (dataset in ['stock_trading_data', 'stock_trading_notes', 'trading_activity', 'stock_custody']) & \
                ((dt.date.today() - dt.datetime.strptime(start_date, "%Y-%m-%d").date()).days >= 365):
                new_data = self.get_tej_data_bulk(dataset, start_date=start_date, end_date=end_date)
            else:
                new_data = self.get_tej_data(dataset, start_date=start_date, end_date=end_date)
            
            # insert time
            new_data['insert_time'] = pd.Timestamp.now()

            if (dataset == 'stock_trading_data') & (not old_data.empty):
                old_data = self._concat_new_trading_data(old_data, new_data)
            else:
                old_data = pd.concat([
                    old_data.dropna(axis=1, how='all'), 
                    new_data.dropna(axis=1, how='all')
                ], ignore_index=True)
                
                # drop_duplicates
                drop_dup_subset = {
                    'dividend_policy': ['stock_id', '盈餘分派_年度'],
                    'capital_formation': ['stock_id', '除權日'],
                    'stock_basic_info': ['stock_id'],
                    'mkt_calendar': ['date']
                }.get(dataset, ['date', 'stock_id'])
                old_data = old_data\
                    .drop_duplicates(subset=drop_dup_subset, keep='last')\
                    .reset_index(drop=True)

            # insert
            self.write_dataset(dataset, old_data)
        
        logging.info(f'Updated {dataset} from tej')


class ProcessedDataHandler(DataBankInfra):
    """處理資料的類別

    此類別負責處理原始資料,計算各種財務、技術、籌碼等指標。

    Args:
        processed_datasets (dict): 已處理資料集的設定,包含:
            - fin_data_chng: 財務資料變化率
            - fin_ratio_diff: 財務比率差分
            - roe_roa: ROE與ROA
            - monthly_rev_chng: 月營收變化率
            - stock_momentum: 股價動量
            - shareholding_pct: 股東持股比例
            - shareholding_pct_diff: 股東持股比例變化率
            - inst_investor_ratio_diff: 法人持股比例變化率
            - inst_investor_money: 法人買賣超金額
            - exp_returns: 個股報酬率

    Returns:
        None

    Examples:
        ```python
        # 初始化處理資料類別
        handler = ProcessedDataHandler()

        # 更新特定資料集
        handler.update_processed_data(dataset='fin_data_chng')

        # 更新所有資料集
        handler.update_processed_data()
        ```

    Note:
        - 每個資料集都有對應的更新函數
        - 可以選擇更新單一資料集或全部資料集
        - 資料處理包含財務面、技術面、籌碼面等多個面向
    """

    def __init__(self):
        super().__init__()
        self.processed_datasets = { 
            'fin_data_chng':{'source':'fin_data', 'func':self._update_fin_data_chng},
            'fin_ratio_diff':{'source':'fin_data', 'func':self._update_fin_ratio_diff},
            'fin_data_lag':{'source':'fin_data', 'func':self._update_fin_data_lag},
            'monthly_rev_chng':{'source':'monthly_rev', 'func':self._update_monthly_rev_chng},
            'monthly_rev_reach_ath':{'source':'monthly_rev', 'func':self._update_monthly_rev_reach_ath},
            'monthly_rev_ath_distance':{'source':'monthly_rev', 'func':self._update_monthly_rev_ath_distance},
            'monthly_rev_lag':{'source':'monthly_rev', 'func':self._update_monthly_rev_lag},
            'stock_momentum':{'source':'stock_trading_data', 'func':self._update_stock_momentum},
            'shareholding_pct':{'source':'stock_custody', 'func':self._update_shareholding_pct},
            'shareholding_pct_diff':{'source':'stock_custody', 'func':self._update_shareholding_pct_diff},
            'inst_investor_ratio_diff':{'source':'trading_activity', 'func':self._update_inst_investor_ratio_diff},
            'inst_investor_money_chng':{'source':'trading_activity', 'func':self._update_inst_investor_money_chng},
            'exp_returns':{'source':'stock_trading_data', 'func':self._update_exp_returns},
            'stock_sector':{'source':'stock_trading_notes', 'func':self._update_stock_sector},
        }
        # self.processed_datasets.update(self.factors_datasets)
    
    # fundamental
    def _update_fin_data_chng(self):
        """計算財務資料變化率

        計算財務資料的季增率(qoq)與年增率(yoy)。

        Returns:
            None: 將計算結果寫入 fin_data_chng 資料集

        Examples:
            ```python
            # 更新財務資料變化率
            db = DataBank()
            db._update_fin_data_chng()
            ```

        Note:
            - 自動計算所有財務欄位的變化率,不包含比率類欄位
            - 變化率包含季增率(qoq)與年增率(yoy)
        """

        columns = [col for col in self.list_columns('fin_data') if not any(col.endswith(c) for c in [
                    '率', '常續ROE', '常續ROA', 
                    '期間別', '序號', '季別', '合併(Y/N)', '幣別', '產業別', 
                    'date', 'release_date', 'stock_id', 't_date', 'insert_time'
            ])
        ]
        df = self.read_dataset('fin_data', columns=['date', 'release_date', 'stock_id', *columns, 't_date'])
        
        # calculate changes
        grouped = df.groupby('stock_id')
        changes = {'QOQ':1, 'YOY':4}
        for col in [col for col in df.columns if any(col.startswith(c) for c in columns)]:
            for k, v in changes.items():
                df[f'{col}_{k}'] = grouped[col].pct_change(periods=v, fill_method=None)

        df = df[['date', 'release_date', 'stock_id'] + [col for col in df.columns if any(col.endswith(c) for c in ['QOQ','YOY'])] + ['t_date']]
        
        return self.write_dataset(dataset='fin_data_chng', df=df)
    
    def _update_fin_ratio_diff(self):
        """計算財務比率的差分

        計算財務比率的差分，包含季差(qoq_diff)與年差(yoy_diff)。

        Returns:
            None: 將計算結果寫入 fin_ratio_diff 資料集

        Examples:
            ```python
            # 更新財務比率差分
            db = DataBank()
            db._update_fin_ratio_diff()
            ```

        Note:
            - 自動計算所有財務比率欄位的差分
            - 差分包含季差(qoq_diff)與年差(yoy_diff)
        """
        columns = [col for col in self.list_columns('fin_data') if any(col.endswith(c) for c in ['率', '常續ROE', '常續ROA'])]
        df = self.read_dataset('fin_data', columns=['date', 'release_date', 'stock_id', *columns, 't_date'])
        
        # calculate rolling average
        # for col in columns:
        #     rolling_avg = {'4q':4, '8q':8, '12q':12}
        #     for k, v in rolling_avg.items():
        #         df[f'{col}_{k}_avg'] = df.groupby(['stock_id'])[col].transform(lambda d: d.rolling(window=v).mean())
            
        # calculate changes
        grouped = df.groupby('stock_id')
        changes = {'QOQ_DIFF':1, 'YOY_DIFF':4}
        for col in [col for col in df.columns if any(col.startswith(c) for c in columns)]:
            for k, v in changes.items():
                df[f'{col}_{k}'] = grouped[col].diff(periods=v).round(3)
        
        df = df[['date', 'release_date', 'stock_id'] + [col for col in df.columns if any(col.endswith(c) for c in ['QOQ_DIFF','YOY_DIFF'])] + ['t_date']]
        
        return self.write_dataset(dataset='fin_ratio_diff', df=df)

    def _update_fin_data_lag(self):
        
        columns = [col for col in self.list_columns('fin_data') if not any(col.endswith(c) for c in [
                    '期間別', '序號', '季別', '合併(Y/N)', '幣別', '產業別', 
                    'date', 'release_date', 'stock_id', 't_date', 'insert_time'])]
        periods = range(1, 9)
        lag_columns = {
            f'{col}_lag{i}': lambda df, i=i: df.groupby('stock_id')[col].shift(i)
            for i in periods for col in columns
        }
        df = self.read_dataset(
                dataset='fin_data', 
                columns=['date', 'stock_id', *columns, 't_date'])\
            .assign(**lag_columns)
        df = df[['date', 'stock_id'] + [col for col in df.columns if 'lag' in col] + ['t_date']]
        return self.write_dataset(dataset='fin_data_lag', df=df)

    def _update_monthly_rev_chng(self):
        """計算單月營收的變化率

        計算單月營收的變化率，包含月增率(mom)、季增率(qoq)、年增率(yoy)。

        Returns:
            None: 將計算結果寫入 monthly_rev_chng 資料集

        Examples:
            ```python
            db = DataBank()
            db._update_monthly_rev_chng()
            ```

        Note:
            - 計算移動平均包含3月、1年、2年、3年
            - 變化率包含月增率(mom)、季增率(qoq)、年增率(yoy)
            - 會自動填補缺失值
        """

        df = self.read_dataset('monthly_rev', columns=[
            'date', 'release_date', 'stock_id', 
            '單月營收(千元)', '單月營收成長率％', '近12月累計營收成長率％', '近3月累計營收成長率％', '近3月累計營收與上月比％', 
            't_date'
            ])
        column = '單月營收(千元)'

        # calculate rolling average
        rolling_avg = {'3M':3, '1Y':12, '2Y':24, '3Y':36,}
        for k, v in rolling_avg.items():
            df[f'{column}_{k}_AVG'] = df.groupby(['stock_id'])[column].transform(lambda d: d.rolling(window=v).mean())
        
        # calculate changes
        changes = { 'MOM':1, 'QOQ':3, 'YOY':12, }
        for col in [column] + [col for col in df.columns if col.endswith('_AVG')]:
            for k, v in changes.items():
                df[f'{col}_{k}'] = round(df.groupby(['stock_id'])[col].pct_change(periods=v, fill_method=None), 3)
        
        # fillna
        fillna_map = {
            '單月營收(千元)_YOY': '單月營收成長率％',
            '單月營收(千元)_1Y_AVG_YOY': '近12月累計營收成長率％',
            '單月營收(千元)_3M_AVG_YOY': '近3月累計營收成長率％',
            '單月營收(千元)_3M_AVG_MOM': '近3月累計營收與上月比％',
        }
        for col, fill_col in fillna_map.items():
            df[col] = df[col].fillna(df[fill_col]/100)

        df = df[['date', 'release_date', 'stock_id'] + [col for col in df.columns for key in changes if col.endswith(key)] + ['t_date']]
            
        return self.write_dataset(dataset='monthly_rev_chng', df=df)

    def _update_monthly_rev_reach_ath(self):
        """計算單月營收是否達到歷史新高

        計算單月營收是否達到歷史新高。

        Returns:
        """
        df = self.read_dataset('monthly_rev', columns=[
            'date', 'release_date', 'stock_id', 
            '單月營收(千元)', 
            't_date'
            ])
        column = '單月營收(千元)'

        periods = {'3M':3, '1Y':12, '2Y':24, '3Y':36}
        for k, v in periods.items():
            df[f'{column}_is_{k}_ATH'] = (df[column] == df.groupby(['stock_id'])[column].transform(lambda x: x.rolling(window=v).max()))

        # Calculate all-time high separately
        df[f'{column}_is_ATH'] = (df[column] == df.groupby(['stock_id'])[column].transform(lambda x: x.expanding().max()))

        df = df[['date', 'release_date', 'stock_id'] + [col for col in df.columns if col.endswith('ATH')] + ['t_date']]

        return self.write_dataset(dataset='monthly_rev_reach_ath', df=df)
    
    def _update_monthly_rev_ath_distance(self):
        """計算單月營收達到歷史新高以來的月數

        計算單月營收達到歷史新高以來的月數。

        """
        df = self.read_dataset('monthly_rev', columns=[
            'date', 'release_date', 'stock_id', 
            '單月營收(千元)', 
            't_date'
            ])
        column = '單月營收(千元)'

        # Calculate how many months the current revenue is the highest for
        df['單月營收_ATH_DISTANCE'] = 0
        
        # For each stock, calculate how many months back the current revenue is highest
        def calc_ath_months(x):
            curr_val = x.iloc[-1]
            for i in range(len(x)-1, -1, -1):
                if x.iloc[i] > curr_val:
                    return len(x) - i - 1
            return len(x)
            
        df['單月營收_ATH_DISTANCE'] = df.groupby('stock_id')[column].transform(
            lambda x: x.expanding().apply(calc_ath_months)
        )

        df = df[['date', 'release_date', 'stock_id', '單月營收_ATH_DISTANCE', 't_date']]

        return self.write_dataset(dataset='monthly_rev_ath_distance', df=df)

    def _update_monthly_rev_lag(self):
        """計算單月營收的時序資料

        計算單月營收的時序資料，包含月增率(mom)、季增率(qoq)、年增率(yoy)。

        """
        periods = range(1, 25)
        lag_columns = {
            f'單月營收(千元)_lag{i}': lambda df, i=i: df.groupby('stock_id')['單月營收(千元)'].shift(i)
            for i in periods
        }
        df = self.read_dataset(
                dataset='monthly_rev', 
                columns=['date', 'stock_id', '單月營收(千元)', 't_date'])\
            .assign(**lag_columns)
        df = df[['date', 'stock_id'] + [col for col in df.columns if 'lag' in col] + ['t_date']]
        return self.write_dataset(dataset='monthly_rev_lag', df=df)
        

    # technical
    def _update_stock_momentum(self):
        """計算股價動量

        計算不同期間的股價動量指標。

        Returns:
            None: 將計算結果寫入 stock_momentum 資料集

        Examples:
            ```python
            db = DataBank()
            db._update_stock_momentum()
            ```

        Note:
            - 計算期間包含1週、1月、3月、6月、1年、3年
            - 使用累積報酬率計算動量
        """

        df = self.read_dataset('stock_trading_data', columns=['date', 'stock_id', '報酬率', 't_date'])
        df['c_return'] = df.groupby('stock_id')['報酬率'].transform(lambda x: (1 + x/100).cumprod())
        
        # calculate changes
        mtm_map = {
            'MTM_1W':5, 'MTM_1M':20, 'MTM_3M':60, 'MTM_6M':120, 'MTM_1Y':240, 'MTM_3Y':720,
        }
        
        for k, v in mtm_map.items():
            df[k] = df.groupby(['stock_id'])['c_return'].transform(lambda x: x.pct_change(periods=v, fill_method=None)).round(3)

        df = df[['date', 'stock_id'] + [col for col in df.columns if col.startswith('MTM')] + ['t_date']]
        return self.write_dataset(dataset='stock_momentum', df=df)

    # chip
    def _update_shareholding_pct(self):
        """計算股東持股比例

        計算不同持股級距的股東持股比例。

        Returns:
            None: 將計算結果寫入 shareholding_pct 資料集

        Examples:
            ```python
            db = DataBank()
            db._update_shareholding_pct()
            ```

        Note:
            - 計算級距包含:未滿400張、超過400張、超過600張、超過800張、超過1000張
            - 比例以總持股數為分母計算
        """
        df = self.read_dataset('stock_custody', columns=[
            'date', 'stock_id',
            '未滿400張集保張數', '超過400張集保張數', '400-600張集保張數', '600-800張集保張數', '800-1000張集保張數', '超過1000張集保張數',
            't_date',
        ])
        conditions = {
            '未滿400張': ['未滿400張集保張數'],
            '超過400張': ['超過400張集保張數'],
            '超過600張': ['600-800張集保張數', '800-1000張集保張數', '超過1000張集保張數'],
            '超過800張': ['800-1000張集保張數', '超過1000張集保張數'],
            '超過1000張': ['超過1000張集保張數'],
            }
        for new_col, cols_to_sum in conditions.items():
            df[new_col] = df[cols_to_sum].sum(axis=1)/df[['未滿400張集保張數','超過400張集保張數']].sum(axis=1)
        df = df[['date', 'stock_id', *conditions.keys(), 't_date']]
        
        return self.write_dataset(dataset='shareholding_pct', df=df)

    def _update_shareholding_pct_diff(self):
        """計算股東持股比例的變化率

        計算不同期間的股東持股比例變化。

        Returns:
            None: 將計算結果寫入 shareholding_pct_diff 資料集

        Examples:
            ```python
            db = DataBank()
            db._update_shareholding_pct_diff()
            ```

        Note:
            - 計算週差(wow_diff)、月差(mom_diff)、季差(qoq_diff)、年差(yoy_diff)
            - 對所有持股級距都計算差分
        """
        df = self.read_dataset('shareholding_pct')
        
        # calculate changes
        changes = { 'WOW_DIFF':1, 'MOM_DIFF':4, 'QOQ_DIFF':13, 'YOY_DIFF':52,}
        for col in [col for col in df.columns if col.endswith('張')]:
            for k, v in changes.items():
                df[f'{col}_{k}'] = round(df.groupby(['stock_id'])[col].diff(periods=v), 3)
        
        df = df[['date', 'stock_id'] + [col for col in df.columns for k in changes if col.endswith(k)] + ['t_date']]
        return self.write_dataset(dataset='shareholding_pct_diff', df=df)

    def _update_inst_investor_ratio_diff(self):
        """計算機構投資人持股比例的變化率

        計算外資、投信、自營商等機構投資人持股比例的變化。

        Returns:
            None: 將計算結果寫入 inst_investor_ratio_diff 資料集

        Examples:
            ```python
            db = DataBank()
            db._update_inst_investor_ratio_diff()
            ```

        Note:
            - 計算1週、1月、1季的移動平均
            - 計算週差、月差、季差、年差
            - 處理外資、投信、自營商三類機構投資人
        """
        columns = ['外資持股率', '投信持股率', '自營商持股率']
        df = self.read_dataset('trading_activity', columns=['date', 'stock_id', *columns, 't_date'])
        
        # Pre-calculate all groupby objects to avoid redundant grouping
        grouped = df.groupby('stock_id')
        
        # Calculate all rolling averages at once
        rolling_avg = {'1W': 5, '1M': 20, '1Q': 60}
        avg_cols = []
        for col in columns:
            for key, window in rolling_avg.items():
                new_col = f'{col}_{key}_avg'
                avg_cols.append(new_col)
                df[new_col] = grouped[col].transform(lambda x: x.rolling(window=window).mean())

        # Calculate all changes at once 
        changes = {'WOW_DIFF': 5, 'MOM_DIFF': 20, 'QOQ_DIFF': 60}
        diff_cols = []
        for col in columns + avg_cols:
            for key, periods in changes.items():
                new_col = f'{col}_{key}'
                diff_cols.append(new_col)
                df[new_col] = grouped[col].transform(lambda x: x.diff(periods=periods)).round(3)

        # Select columns efficiently
        keep_cols = ['date', 'stock_id'] + [col for col in df.columns if any(col.startswith(c) for c in columns)] + ['t_date']
        df = df[keep_cols].dropna(how='all', axis=1)
        
        return self.write_dataset(dataset='inst_investor_ratio_diff', df=df)

    def _update_inst_investor_money_chng(self):
        """計算機構投資人買賣超金額

        計算外資、投信、自營商等機構投資人的買賣超金額。

        Returns:
            None: 將計算結果寫入 inst_investor_money 資料集

        Note:
            - 計算1週、1月、1季的累計買賣超金額
            - 包含外資、投信、自營商(自行)、自營商(避險)
            - 金額單位為千元
        """
        columns = ['外資買賣超金額(千元)', '投信買賣超金額(千元)', 
                  '自營買賣超金額(自行)', '自營買賣超金額(避險)', '合計買賣超金額(千元)']
        df = self.read_dataset('trading_activity', columns=['date', 'stock_id', *columns, 't_date'])
        
        # Pre-calculate groupby object
        grouped = df.groupby('stock_id')
        
        # Calculate all rolling sums at once
        rolling_sum = {'1W': 5, '1M': 20, '1Q': 60}
        sum_cols = []
        
        # Process all columns and periods in a single loop
        for col in columns:
            for period_name, window in rolling_sum.items():
                new_col = f'{col}_{period_name}_sum'
                sum_cols.append(new_col)
                df[new_col] = grouped[col].transform(lambda x: x.rolling(window=window).sum())
        
        # Calculate all changes at once 
        changes = {'WOW': 5, 'MOM': 20, 'QOQ': 60}
        chng_cols = []
        for col in columns + sum_cols:
            for key, periods in changes.items():
                new_col = f'{col}_{key}'
                chng_cols.append(new_col)
                df[new_col] = grouped[col].transform(lambda x: x.pct_change(periods=periods, fill_method=None)).round(3)

        # Select columns efficiently
        keep_cols = ['date', 'stock_id'] + [col for col in df.columns if col.endswith('chng')] + ['t_date']
        df = df[keep_cols].dropna(how='all', axis=1)
        
        return self.write_dataset(dataset='inst_investor_money_chng', df=df)

    # backtest
    def _update_exp_returns(self):
        """計算個股報酬率

        Returns:
            None: 將計算結果寫入 exp_returns 資料集

        Examples:
            ```python
            db = DataBank()
            db._update_exp_returns()
            ```

        Note:
            - 計算下一期的報酬率
            - 報酬率以小數表示
            - 結果以日期為索引,股票代碼為欄位的矩陣形式儲存
        """
        df = self.read_dataset('stock_trading_data', columns=['date', 'stock_id', '報酬率'])\
            .assign(rtn=lambda df: df.groupby('stock_id')['報酬率'].shift(-1) /100)\
            [['date', 'stock_id', 'rtn']]\
            .rename(columns={'date':'t_date'})\
            .set_index(['t_date', 'stock_id'])\
            .unstack('stock_id')\
            .droplevel(0, axis=1)\
            .dropna(axis=0, how='all')
        return self.write_dataset(dataset='exp_returns', df=df)
 
    def _update_stock_sector(self):
        df = self.read_dataset('stock_trading_notes', columns=['date', 'stock_id', '主產業別(中)', '是否為臺灣50成分股', 't_date'])\
            .assign(sector=lambda df: df['主產業別(中)'].str.split(' ').str[-1].str[0:2].fillna('').replace({'其他': '其它', '証券':'證券'}))\
            .drop(columns=['主產業別(中)'])\
            .rename(columns={'是否為臺灣50成分股':'is_tw50'})\
            [['date', 'stock_id', 'sector', 'is_tw50', 't_date']]
        return self.write_dataset(dataset='stock_sector', df=df)
    
    # all/ any
    def update_processed_data(self, dataset:str=None, tej_dataset:str=None):
        """更新處理後的資料集

        根據指定的資料集或原始資料集更新對應的處理後資料。

        Args:
            dataset (str): 要更新的處理後資料集名稱
            tej_dataset (str): 要更新的原始資料集名稱

        Returns:
            None

        Examples:
            ```python
            db = DataBank()
            # 更新特定資料集
            db.update_processed_data(dataset='fin_data_chng')
            # 更新特定原始資料相關的所有處理後資料
            db.update_processed_data(tej_dataset='fin_data')
            # 更新所有資料集
            db.update_processed_data()
            ```

        Note:
            - 可以選擇更新單一資料集或全部資料集
            - 可以根據原始資料集更新相關的所有處理後資料
            - 會記錄更新日誌
        """
        if dataset:
            self.processed_datasets.get(dataset).get('func')()
            self.update_processed_data(tej_dataset=dataset)
            logging.info(f'Updated processed data for {dataset}')
        elif tej_dataset:
            for k, v in self.processed_datasets.items():
                if v.get('source') == tej_dataset:
                    self.update_processed_data(dataset=k)
                    logging.info(f'Updated processed data for {k} from {tej_dataset}')
        elif (not dataset) & (not tej_dataset):
            for k in self.processed_datasets.keys():
                self.update_processed_data(dataset=k)


class FactorModelHandler(DataBankInfra):
    def __init__(self):
        super().__init__()

    def _update_factor_model(self):
        from quantdev.analysis import calc_factor_longshort_return
        from quantdev.backtest import get_rank
        model = pd.DataFrame({
            'MKT':self._calc_market_factor(),
            'PBR':calc_factor_longshort_return(get_rank(self['股價淨值比'], asc=False), rebalance='QR'),
            'MCAP_to_REV':calc_factor_longshort_return(get_rank(self['個股市值(元)']/1000/self['營業收入'], asc=False), rebalance='QR'),
            'SIZE':calc_factor_longshort_return(get_rank(self['個股市值(元)'], asc=False), rebalance='Q'),
            'VOL':calc_factor_longshort_return(get_rank(self['成交金額(元)'].rolling(60).mean(), asc=False), rebalance='Q'),
            'MTM3m':calc_factor_longshort_return(get_rank(self['mtm_3m'], ), rebalance='Q'),
            'MTM6m':calc_factor_longshort_return(get_rank(self['mtm_6m'], ), rebalance='Q'),
            'ROE':calc_factor_longshort_return(get_rank(self['常續ROE']), rebalance='QR'),
            'OPM':calc_factor_longshort_return(get_rank(self['營業利益率']), rebalance='QR'),
            'CMA':calc_factor_longshort_return(get_rank(self['資產成長率']), rebalance='QR'),
        }).dropna(how='all')

        self.write_dataset('factor_model', model)
        logging.info('Factor model updated')

    def _calc_market_factor(self, mkt_idx:Literal['TR', None]=None):
        # define market indices mapping
        mkt_indice = {
            None: {
                'TWSE': {'stock_id': 'IX0001', 'ch_name': '加權指數'}, 
                'TPEX': {'stock_id': 'IX0043', 'ch_name': 'OTC 指數'}
            },
            'TR': {
                'TWSE': {'stock_id': 'IR0001', 'ch_name': '報酬指數'}, 
                'TPEX': {'stock_id': 'IR0043', 'ch_name': '櫃檯報酬指'}
            }
        }
        idx_ids = [v['stock_id'] for v in mkt_indice[mkt_idx].values()]

        # get market cap data
        idx_mkt_cap = self.read_dataset(
            dataset='stock_trading_data',
            columns=['date', 'stock_id', '個股市值(元)'],
            filters=[['stock_id', 'in', idx_ids]]
        ).set_index(['date', 'stock_id']).unstack('stock_id').droplevel(0, axis=1)

        # get return data 
        idx_return = self.read_dataset(
            'stock_trading_data',
            columns=['date', 'stock_id', '報酬率'],
            filters=[['stock_id', 'in', idx_ids]]
        ).set_index(['date', 'stock_id']).unstack('stock_id').droplevel(0, axis=1) / 100

        # calculate market return
        mkt_rtn = (idx_return * idx_mkt_cap).sum(axis=1) / idx_mkt_cap.sum(axis=1)

        # get risk-free rate
        rf_rate = pd.merge_asof(
            pd.DataFrame(mkt_rtn.index),
            self.read_dataset(
                'rf_rate', 
                columns=['date', 'bank', '定存利率_一年期_固定'],
                filters=[['bank', '==', '一銀']]
            ).rename(columns={'定存利率_一年期_固定': 'rf_rate'})[['date', 'rf_rate']].set_index('date')['rf_rate'],
            on='date',
            direction='backward'
        ).assign(
            rf_rate=lambda df: (1 + df['rf_rate']) ** (1 / df.groupby(df['date'].dt.year)['date'].transform('count')) - 1
        ).set_index('date')['rf_rate'].ffill()

        # calculate cumulative excess return
        return (mkt_rtn - rf_rate).shift(-1).rename_axis('t_date')


class PublicDataHandler(DataBankInfra):
    """公開資料處理類別

    處理從公開網站爬取的資料，如公司產業標籤、公司地址等。

    Args:
        stock_industry_tag_url (str): 公司產業標籤爬取網址
        stock_address_url (str): 公司地址爬取網址
        rf_rate_url (str): 無風險利率爬取網址

    Returns:
        None

    Examples:
        ```python
        # 初始化公開資料處理類別
        handler = PublicDataHandler()

        # 更新所有公開資料
        handler.update_public_data()
        ```
    """
    def __init__(self):
        super().__init__()
        self.stock_industry_tag_url = 'https://ic.tpex.org.tw/company_chain.php?stk_code='
        self.stock_address_url = 'https://mops.twse.com.tw/mops/web/t51sb01'
        self.rf_rate_url = 'https://www.cbc.gov.tw/tw/public/data/a13rate.xls'

        self.public_datasets = {
            'stock_industry_tag':{'func':self._update_stock_industry_tag},
            'rf_rate':{'func':self._update_rf_rate},
            
        }

    def _get_stock_industry_tag(self, stock_id:str):
        r = requests.get(f'{self.stock_industry_tag_url}{stock_id}')
        r.encoding = "utf-8"
        html = r.text

        bsobj = BeautifulSoup(html, "lxml")
        try:
            h_data = bsobj.find_all("h4")

            stock_tag_raw=[]
            for h in h_data:    
                stock_tag_raw.append(h.get_text().replace("\n","").replace(" ","").replace("\xa0",""))
            stock_tag_raw = [item.replace('所屬產業鏈如下:', '') for item in stock_tag_raw]
            df = pd.DataFrame({
                'stock_id':[stock_tag_raw[0][:4]],
                'stock_name':[stock_tag_raw[0][4:]], 
                'industry_tag':['\n'.join(stock_tag_raw[1:])],
                'insert_time':dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            if not df.empty:
                print(f'Fetched stock industry tag for {stock_tag_raw[0][:4]} {stock_tag_raw[0][4:]}')
        except:
            df = pd.DataFrame()
            print(f'fail to fetch stock industry tag for {stock_id}')
        return df
    
    def _update_stock_industry_tag(self):
        df = self.read_dataset('stock_basic_info', columns=['stock_id', '下市日'], filters=[('主產業別(中)', '!=', '')])
        stock_list = df[(df['下市日'].isna()) & (df['stock_id'].str.match(r'^\d{4}$'))]['stock_id'].tolist()

        df = pd.DataFrame()
        for stock_id in stock_list:
            df_temp = self._get_stock_industry_tag(stock_id)
            df = pd.concat([df, df_temp], ignore_index=True)\
                .drop_duplicates(subset=['stock_id'], keep='last')\
                .reset_index(drop=True)
        self.write_dataset(dataset='stock_industry_tag', df=df)

    def _get_stock_address(self):
        data = pd.DataFrame()
        for stock_type in ['sii', 'otc', 'rotc']:
            form_data = {
                'encodeURIComponent': 1,
                'step': 1,
                'firstin': 1,
                'off': 1,
                'TYPEK': stock_type,
            }
            r = requests.post(self.stock_address_url,form_data)
            data = pd.concat([data, pd.read_html(r.text)[9][['公司 代號', '外國企業 註冊地國', '住址']]], axis=0, ignore_index=True)
        return data\
            .rename(columns={
                '公司 代號': 'stock_id',
                '外國企業 註冊地國': 'KY_location',
                '住址': 'address'
            })\
            .query('stock_id != "公司 代號"')\
            .assign(address=lambda df: df['address'].str\
                .replace(r'^\([^)]*\)\s*', '', regex=True)\
                .replace(r'^\s*', '', regex=True)\
                .replace('巿', '市', regex=True)\
                .replace('新竹科學園區|新竹市科學工業園區|新竹科學工業園區', '', regex=True)\
                .replace('^新安路', '新竹市新安路', regex=True)\
                .replace('^創新一路', '新竹市創新一路', regex=True)\
                .replace('^力行一路', '新竹市力行一路', regex=True)\
                .replace('^創新三路', '新竹市創新三路', regex=True)\
                .replace('^創新四路', '新竹市創新四路', regex=True)\
                .replace('^北市南京東路', '台北市南京東路', regex=True)\
                .replace('^園區二路', '新竹市園區二路', regex=True)\
                .replace('^篤行路', '新竹市篤行路', regex=True)\
                .replace('^工業東九路', '新竹市工業東九路', regex=True)\
                .replace('^30078新安路', '新竹市新安路', regex=True)
            )\
            .reset_index(drop=True)

    def _update_stock_address(self):
        df = self._get_stock_address()
        df = self._add_latlng(df, 'address')
        return self.write_dataset('stock_address', df)

    def _add_latlng(self, data:pd.DataFrame, address_col:str, na_only:bool=False):
        if not na_only:
            data=data\
                .assign(latlng=data[address_col].apply(lambda x: geocoder.arcgis(x).latlng))
            
            # double check
            time.sleep(5)
        data.loc[data['latlng'].isna(), 'latlng'] = data\
            .loc[data['latlng'].isna(), address_col]\
            .apply(lambda x: geocoder.arcgis(x).latlng)
        return data

    def _update_rf_rate(self):
        rf_xls = pd.ExcelFile(self.rf_rate_url)

        # get column names
        col = rf_xls.parse('台銀')[0:4]\
            .T.\
            reset_index(drop=True)\
            [[1,2,3]]\
            [1:]
                
        for k, v in {1: None, 2: 1}.items():
            col[k] = col[k].ffill(limit=v)
        col.fillna('', inplace=True)
        col['col'] = col[[1, 2, 3]]\
            .apply(lambda x: '_'.join(filter(bool, x.astype(str))), axis=1)\
            .str.replace('\u3000', '', regex=False)

        col = ['date', *col['col'].tolist()]

        # compile data
        rf_rate = pd.DataFrame()
        for bank in rf_xls.sheet_names:
            df = rf_xls.parse(bank)[4:]
            df.columns = col
            df.insert(loc=1, column='bank', value=bank)
            rf_rate = pd.concat([rf_rate, df], ignore_index=True).reset_index(drop=True)

        # change date format
        rf_rate['date'] = pd.to_datetime(
                (rf_rate['date'].str[:3].astype(int) + 1911).astype(str) + '-' + rf_rate['date'].str[3:],
                format='%Y-%m'
            )
        rf_rate.loc[:, rf_rate.columns.difference(['date', 'bank'])] /= 100 # change digit
        rf_rate = rf_rate.sort_values(by=['date', 'bank'], ascending=True).reset_index(drop=True) # re-order
        rf_rate['t_date'] = rf_rate['date'] # insert t_date
        rf_rate['insert_time'] = pd.Timestamp.now() # insert_time

        # insert
        self.write_dataset('rf_rate', rf_rate)

    def update_public_data(self, dataset:str=None):
        if dataset:
            self.public_datasets.get(dataset).get('func')()
            logging.info(f'Updated public data for {dataset}')
        else:
            for k in self.public_datasets.keys():
                self.update_public_data(dataset=k)

        # self._update_stock_address()


class DataLoader(DataBankInfra):
    def __init__(self, sec_type:Union[str, list[str]]=['普通股']):
        super().__init__()
        self.cashe_dict = {}
        self.func_dict = {}
        self.loader_path = os.path.join(self.databank_path, 'data')
        
        # security type
        # ['封閉型基金', 'ETF', '普通股', '特別股', '台灣存託憑證', '指數', 'REIT', '國外ETF', '普通股-海外']
        if self.sec_type:
            self.sec_type = None
            sec_type_data = self['證券種類(中)']
            # sec_type_data = pd.read_parquet(os.path.join(self.loader_path, f'證券種類(中).{self.data_type}'))
            sec_type = sec_type_data.isin([self.sec_type] if isinstance(sec_type, str) else sec_type)
            self.sec_type = sec_type.loc[:, sec_type.sum()>0]

        # items to be stored
        not_item_col = ['date', 'release_date', 'stock_id', 't_date', 'insert_time']
        self.dataset_items = {
            # tej_data
            'monthly_rev':None,
            'fin_data':[col for col in self.list_columns('fin_data') if col not in [*not_item_col, '期間別', '序號', '季別', '合併(Y/N)', '幣別', '產業別',]],
            'stock_trading_data':[col for col in self.list_columns('stock_trading_data') if col not in [*not_item_col, '市場別', '調整係數(除權)']],
            'stock_trading_notes':None,
            'trading_activity':[col for col in self.list_columns('trading_activity') if col not in [*not_item_col, '市場別']],
            'stock_custody':[col for col in self.list_columns('stock_custody') if col not in [*not_item_col, 'release_date_集保庫存', 'release_date_集保股權', '市場別']],
            'self_disclosed':[col for col in self.list_columns('self_disclosed') if col not in [*not_item_col, '期間別', '序號', '季別', '合併(Y/N)', '幣別','產業別']],
            'fin_data_chng':None,
            'fin_ratio_diff':None,
            'fin_data_lag':None,
            'monthly_rev_chng':None,
            'monthly_rev_reach_ath':None,
            'monthly_rev_ath_distance':None,
            'monthly_rev_lag':None,
            'stock_momentum':None,
            'shareholding_pct':None,
            'shareholding_pct_diff':None,
            'inst_investor_ratio_diff':None,
            'inst_investor_money_chng':None,
        }
        self.dataset_items.update({k: [col for col in self.list_columns(k) if col not in not_item_col] for k,v in self.dataset_items.items() if v is None})

    def __getitem__(self, key):
        if key in self.cashe_dict:
            return self.cashe_dict[key]
        elif key in self.func_dict:
            return self.func_dict[key]
        else:
            file_path = os.path.join(self.loader_path, f'{key}.{self.data_type}')
            
            if os.path.exists(file_path):
                try:
                    if self.sec_type is not None:
                        return pd.read_parquet(file_path).reindex_like(self.sec_type)
                    else:
                        return pd.read_parquet(file_path)
                except Exception as e:
                    print(e)
                    raise ValueError(f'Cannot read {file_path}')
            raise ValueError(f'{file_path} not exist')

    def __call__(self, key):
        return self.__getitem__(key)

    def __setitem__(self, key, data:pd.DataFrame):
        file_path = os.path.join(self.loader_path, f'{key}.{self.data_type}')
        if data.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all():
            data[np.isfinite(data)].to_parquet(file_path)
        else:
            data.to_parquet(file_path)
    
    def convert_to_dataloader(self, data:Union[pd.DataFrame, str], t_date:pd.DataFrame=None, has_price:pd.DataFrame=None):
        if isinstance(data, str):
            dataset = self.find_dataset(data)
            data = self.read_dataset(dataset, columns=['t_date', 'stock_id', data])
        
        t_date = self.read_dataset('mkt_calendar', columns=['date'], filters=[('休市原因中文說明(人工建置)','=','')])\
            .rename(columns={'date':'t_date'})\
            .loc[lambda x: x['t_date'] <= pd.to_datetime(dt.datetime.today().date()+dt.timedelta(days=2))] if t_date is None else t_date

        has_price = self.read_dataset('stock_trading_data', columns=['date', 'stock_id', '收盤價'])\
            .rename(columns={'date':'t_date'})\
            .set_index(['t_date', 'stock_id'])\
            .unstack('stock_id')\
            .droplevel(0, axis=1)\
            .dropna(axis=0, how='all')\
            .notna() if has_price is None else has_price

        data = data.drop_duplicates(subset=['t_date','stock_id'], keep='last')\
            .set_index(['t_date','stock_id'])\
            .sort_index()\
            .unstack()\
            .droplevel(0, axis=1)\
            .reindex(index = t_date['t_date'],method='ffill')\
            .ffill()\
            [has_price]
        data.iloc[data.notna().any(axis=1).values.nonzero()[0][-1]:] = data.iloc[data.notna().any(axis=1).values.nonzero()[0][-1]:].ffill()

        return data
    
    def _update_dataloader(self, datasets:Union[str, list[str]]=None):
        t_date = self.read_dataset('mkt_calendar', columns=['date'], filters=[('休市原因中文說明(人工建置)','=','')])\
            .rename(columns={'date':'t_date'})\
            .loc[lambda x: x['t_date'] <= pd.to_datetime(dt.datetime.today().date())]
        has_price = self.read_dataset('stock_trading_data', columns=['date', 'stock_id', '收盤價'])\
            .rename(columns={'date':'t_date'})\
            .set_index(['t_date', 'stock_id'])\
            .unstack('stock_id')\
            .droplevel(0, axis=1)\
            .notna()

        if datasets:
            data_items = [(datasets, self.dataset_items[datasets]) for datasets in datasets]
        else:
            data_items = self.dataset_items.items()
            
        for dataset_name, columns in data_items:
            data = self.read_dataset(dataset_name)
            for item in columns:
                self[item] = self.convert_to_dataloader(data[['t_date', 'stock_id', item]], t_date, has_price)

    def list_dataloader(self):
        parquet_set = set(filter(lambda X:X.endswith(f".{self.data_type}"),os.listdir(self.loader_path)))
        return sorted(map(lambda X:X[:-(len(self.data_type)+1)],list(parquet_set)))


class Databank(TEJHandler, ProcessedDataHandler, PublicDataHandler, FactorModelHandler, DataLoader):
    """quantdev 資料庫

    整合 TEJ、FinMind 與公開資料的資料處理功能。

    Args:
        databank_path (str): 資料庫路徑
        tej_token (str): TEJ API 金鑰
        tej_datasets (dict): TEJ 資料集對照表
        processed_datasets (dict): 處理資料集設定
        finmind_token (str): FinMind API 金鑰

    Examples:
        ```python
        # 初始化資料庫
        db = Databank()

        # 更新資料庫
        db.update_databank()

        # 讀取特定資料集
        df = db.read_dataset('monthly_rev')

        # 讀取特定資料集的資料
        df = db.read_dataset('fin_data', columns=['stock_id', 'date', 'revenue'])

        # 更新特定 TEJ 資料集
        db.update_tej_dataset('monthly_rev')

        # 更新處理後的資料
        db.update_processed_data(dataset='fin_data_chng')

        # 更新公開資料
        db.update_public_data()

        # 寫入資料到資料庫
        db.write_dataset('custom_data', df)

        ```

    Note:
        - 需要在 config 中設定相關 API 金鑰
        - 資料庫更新會自動處理資料相依性
    """
    
    def __init__(self, sec_type:Union[str, list[str]]=['普通股']):
        self.sec_type = sec_type
        super().__init__()

    
    # basic
    def update_databank(self, include:list=[], exclude:list[str]=['stock_industry_tag', 'rf_rate'], update_processed=True, update_dataloader=True):
        logging.info('Starting databank update')
        
        if include:
            if isinstance(include, str):
                datasets = [include]
            else:
                datasets = include
        else:
            datasets = [d for d in self.tej_datasets.keys() if d not in exclude]

        # tej
        for dataset in datasets:
            self.update_tej_dataset(dataset)
                        
            # processed
            if update_processed:
                self.update_processed_data(tej_dataset=dataset)
            # dataloader
            if update_dataloader:
                # update dataloader for both processed and tej datasets
                self._update_dataloader(
                    datasets=[k for k, v in Databank(sec_type=None).processed_datasets.items() if (k in include or v['source'] in include)]
                )

        # public
        for dataset in self.public_datasets.keys():
            if ((not exclude) and (not include)) or (dataset not in exclude) or (dataset in include):
                self.update_public_data(dataset=dataset)
        
        # factor model
        if (('factor_model' in include) & (include != [])) or\
           (('factor_model' not in exclude) & (include==[])):
            self._update_factor_model()

        logging.info('Completed databank update')