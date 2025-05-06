from concurrent.futures import ThreadPoolExecutor
from IPython.display import HTML, display
from typing import Union, Literal
import pyarrow.parquet as pq
import pyarrow as pa
import datetime as dt
import pandas as pd
import numpy as np
import logging
import tejapi
import os
import re

# config
from .config import config

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.set_option('future.no_silent_downcasting', True)

class DataUtils():
    def __init__(self):
        self.datasets_path = config.data_config.get('datasets_path')
        self.databank_path = config.data_config.get('databank_path')
        os.makedirs(self.databank_path, exist_ok=True)
        os.makedirs(self.datasets_path, exist_ok=True)
        
        self.start_date = '2005-01-01'
        self.data_type = 'parquet'
    
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
            db.list_columns('trading_data')
            ```

            output:
            ```
            ['收盤價', '開盤價', '最高價', '最低價', ...]
            ```
        """
        
        path = os.path.join(self.datasets_path, f'{dataset}.{self.data_type}')
        if (keyword is not None) and (os.path.exists(path)):
            return [col for col in pq.ParquetFile(path).schema.names if keyword in col]
        elif not os.path.exists(path):
            return []
        else:
            return pq.ParquetFile(path).schema.names

    def list_datasets(self, keyword:str=None):
        datasets = [file.replace(f'.{self.data_type}', '') \
                    for file in os.listdir(self.datasets_path) \
                    if file.endswith(f'.{self.data_type}')]
        return datasets if keyword is None else [d for d in datasets if keyword in d]

    def read_dataset(self, dataset:str, filter_date:str='date', start:Union[int, str]=None, end:Union[int, str]=None, columns:list=None, filters:list=None) -> pd.DataFrame:
        # Process date filters
        filters = filters or []
        if start:
            start_date = dt.datetime(start, 1, 1) if isinstance(start, int) else pd.to_datetime(start)
            filters.append((filter_date, '>=', start_date))
        if end:
            end_date = dt.datetime(end, 12, 31) if isinstance(end, int) else pd.to_datetime(end)
            filters.append((filter_date, '<=', end_date))

        # Read data
        path = os.path.join(self.datasets_path, f'{dataset}.{self.data_type}')
        if not os.path.exists(path):
            raise FileNotFoundError(f'{dataset}.{self.data_type} not exist')
        
        return pq.read_table(source=path, columns=columns, filters=filters or None).to_pandas()

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
        path = os.path.join(self.datasets_path, f'{dataset}.{self.data_type}')
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path)
        logging.info(f'Saved {dataset} as parquet from DataFrame')


class TEJHandler(DataUtils):
    def __init__(self):
        super().__init__()
        tejapi.ApiConfig.api_key = config.data_config.get('tej_token')
        tejapi.ApiConfig.ignoretz = True
        self.tej_datasets_map = {
            'trading_activity_w': 'TWN/APISHRACTW',
            'trading_activity': 'TWN/APISHRACT',
            'trading_notes': 'TWN/APISTKATTR',
            'trading_data': 'TWN/APIPRCD',
            'monthly_rev': 'TWN/APISALE1',
            'fin_data': 'TWN/AINVFQ1',
            'fin_data_self_disclosed': 'TWN/AFESTM1',
            'capital_formation_listed': 'TWN/ASTK1',
            'capital_formation': 'TWN/APISTK1',
            'div_policy': 'TWN/APIMT1',
            'cash_div': 'TWN/ADIV',
            'twn_rates': 'TWN/ARATE',
            'glbl_rates': 'GLOBAL/WIBOR1',
            'stock_info': 'TWN/APISTOCK',
            'mkt_calendar': 'TWN/TRADEDAY_TWSE',
            'index_components': 'TWN/EWISAMPLE',
        }
    
    # datasets info
    def get_tej_datasets_info(self):
        available_datasets = [k for k in tejapi.ApiConfig.info()['user']['tables'].keys() if k!='TWN/APISALE']
        new_available = [t for t in available_datasets if t not in self.tej_datasets_map.values()]
        not_available = [t for t in self.tej_datasets_map.values() if t not in available_datasets]

        if any([new_available, not_available]):
            raise ValueError(f"datasets_map/ TEJ available datasets mismatch: new: {new_available}, no longer available: {not_available}")

        def get_dataset_info(item):
            k, v = item
            table_info = tejapi.table_info(v)
            return k, {'en':v, 'cn': table_info['name'], 'description': re.sub(r'(<br\s*/?>)+', '<br />', table_info['description'])}
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            info = dict(executor.map(get_dataset_info, self.tej_datasets_map.items()))
        
        return info

    def display_tej_datasets(self, keyword:str=None)-> dict:
        datasets = self.get_tej_datasets_info() if keyword is None \
            else {k:v for k, v in self.get_tej_datasets_info().items() if any([keyword in v['cn'], keyword in v['en'], keyword in v['description'], keyword in k])}
        for dataset in datasets:
            display(HTML(
                dataset + '<br>' + \
                datasets[dataset]['en'] + ' ' + datasets[dataset]['cn'] + '<br>' + \
                datasets[dataset]['description']     
            ))
        
    # get & process raw data
    def get_tej_data(self, dataset:str, start_date:str=None, end_date:str=None, stock_id:Union[str, list[str]]=None, columns:list=None)-> pd.DataFrame:
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
        
        # start_date, end_date
        mdate = {
            'gte': start_date,
            'lte': end_date,
        } if dataset not in ['mkt_calendar', 'stock_info'] else None
        
        # columns
        opts = {'columns':columns} if columns else {}

        #fast get
        data = tejapi.fastget(
            datatable_code = self.tej_datasets_map[dataset], 
            coid=stock_id, 
            mdate=mdate, 
            chinese_column_name=True, 
            paginate=True, 
            opts=opts, 
        )
        
        # process data
        if not data.empty:
            data = self.process_tej_data(dataset, data)

            # log
            if 'date' in data.columns:
                logging.info(f'Fetched {dataset} from {data.date.min().strftime("%Y-%m-%d")} to {data.date.max().strftime("%Y-%m-%d")} from TEJ', )
            else:
                logging.info(f'Fetched {dataset} from TEJ')
        return data
    
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
                dataset='trading_data',
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
            if self.reach_tej_limit():
                break
            
            # continue
            current_start_date = current_end_date + dt.timedelta(days=1)
        
        return self.bulk_tej_data_temp

    def add_t_date(self, data:pd.DataFrame) -> pd.DataFrame:
        """將資料加入交易日期

        將資料的日期或發布日期對應到下一個交易日，用於計算資訊揭露後的市場反應

        Args:
            data (pd.DataFrame): 要加入交易日期的 DataFrame

        Returns:
            pd.DataFrame: 加入交易日期後的 DataFrame，新增欄位:
                - t_date: 下一個交易日期

        Examples:
            ```python
            # 將月營收資料加入交易日期
            db = DataBank()
            data = db.read_dataset('monthly_rev')
            data = db.add_t_date(data)
            ```

        Note:
            - 若資料有 release_date 欄位，則以 release_date 為基準找下一個交易日
            - 若無 release_date 欄位，則以 date 欄位為基準找下一個交易日
            - 若資料已有 t_date 欄位，則會先刪除再重新計算
        """
        
        # import trade date
        t_date = self.read_dataset('mkt_calendar', columns=['date'], filters=[('休市原因中文說明(人工建置)','=','')]).rename(columns={'date':'t_date'})
        if t_date.empty:
            self.update_tej_datasets('mkt_calendar')
            t_date = self.read_dataset('mkt_calendar', columns=['date'], filters=[('休市原因中文說明(人工建置)','=','')]).rename(columns={'date':'t_date'})
        
        date_col = 'release_date' if 'release_date' in data.columns else 'date'
        
        data = data\
            .drop(columns='t_date', errors='ignore')\
            .assign(next_day=lambda x: (x[date_col] + pd.DateOffset(days=1)).astype('datetime64[ms]'))\
            .sort_values('next_day')\
            .reset_index(drop=True)
        data = pd.merge_asof(
            data, 
            t_date, 
            left_on='next_day', 
            right_on='t_date', 
            direction='forward'
        ).drop('next_day', axis=1)
        
        return data

    def process_tej_data(self, dataset:str, data:pd.DataFrame):
        # rename columns
        rename_columns = {
            '公司':'stock_id',
            '證券名稱':'stock_id',
            '證券碼':'stock_id',
            '年月':'date',
            '年/月':'date', 
            '資料日':'date',
            '日期':'date',
            '除息日':'date',
            '營收發布日':'release_date',
            '編表日':'release_date',
            '除息公告日':'release_date',
            '公告日(集保庫存)':'release_date_集保庫存',
            '公告日(集保股權)':'release_date_集保股權',
        }
        if dataset !='stock_info':
            data = data.rename(columns=rename_columns)
        else:
            data = data.rename(columns={'公司簡稱':'stock_id'})

        # leave only 期間別 =='Q' & 序號=='001' for self disclosed and financial data
        if dataset in ['fin_data', 'fin_data_self_disclosed']:
            data = data[(data['期間別']=='Q')&(data['序號']=='001')]
        
        # rename 財務資料_公司自結數
        if dataset=='fin_data_self_disclosed':
            data = data\
                .rename(columns={col: f'{col}_自' for col in data.columns \
                                if col not in ['date', 'stock_id', 'release_date', '期間別', '序號', '季別', '合併(Y/N)', '幣別', '產業別', 't_date']})
        # fillna for monthly rev release date
        elif dataset=='monthly_rev':
            data = data\
                .assign(release_date=lambda x: x['release_date'].fillna(x['date'] + pd.DateOffset(days=9, months=1)))\
                .rename(columns={col: col.replace('/', '_') for col in data.columns if '/' in col})
        # stock calendar
        elif dataset=='mkt_calendar':
            data = data[(data['date'] >= self.start_date) & (data['date'].dt.year <= dt.datetime.now().year+1)]

        # reorder front cols
        front_cols=['date', 'stock_id', 'release_date']
        data=data[[col for col in front_cols if col in data.columns] + [col for col in data.columns if col not in front_cols]]

        # dtype
        for col, dtype in {'date': 'datetime64[ms]', 'release_date': 'datetime64[ms]', 'stock_id': str}.items():
            if col in data.columns:
                data = data.assign(**{col: lambda x, c=col, dt=dtype: x[c].astype(dt)})
        
        # sort value
        sort_columns = [col for col in ['date', 'stock_id'] if col in data.columns]
        if sort_columns:
            data = data.sort_values(by=sort_columns, ascending=True).reset_index(drop=True)
        
        # add t_date
        if (dataset != 'mkt_calendar') & ('date' in data.columns):
            data = self.add_t_date(data)

        return data

    def deal_with_adj_fac(self, data:pd.DataFrame):
        adj_fac_cols = {'調整係數':'adjfac', '調整係數(除權)':'adjfac_a'}
        adj_fac_data = self.get_tej_data(
            dataset='trading_data', 
            columns=['mdate', 'coid', *list(adj_fac_cols.values())], 
            start_date=data['date'].min().strftime('%Y-%m-%d'), 
            end_date=data['date'].max().strftime('%Y-%m-%d'), 
        )\
            .set_index(['date', 'stock_id'])\
            [adj_fac_cols.keys()]\
            .rename(columns=adj_fac_cols)
        return pd.concat([
            data\
                .drop_duplicates(subset=['date', 'stock_id'], keep='last')\
                .set_index(['date', 'stock_id']), 
            adj_fac_data], axis=1)\
            .assign(**{k: lambda x: x[v] for k, v in adj_fac_cols.items()})\
            .drop(columns=list(adj_fac_cols.values()))\
            .dropna(subset=['insert_time'])\
            .reset_index()
    
    # tej api limit
    def reach_tej_limit(self, print_:bool=False) -> bool:
        api_info = tejapi.ApiConfig.info()
        req_usage = api_info['todayReqCount'] / api_info['reqDayLimit']
        row_usage = api_info['todayRows'] / api_info['rowsDayLimit']
        logging.info(f'TEJ API usage: {req_usage:.2%}, {row_usage:.2%}')
        if print_:
            print(f'TEJ API usage: {req_usage:.2%}, {row_usage:.2%}')
        return any([(row_usage >= 0.95), (req_usage >= 0.95)])
    
    # update tej datasets
    def update_tej_datasets(self, dataset:Union[str, list[str]]=None, end_date:str=None, back_period:int=2):
        """更新TEJ資料集

        Args:
            dataset (str): 要更新的資料集名稱
            end_date (str, optional): 更新資料的結束日期. Defaults to None.
            back_period (int, optional): 回溯期間. Defaults to 2.
        Examples:
            ```python
            # 更新股票交易資料
            db.update_tej_dataset('trading_data')

            # 更新到指定日期的財務資料
            db.update_tej_dataset('fin_data', end_date='2023-12-31')
            ```

        Note:
            - 對於交易相關資料(trading_data等)，若更新區間超過365天，會使用批量下載模式
            - 股票交易資料會自動處理除權息調整係數
            - 所有資料會加入insert_time欄位記錄寫入時間
            - 重複資料會保留最新一筆
        """
        datasets = list(self.tej_datasets_map.keys())
        if dataset is None:
            for d in datasets:
                self.update_tej_datasets(d, end_date=end_date, back_period=back_period)
        elif isinstance(dataset, list):
            for d in dataset:
                self.update_tej_datasets(d, end_date=end_date, back_period=back_period)
        elif isinstance(dataset, str):
            if dataset not in datasets:
                raise ValueError(f'{dataset} is not in tej_datasets')

            # mkt calendar, stock basic info, equity indice components, and 除權息增減資-related datasets
            if dataset in ['mkt_calendar', 'stock_info', 'index_components', 'capital_formation', 'capital_formation_listed', 'div_policy', 'cash_div']:
                if (not self.read_dataset(dataset, columns=['insert_time']).empty) and \
                   (self.read_dataset(dataset, columns=['insert_time'])['insert_time'].min().date() == dt.date.today()):
                    pass
                else:
                    data = self.get_tej_data(dataset)
                    data['insert_time'] = dt.datetime.now()
                    self.write_dataset(dataset, data)
            
            # others
            else:
                # old data
                try:
                    old_data = self.read_dataset(dataset)
                except Exception as e:
                    print(e)
                    old_data = pd.DataFrame()

                start_date = self.start_date if old_data.empty else pd.Series(old_data['date'].unique()).nlargest(back_period).min().strftime('%Y-%m-%d')
                end_date = dt.date.today().strftime('%Y-%m-%d') if end_date is None else end_date
                
                # get data, process data
                if (dataset in ['trading_data', 'trading_notes', 'trading_activity', 'trading_activity_w']) & \
                    ((dt.datetime.strptime(end_date, "%Y-%m-%d") - dt.datetime.strptime(start_date, "%Y-%m-%d")).days >= 366):
                    new_data = self.get_tej_data_bulk(dataset, start_date=start_date, end_date=end_date)
                else:
                    new_data = self.get_tej_data(dataset, start_date=start_date, end_date=end_date)
                
                if not new_data.empty:
                    new_data['insert_time'] = dt.datetime.now()

                    # Concatenate old and new data
                    old_data = pd.concat([
                        old_data.dropna(axis=1, how='all'), 
                        new_data.dropna(axis=1, how='all'),
                    ], ignore_index=True)
                    
                    # Deal with adj_fac
                    if (dataset == 'trading_data') & \
                        (not old_data.empty) & \
                        (old_data['insert_time'].min().date() < dt.date.today()):
                        old_data = self.deal_with_adj_fac(old_data)
                    
                    # Reorder columns to move t_date and insert_time to the end
                    last_cols = [col for col in old_data.columns if col in ['t_date', 'insert_time']]
                    old_data = old_data[[col for col in old_data.columns if col not in last_cols] + last_cols]
                    
                    # drop_duplicates
                    subset = [col for col in old_data.columns if col in ['date', 'stock_id']]
                    old_data = old_data\
                        .drop_duplicates(subset=subset, keep='last')\
                        .reset_index(drop=True)

                    # insert
                    self.write_dataset(dataset, old_data)
        
            logging.info(f'Updated {dataset} from tej')


class ProcessedHandler(DataUtils):
    def __init__(self):
        super().__init__()
        self.processed_datasets = {
            'fin_data_lag':{'source':'fin_data', 'func':self.update_fin_data_lag},
            'fin_data_ath':{'source':'fin_data', 'func':self.update_fin_data_ath},
            'monthly_rev_lag':{'source':'monthly_rev', 'func':self.update_monthly_rev_lag},
            'monthly_rev_ath':{'source':'monthly_rev', 'func':self.update_monthly_rev_ath},
            'exp_returns':{'source':'trading_data', 'func':self.update_exp_returns}, 
            'exp_returns_daytrade':{'source':'trading_data', 'func':self.update_exp_returns_daytrade},
            'trading_activity_w_pct':{'source':'trading_activity_w', 'func':self.update_trading_activity_w_pct},
            'trading_activity_w_pct_lag':{'source':'trading_activity_w_pct', 'func':self.update_trading_activity_w_pct_lag},
        }

    # financial data
    def update_fin_data_lag(self, lag_period:int=8):
        ignore_cols = ['期間別', '序號', '季別', '合併(Y/N)', '幣別', '產業別', 'date', 'release_date', 'stock_id', 't_date', 'insert_time']
        
        
        lag_columns = {
            f'{col}_lag{i}': lambda df, i=i, col=col: df.groupby('stock_id')[col].shift(i)
            for i in range(1, lag_period+1) for col in [col for col in self.list_columns('fin_data') if col not in ignore_cols]
        }

        df = self.read_dataset('fin_data')\
            .assign(**lag_columns)
        lag_cols = [col for col in df.columns if 'lag' in col]
        df = df[['date', 'stock_id', 'release_date', *lag_cols, 't_date']].dropna(subset=lag_cols, how='all')
        return self.write_dataset(dataset='fin_data_lag', df=df)

    def update_fin_data_ath(self, selected_cols:list[str]=[
        '常續ROE', '常續ROA', 
        '營業收入', '營業毛利', '稅前淨利', '稅後淨利', '營業利益', '資產總計', 
        '營業毛利率', '營業利益率', '稅前淨利率', '稅後淨利率', 
        '員工人數', '生財設備',
    ]):
        df = self.read_dataset('fin_data', columns=['date', 'stock_id', 'release_date', *selected_cols, 't_date'])

        def calc_ath_months(x):
            curr_val = x.iloc[-1]
            for i in range(len(x)-1, -1, -1):
                if x.iloc[i] > curr_val:
                    return len(x) - i - 1
            return len(x)
                
        for col in selected_cols:
            df[f'{col}_ATH'] = df\
                .groupby('stock_id')\
                [col]\
                .transform(lambda x: x.expanding().apply(calc_ath_months))
            is_ath = df.groupby('stock_id')[col].transform(lambda x: x.iloc[4:] == x.iloc[4:].expanding().max() if len(x) > 4 else False).fillna(False)
            df.loc[is_ath, f'{col}_ATH'] = np.inf
        
        return self.write_dataset(dataset='fin_data_ath', df=df[['date', 'stock_id', 'release_date', *[col for col in df.columns if col.endswith('_ATH')], 't_date']])

    def update_monthly_rev_lag(self, lag_period:int=24):
        lag_columns = {
            f'單月營收(千元)_lag{i}': lambda df, i=i: df.groupby('stock_id')['單月營收(千元)'].shift(i)
            for i in range(1, lag_period+1)
        }
        df = self.read_dataset(
                dataset='monthly_rev', 
                columns=['date', 'stock_id', '單月營收(千元)', 't_date'])\
            .assign(**lag_columns)
        lag_cols = [col for col in df.columns if 'lag' in col]
        df = df[['date', 'stock_id', *lag_cols, 't_date']].dropna(subset=lag_cols, how='all')
        return self.write_dataset(dataset='monthly_rev_lag', df=df)
    
    def update_monthly_rev_ath(self):
        df = self.read_dataset('monthly_rev', columns=['date', 'stock_id', 'release_date', '單月營收(千元)', 't_date'])
        
        df['單月營收_ATH'] = np.nan
        
        def calc_ath_months(x):
            curr_val = x.iloc[-1]
            for i in range(len(x)-1, -1, -1):
                if x.iloc[i] > curr_val:
                    return len(x) - i - 1
            return len(x)
            
        df['單月營收_ATH'] = df\
            .groupby('stock_id')\
            ['單月營收(千元)']\
            .transform(lambda x: x.expanding().apply(calc_ath_months))
        
        is_ath = df.groupby('stock_id')['單月營收(千元)'].transform(lambda x: x.iloc[12:] == x.iloc[12:].expanding().max() if len(x) > 12 else False).fillna(False)
        df.loc[is_ath, '單月營收_ATH'] = np.inf

        return self.write_dataset(dataset='monthly_rev_ath', df=df[['date', 'stock_id', 'release_date', '單月營收_ATH', 't_date']])

    # backtest 
    def update_exp_returns(self):
        df = self.read_dataset('trading_data', columns=['date', 'stock_id', '報酬率'])\
            .assign(rtn=lambda df: df.groupby('stock_id')['報酬率'].shift(-1) /100)\
            [['date', 'stock_id', 'rtn']]\
            .rename(columns={'date':'t_date'})\
            .set_index(['t_date', 'stock_id'])\
            .unstack('stock_id')\
            .droplevel(0, axis=1)\
            .dropna(axis=0, how='all')
        return self.write_dataset(dataset='exp_returns', df=df)
    
    def update_exp_returns_daytrade(self):
        df = self.read_dataset('trading_data', columns=['date', 'stock_id', '開盤價', '收盤價'])\
            .assign(rtn=lambda df: df['收盤價']/df['開盤價']-1)\
            [['date', 'stock_id', 'rtn']]\
            .rename(columns={'date':'t_date'})\
            .set_index(['t_date', 'stock_id'])\
            .unstack('stock_id')\
            .droplevel(0, axis=1)\
            .replace([np.inf, -np.inf], np.nan)\
            .dropna(axis=0, how='all')
        return self.write_dataset(dataset='exp_returns_daytrade', df=df)
    
    # trading_activity_w
    def update_trading_activity_w_pct(self):
        df = self.read_dataset('trading_activity_w', columns=[
            'date', 'stock_id',
            '未滿400張集保張數', '超過400張集保張數', '400-600張集保張數', '600-800張集保張數', '800-1000張集保張數', '超過1000張集保張數',
            't_date',
        ])
        conditions = {
            '未滿400張_pct': ['未滿400張集保張數'],
            '超過400張_pct': ['超過400張集保張數'],
            '超過600張_pct': ['600-800張集保張數', '800-1000張集保張數', '超過1000張集保張數'],
            '超過800張_pct': ['800-1000張集保張數', '超過1000張集保張數'],
            '超過1000張_pct': ['超過1000張集保張數'],
            }
        for new_col, cols_to_sum in conditions.items():
            df[new_col] = df[cols_to_sum].sum(axis=1)/df[['未滿400張集保張數','超過400張集保張數']].sum(axis=1)
        df = df[['date', 'stock_id', *conditions.keys(), 't_date']]
        
        return self.write_dataset(dataset='trading_activity_w_pct', df=df)

    def update_trading_activity_w_pct_lag(self, lag_period:int=12):
        cols_to_lag = ['未滿400張_pct', '超過400張_pct', '超過600張_pct', '超過800張_pct', '超過1000張_pct']
        lag_columns = {
            f'{col}_lag{i}': lambda df, i=i, col=col: df.groupby('stock_id')[col].shift(i)
            for i in range(1, lag_period+1) for col in cols_to_lag
        }
        df = self.read_dataset('trading_activity_w_pct')\
            .assign(**lag_columns)
        lag_cols = [col for col in df.columns if 'lag' in col]
        df = df[['date', 'stock_id', *lag_cols, 't_date']].dropna(subset=lag_cols, how='all')
        return self.write_dataset(dataset='trading_activity_w_pct_lag', df=df)

    # update
    def update_processed_datasets(self):
        for v in self.processed_datasets.values():
            v['func']()

     
class FactorModelHandler(DataUtils):
    def __init__(self):
        super().__init__()
    
    def update_factor_model(self):
        from quantdev.backtest import calc_factor_longshort_return
        data = Databank()
        model = pd.DataFrame({
            'MKT':self.calc_market_factor(),
            'PBR':calc_factor_longshort_return((-1*data['股價淨值比']).to_factor().to_rank(), rebalance='QR'),
            'MCAP_to_REV':calc_factor_longshort_return((-1*data['個股市值(元)']/1000/data['營業收入']).to_factor().to_rank(), rebalance='QR'),
            'SIZE':calc_factor_longshort_return((data['個股市值(元)']).to_factor().to_rank(), rebalance='Q'),
            'VOL':calc_factor_longshort_return((data['成交金額(元)'].rolling(60).mean()).to_factor().to_rank(), rebalance='Q'),
            'MTM3m':calc_factor_longshort_return(((data['收盤價']/data['收盤價'].shift(60))*data['調整係數']-1).to_factor().to_rank(), rebalance='Q'),
            'ROE':calc_factor_longshort_return((data['常續ROE']).to_factor().to_rank(), rebalance='QR'),
            'OPM':calc_factor_longshort_return((data['營業利益率']).to_factor().to_rank(), rebalance='QR'),
        }).dropna(how='all')

        self.write_dataset('factor_model', model)
        logging.info('Factor model updated')

    def calc_market_factor(self, mkt_idx:Literal['TR', None]=None):
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
            dataset='trading_data',
            columns=['date', 'stock_id', '個股市值(元)'],
            filters=[['stock_id', 'in', idx_ids]]
        ).set_index(['date', 'stock_id']).unstack('stock_id').droplevel(0, axis=1).shift(1)

        # get return data 
        idx_return = self.read_dataset(
            'trading_data',
            columns=['date', 'stock_id', '報酬率'],
            filters=[['stock_id', 'in', idx_ids]]
        ).set_index(['date', 'stock_id']).unstack('stock_id').droplevel(0, axis=1) / 100

        # calculate market return
        mkt_rtn = (idx_return * idx_mkt_cap).sum(axis=1) / idx_mkt_cap.sum(axis=1)
        

        # get risk-free rate
        rf_rate = self.read_dataset('TWN_rates', filters=[['銀行別', '==', '5844']], columns=['date', '一年定存'])\
            .set_index('date')\
            .reindex(index = mkt_rtn.index,method='ffill')\
            .reset_index()\
            .assign(rf_rate=lambda df: (1 + df['一年定存']/100) ** (1 / df.groupby(df['date'].dt.year)['date'].transform('count')) - 1)\
            .set_index('date')\
            ['rf_rate']\
            .ffill()

        # calculate cumulative excess return
        return (mkt_rtn - rf_rate).shift(-1).rename_axis('t_date')


class DatasetsHandler(TEJHandler, ProcessedHandler, FactorModelHandler):
    def __init__(self):
        super().__init__()
    
    def update_datasets(self):
        self.update_tej_datasets()
        self.update_processed_datasets()
        self.update_factor_model()


class Databank(DataUtils):
    def __init__(self, sec_type:Union[str, list[str]]=['普通股']):
        super().__init__()
        self._ignore()
        self.cashe_dict = {}
        self.func_dict = {}
        self.sec_type = None
        if sec_type:
            sec_type = [sec_type] if isinstance(sec_type, str) else sec_type
            sec_types = ['封閉型基金', 'ETF', '普通股', '特別股', '台灣存託憑證', '指數', 'REIT', '國外ETF', '普通股-海外']
            if any(t not in sec_types for t in sec_type):
                raise ValueError(f'Security type must be one of {sec_types}')
            stdata = self['證券種類(中)']\
                .isin(sec_type)
            self.sec_type = stdata\
                .loc[:, stdata.sum()>0]
    
    def _ignore(self):
        # ignore
        self.ingore_datasets = [
            'exp_returns', 'exp_returns_daytrade', 
            'div_policy', 'cash_div', 'capital_formation', 'capital_formation_listed',
            'glbl_rates', 'twn_rates', 'index_components',
            'stock_info', 'mkt_calendar', 
        ]
        self.ignore_cols = [
            'date', 'release_date', 'stock_id', 't_date', 'insert_time', 'MTM3m'
            '期間別', '序號', '季別', '合併(Y/N)', '幣別', '產業別', '市場別', '營收發布時點', '備註說明', 'release_date_集保股權', 'release_date_集保庫存', 
        ]

    def find(self, keyword:str):
        datasets = [d for d in self.list_datasets() if d not in self.ingore_datasets]
        columns = [c for d in datasets for c in self.list_columns(d) if c not in self.ignore_cols]
        return [c for c in columns if keyword in c]
    
    def find_dataset(self, key:str):
        datasets = [d for d in self.list_datasets() if d not in self.ingore_datasets]
        map = {d:[c for c in self.list_columns(d) if c not in self.ignore_cols] for d in datasets}
        target_datasets = [k for k, v in map.items() if key in v]
        if len(target_datasets) >1:
            raise ValueError(f'{key} appears in more than one dataset: {target_datasets}')
        elif len(target_datasets) == 0:
            raise ValueError(f'{key} not found in any dataset')
        else:
            return target_datasets[0]
    
    def get_t_date(self):
        t_date = DatasetsHandler().read_dataset('mkt_calendar', columns=['date'], filters=[('休市原因中文說明(人工建置)','=','')])\
            .rename(columns={'date':'t_date'})
        nearest_next_t_date = t_date\
            .loc[lambda x: x['t_date'] > pd.to_datetime(dt.datetime.today().date())]\
            .iloc[0]['t_date']
        return t_date.loc[lambda x: x['t_date'] <= nearest_next_t_date]

    def unstack_data(self, data:pd.DataFrame, t_date:pd.DataFrame=None, has_price:pd.DataFrame=None):
        t_date = self.get_t_date() if t_date is None else t_date

        has_price = self.read_dataset('trading_data', columns=['date', 'stock_id', '收盤價'])\
            .rename(columns={'date':'t_date'})\
            .set_index(['t_date', 'stock_id'])\
            .unstack('stock_id')\
            .droplevel(0, axis=1)\
            .dropna(axis=0, how='all')\
            .notna() if has_price is None else has_price

        unstacked_data = data\
            .drop_duplicates(subset=['t_date','stock_id'], keep='last')\
            .set_index(['t_date','stock_id'])\
            .sort_index()\
            .unstack()\
            .droplevel(0, axis=1)\
            .reindex(index = t_date['t_date'],method='ffill')\
            .ffill()\
            [has_price]
        return unstacked_data
    
    def to_databank(self, key:str):
        dataset = self.find_dataset(key)
        data = self.read_dataset(dataset, columns=['t_date', 'stock_id', key])
        data = self.unstack_data(data)
        self[key] = data

    def __setitem__(self, key:str, value:pd.DataFrame):
        path = os.path.join(self.databank_path, f'{key}.{self.data_type}')
        value.to_parquet(path)

    def __getitem__(self, key):
        if key in self.cashe_dict:
            return self.cashe_dict[key]
        elif key in self.func_dict:
            return self.func_dict[key]
        else:
            path = os.path.join(self.databank_path, f'{key}.{self.data_type}')
            
            if os.path.exists(path):
                try:
                    if pq.read_table(path, columns=['t_date']).to_pandas().index.max() \
                       < pd.Timestamp(dt.datetime.today().date()):
                        self.to_databank(key)
                    if self.sec_type is not None:
                        return pd.read_parquet(path).reindex_like(self.sec_type)
                    else:
                        return pd.read_parquet(path)
                except Exception as e:
                    raise ValueError(f'Cannot read {path}, error: {e}')
            else:
                self.to_databank(key)
                return self[key]

    def __call__(self, key):
        return self.__getitem__(key)
