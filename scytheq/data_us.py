from eodhd import APIClient
from typing import Literal
import pandas as pd
import numpy as np
import tqdm
import time
from .config import config
from joblib import Parallel, delayed
from .data import DataKit
import os

class US_DataHandler(DataKit):
    def __init__(self, universe=['etf', 'stock']):
        self.databank_path = config.us_data_config["databank_path"]
        self.token = config.us_data_config["eodhd_token"]
        self.client = APIClient(self.token)
        self.stock_info = pd.DataFrame(self.client.get_list_of_tickers("US"))\
            .query("Exchange.isin(['NYSE ARCA', 'BATS', 'NASDAQ','NYSE','AMEX', 'NYSE MKT'])")\
            .reset_index(drop=True)
        self.etf_list = self.stock_info.query("Type == 'ETF'")['Code'].tolist()
        self.exclude_stock_list = ['OKLL', 'SCD-R', 'IONZ', 'BUFX', 'BUFH', 'RAL-W']
        self.stock_list = list(set(self.stock_info.query("Type == 'Common Stock'")['Code'].tolist()) - set(self.exclude_stock_list))
        universe = [universe] if isinstance(universe, str) else universe
        for _ in universe:
            self.update_databank(universe=_)

    def get_eodhd_data_single_stock(self, ticker: str, start_date: str = '1990-01-01', end_date: str = f'{pd.Timestamp.today().date()}') -> pd.DataFrame:
        for _ in range(5):
            try:
                data = self.client.get_eod_historical_stock_market_data(
                    symbol=f"{ticker}.US",  # format: {stock_code}.{exchange}
                    period="d",             # daily data
                    from_date=start_date,   # start date
                    to_date=end_date,       # end date
                    order="a"               # ascending order by date
                )
                if len(data) == 0:
                    return None
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                return df\
                    .rename(columns={'adjusted_close': 'adj_close'})\
                    .assign(
                        adj_rate = lambda x: x['adj_close'] / x['close'],
                        **{
                            f'adj_{column}': lambda x: x['adj_rate'] * x[column]
                            for column in ['open', 'high', 'low']
                        })\
                    .sort_index(axis='columns')
            except Exception:
                print(f'Error fetching {ticker}, retrying in 10 seconds...')
                time.sleep(10)
        raise ValueError(f'Failed to fetch data for {ticker}')

    def get_eodhd_data(self, tickers: list, start_date: str = '1990-01-01', end_date: str = f'{pd.Timestamp.today().date()}', n_jobs: int = -1, universe:Literal['etf', 'stock']='stock') -> dict:
        all_data_dict = dict(zip(tickers, Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self.get_eodhd_data_single_stock)(ticker, start_date, end_date) for ticker in tqdm.tqdm(tickers, desc=f'Fetching {universe} data from EODHD')
        )))
        if all_data_dict == {}:
            return pd.DataFrame()
        
        valid_data_dict = {k: v for k, v in all_data_dict.items() if v is not None}
        if not valid_data_dict:
            return pd.DataFrame()
            
        all_data_df = pd.concat(valid_data_dict, axis=1).swaplevel(axis='columns').sort_index(axis='columns')
        all_data_df.columns.names = ['table_name','ticker']
        return all_data_df

    def update_databank(self, universe:Literal['etf', 'stock']='stock', n_jobs: int = -1):
        stock_list = self.stock_list if universe == 'stock' else self.etf_list
        self.data_df = pd.DataFrame()
        for _ in range(5):
            print(f'stock_list:{len(stock_list)}')
            self.data_df = pd.concat([self.data_df,self.get_eodhd_data(stock_list,n_jobs=n_jobs,universe=universe)],axis=1)
            stock_list = list(set(stock_list) - set(self.data_df['close'].columns))
            if len(stock_list)==0:
                break
            elif _ < 4:
                print(f'Detected missing data for the following:{stock_list}')
                print(f'Still need to fetch:{len(stock_list)} tickers')
            else:
                # raise ValueError('Failed to fetch complete data, please check the data status!!!')
                print(f'Failed to fetch {stock_list}, please check the data status!!!')

        if 'SPY' in self.data_df['close'].columns:
            index = self.data_df['close']['SPY'].dropna().index
        elif 'AAPL' in self.data_df['close'].columns:
            index = self.data_df['close']['AAPL'].dropna().index
        else:
            index = self.get_eodhd_data(['SPY'],n_jobs=1)['close']['SPY'].dropna().index
        self.data_df = self.data_df.loc[index]
        print(f'compliance stocks:{len(set(self.data_df.columns.get_level_values(1)))}')

        DB = USDatabank(config.us_data_config["databank_path"], universe)
        for _ in tqdm.tqdm(sorted(set(self.data_df.columns.get_level_values(0))), desc='Saving data to databank'):
            DB[_] = self.data_df[_]


class USDatabank(dict):
    def __init__(self, path:str, universe:Literal['etf', 'stock']='stock'):
        self.path = os.path.join(path, universe)
        os.makedirs(self.path,exist_ok=True)
        self.cashe_dict = {'bool':bool}
        self.func_dict = {}
        self.reindex_like = None
    def __getitem__(self, key):
        if key in self.cashe_dict:
            return self.cashe_dict[key]
        elif key in self.func_dict:
            return self.func_dict[key]
        else:
            file_path = os.path.join(self.path, f'{key}.parquet')
            if os.path.exists(file_path):
                try:
                    if self.reindex_like is not None:
                        return pd.read_parquet(file_path).reindex_like(self.reindex_like)
                    else:
                        return pd.read_parquet(file_path)
                except :
                    raise ValueError(f'{file_path} is corrupted or cannot be read')
            raise ValueError(f'{file_path} does not exist')
    def __call__(self, key):
        return self.__getitem__(key)
    def __setitem__(self, key, value):
        file_path = os.path.join(self.path, f'{key}.parquet')
        value[np.isfinite(value)].to_parquet(file_path)
    def cashe_list(self):
        parquet_set = set(filter(lambda X:X.endswith(f".parquet"),os.listdir(self.path)))
        return sorted(map(lambda X:X[:-len('.parquet')],list(parquet_set)))


class Handler(dict):
    def __init__(self,path,data_type:str = 'parquet'):
        self.path = path
        self.cashe_dict = {'bool':bool}
        self.func_dict = {}
        if data_type == 'pickle':
            data_type = 'pkl'
        self.data_type = data_type
        self.reindex_like = None
        os.makedirs(path,exist_ok=True)
    def __getitem__(self, key):
        if key in self.cashe_dict:
            return self.cashe_dict[key]
        elif key in self.func_dict:
            return self.func_dict[key]
        else:
            file_path = os.path.join(self.path, f'{key}.{self.data_type}')
            # 检查存储的文件是否存在
            if os.path.exists(file_path):
                try:
                    if self.data_type == 'parquet':
                        if self.reindex_like is not None:
                            return pd.read_parquet(file_path).reindex_like(self.reindex_like)
                        else:
                            return pd.read_parquet(file_path)
                    elif self.data_type == 'pkl':
                        if self.reindex_like is not None:
                            return pd.read_pickle(file_path).reindex_like(self.reindex_like)
                        else:
                            return pd.read_pickle(file_path)
                except :
                    # 如果文件损坏或无法读取，返回默认值
                    raise ValueError(f'文件{file_path}损坏或无法读取')
            raise ValueError(f'文件{file_path}不存在')
    def __call__(self, key):
        return self.__getitem__(key)
    def __setitem__(self, key, value):
        file_path = os.path.join(self.path, f'{key}.{self.data_type}')
        value[np.isfinite(value)].to_parquet(file_path)
    def cashe_list(self):
        parquet_set = set(filter(lambda X:X.endswith(f".{self.data_type}"),os.listdir(self.path)))
        return sorted(map(lambda X:X[:-(len(self.data_type)+1)],list(parquet_set)))

