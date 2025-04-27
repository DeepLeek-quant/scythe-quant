from typing import Literal
import pandas as pd
import numpy as np

def largest(df, n):
    return df.rank(axis=1, ascending=False) <= n

def smallest(df, n):
    return df.rank(axis=1, ascending=True) <= n

def to_factor(df, method:Literal['Z']='Z', remove_outlier:Literal['SD', 'IQR', 'WS', 'MAD', None]='MAD', universe:pd.DataFrame=None):
    """將資料去極值、標準化

    Args:
        df (pd.DataFrame): 要轉換的資料，index 為日期，columns 為股票代號
        method (Literal['Z']): 標準化方法，目前只支援 'Z' 方法
        remove_outlier (Literal['SD', 'IQR', 'WS', 'MAD', None]): 去極值方法，預設為 'MAD' 
        universe (pd.DataFrame, optional): 股票池，用於篩選要保留的股票

    Returns:
        pd.DataFrame: 去極值、標準化後的因子值，index 為日期，columns 為股票代號

    Examples:
        將 ROE 資料去極值、標準化:
        ```python
        roe_factor = df_roe.to_factor(method='Z', remove_outlier='MAD')
        ```
    """
    
    params = {'SD': 3, 'IQR': 1.5, 'WS': 0.05, 'MAD': 3}

    df = df[universe] if universe is not None else df

    # remove outlier
    if remove_outlier !=None:
        if remove_outlier=='SD':
            mean = df.mean(axis=1)
            sd = df.std(axis=1)
            sd_threshold = params['SD']
            lower_bound, upper_bound = mean-sd_threshold*sd, mean+sd_threshold*sd
            df = df.where((df.ge(lower_bound, axis=0) & df.le(upper_bound, axis=0)) , np.nan)
        elif remove_outlier=='IQR':
            q1, q3 = df.quantile(0.25, axis=1), df.quantile(0.75, axis=1)
            iqr = q3-q1
            iqr_threshold = params['IQR']
            lower_bound, upper_bound = q1-iqr_threshold*iqr, q3+iqr_threshold*iqr
            df = df.where((df.ge(lower_bound, axis=0) & df.le(upper_bound, axis=0)) , np.nan)
        elif remove_outlier=='WS':
            ws_limit = params['WS']
            lower_bound, upper_bound = df.quantile(ws_limit, axis=1), data.quantile((1-ws_limit), axis=1)
            df = df.clip(lower=lower_bound, upper=upper_bound, axis=0)
        elif remove_outlier=='MAD':
            median = df.median(axis=1)
            mad = df.sub(median, axis=0).abs().median(axis=1)
            mad_threshold = params['MAD']
            lower_bound, upper_bound = median - mad_threshold * 1.4826 * mad, median + mad_threshold * 1.4826 * mad
            df = df.where((df.ge(lower_bound, axis=0)) & (df.le(upper_bound, axis=0)), np.nan)
    
    # z-score standardize
    if method=='Z':
        df = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

    return df

def to_rank(df, asc:bool=True, universe:pd.DataFrame=None):
    """將資料轉換為排序值

    Args:
        df (pd.DataFrame): 要轉換的資料，index 為日期，columns 為股票代號
        asc (bool): 是否為正向因子，True 代表數值越大分數越高，False 代表數值越小分數越高
        universe (pd.DataFrame, optional): 股票池，用於篩選要保留的股票

    Returns:
        pd.DataFrame: 轉換後的因子值，數值介於 0~1 之間，index 為日期，columns 為股票代號

    Examples:
        將 ROE 資料轉換為排序值:
        ```python
        roe_rank = df_roe.to_rank(asc=True)
        ```

        指定股票池:
        ```python
        universe = df_profit >= 0
        roe_rank = df_roe.to_rank(asc=True, universe=universe)
        ```

    Note:
        - 輸出的因子值為橫截面百分比排名
        - 若有指定 universe，只會保留 universe 內的股票資料
        - 可以作為 DataFrame 的方法使用: df.to_rank()
    """
    if universe is not None:
        df = df[universe]
    return df.rank(axis=1, pct=True, ascending=asc)

pd.DataFrame.largest = lambda self, n: largest(self, n)
pd.DataFrame.smallest = lambda self, n: smallest(self, n)
pd.DataFrame.to_factor = lambda self, method='Z', remove_outlier='MAD', universe=None: to_factor(self, method, remove_outlier, universe)
pd.DataFrame.to_rank = lambda self, asc=True, universe=None: to_rank(self, asc, universe)