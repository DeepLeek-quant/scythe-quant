from typing import Literal, Union
import pandas as pd
import numpy as np

# factor
def largest(df, n):
    return df.rank(axis=1, ascending=False) <= n

def smallest(df, n):
    return df.rank(axis=1, ascending=True) <= n

def to_zscore(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

def winsorize(df, method:Literal['SD', 'IQR', 'WS', 'MAD']='MAD', params=None):
    params = params or {'SD': 3, 'IQR': 2.5, 'WS': 0.05, 'MAD': 3}

    # remove outlier
    if method=='SD':
        mean = df.mean(axis=1)
        sd = df.std(axis=1)
        sd_threshold = params['SD']
        lower_bound, upper_bound = mean-sd_threshold*sd, mean+sd_threshold*sd
        return df.where((df.ge(lower_bound, axis=0) & df.le(upper_bound, axis=0)) , np.nan)
    elif method=='IQR':
        q1, q3 = df.quantile(0.25, axis=1), df.quantile(0.75, axis=1)
        iqr = q3-q1
        iqr_threshold = params['IQR']
        lower_bound, upper_bound = q1-iqr_threshold*iqr, q3+iqr_threshold*iqr
        return df.where((df.ge(lower_bound, axis=0) & df.le(upper_bound, axis=0)) , np.nan)
    elif method=='WS':
        ws_limit = params['WS']
        lower_bound, upper_bound = df.quantile(ws_limit, axis=1), data.quantile((1-ws_limit), axis=1)
        return df.clip(lower=lower_bound, upper=upper_bound, axis=0)
    elif method=='MAD':
        median = df.median(axis=1)
        mad = df.sub(median, axis=0).abs().median(axis=1)
        mad_threshold = params['MAD']
        lower_bound, upper_bound = median - mad_threshold * 1.4826 * mad, median + mad_threshold * 1.4826 * mad
        return df.where((df.ge(lower_bound, axis=0)) & (df.le(upper_bound, axis=0)), np.nan)

def to_factor(df, winsorize:Literal['SD', 'IQR', 'WS', 'MAD', None]='MAD', params=None):
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

    # remove outlier
    if winsorize !=None:
        df =  df.winsorize(winsorize, params)
    
    # z-score standardize
    return df.to_zscore()

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

def _df_log(self, base=10):
    arr = self.values
    with np.errstate(invalid='ignore', divide='ignore'):
        arr_safe = np.where(arr > 0, arr, 0)
        if base == 10:
            result = np.log10(arr_safe)
        elif base == np.e:
            result = np.log(arr_safe)
        else:
            result = np.log(arr_safe) / np.log(base)
        result = np.where(np.isneginf(result), 0, result)
    return pd.DataFrame(result, index=self.index, columns=self.columns)

# plot
def plot_cr(df:Union[pd.DataFrame, pd.Series], **kwargs):
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    return df.add(1).cumprod().sub(1).iplot(**kwargs)

pd.DataFrame.largest = lambda self, n: largest(self, n)
pd.DataFrame.smallest = lambda self, n: smallest(self, n)
pd.DataFrame.to_zscore = lambda self: to_zscore(self)
pd.DataFrame.winsorize = lambda self, method='MAD', params=None: winsorize(self, method, params)
pd.DataFrame.to_factor = lambda self, winsorize='MAD', params=None: to_factor(self, winsorize, params)
pd.DataFrame.to_rank = lambda self, asc=True, universe=None: to_rank(self, asc, universe)
pd.DataFrame.log = _df_log
pd.DataFrame.plot_cr = pd.Series.plot_cr = plot_cr

# dfs
def sum_dfs(x_list:list[pd.DataFrame]) -> pd.DataFrame:
    if len(x_list)==2:
        return x_list[0] + x_list[1]
    else:
        return pd.concat(x_list, keys=range(len(x_list))).groupby(level=1).sum()
    
def mean_dfs(x_list:list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(x_list, keys=range(len(x_list))).groupby(level=1).mean()

def std_dfs(x_list:list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(x_list, keys=range(len(x_list))).groupby(level=1).std()

def var_dfs(x_list:list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(x_list, keys=range(len(x_list))).groupby(level=1).var()

def cov_dfs(x_list:list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(x_list, keys=range(len(x_list))).groupby(level=1).cov()

def calc_surprise(item: str, lag:int=12, std_window:int=24, method: Literal['std', 'drift']='drift'):
    try:
        db
    except:
        from .data import Databank
        db = Databank()
    diffs = [db[f'{item}_lag{i}' if i != 0 else item].fillna(0)-db[f'{item}_lag{i+lag}'].fillna(0) for i in range(0, std_window)]
    seasonal_diff = db[item]-db[f'{item}_lag{lag}']
    
    if method == 'std':
        surprise = seasonal_diff / std_dfs(diffs)
    elif method == 'drift':
        surprise = (seasonal_diff - mean_dfs(diffs)) / std_dfs(diffs)
        
    return surprise\
        .replace([np.inf, -np.inf], np.nan)