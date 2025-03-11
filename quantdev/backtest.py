"""回測

Available Functions:
    - get_data: 從資料庫取得資料並進行預處理
    - get_factor: 將資料轉換為因子值
    - backtesting: 進行回測並返回完整的回測結果
    - simple_backtesting: 進行快速回測並返回每日報酬率
    - multi_backtesting: 進行多重策略回測並返回組合策略回測結果
    - simple_multi_backtesting: 進行快速多重策略回測並返回每日報酬率

Examples:
    取得資料並進行回測:
    ```python
    # 取得資料
    roe = get_data('roe')
    pe = get_data('pe')
    
    # 計算因子值
    roe_factor = get_factor(roe, asc=True)
    pe_factor = get_factor(pe, asc=False)
    
    # 合成因子
    composite_factor = get_factor((0.3 * roe_factor + 0.7 * pe_factor), asc=True)
    
    # 進行回測
    result = backtesting(
        data=composite_factor>0.9,
        rebalance='Q',
        signal_shift=1,
        benchmark='0050'
    )
    
    # 查看回測結果
    result.summary
    result.position_info
    result.report
    ```

    進行多重策略回測:
    ```python
    # 建立兩個策略
    strategy1 = backtesting(data1, rebalance='Q')
    strategy2 = backtesting(data2, rebalance='M')

    # 組合策略回測(等權重)
    meta = multi_backtesting({
        'strategy1': strategy1,
        'strategy2': strategy2
    })

    # 組合策略回測(自訂權重)
    meta = multi_backtesting({
        'strategy1': (strategy1, 0.7),
        'strategy2': (strategy2, 0.3)
    })
    ```
"""

from typing import Literal, Union, Tuple
from IPython.display import display, HTML
from abc import ABC
from scipy import stats
import datetime as dt
import pandas as pd
import numpy as np
import calendar
import logging

from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from numerize.numerize import numerize
import plotly.figure_factory as ff
import plotly.graph_objects as go
import panel as pn

from .data import Databank
from .analysis import *

pd.set_option('future.no_silent_downcasting', True)

# databank
_db = Databank()
_t_date = _db.read_dataset('mkt_calendar', columns=['date'], filters=[('休市原因中文說明(人工建置)','=','')]).rename(columns={'date':'t_date'})

def get_data(item:Union[str, pd.DataFrame, Tuple[str, str]], universe:pd.DataFrame=None)-> pd.DataFrame:
    """取得股票資料並轉換為時間序列格式

    Args:
        item (Union[str, pd.DataFrame, Tuple[str, str]]): 資料欄位名稱或已有的 DataFrame，或是 (欄位名稱, 資料集名稱) 的 tuple
        universe (pd.DataFrame, optional): 股票池，用於篩選要保留的股票

    Returns:
        pd.DataFrame: 轉換後的時間序列資料，index 為日期，columns 為股票代號

    Examples:
        從資料庫取得收盤價資料:
        ```python
        close = get_data('收盤價')
        ```

        使用已有的 DataFrame:
        ```python
        df = _db.read_dataset('stock_trading_data', filter_date='t_date', start='2023-01-01', end='2023-01-02', columns=['t_date', 'stock_id', '收盤價'])
        close = get_data(df)
        ```

        指定資料集:
        ```python
        close = get_data(('收盤價', 'stock_trading_data'))
        ```

        指定股票池:
        ```python
        universe = bts.get_data('稅後淨利') >= 0
        close = get_data('收盤價', universe=universe)
        ```

    Note:
        - 若輸入字串，會從資料庫尋找對應欄位資料
        - 若輸入 DataFrame，需包含 t_date、stock_id 及目標欄位
        - 若輸入 tuple，第一個元素為欄位名稱，第二個元素為資料集名稱
        - 輸出資料會自動向前填補最多 240 個交易日的缺失值
        - 會移除全部為空值的股票
        - 若有指定 universe，只會保留 universe 內的股票資料
    """

    if isinstance(item, str):
        raw_data = _db.read_dataset(
            dataset=_db.find_dataset(item), 
            filter_date='t_date', 
            columns=['t_date', 'stock_id', item],
            )
    elif isinstance(item, tuple):
        raw_data = _db.read_dataset(
            dataset=item[1],
            filter_date='t_date',
            columns=['t_date', 'stock_id', item[0]],
            )
    elif isinstance(item, pd.DataFrame):
        raw_data = item

    data = pd.merge_asof(
        _t_date[_t_date['t_date']<=pd.Timestamp.today() + pd.DateOffset(days=5)], 
        raw_data\
            .query('stock_id.str.match("^[0-9]")')\
            .sort_values(by='t_date')\
            .drop_duplicates(subset=['t_date', 'stock_id'], keep='last')\
            .set_index(['t_date', 'stock_id'])\
            .unstack('stock_id')\
            .droplevel(0, axis=1), 
        on='t_date', 
        direction='backward'
        )\
        .ffill(limit=240)\
        .set_index(['t_date'])\
        .dropna(axis=1, how='all')
    return data[universe] if universe is not None else data

def get_factor(item:Union[str, pd.DataFrame, Tuple[str, str]], asc:bool=True, universe:pd.DataFrame=None)-> pd.DataFrame:
    """將資料轉換為因子值

    Args:
        item (Union[str, pd.DataFrame, Tuple[str, str]]): 資料欄位名稱或已有的 DataFrame，或是 (欄位名稱, 資料集名稱) 的 tuple
        asc (bool): 是否為正向因子，True 代表數值越大分數越高，False 代表數值越小分數越高
        universe (pd.DataFrame, optional): 股票池，用於篩選要保留的股票

    Returns:
        pd.DataFrame: 轉換後的因子值，數值介於 0~1 之間，index 為日期，columns 為股票代號

    Examples:
        從資料庫取得 ROE 因子值:
        ```python
        roe_factor = get_factor('roe', asc=True)
        ```

        使用已有的 DataFrame:
        ```python
        df = get_data('roe')
        roe_factor = get_factor(df, asc=True)
        ```

        指定資料集:
        ```python
        roe_factor = get_factor(('roe', 'fin_data'), asc=True)
        ```

        指定股票池:
        ```python
        universe = bts.get_data('稅後淨利') >= 0
        roe_factor = get_factor('roe', asc=True, universe=universe)
        ```

    Note:
        - 若輸入字串，會先呼叫 get_data 取得資料
        - 若輸入 DataFrame，需包含 t_date、stock_id 及目標欄位
        - 若輸入 tuple，第一個元素為欄位名稱，第二個元素為資料集名稱
        - 輸出的因子值為橫截面標準化後的百分比排名
        - 若有指定 universe，只會保留 universe 內的股票資料
    """
    if isinstance(item, (str, tuple)):
        return get_factor(get_data(item), asc, universe)
    elif isinstance(item, pd.DataFrame):
        data = item[universe] if universe is not None else item
        return data.rank(axis=1, pct=True, ascending=asc)

def _calc_release_pct_rebalancing(type_:Literal['MR', 'QR'], pct:int=80):
    """計算財報發布達到特定比例的交易日期列表

    Args:
        type_ (Literal['MR', 'QR']): 財報類型
            - MR: 月營收報表
            - QR: 季財報
        pct (int, optional): 發布比例門檻. Defaults to 80.
            例如: pct=80 表示要等到80%的公司都發布財報後的第一個交易日

    Returns:
        list: 符合條件(達到指定發布比例後的第一個交易日)的交易日期列表
        
    Examples:
        ```python
        # 取得80%公司發布月營收後的交易日:
        dates = _calc_release_pct_rebalancing('MR', pct=80)

        # 取得90%公司發布季報後的交易日:
        dates = _calc_release_pct_rebalancing('QR', pct=90)
        ```

    Note:
        - MR (月營收): 檢查1-30天的發布情況
        - QR (季報): 檢查1-90天的發布情況
    """

    map = {
        'MR': {'dataset': 'monthly_rev','days': range(1, 31)},
        'QR': {'dataset': 'fin_data','days': range(1, 91)}
    }
    
    data = _db.read_dataset(map[type_]['dataset'], columns=['date', 'stock_id', 'release_date'])
    # Create a MultiIndex for all possible combinations of dates and days
    index = pd.MultiIndex.from_product([
            data['date'].unique(),
            map[type_]['days'],
    ], names=['date', 'day'])

    # Create DataFrame with calculated release dates
    # For each date, calculate potential release dates by adding days
    result = (pd.DataFrame(index=index)
        .reset_index()\
        .assign(release_date=lambda df: df['date'] + pd.DateOffset(months=1) - pd.DateOffset(days=1) + pd.to_timedelta(df['day'], unit='D')))
    
    # Calculate cumulative count of companies that have released reports
    cumulative_counts = data\
        .groupby(['date', 'release_date'])\
        .size()\
        .groupby('date')\
        .cumsum()\
        .reset_index()\
        .rename(columns={0: 'release_count'})
    total_count_last_mth = data.groupby('date')['stock_id'].nunique().shift(1).reset_index().rename(columns={'stock_id': 'total_count'})

    # Combine all data and calculate release percentages
    result = result\
        .merge(cumulative_counts, how='left', on=['date', 'release_date'])\
        .merge(total_count_last_mth, how='left', on=['date'])\
        .assign(
            release_count=lambda df: df.groupby('date')['release_count'].ffill(),
            release_pct = lambda df: df['release_count'] / df['total_count']
        )[['date', 'release_date', 'release_pct']]
    result = _db._add_trade_date(result)[['date', 't_date', 'release_pct']]
    
    # Filter and transform the result to get rebalancing dates
    return result\
        [result['release_pct'] >= pct/100]\
        .groupby('date')\
        .first()\
        .reset_index()\
        ['t_date']\
        .tolist()

def _get_rebalance_date(rebalance:Literal['D', 'MR', 'QR', 'W', 'M', 'Q', 'Y'], end_date:Union[pd.Timestamp, str]=None):
    """取得再平衡日期列表

    Args:
        rebalance (Literal['MR', 'QR', 'W', 'M', 'Q', 'Y']): 再平衡頻率

            - MR: 每月10日後第一個交易日
            - QR: 每季財報公布日後第一個交易日(3/31, 5/15, 8/14, 11/14)
            - W: 每週一
            - M: 每月第一個交易日
            - Q: 每季第一個交易日
            - Y: 每年第一個交易日
        end_date (Union[pd.Timestamp, str], optional): 結束日期，若為 None 則為今日加5天

    Returns:
        list: 再平衡日期列表

    Examples:
        取得每月再平衡日期:
        ```python
        dates = _get_rebalance_date('M', end_date='2023-12-31')
        ```

        取得每季財報公布後再平衡日期:
        ```python
        dates = _get_rebalance_date('QR')
        ```

    Note:
        - 開始日期為資料庫中最早的交易日
        - 結束日期若為 None 則為今日加5天
        - 所有再平衡日期都會對應到實際交易日
    """
    
    # dates
    start_date = _t_date['t_date'].min()
    end_date = pd.Timestamp.today() + pd.DateOffset(days=5) if end_date is None else pd.to_datetime(end_date)
    

    if rebalance == 'D':
        return _t_date[(_t_date['t_date'] <= end_date)]['t_date'].to_list()
    elif rebalance.startswith('W'):
        if len(rebalance.split('-')) == 1:
            r_date = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='W-MON'), columns=['r_date'])
        else:
            r_date = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq=rebalance), columns=['r_date'])
    elif rebalance.startswith('MR'):
        if len(rebalance.split('-')) == 1:
            date_list = [
                pd.to_datetime(f'{year}-{month:02d}-10') + pd.DateOffset(days=1)
                for year in range(start_date.year, end_date.year + 1)
                for month in range(1, 13)
                if start_date <= pd.to_datetime(f'{year}-{month:02d}-10') + pd.DateOffset(days=1) <= end_date
            ]
        elif (len(rebalance.split('-')) == 2) and (rebalance.split('-')[1].isdigit()):
            date_list = _calc_release_pct_rebalancing(rebalance.split('-')[0], int(rebalance.split('-')[1]))
        r_date = pd.DataFrame(date_list, columns=['r_date'])
    elif rebalance.startswith('QR'):
        if len(rebalance.split('-')) == 1:
            qr_dates = ['03-31', '05-15', '08-14', '11-14']
            date_list = [
                pd.to_datetime(f'{year}-{md}') + pd.DateOffset(days=1)
                for year in range(start_date.year, end_date.year + 1)
                for md in qr_dates
                if start_date <= pd.to_datetime(f"{year}-{md}") + pd.DateOffset(days=1) <= end_date
            ]
        elif (len(rebalance.split('-')) == 2) and (rebalance.split('-')[1].isdigit()):
            date_list = _calc_release_pct_rebalancing(rebalance.split('-')[0], int(rebalance.split('-')[1]))
        r_date = pd.DataFrame(date_list, columns=['r_date'])
    elif rebalance == 'M':
        r_date = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='MS'), columns=['r_date'])
    elif rebalance == 'Q': 
        r_date = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='QS'), columns=['r_date'])    
    elif rebalance == 'Y':
        r_date = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='YS'), columns=['r_date'])
    else:
        raise ValueError("Invalid frequency. Allowed values are 'QR', 'W', 'M', 'Q', 'Y'.")

    return pd.merge_asof(
        r_date, _t_date, 
        left_on='r_date', 
        right_on='t_date', 
        direction='forward'
        )['t_date'].to_list()

def _get_portfolio(data:pd.DataFrame, return_df:pd.DataFrame, rebalance:Literal['MR', 'QR', 'W', 'M', 'Q', 'Y']='QR', signal_shift:int=0, hold_period:int=None):
    """
    將資料轉換為投資組合權重。

    Args:
        data (pd.DataFrame): 選股條件矩陣，index為日期，columns為股票代碼
        return_df (pd.DataFrame): 股票報酬率矩陣，index為日期，columns為股票代碼
        rebalance (str): 再平衡頻率，可選 'MR'(月營收公布後), 'QR'(財報公布後), 'W'(每週), 'M'(每月), 'Q'(每季), 'Y'(每年)
        signal_shift (int): 訊號延遲天數，用於模擬實際交易延遲
        hold_period (int): 持有期間，若為None則持有至下次再平衡日

    Returns:
        tuple: 包含以下兩個元素:
            - buy_list (pd.DataFrame): 選股結果矩陣，index為再平衡日期，columns為股票代碼
            - portfolio (pd.DataFrame): 投資組合權重矩陣，index為日期，columns為股票代碼

    Examples:
        取得每月第一個交易日再平衡、訊號延遲1天、持有20天的投資組合:
        ```python
        buy_list, portfolio = _get_portfolio(data, return_df, rebalance='M', signal_shift=1, hold_period=20)
        ```

    Note:
        - 投資組合權重為等權重配置
        - 若無持有期間限制，則持有至下次再平衡日
        - 訊號延遲用於模擬實際交易所需的作業時間
    """
    
    # rebalance
    r_date = _get_rebalance_date(rebalance)
    buy_list =  data[data.index.isin(r_date)]
    
    # weight
    portfolio = buy_list\
        .fillna(False)\
        .astype(bool)\
        .astype(int)\
        .apply(lambda x: x / x.sum(), axis=1)

    # shift & hold_period
    portfolio = pd.DataFrame(return_df.index)\
        .merge(portfolio, on='t_date', how='left')\
        .set_index('t_date')\
        .shift(signal_shift)\
        .ffill(limit=hold_period)\
        .dropna(how='all')
    return buy_list, portfolio

def _stop_loss_or_profit(buy_list:pd.DataFrame, portfolio_df:pd.DataFrame, return_df:pd.DataFrame, pct:float, stop_at:Literal['intraday', 'next_day']='intraday'):
    # calculate portfolio returns by multiplying portfolio positions (0/1) with return data
    portfolio_return = (portfolio_df != 0).astype(int) * return_df[portfolio_df.columns].loc[portfolio_df.index]
    
    # initialize empty DataFrame to store portfolio after stop loss/profit adjustments
    aftermath_portfolio_df = pd.DataFrame(index=portfolio_return.index, columns=portfolio_return.columns)
    
    # get rebalancing period dates
    period_starts = portfolio_df.index.intersection(buy_list.index)
    period_ends = period_starts[1:].map(lambda x: portfolio_df.loc[:x].index[-1])
    period_ends = period_ends.append(pd.Index([portfolio_return.index[-1]]))

    # convert DataFrames to numpy arrays for faster computation
    returns_array = portfolio_return.values
    dates_array = portfolio_return.index.values

    # process each rebalancing period
    for start_date, end_date in zip(period_starts, period_ends):
        # get array indices for current period
        start_idx = portfolio_return.index.get_indexer([start_date])[0]
        end_idx = portfolio_return.index.get_indexer([end_date])[0] + 1
        
        # extract returns and dates for current period
        period_returns = returns_array[start_idx:end_idx]
        period_dates = dates_array[start_idx:end_idx]
        
        # calculate cumulative returns for period
        if isinstance(pct, float):
            cum_returns = np.cumprod(1 + period_returns, axis=0) - 1
            
            # create mask for stop loss/profit triggers based on threshold
            if pct<0:
                stop_loss_mask = cum_returns <= pct  # stop loss
            else:
                stop_loss_mask = cum_returns >= pct  # stop profit
        elif isinstance(pct, pd.DataFrame):
            pass
        
        # skip if period has no data
        if len(stop_loss_mask) == 0:
            continue
        
        # find first trigger point for each stock
        trigger_idx = np.argmax(stop_loss_mask, axis=0)
        has_trigger = np.any(stop_loss_mask, axis=0)
        
        # for 'next_day' mode, shift trigger point one day forward
        if stop_at=='next_day':
            trigger_idx = np.where(has_trigger & (trigger_idx < len(period_dates) - 1), trigger_idx + 1, trigger_idx)
        
        # create mask for dates after trigger points
        date_matrix = period_dates[:, None] > period_dates[trigger_idx]
        
        # only apply mask to stocks that hit stop loss/profit
        final_mask = np.where(has_trigger, date_matrix, False)
        
        # get portfolio values and zero out positions after stop loss/profit
        period_portfolio = portfolio_df.values[start_idx:end_idx]
        filtered_portfolio = np.where(final_mask, 0, period_portfolio)
        
        # store results
        aftermath_portfolio_df.iloc[start_idx:end_idx] = filtered_portfolio

    return aftermath_portfolio_df

def backtesting(
    data:pd.DataFrame, 
    rebalance:Literal['MR', 'QR', 'W', 'M', 'Q', 'Y']='QR', 
    signal_shift:int=0, hold_period:int=None, 
    stop_loss:Union[float, pd.DataFrame]=None, stop_profit:Union[float, pd.DataFrame]=None, stop_at:Literal['intraday', 'next_day']='next_day',
    start:Union[int, str]=None, end:Union[int, str]=None, 
    benchmark:Union[str, list[str]]='0050')-> 'Strategy':
    """
    進行回測並返回回測結果。

    Args:
        data (pd.DataFrame): 選股條件矩陣，index為日期，columns為股票代碼
        rebalance (str): 再平衡頻率，可選 'MR'(月營收公布後), 'QR'(財報公布後), 'W'(每週), 'M'(每月), 'Q'(每季), 'Y'(每年)
        signal_shift (int): 訊號延遲天數，用於模擬實際交易延遲
        hold_period (int): 持有期間，若為None則持有至下次再平衡日
        stop_loss (float | pd.DataFrame): 停損點，例如-0.1代表跌幅超過10%時停損
        stop_profit (float | pd.DataFrame): 停利點，例如0.2代表漲幅超過20%時停利
        stop_at (str): 停損停利執行時點，可選 'intraday'(當日) 或 'next_day'(次日)
        start (int | str): 回測起始日期
        end (int | str): 回測結束日期
        benchmark (str | list[str]): 基準指標，可為單一或多個股票代碼

    Returns:
        Strategy: 回測結果物件，包含:

            - summary: 策略績效摘要，包含總報酬率、年化報酬率、最大回撤等重要指標
            - position_info: 持股資訊，包含每日持股數量、換手率、個股權重等資訊
            - report: 完整的回測報告，包含:
                - Equity curve, Relative return, Portfolio style, MAE/MFE... etc.

    Examples:
        ```python
        result = backtesting(
            data, 
            rebalance='Q', # 每季再平衡
            signal_shift=1, # 訊號延遲1天
            hold_period=20, # 持有20天
            stop_loss=-0.1, # 停損10%
            stop_profit=0.2, # 停利20%
            stop_at='next_day', # 停損/停利於觸及條件後次日執行
            start='2023-01-01', # 回測起始日期
            end='2023-12-31', # 回測結束日期
            benchmark='0050' # 基準指標
        )
        ```

    Note:
        - 投資組合權重為等權重配置
        - 若無持有期間限制，則持有至下次再平衡日
        - 停損停利點若未設定則不啟用
    """

    # return & weight
    return_df = _db.read_dataset('stock_return', filter_date='t_date', start=start, end=end)
    # if trade_at == 'open':
    #     return_df = return_df.shift(1)
    
    # get data
    buy_list, portfolio_df = _get_portfolio(data, return_df, rebalance, signal_shift, hold_period)
    
    # stop loss or profit
    if stop_loss is not None:
        portfolio_df = _stop_loss_or_profit(buy_list, portfolio_df, return_df, pct=abs(stop_loss)*-1, stop_at=stop_at).infer_objects(copy=False).replace({np.nan: 0})
    if stop_profit is not None:
        portfolio_df = _stop_loss_or_profit(buy_list, portfolio_df, return_df, pct=stop_profit, stop_at=stop_at).infer_objects(copy=False).replace({np.nan: 0})

    # backtest
    backtest_df = (return_df * portfolio_df)\
        .dropna(axis=0, how='all')

    return Strategy(
        return_df,
        buy_list,
        portfolio_df,
        backtest_df,
        benchmark,
        rebalance
    )

def simple_backtesting(
    data:pd.DataFrame, 
    rebalance:Literal['MR', 'QR', 'W', 'M', 'Q', 'Y']='QR', 
    signal_shift:int=0, 
    hold_period:int=None, 
    return_df:pd.DataFrame=None,
    )-> pd.DataFrame:
    """快速回測函數，僅返回投資組合每日報酬率序列。

    Attributes:
        data: 策略選股條件，True代表買入訊號
        rebalance: 再平衡頻率，可選MR(月營收公布日)、QR(季報公布日)、W(週)、M(月)、Q(季)、Y(年)
        signal_shift: 訊號延遲天數，用於模擬實際交易延遲
        hold_period: 持有期間天數，若為None則持有至下次再平衡日

    Returns:
        pd.DataFrame: 投資組合每日報酬率序列

    Examples:
        ```python
        daily_return = simple_backtesting(
            data,
            rebalance='QR', # 季報公布日再平衡
            signal_shift=1,  # 訊號延遲1天
            hold_period=20   # 持有20天
        )
        ```

    Note:
        - 投資組合權重為等權重配置
        - 若無持有期間限制，則持有至下次再平衡日
    """
    if return_df is None:
        return_df = _db.read_dataset('stock_return', filter_date='t_date')
    _, portfolio_df = _get_portfolio(data, return_df, rebalance, signal_shift, hold_period)
    return (return_df * portfolio_df)\
        .dropna(axis=0, how='all')\
        .sum(axis=1)

def multi_backtesting(
    strategies:dict[str, Union['Strategy', Tuple['Strategy', float]]], 
    rebalance:Literal['MR', 'QR', 'W', 'M', 'Q', 'Y']=None,
    benchmark:Union[str, list[str]]='0050')-> 'MultiStrategy':
    
    """多重策略回測並返回組合策略回測結果。

    Args:
        
        - strategies (dict[str, Union['Strategy', Tuple['Strategy', float]]]): 策略字典，格式為:  
            - key: 策略名稱
            - value: Strategy物件，或是(Strategy物件, 權重)的tuple
        - rebalance (Literal['MR', 'QR', 'W', 'M', 'Q', 'Y'], optional): 再平衡頻率，可選:  
            - MR: 每月10日後第一個交易日
            - QR: 每季財報公布日後第一個交易日(3/31, 5/15, 8/14, 11/14)
            - W: 每週一
            - M: 每月第一個交易日
            - Q: 每季第一個交易日
            - Y: 每年第一個交易日
        - benchmark (Union[str, list[str]], optional): 基準指標，可為單一或多個股票代碼

    Returns:
        MultiStrategy: 組合策略回測結果物件，包含:

            - summary: 策略績效摘要，包含總報酬率、年化報酬率、最大回撤等重要指標
            - position_info: 持股資訊，包含每日持股數量、換手率、個股權重等資訊
            - report: 完整的回測報告，包含:
                - Equity curve, Efficiency frontier, Portfolio style, MAE/MFE... etc.

    Examples:
        ```python
        # 建立兩個策略
        strategy1 = backtesting(data1, rebalance='Q')
        strategy2 = backtesting(data2, rebalance='M')

        # 組合策略回測(等權重)
        meta = multi_backtesting({
            'strategy1': strategy1,
            'strategy2': strategy2
        })

        # 組合策略回測(自訂權重)
        meta = multi_backtesting({
            'strategy1': (strategy1, 0.7),
            'strategy2': (strategy2, 0.3)
        })

        # 組合策略回測(每月再平衡)
        meta = multi_backtesting({
            'strategy1': (strategy1, 0.7),
            'strategy2': (strategy2, 0.3)
        }, rebalance='M')
        ```

    Note:
        - 若未指定權重，則預設為等權重配置
        - 若未指定再平衡頻率，則不進行再平衡
        - 若指定再平衡頻率，則會在再平衡日重新配置權重
        - 回測區間為所有策略重疊的時間區間
    """

    if all(isinstance(v, Strategy) for v in strategies.values()):
        strategies = {k: (v, 1/len(strategies)) for k, v in strategies.items()}
    
    strategies_rtn = pd.concat([v[0].daily_return.rename(k) for k, v in strategies.items()], axis=1).dropna()
    rebalance_weights = pd.DataFrame({k:v[1]/sum(v[1] for v in strategies.values()) for k, v in strategies.items()}, index=[strategies_rtn.index[0]])

    if rebalance is not None:
        rebalance_dates = [d for d in _get_rebalance_date(rebalance) if d in strategies_rtn.index]
        period_starts = pd.Series(rebalance_dates)
        period_ends = pd.Series(rebalance_dates + [strategies_rtn.index[-1]]).shift(-1).dropna()
        equity = pd.DataFrame(columns=strategies_rtn.columns, index=strategies_rtn.index)

        if period_starts[0] != strategies_rtn.index[0]:
            period_starts = pd.Series([strategies_rtn.index[0]] + list(period_starts))
            period_ends = pd.Series([period_starts[1]] + list(period_ends))

        for start_date, end_date in zip(period_starts, period_ends):
            start_idx = strategies_rtn.index.get_indexer([start_date])[0]
            end_idx = strategies_rtn.index.get_indexer([end_date])[0]
            if start_idx == 0: # first day
                equity.iloc[start_idx:end_idx, :] = (1+strategies_rtn.iloc[start_idx:end_idx, :]).cumprod() * rebalance_weights.values
            elif end_idx == strategies_rtn.index.size-1: # last day
                prev_equity = equity.iloc[start_idx-1].sum()
                end_idx += 1
                equity.iloc[start_idx:end_idx, :] = (1+strategies_rtn.iloc[start_idx:end_idx, :]).cumprod() * rebalance_weights.values * prev_equity    
            else:
                prev_equity = equity.iloc[start_idx-1].sum()
                equity.iloc[start_idx:end_idx, :] = (1+strategies_rtn.iloc[start_idx:end_idx, :]).cumprod() * rebalance_weights.values * prev_equity
    else:
        equity = (1+strategies_rtn).cumprod()* rebalance_weights.values

    # daily return
    daily_return = (equity.sum(axis=1).pct_change().combine_first(
        pd.Series([equity.sum(axis=1).iloc[0]-1], index=[equity.sum(axis=1).index[0]])
    )).rename_axis('t_date').astype(float)
    
    # return df
    indices = [v[0].return_df.index for v in strategies.values()]
    columns = [v[0].return_df.columns for v in strategies.values()]

    common_index = indices[0]
    for idx in indices[1:]:
        common_index = common_index.intersection(idx)
    common_columns = columns[0]
    for cols in columns[1:]:
        common_columns = common_columns.intersection(cols)
    first_strategy = list(strategies.values())[0][0]
    return_df = first_strategy.return_df.loc[common_index, common_columns]
    
    return MultiStrategy(strategies, rebalance=rebalance, benchmark=benchmark, return_df=return_df, daily_return=daily_return)

def simple_multi_backtesting(strategies:dict[str, Union['Strategy', Tuple['Strategy', float]]], 
    rebalance:Literal['MR', 'QR', 'W', 'M', 'Q', 'Y']=None)-> 'MultiStrategy':
    
    if all(isinstance(v, Strategy) for v in strategies.values()):
        strategies = {k: (v, 1/len(strategies)) for k, v in strategies.items()}
    
    strategies_rtn = pd.concat([v[0].daily_return.rename(k) for k, v in strategies.items()], axis=1).dropna()
    rebalance_weights = pd.DataFrame({k:v[1]/sum(v[1] for v in strategies.values()) for k, v in strategies.items()}, index=[strategies_rtn.index[0]])

    if rebalance is not None:
        rebalance_dates = [d for d in _get_rebalance_date(rebalance) if d in strategies_rtn.index]
        period_starts = pd.Series(rebalance_dates)
        period_ends = pd.Series(rebalance_dates + [strategies_rtn.index[-1]]).shift(-1).dropna()
        equity = pd.DataFrame(columns=strategies_rtn.columns, index=strategies_rtn.index)

        if period_starts[0] != strategies_rtn.index[0]:
            period_starts = pd.Series([strategies_rtn.index[0]] + list(period_starts))
            period_ends = pd.Series([period_starts[1]] + list(period_ends))

        for start_date, end_date in zip(period_starts, period_ends):
            start_idx = strategies_rtn.index.get_indexer([start_date])[0]
            end_idx = strategies_rtn.index.get_indexer([end_date])[0]
            if start_idx == 0: # first day
                equity.iloc[start_idx:end_idx, :] = (1+strategies_rtn.iloc[start_idx:end_idx, :]).cumprod() * rebalance_weights.values
            elif end_idx == strategies_rtn.index.size-1: # last day
                prev_equity = equity.iloc[start_idx-1].sum()
                end_idx += 1
                equity.iloc[start_idx:end_idx, :] = (1+strategies_rtn.iloc[start_idx:end_idx, :]).cumprod() * rebalance_weights.values * prev_equity    
            else:
                prev_equity = equity.iloc[start_idx-1].sum()
                equity.iloc[start_idx:end_idx, :] = (1+strategies_rtn.iloc[start_idx:end_idx, :]).cumprod() * rebalance_weights.values * prev_equity
    else:
        equity = (1+strategies_rtn).cumprod()* rebalance_weights.values

    # daily return
    daily_return = (equity.sum(axis=1).pct_change().combine_first(
        pd.Series([equity.sum(axis=1).iloc[0]-1], index=[equity.sum(axis=1).index[0]])
    )).rename_axis('t_date').astype(float)
    
    return daily_return

class PlotMaster(ABC):
    """
    繪圖基礎類別，提供繪圖相關的共用功能。

    Attributes:
        fig_param (dict): 繪圖參數設定，包含:
            - size (dict): 圖表大小設定
            - margin (dict): 圖表邊界設定 
            - template (str): 圖表樣式模板
            - colors (dict): 顏色設定
            - font (dict): 字型設定
            - pos_colorscale (list): 正值色階設定
            - neg_colorscale (list): 負值色階設定

    Returns:
        PlotMaster: 繪圖基礎物件，提供:
            - _bold(): 文字加粗功能
            - show_colors(): 顯示顏色設定功能

    Examples:
        ```python
        # 建立繪圖物件
        plot = PlotMaster()
        
        # 顯示顏色設定
        plot.show_colors()
        ```

    Note:
        - 需繼承此類別以使用繪圖功能
        - 子類別需實作 __required_attrs__ 中定義的屬性
    """
    
    def __init__(self):
        self.fig_param = dict(
            size=dict(w=800, h=600),
            margin=dict(t=50, b=50, l=50, r=50),
            template='plotly_dark',
            colors={
                'Bright Green':'#00FF00', 'Bright Red':'#FF0000', 'Bright Yellow':'#FFCC00', 
                'Bright Cyan':'#00FFFF', 'Bright Orange':'#FF9900', 'Bright Magenta':'#FF00FF',
                'Light Grey':'#AAAAAA', 'Light Blue':'#636EFA', 'Light Red':'#EF553B',
                'Dark Blue':'#0000FF', 'Dark Grey':'#7F7F7F', 'White':'#FFFFFF', 
            },
            font=dict(family='Arial'),
            pos_colorscale = [[0,'#414553'],[0.5,'#6E0001'],[1,'#FF0001']],
            neg_colorscale = [[0,'#02EF02'],[0.5,'#017301'],[1,'#405351']],
        )

        __required_attrs__ = {
            'return_df': 'pd.DataFrame',
            'buy_list': 'pd.DataFrame',
            'portfolio_df': 'pd.DataFrame', 
            'backtest_df': 'pd.DataFrame',
            'benchmark': 'list',
            'rebalance': 'str',
            'daily_return': 'pd.Series or pd.DataFrame',
            'maemfe': 'pd.DataFrame',
            'betas': 'pd.DataFrame',
            
            # if meta strategy
            'strategies': 'dict',
        }

    def _bold(self, text):
        return f'<b>{text}</b>'

    def show_colors(self):
        fig = go.Figure()
        
        # Calculate positions
        colors = self.fig_param['colors']
        y_start = 0.8
        spacing = 0.1
        x_base = 0.15
        
        for i, (name, color) in enumerate(colors.items()):
            y_pos = y_start - i*spacing
            
            # Lines
            fig.add_trace(go.Scatter(
                x=[x_base, x_base+0.1], y=[y_pos]*2,
                mode='lines', line=dict(color=color, width=4),
                name=f"{name} ({color})", showlegend=True
            ))
            # Dots
            fig.add_trace(go.Scatter(
                x=[x_base+0.2], y=[y_pos],
                mode='markers', marker=dict(color=color, size=12),
                showlegend=False
            ))
            # Boxes
            fig.add_shape(
                type="rect", fillcolor=color, line_color=color,
                x0=x_base+0.3, x1=x_base+0.4,
                y0=y_pos-0.02, y1=y_pos+0.02
            )
            # Text
            fig.add_trace(go.Scatter(
                x=[x_base+0.5], y=[y_pos],
                mode='text', text=f"{name} ({color})",
                textposition='middle right',
                textfont=dict(size=12, color='white'),
                showlegend=False
            ))

        # Update layout
        fig.update_layout(
            showlegend=False,
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False,
                    range=[y_start-len(colors)*spacing-0.1, y_start+0.1]),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False,
                    range=[0, 1]),
            height=self.fig_param['size']['h'],
            width=self.fig_param['size']['w'], 
            margin=self.fig_param['margin'],
            template=self.fig_param['template'],
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    # analysis
    def _liquidity_analysis(self, portfolio_df:pd.DataFrame=None):
        portfolio_df = portfolio_df or self.portfolio_df
        
        # get buy sell dates for each trade
        last_portfolio = portfolio_df.shift(1).fillna(0).infer_objects(copy=False)
        buys = (last_portfolio == 0) & (portfolio_df != 0)
        sells = (last_portfolio != 0) & (portfolio_df == 0)

        buy_sells = pd.concat([
            buys.stack().loc[lambda x: x].reset_index(name='action').assign(action='b'),
            sells.stack().loc[lambda x: x].reset_index(name='action').assign(action='s')
            ])\
            .rename(columns={'t_date':'buy_date', 'level_1': 'stock_id'})\
            .sort_values(by=['stock_id', 'buy_date'])\
            .assign(sell_date=lambda x: x.groupby('stock_id')['buy_date'].shift(-1))

        buy_sells = buy_sells[buy_sells['action'] == 'b']\
            .dropna()\
            .sort_values(by=['stock_id', 'buy_date'])\
            .reset_index(drop=True)\
            [['stock_id', 'buy_date', 'sell_date']]

        # load data
        filters=[
            ('stock_id', 'in', list(buy_sells['stock_id'].unique())),
            ('date', 'in', list(set(list(buy_sells['buy_date'].unique().strftime('%Y-%m-%d')) + list(buy_sells.sell_date.unique().strftime('%Y-%m-%d'))))),
          ]

        liquidity_status=pd.merge(
          _db.read_dataset(
              'stock_trading_notes', 
              columns=['date', 'stock_id', '是否為注意股票', '是否為處置股票', '是否全額交割', '漲跌停註記'],
              filters=filters
          ),
          _db.read_dataset(
              'stock_trading_data',
              columns=['date', 'stock_id', '成交量(千股)', '成交金額(元)'],
              filters=filters,
          ),
          on=['date', 'stock_id'],
          how='inner',
          )

        # merge
        return pd.concat([
            pd.merge(buy_sells, liquidity_status, left_on=['stock_id', 'buy_date'], right_on=['stock_id', 'date'], how='left'), 
            pd.merge(buy_sells, liquidity_status, left_on=['stock_id', 'sell_date'], right_on=['stock_id', 'date'], how='left'),
          ])
    
    def _calc_relative_return(self, daily_return:pd.DataFrame=None, downtrend_window:int=3):
        daily_rtn = daily_return or self.daily_return
        benchmark_daily_returns = self.return_df[self.benchmark[0]]
        relative_rtn = (1 + daily_rtn - benchmark_daily_returns).cumprod() - 1

        relative_rtn_diff = relative_rtn.diff()
        downtrend_mask = relative_rtn_diff < 0
        downtrend_mask = pd.DataFrame({'alpha': relative_rtn, 'downward': downtrend_mask})
        downtrend_mask['group'] = (downtrend_mask['downward'] != downtrend_mask['downward'].shift()).cumsum()
        downtrend_returns = np.where(
            downtrend_mask['downward'] & downtrend_mask.groupby('group')['downward'].transform('sum').ge(downtrend_window), 
            downtrend_mask['alpha'], 
            np.nan
        )

        return relative_rtn, downtrend_returns

    def _calc_rolling_beta(self, window:int=240, daily_rtn:pd.DataFrame=None, benchmark_daily_returns:pd.DataFrame=None):
        daily_rtn = daily_rtn or self.daily_return
        benchmark_daily_returns = benchmark_daily_returns or self.return_df[self.benchmark[0]]

        rolling_cov = daily_rtn.rolling(window).cov(benchmark_daily_returns)
        rolling_var = benchmark_daily_returns.rolling(window).var()
        return (rolling_cov / rolling_var).dropna()

    def _calc_ir(self, window:int=240, daily_rtn:pd.DataFrame=None, benchmark_daily_returns:pd.DataFrame=None):
        daily_rtn = daily_rtn or self.daily_return
        benchmark_daily_returns = benchmark_daily_returns or self.return_df[self.benchmark[0]]

        excess_return = (1 + daily_rtn - benchmark_daily_returns).rolling(window).apply(np.prod, raw=True) - 1
        tracking_error = excess_return.rolling(window).std()
        return (excess_return / tracking_error).dropna()

    def _calc_ic(self,factor:pd.DataFrame, group:int=10, rebalance:Literal['MR', 'QR', 'W', 'M', 'Q', 'Y']=None):
        # factor rank
        rebalance = rebalance or self.rebalance
        r_date = _get_rebalance_date(rebalance)
        f_rank = (pd.qcut(factor[factor.index.isin(r_date)].stack(), q=group, labels=False, duplicates='drop') + 1)\
        .reset_index()\
        .rename(columns={'level_1':'stock_id', 0:'f_rank'})    

        # return rank
        return_df = self.return_df
        factor_r =  return_df\
            .groupby(pd.cut(return_df.index, bins=r_date))\
            .apply(lambda x: (1 + x).prod() - 1).dropna()\
            .stack()\
            .reset_index()\
            .rename(columns={'level_0':'t_date', 'level_1':'stock_id', 0: 'return'})\
            .assign(t_date=lambda df: df['t_date'].apply(lambda x: x.left).astype('datetime64[ns]'))
        
        # ic
        factor_return_rank = pd.merge(f_rank, factor_r, on=['t_date', 'stock_id'], how='inner').dropna()\
            .groupby(['t_date', 'f_rank'])['return']\
            .mean()\
            .reset_index()\
            .assign(r_rank = lambda df: df.groupby('t_date')['return'].rank())
        return factor_return_rank.groupby('t_date', group_keys=False).apply(
            lambda group: stats.spearmanr(group['f_rank'], group['r_rank'])[0], include_groups=False)

    # plot
    def _plot_equity_curve(self, c_return:pd.DataFrame=None, bmk_c_return:pd.DataFrame=None, portfolio_df:pd.DataFrame=None):
        
        c_return = c_return or (1 + self.daily_return).cumprod() - 1
        bmk_c_return = bmk_c_return or (1 + self.return_df[self.benchmark[0]]).cumprod() - 1
        portfolio_df = portfolio_df or self.portfolio_df
        
        fig = make_subplots(
            rows=3, cols=1, vertical_spacing=0.05, 
            shared_xaxes=True,
            )
        
        # meta strategy section
        if hasattr(self, 'strategies') and self.strategies:
            strategies_rtn = pd.concat([v[0].daily_return.rename(k) for k, v in self.strategies.items()], axis=1).dropna()
            strategies_c_rtn = (1 + strategies_rtn).cumprod() - 1
            strategies_drawdown = (strategies_c_rtn+1 - (strategies_c_rtn+1).cummax()) / (strategies_c_rtn+1).cummax()


            n_strategies = len(strategies_c_rtn.columns)
            colors = sample_colorscale('Teal', [0.6 + (n * 0.4/(n_strategies-1)) for n in range(n_strategies)])
            # colors = sample_colorscale(c, [n / (n_strategies-1) for n in range(n_strategies)])
            color_scale = dict(zip(strategies_c_rtn.columns, colors))
            
            for col in strategies_c_rtn.columns:
                fig.append_trace(go.Scatter(
                    x=strategies_c_rtn.index,
                    y=strategies_c_rtn[col].values,
                    name=col,
                    legendgroup=col,
                    legendrank=10000+strategies_c_rtn.columns.get_loc(col),
                    line=dict(color=color_scale[col], width=2),
                    ), row=1, col=1)
                fig.append_trace(go.Scatter(
                    x=strategies_drawdown.index,
                    y=strategies_drawdown[col].values,
                    name=col,
                    legendgroup=col,
                    legendrank=10000+strategies_c_rtn.columns.get_loc(col),
                    showlegend=False,
                    line=dict(color=color_scale[col], width=1.5),
                    ), row=2, col=1)

        # equity curve
        fig.append_trace(go.Scatter(
            x=bmk_c_return.index,
            y=bmk_c_return.values,
            showlegend=False,
            name='Benchmark',
            line=dict(color=self.fig_param['colors']['Dark Grey'], width=2),
            ), row=1, col=1)
        fig.append_trace(go.Scatter(
            x=c_return.index,
            y=c_return.values,
            name='Strategy',
            line=dict(color=self.fig_param['colors']['Dark Blue'], width=2),
            ), row=1, col=1)

        # drawdown
        drawdown = (c_return+1 - (c_return+1).cummax()) / (c_return+1).cummax()
        bmk_drawdown = (bmk_c_return+1 - (bmk_c_return+1).cummax()) / (bmk_c_return+1).cummax()

        fig.append_trace(go.Scatter(
            x=bmk_drawdown.index,
            y=bmk_drawdown.values,
            name='Benchmark-DD',
            showlegend=False,
            line=dict(color=self.fig_param['colors']['Dark Grey'], width=2),
            ), row=2, col=1)
        fig.append_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name='Strategy-DD',
            line=dict(color=self.fig_param['colors']['Bright Magenta'], width=1.5),
            ), row=2, col=1)

        # portfolio size
        p_size = portfolio_df.apply(lambda row: np.count_nonzero(row), axis=1)
        fig.append_trace(go.Scatter(
            x=p_size.index,
            y=p_size.values,
            name='Strategy-size',
            line=dict(color=self.fig_param['colors']['Bright Orange'], width=2),
            ), row=3, col=1)

        # adjust time range
        fig.update_xaxes(rangeslider_visible=True, rangeslider=dict(thickness=0.01), row=3, col=1)

        # asjust y as % for equity curve & drawdown
        fig.update_yaxes(tickformat=".0%", row=1, col=1)
        fig.update_yaxes(tickformat=".0%", row=2, col=1)

        # position
        fig.update_yaxes(domain=[0.5, 0.95], row=1, col=1)
        fig.update_yaxes(domain=[0.2, 0.4], row=2, col=1)
        fig.update_yaxes(domain=[0, 0.1], row=3, col=1)

        # titles
        fig.add_annotation(text=self._bold(f'Cumulative Return'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
        fig.add_annotation(text=self._bold(f'Drawdown'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
        fig.add_annotation(text=self._bold(f'Portfolio Size'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=3, col=1)

        # fig layout
        fig.update_layout(
            legend=dict(x=0.01, y=0.95, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)', traceorder='normal'),
            width=self.fig_param['size']['w'],
            height=self.fig_param['size']['h'],
            margin=self.fig_param['margin'],
            template=self.fig_param['template'],
        )
        return fig

    def _plot_return_heatmap(self, daily_return:pd.DataFrame=None):
        # prepare data
        daily_return = daily_return or self.daily_return
        monthly_return = ((1 + daily_return).resample('ME').prod() - 1)\
            .reset_index()\
            .assign(
                y=lambda x: x['t_date'].dt.year.astype('str'),
                m=lambda x: x['t_date'].dt.month,
                )
        annual_return = (daily_return + 1).groupby(daily_return.index.year).prod() - 1
        my_heatmap = monthly_return.pivot(index='y', columns='m', values=0).iloc[::-1]
        avg_mth_return = monthly_return.groupby('m')[0].mean()

        # plot
        fig = make_subplots(rows=3, cols=1, vertical_spacing=0.1)
        pos_colorscale = self.fig_param['pos_colorscale']
        neg_colorscale = self.fig_param['neg_colorscale']

        # annual return heatmap
        annual_return_pos = np.where(annual_return > 0, annual_return, np.nan)
        annual_return_neg = np.where(annual_return <= 0, annual_return, np.nan)

        # annual return for positive values
        fig.append_trace(
            go.Heatmap(
                z=annual_return_pos.reshape(1, -1),
                x=annual_return.index.astype('str'),
                y=[''], 
                text=annual_return.apply(lambda x: f"{x:.1%}").values.reshape(1, -1),
                texttemplate="%{text}",
                textfont=dict(size=9),
                colorscale = pos_colorscale,
                showscale = False,
                zmin=0,
                zmax=0.4,
            ),
            row=1, col=1
        )

        # annual return for negative values
        fig.append_trace(
            go.Heatmap(
                z=annual_return_neg.reshape(1, -1),
                x=annual_return.index.astype('str'),
                y=[''], 
                text=annual_return.apply(lambda x: f"{x:.1%}").values.reshape(1, -1),
                texttemplate="%{text}",
                textfont=dict(size=9),
                colorscale = neg_colorscale,
                showscale = False,
                zmin=-0.4,
                zmax=0,
            ),
            row=1, col=1
        )


        # monthly return heatmap
        my_heatmap_pos = np.where(my_heatmap > 0, my_heatmap, np.nan)
        my_heatmap_neg = np.where(my_heatmap <= 0, my_heatmap, np.nan)

        # monthly return for positive values
        fig.append_trace(
            go.Heatmap(
                z=my_heatmap_pos,
                x=my_heatmap.columns,
                y=my_heatmap.index,
                text=np.where(my_heatmap_pos > 0, np.vectorize(lambda x: f"{x:.1%}")(my_heatmap_pos), ""),
                texttemplate="%{text}",
                textfont=dict(size=10),
                colorscale = pos_colorscale,
                showscale = False,
                zmin=0,
                zmax=0.1,
            ),
            row=2, col=1
        )

        # monthly return for negative values
        fig.append_trace(
            go.Heatmap(
                z=my_heatmap_neg,
                x=my_heatmap.columns,
                y=my_heatmap.index,
                text=np.where(my_heatmap_neg <= 0, np.vectorize(lambda x: f"{x:.1%}")(my_heatmap_neg), ""),
                texttemplate="%{text}",
                textfont=dict(size=10),
                colorscale = neg_colorscale,
                showscale = False,
                zmin=-0.1,
                zmax=0,
            ),
            row=2, col=1
        )

        # monthly average return
        fig.append_trace(
        go.Bar(
            x=avg_mth_return.index,
            y=avg_mth_return.values,
            marker=dict(color=self.fig_param['colors']['Bright Orange']),
            name='Avg. return',
            
        ),
        row=3, col=1
        )
        fig.add_hline(
            y=0,
            line=dict(color="white", width=1),
            row=3, col=1
        )

        # adjust for axes display
        fig.update_xaxes(tickfont=dict(size=11), row=1, col=1)
        fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=[calendar.month_abbr[i] for i in range(1, 13)], row=2, col=1)
        fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=[calendar.month_abbr[i] for i in range(1, 13)], row=3, col=1, matches='x2')
        fig.update_yaxes(tickformat=".01%", row=3, col=1)

        # position
        fig.update_yaxes(domain=[0.9, 0.95], row=1, col=1)
        fig.update_yaxes(domain=[0.3, 0.75], row=2, col=1)
        fig.update_yaxes(domain=[0, 0.15], row=3, col=1)

        # titles
        fig.add_annotation(text=self._bold(f'Annual Return'), x=0, y=1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
        fig.add_annotation(text=self._bold(f'Monthly Return'), x=0, y=1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
        fig.add_annotation(text=self._bold(f'Avg. monthly Return'), x=0, y=1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=3, col=1)

        # fig size, colorscale & bgcolor
        fig.update_layout(
            width=self.fig_param['size']['w'],
            height=self.fig_param['size']['h'],
            margin=self.fig_param['margin'],
            template=self.fig_param['template'],
        )

        return fig

    def _plot_liquidity(self, portfolio_df:pd.DataFrame=None, money_threshold:int=500000, volume_threshold:int=50):
        df=self._liquidity_analysis(portfolio_df)

        # lqd_heatmap
        low_money = (df['成交金額(元)'] < money_threshold).mean()
        low_volume = (df['成交量(千股)'] < volume_threshold).mean()
        全額交割 = (df['是否全額交割']=='Y').mean()
        注意股票 = (df['是否為注意股票']=='Y').mean()
        處置股票 = (df['是否為處置股票']=='Y').mean()
        buy_limit = ((df['date']==df['sell_date'])&(df['漲跌停註記'] == '-')).mean()
        sell_limit = ((df['date']==df['buy_date'])&(df['漲跌停註記'] == '+')).mean()
        lqd_heatmap = pd.DataFrame({      
            f'money < {numerize(money_threshold)}': [low_money],
            f'volume < {numerize(volume_threshold)}': [low_volume],
            '全額交割': [全額交割],
            '注意股票': [注意股票],
            '處置股票': [處置股票],
            'buy limit up': [buy_limit],
            'sell limit down': [sell_limit],
            },
            index=[0],
        )

        # safety capacity
        principles = [500000, 1000000, 5000000, 10000000, 50000000, 100000000]
        buys_lqd = df[df['buy_date']==df['date']]
        vol_ratio = 10
        capacity = pd.DataFrame({
            'principle': [numerize(p) for p in principles], # Format principle for better readability
            'safety_capacity': [
                (buys_lqd.assign(
                capacity=lambda x: x['成交金額(元)'] / vol_ratio - (principle / x.groupby('buy_date')['stock_id'].transform('nunique')),
                spill=lambda x: np.where(x['capacity'] < 0, x['capacity'], 0)
                )
                .groupby('buy_date')['spill'].sum() * -1 / principle).mean()
                for principle in principles
                ]
        })
        capacity['safety_capacity'] = 1 - capacity['safety_capacity']

        # entry, exit volume
        def volume_threshold(df):
            thresholds = [100000, 500000, 1000000, 10000000, 100000000]
            labels = [f'{numerize(thresholds[i-1])}-{numerize(th)}' 
                        if i > 0 else f'<= {numerize(th)}'
                        for i, th in enumerate(thresholds)]
            labels.append(f'> {numerize(thresholds[-1])}')
            
            return df.assign(
                money_threshold=pd.cut(df['成交金額(元)'], [0] + thresholds + [np.inf], labels=labels)
                )\
                .groupby('money_threshold', observed=True)['stock_id']\
                .count()\
                .pipe(lambda x: (x / x.sum()))

        ee_volume = pd.concat([
            volume_threshold(df[df['buy_date']==df['date']]).rename('buy'),
            volume_threshold(df[df['sell_date']==df['date']]).rename('sell')
        ], axis=1)

        # plot
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"colspan": 2}, None],
                [{}, {},]
            ],
            vertical_spacing=0.05,
            horizontal_spacing=0.1,
            )

        # lqd_heatmap
        fig.add_trace(go.Heatmap(
            z=lqd_heatmap.loc[0].values.reshape(-1, 1), 
            x=lqd_heatmap.columns.values, 
            y=[''], 
            text = lqd_heatmap.loc[0].apply(lambda x: f"{x:.2%}").values.reshape(1, -1),
            texttemplate="%{text}",
            transpose=True,
            textfont=dict(size=10),
            colorscale = 'Plotly3',
            showscale = False,
            zmin=0,
            zmax=0.1,
        ),
        row=1, col=1)

        # safety capacity
        fig.add_trace(go.Bar(
            x=capacity['principle'],
            y=capacity['safety_capacity'],
            showlegend=False,
            marker_color=self.fig_param['colors']['Bright Orange'],
            name='Safety cap',
            text=capacity.safety_capacity.apply(lambda x: f"{x:.2%}").values,
            textposition='inside',
            width=0.7
        ),
        row=2, col=1)

        # entry, exit volume
        fig.add_trace(go.Bar(
            x=ee_volume.index,
            y=ee_volume.buy.values,
            name='Entry',
            marker_color=self.fig_param['colors']['Bright Cyan'], 
        ),
        row=2, col=2)
        fig.add_trace(go.Bar(
            x=ee_volume.index,
            y=ee_volume.sell.values,
            name='Exit',
            marker_color=self.fig_param['colors']['Bright Red'], 
        ),
        row=2, col=2)

        # adjust axes
        fig.update_xaxes(tickfont=dict(size=11), side='top', row=1, col=1)
        fig.update_xaxes(tickfont=dict(size=11), title_text='total capital', row=2, col=1)
        fig.update_xaxes(tickfont=dict(size=11), title_text="trading money", row=2, col=2)

        fig.update_yaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(tickformat=".0%", row=2, col=1)
        fig.update_yaxes(tickformat=".0%", row=2, col=2)

        # add titles
        fig.add_annotation(text=self._bold(f'Liquidity Heatmap'), x=0, y = 1, yshift=50, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
        fig.add_annotation(text=self._bold(f'Safety Capacity'), align='left', x=0, y = 1, yshift=50, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
        fig.add_annotation(text=f'(Ratio of buy money ≤ 1/{vol_ratio} of volume for all trades)', align='left', x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, font=dict(size=12), row=2, col=1)
        fig.add_annotation(text=self._bold(f'Trading Money at Entry/Exit'), x=0, y = 1, yshift=50, xref="x domain", yref="y domain", showarrow=False, row=2, col=2)

        # position
        fig.update_yaxes(domain=[0.87, 0.92], row=1, col=1)
        fig.update_yaxes(domain=[0.2, 0.65], row=2, col=1, range=[0, 1])
        fig.update_yaxes(domain=[0.2, 0.65], row=2, col=2, dtick=0.1)

        # laoyout
        fig.update_layout(
            legend=dict(x=0.55, y=0.65, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)', traceorder='normal'),
            coloraxis1_showscale=False,
            coloraxis1=dict(colorscale='Plasma'),
            width=self.fig_param['size']['w'],
            height=self.fig_param['size']['h'],
            margin=self.fig_param['margin'],
            template=self.fig_param['template'],
            )
        return fig
    
    def _plot_maemfe(self):
        maemfe = self.maemfe
        win = maemfe[maemfe['return']>0]
        lose = maemfe[maemfe['return']<=0]

        fig = make_subplots(
            rows=2, cols=3,
            vertical_spacing=0,
            horizontal_spacing=0,
            )

        # colors
        light_blue = self.fig_param['colors']['Light Blue']
        light_red = self.fig_param['colors']['Light Red']
        dark_blue = self.fig_param['colors']['Dark Blue']
        dark_red = self.fig_param['colors']['Bright Red']

        # return distribution
        fig.add_trace(go.Histogram(
            x=win['return'], 
            marker_color=light_blue, 
            name='win',
            legendgroup='win',
        ), row=1, col=1)
        fig.add_trace(go.Histogram(
            x=lose['return'], 
            marker_color=light_red, 
            name ='lose',
            legendgroup ='lose',
        ), row=1, col=1)
        fig.add_vline(
            x=maemfe['return'].mean(), 
            line_width=2, 
            line_dash="dash", 
            line_color="Blue",
            annotation_text=f'avg. rtn: {maemfe["return"].mean() :.2%}',
            annotation_font_color='Blue',
        row=1, col=1)
        
        # bmfe/ mae
        fig.add_trace(go.Scatter(
            x = win['gmfe'], 
            y = win['mae']*-1, 
            mode='markers', 
            marker_color=light_blue, 
            text=win.index,
            textposition="top center",
            name='win',
            legendgroup='win',
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x = lose['gmfe'], 
            y = lose['mae']*-1, 
            mode='markers', 
            marker_color=light_red, 
            text=lose.index,
            textposition="top center",
            name='lose',
            legendgroup ='lose',
        ), row=1, col=2)
        
        # gmfe/mdd
        fig.add_trace(go.Scatter(
            x = win['gmfe'], 
            y = win['mdd']*-1, 
            mode='markers', 
            marker_color=light_blue, 
            name='win',
            text=win.index,
            textposition="top center",
            legendgroup='win',
        ), row=1, col=3)
        fig.add_trace(go.Scatter(
            x = lose['gmfe'], 
            y = lose['mdd']*-1, 
            mode='markers', 
            marker_color=light_red, 
            name='lose',
            text=lose.index,
            textposition="top center",
            legendgroup ='lose',
        ), row=1, col=3)
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=1, y1=1,
            xref='x3', yref='y3',
            line=dict(color="Orange", width=2, dash="dash")
        )

        # positions
        fig.update_xaxes(tickformat=".0%", domain=[0, 0.25], title_text='return', row=1, col=1)
        fig.update_xaxes(tickformat=".0%", domain=[0.25+0.125, 0.5+0.125], title_text="gmfe", row=1, col=2)
        fig.update_xaxes(tickformat=".0%", domain=[0.75, 1], title_text="gmfe", row=1, col=3)
        
        fig.update_yaxes(domain=[0.625, 0.95], title_text='count', title_standoff=2, row=1, col=1)
        fig.update_yaxes(tickformat=".0%", domain=[0.625, 0.95], title_text='mae', title_standoff=2, row=1, col=2)
        fig.update_yaxes(tickformat=".0%", domain=[0.625, 0.95], title_text='mdd', title_standoff=2, row=1, col=3)

        # distributions
        def plot_distributions(item, x):
            jitter = 1e-10
            win_data = x*win[item] + jitter
            lose_data = x*lose[item] + jitter
            
            distplot=ff.create_distplot(
                [win_data, lose_data], 
                ['win', 'lose'], 
                colors=['#636EFA', '#EF553B'],
                bin_size=0.01, 
                histnorm='probability', 
                # curve_type='normal',
                show_rug=False
            )
            distplot.update_traces(dict(marker_line_width=0))
            return distplot

        distributions={
            'mae':{'row':2, 'col':1},
            'bmfe':{'row':2, 'col':2},
            'gmfe':{'row':2, 'col':3},
        }

        for key, value in distributions.items():
            x = -1 if key =='mae' else 1
            distplot = plot_distributions(key, x)
            
            # histogram
            fig.add_trace(go.Histogram(
                distplot['data'][0], 
                name='win',
                legendgroup='win', 
            ), row=value.get('row'), col=value.get('col'))
            fig.add_trace(go.Histogram(
                distplot['data'][1], 
                legendgroup ='lose', 
            ), row=value.get('row'), col=value.get('col'))
            
            # curve lines
            fig.add_trace(go.Scatter(
                distplot['data'][2],
                marker_color='blue'
            ), row=value.get('row'), col=value.get('col'))
            fig.add_trace(go.Scatter(
                distplot['data'][3],
                marker_color='red'
            ), row=value.get('row'), col=value.get('col'))
            
            # add v lines & annotations
            q3_win, q3_lose  = (x*win[key]).quantile(0.75), (x*lose[key]).quantile(0.75)
            fig.add_vline(
                x=q3_win, 
                line_width=2, line_dash="dash", line_color=dark_blue,
                row=value.get('row'), col=value.get('col'))
            fig.add_vline(
                x=q3_lose, 
                line_width=2, line_dash="dash", line_color=dark_red,
                row=value.get('row'), col=value.get('col'))
            fig.add_annotation(
                x=q3_win,
                y = 1,
                yref="y domain",
                # y='paper',
                text=f'Q3 {key} win: {q3_win:.2%}',
                showarrow=False,
                xshift=70,
                font=dict(color=dark_blue),
                row=value.get('row'), col=value.get('col'))
            fig.add_annotation(
                x=q3_lose,
                y = 0.9,
                yref="y domain",
                text=f'Q3 {key} lose: {q3_lose:.2%}',
                showarrow=False,
                xshift=70,
                # yshift=-20,
                font=dict(color=dark_red),
                row=value.get('row'), col=value.get('col'))
            
            # update yaxes
            fig.update_yaxes(domain=[0.05, 0.425], row=value.get('row'), col=value.get('col'))

            # update xaxes
            col=value.get('col')
            domain_start = (col - 1) * 0.375
            domain_end = domain_start + 0.25
            fig.update_xaxes(tickformat=".0%", domain=[domain_start, domain_end], title_text=key, row=value.get('row'), col=value.get('col'))

        # add titles
        fig.add_annotation(text=self._bold(f'Return Distribution<br>win rate: {len(win)/len(maemfe):.2%}'), x=0.5, y = 1, yshift=45, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
        fig.add_annotation(text=self._bold(f'GMFE/ MAE'), x=0.5, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=2)
        fig.add_annotation(text=self._bold(f"GMFE/ MDD<br>missed profit pct:{len(win[win['mdd']*-1 > win['gmfe']])/len(win):.2%}"), x=0.5, y = 1, yshift=45, xref="x domain", yref="y domain", showarrow=False, row=1, col=3)
        fig.add_annotation(text=self._bold(f'MAE distribution'), x=0.5, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
        fig.add_annotation(text=self._bold(f'BMFE distribution'), x=0.5, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=2, col=2)
        fig.add_annotation(text=self._bold(f'GMFE distribution'), x=0.5, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=2, col=3)

        # layout
        fig.update_layout(
            width=self.fig_param['size']['w'],
            height=self.fig_param['size']['h'],
            margin=self.fig_param['margin'],
            template=self.fig_param['template'],
        )
        
        return fig
    
    def _plot_relative_return(self, daily_return:pd.DataFrame=None, bmk_equity_df:pd.DataFrame=None):
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"colspan": 2, "secondary_y": True}, None],
                [{}, {}],
            ],
            vertical_spacing=0.05,
            horizontal_spacing=0.1,
            shared_xaxes=True,
        )

        # relative return
        relative_rtn, downtrend_returns = self._calc_relative_return(daily_return=daily_return)
        bmk_rtn = bmk_equity_df or (1+self.return_df[self.benchmark[0]]).cumprod() - 1

        strategy_trace = go.Scatter(
            x=relative_rtn.index,
            y=relative_rtn.values,
            name='Relative Return (lhs)',
            line=dict(color=self.fig_param['colors']['Dark Blue'], width=2),
            mode='lines',
            yaxis='y1'
        )

        downtrend_trace = go.Scatter(
            x=relative_rtn.index,
            y=downtrend_returns,
            name='Downtrend (lhs)',
            line=dict(color=self.fig_param['colors']['Bright Red'], width=2),
            mode='lines',
            yaxis='y1'
        )

        benchmark_trace = go.Scatter(
            x=bmk_rtn.index,
            y=bmk_rtn.values,
            name='Benchmark Return (rhs)',
            line=dict(color=self.fig_param['colors']['Dark Grey'], width=2),
            mode='lines',
            yaxis='y2'
        )

        fig.add_trace(strategy_trace, row=1, col=1)
        fig.add_trace(downtrend_trace, row=1, col=1)
        fig.add_trace(benchmark_trace, row=1, col=1, secondary_y=True)  

        # rolling beta
        rolling_beta = self._calc_rolling_beta()

        beta_trace = go.Scatter(
            x=rolling_beta.index,
            y=rolling_beta.values,
            name='Rolling Beta',
            line=dict(color=self.fig_param['colors']['Bright Orange'], width=1),
            showlegend=False,
        )
        
        fig.add_trace(beta_trace, row=2, col=1)
        fig.add_hline(
                    y=rolling_beta.mean(),
                    line=dict(color="white", width=1),
                    line_dash="dash",
                    annotation_text=f'avg. beta: {rolling_beta.mean() :.2}',
                    annotation_position="bottom left",
                    annotation_textangle = 0,
                    row=2, col=1
                )

        # information ratio
        info_ratio = calc_info_ratio(daily_rtn=self.daily_return, benchmark_daily_returns=self.return_df[self.benchmark[0]])

        info_ratio_trace = go.Scatter(
            x=info_ratio.index,
            y=info_ratio.values,
            name='Rolling info Ratio',
            line=dict(color=self.fig_param['colors']['Bright Magenta'], width=1),
            showlegend=False,
        )
        fig.add_trace(info_ratio_trace, row=2, col=2)
        fig.add_hline(
                    y=info_ratio.mean(),
                    line=dict(color="white", width=1),
                    line_dash="dash",
                    annotation_text=f'avg. info ratio: {info_ratio.mean() :.2}',
                    annotation_position="bottom left",
                    annotation_textangle = 0,
                    row=2, col=2
                )

        # rangeslider_visible
        fig.update_xaxes(rangeslider_visible=True, rangeslider=dict(thickness=0.01), row=1, col=1)

        # adjust axes
        fig.update_xaxes(tickfont=dict(size=12), row=1, col=1)
        fig.update_yaxes(tickformat=".0%", row=1, col=1, secondary_y=False, dtick=2)
        fig.update_yaxes(tickformat=".0%", row=1, col=1, secondary_y=True, showgrid=False) # second y-axis

        # position
        fig.update_xaxes(domain=[0.025, 0.975], row=1, col=1)
        fig.update_xaxes(domain=[0.025, 0.45], row=2, col=1)
        fig.update_xaxes(domain=[0.55, 0.975], row=2, col=2)

        fig.update_yaxes(domain=[0.6, .95], row=1, col=1)
        fig.update_yaxes(domain=[0, 0.4], row=2, col=1)
        fig.update_yaxes(domain=[0, 0.4], row=2, col=2)

        # title
        fig.add_annotation(text=self._bold(f'Relative Return'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
        fig.add_annotation(text=self._bold(f'Beta'), align='left', x=0, y = 1, yshift=40, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
        fig.add_annotation(text=self._bold(f'Information Ratio(IR)'), align='left', x=0, y = 1, yshift=40, xref="x domain", yref="y domain", showarrow=False, row=2, col=2)
        fig.add_annotation(text='(Relative return / tracking error)', align='left', x=0, y = 1, yshift=20, xref="x domain", yref="y domain", showarrow=False, font=dict(size=12), row=2, col=2)

        # laoyout
        fig.update_layout(
            legend=dict(x=0.05, y=0.94, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)'),
            width= self.fig_param['size']['w'],
            height= self.fig_param['size']['h'],
            margin= self.fig_param['margin'],
            template= self.fig_param['template'],
            )
        
        return fig

    def _plot_style_analysis(self):
        betas = self.betas
        fig = make_subplots(
            rows=2, cols=1,
            specs=[
                [{"secondary_y": True}],
                [{}],
            ],
            vertical_spacing=0.05,
            horizontal_spacing=0.1,
        )

        # rolling beta
        alpha = (1 + betas.iloc[:-1]['const']).cumprod() - 1
        betas = betas.drop(columns=['const'])
        betas = betas.reindex(columns=betas.iloc[-1].sort_values(ascending=False).index).round(3)

        colors = sample_colorscale('Spectral', len(betas.columns))
        for i, col in enumerate(betas.columns):
            rolling_beta = betas.iloc[:-1]
            total_beta = betas.iloc[-1]
            # return (total_beta)
            fig.add_trace(
                go.Scatter(
                    x=rolling_beta.index, 
                    y=rolling_beta[col], 
                    name=col,
                    line=dict(color=colors[i]),
                    showlegend=True,
                    legendgroup=col,
                ),
                secondary_y=True,
                row=1, col=1,
            )

            fig.add_trace(
                go.Bar(
                    x=[col],
                    y=[total_beta[col]],
                    text=[f'{total_beta[col]:.3f}'],
                    textposition='auto',
                    marker_color=colors[i],
                    name=col,
                    showlegend=False,
                    legendgroup=col,
                    width=0.6,
                ),
                row=2, col=1
            )

        fig.add_trace(
            go.Scatter(
                x=alpha.index, 
                y=alpha.values, 
                name='Alpha (rhs)',
                line=dict(color=self.fig_param['colors']['Light Grey'], width=2),
                showlegend=True,
            ),
            secondary_y=False,
            row=1, col=1,
        )

        # adjust axes
        fig.update_yaxes(tickformat=".0%", row=1, col=1, secondary_y=False, showgrid=False)
        fig.update_yaxes(row=2, col=1, showgrid=False)

        # position
        fig.update_xaxes(domain=[0.025, 0.975], row=1, col=1)
        fig.update_xaxes(domain=[0.025, 0.975], row=2, col=1)

        fig.update_yaxes(domain=[0.55, 0.975], row=1, col=1)
        fig.update_yaxes(domain=[0, 0.3], row=2, col=1)

        # titles
        fig.add_annotation(text=self._bold('Rolling beta'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
        fig.add_annotation(text=self._bold('Total beta'), x=0, y = 1, yshift=40, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)

        # layout
        fig.update_layout(
            legend = dict(x=0.05, y=0.5, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)', tracegroupgap=.25, orientation='h'),
            width = self.fig_param['size']['w'],
            height = self.fig_param['size']['h'],
            margin = self.fig_param['margin'],
            template = self.fig_param['template'],
            yaxis = dict(side='right'),
            yaxis2 = dict(side='left'),
        )
        fig.update_polars(radialaxis_showline=False)

        return fig

    def _plot_efficiency_frontier(self):
        returns = pd.concat([v[0].daily_return.rename(k) for k, v in self.strategies.items()], axis=1).dropna()
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.05)

        results = create_random_portfolios(returns)
        max_sharpe = results.loc[results['sharpe'].argmax()]
        min_vol = results.loc[results['std_dev'].argmin()]
        
        fig.add_trace(go.Scatter(
            x=results['std_dev'],
            y=results['return'],
            mode='markers',
            marker=dict(
                size=5,
                color=results['sharpe'],
                colorscale='PuBu',
                # showscale=True,
                opacity=0.3
            ),
            name='Weights',
            text=results['weights'].astype(str),
            showlegend=False,
        ), row=1, col=1)

        # Add maximum Sharpe ratio point
        fig.add_trace(go.Scatter(
            x=[max_sharpe['std_dev']],
            y=[max_sharpe['return']],
            mode='markers',
            marker=dict(size=10, symbol='diamond'),
            name=f"Max Sharpe: {str(max_sharpe['weights'])}",
            text=[str(max_sharpe['weights'])],
            showlegend=True,
        ), row=1, col=1)

        # Add minimum volatility point
        fig.add_trace(go.Scatter(
            x=[min_vol['std_dev']],
            y=[min_vol['return']],
            mode='markers',
            marker=dict(size=10, symbol='diamond'),
            name=f"Min volatility: {str(min_vol['weights'])}",
            text=[str(min_vol['weights'])],
            showlegend=True
        ), row=1, col=1)
        
        corr_plot = returns.corr()
        corr_plot = corr_plot\
            .mask(np.tril(np.ones(corr_plot.shape)).astype(bool))

        fig.add_trace(go.Heatmap(
            z=corr_plot.values,
            x=corr_plot.columns,
            y=corr_plot.index,
            colorscale='RdBu_r',
            text=corr_plot.map("{:.2f}".format).replace('nan', '').values,
            texttemplate="%{text}",
            showscale=False,
        ), row=2, col=1)
        
        fig.update_yaxes(tickformat=".0%", row=1, col=1)
        fig.update_xaxes(tickformat=".0%", row=1, col=1)
        
        # position
        fig.update_yaxes(domain=[0.0, 1.0], row=1, col=1)
        fig.update_yaxes(domain=[0.15, 0.45], row=2, col=1)
        fig.update_xaxes(domain=[0.0, 1.0], row=1, col=1)
        fig.update_xaxes(domain=[0.7, 1.0], row=2, col=1)

        # title
        fig.add_annotation(text=self._bold(f'Efficiency frontier'), x=0, y=1, yshift=40, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
        fig.add_annotation(text=self._bold(f'Corr'), x=0.1, y=1, yshift=20, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)

        # Update layout
        fig.update_layout(
            legend=dict(x=0.01, y=1, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)', traceorder='normal'),
            xaxis_title='Annualised volatility',
            yaxis_title='Annualised returns',
            width=self.fig_param['size']['w'],
            height=self.fig_param['size']['h'],
            template=self.fig_param['template'],
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            xaxis2=dict(showgrid=False),
            yaxis2=dict(showgrid=False),
        )

        return fig


class Strategy(PlotMaster):
    """
    回測策略類別，用於儲存回測結果並提供分析功能。

    Returns:
        Strategy: 回測策略物件，包含以下功能:
        
            - summary: 回測績效摘要，包含年化報酬率、總報酬率、最大回撤、波動率等指標
            - position_info: 持倉資訊，包含股票代碼、名稱、買賣日期、警示狀態、漲跌停、成交量額、財報公佈日期、板塊產業別等資訊
            - report: 回測報告圖表，包含淨值曲線、相對報酬、報酬熱圖、流動性分析等

    Examples:
        ```python
        # 建立回測策略
        strategy = backtesting(
            data,
            rebalance='Q',
            signal_shift=1,
            benchmark='0050'
        )

        # 查看績效摘要
        strategy.summary
        '''
                           strategy    0050.TT
        Annual return       12.5%       8.2%
        Total return        45.3%      28.4%
        Max drawdown       -15.2%     -18.5%
        Annual vol          18.2%      16.8%
        Sharpe ratio         0.69       0.49
        Calmar ratio         0.82       0.44
        beta                 0.85         --
        '''

        # 查看持倉資訊
        strategy.position_info
        '''
                    name       buy_date   sell_date  警示狀態  漲跌停  前次成交量_K  前次成交額_M  季報公佈日期    月營收公佈日期  板塊別      主產業別
        stock_id                                                                       
        1436         全家     2024-11-15  2025-04-01    =      =        32        6.08    2024-11-06  2025-01-09   上櫃一般版  OTC38 OTC 居家生活
        1446         宏和     2024-11-15  2025-04-01    =      =       254        9.79    2024-11-12  2025-01-10   上市一般版  M1400 紡織纖維  
        1519         華城     2024-11-15  2025-04-01    =      =      8714     4780.49    2024-11-12  2025-01-09   上市一般版  M1500 電機機械
        2101         南港     2024-11-15  2025-04-01    =      =      1231       55.48    2024-11-13  2025-01-07   上市一般版  M2100 橡膠工業
        2404         漢唐     2024-11-15  2025-04-01    =      =       170       18.47    2024-11-12  2025-01-10   上市一般版  M2300 電子工業
        '''

        # 顯示回測報告
        strategy.report
        '''
        [Equity Curve] [Relative Return] [Return Heatmap] [Liquidity]
        '''
        ```

    Note:
        - summary、position_info、report 為回測結果物件的主要功能
        - 可透過這些功能快速了解策略績效表現
        - report 提供互動式圖表分析工具
    """
    
    def __init__(self, return_df:pd.DataFrame, buy_list:Union[pd.DataFrame, pd.Series], portfolio_df:Union[pd.DataFrame, pd.Series], backtest_df:Union[pd.DataFrame, pd.Series], benchmark:Union[str, list[str]], rebalance:str):
        super().__init__()
        self.return_df = return_df
        self.buy_list = buy_list
        self.portfolio_df = portfolio_df
        self.backtest_df = backtest_df
        self.benchmark = [benchmark] if isinstance(benchmark, str) else benchmark
        self.rebalance = rebalance
        
        # analysis
        self.daily_return = backtest_df.sum(axis=1)
        self.c_return = (1 + self.daily_return).cumprod() - 1
        self.maemfe = calc_maemfe(self.buy_list, self.portfolio_df, self.return_df)
        self.betas = calc_portfolio_style(self.daily_return, total=True)
        
        # results
        self.summary = self._generate_summary()
        self.report = self._generate_report()
        self.position_info = self._generate_position_info()

        # display
        display(self.summary)

        logging.info(f"Created Strategy with AAR: {self.summary['Strategy']['annual_return']}; MDD: {self.summary['Strategy']['mdd']}; Avol: {self.summary['Strategy']['annual_vol']}")

    def _generate_summary(self) -> pd.DataFrame:
        
        summary = calc_metrics(self.daily_return, self.return_df[self.benchmark[0]]).to_frame('Strategy')
        for bmk in self.benchmark:
            summary = pd.concat([
                summary, 
                calc_metrics(self.return_df[bmk], None).to_frame(f'{bmk}.TT')
            ], axis=1)
        return summary
    
    def _generate_position_info(self) -> pd.DataFrame:
        """顯示持倉資訊

        Args:
            rebalance (str): 再平衡方式

        Returns:
            DataFrame: 包含股票代碼、名稱、買入日期、賣出日期、警示狀態、漲跌停、前次成交量、前次成交額、季報公佈日期、月營收公佈日期、市場別、主產業別的資料表
        """
        # print('make sure to update databank before get info!')

        # 獲取買入和賣出日期
        rebalance_dates = _get_rebalance_date(self.rebalance, end_date=_t_date['t_date'].max())
        buy_date = self.buy_list.index[-1]
        try:
            sell_date = next(d for d in rebalance_dates if d > buy_date)
        except StopIteration as e:
            logging.error(e)
            print(e)
            sell_date = None
        
        # 獲取持倉資訊
        position_info = self.buy_list.iloc[-1, :]
        position_info = position_info[position_info]\
            .reset_index()\
            .rename(columns={'index':'stock_id', position_info.name:'buy_date'})\
            .assign(buy_date=position_info.name)\
            .assign(sell_date=sell_date)

        filters=[('stock_id', 'in', position_info['stock_id'].to_list())]
        
        # 獲取股票中文名稱
        ch_name = _db.read_dataset('stock_basic_info', columns=['stock_id', '證券名稱'], filters=filters).rename(columns={'證券名稱':'name'})

        # 獲取財報和月營收公佈日期
        quarterly_report_release_date = _db.read_dataset('fin_data', columns=['stock_id', 'release_date'], filters=filters)\
            .loc[lambda x: x.groupby('stock_id')['release_date'].idxmax()]\
            .rename(columns={'release_date':'季報公佈日期'})
        monthly_rev_release_date = _db.read_dataset('monthly_rev', columns=['stock_id', 'release_date'], filters=filters)\
            .loc[lambda x: x.groupby('stock_id')['release_date'].idxmax()]\
            .rename(columns={'release_date':'月營收公佈日期'})

        filters.append(('date', '=', self.return_df.index.max()))

        # 獲取市場類型、產業類別、交易狀態
        trading_notes = _db.read_dataset('stock_trading_notes', columns=['date', 'stock_id', '是否為注意股票', '是否為處置股票', '是否暫停交易', '是否全額交割', '漲跌停註記', '板塊別(中)', '主產業別(中)'], filters=filters)\
        .assign(
            警示狀態=lambda x: x.apply(
                lambda row: ', '.join(filter(None, [
                    '注意股' if row['是否為注意股票'] == 'Y' else '',
                    '處置股' if row['是否為處置股票'] == 'Y' else '', 
                    '暫停交易' if row['是否暫停交易'] == 'Y' else '',
                    '全額交割' if row['是否全額交割'] == 'Y' else '',
                ])), axis=1
            ).replace('', '='),
            漲跌停=lambda x: x['漲跌停註記'].replace('', '=')
        )\
        .rename(columns={'主產業別(中)':'主產業別', '板塊別(中)':'板塊別'})\
        [['stock_id', '警示狀態', '漲跌停', '板塊別', '主產業別']]
        
        # 獲取交易量和交易金額
        trading_vol = _db.read_dataset('stock_trading_data', columns=['stock_id', '成交量(千股)', '成交金額(元)'], filters=filters)\
        .rename(columns={
            '成交量(千股)':'前次成交量_K',
            '成交金額(元)':'前次成交額_M',
                        })\
        .assign(前次成交額_M=lambda x: (x['前次成交額_M']/1e6).round(2))

        # 合併所有資訊
        return position_info\
            .merge(ch_name, on='stock_id', how='left')\
                .merge(quarterly_report_release_date, on='stock_id', how='left')\
                .merge(monthly_rev_release_date, on='stock_id', how='left')\
                .merge(trading_notes, on='stock_id', how='left')\
                .merge(trading_vol, on='stock_id', how='left')\
                [[
                    'stock_id', 'name', 'buy_date', 'sell_date', 
                    '警示狀態', '漲跌停',
                    '前次成交量_K', '前次成交額_M', 
                    '季報公佈日期', '月營收公佈日期', 
                    '板塊別', '主產業別',
                ]].set_index('stock_id')

    def _generate_report(self, exclude_tabs:list[str]=[]) -> pn.Tabs:
        # main plots
        pn.extension('plotly')
        
        # Define mapping of tab names to plot functions
        plot_funcs = {
            'Equity curve': self._plot_equity_curve,
            'Relative return': self._plot_relative_return,
            'Style analysis': self._plot_style_analysis,
            'Return heatmap': self._plot_return_heatmap,
            'Liquidity': self._plot_liquidity,
            'MAE/MFE': self._plot_maemfe,
            # 'Optimal Params': self._plot_optimal_params,
        }

        figs = {
            name: pn.pane.Plotly(plot_funcs[name]())
            for name in plot_funcs
            if name not in exclude_tabs
        }
        tab_items = list(figs.items())

        return pn.Tabs(*tab_items)


class MultiStrategy(Strategy):
    """
    多重策略組合類別，用於組合多個策略並提供分析功能。

    Attributes:
    - strategies (dict[str, Tuple[Strategy, Union[float, int]]]): 策略字典，格式為:
            
            - key: 策略名稱
            - value: (Strategy物件, 權重)的tuple
    - benchmark (Union[str, list[str]]): 基準指標，可為單一或多個股票代碼
    - rebalance (str): 再平衡頻率，可選:

            - MR: 每月10日後第一個交易日
            - QR: 每季財報公布日後第一個交易日(3/31, 5/15, 8/14, 11/14)
            - W: 每週一
            - M: 每月第一個交易日
            - Q: 每季第一個交易日
            - Y: 每年第一個交易日
        return_df (Union[pd.Series, pd.DataFrame]): 個股報酬率資料
        daily_return (Union[pd.Series, pd.DataFrame]): 策略每日報酬率序列

    Returns:
        MultiStrategy: 多重策略組合物件，包含:

            - summary: 策略績效摘要，包含總報酬率、年化報酬率、最大回撤等重要指標
            - position_info: 持股資訊，包含每日持股數量、換手率、個股權重等資訊
            - report: 完整的回測報告，包含:
                - Equity curve: 淨值曲線
                - Efficiency frontier: 效率前緣
                - Portfolio style: 投資風格分析
                - Return heatmap: 報酬率熱圖
                - MAE/MFE: 最大不利變動/最大有利變動分析

    Examples:
        ```python
        # 建立兩個策略
        strategy1 = backtesting(data1, rebalance='Q')
        strategy2 = backtesting(data2, rebalance='M')

        # 組合策略回測(等權重)
        meta = multi_backtesting({
            'strategy1': strategy1,
            'strategy2': strategy2
        })

        # 組合策略回測(自訂權重)
        meta = multi_backtesting({
            'strategy1': (strategy1, 0.7),
            'strategy2': (strategy2, 0.3)
        })

        # 組合策略回測(每月再平衡)
        meta = multi_backtesting({
            'strategy1': (strategy1, 0.7),
            'strategy2': (strategy2, 0.3)
        }, rebalance='M')
        ```

    Note:
        - 繼承自Strategy類別，保留原有的分析功能
        - 新增效率前緣分析，用於最佳化策略權重配置
        - 可設定再平衡頻率，定期調整策略權重
        - 支援多個基準指標的比較
    """
    
    def __init__(self, strategies:dict[str, Tuple[Strategy, Union[float, int]]], benchmark:Union[str, list[str]], rebalance:str, return_df:Union[pd.Series, pd.DataFrame], daily_return:Union[pd.Series, pd.DataFrame]):
        PlotMaster.__init__(self)
        self.strategies = strategies
        self.rebalance = rebalance
        self.benchmark = [benchmark] if isinstance(benchmark, str) else benchmark
        self.return_df = return_df
        self.daily_return = daily_return
        self.portfolio_df = sum(v[0].portfolio_df * v[1]/sum(w[1] for w in strategies.values()) for v in strategies.values()).dropna(how='all', axis=0).fillna(0)

        # analysis
        self.c_return = (1 + daily_return).cumprod() - 1
        self.maemfe = pd.concat([
            v[0].maemfe.reset_index()\
                .assign(index=lambda x: k+' '+x['index'].astype(str))\
                .set_index('index')\
            for k, v in self.strategies.items()
        ], axis=0)
        self.betas = calc_portfolio_style(self.daily_return, total=True)
        # self.brinson_model = calc_brinson_model(self.portfolio_df, self.return_df, self.benchmark)

        # material
        self.summary = self._generate_meta_summary()
        self.position_info = self._generate_meta_position_info()
        self.report = self._generate_meta_report()
        display(self.summary)

    def _generate_meta_summary(self) -> pd.DataFrame:
        bmk = self.return_df[self.benchmark[0]]
        summary = calc_metrics(self.daily_return, bmk).to_frame('Strategy')
        
        for k, v in self.strategies.items():
            summary = pd.concat([
                summary,
                calc_metrics(v[0].daily_return.loc[self.daily_return.index], bmk).to_frame(k)
            ], axis=1)

        for bmk in self.benchmark:
            summary = pd.concat([
                summary, 
                calc_metrics(self.return_df[bmk], None).to_frame(f'{bmk}.TT')
            ], axis=1)
        return summary

    def _generate_meta_position_info(self) -> pd.DataFrame:
        return pd.concat([
            v[0].position_info.reset_index()\
                .assign(strategy=k)\
                .set_index('strategy')
                .assign(weight=lambda x: v[1]/sum(v1[1] for v1 in self.strategies.values()) * 1/len(x))\
            for k, v in self.strategies.items()
        ], axis=0)

    def _generate_meta_report(self, exclude_tabs:list[str]=[]) -> pn.Tabs:
        # main plots
        pn.extension('plotly')
        
        # Define mapping of tab names to plot functions
        plot_funcs = {
            'Equity curve': self._plot_equity_curve,
            'Efficiency frontier': self._plot_efficiency_frontier,
            'Portfolio style': self._plot_style_analysis,
            'Return heatmap': self._plot_return_heatmap,
            'MAE/MFE': self._plot_maemfe,
            # 'Liquidity': self._plot_liquidity,
        }

        figs = {
            name: pn.pane.Plotly(plot_funcs[name]())
            for name in plot_funcs
            if name not in exclude_tabs
        }
        tab_items = list(figs.items())

        return pn.Tabs(*tab_items)


# factor analysis
def factor_analysis(factor:pd.DataFrame, asc:bool=True, rebalance:str='QR', group:int=10, benchmark:str='0050')-> 'FactorAnalysis':
    return FactorAnalysis(factor, asc=asc, rebalance=rebalance, group=group, benchmark=benchmark)

class FactorAnalysis(PlotMaster):
    def __init__(self, factor:Union[pd.DataFrame, str], asc:bool=True, rebalance:str='QR', group:int=10, benchmark:str='0050'):
        super().__init__()
        self.factor = get_factor(factor, asc=asc)
        self.rebalance = rebalance
        self.group = group

        # data
        from quantdev.data import Databank
        db = Databank()
        self.return_df = db.read_dataset('stock_return', filter_date='t_date')
        self.quantiles_returns = calc_factor_quantiles_return(self.factor, rebalance=rebalance, group=group, return_df=self.return_df)
        self.ls_returns = (1 + self.quantiles_returns[max(self.quantiles_returns.keys())]) - (1 + self.quantiles_returns[0])
        self.benchmark = benchmark
        self.benchmark_returns = self.return_df[benchmark]

        # analysis
        self.betas = calc_portfolio_style(self.ls_returns, total=True)
        self.info_coef = calc_info_coef(self.factor, self.return_df, 10, 'W')
        
        # result
        self.summary = self._generate_summary()
        self.report = self._generate_report()

        display(self.summary)

    def _plot_factor_returns(self):
        returns = self.quantiles_returns
        return_df = self.return_df

        
        fig = make_subplots(
                    rows=2, cols=2,
                    specs=[
                        [{"colspan": 2, "secondary_y": True}, None],
                        [{}, {}],
                    ],
                    vertical_spacing=0.05,
                    horizontal_spacing=0.1,
                )

        # relative return
        # base_key = 0
        # # for key in sorted(results.keys()):
        # #     if (results[key]).isna().count() / len(pd.DataFrame(results)) >0.5:
        # #         base_key = key
        # #         break
        ls_daily_returns = self.ls_returns
        benchmark_daily_returns = return_df['0050']

        relative_returns, downtrend_returns = calc_relative_return(ls_daily_returns, benchmark_daily_returns)
        benchmark_c_returns = (1+benchmark_daily_returns).cumprod()-1

        # relative return
        strategy_trace = go.Scatter(
            x=relative_returns.index,
            y=relative_returns.values,
            name='Relative Return (lhs)',
            line=dict(color=self.fig_param['colors']['Dark Blue'], width=2),
            mode='lines',
            yaxis='y1',
        )

        # downtrend
        downtrend_trace = go.Scatter(
            x=relative_returns.index,
            y=downtrend_returns,
            name='Downtrend (lhs)',
            line=dict(color=self.fig_param['colors']['Bright Red'], width=2),
            mode='lines',
            yaxis='y1',
        )

        # factor return
        ls_c_returns  = (1+ls_daily_returns).cumprod()-1
        factor_return_trace = go.Scatter(
            x=ls_c_returns.index,
            y=ls_c_returns.values,
            name='Factor Return (rhs)',
            line=dict(color=self.fig_param['colors']['White'], width=2),
            mode='lines',
            yaxis='y2',
        )


        benchmark_trace = go.Scatter(
            x=benchmark_c_returns.index,
            y=benchmark_c_returns.values,
            name='Benchmark Return (rhs)',
            line=dict(color=self.fig_param['colors']['Dark Grey'], width=2),
            mode='lines',
            yaxis='y2',
        )

        fig.add_trace(strategy_trace, row=1, col=1)
        fig.add_trace(downtrend_trace, row=1, col=1)
        fig.add_trace(factor_return_trace, row=1, col=1, secondary_y=True)
        fig.add_trace(benchmark_trace, row=1, col=1, secondary_y=True)

        # c return
        num_lines = len(returns)
        color_scale = sample_colorscale('PuBu', [n / num_lines for n in range(num_lines)])
        for (key, value), color in zip(reversed(list(returns.items())), color_scale):
            value = (1+value).cumprod()-1
            annual_rtn = (1 + value.iloc[-1]) ** (240 / len(value)) - 1 if not value.empty else 0
            fig.add_trace(go.Scatter(
                x=value.index,
                y=value.values,
                name=int(key),
                line=dict(color=color),
                showlegend=False,
            ),
            row=2, col=1)

            # annual return
            fig.add_annotation(
                x=value.index[-1],
                y=value.values[-1], 
                xref="x", 
                yref="y", 
                text=f'{int(key)}: {annual_rtn: .2%}',  
                showarrow=False, 
                xanchor="left", 
                yanchor="middle", 
                font=dict(color="white", size=10),
            row=2, col=1)

        # IC
        ic = self.info_coef.rolling(window=20).mean()

        fig.add_trace(go.Scatter(
            x=ic.index, 
            y=ic.values, 
            mode='lines', 
            name='IC', 
            line=dict(color=self.fig_param['colors']['Bright Orange'], width=1),
            showlegend=False,
            ),
            row=2, col=2)
        fig.add_hline(
            y=0,
            line=dict(color="white", width=1),
            row=2, col=2
        )

        # rangeslider_visible
        fig.update_xaxes(rangeslider_visible=True, rangeslider=dict(thickness=0.01), row=1, col=1)

        # position
        fig.update_xaxes(domain=[0.025, 0.975], row=1, col=1)
        fig.update_xaxes(domain=[0.025, 0.45], row=2, col=1)
        fig.update_xaxes(domain=[0.55, 0.975], row=2, col=2)

        fig.update_yaxes(domain=[0.6, .95], row=1, col=1)
        fig.update_yaxes(domain=[0, 0.4], row=2, col=1)
        fig.update_yaxes(domain=[0, 0.4], row=2, col=2)

        # adjust axes
        fig.update_xaxes(tickfont=dict(size=12), row=1, col=1)
        fig.update_yaxes(tickformat=".0%", row=1, col=1, secondary_y=False)
        fig.update_yaxes(tickformat=".0%", row=1, col=1, secondary_y=True, showgrid=False) # second y-axis
        fig.update_yaxes(tickformat=".0%", row=2, col=1)

        # add titles
        fig.add_annotation(text=self._bold(f'Factor Relative Returns'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
        fig.add_annotation(text=self._bold(f'Factor Return by Quantiles'), x=0, y = 1, yshift=50, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
        fig.add_annotation(text=self._bold(f'Information Coefficient(IC)'), x=0, y = 1, yshift=50, xref="x domain", yref="y domain", showarrow=False, row=2, col=2)

        ic_mean = ic.mean()
        ic_std = ic.std()
        info_ratio = ic_mean/ic_std
        positive_ratio = len(ic[ic>0])/len(ic)
        fig.add_annotation(text=f'Mean: {ic_mean :.2f}, SD: {ic_std :.2f}, IR: {info_ratio :.2f}, Positive Ratio: {positive_ratio:.0%}', x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=2, col=2)

        # wrap up
        fig.update_layout(
            legend=dict(x=0.05, y=0.94, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)'),    
            height=self.fig_param['size']['h'], 
            width=self.fig_param['size']['w'], 
            margin= self.fig_param['margin'],
            template=self.fig_param['template'],
        )
        return fig

    def _generate_summary(self):
        self.summary_dfs = self._generate_summary_dfs()
        
        html = """
        <div style="display: flex; gap: 5px;row-gap: 10px;">
            <div>{}<br>{}</div>
            <div>{}</div>
            <div>{}</div>
        </div>
        """.format(
            self.summary_dfs['summary_metrics'].to_html(), 
            self.summary_dfs['ic'].to_html(), 
            self.summary_dfs['quantiles'].to_html(), 
            self.summary_dfs['styles'].to_html(),
        )
        return HTML(html)
    
    def _generate_summary_dfs(self):
        # Factor Returns
        returns = pd.concat([self.ls_returns.rename('f_return'), self.benchmark_returns], axis=1)\
            .dropna()\
            .assign(rel_return = lambda df: df['f_return']-df[self.benchmark])\
            [['f_return', 'rel_return', self.benchmark]]\
            .rename(columns={self.benchmark: f'{self.benchmark}.TT'})

        summary_metrics = pd.concat([calc_metrics(returns[col]).to_frame(('Performance', col)) for col in returns.columns], axis=1)\
            .drop(['beta'], axis=0)
        summary_metrics.columns = pd.MultiIndex.from_tuples(summary_metrics.columns)

        # Quantiles Returns
        quantiles = pd.DataFrame(self.quantiles_returns, columns=range(len(self.quantiles_returns))).dropna()\
            .assign(LS = lambda df: df[9]-df[0])\
            .apply(lambda x: ((1+x).cumprod()).iloc[-1] ** (240 / len(x)) -1)\
            .apply(lambda x: f'{x:.2%}')\
            .to_frame('Quantiles Returns')

        # Style
        betas = self.betas\
            .loc['total']\
            .drop('const')\
            .sort_values(ascending=False)\
            .to_frame('Style')\
            .round(4)
        alpha = self.betas[['const']]\
            .iloc[:-1]\
            .apply(lambda x: (1+x).cumprod().iloc[-1]** (240 / len(x)) -1)\
            .apply(lambda x: f'{x:.2%}')\
            .rename(index={'const': 'Annualized Alpha'})\
            .to_frame('Style')
        styles = pd.concat([betas, alpha], axis=0)

        # IC
        ic = self.info_coef.agg({
            'mean': lambda x: x.mean().round(4),
            'std': lambda x: x.std().round(4),
            'IR': lambda x: f'{x.mean()/x.std():.2%}', 
            'positive ratio': lambda x: f'{len(x[x>0])/len(x):.2%}',
        }).to_frame('Information Coefficient')

        return {
            'summary_metrics': summary_metrics,
            'quantiles': quantiles,
            'styles': styles,
            'ic': ic,
        }

    def _generate_report(self):
        
        # main plots
        pn.extension('plotly')
        
        # Define mapping of tab names to plot functions
        plot_funcs = {
            'Factor returns': self._plot_factor_returns,
            'Style analysis': self._plot_style_analysis,
        }

        figs = {
            name: pn.pane.Plotly(plot_funcs[name]())
            for name in plot_funcs
        }
        tab_items = list(figs.items())

        return pn.Tabs(*tab_items)