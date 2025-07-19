"""回測

Available Functions:
    - get_data: 從資料庫取得資料並進行預處理
    - get_factor: 將資料轉換為因子值
    - backtesting: 進行回測並返回完整的回測結果
    - fast_backtesting: 進行快速回測並返回每日報酬率
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

from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Union, Tuple
from IPython.display import display, HTML
import datetime as dt
import pandas as pd
import numpy as np
import panel as pn
import logging
import tqdm

from .analysis import *
from .data import DatasetsHandler
from .plot import *
from .utils import *

pd.set_option('future.no_silent_downcasting', True)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# backtest
def get_release_pct_rebalance(type_:Literal['MR', 'QR'], pct:int=80):
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
        dates = get_release_pct_rebalance('MR', pct=80)

        # 取得90%公司發布季報後的交易日:
        dates = get_release_pct_rebalance('QR', pct=90)
        ```

    Note:
        - MR (月營收): 檢查1-30天的發布情況
        - QR (季報): 檢查1-90天的發布情況
    """

    map = {
        'MR': {'dataset': 'monthly_rev','days': range(1, 31)},
        'QR': {'dataset': 'fin_data','days': range(1, 91)}
    }
    
    data = DatasetsHandler().read_dataset(map[type_]['dataset'], columns=['date', 'stock_id', 'release_date'])
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
    result = DatasetsHandler().add_t_date(result)[['date', 't_date', 'release_pct']]
    
    # Filter and transform the result to get rebalancing dates
    return result\
        [result['release_pct'] >= pct/100]\
        .groupby('date')\
        .first()\
        .reset_index()\
        ['t_date']\
        .tolist()

def get_rebalance_date(rebalance:Literal['D', 'MR', 'QR', 'W', 'M', 'Q', 'Y'], start:Union[str, int]=None, end:Union[str, int]=None):
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
        dates = get_rebalance_date('M', end_date='2023-12-31')
        ```

        取得每季財報公布後再平衡日期:
        ```python
        dates = get_rebalance_date('QR')
        ```

    Note:
        - 開始日期為資料庫中最早的交易日
        - 結束日期若為 None 則為今日加5天
        - 所有再平衡日期都會對應到實際交易日
    """
    
    # dates
    end = (dt.datetime.today() + dt.timedelta(days=1)).strftime('%Y-%m-%d') if end is None else end
    t_date = DatasetsHandler().read_dataset('mkt_calendar', columns=['date'], filters=[('休市原因中文說明_人工建置','=','')], start=start, end=end).rename(columns={'date':'t_date'})
    start_date = t_date['t_date'].min()
    end_date = t_date['t_date'].max()

    if rebalance.startswith('W'):
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
            date_list = get_release_pct_rebalance(rebalance.split('-')[0], int(rebalance.split('-')[1]))
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
            date_list = get_release_pct_rebalance(rebalance.split('-')[0], int(rebalance.split('-')[1]))
        r_date = pd.DataFrame(date_list, columns=['r_date'])
    elif rebalance == 'M':
        r_date = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='MS'), columns=['r_date'])
    elif rebalance == 'Q': 
        r_date = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='QS'), columns=['r_date'])    
    elif rebalance == 'Y':
        r_date = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='YS'), columns=['r_date'])
    elif rebalance in ['otcs', 'otc_short']:
        return None
    else:
        raise ValueError("Invalid frequency. Allowed values are 'W', 'MR, 'QR', 'M', 'Q', 'Y'.")

    r_date['r_date'] = r_date['r_date'].astype('datetime64[ms]')
    
    return pd.merge_asof(
        r_date, t_date, 
        left_on='r_date', 
        right_on='t_date', 
        direction='forward'
        )['t_date'].to_list()

def get_portfolio(
    data:pd.DataFrame, exp_returns:pd.DataFrame, rebalance_dates:list=None, 
    weights:pd.DataFrame=None, 
    signal_shift:int=0, 
    hold_period:int=None, 
    weight_limit:float=None,
    rebalance:str=None,
)-> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    將資料轉換為投資組合權重。

    Args:
        data (pd.DataFrame): 選股條件矩陣，index為日期，columns為股票代碼
        exp_returns (pd.DataFrame): 股票報酬率矩陣，index為日期，columns為股票代碼
        rebalance (str): 再平衡頻率，可選 'MR'(月營收公布後), 'QR'(財報公布後), 'W'(每週), 'M'(每月), 'Q'(每季), 'Y'(每年), 'D'(每日/ 當沖)
        weights (pd.DataFrame): 自定義權重矩陣，index為日期，columns為股票代碼
        signal_shift (int): 訊號延遲天數，用於模擬實際交易延遲
        hold_period (int): 持有期間，若為None則持有至下次再平衡日
        weight_limit (float): 權重限制，例如0.5代表單個股權重不能超過50%

    Returns:
        tuple: 包含以下兩個元素:
            - buy_list (pd.DataFrame): 選股結果矩陣，index為再平衡日期，columns為股票代碼
            - portfolio (pd.DataFrame): 投資組合權重矩陣，index為日期，columns為股票代碼

    Examples:
        取得每月第一個交易日再平衡、訊號延遲1天、持有20天的投資組合:
        ```python
        buy_list, portfolio = get_portfolio(data, exp_returns, rebalance='M', signal_shift=1, hold_period=20)
        ```

    Note:
        - 投資組合權重為等權重配置，除非提供自定義權重
        - 若無持有期間限制，則持有至下次再平衡日
        - 訊號延遲用於模擬實際交易所需的作業時間
    """
    
    # filter 漲跌停
    # buy_list = buy_list[buy_list.apply(lambda x: x.isin([0, 1]).all())]
    
    # weight
    if rebalance in ['dt_short', 'DTS']:
        buy_list = data
        portfolio = buy_list.astype(int)\
            .apply(lambda x: x / x.sum(), axis=1)
        return buy_list, portfolio
    if rebalance == 'event':
        buy_list = data
        portfolio = buy_list\
            .astype(int)\
            .replace({0:np.nan})\
            .reindex(index = exp_returns.index)\
            .ffill(limit=hold_period)\
            .apply(lambda x: x / x.sum(), axis=1)\
            .fillna(0)
        return buy_list, portfolio
    else:
        buy_list =  data[data.index.isin(rebalance_dates)]
        portfolio = (buy_list.astype(int) if weights is None else (buy_list * weights))\
            .apply(lambda x: x / x.sum(), axis=1)

        if weight_limit is not None:
            portfolio = portfolio.clip(upper=weight_limit)

        # shift & hold_period
        if signal_shift in [0, None]:
            portfolio = portfolio\
                .reindex(index = exp_returns.index, method='ffill', limit=hold_period)\
                .fillna(0)
        else:
            portfolio = portfolio\
                .reindex(index = exp_returns.index)\
                .shift(signal_shift)\
                .ffill(limit=hold_period)\
                .fillna(0)
        return buy_list, portfolio

def stop_loss_or_profit(buy_list:pd.DataFrame, portfolio_df:pd.DataFrame, exp_returns:pd.DataFrame, pct:float, stop_at:Literal['intraday', 'next_day']='intraday'):
    # calculate portfolio returns by multiplying portfolio positions (0/1) with return data
    portfolio_return = (portfolio_df != 0).astype(int) * exp_returns[portfolio_df.columns].loc[portfolio_df.index]
    
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
    rebalance:Literal['MR', 'QR', 'W', 'M', 'Q', 'Y', 'otcs', 'otc_short', 'event']='QR', 
    longshort:Union[Literal[1], Literal[-1]]=1, 
    weights:pd.DataFrame=None, signal_shift:int=0, hold_period:int=None, weight_limit:float=None,
    fee:float=0.001425, fee_discount:float=0.25, tax:float=0.003, slippage:float=0.001, trading_cost:bool=True,
    stop_loss:Union[float, pd.DataFrame]=None, stop_profit:Union[float, pd.DataFrame]=None, stop_at:Literal['intraday', 'next_day']='next_day', 
    start:Union[int, str]='2006-01-01', end:Union[int, str]=None, 
    benchmark_id:str='TRTEJ', report:bool=False,
    exp_returns:pd.DataFrame=None,
    **kwargs,
)-> Union[pd.DataFrame, 'Report']:
    """回測函數，返回投資組合每日報酬率序列。

    Attributes:
        data: 選股條件矩陣，index為日期，columns為股票代碼，True代表交易訊號
        longshort: 多空方向，1為做多，-1為做空
        rebalance: 再平衡頻率，可選MR(月營收公布日)、QR(季報公布日)、W(週)、M(月)、Q(季)、Y(年)、D(當沖/日)
        weights: 自定義權重矩陣，若為None則使用等權重
        signal_shift: 訊號延遲天數，用於模擬實際交易延遲
        hold_period: 持有期間天數，若為None則持有至下次再平衡日
        weight_limit: 單一標的權重上限
        fee: 交易手續費率
        fee_discount: 手續費折扣
        tax: 證券交易稅率
        start: 回測起始日期
        end: 回測結束日期
        exp_returns: 預期報酬率矩陣，若為None則自動讀取

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
        - 投資組合權重預設為等權重配置，除非提供自定義權重
        - 若無持有期間限制，則持有至下次再平衡日
    """
    if rebalance in ['dt_short', 'DTS']:
        exp_returns = DatasetsHandler().read_dataset('update_exp_returns_otc_short', filter_date='t_date', start=start, end=end) if exp_returns is None else exp_returns.copy()
        rebalance_dates = None
    elif rebalance == 'event':
        exp_returns = DatasetsHandler().read_dataset('exp_returns', filter_date='t_date', start=start, end=end) if exp_returns is None else exp_returns
        rebalance_dates = None
        hold_period = 1 if hold_period is None else hold_period
        exp_returns *= longshort
    else:
        exp_returns = DatasetsHandler().read_dataset('exp_returns', filter_date='t_date', start=start, end=end) if exp_returns is None else exp_returns
        rebalance_dates = get_rebalance_date(rebalance)
        exp_returns *= longshort

    buy_list, portfolio_df = get_portfolio(
        data=data, 
        exp_returns=exp_returns, 
        rebalance_dates=rebalance_dates, 
        weights=weights, 
        weight_limit=weight_limit,
        signal_shift=signal_shift, 
        hold_period=hold_period, 
        rebalance=rebalance,
    )

    # stop loss or profit
    if stop_loss is not None:
        portfolio_df = stop_loss_or_profit(buy_list, portfolio_df, exp_returns, pct=abs(stop_loss)*-1, stop_at=stop_at).infer_objects(copy=False).replace({np.nan: 0})
    if stop_profit is not None:
        portfolio_df = stop_loss_or_profit(buy_list, portfolio_df, exp_returns, pct=stop_profit, stop_at=stop_at).infer_objects(copy=False).replace({np.nan: 0})

    # trading cost
    if trading_cost:
        if rebalance in ['dt_short', 'DTS']:
            exp_returns -= ((fee*fee_discount)*2 + tax/2 + slippage*2)
        else:
            buy = ((portfolio_df.shift(1) == 0) & (portfolio_df != portfolio_df.shift(1))).reindex(columns=exp_returns.columns, fill_value=False)
            sell = ((portfolio_df.shift(-1) == 0) & (portfolio_df != portfolio_df.shift(-1))).reindex(columns=exp_returns.columns, fill_value=False)
            exp_returns.iloc[:,:] = np.where(
                buy,
                exp_returns - (fee*fee_discount+slippage),
                exp_returns
            ) # buy
            exp_returns.iloc[:,:] = np.where(
                sell,
                exp_returns - (fee*fee_discount+tax+slippage),
                exp_returns
            ) # sell

    if report:
        return Report(
            portfolio_df=portfolio_df, 
            buy_list=buy_list,
            exp_returns=exp_returns,
            benchmark_id=benchmark_id,
            rebalance=rebalance,
        )
    else:
        return (exp_returns * portfolio_df).dropna(how='all').sum(axis=1)

def multi_backtesting(
    reports:dict[str, Union['Report', Tuple['Report', float]]], 
    rebalance:Literal['MR', 'QR', 'W', 'M', 'Q', 'Y']=None,
    benchmark_id:str='TRTEJ')-> 'MultiReport':
    
    """多重策略回測並返回組合策略回測結果。

    Args:
        - strategies (dict[str, Union['Report', Tuple['Report', float]]]): 策略字典，格式為:  
            - key: 策略名稱
            - value: Report物件，或是(Report物件, 權重)的tuple
        - rebalance (Literal['MR', 'QR', 'W', 'M', 'Q', 'Y'], optional): 再平衡頻率，可選:  
            - MR: 每月10日後第一個交易日
            - QR: 每季財報公布日後第一個交易日(3/31, 5/15, 8/14, 11/14)
            - W: 每週一
            - M: 每月第一個交易日
            - Q: 每季第一個交易日
            - Y: 每年第一個交易日
        - benchmark (Union[str, list[str]], optional): 基準指標，可為單一或多個股票代碼

    Returns:
        MultiReport: 組合策略回測結果物件，包含:

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

    if all(isinstance(v, Report) for v in reports.values()):
        reports = {k: (v, 1/len(reports)) for k, v in reports.items()}
    
    strategies_rtn = pd.concat([v[0].daily_return.rename(k) for k, v in reports.items()], axis=1).dropna()
    rebalance_weights = pd.DataFrame({k:v[1]/sum(v[1] for v in reports.values()) for k, v in reports.items()}, index=[strategies_rtn.index[0]])

    if rebalance is not None:
        rebalance_dates = [d for d in get_rebalance_date(rebalance) if d in strategies_rtn.index]
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
    
    # exp_returns
    exp_returns = DatasetsHandler().read_dataset('exp_returns')
    
    return MultiReport(reports=reports, rebalance=rebalance, benchmark_id=benchmark_id, exp_returns=exp_returns, daily_return=daily_return)

class Report:
    def __init__(self, portfolio_df:pd.DataFrame, buy_list:pd.DataFrame, exp_returns:pd.DataFrame, benchmark_id:str, rebalance:str, **kwargs):
        self.portfolio_df = portfolio_df
        self.buy_list = buy_list
        self.exp_returns = exp_returns
        self.benchmark_id = benchmark_id
        self.rebalance = rebalance
        self.daily_return = (exp_returns * portfolio_df).dropna(how='all').sum(axis=1)
        
        # metrics
        self.metrics = self.calc_metrics()
        display(self.metrics)

        # calculated
        self.liquidity = calc_liquidity(self.portfolio_df)
        self.style = calc_style(self.daily_return, window=None, total=True)
        # self.maemfe = calc_maemfe(self.buy_list, self.portfolio_df, self.exp_returns)
        self.relative_return = calc_relative_return(self.daily_return, self.exp_returns[self.benchmark_id])
        # tabs
        self.tabs = self.plot_tabs()

    def plot_tabs(self)-> pn.Tabs:
        pn.extension('plotly')

        plot_funcs = {
            'Equity curve': plot_equity_curve(
                (1+self.daily_return).cumprod()-1, 
                (1+self.exp_returns[self.benchmark_id]).cumprod()-1, 
                self.portfolio_df.apply(lambda row: np.count_nonzero(row), axis=1)
            ),
            'Relative Return': plot_relative_return(self.relative_return),
            'Return heatmap': plot_return_heatmap(self.daily_return),
            'Liquidity': plot_liquidity(self.liquidity),
            'Style': plot_style(self.style),
            # 'MAE/MFE': plot_maemfe(self.maemfe),
        }
        return pn.Tabs(*[(k, pn.pane.Plotly(v)) for k, v in plot_funcs.items()])

    def calc_metrics(self)-> pd.DataFrame:
        benchmark_daily_return = self.exp_returns[self.benchmark_id].dropna()
        return calc_metrics(
            pd.concat([
                self.daily_return.rename('Strategy'), 
                benchmark_daily_return.rename(f'Benchmark: {self.benchmark_id}'), 
                (self.daily_return-benchmark_daily_return).rename('Excess Return')
                ], axis=1))


class MultiReport:
    def __init__(self, daily_return:Union[pd.Series, pd.DataFrame], reports:dict[str, Tuple[Report, Union[float, int]]], rebalance:str, benchmark_id:str, exp_returns:Union[pd.Series, pd.DataFrame]):
        self.daily_return = daily_return
        self.reports = reports
        self.rebalance = rebalance
        self.exp_returns = exp_returns
        self.benchmark_id = benchmark_id
        self.portfolio_df = sum(v[0].portfolio_df for v in self.reports.values())
        self.metrics = self.calc_metrics()
        display(self.metrics)
        
        # analysis
        self.style = calc_style(self.daily_return, total=True)
        # self.maemfe = pd.concat([
        #     v[0].maemfe.reset_index()\
        #         .assign(index=lambda x: k+' '+x['index'].astype(str))\
        #         .set_index('index')\
        #     for k, v in self.reports.items()
        # ], axis=0)

        # material
        self.tabs = self.plot_tabs()
    
    def calc_metrics(self)-> pd.DataFrame:
        benchmark_daily_return = self.exp_returns[self.benchmark_id]
        return calc_metrics(pd.concat([
            self.daily_return.rename('Strategy'),
            benchmark_daily_return.rename(f'Benchmark: {self.benchmark_id}'),
            (self.daily_return - benchmark_daily_return).rename('Excess Return'),
            *[v[0].daily_return.rename(k) for k, v in self.reports.items()]
        ], axis=1).dropna())

    def plot_tabs(self)-> pn.Tabs:
        pn.extension('plotly')

        plot_funcs = {
            'Equity curve': plot_equity_curve(
                (1+pd.concat([
                    self.daily_return.rename('Strategy'), 
                    pd.concat({k:v[0].daily_return for k, v in self.reports.items()}, axis=1),
                ], axis=1)).cumprod()-1, 
                (1+self.exp_returns[self.benchmark_id]).cumprod()-1, 
                self.portfolio_df.apply(lambda row: np.count_nonzero(row), axis=1)
            ),
            'Efficiency frontier': plot_efficiency_frontier(
                pd.concat([v[0].daily_return.rename(k) for k, v in self.reports.items()], axis=1).dropna()
            ),
            'Return heatmap': plot_return_heatmap(self.daily_return),
            'Style': plot_style(self.style),
            # 'MAE/MFE': plot_maemfe(self.maemfe),
        }
        return pn.Tabs(*[(k, pn.pane.Plotly(v)) for k, v in plot_funcs.items()])


# factor
def calc_factor_longshort_return(factor:pd.DataFrame, rebalance:str='QR', group:int=10, **kwargs):
    """計算因子多空組合報酬率

    Args:
        factor (Union[pd.DataFrame, str]): 因子值矩陣或因子名稱
        asc (bool, optional): 因子值是否為越小越好. Defaults to True.
        rebalance (str, optional): 調倉頻率. Defaults to 'QR'.
        group (int, optional): 分組數量. Defaults to 10.

    Returns:
        pd.Series: 多空組合報酬率時間序列

    Examples:
        計算ROE因子的多空組合報酬率:
        ```python
        roe_ls = calc_factor_longshort_return('roe', asc=True, rebalance='QR', group=10)
        ```

    Note:
        - 多頭為因子值前1/group的股票
        - 空頭為因子值後1/group的股票
        - 報酬率為多頭減去空頭的報酬率
    """
    
    exp_returns = kwargs.get('exp_returns')
    if exp_returns is None:
        exp_returns = DatasetsHandler().read_dataset('exp_returns', start=kwargs.get('start'))
    long = backtesting(factor>=(1-1/group), rebalance=rebalance, exp_returns=exp_returns, trading_cost=False)
    short = backtesting(factor<(1/group), rebalance=rebalance, exp_returns=exp_returns, trading_cost=False)

    return (long-short).dropna()

def calc_factor_quantiles_return(factor:pd.DataFrame, rebalance:str='QR', group:int=10, add_ls:bool=False, start:str='2006-01-01', **kwargs):
    """計算因子分位數報酬率

    Args:
        factor (Union[pd.DataFrame, str]): 因子值矩陣或因子名稱
        asc (bool, optional): 因子值是否為越小越好. Defaults to True.
        rebalance (str, optional): 調倉頻率. Defaults to 'QR'.
        group (int, optional): 分組數量. Defaults to 10.

    Returns:
        dict: 各分位數報酬率時間序列

    Examples:
        計算ROE因子的十分位數報酬率:
        ```python
        roe_quantiles = calc_factor_quantiles_return('roe', asc=True, rebalance='QR', group=10)
        ```

    Note:
        - 將因子值依照大小分成group組
        - 每組報酬率為等權重持有該組內所有股票
        - 字典的key為分位數起始點*group (例如: 0代表第一組, 1代表第二組...)
    """
    
    vals = factor.to_rank().values
    groups = {
        lower_bound/group : pd.DataFrame((lower_bound/group < vals) & (vals <= upper_bound/group), index=factor.index, columns=factor.columns)
        for lower_bound, upper_bound in zip(range(0, group), range(1, group+1))
    }

    # return
    exp_returns = kwargs.get('exp_returns')
    if exp_returns is None:
        exp_returns = DatasetsHandler().read_dataset('exp_returns', filter_date='t_date', start=start) if rebalance not in ['otcs', 'otc_short'] else DatasetsHandler().read_dataset('update_exp_returns_otc_short', filter_date='t_date', start=start)
    rebalance_dates = get_rebalance_date(rebalance=rebalance)
    
    def process_group(k_v):
        k, v = k_v
        return k, (get_portfolio(
            data=v,
            exp_returns=exp_returns,
            rebalance_dates=rebalance_dates,
            rebalance=rebalance
        )[1] * exp_returns).dropna(how='all').sum(axis=1)

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = dict(tqdm.tqdm(executor.map(process_group, groups.items()), total=len(groups)))
    if add_ls:
        results['LS'] = results[max(results.keys())] - results[min(results.keys())]
    return results

def calc_independent_double_sorting(factors:Tuple[pd.DataFrame, pd.DataFrame], group:Union[int, list[int], Tuple[int, int]]=10, rebalance:str='QR', **kwargs):
    main_f, ctrl_f = factors

    # factor
    if isinstance(group, int):
        main_g_num = group
        ctrl_g_num = group
    elif isinstance(group, (list, tuple)):
        main_g_num, ctrl_g_num = group[0], group[1]

    main_vals = main_f.values
    ctrl_vals = ctrl_f.values
    main_groups = {
        lower_bound/main_g_num : pd.DataFrame((lower_bound/main_g_num < main_vals) & (main_vals <= upper_bound/main_g_num), index=main_f.index, columns=main_f.columns)
        for lower_bound, upper_bound in zip(range(0, main_g_num), range(1, main_g_num+1))
    }
    ctrl_groups = {
        lower_bound/ctrl_g_num : pd.DataFrame((lower_bound/ctrl_g_num < ctrl_vals) & (ctrl_vals <= upper_bound/ctrl_g_num), index=ctrl_f.index, columns=ctrl_f.columns)
        for lower_bound, upper_bound in zip(range(0, ctrl_g_num), range(1, ctrl_g_num+1))
    }
    total_groups = {
        (i, j): (main_groups[i] & ctrl_groups[j]).fillna(False)
        for j in ctrl_groups.keys()
        for i in main_groups.keys()
    }

    # return
    exp_returns = kwargs.get('exp_returns')
    if exp_returns is None:
        exp_returns = DatasetsHandler().read_dataset('exp_returns', filter_date='t_date', start=kwargs.get('start')) if rebalance not in ['dt_short', 'DTS'] else DatasetsHandler().read_dataset('update_exp_returns_otc_short', filter_date='t_date', start=kwargs.get('start'))
    rebalance_dates = get_rebalance_date(rebalance=rebalance)
    
    def process_group(k_v):
        k, v = k_v
        return k, (get_portfolio(
            data=v,
            exp_returns=exp_returns,
            rebalance_dates=rebalance_dates,
            rebalance=rebalance
        )[1] * exp_returns).dropna(how='all').sum(axis=1)

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = dict(tqdm.tqdm(executor.map(process_group, total_groups.items()), total=len(total_groups)))
    return results

# factor
def factor_analysis(
    ranked_factor:pd.DataFrame, 
    rebalance:str='QR', 
    group:int=10, 
    benchmark_id:str='TRTEJ',
    exp_returns:pd.DataFrame=None,
    start:str='2006-01-01',
    end:str=None,
)-> 'FactorReport':
    if exp_returns is None:
        exp_returns = DatasetsHandler().read_dataset('exp_returns', filter_date='t_date', start=start, end=end)

    quantiles_returns = pd.DataFrame(
        calc_factor_quantiles_return(ranked_factor, rebalance=rebalance, group=group, exp_returns=exp_returns, start=start)
    )
    
    return FactorReport(ranked_factor, quantiles_returns, rebalance, group, benchmark_id, exp_returns)

class FactorReport:
    def __init__(self, factor:pd.DataFrame, quantiles_returns:dict, rebalance:str, group:int, benchmark_id:str, exp_returns:pd.DataFrame):
        self.factor = factor
        self.quantiles_returns = quantiles_returns
        self.ls_returns = quantiles_returns[quantiles_returns.columns.max()] - quantiles_returns[quantiles_returns.columns.min()]
        self.rebalance = rebalance
        self.group = group
        self.benchmark_id = benchmark_id
        self.exp_returns = exp_returns
        
        # analysis
        self.style = calc_style(self.ls_returns)
        self.ic = calc_ic(self.factor, self.exp_returns, rebalance)
        self.ir = calc_ir(self.factor, self.exp_returns, rebalance)

        # metrics
        self.metrics = self.calc_metrics()
        display(self.metrics)

        # plot
        self.tabs = self.plot_tabs()
    
    def calc_metrics(self)-> pd.DataFrame:
        main = calc_metrics(self.quantiles_returns.assign(LS=lambda df:df[df.columns.max()] - df[df.columns.min()]))
        
        ic = calc_ic_metrics(ic=self.ic).to_frame('Rank IC')

        html = f"""
        <div style="display: flex; gap: 5px;row-gap: 10px;">
            <div>{main.to_html()}<br>{ic.to_html()}</div>
        </div>
        """
        return HTML(html)

    def plot_tabs(self)-> pn.Tabs:
        pn.extension('plotly')

        plot_funcs = {
            'Quantile Returns': plot_quantile_returns(self.quantiles_returns, self.exp_returns[self.benchmark_id]),
            'Rank IC/IR': plot_icir(self.ic, self.ir),
            'Style': plot_style(self.style),
        }
        return pn.Tabs(*[(k, pn.pane.Plotly(v)) for k, v in plot_funcs.items()])

# portofolio analysis
def portofolio_analysis(portofolio, rebalance:str='QR', features:list[str]=None)-> pd.DataFrame:
    pass