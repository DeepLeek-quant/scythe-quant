from typing import Union, Literal
from statsmodels.regression.rolling import RollingOLS
from statsmodels.regression.linear_model import OLS
from numerize.numerize import numerize
import statsmodels.api as sm
from scipy import stats
import pandas as pd
import numpy as np

# utils
def calc_metrics(daily_returns:Union[pd.Series, pd.DataFrame], benchmark_daily_returns:Union[pd.Series, pd.DataFrame]=None) -> dict:
    metrics = daily_returns\
    .agg({
        'total_return': lambda x: ((1 + x).cumprod() - 1).iloc[-1],
        'annual_return': lambda x: (1 + ((1 + x).cumprod() - 1).iloc[-1]) ** (240 / len(x)) - 1,
        'mdd': lambda x: ((1 + (1 + x).cumprod() - 1) / (1 + (1 + x).cumprod() - 1).cummax() - 1).min(),
        'annual_vol': lambda x: x.std() * np.sqrt(240)
    })
    metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['mdd'])
    metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['annual_vol']

    if benchmark_daily_returns is not None:
        metrics['beta'] = daily_returns.cov(benchmark_daily_returns) / benchmark_daily_returns.var() # beta
    else:
        metrics['beta'] = '-'

    return pd.Series({
        name: f'{metrics[name]:.2%}' if name in ['total_return', 'annual_return', 'mdd', 'annual_vol'] 
        else f'{metrics[name]:.2}'
        for name in metrics.index
    })

def calc_maemfe(buy_list:pd.DataFrame, portfolio_df:pd.DataFrame, return_df:pd.DataFrame):
    """計算每筆交易的最大不利變動(MAE)和最大有利變動(MFE)

    Args:
        buy_list (pd.DataFrame): 買入訊號矩陣，index為日期，columns為股票代號
        portfolio_df (pd.DataFrame): 持倉矩陣，index為日期，columns為股票代號
        return_df (pd.DataFrame): 報酬率矩陣，index為日期，columns為股票代號

    Returns:
        pd.DataFrame: 包含以下欄位的DataFrame:
            
            - return: 每筆交易的報酬率
            - gmfe: 每筆交易的全域最大有利變動(Global Maximum Favorable Excursion)
            - bmfe: 每筆交易在最大不利變動前的最大有利變動(Before MAE Maximum Favorable Excursion)
            - mae: 每筆交易的最大不利變動(Maximum Adverse Excursion)
            - mdd: 每筆交易的最大回撤(Maximum Drawdown)

    Examples:
        計算每筆交易的MAE/MFE:
        ```python
        maemfe = calc_maemfe(buy_list, portfolio_df, return_df)
        ```

    Note:
        - 交易期間的報酬率會以 'YY/MM/DD stock_id' 的格式作為index
        - MAE只計算負值，正值會被設為0
    """
    portfolio_return = (portfolio_df != 0).astype(int) * return_df[portfolio_df.columns].loc[portfolio_df.index]
    period_starts = portfolio_df.index.intersection(buy_list.index)[:-1]
    period_ends = period_starts[1:].map(lambda x: portfolio_df.loc[:x].index[-1])
    trades = pd.DataFrame()
    for start_date, end_date in zip(period_starts, period_ends):
        start_idx = portfolio_return.index.get_indexer([start_date])[0]
        end_idx = portfolio_return.index.get_indexer([end_date])[0]
        r = portfolio_return.iloc[start_idx:end_idx, :]
        trades = pd.concat([trades, r\
            .loc[:, (r != 0).any() & ~r.isna().any()]\
            .rename(columns=lambda x: f"{r.index[0].strftime('%y/%m/%d')} {x}")\
            .reset_index(drop=True)], axis=1)
    
    c_returns = (1 + trades).cumprod(axis=0) - 1
    returns = c_returns.apply(lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan, axis=0)
    gmfe = c_returns.max(axis=0)
    bmfe = c_returns.apply(lambda x: x.iloc[:x.idxmin()+1].max(), axis=0)
    mae = c_returns.min(axis=0).where(lambda x: x <= 0, 0)
    mdd = ((1 + c_returns) / (1 + c_returns).cummax(axis=0) - 1).min(axis=0)

    return pd.DataFrame({
        'return': returns,
        'gmfe': gmfe,
        'bmfe': bmfe,
        'mae': mae,
        'mdd': mdd
    }, index=trades.columns)

def calc_liquidity_metrics(portfolio_df:pd.DataFrame=None, money_threshold:int=500000, volume_threshold:int=50):
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
    df = pd.concat([
        pd.merge(buy_sells, liquidity_status, left_on=['stock_id', 'buy_date'], right_on=['stock_id', 'date'], how='left'), 
        pd.merge(buy_sells, liquidity_status, left_on=['stock_id', 'sell_date'], right_on=['stock_id', 'date'], how='left'),
        ])
        
    # Calculate warning stats
    warning_stats = pd.DataFrame({      
        f'money < {numerize(money_threshold)}': [(df['成交金額(元)'] < money_threshold).mean()],
        f'volume < {numerize(volume_threshold)}': [(df['成交量(千股)'] < volume_threshold).mean()],
        '全額交割': [(df['是否全額交割']=='Y').mean()],
        '注意股票': [(df['是否為注意股票']=='Y').mean()],
        '處置股票': [(df['是否為處置股票']=='Y').mean()],
        'buy limit up': [((df['date']==df['sell_date'])&(df['漲跌停註記'] == '-')).mean()],
        'sell limit down': [((df['date']==df['buy_date'])&(df['漲跌停註記'] == '+')).mean()],
    }, index=[0])

    # Calculate safety capacity
    principles = [500000, 1000000, 5000000, 10000000, 50000000, 100000000]
    buys_lqd = df[df['buy_date']==df['date']]
    vol_ratio = 10
    capacity = pd.DataFrame({
        'principle': [numerize(p) for p in principles],
        'safety_capacity': [
            1 - (buys_lqd.assign(
                capacity=lambda x: x['成交金額(元)'] / vol_ratio - (principle / x.groupby('buy_date')['stock_id'].transform('nunique')),
                spill=lambda x: np.where(x['capacity'] < 0, x['capacity'], 0)
            )
            .groupby('buy_date')['spill'].sum() * -1 / principle).mean()
            for principle in principles
        ]
    })

    # Calculate volume distribution
    def calc_volume_dist(df):
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

    volume_dist = pd.concat([
        calc_volume_dist(df[df['buy_date']==df['date']]).rename('buy'),
        calc_volume_dist(df[df['sell_date']==df['date']]).rename('sell')
    ], axis=1)

    return {
        'warning_stats': warning_stats,
        'capacity': capacity,
        'volume_dist': volume_dist,
        'vol_ratio': vol_ratio
    }

def calc_info_ratio(daily_rtn:pd.DataFrame, benchmark_daily_returns:pd.DataFrame, window:int=240):
    excess_return = (1 + daily_rtn - benchmark_daily_returns).rolling(window).apply(np.prod, raw=True) - 1
    tracking_error = excess_return.rolling(window).std()
    return (excess_return / tracking_error).dropna()

def calc_info_coef(factor:pd.DataFrame, return_df:pd.DataFrame, group:int=10, rebalance:Literal['MR', 'QR', 'W', 'M', 'Q', 'Y']=None):
    # factor rank
    from quantdev.backtest import _get_rebalance_date
    r_date = _get_rebalance_date(rebalance)
    r_date = pd.Index(r_date).intersection(return_df.index)
    f_rank = (pd.qcut(factor[factor.index.isin(r_date)].stack(), q=group, labels=False, duplicates='drop') + 1)\
    .reset_index()\
    .rename(columns={'level_1':'stock_id', 0:'f_rank'})    

    # return rank
    factor_r =  return_df\
        .groupby(pd.cut(return_df.index, bins=r_date), observed=True)\
        .apply(lambda x: (1 + x).prod() - 1).dropna()\
        .stack()\
        .reset_index()\
        .rename(columns={'level_0':'t_date', 'level_1':'stock_id', 0: 'return'})\
        .assign(t_date=lambda df: df['t_date'].apply(lambda x: x.left).astype('datetime64[ns]'))
    
    # ic
    factor_return_rank = pd.merge(f_rank, factor_r, on=['t_date', 'stock_id'], how='inner')\
        .dropna()\
        .groupby(['t_date', 'f_rank'])['return']\
        .mean()\
        .reset_index()\
        .assign(r_rank = lambda df: df.groupby('t_date')['return'].rank())
    return factor_return_rank.groupby('t_date', group_keys=False)\
        .apply(lambda group: stats.spearmanr(group['f_rank'], group['r_rank'])[0], include_groups=False)

def calc_relative_return(daily_return:pd.DataFrame, bmk_daily_return:pd.DataFrame, downtrend_window:int=3):
    relative_return = (1 + daily_return - bmk_daily_return).cumprod() - 1

    relative_return_diff = relative_return.diff()
    downtrend_mask = relative_return_diff < 0
    downtrend_mask = pd.DataFrame({'alpha': relative_return, 'downward': downtrend_mask})
    downtrend_mask['group'] = (downtrend_mask['downward'] != downtrend_mask['downward'].shift()).cumsum()
    downtrend_return = np.where(
        downtrend_mask['downward'] & downtrend_mask.groupby('group')['downward'].transform('sum').ge(downtrend_window), 
        downtrend_mask['alpha'], 
        np.nan
    )

    return relative_return, downtrend_return

# factor analysis
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
    
    from quantdev.backtest import simple_backtesting
    return_df = kwargs.get('return_df')
    if return_df is None:
        from quantdev.data import Databank
        db = Databank()
        return_df = db.read_dataset('stock_return', filter_date='t_date')
    # factor = get_factor(item=factor, asc=asc)
    long = simple_backtesting(factor>=(1-1/group), rebalance=rebalance, return_df=return_df)
    short = simple_backtesting(factor<(1/group), rebalance=rebalance, return_df=return_df)

    return (long-short).dropna()

def calc_factor_quantiles_return(factor:pd.DataFrame, rebalance:str='QR', group:int=10, **kwargs):
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
    from quantdev.backtest import simple_backtesting
    results = {}
    return_df = kwargs.get('return_df')
    if return_df is None:
        from quantdev.data import Databank
        db = Databank()
        return_df = db.read_dataset('stock_return', filter_date='t_date')
    
    for q_start, q_end in [(i/group, (i+1)/group) for i in range(group)]:
        condition = (q_start <= factor) & (factor < q_end)
        results[q_start*group] = simple_backtesting(condition, rebalance=rebalance, return_df=return_df)

    return results

def calc_factor_quantiles_return_thread(factor:Union[pd.DataFrame, str], asc:bool=True, rebalance:str='QR', group:int=10, **kwargs):
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
    from quantdev.backtest import get_factor, simple_backtesting
    import concurrent.futures
    import threading
    
    results = {}
    results_lock = threading.Lock()
    factor = get_factor(item=factor, asc=asc)
    return_df = kwargs.get('return_df')
    if return_df is None:
        from quantdev.data import Databank
        db = Databank()
        return_df = db.read_dataset('stock_return', filter_date='t_date')

    def calc_quantile(q_start, q_end):
        condition = (q_start <= factor) & (factor < q_end)
        result = simple_backtesting(condition, rebalance=rebalance, return_df=return_df)
        with results_lock:
            results[q_start*group] = result

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for q_start, q_end in [(i/group, (i+1)/group) for i in range(group)]:
            futures.append(executor.submit(calc_quantile, q_start, q_end))
        
        concurrent.futures.wait(futures)

    return results

def resample_returns(data: pd.DataFrame, t: Literal['YE', 'QE', 'ME', 'W-FRI']):
    """重新取樣報酬率時間序列

    Args:
        data (pd.DataFrame): 報酬率時間序列
        t (Literal['YE', 'QE', 'ME', 'W-FRI']): 重新取樣頻率
            - YE: 年底
            - QE: 季底
            - ME: 月底
            - W-FRI: 週五

    Returns:
        pd.DataFrame: 重新取樣後的報酬率時間序列

    Examples:
        將日報酬率轉換為月報酬率:
        ```python
        monthly_returns = resample_returns(daily_returns, 'ME')
        ```

    Note:
        - 報酬率會以複利方式計算
        - 若某期間內所有值皆為NA，則該期間的報酬率為NA
    """
    return data.resample(t).apply(lambda x: (np.nan if x.isna().all() else (x + 1).prod() - 1))

# portfolio analysis
def calc_portfolio_style(portfolio_daily_rtn:Union[pd.DataFrame, pd.Series], window:int=None, total:bool=False):
    """計算投資組合的風格分析

    Args:
        portfolio_daily_rtn (Union[pd.DataFrame, pd.Series]): 投資組合的日報酬率
        window (int, optional): 滾動視窗大小. Defaults to None.
        total (bool, optional): 是否計算全期間的風格分析. Defaults to False.

    Returns:
        pd.DataFrame: 投資組合的風格分析結果
            - 若total=False，回傳滾動視窗的風格分析結果
            - 若total=True，回傳滾動視窗及全期間的風格分析結果

    Examples:
        計算投資組合的滾動風格分析:
        ```python
        style = calc_portfolio_style(portfolio_returns, window=60)
        ```

    Note:
        - 使用因子模型進行風格分析
        - 若window=None，則使用全期間20%的資料作為滾動視窗
        - 因子包含市場、規模、價值等常見因子
    """

    from quantdev.data import Databank
    db = Databank()
    model = db.read_dataset('factor_model').drop(['MTM6m', 'CMA'], axis=1)
    data = pd.concat([portfolio_daily_rtn, model], axis=1).dropna()
    
    # rolling
    X = sm.add_constant(data[model.columns])
    y = data.drop(columns=model.columns)

    
    rolling_reg = RollingOLS(y, X, window=round(len(data)*.2) if window is None else window).fit().params.dropna(how='all')

    if total:
        reg = OLS(y, X)
        return pd.concat([rolling_reg, pd.DataFrame(reg.fit().params, columns=['total']).T])
    else:
        return rolling_reg

def calc_brinson_model(portfolio_df:pd.DataFrame, return_df:pd.DataFrame, benchmark:list[str]):
    from quantdev.data import Databank
    db = Databank()

    portfolio_df = portfolio_df[portfolio_df!=0].stack()

    benchmark_sector = db.read_dataset('stock_sector',
                    columns=['stock_id', 't_date', 'sector'],
                    filters=[['is_tw50', '==', 'Y']])

    portfolio_sector = db.read_dataset('stock_sector',
                    columns=['stock_id', 't_date', 'sector'],
                    filters=[['stock_id', 'in', portfolio_df.index.get_level_values(1).unique()]])

    benckmark_weight = db.read_dataset('stock_trading_data',
                    columns=['t_date', 'stock_id', '個股市值(元)'],
                    filters=[['stock_id', 'in', benchmark_sector['stock_id'].unique()]])\
                    .rename(columns={'個股市值(元)':'weight'})

    portfolio_weight = portfolio_df\
        .reset_index()\
        .rename(columns={0:'weight', 'level_1':'stock_id'})

    rtn = return_df.stack().reset_index().rename(columns={0:'rtn'})

    # Process benchmark data
    benchmark_data = (benchmark_sector
        .merge(benckmark_weight, on=['t_date', 'stock_id'])
        .merge(rtn, on=['t_date', 'stock_id'])
        .dropna()
        .assign(w_rtn=lambda x: x['weight'] * x['rtn'])
        .groupby(['t_date', 'sector'], as_index=False)
        .agg({'weight': 'sum', 'w_rtn': 'sum'})
    ).assign(
        benchmark_w=lambda x: x['weight'] / x.groupby('t_date')['weight'].transform('sum'),
        benchmark_r=lambda x: x['w_rtn'] / x['weight']
    ).set_index(['t_date', 'sector'])[['benchmark_w', 'benchmark_r']]

    # Process portfolio data similarly
    portfolio_data = (portfolio_sector
        .merge(portfolio_weight, on=['t_date', 'stock_id'])
        .merge(rtn, on=['t_date', 'stock_id'])
        .dropna()
        .assign(w_rtn=lambda x: x['weight'] * x['rtn'])
        .groupby(['t_date', 'sector'], as_index=False)
        .agg({'weight': 'sum', 'w_rtn': 'sum'})
    ).assign(
        portfolio_w=lambda x: x['weight'] / x.groupby('t_date')['weight'].transform('sum'),
        portfolio_r=lambda x: x['w_rtn'] / x['weight']
    ).set_index(['t_date', 'sector'])[['portfolio_w', 'portfolio_r']]

    # Combine the data
    data = pd.concat([benchmark_data, portfolio_data], axis=1)\
        .fillna(0)\
        .join(
            pd.DataFrame(return_df[benchmark[0]].rename('benchmark_R')),
            on='t_date',
            how='left'
        )
    
    # BF
    brinson_model = data.assign(
        allocation_effect=lambda x: (x['portfolio_w'] - x['benchmark_w']) * (x['benchmark_r'] - x['benchmark_R']),
        selection_effect=lambda x: x['portfolio_w'] * (x['portfolio_r'] - x['benchmark_r']),
    )\
    [['allocation_effect', 'selection_effect']]

    return brinson_model

# efficiency frontier
def create_random_portfolios(returns:pd.DataFrame, num_portfolios:int=None):
    def _calc_portfolio_return(weights, mean_returns, cov_matrix):
        returns = np.sum(mean_returns*weights ) *240
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(240)
        return std, returns
    
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = returns.shape[1]
    
    results_list = []
    num_portfolios = 2000*num_assets if num_portfolios is None else num_portfolios
    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_std_dev, portfolio_return = _calc_portfolio_return(weights, mean_returns, cov_matrix)
        results_list.append({
            'std_dev': portfolio_std_dev,
            'return': portfolio_return,
            'sharpe': portfolio_return / portfolio_std_dev,
            'weights': np.round(weights, 4).tolist()
        })
    
    results = pd.DataFrame(results_list).round({'std_dev': 5, 'return': 5, 'sharpe': 5})
    return results