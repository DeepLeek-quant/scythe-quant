from typing import Union, Literal
from statsmodels.regression.rolling import RollingOLS
from statsmodels.regression.linear_model import OLS

from scipy.optimize import minimize
import statsmodels.api as sm
from scipy import stats
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in reduce")

from .utils import *


# utils
def mdd(rtns, mdd_type: Literal['cumprod','cumsum']='cumprod'):
    if mdd_type == 'cumprod':
        cum_rtns = (1 + rtns).cumprod()
    elif mdd_type == 'cumsum':
        cum_rtns = (rtns).cumsum() + 1
    return ((cum_rtns) / (cum_rtns).cummax() - 1).min()

def cagr(rtns):
    return (1 + ((1 + rtns).cumprod() - 1).iloc[-1]) ** (240 / len(rtns)) - 1

def calmar(rtns):
    return cagr(rtns) / abs(mdd(rtns, 'cumprod'))

def calc_metrics(daily_returns:Union[pd.DataFrame, pd.Series, dict]):
    if isinstance(daily_returns,pd.Series):
        daily_returns = pd.DataFrame({'Strategy':daily_returns})
    elif isinstance(daily_returns, dict):
        daily_returns = pd.DataFrame(daily_returns)
    
    return pd.concat({
        'CAGR(%)': cagr(daily_returns)*100,
        'Sharpe': daily_returns.mean()/daily_returns.std()*240**0.5,
        'Calmar': calmar(daily_returns),
        'MDD(%)': mdd(daily_returns, 'cumprod')*100,
        'Simple MDD(%)': mdd(daily_returns, 'cumsum')*100,
        'Win Rate(%)': daily_returns.apply(lambda x:((x.dropna()>0).sum() / x.dropna().shape[0])*100),
        'Weekly Win(%)': daily_returns.apply(lambda x:((x.dropna().add(1).resample('W').prod().sub(1)>0).sum() / x.dropna().add(1).resample('W').prod().sub(1).dropna().shape[0])*100),
        'Monthly Win(%)': daily_returns.apply(lambda x:((x.dropna().add(1).resample('ME').prod().sub(1)>0).sum() / x.dropna().add(1).resample('ME').prod().sub(1).shape[0])*100),
        'Yearly Win(%)': daily_returns.apply(lambda x:((x.dropna().add(1).resample('YE').prod().sub(1)>0).sum() / x.dropna().add(1).resample('YE').prod().sub(1).shape[0])*100),
        'Win/Loss Ratio': daily_returns.apply(lambda x:(x[x > 0].mean() / abs(x[x < 0].mean()))),
        'Expected Return(bps)': ((1 + daily_returns).prod() ** (1 / len(daily_returns)) - 1)*10000,
        'Sample Size': daily_returns.apply(lambda x:x.dropna().count()),
        't': daily_returns.apply(lambda x: (x.mean() / (x.std() / np.sqrt(x.dropna().count())))),
    }, axis=1).round(2).T

def calc_ic_metrics(ic:Union[pd.DataFrame, pd.Series]=None, factor:pd.DataFrame=None, exp_returns=None, rebalance='QR', rank=True)->pd.DataFrame:
    if ic is None and factor is not None:
        ic = calc_ic(factor=factor, exp_returns=exp_returns, rebalance=rebalance, rank=rank)
    if isinstance(ic, pd.Series):
        ic = pd.DataFrame({'Factor':ic})

    return pd.concat({
            'IC mean': ic.mean().round(4),
            'IC std': ic.std().round(4),
            'IR': ic.apply(lambda x:f'{x.mean()/x.std():.2%}'),
            'positive ratio': ic.apply(lambda x:f'{len(x.dropna()[x.dropna()>0])/len(x.dropna()):.2%}'),
        }, axis=1).T

def calc_ic(factor:pd.DataFrame, exp_returns:pd.DataFrame=None, rebalance:Literal['MR', 'QR', 'W', 'M', 'Q', 'Y']='QR', rank:bool=True)->pd.Series:
    from scytheq.backtest import get_rebalance_date
    from scytheq.data import DatasetsHandler
    exp_returns = DatasetsHandler().read_dataset('exp_returns') if exp_returns is None else exp_returns
    r_date = pd.Index(get_rebalance_date(rebalance)).intersection(exp_returns.index)
    
    factor = factor[factor.index.isin(r_date)].stack().rename('factor')
    returns = resample_returns(exp_returns, rebalance).stack().rename_axis(['t_date', 'stock_id']).rename('return')
    
    return pd.concat([factor, returns], axis=1)\
        .dropna()\
        .groupby('t_date')\
        .apply(lambda x: stats.spearmanr(x['factor'], x['return'])[0] if rank else stats.pearsonr(x['factor'], x['return'])[0])

def calc_ir(factor:pd.DataFrame, exp_returns:pd.DataFrame, rebalance:Literal['MR', 'QR', 'W', 'M', 'Q', 'Y']=None, rank:bool=True, window:int=4)->pd.Series:
    ic = calc_ic(factor, exp_returns, rebalance, rank)
    return ic.rolling(window).mean()/ic.rolling(window).std()

def resample_returns(returns: pd.DataFrame, t: Literal['MR', 'QR', 'W', 'M', 'Q', 'Y', 'D']):
    """重新取樣報酬率時間序列

    Args:
        returns (pd.DataFrame): 報酬率時間序列
        t (Literal['MR', 'QR', 'W', 'M', 'Q', 'Y']): 重新取樣頻率
            - MR: 每月10日後第一個交易日
            - QR: 每季財報公布日後第一個交易日(3/31, 5/15, 8/14, 11/14)
            - W: 每週一
            - M: 每月第一個交易日
            - Q: 每季第一個交易日
            - Y: 每年第一個交易日

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
        - 優先使用回測模組的調倉日期進行重新取樣
        - 若無法使用回測模組，則使用pandas的resample功能
    """
    
    if t == 'D':
        return returns
    try:
        from scytheq.backtest import get_rebalance_date
        dates =  pd.DatetimeIndex(get_rebalance_date(t)+[returns.index.max()])

        def cum_rtns(group):
            all_na_cols = group.isna().all()
            if (len(group.shape)>1) and (group.shape[1]>1):
                result = (group + 1).prod() - 1    
                result[all_na_cols] = np.nan
                return result
            else:
                if all_na_cols:
                    return np.nan
                else:
                    return (group + 1).prod() - 1

        return returns\
            .groupby(dates[dates.searchsorted(returns.index)], group_keys=True)\
            .apply(cum_rtns)\
            .shift(-1)\
            .dropna(how='all')
    except:
        return returns\
            .resample(t)\
            .apply(lambda x: (np.nan if x.isna().all() else (x + 1).prod() - 1))

# plot
def calc_liquidity(portfolio_df:pd.DataFrame)-> pd.DataFrame:
        
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
        ('date', 'in', list(set(list(buy_sells['buy_date'].unique().strftime('%Y-%m-%d')) + list(buy_sells['sell_date'].unique().strftime('%Y-%m-%d'))))),
        ]

    from scytheq.data import DatasetsHandler
    liquidity_status=pd.merge(
        DatasetsHandler().read_dataset(
            'trading_notes', 
            columns=['date', 'stock_id', '是否為注意股票', '是否為處置股票', '是否全額交割', '漲跌停註記'],
            filters=filters
        ),
        DatasetsHandler().read_dataset(
            'trading_data',
            columns=['date', 'stock_id', '成交量(千股)', '成交金額(元)'],
            filters=filters,
        ),
        on=['date', 'stock_id'],
        how='inner',
        ).rename(columns={'成交量(千股)':'成交量', '成交金額(元)':'成交金額'})

    # merge
    return pd.concat([
        pd.merge(buy_sells, liquidity_status, left_on=['stock_id', 'buy_date'], right_on=['stock_id', 'date'], how='left'), 
        pd.merge(buy_sells, liquidity_status, left_on=['stock_id', 'sell_date'], right_on=['stock_id', 'date'], how='left'),
        ])

def calc_maemfe(buy_list:pd.DataFrame, portfolio_df:pd.DataFrame, exp_returns:pd.DataFrame):
    """計算每筆交易的最大不利變動(MAE)和最大有利變動(MFE)

    Args:
        buy_list (pd.DataFrame): 買入訊號矩陣，index為日期，columns為股票代號
        portfolio_df (pd.DataFrame): 持倉矩陣，index為日期，columns為股票代號
        exp_returns (pd.DataFrame): 報酬率矩陣，index為日期，columns為股票代號

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
        maemfe = calc_maemfe(buy_list, portfolio_df, exp_returns)
        ```

    Note:
        - 交易期間的報酬率會以 'YY/MM/DD stock_id' 的格式作為index
        - MAE只計算負值，正值會被設為0
    """
    portfolio_return = (portfolio_df != 0).astype(int) * exp_returns[portfolio_df.columns].loc[portfolio_df.index]
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

    # Handle empty trades case
    if trades.empty:
        return pd.DataFrame(columns=['return', 'gmfe', 'bmfe', 'mae', 'mdd'])
        
    return pd.DataFrame({
        'return': returns,
        'gmfe': gmfe,
        'bmfe': bmfe,
        'mae': mae,
        'mdd': mdd
    }, index=trades.columns)\
    .rename_axis('index', axis=0)

def calc_style(daily_return:Union[pd.DataFrame, pd.Series], window:int=None, total:bool=False):
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

    from scytheq.data import DatasetsHandler
    model = DatasetsHandler().read_dataset('factor_model')
    if isinstance(daily_return, pd.DataFrame):
        daily_return = daily_return.iloc[:,0]

    data = pd.concat([daily_return.rename(0), model], axis=1).dropna()
    
    # rolling
    X = sm.add_constant(data[model.columns])
    y = data.drop(columns=model.columns)
    rolling_reg = RollingOLS(y, X, window=round(len(data)*.2) if window is None else window).fit().params.dropna(how='all')

    if total:
        reg = OLS(y, X)
        return pd.concat([rolling_reg, pd.DataFrame(reg.fit().params, columns=['total']).T])
    else:
        return rolling_reg

def calc_relative_return(daily_return:pd.DataFrame, benchmark_daily_return:pd.DataFrame, downtrend_window:int=3, beta_window:int=240):
    relative_return = (1 + daily_return - benchmark_daily_return).cumprod() - 1

    relative_return_diff = relative_return.diff()
    downtrend_mask = relative_return_diff < 0
    downtrend_mask = pd.DataFrame({'alpha': relative_return, 'downward': downtrend_mask})
    downtrend_mask['group'] = (downtrend_mask['downward'] != downtrend_mask['downward'].shift()).cumsum()
    downtrend_return = pd.Series(
        np.where(
            downtrend_mask['downward'] & downtrend_mask.groupby('group')['downward'].transform('sum').ge(downtrend_window), 
            downtrend_mask['alpha'], 
            np.nan
        ),
        index=relative_return.index
    )

    # beta
    rolling_cov = daily_return.rolling(beta_window).cov(benchmark_daily_return)
    rolling_var = benchmark_daily_return.rolling(beta_window).var()
    beta = (rolling_cov / rolling_var)

    # info ratio
    excess_return = (1 + daily_return - benchmark_daily_return).rolling(beta_window).apply(np.prod, raw=True) - 1
    tracking_error = excess_return.rolling(beta_window).std()
    info_ratio = (excess_return / tracking_error)

    return pd.concat([
        ((1+benchmark_daily_return).cumprod()-1).rename('benchmark_return'), 
        relative_return.rename('relative_return'), 
        downtrend_return.rename('downtrend_return'),
        beta.rename('beta'),
        info_ratio.rename('info_ratio')
    ], axis=1)

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

# combine factors
def calc_reg_returns(factors:Union[dict[str, pd.DataFrame], list[pd.DataFrame]], rebalance:str='MR', method:Literal['OLS', 'GLS']='GLS'):
    from scytheq.data import DatasetsHandler
    returns = resample_returns(DatasetsHandler().read_dataset('exp_returns'), t=rebalance)\
        .stack()\
        .reset_index()\
        .rename(columns={'level_0':'t_date', 0:'R'})\
        .set_index(['t_date', 'stock_id'])
    
    reg_methods = {'GLS': sm.GLS, 'OLS': sm.OLS}
    data = pd.concat([returns, factors], axis=1).dropna()
    return data.groupby('t_date').apply(
        lambda x: pd.Series(reg_methods[method](x['R'], sm.add_constant(x.drop(columns=['R']))).fit().params)
    ).drop(columns=['const'])

def calc_max_ir_weights(ic_df:pd.DataFrame, window:int=4, neg_weight:bool=False) -> pd.DataFrame:
    """
    計算最大化 IC_IR 的權重，可選擇是否允許負權重
    
    Args:
        ic_window: IC值的時間序列
        window: 計算權重的滾動窗口大小
        allow_negative: 是否允許負權重，預設為False
    """
    def optimize_window(window_df, neg_weight=False):
        factor_names = window_df.columns
        bar_ic = window_df.mean().values  # shape: (N,)
        sigma = window_df.cov().values    # shape: (N, N)
        N = len(bar_ic)

        # 定義目標函數（負的 IC_IR，因為 minimize 是求最小）
        def neg_ic_ir(w):
            numerator = np.dot(w, bar_ic)
            denominator = np.sqrt(np.dot(w.T, sigma @ w))
            return -numerator / (denominator if denominator != 0 else 1e6)


        # 初始猜測（均勻分配）
        x0 = np.ones(N) / N

        # 根據allow_negative設定權重限制
        bounds = [(None, None) if neg_weight else (0, None) for _ in range(N)]

        # 可以加入權重總和為 1 的額外約束（不是必要的）
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        result = minimize(neg_ic_ir, x0, bounds=bounds, constraints=constraints)

        if result.success:
            return pd.Series(result.x, index=factor_names)
        else:
            return pd.Series([np.nan] * N, index=factor_names)

    # 對每個時間點計算最優權重
    weights = pd.DataFrame(index=ic_df.index, columns=ic_df.columns)
    for i in range(len(ic_df)):
        if i < window:  # 需要至少window個月的數據
            continue
        weights.iloc[i] = optimize_window(ic_df.iloc[i-window:i], neg_weight=neg_weight)
    
    return weights

def calc_max_ic_weights(ic_df: pd.DataFrame, window:int=4, neg_weight: bool=False) -> pd.DataFrame:
    """
    計算「最大化加權IC」的解析解權重：
    w = V^(-1) * mean(IC)，再做歸一化

    Args:
        ic_df: 每月IC值（index為日期，columns為因子名）
        window: 用來估平均IC和協方差的滾動視窗長度
        neg_weight: 是否允許負權重（預設為False）

    Returns:
        weights_df: 每月對每個因子的最佳權重
    """
    weights = pd.DataFrame(index=ic_df.index, columns=ic_df.columns, dtype=float)

    for i in range(window, len(ic_df)):
        window_df = ic_df.iloc[i - window:i]

        bar_ic = window_df.mean().values           # 平均IC
        V = window_df.cov().values                 # 協方差矩陣
        factor_names = ic_df.columns
        try:
            V_inv = np.linalg.pinv(V)              # 計算 V 的逆（使用 pseudo-inverse 較穩定）
            raw_w = V_inv @ bar_ic                 # 解析解：V^(-1) * IC
        except np.linalg.LinAlgError:
            raw_w = np.ones(len(bar_ic))           # 如果矩陣有問題，就給平均分

        # 負權重不允許時，將負值設為0
        if not neg_weight:
            raw_w = np.maximum(raw_w, 0)

        # 歸一化權重（讓總和為1）
        if raw_w.sum() > 0:
            norm_w = raw_w / raw_w.sum()
        else:
            norm_w = np.ones_like(raw_w) / len(raw_w)  # 萬一全為0，就平均分配

        weights.iloc[i] = pd.Series(norm_w, index=factor_names)

    return weights

def combine_factors(
    factors:list[pd.DataFrame], 
    method:Literal['EW', 'HR', 'HR-decay', 'IC', 'IR', 'MAX_IC', 'MAX_IR']=None, 
    rebalance:str='MR',
    params:dict={'window': 12, 'neg_weight': False},
    universe:pd.DataFrame=None
)-> pd.DataFrame:
    
    def calc_weighted_factors(factors_df:pd.DataFrame, weights:pd.DataFrame)->pd.DataFrame:
        return factors_df.merge(weights.add_suffix('_w'), left_index=True, right_index=True)\
            .assign(factor = lambda x: sum([x[col]*x[f'{col}_w'] for col in weights.columns])/sum([x[f'{col}_w'] for col in weights.columns]))\
            [['factor']]\
            .unstack()\
            .droplevel(0, axis=1)
    
    # methods
    if method == 'EW':
        return mean_dfs(factors)
    elif method.startswith('HR'):
        # Calculate factor weights using historical regression
        from scytheq.backtest import get_rebalance_date
        factors_df = pd.concat([f[f.index.isin(get_rebalance_date(rebalance))].stack() for f in factors], axis=1).dropna()
        weights = calc_reg_returns(factors_df, rebalance)
        
        # Apply rolling mean or exponential weighted mean based on method
        if len(method.split('-')) == 1:
            weights = weights.rolling(params['window']).mean().shift(1)
        elif (len(method.split('-'))==2) and (method.split('-')[1] == 'decay'):
            weights = weights.rolling(params['window']).apply(lambda x: x.ewm(halflife=params['window']/2, adjust=False).mean().iloc[-1])
        
        # Combine factors with weights
        combined_factor = calc_weighted_factors(factors_df, weights)
        return combined_factor
    elif (method.endswith('IC')) or (method.endswith('IR')):
        from scytheq.backtest import get_rebalance_date
        factors_df = pd.concat([f[f.index.isin(get_rebalance_date(rebalance))].stack() for f in factors], axis=1).dropna()
        
        from scytheq.data import DatasetsHandler
        exp_returns = DatasetsHandler().read_dataset('exp_returns')
        ic = pd.concat([calc_ic(f, exp_returns, rebalance=rebalance) for f in factors], axis=1).dropna()
        if method == 'IC':
            weights = ic.rolling(params['window']).mean().shift(1)
        elif method == 'IR':
            weights = (ic.rolling(params['window']).mean()/ic.rolling(params['window']).std()).shift(1)
        elif method == 'MAX_IC':
            weights = calc_max_ic_weights(ic, window=params['window'], neg_weight=params['neg_weight']).shift(1)
        elif method == 'MAX_IR':
            weights = calc_max_ir_weights(ic, window=params['window'], neg_weight=params['neg_weight']).shift(1)
        else:
            raise ValueError(f"Invalid method: {method}")
        combined_factor = calc_weighted_factors(factors_df, weights)
        return combined_factor
