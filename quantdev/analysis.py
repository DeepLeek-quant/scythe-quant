from typing import Union, Literal, Tuple
from numerize.numerize import numerize
from statsmodels.regression.rolling import RollingOLS
from statsmodels.regression.linear_model import OLS
from linearmodels import FamaMacBeth
import statsmodels.api as sm
from scipy import stats
import pandas as pd
import numpy as np

from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# utils
def calc_maemfe(buy_list:pd.DataFrame, portfolio_df:pd.DataFrame, return_df:pd.DataFrame):        
        portfolio_return = (portfolio_df != 0).astype(int) * return_df[portfolio_df.columns].loc[portfolio_df.index]
        period_starts = portfolio_df.index.intersection(buy_list.index)[:-1]
        period_ends = period_starts[1:].map(lambda x: portfolio_df.loc[:x].index[-1])
        trades = pd.DataFrame()
        for start_date, end_date in zip(period_starts, period_ends):
            start_idx = portfolio_return.index.get_indexer([start_date])[0]
            end_idx = portfolio_return.index.get_indexer([end_date])[0] + 1
            r = portfolio_return.iloc[start_idx:end_idx, :]
            trades = pd.concat([trades, r\
                .loc[:, (r != 0).any() & ~r.isna().any()]\
                .rename(columns=lambda x: r.index[0].strftime('%Y%m%d') + '_' + str(x))\
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

def calc_info_ratio(daily_rtn:pd.DataFrame, bmk_daily_rtn:pd.DataFrame, window:int=240):
    excess_return = (1 + daily_rtn - bmk_daily_rtn).rolling(window).apply(np.prod, raw=True) - 1
    tracking_error = excess_return.rolling(window).std()
    return (excess_return / tracking_error).dropna()

def calc_info_coef(factor:pd.DataFrame, return_df:pd.DataFrame, group:int=10, rebalance:Literal['MR', 'QR', 'W', 'M', 'Q', 'Y']=None):
    # factor rank
    from quantdev.backtest import _get_rebalance_date
    r_date = _get_rebalance_date(rebalance)
    f_rank = (pd.qcut(factor[factor.index.isin(r_date)].stack(), q=group, labels=False, duplicates='drop') + 1)\
    .reset_index()\
    .rename(columns={'level_1':'stock_id', 0:'f_rank'})    

    # return rank
    factor_r =  return_df\
        .groupby(pd.cut(return_df.index, bins=r_date))\
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
def calc_factor_longshort_return(factor:Union[pd.DataFrame, str], asc:bool=True, rebalance:str='QR', group:int=10):
    from quantdev.backtest import get_factor, quick_backtesting
    factor = get_factor(item=factor, asc=asc)
    long = quick_backtesting(factor>=(1-1/group), rebalance=rebalance)
    short = quick_backtesting(factor<(1/group), rebalance=rebalance)

    return (long-short).dropna()

def calc_factor_quantiles_return(factor:pd.DataFrame, asc:bool=True, rebalance:str='QR', group:int=10):
    from quantdev.backtest import get_factor, quick_backtesting
    results = {}
    factor = get_factor(item=factor, asc=asc)
    for q_start, q_end in [(i/group, (i+1)/group) for i in range(group)]:
        condition = (q_start <= factor) & (factor < q_end)
        results[q_start*group] = quick_backtesting(condition, show=False, rebalance=rebalance)

    return results

def resample_returns(data: pd.DataFrame, t: Literal['YE', 'QE', 'ME', 'W-FRI']):
    return data.resample(t).apply(lambda x: (np.nan if x.isna().all() else (x + 1).prod() - 1))

# style analysis
def calc_portfolio_style(portfolio_daily_rtn:Union[pd.DataFrame, pd.Series], window:int=None, total:bool=False):
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

# factor analysis
def plot_factor(factor:pd.DataFrame, asc:bool=True, rebalance:str='QR', group:int=10):
    results = calc_factor_quantiles_return(factor, asc=asc, rebalance=rebalance, group=group)

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
    base_key = 0
    # for key in sorted(results.keys()):
    #     if (results[key]).isna().count() / len(pd.DataFrame(results)) >0.5:
    #         base_key = key
    #         break
    f_daily_rtn = (1 + results[max(results.keys())]).pct_change() - (1 + results[base_key]).pct_change()
    f_rtn  = (1+f_daily_rtn).cumprod()-1
    f_relative_rtn, downtrend_rtn = calc_relative_return(f_daily_rtn, bmk_daily_rtn)
    bmk_rtn = self.bmk_equity_df

    # relative return
    strategy_trace = go.Scatter(
        x=f_relative_rtn.index,
        y=f_relative_rtn.values,
        name='Relative Return (lhs)',
        line=dict(color=self.fig_param['colors']['Dark Blue'], width=2),
        mode='lines',
        yaxis='y1',
    )

    # downtrend
    downtrend_trace = go.Scatter(
        x=f_relative_rtn.index,
        y=downtrend_rtn,
        name='Downtrend (lhs)',
        line=dict(color=self.fig_param['colors']['Bright Red'], width=2),
        mode='lines',
        yaxis='y1',
    )

    # factor return
    factor_return_trace = go.Scatter(
        x=f_rtn.index,
        y=f_rtn.values,
        name='Factor Return (rhs)',
        line=dict(color=self.fig_param['colors']['White'], width=2),
        mode='lines',
        yaxis='y2',
    )


    benchmark_trace = go.Scatter(
        x=bmk_rtn.index,
        y=bmk_rtn.values,
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
    num_lines = len(results)
    color_scale = sample_colorscale('PuBu', [n / num_lines for n in range(num_lines)])
    for (key, value), color in zip(reversed(list(results.items())), color_scale):
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
    from quantdev.data import Databank
    _db = Databank()
    return_df = _db.read_dataset('stock_return', filter_date='t_date')
    ic = calc_info_coef(factor, return_df, group, rebalance)

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
    fig.add_annotation(text=self.bold(f'Factor Relative Returns'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
    fig.add_annotation(text=self.bold(f'Factor Return by Quantiles'), x=0, y = 1, yshift=50, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
    fig.add_annotation(text=self.bold(f'Information Coefficient(IC)'), x=0, y = 1, yshift=50, xref="x domain", yref="y domain", showarrow=False, row=2, col=2)

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
