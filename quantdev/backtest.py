from typing import Literal, Union, Callable
from abc import ABC, abstractmethod
from scipy import stats
import datetime as dt
import pandas as pd
import numpy as np
import calendar

from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from numerize.numerize import numerize
from IPython.display import display
import plotly.figure_factory as ff
import plotly.graph_objects as go
import panel as pn

from .data import Databank

# databank
db = Databank()
_t_date = db.read_dataset('mkt_calendar', columns=['date'], filters=[('休市原因中文說明(人工建置)','=','')]).rename(columns={'date':'t_date'})

def get_data(item:Union[str, pd.DataFrame], source_dataset:str=None)-> pd.DataFrame:
    """取得股票資料並轉換為時間序列格式

    Args:
        item (Union[str, pd.DataFrame]): 資料欄位名稱或已有的 DataFrame
        source_dataset (str, optional): 資料來源資料集名稱，若為 None 則自動尋找

    Returns:
        pd.DataFrame: 轉換後的時間序列資料，index 為日期，columns 為股票代號

    Examples:
        從資料庫取得收盤價資料:
        ```python
        close = get_data('收盤價')
        ```

        使用已有的 DataFrame:
        ```python
        df = db.read_dataset('stock_trading_data', filter_date='t_date', start='2023-01-01', end='2023-01-02', columns=['t_date', 'stock_id', '收盤價'])
        close = get_data(df)
        ```

    Note:
        - 若輸入字串，會從資料庫尋找對應欄位資料
        - 若輸入 DataFrame，需包含 t_date、stock_id 及目標欄位
        - 輸出資料會自動向前填補最多 240 個交易日的缺失值
        - 會移除全部為空值的股票
    """
    if isinstance(item, str):
        raw_data = db.read_dataset(
            dataset=source_dataset or db.find_dataset(item), 
            filter_date='t_date', 
            columns=['t_date', 'stock_id', item],
            )
    elif isinstance(item, pd.DataFrame):
        raw_data = item
    
    return pd.merge_asof(
        _t_date[_t_date['t_date']<=pd.Timestamp.today() + pd.DateOffset(days=5)], 
        raw_data\
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

def get_factor(item:Union[str, pd.DataFrame], asc:bool=True, source_dataset:str=None)-> pd.DataFrame:
    """將資料轉換為因子值

    Args:
        item (Union[str, pd.DataFrame]): 資料欄位名稱或已有的 DataFrame
        asc (bool): 是否為正向因子，True 代表數值越大分數越高，False 代表數值越小分數越高
        source_dataset (str, optional): 資料來源資料集名稱，若為 None 則自動尋找

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

    Note:
        - 若輸入字串，會先呼叫 get_data 取得資料
        - 輸出的因子值為橫截面標準化後的百分比排名
    """
    if isinstance(item, str):
        return get_factor(get_data(item, source_dataset), asc)
    elif isinstance(item, pd.DataFrame):
        return item.rank(axis=1, pct=True, ascending=asc)

def _get_rebalance_date(rebalance:Literal['MR', 'QR', 'W', 'M', 'Q', 'Y'], end_date:Union[pd.Timestamp, str]=None):
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
    
    if rebalance == 'MR':
        date_list = [
            pd.to_datetime(f'{year}-{month:02d}-10') + pd.DateOffset(days=1)
            for year in range(start_date.year, end_date.year + 1)
            for month in range(1, 13)
            if start_date <= pd.to_datetime(f'{year}-{month:02d}-10') + pd.DateOffset(days=1) <= end_date
        ]
        r_date = pd.DataFrame(date_list, columns=['r_date'])
    elif rebalance == 'QR':
        qr_dates = ['03-31', '05-15', '08-14', '11-14']
        date_list = [
            pd.to_datetime(f'{year}-{md}') + pd.DateOffset(days=1)
            for year in range(start_date.year, end_date.year + 1)
            for md in qr_dates
            if start_date <= pd.to_datetime(f"{year}-{md}") + pd.DateOffset(days=1) <= end_date
        ]
        r_date = pd.DataFrame(date_list, columns=['r_date'])   
    elif rebalance == 'M':
        r_date = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='MS'), columns=['r_date'])
    elif rebalance == 'W':
        r_date = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='W-MON'), columns=['r_date'])
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

def backtesting(
    data:pd.DataFrame, 
    rebalance:Literal['MR', 'QR', 'W', 'M', 'Q', 'Y']='QR', signal_shift:int=0, hold_period:int=None, 
    stop_loss:float=None, stop_profit:float=None,
    start:Union[int, str]=None, end:Union[int, str]=None, 
    universe:pd.DataFrame=None, benchmark:Union[str, list[str]]='0050')-> 'Strategy':
    """
    進行回測並返回回測結果

    Args:
        data (pd.DataFrame): 選股條件矩陣，index為日期，columns為股票代碼
        rebalance (str): 再平衡頻率，可選 'MR'(月營收公布後), 'QR'(財報公布後), 'W'(每週), 'M'(每月), 'Q'(每季), 'Y'(每年)
        signal_shift (int): 訊號延遲天數，用於模擬實際交易延遲
        hold_period (int): 持有期間，若為None則持有至下次再平衡日
        stop_loss (float): 停損點，例如-0.1代表跌幅超過10%時停損
        stop_profit (float): 停利點，例如0.2代表漲幅超過20%時停利
        start (Union[int, str]): 回測起始日期
        end (Union[int, str]): 回測結束日期
        universe (pd.DataFrame): 投資範圍，用於篩選股票池
        benchmark (Union[str, list[str]]): 基準指標，可為單一或多個股票代碼

    Returns:
        Strategy: 回測結果物件，包含:
            - summary: 策略績效摘要，包含總報酬率、年化報酬率、最大回撤等重要指標
            - position_info: 持股資訊，包含每日持股數量、換手率、個股權重等資訊
            - report: 完整的回測報告，包含月報酬分析、風險指標、績效歸因等詳細分析

    Examples:
        進行每季再平衡、訊號延遲1天的回測:
        ```python
        result = backtesting(
            data, 
            rebalance='Q',
            signal_shift=1,
            benchmark='0050'
        )
        ```

    Note:
        - 投資組合權重為等權重配置
        - 若無持有期間限制，則持有至下次再平衡日
        - 停損停利點若未設定則不啟用
    """

    # universe
    data = data[universe] if universe is not None else data
    
    # return & weight
    return_df = db.read_dataset('stock_return', filter_date='t_date', start=start, end=end)
    
    # get data
    buy_list, portfolio_df = _get_portfolio(data, return_df, rebalance, signal_shift, hold_period)
    
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
            'daily_return': 'pd.Series or pd.DataFrame'
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
        last_portfolio = portfolio_df.shift(1).fillna(0)
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
          db.read_dataset(
              'stock_trading_notes', 
              columns=['date', 'stock_id', '是否為注意股票', '是否為處置股票', '是否全額交割', '漲跌停註記'],
              filters=filters
          ),
          db.read_dataset(
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
    
    def _maemfe_analysis(self, portfolio_df:pd.DataFrame=None, return_df:pd.DataFrame=None):
        portfolio_df = portfolio_df or self.portfolio_df
        return_df = return_df or self.return_df

        buy_list = portfolio_df[portfolio_df !=0].stack().reset_index(name='buy').rename(columns={'level_1':'stock_id'})
        portfolio_returns = return_df[portfolio_df !=0]

        trades = {}
        n = 0
        for _, row in buy_list.iterrows():
            stock_id = row['stock_id']
            buy_date = row['t_date']
            sell_date = buy_list[(buy_list['stock_id'] == stock_id) & (buy_list['t_date'] > buy_date)]['t_date'].min()
            if pd.isna(sell_date):
                sell_date = portfolio_returns.index.max()
            period_returns = portfolio_returns.loc[buy_date:sell_date, stock_id].dropna()
            if len(period_returns) !=0:
                trades[f'{n+1}'] = period_returns.dropna().reset_index(drop=True)
                n+=1
        trades = pd.DataFrame(trades)
        
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

    def _calc_relative_return(self, daily_return:pd.DataFrame=None, downtrend_window:int=3):
        daily_rtn = daily_return or self.daily_return
        bmk_daily_rtn = self.return_df[self.benchmark[0]]
        relative_rtn = (1 + daily_rtn - bmk_daily_rtn).cumprod() - 1

        relative_rtn_diff = relative_rtn.diff()
        downtrend_mask = relative_rtn_diff < 0
        downtrend_mask = pd.DataFrame({'alpha': relative_rtn, 'downward': downtrend_mask})
        downtrend_mask['group'] = (downtrend_mask['downward'] != downtrend_mask['downward'].shift()).cumsum()
        downtrend_rtn = np.where(
            downtrend_mask['downward'] & downtrend_mask.groupby('group')['downward'].transform('sum').ge(downtrend_window), 
            downtrend_mask['alpha'], 
            np.nan
        )

        return relative_rtn, downtrend_rtn

    def _calc_rolling_beta(self, window:int=240, daily_rtn:pd.DataFrame=None, bmk_daily_rtn:pd.DataFrame=None):
        daily_rtn = daily_rtn or self.daily_return
        bmk_daily_rtn = bmk_daily_rtn or self.return_df[self.benchmark[0]]

        rolling_cov = daily_rtn.rolling(window).cov(bmk_daily_rtn)
        rolling_var = bmk_daily_rtn.rolling(window).var()
        return (rolling_cov / rolling_var).dropna()

    def _calc_ir(self, window:int=240, daily_rtn:pd.DataFrame=None, bmk_daily_rtn:pd.DataFrame=None):
        daily_rtn = daily_rtn or self.daily_return
        bmk_daily_rtn = bmk_daily_rtn or self.return_df[self.benchmark[0]]

        excess_return = (1 + daily_rtn - bmk_daily_rtn).rolling(window).apply(np.prod, raw=True) - 1
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
            legend=dict(x=0.01, y=0.95, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)'),
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
            legend=dict(x=0.55, y=0.65, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)'),
            coloraxis1_showscale=False,
            coloraxis1=dict(colorscale='Plasma'),
            width=self.fig_param['size']['w'],
            height=self.fig_param['size']['h'],
            margin=self.fig_param['margin'],
            template=self.fig_param['template'],
            )
        return fig
    
    def _plot_maemfe(self, portfolio_df:pd.DataFrame, return_df:pd.DataFrame):
        maemfe = self._maemfe_analysis(portfolio_df, return_df)
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
            name='win',
            legendgroup='win',
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x = lose['gmfe'], 
            y = lose['mae']*-1, 
            mode='markers', 
            marker_color=light_red, 
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
            legendgroup='win',
        ), row=1, col=3)
        fig.add_trace(go.Scatter(
            x = lose['gmfe'], 
            y = lose['mdd']*-1, 
            mode='markers', 
            marker_color=light_red, 
            name='lose',
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
            distplot=ff.create_distplot(
                [x*win[item], x*lose[item]], 
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
        relative_rtn, downtrend_rtn = self._calc_relative_return(daily_return=daily_return)
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
            y=downtrend_rtn,
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
        info_ratio = self._calc_ir()

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

    def report_tabs(self, tabs:list[str]=['Equity Curve', 'Relative Return', 'Return Heatmap', 'Liquidity']):
        # main plots
        pn.extension('plotly')
        
        # Define mapping of tab names to plot functions
        plot_funcs = {
            'Equity Curve': self._plot_equity_curve,
            'Relative Return': self._plot_relative_return,
            'Return Heatmap': self._plot_return_heatmap,
            'Liquidity': self._plot_liquidity,
            'MAE/MFE': self._plot_maemfe
        }

        # Only generate requested plots
        if tabs:
            figs = {
                name: pn.pane.Plotly(plot_funcs[name]())
                for name in tabs 
                if name in plot_funcs
            }
            tab_items = list(figs.items())
        else:
            # If no tabs specified, generate all plots
            figs = {
                name: pn.pane.Plotly(func())
                for name, func in plot_funcs.items()
            }
            tab_items = list(figs.items())

        return pn.Tabs(*tab_items)


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
        # performance
        self.daily_return = backtest_df.sum(axis=1)
        self.c_return = (1 + self.daily_return).cumprod() - 1
        self.summary = self._calc_summary()
        self.report = self.report_tabs()
        self.position_info = self._position_info()

    def _calc_summary(self) -> pd.DataFrame:
        
        # performance
        metrics_dfs = [pd.DataFrame.from_dict(self._calc_metrics(self.c_return), orient='index', columns=['strategy'])]
        
        bmk_c_return = (1 + self.return_df[self.benchmark]).cumprod() - 1
        for bmk in self.benchmark:
            metrics_dfs.append(pd.DataFrame.from_dict(
                self._calc_metrics(bmk_c_return[bmk]), 
                orient='index', 
                columns=[f'{bmk}.TT']
            ))
        return pd.concat(metrics_dfs, axis=1)

    def _calc_metrics(self, c_return) -> dict:
        total_return = c_return.iloc[-1]
        daily_returns = (1 + c_return).pct_change().dropna()
        bmk_daily_returns = (1 + self.return_df[self.benchmark[0]])
        
        annual_return = (1 + total_return) ** (240 / len(c_return)) - 1 # annual return
        mdd = ((1 + c_return) / (1 + c_return).cummax() - 1).min() # mdd
        annual_vol = daily_returns.std() * np.sqrt(240) # annual vol
        calmar_ratio = annual_return / abs(mdd) # calmar Ratio
        sharpe_ratio = annual_return/ annual_vol # sharpe Ratio
        beta = daily_returns.cov(bmk_daily_returns) / bmk_daily_returns.var() # beta

        return {
            'Annual return':f'{annual_return:.2%}',
            'Total return':f'{total_return:.2%}',
            'Max drawdown':f'{mdd:.2%}',
            'Annual volatility':f'{annual_vol:.2%}',
            'Sharpe ratio':f'{sharpe_ratio:.2}',
            'Calmar ratio':f'{calmar_ratio:.2}',
            'beta':f'{beta:.2}',
        }
    
    def _position_info(self) -> pd.DataFrame:
        """顯示持倉資訊

        Args:
            rebalance (str): 再平衡方式

        Returns:
            DataFrame: 包含股票代碼、名稱、買入日期、賣出日期、警示狀態、漲跌停、前次成交量、前次成交額、季報公佈日期、月營收公佈日期、市場別、主產業別的資料表
        """
        print('make sure to update databank before get info!')

        # 獲取買入和賣出日期
        rebalance_dates = _get_rebalance_date(self.rebalance, end_date=_t_date['t_date'].max())
        buy_date = self.buy_list.index[-1]
        sell_date = next(d for d in rebalance_dates if d > buy_date)
        
        # 獲取持倉資訊
        position_info = self.buy_list.iloc[-1, :]
        position_info = position_info[position_info]\
            .reset_index()\
            .rename(columns={'index':'stock_id', position_info.name:'buy_date'})\
            .assign(buy_date=position_info.name)\
            .assign(sell_date=sell_date)

        filters=[('stock_id', 'in', position_info['stock_id'].to_list())]
        
        # 獲取股票中文名稱
        ch_name = db.read_dataset('stock_basic_info', columns=['stock_id', '證券名稱'], filters=filters).rename(columns={'證券名稱':'name'})

        # 獲取財報和月營收公佈日期
        quarterly_report_release_date = db.read_dataset('fin_data', columns=['stock_id', 'release_date'], filters=filters)\
            .loc[lambda x: x.groupby('stock_id')['release_date'].idxmax()]\
            .rename(columns={'release_date':'季報公佈日期'})
        monthly_rev_release_date = db.read_dataset('monthly_rev', columns=['stock_id', 'release_date'], filters=filters)\
            .loc[lambda x: x.groupby('stock_id')['release_date'].idxmax()]\
            .rename(columns={'release_date':'月營收公佈日期'})

        filters.append(('date', '=', self.return_df.index.max()))

        # 獲取市場類型、產業類別、交易狀態
        trading_notes = db.read_dataset('stock_trading_notes', columns=['date', 'stock_id', '是否為注意股票', '是否為處置股票', '是否暫停交易', '是否全額交割', '漲跌停註記', '板塊別(中)', '主產業別(中)'], filters=filters)\
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
        trading_vol = db.read_dataset('stock_trading_data', columns=['stock_id', '成交量(千股)', '成交金額(元)'], filters=filters)\
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

    
    # test factor
    def test_factor(self, factor:pd.DataFrame, rebalance:str='QR', group:int=10):
        results = {}
        for q_start, q_end in [(i/group, (i+1)/group) for i in range(group)]:
            condition = (q_start <= factor) & (factor < q_end)
            self.backtesting(condition, show=False, rebalance=rebalance)
            results[q_start*group] = self.equity_df

        return results

    def plot_test_factor(self, factor:pd.DataFrame, rebalance:str='QR', group:int=10):
        results = self.test_factor(factor, rebalance=rebalance, group=group)

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
        f_relative_rtn, downtrend_rtn = self._calc_relative_return(f_daily_rtn)
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
        ic = self._calc_ic(factor)

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

    # surprise
    def calc_surprise(self,
        data: pd.DataFrame, lag:int=8, std_window:int=8,
        method: Literal['std', 'drift']='std', date_col:str='t_date'
    ):  
        pivot_data = data\
            .dropna()\
            .drop_duplicates(subset=[date_col, 'stock_id'], keep='last')\
            .set_index([date_col, 'stock_id'])\
            .unstack('stock_id')\
            .droplevel(0, axis=1)
        
        seasonal_diff = pivot_data - pivot_data.shift(lag)
        rolling_std = seasonal_diff.rolling(std_window).std()
        
        if method == 'std':
            surprise = seasonal_diff / rolling_std
        elif method == 'drift':
            surprise = (pivot_data - (pivot_data.shift(lag) + seasonal_diff.rolling(std_window).mean())) / rolling_std
            
        return surprise\
            .replace([np.inf, -np.inf], np.nan)\
            .stack()
