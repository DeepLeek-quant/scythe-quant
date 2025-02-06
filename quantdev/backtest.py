from typing import Literal, Union, Callable
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
t_date = db.read_dataset('mkt_calendar', columns=['date'], filters=[('休市原因中文說明(人工建置)','=','')]).rename(columns={'date':'t_date'})

def get_data(item:Union[str, pd.DataFrame], source_dataset:str=None)-> pd.DataFrame:
    if isinstance(item, str):
        raw_data = db.read_dataset(
            dataset=source_dataset or db.find_dataset(item), 
            filter_date='t_date', 
            columns=['t_date', 'stock_id', item],
            )
    elif isinstance(item, pd.DataFrame):
        raw_data = item
    
    return pd.merge_asof(
        t_date[t_date['t_date']<=pd.Timestamp.today() + pd.DateOffset(days=5)], 
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
        .dropna(axis=1, how='all')\
    
    return data #.filter(regex='^\d')

def get_factor(item:Union[str, pd.DataFrame], asc:bool=True, source_dataset:str=None)-> pd.DataFrame:
    if isinstance(item, str):
        return get_factor(get_data(item, source_dataset), asc)
    elif isinstance(item, pd.DataFrame):
        return item.rank(axis=1, pct=True, ascending=asc)

def _get_rebalance_date(rebalance:Literal['MR', 'QR', 'W', 'M', 'Q', 'Y'], end_date:str | pd.Timestamp=None):
    # dates
    start_date = t_date['t_date'].min()
    end_date = pd.Timestamp.today() + pd.DateOffset(days=2) if end_date is None else pd.to_datetime(end_date)
    
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
        r_date, t_date, 
        left_on='r_date', 
        right_on='t_date', 
        direction='forward'
        )['t_date'].to_list()

def _get_portfolio(data:pd.DataFrame, signal_shift:int=0, rebalance:Literal['QR', 'W', 'M', 'Q', 'Y']='QR', hold_period:int=None):
      
    # rebalance
    r_date = _get_rebalance_date(rebalance)
    
    # weight
    portfolio = data[data.index.isin(r_date)]\
        .fillna(False)\
        .astype(int)\
        .apply(lambda x: x / x.sum(), axis=1)
    
    # shift & hold_period
    portfolio = t_date\
        .merge(portfolio, on='t_date', how='left')\
        .set_index('t_date')\
        .shift(signal_shift)\
        .ffill(limit=hold_period)\
        .dropna(how='all')
    return portfolio

def backtesting(
    data:pd.DataFrame, signal_shift:int=0, rebalance:Literal['QR', 'W', 'M', 'Q', 'Y']='QR', hold_period:int=None, 
    start:Union[int, str]=None, end:Union[int, str]=None, 
    benchmark:str='0050'
):
    # return & weight
    return_df = db.read_dataset('stock_return', filter_date='t_date', start=start, end=end)
    benchmark_return = return_df[benchmark]
    
    # get data
    portfolio_df = _get_portfolio(data, signal_shift, rebalance, hold_period)
    
    # backtest
    backtest_df = (return_df * portfolio_df).dropna(axis=0, how='all')

    # equity curve
    daily_return = backtest_df.sum(axis=1)
    c_return = (1 + daily_return).cumprod() - 1
    
    bmk_return = benchmark


    return Strategy(c_return)

class Backtest:
    def __init__(self, start=None, end=None, benchmark:str='0050', db_path:str=None, db_token:str=None):
        self.db = Databank(path=db_path, token=db_token)
        self.return_df = self.db.read_dataset('stock_return', filter_date='t_date', start=start, end=end)
        self.bmk_stock_id = benchmark
        self.universe = None
        self.t_date = self.get_t_date()
        
        # backtest
        self.buy_list = None
        self.portfolio_df = None
        self.backtest_df = None

        # performance
        self.daily_return = None
        self.equity_df = None
        self.bmk_equity_df = (1 + self.return_df[self.bmk_stock_id]).cumprod() - 1
        self.performance_df = None

        # fig
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

    # load data
    def get_t_date(self):
        return self.db.read_dataset('mkt_calendar', 
                                columns=['date'], 
                                filters=[('休市原因中文說明(人工建置)','=','')],
                                start=self.return_df.index.min().strftime('%Y-%m-%d'),
                                end=None,
                                )\
                            .rename(columns={'date':'t_date'})

    def get_data(self, item:Union[str, pd.DataFrame], source:str=None):
        if isinstance(item, str):
            raw_data = self.db.read_dataset(
                dataset=self.db.find_dataset(item) if source is None else source, 
                filter_date='t_date', 
                columns=['t_date', 'stock_id', item],
                )
        elif isinstance(item, pd.DataFrame):
            raw_data = item

        t_date = self.t_date
        data = pd.merge_asof(
            t_date[t_date['t_date']<=pd.Timestamp.today() + pd.DateOffset(days=2)], 
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
            .dropna(axis=1, how='all')\
        
        if self.universe is not None:
            return data[self.universe].filter(regex='^\d')
        else:
            return data.filter(regex='^\d')
    
    def get_factor(self, item:str | pd.DataFrame, asc:bool=True, source:str=None):
        if isinstance(item, str):
            return self.get_data(item=item, source=source).rank(axis=1, pct=True, ascending=asc)
        elif isinstance(item, pd.DataFrame):
            return item.rank(axis=1, pct=True, ascending=asc)

    # trade
    def show_position_info(self, data:pd.DataFrame, rebalance:str) -> pd.DataFrame:
        """顯示持倉資訊

        Attributes:
            rebalance (str): 再平衡方式

        Returns:
            DataFrame: 包含股票代碼、名稱、買入日期、賣出日期、警示狀態、漲跌停、前次成交量、前次成交額、季報公佈日期、月營收公佈日期、市場別、主產業別的資料表
        """
        print('make sure to update databank before get info!')

        # 獲取買入和賣出日期
        buy_date = self.get_rebalance_date(rebalance)[-1]
        sell_date = [d for d in self.get_rebalance_date(rebalance=rebalance, end_date=self.t_date['t_date'].max()) if d > buy_date][0]
        
        # 獲取持倉資訊
        position_info = data.loc[buy_date]
        position_info = position_info[position_info]\
            .reset_index()\
            .rename(columns={'index':'stock_id', position_info.name:'buy_date'})\
            .assign(buy_date=position_info.name)\
            .assign(sell_date=sell_date)

        filters=[('stock_id', 'in', position_info['stock_id'].to_list())]
        
        # 獲取股票中文名稱
        ch_name = self.db.read_dataset('stock_basic_info', columns=['stock_id', '證券名稱'], filters=filters).rename(columns={'證券名稱':'name'})

        # 獲取財報和月營收公佈日期
        quarterly_report_release_date = self.db.read_dataset('fin_data', columns=['stock_id', 'release_date'], filters=filters)\
            .loc[lambda x: x.groupby('stock_id')['release_date'].idxmax()]\
            .rename(columns={'release_date':'季報公佈日期'})
        monthly_rev_release_date = self.db.read_dataset('monthly_rev', columns=['stock_id', 'release_date'], filters=filters)\
            .loc[lambda x: x.groupby('stock_id')['release_date'].idxmax()]\
            .rename(columns={'release_date':'月營收公佈日期'})

        filters.append(('date', '=', self.return_df.index.max()))

        # 獲取市場類型、產業類別、交易狀態
        trading_notes = self.db.read_dataset('stock_trading_notes', columns=['date', 'stock_id', '是否為注意股票', '是否為處置股票', '是否暫停交易', '是否全額交割', '漲跌停註記', '市場別', '主產業別(中)'], filters=filters)\
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
        .rename(columns={'主產業別(中)':'主產業別'})\
        [['stock_id', '警示狀態', '漲跌停', '市場別', '主產業別']]
        
        # 獲取交易量和交易金額
        trading_vol = self.db.read_dataset('stock_trading_data', columns=['stock_id', '成交量(千股)', '成交金額(元)'], filters=filters)\
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
                '市場別', '主產業別',
            ]].set_index('stock_id')
      
    # backtest
    def get_rebalance_date(self, rebalance:Literal['MR', 'QR', 'W', 'M', 'Q', 'Y'], end_date:str | pd.Timestamp=None):
        # dates
        start_date = self.t_date['t_date'].min()
        end_date = pd.Timestamp.today() + pd.DateOffset(days=2) if end_date is None else pd.to_datetime(end_date)
        
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
            r_date,
            self.t_date, 
            left_on='r_date', 
            right_on='t_date', 
            direction='forward'
            )['t_date'].to_list()
    
    def get_portfolio(self, data:pd.DataFrame, signal_shift:int=0, rebalance:Literal['QR', 'W', 'M', 'Q', 'Y']='QR', hold_period:int=None):
      
        # rebalance
        r_date = self.get_rebalance_date(rebalance)
        self.buy_list =  data[data.index.isin(r_date)]
        
        # weight
        portfolio = self.buy_list\
            .fillna(False)\
            .astype(int)\
            .apply(lambda x: x / x.sum(), axis=1)

        # shift & hold_period
        portfolio = pd.DataFrame(self.return_df.index)\
            .merge(portfolio, on='t_date', how='left')\
            .set_index('t_date')\
            .shift(signal_shift)\
            .ffill(limit=hold_period)\
            .dropna(how='all')
        self.portfolio_df = portfolio
        return portfolio
        
    def backtesting(self, data:pd.DataFrame, signal_shift:int=0, rebalance:Literal['QR', 'W', 'M', 'Q', 'Y']='QR', hold_period:int=None, stop_loss:float=None, show=True):

        # return & weight
        return_df = self.return_df
        
        # get data
        portfolio_df = self.get_portfolio(
            data, 
            signal_shift, 
            rebalance, 
            hold_period
        )
        
        # backtest
        backtest_df = (return_df * portfolio_df)\
            .dropna(axis=0, how='all')
        self.backtest_df = backtest_df

        # equity curve
        self.daily_return = self.backtest_df.sum(axis=1)
        self.equity_df = (1 + self.daily_return).cumprod() - 1
        
        # performance display
        self.performance_df = pd.concat([
            pd.DataFrame.from_dict(self.performance_metrics(self.equity_df), orient='index', columns=['strategy']),
            pd.DataFrame.from_dict(self.performance_metrics(self.bmk_equity_df), orient='index', columns=[f'{self.bmk_stock_id}.TT']),
        ], axis=1)

        if show:
            display(self.performance_df)
    

    # analysis
    def performance_metrics(self, equity_curve):
        if not equity_curve.empty:
            cumulative_return = equity_curve.iloc[-1]
            daily_returns = (1 + equity_curve).pct_change().dropna()
            bmk_daily_returns = (1 + self.bmk_equity_df).pct_change().dropna()
            
            annual_return = (1 + cumulative_return) ** (240 / len(equity_curve)) - 1 # annual return
            mdd = ((1 + equity_curve) / (1 + equity_curve).cummax() - 1).min() # mdd
            annual_vol = daily_returns.std() * np.sqrt(240) # annual vol
            calmar_ratio = annual_return / abs(mdd) # calmar Ratio
            sharpe_ratio = annual_return/ annual_vol # sharpe Ratio
            beta = daily_returns.cov(bmk_daily_returns) / bmk_daily_returns.var() # beta
        else:
            annual_return, cumulative_return, mdd, annual_vol, sharpe_ratio, calmar_ratio, beta = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        return {
            'Annual return':f'{annual_return:.2%}',
            'Cumulative return':f'{cumulative_return:.2%}',
            'Max drawdown':f'{mdd:.2%}',
            'Annual volatility':f'{annual_vol:.2%}',
            'Sharpe ratio':f'{sharpe_ratio:.2}',
            'Calmar ratio':f'{calmar_ratio:.2}',
            'beta':f'{beta:.2}',
        }
    
    def liquidity_analysis(self):
      
        # get buy sell dates for each trade
        portfolio = self.portfolio_df
        last_portfolio = portfolio.shift(1).fillna(0)
        buys = (last_portfolio == 0) & (portfolio != 0)
        sells = (last_portfolio != 0) & (portfolio == 0)

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
          self.db.read_dataset(
              'stock_trading_notes', 
              columns=['date', 'stock_id', '是否為注意股票', '是否為處置股票', '是否全額交割', '漲跌停註記'],
              filters=filters
          ),
          self.db.read_dataset(
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
    
    def maemfe_analysis(self):
        buy_list = self.buy_list[self.buy_list].stack().reset_index(name='buy').rename(columns={'level_1':'stock_id'})
        portfolio_returns = self.return_df[self.portfolio_df !=0]

        trades = {}
        n = 0
        for index, row in buy_list.iterrows():
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

    def calculate_relative_return(self, daily_return:pd.DataFrame=None, downtrend_window:int=3):
        d_rtn = self.daily_return if daily_return is None else daily_return
        bmk_d_rtn = self.return_df[self.bmk_stock_id]
        relative_rtn = (1 + d_rtn - bmk_d_rtn).cumprod() - 1

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
    
    def calculate_rolling_beta(self, window=240):
        daily_rtn = self.daily_return
        bmk_daily_rtn = self.return_df[self.bmk_stock_id]

        rolling_cov = daily_rtn.rolling(window).cov(bmk_daily_rtn)
        rolling_var = bmk_daily_rtn.rolling(window).var()
        return (rolling_cov / rolling_var).dropna()

    def calculate_ir(self, window=240):
        daily_rtn = self.daily_return
        bmk_daily_rtn = self.return_df[self.bmk_stock_id]

        excess_return = (1 + daily_rtn - bmk_daily_rtn).rolling(window).apply(np.prod, raw=True) - 1
        tracking_error = excess_return.rolling(window).std()
        return (excess_return / tracking_error).dropna()

    def calculate_ic(self,factor:pd.DataFrame, group:int=10, rebalance ='QR'):
        # factor rank
        r_date = self.get_rebalance_date(rebalance)
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

    def report(self, tab=True):
        # main plots
        ec = self.plot_equity_curve()
        hm = self.plot_return_heatmap()
        lqt = self.plot_liquidity()
        mfe = None #self.plot_maemfe()
        rr = self.plot_relative_return()
        

        # show as tabs
        if tab:
            raw_css = """
            .bk-root .bk-tabs-header.bk-above .bk-headers, .bk-root .bk-tabs-header.bk-below .bk-headers{
            padding: 15px 20px;
            }
            """
            # pn.extension(raw_css=[raw_css])
            pn.extension('plotly', raw_css=[raw_css])
            figs={
                    'Equity Curve':pn.pane.Plotly(ec),
                    'Relative Return':pn.pane.Plotly(rr),
                    'Return Heatmap':pn.pane.Plotly(hm),
                    'Liquidity':pn.pane.Plotly(lqt),
                    'MAE/MFE':pn.pane.Plotly(mfe),
                }

            tabs = pn.Tabs(*figs.items())
            return tabs
        else:
            display(ec, hm, lqt, rr, mfe)

    # plot
    def bold(self, text):
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

    def plot_equity_curve(self, equity_curve:pd.DataFrame=None):
        equity_df = self.equity_df if equity_curve is None else equity_curve
        fig = make_subplots(
            rows=3, cols=1, vertical_spacing=0.05, 
            shared_xaxes=True,
            )

        # equity curve
        fig.append_trace(go.Scatter(
            x=self.bmk_equity_df.index,
            y=self.bmk_equity_df.values,
            showlegend=False,
            name='Benchmark',
            line=dict(color=self.fig_param['colors']['Dark Grey'], width=2),
            ), row=1, col=1)
        fig.append_trace(go.Scatter(
            x=equity_df.index,
            y=equity_df.values,
            name='Strategy',
            line=dict(color=self.fig_param['colors']['Dark Blue'], width=2),
            ), row=1, col=1)

        # drawdown
        drawdown = (equity_df+1 - (equity_df+1).cummax()) / (equity_df+1).cummax()
        bmk_drawdown = (self.bmk_equity_df+1 - (self.bmk_equity_df+1).cummax()) / (self.bmk_equity_df+1).cummax()

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
        p_size = self.portfolio_df.apply(lambda row: np.count_nonzero(row), axis=1) if equity_curve is None else equity_df.apply(lambda row: 0)
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
        fig.add_annotation(text=self.bold(f'Cumulative Return'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
        fig.add_annotation(text=self.bold(f'Drawdown'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
        fig.add_annotation(text=self.bold(f'Portfolio Size'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=3, col=1)

        # fig layout
        fig.update_layout(
            legend=dict(x=0.01, y=0.95, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)'),
            width=self.fig_param['size']['w'],
            height=self.fig_param['size']['h'],
            margin=self.fig_param['margin'],
            template=self.fig_param['template'],
        )
        return fig
    
    def plot_return_heatmap(self):
        # prepare data
        daily_returns = self.daily_return
        monthly_returns = ((1 + daily_returns).resample('ME').prod() - 1)\
            .reset_index()\
            .assign(
                y=lambda x: x['t_date'].dt.year.astype('str'),
                m=lambda x: x['t_date'].dt.month,
                )
        annual_return = (daily_returns + 1).groupby(daily_returns.index.year).prod() - 1
        my_heatmap = monthly_returns.pivot(index='y', columns='m', values=0).iloc[::-1]
        avg_mth_return = monthly_returns.groupby('m')[0].mean()

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
        fig.add_annotation(text=self.bold(f'Annual Return'), x=0, y=1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
        fig.add_annotation(text=self.bold(f'Monthly Return'), x=0, y=1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
        fig.add_annotation(text=self.bold(f'Avg. monthly Return'), x=0, y=1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=3, col=1)

        # fig size, colorscale & bgcolor
        fig.update_layout(
            width=self.fig_param['size']['w'],
            height=self.fig_param['size']['h'],
            margin=self.fig_param['margin'],
            template=self.fig_param['template'],
        )

        return fig

    def plot_liquidity(self, money_threshold:int=500000, volume_threshold:int=50):
        money_threshold=500000
        volume_threshold=50
        df=self.liquidity_analysis()

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
                .groupby('money_threshold')['stock_id']\
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
        fig.add_annotation(text=self.bold(f'Liquidity Heatmap'), x=0, y = 1, yshift=50, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
        fig.add_annotation(text=self.bold(f'Safety Capacity'), align='left', x=0, y = 1, yshift=50, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
        fig.add_annotation(text=f'(Ratio of buy money ≤ 1/{vol_ratio} of volume for all trades)', align='left', x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, font=dict(size=12), row=2, col=1)
        fig.add_annotation(text=self.bold(f'Trading Money at Entry/Exit'), x=0, y = 1, yshift=50, xref="x domain", yref="y domain", showarrow=False, row=2, col=2)

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
    
    def plot_maemfe(self):
        maemfe = self.maemfe_analysis()
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
        fig.add_annotation(text=self.bold(f'Return Distribution<br>win rate: {len(win)/len(maemfe):.2%}'), x=0.5, y = 1, yshift=45, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
        fig.add_annotation(text=self.bold(f'GMFE/ MAE'), x=0.5, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=2)
        fig.add_annotation(text=self.bold(f"GMFE/ MDD<br>missed profit pct:{len(win[win['mdd']*-1 > win['gmfe']])/len(win):.2%}"), x=0.5, y = 1, yshift=45, xref="x domain", yref="y domain", showarrow=False, row=1, col=3)
        fig.add_annotation(text=self.bold(f'MAE distribution'), x=0.5, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
        fig.add_annotation(text=self.bold(f'BMFE distribution'), x=0.5, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=2, col=2)
        fig.add_annotation(text=self.bold(f'GMFE distribution'), x=0.5, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=2, col=3)

        # layout
        fig.update_layout(
            width=self.fig_param['size']['w'],
            height=self.fig_param['size']['h'],
            margin=self.fig_param['margin'],
            template=self.fig_param['template'],
        )
        
        return fig
    
    def plot_relative_return(self, daily_return:pd.DataFrame=None):
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
        relative_rtn, downtrend_rtn = self.calculate_relative_return(daily_return=daily_return)
        bmk_rtn = self.bmk_equity_df

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
        rolling_beta = self.calculate_rolling_beta()

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
        info_ratio = self.calculate_ir()

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
        fig.add_annotation(text=self.bold(f'Relative Return'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
        fig.add_annotation(text=self.bold(f'Beta'), align='left', x=0, y = 1, yshift=40, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
        fig.add_annotation(text=self.bold(f'Information Ratio(IR)'), align='left', x=0, y = 1, yshift=40, xref="x domain", yref="y domain", showarrow=False, row=2, col=2)
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
        f_relative_rtn, downtrend_rtn = self.calculate_relative_return(f_daily_rtn)
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
        ic = self.calculate_ic(factor)

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
    
class Strategy():
    def __init__(self, name:str, c_return:Union[pd.DataFrame, pd.Series], bemchmark:Union[pd.DataFrame, pd.Series]):
        self.name = name
        self.c_return = c_return
        self.bmk_c_return = (1 + return_df[self.bmk_stock_id]).cumprod() - 1
        
        
        
        (1 + return_df[self.bmk_stock_id]).cumprod() - 1

        self.summary = self._calc_summary()
        self.position = pd.DataFrame()
        self.c_return = pd.DataFrame()
    
    def _calc_summary(self, equity_curve):
        if not equity_curve.empty:
            cumulative_return = equity_curve.iloc[-1]
            daily_returns = (1 + equity_curve).pct_change().dropna()
            bmk_daily_returns = (1 + self.bmk_equity_df).pct_change().dropna()
            
            annual_return = (1 + cumulative_return) ** (240 / len(equity_curve)) - 1 # annual return
            mdd = ((1 + equity_curve) / (1 + equity_curve).cummax() - 1).min() # mdd
            annual_vol = daily_returns.std() * np.sqrt(240) # annual vol
            calmar_ratio = annual_return / abs(mdd) # calmar Ratio
            sharpe_ratio = annual_return/ annual_vol # sharpe Ratio
            beta = daily_returns.cov(bmk_daily_returns) / bmk_daily_returns.var() # beta
        else:
            annual_return, cumulative_return, mdd, annual_vol, sharpe_ratio, calmar_ratio, beta = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        return {
            'Annual return':f'{annual_return:.2%}',
            'Cumulative return':f'{cumulative_return:.2%}',
            'Max drawdown':f'{mdd:.2%}',
            'Annual volatility':f'{annual_vol:.2%}',
            'Sharpe ratio':f'{sharpe_ratio:.2}',
            'Calmar ratio':f'{calmar_ratio:.2}',
            'beta':f'{beta:.2}',
        }

    @staticmethod
    def position_info(self, data:pd.DataFrame, rebalance:str) -> pd.DataFrame:
        """顯示持倉資訊

        Attributes:
            rebalance (str): 再平衡方式

        Returns:
            DataFrame: 包含股票代碼、名稱、買入日期、賣出日期、警示狀態、漲跌停、前次成交量、前次成交額、季報公佈日期、月營收公佈日期、市場別、主產業別的資料表
        """
        print('make sure to update databank before get info!')

        # 獲取買入和賣出日期
        rebalance_dates = _get_rebalance_date(rebalance, end_date=self.t_date['t_date'].max())
        buy_date = rebalance_dates[-1]
        sell_date = next(d for d in rebalance_dates if d > buy_date)
        
        # 獲取持倉資訊
        position_info = data.loc[buy_date]
        position_info = position_info[position_info]\
            .reset_index()\
            .rename(columns={'index':'stock_id', position_info.name:'buy_date'})\
            .assign(buy_date=position_info.name)\
            .assign(sell_date=sell_date)

        filters=[('stock_id', 'in', position_info['stock_id'].to_list())]
        
        # 獲取股票中文名稱
        ch_name = self.db.read_dataset('stock_basic_info', columns=['stock_id', '證券名稱'], filters=filters).rename(columns={'證券名稱':'name'})

        # 獲取財報和月營收公佈日期
        quarterly_report_release_date = self.db.read_dataset('fin_data', columns=['stock_id', 'release_date'], filters=filters)\
            .loc[lambda x: x.groupby('stock_id')['release_date'].idxmax()]\
            .rename(columns={'release_date':'季報公佈日期'})
        monthly_rev_release_date = self.db.read_dataset('monthly_rev', columns=['stock_id', 'release_date'], filters=filters)\
            .loc[lambda x: x.groupby('stock_id')['release_date'].idxmax()]\
            .rename(columns={'release_date':'月營收公佈日期'})

        filters.append(('date', '=', self.return_df.index.max()))

        # 獲取市場類型、產業類別、交易狀態
        trading_notes = self.db.read_dataset('stock_trading_notes', columns=['date', 'stock_id', '是否為注意股票', '是否為處置股票', '是否暫停交易', '是否全額交割', '漲跌停註記', '市場別', '主產業別(中)'], filters=filters)\
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
        .rename(columns={'主產業別(中)':'主產業別'})\
        [['stock_id', '警示狀態', '漲跌停', '市場別', '主產業別']]
        
        # 獲取交易量和交易金額
        trading_vol = self.db.read_dataset('stock_trading_data', columns=['stock_id', '成交量(千股)', '成交金額(元)'], filters=filters)\
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
                    '市場別', '主產業別',
                ]].set_index('stock_id')