from typing import Literal, Dict, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from fugle_marketdata import RestClient
from functools import wraps
import datetime as dt
import pandas as pd
import numpy as np
import time
import json

import shioaji as sj
import fubon_neo

from .backtest import Report
from .config import config

SJ_LOG_PATH = config.sinopac_config.get('sj_log_path')
quote_api = RestClient(api_key=config.fugle_config.get('fugle_api_key'))

def get_quote(stock_id: str, lot_type: Literal[0, 1, None] = None, raw: bool = False):
    """取得股票報價資訊

       Args:
           stock_id (str): 股票代號
           lot_type (Literal[0, 1, None]): 0 代表整股，1 代表零股，None 代表兩者都要
           raw (bool): True 代表回傳原始資料，False 代表回傳處理過的資料

       Examples:
           取得台積電整股報價：
           ```py
           get_quote(stock_id='2330', lot_type=0)
           ```
           output
           ```
           {0: {
               'close_price': 580.0,
               'bids': {579.0: 1000, 578.0: 500, ...},
               'asks': {581.0: 800, 582.0: 1200, ...}
           }}
           ```
    """
    quote = {}
    if lot_type in [0, None]:
        quote[0] = quote_api.stock.intraday.quote(symbol=stock_id)
    if lot_type in [1, None]: 
        quote[1] = quote_api.stock.intraday.quote(symbol=stock_id, type='oddlot')
    
    if not raw:
        return {
            i: {
                'close_price': quote[i].get('closePrice') or quote[i].get('previousClose'),
                'bids': {item['price']: item['size'] for item in quote[i]['bids']},
                'asks': {item['price']: item['size'] for item in quote[i]['asks']}
            }
            for i in quote.keys()
        }
    return quote

def price_pct_to_tick(price: Union[int, float], pct: float):
        """根據價格和漲跌幅計算調整後的價格

        Args:
            price (int or float): 股票價格
            pct (float): 漲跌幅，例如 0.01 代表漲 1%，-0.01 代表跌 1%

        Examples:
            例如，計算股價 52.3 元漲 1% 後的價格：
            ```py
            price_pct_to_tick(price=52.3, pct=0.01)
            ```
            output
            ```
            52.85
            ```
        52.3 * 1.01 = 52.823，但因為 50~100 元的股票，最小跳動單位是 0.05，
        所以會無條件捨去到最接近的 0.05 倍數，即 52.85
        """
        
        def tick_size(price):
            ticks = [(10, 0.01), (50, 0.05), (100, 0.1), (500, 0.5), (1000, 1.0)]
            for limit, tick in ticks:
                if price <= limit:
                    return tick
            return 5.0
        
        price_pct = float(price) * (pct+1)
        tick = tick_size(price_pct)
        price_tick = round((price_pct // tick) * tick, 2)
        return price_tick

def calc_price(stock_id: str, lot_type: Literal[0, 1], direction: int, price_tolerance: float = 0.01):
        """根據股票代碼、交易類型、交易方向和價格容忍度計算下單價格

        Args:
            stock_id (str): 股票代碼
            lot_type (Literal[0, 1]): 0 表示整股，1 表示零股
            direction (int): 交易方向，1 表示買入，-1 表示賣出
            price_tolerance (float): 價格容忍度，預設為 0.01 (1%)

        Examples:
            例如，計算股票 2330 整股買入時的下單價格：
            ```py
            calc_price(stock_id='2330', lot_type=0, direction=1, price_tolerance=0.01)
            ```
            output
            ```
            505.0
            ```
        若目前市場報價的買賣價差在容忍範圍內，則使用市場報價
        否則根據最新成交價和容忍度計算下單價格。
        """
        quote = get_quote(stock_id=stock_id, lot_type=lot_type)
        direction = np.sign(direction) if abs(direction)>1 else direction

        # check spread
        last_price = quote[lot_type]['close_price']
        try:
            if direction > 0:
                if quote[lot_type]['asks']:
                    quote_price = min([k for k in quote[lot_type]['asks'].keys()])
                else:
                    print(f'no ask for {stock_id}')
                    return last_price
                spread = (quote_price - last_price) / last_price  # to buy, but ask is too high
            else:
                if quote[lot_type]['bids']:
                    quote_price = max([k for k in quote[lot_type]['bids'].keys()])
                else:
                    print(f'no bid for {stock_id}')
                    return last_price
                spread = (last_price - quote_price) / last_price  # to sell, but bid is too low
            # decide order price
            pct = price_tolerance if direction > 0 else -1 * price_tolerance
            return quote_price if spread < price_tolerance else price_pct_to_tick(price=last_price, pct=pct)
        except Exception as e:
            print(e)
            return last_price

def mkt_is_open():
    current_day = dt.datetime.now()
    current_time = current_day.time()
    market_open, market_close = dt.time(9, 0), dt.time(13, 30)
    current_day = current_day.weekday()

    return (current_day<5) and (market_open<=current_time<=market_close)

def gen_position_info(report:Report)-> pd.DataFrame:
    from .backtest import get_rebalance_date
    from .data import DatasetsHandler

    last_buy_date = report.buy_list.index[-1]
    exp_sell_date = pd.Series(get_rebalance_date(report.rebalance, start=last_buy_date, end=(dt.datetime.today()+pd.DateOffset(years=1)))).loc[1]
    last_position = report.portfolio_df\
        .loc[last_buy_date]\
        [report.buy_list.iloc[-1, :].replace({False:np.nan}).dropna().index.tolist()]\
        .to_frame('weight')\
        .assign(
            buy_date=last_buy_date,
            exp_sell_date=exp_sell_date
        )

    if last_position.empty:
        return pd.DataFrame()

    filters=[('stock_id', 'in', last_position.index.to_list())]
        
    # Stock CN name
    ch_name = DatasetsHandler().read_dataset('stock_info', columns=['stock_id', '證券名稱'], filters=filters)\
        .rename(columns={'證券名稱':'name'})\
        .set_index('stock_id')
    # Release date
    q_report_release_date = DatasetsHandler().read_dataset('fin_data', columns=['stock_id', 'release_date'], filters=filters)\
        .loc[lambda x: x.groupby('stock_id')['release_date'].idxmax()]\
        .rename(columns={'release_date':'季報公佈日期'})\
        .set_index('stock_id')
    m_rev_release_date = DatasetsHandler().read_dataset('monthly_rev', columns=['stock_id', 'release_date'], filters=filters)\
        .loc[lambda x: x.groupby('stock_id')['release_date'].idxmax()]\
        .rename(columns={'release_date':'月營收公佈日期'})\
        .set_index('stock_id')

    filters.append(('date', '=', report.exp_returns.index.max()))

    # market status
    trading_notes = DatasetsHandler().read_dataset('trading_notes', columns=['date', 'stock_id', '是否為注意股票', '是否為處置股票', '是否暫停交易', '是否全額交割', '漲跌停註記', '板塊別(中)', '主產業別(中)'], filters=filters)\
    .assign(
        警示狀態=lambda x: x.apply(
            lambda row: ', '.join(filter(None, [
                '注意股' if row['是否為注意股票'] == 'Y' else '',
                '處置股' if row['是否為處置股票'] == 'Y' else '', 
                '暫停交易' if row['是否暫停交易'] == 'Y' else '',
                '全額交割' if row['是否全額交割'] == 'Y' else '',
            ])), axis=1
        ),
        漲跌停=lambda x: x['漲跌停註記']
    )\
    .rename(columns={'主產業別(中)':'主產業別', '板塊別(中)':'板塊別'})\
    [['stock_id', '警示狀態', '漲跌停', '板塊別', '主產業別']]\
    .set_index('stock_id')

    # trading volume and amount
    trading_vol = DatasetsHandler().read_dataset('trading_data', columns=['stock_id', '成交量(千股)', '成交金額(元)'], filters=filters)\
        .rename(columns={'成交量(千股)':'前次成交量_K', '成交金額(元)':'前次成交額_M'})\
        .assign(前次成交額_M=lambda x: (x['前次成交額_M']/1e6).round(2))\
        .set_index('stock_id')

    return pd.concat([
        last_position,
        ch_name,
        q_report_release_date,
        m_rev_release_date,
        trading_notes,
        trading_vol
    ], axis=1)\
    [[
        'name', 'buy_date', 'exp_sell_date', 
        '警示狀態', '漲跌停',
        '前次成交量_K', '前次成交額_M', 
        '季報公佈日期', '月營收公佈日期', 
        '板塊別', '主產業別',
    ]]

class Position:
    """建構 Position

    Args:
        - data (Union[Tuple[str, Union[int, float]], sj.position.StockPosition]): 代表部位的資料，可以是一個元組 (stock_id, money) 或是 shioaji 的 StockPosition 物件。
        - price_tolerance (float): 價格容忍度，預設為 0.02 (2%)。
        - odd (bool): 是否允許零股交易，預設為 True。

    Examples:
        **建立新部位**
        ```python
        # 使用 tuple 建立部位
        position = Position(('2330', 100000))  # 買入台積電 10 萬元

        # 使用 shioaji StockPosition 建立部位
        position = Position(stock_position)  # stock_position 為 shioaji 的部位物件
        ```

        **部位運算**
        ```python
        # 部位相加
        total_position = position1 + position2

        # 部位相減
        diff_position = position1 - position2

        # 部位乘以倍數
        double_position = position * 2
        ```
    """
    def __init__(self, data:Union[Tuple[str, Union[int, float]], sj.position.StockPosition], price_tolerance:float=0.02, odd:bool=True):
        self.inventory_data = {}
        self.data = {
            'stock_id': np.nan, 'buy_money': np.nan,
            'round_price': np.nan, 'round_qty': np.nan,
            'odd_price': np.nan, 'odd_qty': np.nan,
            'last_update': np.nan,
        }
        
        if isinstance(data, tuple):
            self.data.update({
                'stock_id': data[0],
                'buy_money': data[1],
            })
        
        elif isinstance(data, sj.position.StockPosition) or isinstance(data, dict):
            data = data.__dict__ if isinstance(data, sj.position.StockPosition) else data
            self.inventory_data = data
            self.data.update({
                'stock_id': data.get('code', data.get('stock_id')),
                'buy_money': np.floor(data['price'] * (data.get('quantity', 0) - data.get('yd_quantity', 0)) + data.get('last_price', np.nan) * data.get('yd_quantity', 0)),
                'round_qty': (abs(data['quantity']) // 1000) * 1000 * np.sign(data['quantity']),
                'odd_qty': abs(data['quantity']) % 1000 * np.sign(data['quantity']),
            })
        
        else:
            raise ValueError("Invalid data type.")
        
        self.refresh_quotes(price_tolerance=price_tolerance, odd=odd)

    # quote data
    def refresh_quotes(self, price_tolerance:float=0.02, odd:bool=True):
        """更新報價資訊

        更新部位的報價資訊，包括整股與零股的價格。

        Args:
            price_tolerance (float): 價格容忍度，預設為 0.02。
            odd (bool): 是否計算零股價格，預設為 True。

        Examples:
            ```python
            position = Position(('2330', 100000))
            position.refresh_quotes(price_tolerance=0.02, odd=False)
            ```
        """
        if not self.inventory_data:
            stock_id, money = self.data['stock_id'], self.data['buy_money']
            direction = np.sign(money)
            round_price = calc_price(stock_id=stock_id, lot_type=0, direction=direction, price_tolerance=price_tolerance)
            round_quantity = np.floor(abs(money) / (round_price * 1000)) * direction * 1000
            odd_price = calc_price(stock_id=stock_id, lot_type=1, direction=direction, price_tolerance=price_tolerance) if odd else 0
            odd_quantity = int(np.floor(abs(money-round_price*round_quantity) / (odd_price))) * direction if odd else 0
            self.data.update({
                'round_price': round_price,
                'round_qty': round_quantity,
                'odd_price': odd_price,
                'odd_qty': odd_quantity,
                'last_update': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            stock_id = self.data['stock_id']
            direction = np.sign(self.data['buy_money'])
            round_price = calc_price(stock_id=stock_id, lot_type=0, direction=direction, price_tolerance=price_tolerance) if self.data['round_qty'] != 0 else 0
            odd_price = calc_price(stock_id=stock_id, lot_type=1, direction=direction, price_tolerance=price_tolerance) if self.data['odd_qty'] != 0 else 0
            self.data.update({
                'round_price': round_price,
                'odd_price': odd_price,
                'last_update': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }) 

    # arithmetic operations
    def check_same_stock_id(func):
        @wraps(func)
        def wrapper(self, other):
            if self.data['stock_id'] != other.data['stock_id']:
                raise ValueError(f"Stock id mismatch: {self.data['stock_id']} vs {other.data['stock_id']}")
            return func(self, other)
        return wrapper
    
    @check_same_stock_id
    def __add__(self, other:'Position'):
        return Position((self.data['stock_id'], self.data['buy_money'] + other.data['buy_money']))

    @check_same_stock_id
    def __sub__(self, other:'Position'):
        return Position((self.data['stock_id'], self.data['buy_money'] - other.data['buy_money']))
    
    def __mul__(self, other:int):
        """將持倉乘以一個標量值。

        Args:
            other (int): 要乘以的標量值。
        
        Examples:
            ```py
            position = Position(('2330', 100000))
            position_2x = position * 2  # 將持倉規模加倍
            ```
        若持倉來自庫存資料(inventory_data)，則直接將 quantity 和 yd_quantity 乘以標量值
        若持倉來自初始資金(buy_money)，則直接將 buy_money 乘以標量值
        """

        if self.inventory_data:
            self.inventory_data.update({
                'quantity': self.inventory_data['quantity']*other,
                'yd_quantity': self.inventory_data['yd_quantity']*other,
            })
            return Position(self.inventory_data)
        else:
            return Position((self.data['stock_id'], self.data['buy_money'] * other))

    def __rmul__(self, other:int):
        return self.__mul__(other)
    
    def __repr__(self):
        return str(self.data)