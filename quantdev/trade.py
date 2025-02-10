from typing import Literal, Dict, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from fugle_marketdata import RestClient
from functools import wraps
import datetime as dt
import pandas as pd
import numpy as np
import shioaji as sj
import time
import json

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

def calc_price( stock_id: str, lot_type: Literal[0, 1], direction: int, price_tolerance: float = 0.01):
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
    def check_stock_id(func):
        @wraps(func)
        def wrapper(self, other):
            if self.data['stock_id'] != other.data['stock_id']:
                raise ValueError(f"Stock id mismatch: {self.data['stock_id']} vs {other.data['stock_id']}")
            return func(self, other)
        return wrapper
    
    @check_stock_id
    def __add__(self, other:'Position'):
        return Position((self.data['stock_id'], self.data['buy_money'] + other.data['buy_money']))

    @check_stock_id
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

class Portfolio:
    """ 建構 Portfolio

        用於管理多個策略的投資部位，支持部位的加減運算和報價更新。

        Args:
        data (Dict[str, Union[List[Position], Tuple[List[str], int]]]): 投資組合資料，可以是以下兩種格式之一：

        1. {策略名稱: ([股票代號列表], 投資金額)}
        2. {策略名稱: [Position物件列表]}
        
        n_workers (int): 多執行緒數量，預設為20

        Examples:
            **建立新的投資組合**
            ```python
            # 使用股票列表和金額建立
            portfolio = Portfolio({
                'strategy1': [['2330', '2317'], 100000],
                'strategy2': [['2308', '2454'], 200000]
            })

            # 使用Position物件列表建立
            portfolio = Portfolio({
                'strategy1': [Position(('2330', 50000)), Position(('2317', 50000))],
                'strategy2': [Position(('2308', 100000)), Position(('2454', 100000))]
            })
            ```

            **投資組合運算**
            ```python
            # 投資組合相加
            total_portfolio = portfolio1 + portfolio2

            # 投資組合相減
            diff_portfolio = portfolio1 - portfolio2
            ```

            **更新投資組合資料**
            ```python
            # 更新投資組合的部位資料
            portfolio.update(new_data)
            ```

            更新投資組合的資料。new_data 參數必須符合初始化時的資料格式，可以是以下兩種格式之一：
            1. {策略名稱: [(股票代號列表), 投資金額]}
            2. {策略名稱: [Position物件列表]}

            更新時會先清除原有的部位資料，再載入新的部位資料。如果新的部位資料中有相同的策略名稱，
            則會覆蓋原有的策略部位。
        """
    def __init__(self, data:Dict[str, Union[List[Position], Tuple[List[str], int]]], n_workers:int=20):
        self.data = data
        self.n_workers = n_workers

        if not self.data or any(not positions for positions in self.data.values()):
            self.data = {}
        elif all(isinstance(v[0], list) and isinstance(v[1], Union[int, float]) for v in self.data.values()):
            self.data = self.create_portfolio()
            self.clear_zero_position()
        elif isinstance(self.data, dict) and \
            all(isinstance(k, str) and isinstance(v, list) and all(isinstance(pos, Position) for pos in v) for k, v in self.data.items()):
            self.refresh_quotes()
        else:
            raise ValueError("Invalid data type.")

    def refresh_quotes(self, n_workers:int=20):
        positions = [pos for pos_list in self.data.values() for pos in pos_list]
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            list(executor.map(lambda p: p.refresh_quotes(), positions))
        self.clear_zero_position()
    
    def create_portfolio(self, n_workers:int=20):
        portfolio = {
            strategy: [((stock_id, np.floor(money/len(stocks)))) for stock_id in stocks]
            for strategy, (stocks, money) in self.data.items()
        }

        def to_position(t):
            return Position(t)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for strategy, tuples in portfolio.items():
                portfolio[strategy] = list(executor.map(to_position, tuples))
        return portfolio

    def clear_zero_position(self):
        """清除投資組合中的零部位

        移除投資組合中所有數量為零的部位（包括整股和零股），並清除沒有任何部位的策略。
        零部位可能來自於：
        1. 部位相減運算後的結果
        2. 報價更新後的結果

        Examples:
            ```python
            portfolio = Portfolio({
                'strategy1': [Position(('2330', 100000))],
                'strategy2': [Position(('2317', 0))]  # 零部位
            })
            
            portfolio.clear_zero_position()
            # 結果：strategy2 會被移除，因為其部位數量為 0
            # portfolio.data = {
            #     'strategy1': [Position(('2330', 100000))]
            # }
            ```
        """
        
        for strategy in self.data.keys():
            self.data[strategy] = [pos for pos in self.data[strategy] if (pos.data['round_qty'] != 0) or (pos.data['odd_qty'] != 0)]
        self.data = {strategy: positions for strategy, positions in self.data.items() if positions}
    
    def __add__(self, other):
        """將兩個投資組合相加

        將兩個投資組合的部位相加，如果兩個組合有相同的策略和股票，則將其部位相加。
        如果某個策略或股票只存在於其中一個組合，則保留該部位。

        Args:
            other (Portfolio): 要相加的另一個投資組合

        Returns:
            Portfolio: 相加後的新投資組合

        Examples:
            ```python
            portfolio1 = Portfolio({
                'strategy1': [Position(('2330', 50000))]
            })
            portfolio2 = Portfolio({
                'strategy1': [Position(('2330', 50000))],
                'strategy2': [Position(('2317', 100000))]
            })

            # 相加後的結果
            result = portfolio1 + portfolio2
            # result.data = {
            #     'strategy1': [Position(('2330', 100000))],
            #     'strategy2': [Position(('2317', 100000))]
            # }
            ```
        """
        
        if not isinstance(other, Portfolio):
            raise ValueError("Can only add Portfolio objects")
        result = {}
        for strategy in set(self.data) | set(other.data):
            self_pos = {p.data['stock_id']: p for p in self.data.get(strategy, [])}
            other_pos = {p.data['stock_id']: p for p in other.data.get(strategy, [])}
            
            result[strategy] = [
                self_pos[stock_id] + other_pos[stock_id] if (stock_id in self_pos) and (stock_id in other_pos)
                else self_pos[stock_id] if (stock_id in self_pos)
                else (other_pos[stock_id])
                for stock_id in (set(self_pos) | set(other_pos))
            ]
        return Portfolio(result)
    
    def __sub__(self, other):
        """將兩個投資組合相減

        將兩個投資組合的部位相減，如果兩個組合有相同的策略和股票，則將其部位相減。
        如果某個股票只存在於第一個組合，則保留該部位。
        如果某個股票只存在於第二個組合，則將該部位反轉（乘以-1）。

        Args:
            other (Portfolio): 要相減的另一個投資組合

        Returns:
            Portfolio: 相減後的新投資組合

        Examples:
            ```python
            portfolio1 = Portfolio({
                'strategy1': [Position(('2330', 100000))]
            })
            portfolio2 = Portfolio({
                'strategy1': [Position(('2330', 50000))],
                'strategy2': [Position(('2317', 100000))]
            })

            # 相減後的結果
            result = portfolio1 - portfolio2
            # result.data = {
            #     'strategy1': [Position(('2330', 50000))],
            #     'strategy2': [Position(('2317', -100000))]
            # }
            ```
        """
        
        if not isinstance(other, Portfolio):
            raise ValueError("Can only subtract Portfolio objects")
        result = {}
        for strategy in set(self.data) | set(other.data):
            self_pos = {p.data['stock_id']: p for p in self.data.get(strategy, [])}
            other_pos = {p.data['stock_id']: p for p in other.data.get(strategy, [])}
            
            result[strategy] = [
                self_pos[stock_id] - other_pos[stock_id] if stock_id in self_pos and stock_id in other_pos
                else self_pos[stock_id] if stock_id in self_pos
                else other_pos[stock_id]*-1
                for stock_id in set(self_pos) | set(other_pos)
            ]
        return Portfolio(result)

    def update(self, strategy:Union[Dict[str, List[Position]], Dict[str, Tuple[List[str], int]]]):
        """更新投資組合的資料

        更新投資組合的部位資料。strategy 參數必須符合初始化時的資料格式，可以是以下兩種格式之一：
        1. {策略名稱: [(股票代號列表), 投資金額]}
        2. {策略名稱: [Position物件列表]}

        Args:
            strategy (Union[Dict[str, List[Position]], Dict[str, Tuple[List[str], int]]]): 新的部位資料，格式必須符合以下之一：
            
            - {策略名稱: [Position物件列表]}
            - {策略名稱: [(股票代號列表), 投資金額]}

        Examples:
            ```python
            # 使用Position物件列表更新
            portfolio.update({
                'strategy1': [Position(('2330', 100000))],
                'strategy2': [Position(('2317', 50000))]
            })

            # 使用股票列表和金額更新
            portfolio.update({
                'strategy1': [['2330'], 100000],
                'strategy2': [['2317'], 50000]
            })
            ```

        Note:
            更新時會先清除原有的部位資料，再載入新的部位資料。如果新的部位資料中有相同的策略名稱，
            則會覆蓋原有的策略部位。更新後會自動重新整理報價資訊。
        """
        
        if isinstance(strategy, dict) and all(isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], list) and isinstance(v[1], (int, float)) for v in strategy.values()):
            strategy_portfolio = {
                strategy_name: [((stock_id, np.floor(money/len(stocks)))) for stock_id in stocks]
                for strategy_name, (stocks, money) in strategy.items()
            }
            def to_position(t):
                return Position(t)
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                for strategy_name, tuples in strategy_portfolio.items():
                    strategy_portfolio[strategy_name] = list(executor.map(to_position, tuples))
            self.data.update(strategy_portfolio)
        elif isinstance(strategy, dict) and all(isinstance(v, list) and all(isinstance(pos, Position) for pos in v) for v in strategy.values()):
            self.data.update(strategy)
        else:
            raise ValueError("Invalid data type.")
        
        self.refresh_quotes()

    def __repr__(self):
        return str(self.data)
 
class SinoPacAccount(sj.Shioaji):
    def __init__(self, key: dict = None, sim: bool = False):
        """SinoPac Account

        繼承自 shioaji.Shioaji，提供自動登入和憑證啟用功能。

        Args:
            key (dict): API 金鑰和憑證資訊，必須包含以下欄位：
                - api_key: API 金鑰
                - secret_key: API 密鑰
                - ca_path: 憑證路徑
                - ca_passwd: 憑證密碼
                - person_id: 身分證字號
            sim (bool): 是否使用模擬環境，預設為 False

        Examples:
            ```python
            # 建立 API 物件
            api_key = {
                'api_key': 'your_api_key',
                'secret_key': 'your_secret_key',
                'ca_path': '/path/to/ca.pfx',
                'ca_passwd': 'your_ca_password',
                'person_id': 'your_person_id'
            }
            api = SinoPacAccount(key=api_key, sim=False)
            ```

        Available Functions:
            - login: 登入 API
            - logout: 登出 API
            - activate_ca: 啟用憑證
            - list_accounts: 列出帳戶
            - set_default_account: 設定預設帳戶
            - get_account_margin: 取得帳戶保證金
            - get_account_openposition: 取得帳戶未平倉部位
            - get_account_settle_profitloss: 取得帳戶已實現損益
            - get_stock_account_funds: 取得股票帳戶資金
            - get_stock_account_unreal_profitloss: 取得股票帳戶未實現損益
            - get_stock_account_real_profitloss: 取得股票帳戶已實現損益
            - place_order: 下單
            - update_order: 更新委託
            - update_status: 更新狀態
            - list_trades: 列出交易
        """
        
        super().__init__(simulation=sim)
        self.__key = key or config.sinopac_config
        
        # login
        self.login(
            api_key=self.__key['api_key'],
            secret_key=self.__key['secret_key'],
        )
        # CA
        self.activate_ca(
            ca_path=self.__key['ca_path'],
            ca_passwd=self.__key['ca_passwd'],
            person_id=self.__key['person_id'],
        )
    
    @property
    def usage(self) -> pd.DataFrame:
        """Returns current API usage as a DataFrame"""
        return pd.DataFrame([dict(super().usage())])

class Trader:
    def __init__(self, api: SinoPacAccount):
        """ Trader 類別

        提供基本的交易功能，包括查詢庫存、下單、取消委託等。

        Args:
            api (SinoPacAccount): SinoPacAccount 物件，用於執行交易相關操作

        Examples:
            ```python
            # 建立 Trader 物件
            api = SinoPacAccount(key=api_key, sim=False)
            trader = Trader(api=api)

            # 查詢庫存
            inventories = trader.get_inventories()

            # 下單
            trader.place_order(
                stock_id='2330',
                action='buy',
                intraday_odd=0,  # 0:整股, 1:零股
                price=500.0,
                quantity=1000
            )

            # 取消委託
            orders = trader.list_orders()
            for order in orders:
                trader.cancel_order(order)
            ```

        Available Functions:
            - get_inventories: 取得庫存部位
            - list_orders: 列出委託單
            - place_order: 下單
            - cancel_order: 取消委託
            - get_bank_balance: 取得銀行餘額
            - get_settlement: 取得交割資訊
            - update_status: 更新狀態
        """
        
        self.api = api

    def get_inventories(self) -> List[sj.position.StockPosition]:
            return self.api.list_positions(self.api.stock_account, unit=sj.constant.Unit.Share)

    def list_orders(self) -> List[sj.order.Order]:
        return self.api.list_trades()
    
    def place_order(self, 
        stock_id:str, action:Literal['buy', 'sell'], intraday_odd:Literal[0, 1], price:float, quantity:int, 
        price_type:str='limit', order_type:str='rod', trade_type:str='cash', 
    ):
        param = {
            'action':{
                'buy':sj.constant.Action.Buy, 
                'sell':sj.constant.Action.Sell,
            },
            'intraday_odd':{
               0:sj.constant.StockOrderLot.Common, 
               1:sj.constant.StockOrderLot.IntradayOdd, 
            #  2:sj.constant.StockOrderLot.Fixing, 
            #  3:sj.constant.StockOrderLot.Odd, 
            },
            'price_type':{
                'limit':sj.constant.StockPriceType.LMT,
                'mkt':sj.constant.StockPriceType.MKT,
                # 'mkp':sj.constant.StockPriceType.MKP,
            },
            'order_type':{
                'fok':sj.constant.OrderType.FOK,
                'ioc':sj.constant.OrderType.IOC,
                'rod':sj.constant.OrderType.ROD,
            },
            'trade_type':{
                'cash':'Cash',
                'margin':'MarginTrading',
                'short':'ShortSelling',
            },
        }
        contract = self.api.Contracts.Stocks[stock_id]
        order = self.api.Order(
            quantity = quantity/1000 if intraday_odd==0 else quantity,
            price = price,
            action = param['action'][action],
            price_type = param['price_type'][price_type],
            order_type = param['order_type'][order_type],
            order_cond = param['trade_type'][trade_type],
            order_lot = param['intraday_odd'][intraday_odd],
            account=self.api.stock_account,
        )

        print(f'{action} {quantity} of {stock_id} at {price}')
        return self.api.place_order(contract, order)

    def cancel_order(self, order):
        return self.api.cancel_order(order)
    
    def get_bank_balance(self)-> float:
        return self.api.account_balance().acc_balance
    
    def get_settlement(self, as_df:bool=False):
        settlements = self.api.settlements(self.api.stock_account)
        if not as_df:
            return settlements
        else:
            return pd.DataFrame([s.__dict__ for s in settlements]).set_index("T")

    def update_status(self):
        return self.api.update_status(self.api.stock_account)

class PortfolioManager(Trader):
    """投資組合管理執行類別

        繼承自 Trader 類別，提供投資組合管理和執行相關功能，包括同步庫存、更新部位、執行交易等。

        Args:
            api (SinoPacAccount): SinoPacAccount 物件，用於執行交易相關操作
            portfolio (Union[Portfolio, str], optional): 初始投資組合或 JSON 檔案路徑
            json_path (str): 投資組合 JSON 檔案的預設儲存路徑
            actual_portfolio (Portfolio): 實際持有的投資組合
            target_portfolio (Portfolio): 目標投資組合

        Examples:
            ```python
            # 建立 PortfolioManager 物件
            api = SinoPacAccount(key=api_key, sim=False)
            pm = PortfolioManager(api=api)

            # 從 JSON 檔案載入投資組合
            pm = PortfolioManager(api=api, portfolio='path/to/portfolio.json')

            # 使用 Portfolio 物件初始化
            portfolio = Portfolio({
                'strategy1': [Position(('2330', 100000))],
                'strategy2': [Position(('2317', 50000))]
            })
            pm = PortfolioManager(api=api, portfolio=portfolio)

            # 執行投資組合管理
            pm.run_portfolio_manager(
                freq=1,  # 更新頻率（分鐘）
                ignore_bank_balance=False,  # 是否忽略銀行餘額檢查
                ignore_mkt_open=False  # 是否忽略市場開盤時間檢查
            )
            ```

        Available Functions:
            - sync: 同步實際持有部位與目標部位
            - update: 更新目標投資組合
            - to_json: 將實際持有部位儲存為 JSON 檔案
            - from_json: 從 JSON 檔案載入投資組合
            - bank_balance_enough: 檢查銀行餘額是否足夠
            - trade_position: 執行單一部位交易
            - trade_portfolio: 執行整個投資組合交易
            - run_portfolio_manager: 執行投資組合管理循環
        """
    def __init__(self, api: SinoPacAccount, portfolio:Union[Portfolio, str]=None):
        super().__init__(api)
        self.json_path = '/Users/jianrui/Desktop/Research/Quant/portfolio/portfolio.json'
        self.actual_portfolio:Portfolio = None
        if not portfolio:
            self.target_portfolio:Portfolio = self.from_json()
        elif isinstance(portfolio, Portfolio):
            self.target_portfolio:Portfolio = portfolio
        self.sync()
    
    def sync(self):
        """同步實際持有部位與目標部位

        將實際庫存部位與目標投資組合進行同步。此方法會：
        1. 取得目前所有庫存部位
        2. 根據目標投資組合的策略分配庫存部位
        3. 將無法對應到目標策略的部位歸類到 'residual' 中

        同步規則：
        - 優先分配昨日部位(yd_quantity)
        - 再分配今日新增部位(new_quantity)
        - 如果某部位無法完全分配到目標策略中，剩餘部分會被歸類到 'residual' 策略

        Examples:
            ```python
            pm = PortfolioManager(api=api)
            
            # 目標投資組合
            pm.target_portfolio = Portfolio({
                'strategy1': [Position(('2330', 100000))]
            })

            # 同步實際部位
            pm.sync()
            
            # 如果實際持有 2330:1000股，其中 600股是昨日部位，400股是今日新增
            # 且目標策略只需要 800股，則：
            # pm.actual_portfolio.data = {
            #     'strategy1': [Position(2330, quantity=800, yd_quantity=600)],
            #     'residual': [Position(2330, quantity=200, yd_quantity=0)]
            # }
            ```

        Note:
            - 此方法會更新 actual_portfolio
            - 'residual' 策略用於追蹤無法對應到目標策略的部位
            - 同步後的部位數量總和會等於實際庫存數量
        """
        
        target_portfolio = self.target_portfolio
        inventory_positions = self.get_inventories()
        actual_portfolio = {strategy: [] for strategy in target_portfolio.data.keys()}
        actual_portfolio['residual'] = []

        # Process each inventory position
        for inv_pos in inventory_positions:
            stock_id = inv_pos['code']
            total_quantity = inv_pos['quantity']
            yd_quantity = inv_pos['yd_quantity']
            new_quantity = total_quantity - yd_quantity
            price = inv_pos['price']
            last_price = inv_pos['last_price']
            
            # Track remaining quantities
            remaining_yd = yd_quantity
            remaining_new = new_quantity
            
            # Find all strategies that want this stock
            for strategy, positions in target_portfolio.data.items():
                for target_pos in positions:
                    if target_pos.data['stock_id'] == stock_id:
                        target_quantity = target_pos.data['round_qty'] + target_pos.data['odd_qty']
                        
                        # First distribute yd_quantity
                        filled_yd = min(remaining_yd, target_quantity)
                        target_quantity -= filled_yd
                        remaining_yd -= filled_yd
                        
                        # Then distribute new quantity if needed
                        filled_new = min(remaining_new, target_quantity)
                        remaining_new -= filled_new
                        
                        total_filled = filled_yd + filled_new
                        if total_filled > 0:
                            actual_portfolio[strategy].append(Position(
                                data = {
                                    'code': stock_id,
                                    'quantity': total_filled,
                                    'yd_quantity': filled_yd,
                                    'price': price,
                                    'last_price': last_price,
                                }
                            ))                    
                        if remaining_yd == 0 and remaining_new == 0:
                            break
                if remaining_yd == 0 and remaining_new == 0:
                    break
            
            remaining_total = remaining_yd + remaining_new
            if remaining_total > 0:
                actual_portfolio['residual'].append(Position(
                    data = {
                        'code': stock_id,
                        'quantity': remaining_total,
                        'yd_quantity': remaining_yd,
                        'price': price,
                        'last_price': last_price,
                    }
                ))
        self.actual_portfolio = Portfolio(actual_portfolio)
    
    def update(self, strategy:Union[Dict[str, List[Position]], Dict[str, List[Tuple[str, int]]]]):
        """更新目標投資組合

        更新目標投資組合的部位資料，並同步實際持有部位。

        Args:
            strategy (Union[Dict[str, List[Position]], Dict[str, List[Tuple[str, int]]]]): 新的目標投資組合資料，格式為：
                {策略名稱: [Position物件列表]}

        Examples:
            ```python
            # 更新特定策略的部位
            pm.update({
                'strategy1': [Position(('2330', 100000))],
                'strategy2': [Position(('2317', 50000))]
            })

            # 清空特定策略的部位
            pm.update({
                'strategy1': []
            })

            # 新增策略
            pm.update({
                'new_strategy': [Position(('2330', 100000))]
            })
            ```

        Note:
            - 此方法會先更新 target_portfolio
            - 然後自動呼叫 sync() 同步實際部位
            - 如果更新的策略已存在，則會覆蓋原有的部位
            - 如果是新策略，則會新增到目標投資組合中
        """
        
        self.target_portfolio.update(strategy)
        self.sync()
    
    # save/load the portfolio
    def to_json(self, path: str=None):
        """將實際持有部位儲存為 JSON 檔案

        將實際持有部位的資料儲存到指定的 JSON 檔案中。

        Args:
            path (str, optional): JSON 檔案的路徑，預設為 self.json_path
        """
        
        self.sync()
        path = path or self.json_path
        portfolio = self.actual_portfolio
        
        if 'residual' in portfolio.data:
            del portfolio.data['residual']
        portfolio_dict = {
            'portfolio': {strategy: {pos.data['stock_id']: int(pos.data['round_qty']+pos.data['odd_qty']) for pos in positions} for strategy, positions in portfolio.data.items()},
            'update_time': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(f'{path}', 'w') as json_file:
            print(f'Inventory Portfolio saved to {path}')
            json.dump(portfolio_dict, json_file)
    
    def from_json(self, path: str=None):
        """從 JSON 檔案載入投資組合

        從指定的 JSON 檔案中載入投資組合資料，並同步實際持有部位。

        Args:
            path (str, optional): JSON 檔案的路徑，預設為 self.json_path

        Returns:
            Portfolio: 載入的投資組合
        """
        
        path = path or self.json_path
        with open(f'{path}', 'r') as json_file:
            print(f'Portfolio loaded from {path}')
            portfolio_dict = json.load(json_file)['portfolio']
        inventory = self.get_inventories()
        portfolio = {strategy: [] for strategy in portfolio_dict.keys()}

        # Process each inventory position
        for inv_pos in inventory:
            stock_id = inv_pos['code']
            total_quantity = inv_pos['quantity']
            yd_quantity = inv_pos['yd_quantity']
            new_quantity = total_quantity - yd_quantity
            price = inv_pos['price']
            last_price = inv_pos['last_price']
            
            # Track remaining quantities
            remaining_yd = yd_quantity
            remaining_new = new_quantity
            
            # Find all strategies that want this stock
            for strategy, target_quantity in portfolio_dict.items():
                if stock_id in target_quantity:
                    target_qty = target_quantity[stock_id]
                    
                    # First distribute yd_quantity
                    filled_yd = min(remaining_yd, target_qty)
                    target_qty -= filled_yd
                    remaining_yd -= filled_yd
                    
                    # Then distribute new quantity if needed
                    filled_new = min(remaining_new, target_qty)
                    remaining_new -= filled_new
                    
                    total_filled = filled_yd + filled_new
                    if total_filled > 0:
                        portfolio[strategy].append(Position(
                            data = {
                                'code': stock_id,
                                'quantity': total_filled,
                                'yd_quantity': filled_yd,
                                'price': price,
                                'last_price': last_price,
                            }
                        ))
                    if remaining_yd == 0 and remaining_new == 0:
                        break
        return Portfolio(portfolio)
    
    # check
    def bank_balance_enough(self) -> bool:
        """檢查銀行餘額是否足夠

        檢查銀行餘額是否足夠，以確保可以執行交易。計算方式如下：

        1. 取得目前銀行餘額 (bank_balance)
        2. 計算交割金額總和 (settlement)：所有待交割金額的總和
        3. 計算目標投資組合所需金額 (portfolio_money)：
           - 計算目標投資組合與實際投資組合的差額 (target - actual)
           - 加總所有部位的買入金額
        4. 檢查可用金額是否足夠：
           bank_balance + settlement - portfolio_money >= 0

        Returns:
            bool: 如果銀行餘額足夠，則返回 True；否則返回 False
        """
        
        try:
            bank_balance = self.get_bank_balance()
            settlement = sum([int(i['amount']) for i in self.get_settlement()])
            portfolio_money = sum([int(pos.data['buy_money']) for pos_list in (self.target_portfolio - self.actual_portfolio).data.values() for pos in pos_list])
            
            return (bank_balance + settlement - portfolio_money) >= 0
        except:
            return True
    
    # trade
    def trade_position(self, position:Position):
        """執行單一部位交易

        執行單一部位交易，包括買入和賣出。

        Args:
            position (Position): 要交易的部位
        """
        
        position.refresh_quotes()
        stock_id = position.data['stock_id']
        action = 'buy' if (position.data['buy_money']>0) else 'sell'
        if position.data['round_qty']!=0:
            self.place_order(
                stock_id=stock_id,
                action=action,
                intraday_odd=0,
                price=position.data['round_price'],
                quantity=abs(position.data['round_qty']),
            )
        
        if position.data['odd_qty']!=0:
            self.place_order(
                stock_id=stock_id,
                action=action,
                intraday_odd=1,
                price=position.data['odd_price'],
                quantity=abs(position.data['odd_qty']),
            )
    
    def trade_portfolio(self, portfolio:Portfolio):
        """執行整個投資組合交易

        執行整個投資組合交易，包括買入和賣出。

        Args:
            portfolio (Portfolio): 要交易的投資組合
        """
        
        for positions in portfolio.data.values():
            for position in positions:
                self.trade_position(position)
    
    def run_portfolio_manager(self, freq=1, ignore_bank_balance:bool=False, ignore_mkt_open:bool=False):
        """執行投資組合管理循環

        持續執行投資組合管理，包括同步部位、檢查條件、取消舊委託、執行新委託等。
        此方法會在以下情況停止執行：
        1. 市場已收盤（除非 ignore_mkt_open=True）
        2. 銀行餘額不足（除非 ignore_bank_balance=True）
        3. 投資組合已完全執行（actual_portfolio 等於 target_portfolio）

        Args:
            freq (int): 更新頻率，單位為分鐘，預設為 1 分鐘
            ignore_bank_balance (bool): 是否忽略銀行餘額檢查，預設為 False
                - True: 即使銀行餘額不足也繼續執行
                - False: 當銀行餘額不足時停止執行
            ignore_mkt_open (bool): 是否忽略市場開盤時間檢查，預設為 False
                - True: 不論市場是否開盤都執行
                - False: 只在市場開盤時執行（9:00-13:30）

        執行流程：
        1. 同步實際部位（sync）
        2. 檢查執行條件：
            - 市場是否開盤
            - 銀行餘額是否足夠
        3. 取消所有未完成的委託
        4. 計算目標部位和實際部位的差異
        5. 執行差異部位的交易
        6. 等待指定時間後重複執行

        Examples:
            ```python
            # 基本使用
            pm = PortfolioManager(api=api, portfolio=portfolio)
            pm.run_portfolio_manager()  # 每分鐘更新一次

            # 自定義更新頻率
            pm.run_portfolio_manager(freq=5)  # 每5分鐘更新一次

            # 忽略檢查條件
            pm.run_portfolio_manager(
                ignore_bank_balance=True,  # 忽略銀行餘額檢查
                ignore_mkt_open=True  # 忽略市場開盤時間檢查
            )
            ```

        Note:
            - 當程式停止執行時，會自動儲存最後的部位狀態
            - 建議在正式環境中不要設置 ignore_bank_balance=True
        """
        while True:
            self.sync()
            
            # machine time
            print(f"Current time: {dt.datetime.now().time().strftime('%H:%M:%S')}")

            if (not ignore_mkt_open) and (not mkt_is_open()):
                print("Market is closed.")
                self.to_json()
                break

            if (not ignore_bank_balance) and (not self.bank_balance_enough()):
                print("Bank balance is not enough for position!")
                self.to_json()
                break

            # cancel all orders
            for order in self.list_orders():
                self.update_status()
                self.cancel_order(order)

            # excecute portfolio
            difference = self.target_portfolio - self.actual_portfolio
            if difference.data:
                self.trade_portfolio(difference)
            else:
                print("Portfolio is filled.")
                self.to_json()
                break

            time.sleep(60*freq)