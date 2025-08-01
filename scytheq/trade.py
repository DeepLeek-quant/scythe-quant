from typing import Literal, List, Tuple, Union, Any, Dict
from functools import wraps
import datetime as dt
import pandas as pd
import numpy as np
import time
import json

from concurrent.futures import ThreadPoolExecutor, as_completed


from fugle_marketdata import RestClient
from fubon_neo.sdk import FubonSDK, Order, Condition, ConditionDayTrade, ConditionOrder
from fubon_neo.constant import *
import shioaji as sj

from .backtest import Report
from .config import config
from .utils import *
SJ_LOG_PATH = config.trade_api_config.get('sinopac_api').get('sj_log_path')
quote_api = RestClient(api_key=config.trade_api_config.get('fugle_marketdata').get('fugle_api_key'))

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

def gen_position_info(report:Report, portfolio:pd.DataFrame=None)-> pd.DataFrame:
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

    filters.append(('date', '=', report.exp_returns.dropna(how='all').index.max()))

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
        'name', 'weight', 'buy_date', 'exp_sell_date', 
        '警示狀態', '漲跌停',
        '前次成交量_K', '前次成交額_M', 
        '季報公佈日期', '月營收公佈日期', 
        '板塊別', '主產業別',
    ]]

# portfolio log
def write_portfolio(portfolio_dict:dict, path:str):
    with open(f'{path}', 'w') as f:
        json.dump(portfolio_dict, f)
    print(f'portfolio.json saved to {path}')

def read_portfolio(path:str)-> dict:
    try:
        with open(f'{path}', 'r') as f:
            portfolio = json.load(f)
        print(f'portfolio.json loaded from {path}')
        return portfolio
    except FileNotFoundError:
        empty_portfolio = {
            'portfolio': {},
            'update_time': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        write_portfolio(empty_portfolio, path)
        return read_portfolio(path)
    
# generate order
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

    def __init__(self):
        pass

    # quote data
    def from_quantity(self, stock_id:str, quantity:int, price_tolerance:float=0.02, odd:bool=True):
        direction = np.sign(quantity)
        self.stock_id = stock_id
        self.round_price = calc_price(stock_id=stock_id, lot_type=0, direction=direction, price_tolerance=price_tolerance)
        self.odd_price = calc_price(stock_id=stock_id, lot_type=1, direction=direction, price_tolerance=price_tolerance) if odd else 0
        self.round_qty = np.floor(abs(quantity) / 1000) * 1000 * direction
        self.odd_qty = abs(quantity) % 1000 * direction
        # self.last_update = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self
    
    def from_money(self, stock_id:str, money:int, price_tolerance:float=0.02, odd:bool=True):
        direction = np.sign(money)
        self.stock_id = stock_id
        self.round_price = calc_price(stock_id=stock_id, lot_type=0, direction=direction, price_tolerance=price_tolerance)
        self.round_qty = np.floor(abs(money) / (self.round_price * 1000)) * direction * 1000
        self.odd_price = calc_price(stock_id=stock_id, lot_type=1, direction=direction, price_tolerance=price_tolerance) if odd else 0
        self.odd_qty = int(np.floor(abs(money-self.round_price*self.round_qty) / (self.odd_price))) * direction if odd else 0
        # self.last_update = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self
    
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
        
        direction = np.sign(self.round_qty + self.odd_qty)
        self.round_price = calc_price(stock_id=self.stock_id, lot_type=0, direction=direction, price_tolerance=price_tolerance) if self.round_qty != 0 else 0
        self.odd_price = calc_price(stock_id=self.stock_id, lot_type=1, direction=direction, price_tolerance=price_tolerance) if self.odd_qty != 0 else 0
        # self.last_update = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def __repr__(self):
        return str(self.__dict__)

    @property
    def empty(self):
        return (self.round_qty+self.odd_qty) == 0


class SinoPacAccount(sj.Shioaji):
    def __init__(self, key: dict = None, sim: bool = False):
        """SinoPac Account

        繼承自 shioaji.Shioaji，提供自動登入和憑證啟用功能。

        Args:
            key (dict): API 金鑰和憑證資訊，必須包含以下欄位：
                - api_key: API 金鑰
                - secret_key: API 密鑰
                - ca_path: 憑證路徑
                - ca_password: 憑證密碼
                - person_id: 身分證字號
            sim (bool): 是否使用模擬環境，預設為 False

        Examples:
            ```python
            # 建立 API 物件
            api_key = {
                'api_key': 'your_api_key',
                'secret_key': 'your_secret_key',
                'ca_path': '/path/to/ca.pfx',
                'ca_password': 'your_ca_password',
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
            - _update_status: 更新狀態
            - list_trades: 列出交易
        """
        
        super().__init__(simulation=sim)
        key = key or config.trade_api_config['sinopac_api']
        
        # login
        self.login(
            api_key=key['api_key'],
            secret_key=key['secret_key'],
        )
        # CA
        self.activate_ca(
            ca_path=key['ca_path'],
            ca_passwd=key['ca_password'],
            person_id=key['person_id'],
        )
    
    @property
    def usage(self) -> pd.DataFrame:
        """Returns current API usage as a DataFrame"""
        return pd.DataFrame([dict(super().usage())])


class SinoPacTrader:
    def __init__(self, api:SinoPacAccount, portfolio_path:str=None, price_tolerance:float=0.02, odd:bool=True):
        self.api = api
        self.price_tolerance = price_tolerance
        self.odd = odd

        # portfolio
        self.portfolio_path = portfolio_path or config.portfolio_path_config.get('sinopac_portfolio_path')
        self.portfolio = read_portfolio(self.portfolio_path)
        self.check_portfolio_match_inventories(self.portfolio)
    
    def _update_status(self):
        return self.api.update_status(self.api.stock_account)

    # basic
    def list_inventories(self) -> List[sj.position.StockPosition]:
        return self.api.list_positions(self.api.stock_account, unit=sj.constant.Unit.Share)

    def list_orders(self) -> List[sj.order.Order]:
        self._update_status()
        return self.api.list_trades()

    def place_order(self, 
        stock_id:str, action:Literal['buy', 'sell'], intraday_odd:Literal[0, 1], 
        price:float, quantity:int, 
        price_type:Literal['limit', 'mkt']='limit', 
        order_type:Literal['fok', 'ioc', 'rod']='rod', 
        trade_type:Literal['cash', 'margin', 'short']='cash', 
        note:str=None,
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
            custom_field = note,
            account=self.api.stock_account,
        )

        print((f'{action} {quantity} of {stock_id} at {price}') + (f' for strategy {note}' if note else ''))
        return self.api.place_order(contract, order)

    def cancel_order(self, order:sj.order.Order):
        return self.api.cancel_order(order)
   
    def get_bank_balance(self)-> float:
        return self.api.account_balance().acc_balance
    
    def get_settlement(self, as_df:bool=False):
        settlements = self.api.settlements(self.api.stock_account)
        if not as_df:
            return settlements
        else:
            return pd.DataFrame([s.__dict__ for s in settlements]).set_index("T")

    # portfolio
    def check_portfolio_match_inventories(self, portfolio:dict):
        agg_json_portfolio = {}
        for p in portfolio['portfolio'].values():
            for stock_id, quantity in p.items():
                if stock_id not in agg_json_portfolio:
                    agg_json_portfolio[stock_id] = quantity
                else:
                    agg_json_portfolio[stock_id] += quantity

        agg_inv_portfolio = {i.code:i.quantity * (-1 if i.direction == 'Sell' else 1)  for i in self.list_inventories()}
        diff_portfolio = {}
        for k, v in agg_inv_portfolio.items():
            if k not in agg_json_portfolio:
                diff_portfolio[k] = v
            else:
                diff_portfolio[k] = v - agg_json_portfolio[k]
        diff_portfolio = {k:v for k, v in diff_portfolio.items() if v != 0}
        if diff_portfolio=={}:
            print('portfolio.json matches inventories.')
        else:
            print(f'portfolio.json does not match inventories: {diff_portfolio}!')

    def update_portfolio(self):
        current_portfolio = self.portfolio['portfolio'].copy()
        last_update_time = dt.datetime.strptime(self.portfolio['update_time'], '%Y-%m-%d %H:%M:%S')
        traded_orders = {}
        for order in self.list_orders():
            if (order.order.custom_field, order.contract.code) not in traded_orders:
                traded_orders[(order.order.custom_field, order.contract.code)] = 0
            traded_orders[(order.order.custom_field, order.contract.code)] += \
                (1000 if order.order.order_lot=='Common' else 1 if order.contract.unit=='IntradayOdd' else 1) * \
                (-1 if order.order.action=='Sell' else 1) * \
                sum([deal.quantity for deal in order.status.deals if dt.datetime.fromtimestamp(deal.ts) > last_update_time])

        for k, v in traded_orders.items():
            if not current_portfolio.get(k[0]):
                current_portfolio[k[0]]={}
            if not current_portfolio[k[0]].get(k[1]):
                current_portfolio[k[0]][k[1]] = v
            else:
                current_portfolio[k[0]][k[1]] += v
        if self.list_orders() != []:
            self.portfolio['portfolio'] = current_portfolio
            self.portfolio['update_time'] = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # trade
    def settlement_is_enough(self, diff_portfolio:Dict[str, List[Position]]) -> bool:
        try:
            bank_balance = self.get_bank_balance()
            settlement = sum([int(i['amount']) for i in self.get_settlement()])
            portfolio_money = sum([sum([p.round_price*p.round_qty+p.odd_price*p.odd_qty for p in positions]) for positions in diff_portfolio.values()])
            
            return (bank_balance + settlement - portfolio_money) >= 0
        except Exception as e:
            print(f'error when checking settlement: {e}')
            return True
        
    def check_portfolio_difference(self,new_portfolio:Dict[str, List[Dict[str, Union[int, float]]]], by:Literal['money', 'quantity']='money'):
        difference_portfolio = {}
        if by == 'money':
            for strategy, positions in new_portfolio.items():
                difference_portfolio[strategy] = []
                current_strategy = self.portfolio['portfolio'].get(strategy, {})
                
                def process_money(stock_id, money):
                    inv_money = current_strategy.get(stock_id, 0) * next((inv.price for inv in self.list_inventories() if inv.code == stock_id), 0)
                    position = Position().from_money(stock_id, money - inv_money)
                    if not position.empty:
                        return position
                    return None
                
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(process_money, stock_id, money) for stock_id, money in positions.items()]
                    for future in as_completed(futures):
                        position = future.result()
                        if position:
                            difference_portfolio[strategy].append(position)   
            difference_portfolio = {k:[i for i in v if abs(i.round_price*i.round_qty+i.odd_price*i.odd_qty) > 500] for k, v in difference_portfolio.items()}
        elif by == 'quantity':
            for strategy, positions in new_portfolio.items():
                difference_portfolio[strategy] = []
                current_strategy = self.portfolio['portfolio'].get(strategy, {})
                
                def process_quantity(stock_id, quantity):
                    inv_quantity = current_strategy.get(stock_id, 0)
                    position = Position().from_quantity(stock_id, quantity - inv_quantity)
                    if not position.empty:
                        return position
                    return None
                
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(process_quantity, stock_id, quantity) for stock_id, quantity in positions.items()]
                    for future in as_completed(futures):
                        position = future.result()
                        if position:
                            difference_portfolio[strategy].append(position)
        
        return difference_portfolio
    
    def trade_position(self, position:Position, strategy_name:str=None):
        action = 'buy' if ((position.round_qty+position.odd_qty)>0) else 'sell'
        # place round order
        if position.round_qty!=0:
            self.place_order(
                stock_id=position.stock_id,
                action=action,
                intraday_odd=0,
                price=position.round_price,
                quantity=abs(position.round_qty),
                note=strategy_name,
            )
        
        # place odd order
        if position.odd_qty!=0:
            self.place_order(
                stock_id=position.stock_id,
                action=action,
                intraday_odd=1,
                price=position.odd_price,
                quantity=abs(position.odd_qty),
                note=strategy_name,
            )

    def rebalance(self, new_portfolio:Dict[str, List[Dict[str, Union[int, float]]]], by:Literal['money', 'quantity']='money', freq:int=1, ignore_mkt_open:bool=False, ignore_settlement:bool=False):
        while True:
            # machine time
            print(f"Current time: {dt.datetime.now().time().strftime('%H:%M:%S')}")
            if (not ignore_mkt_open) and (not mkt_is_open()):
                print("Market is closed.")
                break
            
            # cancel all orders
            for order in self.list_orders():
                if order['status']['status'] not in ['Filled', 'Cancelled', 'Failed']:
                    self.cancel_order(order)

            # update portfolio
            self.update_portfolio()

            # check difference
            diff_portfolio = self.check_portfolio_difference(new_portfolio, by=by)

            if (not ignore_settlement) and (not self.settlement_is_enough(diff_portfolio)):
                print("Settlement is not enough for position!")
                break
            
            if not all(v==[] for v in diff_portfolio.values()):
                # trade
                for strategy_name, positions in diff_portfolio.items():
                    for position in positions:
                        self.trade_position(position, strategy_name)
            else:
                print('Portfolio is all filled.')
                write_portfolio(self.portfolio, self.portfolio_path)
                break
            
            time.sleep(60*freq)


class FubonAccount:
    def __init__(self, key:dict=None):
        key = key or config.trade_api_config['fubon_api']
        self.sdk = FubonSDK()
        self.accounts = self.sdk.login(key['id_number'], key['password'], key['ca_path'], key['ca_password'])
        self.account = self.accounts.data[0]

        
    def logout(self):
        return self.sdk.logout()


class FubonTrader:
    def __init__(self, account:FubonAccount, portfolio_path:str=None, price_tolerance:float=0.02, odd:bool=True):
        self.account = account.account
        self.sdk = account.sdk
        self.price_tolerance = price_tolerance
        self.odd = odd

        # portfolio
        self.portfolio_path = portfolio_path or config.portfolio_path_config.get('fubon_portfolio_path')
        self.portfolio = read_portfolio(self.portfolio_path)
        self.check_portfolio_match_inventories(self.portfolio)

        self.fubon_order_status = {
            0: '預約單',
            4: '系統將委託送往後台',
            9: '連線逾時',
            10: '委託成功',
            30: '未成交刪單成功',
            40: '部分成交，剩餘取消',
            50: '完全成交',
            90: '失敗',
        }

    # basic
    def list_inventories(self):
        return self.sdk.accounting.inventories(self.account).data
    
    def list_unrealized_pnl(self):
        return self.sdk.accounting.unrealized_gains_and_loses(self.account).data

    def list_orders(self):
        return self.sdk.stock.get_order_results(self.account).data
    
    def place_order(self,
        stock_id:str, action:Literal['buy', 'sell'], intraday_odd:Literal[0, 1], price:float, quantity:int, 
        price_type:str='limit', order_type:str='rod', trade_type:str='cash', note:str=None,
    ):
        param = {
            'action':{
                'buy':BSAction.Buy, 
                'sell':BSAction.Sell,
            },
            'intraday_odd':{
               0:MarketType.Common, 
               1:MarketType.IntradayOdd, 
            },
            'price_type':{
                'limit':PriceType.Limit,
                'mkt':PriceType.Market,
                'limit_up':PriceType.LimitUp,
                'limit_down':PriceType.LimitDown,
            },
            'order_type':{
                'fok':TimeInForce.FOK,
                'ioc':TimeInForce.IOC,
                'rod':TimeInForce.ROD,
            },
            'trade_type':{
                'cash':OrderType.Stock,
                'margin':OrderType.Margin,
                'short':OrderType.Short,
                'daytrade':OrderType.DayTrade,
            },
        }

        order = Order(
            buy_sell = param['action'][action],
            symbol = stock_id,
            price =  str(price),
            quantity =  quantity,
            market_type = param['intraday_odd'][intraday_odd],
            price_type = param['price_type'][price_type],
            time_in_force = param['order_type'][order_type],
            order_type = param['trade_type'][trade_type],
            user_def = note,
        )
        print((f'{action} {quantity} of {stock_id} at {price}') + (f' for strategy {note}' if note else ''))
        return self.sdk.stock.place_order(self.account, order)

    def place_batch_order(self, orders:list[Dict[str, Any]]):
        param = {
            'action':{
                'buy':BSAction.Buy, 
                'sell':BSAction.Sell,
            },
            'intraday_odd':{
               0:MarketType.Common, 
               1:MarketType.IntradayOdd, 
            },
            'price_type':{
                'limit':PriceType.Limit,
                'mkt':PriceType.Market,
                'limit_up':PriceType.LimitUp,
                'limit_down':PriceType.LimitDown,
            },
            'order_type':{
                'fok':TimeInForce.FOK,
                'ioc':TimeInForce.IOC,
                'rod':TimeInForce.ROD,
            },
            'trade_type':{
                'cash':OrderType.Stock,
                'margin':OrderType.Margin,
                'short':OrderType.Short,
                'daytrade':OrderType.DayTrade,
            },
        }
        
        orders = [
            Order(
                buy_sell = param[order['action']],
                symbol = order['stock_id'],
                price =  str(order['price']),
                quantity =  order['quantity'],
                market_type = param['intraday_odd'][order['intraday_odd']],
                price_type = param['price_type'][order['price_type']],
                time_in_force = param['order_type'][order['order_type']],
                order_type = param['trade_type'][order['trade_type']],
            ) for order in orders
        ]
        return self.sdk.stock.batch_place_order(self.account, orders)
    
    def place_condition_order(self, order, start_date, end_date):
        condition = Condition(
            market_type = TradingType.Reference,        
            symbol = "2881",
            trigger = TriggerContent.MatchedPrice,
            trigger_value = "66",
            comparison = Operator.LessThan
        )

        order = ConditionOrder(
            buy_sell = BSAction.Buy,
            symbol = "2881",
            price =  "66",
            quantity =  1000,
            market_type = ConditionMarketType.Common,
            price_type = ConditionPriceType.Limit,
            time_in_force = TimeInForce.ROD,
            order_type = ConditionOrderType.Stock
        )

        # 停損停利若為Market , price 則填空值""

        tp = TPSLOrder(
            time_in_force=TimeInForce.ROD,
            price_type=ConditionPriceType.Limit,
            order_type=ConditionOrderType.Stock,
            target_price="85",
            price="85",
            # trigger=TriggerContent.MatchedPrice  # v2.2.0 新增
        )


        sl = TPSLOrder(
            time_in_force=TimeInForce.ROD,
            price_type=ConditionPriceType.Limit,
            order_type=ConditionOrderType.Stock,
            target_price="60",
            price="60",
            # trigger=TriggerContent.MatchedPrice  # v2.2.0 新增
        )

        tpsl = TPSLWrapper(
            stop_sign= StopSign.Full,
            tp=tp,               # optional field 
            sl=sl,               # optional field
            end_date="20240517",
            intraday =False       # optional field
        )

        return self.sdk.stock.single_condition(self.account, start_date, start_date, StopSign.Full, condition, order, tpsl)

    def cancel_order(self, order):
        self.sdk.stock.cancel_order(self.account, order)
    
    def get_bank_balance(self):
        return self.sdk.accounting.bank_remain(self.account).data.balance
    
    def get_settlement(self, days:int=3):
        return self.sdk.accounting.query_settlement(self.account, f'{days}d').data

    # portfolio
    def settlement_is_enough(self, diff_portfolio:Dict[str, List[Position]]) -> bool:
        try:
            bank_balance = self.get_bank_balance()
            settlement = sum([s.total_settlement_amount if s.total_settlement_amount else 0 for s in self.get_settlement().details])
            portfolio_money = sum([sum([p.round_price*p.round_qty+p.odd_price*p.odd_qty for p in positions]) for positions in diff_portfolio.values()])
            
            return (bank_balance + settlement - portfolio_money) >= 0
        except Exception as e:
            print(f'error when checking settlement: {e}')
            return True

    def check_portfolio_match_inventories(self, portfolio:dict):
        portfolio = self.portfolio.copy()
        agg_json_portfolio = {}
        for p in portfolio['portfolio'].values():
            for stock_id, quantity in p.items():
                if stock_id not in agg_json_portfolio:
                    agg_json_portfolio[stock_id] = quantity
                else:
                    agg_json_portfolio[stock_id] += quantity
        agg_inv_portfolio = {i.stock_no:sum([i.today_qty, i.odd.today_qty])  for i in self.list_inventories()}
        diff_portfolio = {}
        for k, v in agg_inv_portfolio.items():
            if k not in agg_json_portfolio:
                diff_portfolio[k] = v
            else:
                diff_portfolio[k] = v - agg_json_portfolio[k]
        diff_portfolio = {k:v for k, v in diff_portfolio.items() if v != 0}
        if diff_portfolio=={}:
            print('portfolio.json matches inventories.')
        else:
            print(f'portfolio.json does not match inventories: {diff_portfolio}!')
    
    def update_portfolio(self):
        current_portfolio = self.portfolio['portfolio'].copy()
        last_update_time = dt.datetime.strptime(self.portfolio['update_time'], '%Y-%m-%d %H:%M:%S')
        traded_orders={
            (order.user_def, order.stock_no):\
            order.filled_qty * \
            (-1 if order.buy_sell=='Sell' else 1)\
            for order in self.list_orders() if \
            (self.fubon_order_status[order.status] in ['完全成交', '部分成交，剩餘取消']) & (dt.datetime.strptime(f"{order.date} {order.last_time}", "%Y/%m/%d %H:%M:%S.%f") > last_update_time)
        }

        for k, v in traded_orders.items():
            if not current_portfolio.get(k[0]):
                current_portfolio[k[0]]={}
            if not current_portfolio[k[0]].get(k[1]):
                current_portfolio[k[0]][k[1]] = v
            else:
                current_portfolio[k[0]][k[1]] += v
        
        # clear empty portfolio & position
        current_portfolio = {k:{k1:v1 for k1, v1 in v.items() if v1!=0} for k, v in current_portfolio.items() if v != {}}
        
        self.portfolio['portfolio'] = current_portfolio
        self.portfolio['update_time'] = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # trade
    def check_portfolio_difference(self,new_portfolio:Dict[str, List[Dict[str, Union[int, float]]]], by:Literal['money', 'quantity']='money'):
        difference_portfolio = {}
        if by == 'money':
            for strategy, positions in new_portfolio.items():
                difference_portfolio[strategy] = []
                current_strategy = self.portfolio['portfolio'].get(strategy, {})
                
                def process_money(stock_id, money):
                    inv_money = current_strategy.get(stock_id, 0) * next((inv.cost_price for inv in self.list_unrealized_pnl() if inv.stock_no == stock_id), 0)
                    position = Position().from_money(stock_id, money - inv_money)
                    if not position.empty:
                        return position
                    return None
                
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(process_money, stock_id, money) for stock_id, money in positions.items()]
                    for future in as_completed(futures):
                        position = future.result()
                        if position:
                            difference_portfolio[strategy].append(position)   
        elif by == 'quantity':
            for strategy, positions in new_portfolio.items():
                difference_portfolio[strategy] = []
                current_strategy = self.portfolio['portfolio'].get(strategy, {})
                
                def process_quantity(stock_id, quantity):
                    inv_quantity = current_strategy.get(stock_id, 0)
                    position = Position().from_quantity(stock_id, quantity - inv_quantity)
                    if not position.empty:
                        return position
                    return None
                
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(process_quantity, stock_id, quantity) for stock_id, quantity in positions.items()]
                    for future in as_completed(futures):
                        position = future.result()
                        if position:
                            difference_portfolio[strategy].append(position)
        return difference_portfolio
    
    def trade_position(self, position:Position, strategy_name:str=None):
        action = 'buy' if ((position.round_qty+position.odd_qty)>0) else 'sell'
        # place round order
        if position.round_qty!=0:
            self.place_order(
                stock_id=position.stock_id,
                action=action,
                intraday_odd=0,
                price=position.round_price,
                quantity=abs(position.round_qty),
                note=strategy_name,
            )
        
        # place odd order
        if position.odd_qty!=0:
            self.place_order(
                stock_id=position.stock_id,
                action=action,
                intraday_odd=1,
                price=position.odd_price,
                quantity=abs(position.odd_qty),
                note=strategy_name,
            )

    def rebalance(self, new_portfolio:Dict[str, List[Dict[str, Union[int, float]]]], by:Literal['money', 'quantity']='money', freq:int=1, ignore_mkt_open:bool=False, ignore_settlement:bool=False):
        while True:
            # machine time
            print(f"Current time: {dt.datetime.now().time().strftime('%H:%M:%S')}")
            if (not ignore_mkt_open) and (not mkt_is_open()):
                print("Market is closed.")
                break
            
            # cancel all orders
            for order in self.list_orders():
                if self.fubon_order_status[order.status] not in ['完全成交', '部分成交，剩餘取消']:
                    self.cancel_order(order)

            # update portfolio
            self.update_portfolio()

            # check difference
            diff_portfolio = self.check_portfolio_difference(new_portfolio, by=by)

            if (not ignore_settlement) and (not self.settlement_is_enough(diff_portfolio)):
                print("Settlement is not enough for position!")
                break
            
            if not all(v==[] for v in diff_portfolio.values()):
                # trade
                for strategy_name, positions in diff_portfolio.items():
                    for position in positions:
                        self.trade_position(position, strategy_name)
            else:
                print('Portfolio is all filled.')
                write_portfolio(self.portfolio, self.portfolio_path)
                break
            
            time.sleep(60*freq)

