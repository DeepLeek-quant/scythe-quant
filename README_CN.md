
# QuantDev

[English](README.md) | [中文](README_CN.md)

一個用於量化交易的 Python 模組。

## 目錄
1. 配置設定
2. 資料管理
3. 回測
4. 投資組合管理

## 配置設定

在使用 QuantDev 之前，您需要在 `config` 目錄中設置配置文件。該模組支援自定義配置路徑：

```python
import quantdev
quantdev.set_config_dir('/path/to/config')
```

## 資料管理

`Databank` 類別整合多個資料源並提供統一的存取方式：

- TEJ (Taiwan Economic Journal) 資料
- FinMind 資料
- 公開資料 (公司資訊, 無風險利率)
- 處理過的資料 (自行計算的技術指標)

### 範例

```python
from quantdev.data import Databank

# 初始化資料管理器
db = Databank()

# 讀取特定資料集
df = db.read_dataset('monthly_rev')

# 讀取帶有過濾條件和列選擇的資料
df = db.read_dataset(
    dataset='stock_trading_data',
    filter_date='t_date',
    start='2023-01-01',
    end='2023-12-31',
    columns=['stock_id', '收盤價']
)

# 列出所有可用的資料集和列
datasets = db.list_datasets()
columns = db.list_columns('stock_trading_data')
dataset = db.find_dataset('收盤價')  # Returns 'stock_trading_data'

# 更新資料
db.update_databank(include=['monthly_rev'])  # 更新特定資料集
db.update_databank(exclude=['stock_basic_info'])  # 更新所有資料集，排除指定資料集
db.update_processed_data(dataset='fin_data_chng')  # 更新處理過的資料
```

## Backtesting

`backtesting()` 函數測試交易策略並返回 `Strategy` 實例，包含分析工具：
- 績效指標 vs 基準
- 位置資訊追蹤
- 互動式報告，多個視圖：
  - 淨值曲線
  - 相對報酬
  - 報酬熱力圖
  - 流動性分析
  - ...

```python
import quantdev.backtest as bts

# 獲取策略條件的資料
data = bts.get_data('ROE')  # 獲取 ROE 資料

# 定義策略條件
condition = data > 0

# 使用季度再平衡運行回測
strategy = bts.backtesting(
    data=condition,
    rebalance='QR',         # 季度報告發布日期再平衡
    signal_shift=1,         # 信號執行延遲 1 天
    hold_period=None,       # 持有直到下一個再平衡日期
    stop_loss=None,         # 無停損
    stop_profit=None,       # 無停利
    benchmark='0050',      # 使用 0050 作為基準
    start='2020-01-01',   # 回測開始日期
    end='2023-12-31'      # 回測結束日期
)
```

### Examples

```python
# 獲得策略績效指標
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

# 獲得詳細的位置資訊
strategy.position_info
'''
            name       buy_date   sell_date  警示狀態  漲跌停  前次成交量_K  前次成交額_M  季報公佈日期    月營收公佈日期  板塊別      主產業別
stock_id                                                                                       
1436         全家     2024-11-15  2025-04-01    =      =        32        6.08    2024-11-06  2025-01-09   上櫃一般版  OTC38 OTC 居家生活
1446         宏和     2024-11-15  2025-04-01    =      =       254        9.79    2024-11-12  2025-01-10   上市一般版  M1400 紡織纖維  
'''

# 生成互動式分析報告
strategy.report  # 顯示包含多個分析選項的互動式圖表
```

## 投資組合管理

投資組合管理系統通過 SinoPac 的 API 執行交易，並使用 Fugle MarketData 的即時資料。系統由三個主要組件組成，共同工作：

### Position Class

`Position` 類別表示將在投資組合中管理的個別股票位置：

```python
from quantdev.trade import Position

# 為您的交易策略創建位置
tsmc_position = Position(('2330', 100000))    # 100,000 TWD of TSMC
hon_hai_position = Position(('2317', 50000))   # 50,000 TWD of Hon Hai
mediatek_position = Position(('2454', 80000))  # 80,000 TWD of MediaTek

# 在添加到投資組合之前更新最新的報價
tsmc_position.refresh_quotes(price_tolerance=0.02, odd=True)
hon_hai_position.refresh_quotes(price_tolerance=0.02, odd=True)
mediatek_position.refresh_quotes(price_tolerance=0.02, odd=True)
```

### Portfolio Class

`Portfolio` 類別將多個位置組合為基於策略的投資組合，將被執行：

```python
from quantdev.trade import Portfolio

# 創建一個包含多個交易策略的投資組合
portfolio = Portfolio({
    'tech_leaders': [tsmc_position, hon_hai_position],
    'semiconductor': [mediatek_position]
})

# 直接使用股票代碼和分配創建投資組合
portfolio = Portfolio({
    'tech_leaders': (['2330', '2317'], 150000),  # 150K TWD split between TSMC and Hon Hai
    'semiconductor': (['2454'], 80000)            # 80K TWD in MediaTek
})

# 確保所有位置報價都是最新的
portfolio.refresh_quotes()
```

### Portfolio Manager

`PortfolioManager` 類別在實時執行和監控您的投資組合策略：

```python
from quantdev.trade import SinoPacAccount, PortfolioManager

# 初始化 SinoPac 交易帳戶
account = SinoPacAccount()  # 自動處理登入

# 使用您的投資組合創建投資組合管理器
pm = PortfolioManager(
    api=account,
    portfolio=None  # 自動從 config/portfolio_path.json 獲取投資組合
)

# 新的投資組合 (例如，基於新的策略信號)
new_portfolio = Portfolio({
    'tech_leaders': [
        Position(('2330', 120000)),  # 增加 台積電 位置
        Position(('2317', 30000))    # 減少 鴻海 位置
    ],
    'semiconductor': [
        Position(('2454', 100000)),  # 增加 聯發科 位置
        Position(('2379', 50000))    # 增加 瑞昱 位置
    ]
})

# 更新投資組合
pm.update(new_portfolio)

# 開始自動投資組合管理 - 這將：
# 1. 計算位置差異
# 2. 生成必要的買賣訂單
# 3. 執行訂單
pm.run_portfolio_manager(
    freq=1,              # 每分鐘更新位置
    ignore_bank_balance=False,  # 確保足夠的資金
    ignore_mkt_open=False      # 僅在市場開盤時交易
)

```

上述物件共同工作以：
1. 使用 `Position` 類別定義個別位置
2. 使用 `Portfolio` 類別將位置組合為策略
3. 通過 `PortfolioManager` 執行和監控交易
4. 基於策略信號動態更新位置