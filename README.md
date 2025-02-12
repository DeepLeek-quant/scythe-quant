# QuantDev

[English](README.md) | [中文](README_CN.md)

A Python module for quantitative trading.

## Table of Contents
1. Configuration
2. Data Management
3. Backtesting
4. Portfolio Management

## Configuration

Before using QuantDev, you'll need to set up your configuration files in a `config` directory. The module supports custom configuration paths:

```python
import quantdev
quantdev.set_config_dir('/path/to/config') # if not set, the default is ./config
```

## Data Management

The `Databank` class integrates multiple data sources and provides unified access to:

- TEJ (Taiwan Economic Journal) data
- FinMind data
- Public data (company information, risk-free rates)
- Processed (self-calculated) data (e.g., technical indicators)

### Examples

```python
from quantdev.data import Databank

# Initialize the data manager
db = Databank()

# Read specific dataset
df = db.read_dataset('monthly_rev')

# Read with filters and column selection
df = db.read_dataset(
    dataset='stock_trading_data',
    filter_date='t_date',
    start='2023-01-01',
    end='2023-12-31',
    columns=['stock_id', '收盤價']
)

# List available datasets and columns
datasets = db.list_datasets()
columns = db.list_columns('stock_trading_data')
dataset = db.find_dataset('收盤價')  # Returns 'stock_trading_data'

# Update data
db.update_databank(include=['monthly_rev'])  # Update specific dataset
db.update_databank(exclude=['stock_basic_info'])  # Update all except specified
db.update_processed_data(dataset='fin_data_chng')  # Update processed data
```

## Backtesting

The `backtesting()` function tests trading strategies and returns a `Strategy` instance with analysis tools including:
- Performance metrics vs benchmark
- Position information tracking
- Interactive reporting with multiple views:
  - Equity Curve
  - Relative Return
  - Return Heatmap
  - Liquidity Analysis
  - ...

```python
import quantdev.backtest as bts

# Get data for strategy condition
data = bts.get_data('ROE')  # Get ROE data

# Define strategy condition
condition = data > 0

# Run backtest with quarterly rebalancing
strategy = bts.backtesting(
    data=condition,
    rebalance='QR',         # Quarterly report release date rebalancing
    signal_shift=1,         # 1 day delay for signal execution
    hold_period=None,       # Hold until next rebalance date
    stop_loss=None,         # No stop loss
    stop_profit=None,       # No stop profit
    benchmark='0050',      # Use 0050 as benchmark
    start='2020-01-01',   # Backtest start date
    end='2023-12-31'      # Backtest end date
)
```

### Examples

```python
# Access strategy performance metrics
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

# Get detailed position information
strategy.position_info
'''
            name       buy_date   sell_date  警示狀態  漲跌停  前次成交量_K  前次成交額_M  季報公佈日期    月營收公佈日期  板塊別      主產業別
stock_id                                                                                       
1436         全家     2024-11-15  2025-04-01    =      =        32        6.08    2024-11-06  2025-01-09   上櫃一般版  OTC38 OTC 居家生活
1446         宏和     2024-11-15  2025-04-01    =      =       254        9.79    2024-11-12  2025-01-10   上市一般版  M1400 紡織纖維  
'''

# Generate interactive analysis report
strategy.report  # Shows interactive plots with multiple analysis tabs
```

## Portfolio Management

The portfolio management system executes trades via SinoPac's API with real-time data from Fugle MarketData. The system consists of three main components that work together:

### Position Class

The `Position` class represents individual stock positions that will be managed in your portfolio:

```python
from quantdev.trade import Position

# Create positions for your trading strategy
tsmc_position = Position(('2330', 100000))    # 100,000 TWD of TSMC
hon_hai_position = Position(('2317', 50000))   # 50,000 TWD of Hon Hai
mediatek_position = Position(('2454', 80000))  # 80,000 TWD of MediaTek

# Update latest quotes before adding to portfolio
tsmc_position.refresh_quotes(price_tolerance=0.02, odd=True)
hon_hai_position.refresh_quotes(price_tolerance=0.02, odd=True)
mediatek_position.refresh_quotes(price_tolerance=0.02, odd=True)
```

### Portfolio Class

The `Portfolio` class combines multiple positions into strategy-based portfolios that will be executed:

```python
from quantdev.trade import Portfolio

# Create a portfolio with multiple trading strategies
portfolio = Portfolio({
    'tech_leaders': [tsmc_position, hon_hai_position],
    'semiconductor': [mediatek_position]
})

# Alternatively, create portfolio directly with stock symbols and allocation
portfolio = Portfolio({
    'tech_leaders': (['2330', '2317'], 150000),  # 150K TWD split between TSMC and Hon Hai
    'semiconductor': (['2454'], 80000)            # 80K TWD in MediaTek
})

# Ensure all position quotes are up-to-date before execution
portfolio.refresh_quotes()
```

### Portfolio Manager

The `PortfolioManager` class executes and monitors your portfolio strategies in real-time:

```python
from quantdev.trade import SinoPacAccount, PortfolioManager

# Initialize SinoPac trading account
account = SinoPacAccount()  # Handles login automatically

# Create portfolio manager with your portfolio
pm = PortfolioManager(
    api=account,
    portfolio=portfolio  # Pass your portfolio object
)

# Synchronize with actual positions before trading
pm.sync()

# Start automated portfolio management
pm.run_portfolio_manager(
    freq=1,              # Update positions every minute
    ignore_bank_balance=False,  # Ensure sufficient funds
    ignore_mkt_open=False      # Trade only during market hours
)

```

The above components work together to:
1. Define individual positions with `Position` class
2. Group positions into strategies using `Portfolio` class
3. Execute and monitor trades through `PortfolioManager`


### Portfolio Manager

The `PortfolioManager` class executes and monitors your portfolio strategies in real-time:

```python
from quantdev.trade import SinoPacAccount, PortfolioManager

# Initialize SinoPac trading account
account = SinoPacAccount()  # Handles login automatically

# Create portfolio manager with your portfolio
pm = PortfolioManager(
    api=account,
    portfolio=None  # Automatically fetch portfolio from config/portfolio_path.json
)

# new portfolio (e.g., based on new strategy signals)
new_portfolio = Portfolio({
    'tech_leaders': [
        Position(('2330', 120000)),  # Increase TSMC position
        Position(('2317', 30000))    # Decrease Hon Hai position
    ],
    'semiconductor': [
        Position(('2454', 100000)),  # Increase MediaTek position
        Position(('2379', 50000))    # Add new position in Realtek
    ]
})

# Update the portfolio
pm.update(new_portfolio)

# Start automated portfolio management - this will:
# 1. Calculate position differences
# 2. Generate necessary buy/sell orders
# 3. Queue orders for execution
pm.run_portfolio_manager(
    freq=1,              # Update positions every minute
    ignore_bank_balance=False,  # Ensure sufficient funds
    ignore_mkt_open=False      # Trade only during market hours
)

```

The above components work together to:
1. Define individual positions with `Position` class
2. Group positions into strategies using `Portfolio` class
3. Execute and monitor trades through `PortfolioManager`
4. Update positions dynamically based on strategy signals