import ccxt
import math
import matplotlib.pyplot as plt
import pandas as pd
import math

from pandas.core.frame import DataFrame
from pandas._libs.tslibs.timestamps import Timestamp
from actions.actions import Actions, ActionSimple, Investments
from agents.agent import Agent
from agents.basic.average_directional_index import AdxAgent
from agents.basic.average_true_range import AtrAgent
from agents.basic.bollinger_bands import BbAgent
from agents.basic.chaikin_money_flow import CmfAgent
from agents.basic.exponential_moving_average import EmaAgent
from agents.basic.fibonacci_retracement import FibonacciRetracementAgent
from agents.basic.moving_average_convergence_divergence import MacdAgent
from agents.basic.on_balance_volume import ObvAgent
from agents.basic.relative_strength_index import RsiAgent
from agents.basic.sentiment_analysis import SentimentAnalysisAgent
from agents.basic.simple_moving_average import SmaAgent
from agents.basic.stochastic_oscillator import SoAgent
from typing import List

def get_coin_data(coin: str, timeframe: str = '1h', start_date: Timestamp = None, end_date: Timestamp = None) -> DataFrame:
    """
    Get the coin data from binance exchange.

    Args:
        coin (str): The coin to get the data for.
        timeframe (str): The timeframe to get the data for.
            {'1s', '1m', '3m','5m', '15m','30m','1h','2h','4h',6h',8h,'12h',1d', '3d', '1w', '1M'}
        start_date (Timestamp): The start date of the data.
        end_date (Timestamp): The end date of the data.

    Returns:
        DataFrame: The coin data
    """
    exchange = ccxt.binance()
    exchange.load_markets()

    # print(exchange.timeframes)
    # > {'1s', '1m', '3m','5m', '15m','30m','1h','2h','4h',6h',8h,'12h',1d', '3d', '1w', '1M'}
    if start_date is None and end_date is None:
        ohlcv_data = exchange.fetch_ohlcv(coin, timeframe=timeframe)
    else:
        start_timestamp_ms = int(start_date.timestamp() * 1000)
        end_timestamp_ms = None
        if end_date is not None:
            end_timestamp_ms = int(end_date.timestamp() * 1000)
            ms_per_timeframe = _get_ms_per_timeframe(timeframe)

        if end_date is None:
            limit = 1000
        else:
            limit = min(
                1000,
                math.ceil((end_timestamp_ms - start_timestamp_ms) / ms_per_timeframe)
            )

        ohlcv_data = []
        while True:
            ohlcv = exchange.fetch_ohlcv(
                coin,
                timeframe=timeframe,
                since=start_timestamp_ms,
                limit=limit
            )

            if not ohlcv:
                break

            ohlcv_data.extend(ohlcv)

            start_timestamp_ms = ohlcv[-1][0] + 1
            if end_timestamp_ms is not None:
                limit = min(
                    limit,
                    math.ceil((end_timestamp_ms - start_timestamp_ms) / ms_per_timeframe)
                )
                if limit <= 0:
                    break
    
    df = pd.DataFrame(ohlcv_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    return df

def add_indicators(data: pd.DataFrame, indicators: List[Agent] = None) -> pd.DataFrame:
    """
    Function that adds indicator features.

    Args:
        data (pd.DataFrame): The data to add the indicators to
        indicators (List[Indicator], optional): The indicators to add. Defaults to None.

    Returns:
        pd.DataFrame: The data with the added indicators
    """
    if indicators is None:
        indicators = _get_default_indicators()   

    indicator_data = {}
    for indicator in indicators:
        indicator_data[indicator.__class__.__name__] = indicator.act(data)[Actions.INDICATOR_STRENGTH]
        if not indicator.is_action_strength_normalized():
            print(f'Indicator {indicator.__class__.__name__} is not normalized!')

    data = data.join(indicator_data)
    initial_intervals = max([indicator.get_initial_intervals() for indicator in indicators])
    
    data = data.iloc[initial_intervals:]
    return data

def _get_default_indicators() -> List[Agent]:
    """
    Function that returns the default indicators.
    """
    # TODO: idea, update Indicator class so that it can use different normalization methods
    return [
        AdxAgent(),
        AtrAgent(),     # TODO: normalize feature with pct_change
        BbAgent(),
        CmfAgent(),
        EmaAgent(),     # TODO: normalize feature close / ema. Looks like change from previous value is also important, ie: close was below ema and now is above => buy
        # TODO: consider EMA with different windows, 200!
        FibonacciRetracementAgent(),        # TODO: normalize by Close / R2, Close / S2
        MacdAgent(),
        ObvAgent(),     # TODO: normalize with pct_change
        RsiAgent(),
        SentimentAnalysisAgent(),   # TODO: replace Nan values with 0
        SmaAgent(),      # TODO: normalize
        SoAgent()
    ]

def normalize_features(data: pd.DataFrame, data_window: int = 60) -> pd.DataFrame:
    """
    Function that normalizes the data chart features: Close, High, Low, Open, Volume

    The normalization operation are done as follows:
        - Close = Close[-1] / Close[-2]
        - Open = Open[-1] / Close[-1]
        - High = High[-1] / Close[-1]
        - Low = Low[-1] / Close[-1]
        - Volume = Volume[-1] / Volume.rolling(data_window).max
    
    Args:
        data (pd.DataFrame): The data to normalize
        data_window (int): The window size for the volume feature

    Returns:
        pd.DataFrame: The data with normalized features. Normalization operations are done in place.
    """
    data['feature_Close'] = data['Close'].pct_change()
    data['feature_High'] = data['High'] / data['Close']
    data['feature_Low'] = data['Low'] / data['Close']
    data['feature_Open'] = data['Open'] / data['Close']
    data['feature_Volume'] = data['Volume'] / data['Volume'].rolling(data_window).max()

    data.dropna(inplace=True)
    return data

def _get_ms_per_timeframe(timeframe: str) -> int:
    """
    Function returns the amount of milliseconds per timeframe.

    Args:
        timeframe (str): The timeframe to get the amount of milliseconds for.

    Returns:
        int: The amount of milliseconds per timeframe.
    """
    if timeframe == '1s':
        return 1000
    elif timeframe == '1m':
        return 60 * 1000
    elif timeframe == '3m':
        return 3 * 60 * 1000
    elif timeframe == '5m':
        return 5 * 60 * 1000
    elif timeframe == '15m':
        return 15 * 60 * 1000
    elif timeframe == '30m':
        return 30 * 60 * 1000
    elif timeframe == '1h':
        return 60 * 60 * 1000
    elif timeframe == '2h':
        return 2 * 60 * 60 * 1000
    elif timeframe == '4h':
        return 4 * 60 * 60 * 1000
    elif timeframe == '6h':
        return 6 * 60 * 60 * 1000
    elif timeframe == '8h':
        return 8 * 60 * 60 * 1000
    elif timeframe == '12h':
        return 12 * 60 * 60 * 1000
    elif timeframe == '1d':
        return 24 * 60 * 60 * 1000
    elif timeframe == '3d':
        return 3 * 24 * 60 * 60 * 1000
    elif timeframe == '1w':
        return 7 * 24 * 60 * 60 * 1000
    elif timeframe == '1M':
        return 30 * 24 * 60 * 60 * 1000
    else:
        raise Exception('Timeframe not supported: ' + timeframe)

def plot_actions(coin_data: DataFrame, actions: Actions, coin: str, agent_name: str) -> None:
    """
    Function plots the actions on the coin data.

    Args:
        coin_data (DataFrame): The coin data
        actions (DataFrame): The actions to plot
        coin (str): The coin to plot
        agent_name (str): The name of the agent
    """
    plt.figure(figsize=(20, 10))
    plt.title(f'{coin} price with {agent_name} actions')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    
    plt.plot(coin_data.index, coin_data['Close'], label=f'{coin} price', color='black')
    
    # TODO: have integer deciding the strength of the action
    # plot strength as transparency/alpha
    buys = actions[actions[Actions.ACTION] == ActionSimple.BUY]
    sells = actions[actions[Actions.ACTION] == ActionSimple.SELL]

    plt.scatter(buys.index, coin_data.loc[buys.index]['Close'], label="Buy's", marker='^', color='green', s=80)
    plt.scatter(sells.index, coin_data.loc[sells.index]['Close'], label="Sell's", marker='v', color='red', s=80)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_profit(coin_data: DataFrame, investments: Investments, coin: str, agent_name: str) -> None:
    """
    Function plots the profit with the given investments.
    Plots acquired value over time (usd value + coins value)

    Show small histogram at bottom of plot with percentage of portfolio in USD and coins.

    Args:
        coin_data (DataFrame): The coin data
        investments (Investments): The investments to plot
        coin (str): The coin to plot
        agent_name (str): The name of the agent
    """
    plt.figure(figsize=(20, 10))
    plt.title(f'{coin} profit with {agent_name} investments')
    plt.xlabel('Date')
    plt.ylabel('Value (USD)')

    # plot value of invested USD
    total_value = []
    usd_value = []
    coins_value = []

    usd_invested = 0
    coins_invested = 0
    for i in range(len(coin_data)):
        if coin_data.index[i] in investments.index:
            usd_invested = investments.loc[coin_data.index[i]][Investments.FIAT_AMOUNT_INVESTED]
            coins_invested = investments.loc[coin_data.index[i]][Investments.COIN_AMOUNT_INVESTED]

        usd_value.append(usd_invested)
        coins_value.append(coins_invested * coin_data.loc[coin_data.index[i]]['Close'])

        total_value.append(usd_value[i] + coins_value[i])
    
    plt.plot(coin_data.index, total_value, label='Total value', color='black')
    plt.grid(True)
    plt.legend()
    plt.show()

    # plot percentage of portfolio in USD and coins
    plt.figure(figsize=(20, 10))
    plt.title(f'Percentage of portfolio in USD and {coin} value with {agent_name} investments')
    plt.xlabel('Date')
    plt.ylabel('Percentage of portfolio')

    # plot usd_value / total_value with background color green below the plot line, and red above
    ratio = [usd_value[i] / total_value[i] for i in range(len(total_value))]

    plt.plot(coin_data.index, ratio, label='USD value portfolio percentage', color='black')
    plt.fill_between(
        coin_data.index,
        ratio,
        [1 for _ in range(len(ratio))],
        interpolate=True,
        color='red',
        alpha=0.5
    )
    plt.fill_between(
        coin_data.index,
        ratio,
        [0 for _ in range(len(ratio))],
        interpolate=True,
        color='green',
        alpha=0.5
    )
    plt.grid(True)
    plt.legend()
    plt.show()

    # plot profit and coin crease over time
    plt.figure(figsize=(20, 10))
    plt.title(f'{coin} value and {agent_name} profit over time')
    plt.xlabel('Date')
    plt.ylabel('Change %')

    coin_diff_percentage = []
    for i in range(len(coin_data)):
        coin_diff_percentage.append((coin_data.iloc[i]['Close'] - coin_data.iloc[0]['Close']) / coin_data.iloc[0]['Close'])

    plt.plot(coin_data.index, coin_diff_percentage, label=f'{coin} value change', color='black')

    profit = []
    for i in range(len(total_value)):
        profit.append((total_value[i] - total_value[0]) / total_value[0])
    
    plt.plot(coin_data.index, profit, label='Profit', color='blue')
    plt.grid(True)
    plt.legend()
    plt.show()
