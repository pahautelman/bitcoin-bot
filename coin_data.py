import matplotlib.pyplot as plt
import ccxt
import pandas as pd

from pandas.core.frame import DataFrame
from actions.actions import Actions, ActionSimple, Investments


def get_coin_data(coin: str, timestamp: str) -> DataFrame:
    """
    Get the coin data from binance exchange.

    Args:
        coin (str): The coin to get the data for
        timestamp (str): The timestamp to get the data for (1m, 1h, 1d, etc)

    Returns:
        DataFrame: The coin data
    """
    exchange = ccxt.binance()

    exchange.load_markets()

    # {'1s', '1m', '3m','5m', '15m','30m','1h','2h','4h',6h',8h,'12h',1d', '3d', '1w', '1M'}
    # print(exchange.timeframes)
    data = exchange.fetch_ohlcv(coin, timeframe=timestamp)
    
    # data = exchange.fetch_ohlcv(coin)
    
    # params = {
    #     'price': 'mark',
    #     'until': 0,
    # }
    # :param str [params.price]: "mark" or "index" for mark price and index price candles
    # :param int [params.until]: timestamp in ms of the latest candle to fetch
    # :param boolean [params.paginate]: default False, when True will automatically paginate by calling self endpoint multiple times. See in the docs all the [availble parameters](https://github.com/ccxt/ccxt/wiki/Manual#pagination-params)
    # data = exchange.fetch_ohlcv(
    #     coin, 
    #     timeframe='str timeframe: the length of time each candle represents', 
    #     since='int [since]: timestamp in ms of the earliest candle to fetch',
    #     params=params
    # )
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'Close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


def plot_actions(coin_data: DataFrame, actions: Actions, coin: str) -> None:
    """
    Function plots the actions on the coin data.

    Args:
        coin_data (DataFrame): The coin data
        actions (DataFrame): The actions to plot
        coin (str): The coin to plot
    """
    plt.figure(figsize=(20, 10))
    plt.title(f'{coin} price with actions')
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
            usd_invested = investments.loc[coin_data.index[i]][Investments.USD_AMOUNT_INVESTED]
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
    plt.title(f'Percentage of portfolio in USD and {coin} value')
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
