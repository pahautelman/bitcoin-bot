import yfinance as yf
import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame
from actions.actions import Actions, ActionSimple, Investments

def get_coin_data(coin: str, interval: str, period: str) -> DataFrame:
    """
    Get the coin data from Yahoo Finance API

    Args:
        coin (str): The coin to get the data for
        interval (str): The interval to get the data for
        period (str): The period to get the data for

    Returns:
        DataFrame: The coin data
    """
    coin_data = yf.Ticker(coin).history(interval=interval, period=period)
    return coin_data

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
    coins_acquired = 0
    coins_value = []
    usd_value = []
    for i in range(len(coin_data)):
        if coin_data.index[i] in investments.index:
            coins_acquired += investments.loc[coin_data.index[i]][Investments.COIN_AMOUNT_INVESTED]
            assert coins_acquired >= 0
            coins_value.append(coins_acquired * coin_data.loc[coin_data.index[i]]['Close'])

            usd_value.append(investments.loc[coin_data.index[i]][Investments.USD_AMOUNT_INVESTED])
            assert usd_value[-1] >= 0
        else:
            coins_value.append(coins_acquired * coin_data.loc[coin_data.index[i]]['Close'])
            usd_value.append(usd_value[-1])
        
        total_value.append(coins_value[i] + usd_value[i])
        assert total_value[-1] >= 0
    
    plt.plot(coin_data.index, total_value, label='Total value', color='black')
    plt.show()

    # plot percentage of portfolio in USD and coins
    plt.figure(figsize=(20, 10))
    plt.title(f'{coin} profit with {agent_name} investments')
    plt.xlabel('Date')
    plt.ylabel('Percentage of portfolio')
    plt.bar(
        coin_data.index, 
        [usd_value[i] / total_value[i] for i in range(len(total_value))], 
        label='USD', 
        color='green'
    )
    plt.bar(
        coin_data.index, 
        [coins_value[i] / total_value[i] for i in range(len(total_value))], 
        bottom=[usd_value[i] / total_value[i] for i in range(len(total_value))], 
        label=f'{coin} value', 
        color='red'
    )



