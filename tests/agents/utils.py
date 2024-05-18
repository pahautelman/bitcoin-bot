import matplotlib.pyplot as plt

from pandas import DataFrame
from pandas._libs.tslibs.timestamps import Timestamp


import sys
sys.path.append('../../..')

from coin_data import get_coin_data
from agents.agent import Agent
from actions.actions import Actions


def get_test_coin_data() -> DataFrame:
    """
    Get hourly BTC data for the time interval 2023-09-24 - 2023-11-24.

    Returns:
        DataFrame: The BTC data
    """
    return get_coin_data(
        coin='BTC/USDT',
        timeframe='1h',
        start_date=Timestamp('2023-09-24'),
        end_date=Timestamp('2023-11-24')
    )

def get_short_test_coin_data() -> DataFrame:
    """
    Get hourly BTC data for the time interval 2023-09-24 - 2023-09-30.

    Returns:
        DataFrame: The BTC data
    """
    return get_coin_data(
        coin='BTC/USDT',
        timeframe='1h',
        start_date=Timestamp('2023-09-24'),
        end_date=Timestamp('2023-09-30')
    )

def plot_indicator_strength(
    coin_data: DataFrame,
    agent: Agent,
    overlap: bool = False,
    ) -> None:
    """
    Function plots the coin price and the agent indicator strength.

    Args:
        coin_data (DataFrame): The coin data
        agent (Agent): The agent
        overlap (bool): Whether to plot the agent actions on the same plot as the coin
    """
    actions = agent.act(coin_data)

    if not overlap:
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(20, 20))
        fig.suptitle(f'{agent.__class__.__name__} indicator strength')

        # plot coin data
        axs[0].plot(coin_data.index, coin_data['Close'], label='Close Price')
        axs[0].set_ylabel('Price (USD)')
        axs[0].grid(True)
        axs[0].legend()

        # plot indicator strength
        axs[1].plot(actions.index, actions[Actions.INDICATOR_STRENGTH], label='Indicator strength')
        axs[1].set_ylabel('Indicator strength')
        axs[1].grid(True)
        axs[1].legend()

        plt.xlabel('Date')
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(20, 10))
        plt.title(f'{agent.__class__.__name__} indicator strength')
        plt.plot(coin_data.index, coin_data['Close'], label='Close Price')
        plt.plot(actions.index, actions[Actions.INDICATOR_STRENGTH], label='Indicator strength')
        
        plt.xlabel('Date')
        plt.ylabel('Price / Indicator')
        plt.grid(True)
        plt.legend()
        plt.show()

def plot_indicator(
    coin_data: DataFrame,
    agent: Agent,
    indicator_names: list[str],
    overlap: bool = False,
    overlap_indicators = False
    ) -> None:
    """
    Function plots the coin price and the indicator.

    Args:
        coin_data (DataFrame): The coin data
        agent (Agent): The agent
        indicator_names (list[str]): The indicator DataFrame columns to plot on map
        overlap (bool): Whether to plot the indicator on the same plot as the coin
        overlap_indicators (bool): Whether to plot the indicators on the same plot
    """
    indicators = agent.get_indicator(coin_data)        

    if not overlap:
        if not overlap_indicators:
            fig, axs = plt.subplots(nrows=len(indicator_names) + 1, ncols=1, figsize=(20, 10*(len(indicator_names) + 1)))
        else:
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
        fig.suptitle(f'Agent {agent.__class__.__name__}')

        # Plot coin data
        axs[0].plot(coin_data.index, coin_data['Close'], label='Close Price')
        axs[0].set_ylabel('Price (USD)')
        axs[0].grid(True)
        axs[0].legend()

        # Plot each indicator
        for i, indicator_name in enumerate(indicator_names, start=1):            
            # nan value should be plotted as -1
            indicators[indicator_name] = indicators[indicator_name].fillna(-1)

            if not overlap_indicators:
                index = i
            else:
                index = 1
            axs[index].plot(indicators.index, indicators[indicator_name], label=indicator_name)
            axs[index].set_ylabel('Indicator value')
            axs[index].grid(True)
            axs[index].legend()

        plt.xlabel('Date')
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(20, 10))
        plt.title(f'Agent {agent.__class__.__name__}')
        plt.plot(coin_data.index, coin_data['Close'], label='Close Price')
        
        for indicator_name in indicator_names:
            plt.plot(indicators.index, indicators[indicator_name], label=indicator_name)

        plt.xlabel('Date')
        plt.ylabel('Price / Indicator')
        plt.grid(True)
        plt.legend()
        plt.show()

def plot_actions(coin_data: DataFrame, agent: Agent) -> None:
    """
    Function plots the coin price and the agent actions (BUY, SELL).

    Args:
        coin_data (DataFrame): The coin data
        agent (Agent): The agent
    """
    actions = agent.act(coin_data)

    plt.figure(figsize=(20, 10))
    plt.title(f'{agent.__class__.__name__} actions')
    plt.plot(coin_data.index, coin_data['Close'], label='Close Price', zorder=1)
    
    plt.scatter(actions.index[actions[Actions.ACTION] == 'BUY'], coin_data['Close'][actions[Actions.ACTION] == 'BUY'], color='green', marker='^', label='Buy signal', lw=0, zorder=2, s=100)
    plt.scatter(actions.index[actions[Actions.ACTION] == 'SELL'], coin_data['Close'][actions[Actions.ACTION] == 'SELL'], color='red', marker='v', label='Sell signal', lw=0, zorder=2, s=100)
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    plt.show()
