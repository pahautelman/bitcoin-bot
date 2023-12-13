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
    overlap: bool = False 
    ) -> None:
    """
    Function plots the coin price and the agent indicator strength.

    Args:
        coin_data (DataFrame): The coin data
        agent (Agent): The agent
        overlap (bool): Whether to plot the agent actions on the same plot as the coin
    """
    actions = agent.act(coin_data)

    plt.figure(figsize=(20, 10))
    plt.title(f'{agent.__class__.__name__} indicator strength')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.plot(coin_data.index, coin_data['Close'])
    
    if not overlap:
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(20, 10))
        plt.title(f'{agent.__class__.__name__} indicator strength')
        plt.xlabel('Date')
        plt.ylabel('Indicator strength')
        plt.plot(actions.index, actions[Actions.INDICATOR_STRENGTH])
    else:
        # TODO: check this shyte
        # plot the agent actions on the same plot as the coin
        plt.twinx()
        plt.plot(actions.index, actions[Actions.INDICATOR_STRENGTH], color='red')

    plt.grid(True)
    plt.show()

def plot_indicator(
    coin_data: DataFrame,
    agent: Agent,
    indicator_names: list[str],
    overlap: bool = False
    ) -> None:
    """
    Function plots the coin price and the indicator.

    Args:
        coin_data (DataFrame): The coin data
        agent (Agent): The agent
        indicator_names (list[str]): The indicator DataFrame columns to plot on map
        overlap (bool): Whether to plot the indicator on the same plot as the coin
    """
    indicators = agent.get_indicator(coin_data)

    plt.figure(figsize=(20, 10))
    plt.title(f'Agent {agent.__class__.__name__}')
    plt.plot(coin_data.index, coin_data['Close'])
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')

    if not overlap:
        plt.grid(True)
        plt.show()
        
        for indicator_name in indicator_names:
            plt.figure(figsize=(20, 10))
            plt.title(f'Indicator {indicator_name}')
            plt.xlabel('Date')
            plt.ylabel('Indicator strength')
            plt.plot(indicators.index, indicators[indicator_name], label=indicator_name)

        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        for indicator_name in indicator_names:
            plt.plot(indicators.index, indicators[indicator_name], label=indicator_name)

        plt.grid(True)
        plt.legend()
        plt.show()
