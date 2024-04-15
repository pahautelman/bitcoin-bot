from actions.actions import Actions, ActionSimple
from agents.agent import Indicator
from finta import TA
from pandas import DataFrame
from typing import Tuple

class MacdAgent(Indicator):
    """
    Agent that implements *moving average convergence divergence* (MACD) strategy.
    MACD is a trend-following momentum lagging indicator that shows the relationship between two moving averages of an asset's price.

    MACD is calculated using the following steps:
        1. Calculate the short-term Exponential Moving Average (EMA) of the asset's price using fast_period.
        2. Calculate the long-term EMA of the asset's price using slow_period.
        3. Subtract the long-term EMA from the short-term EMA to obtain the MACD line.
        4. Calculate the signal line, which is a signal_period EMA of the MACD line.
        5. Calculate the MACD histogram, which represents the difference between the MACD line and the signal line.
    
    When the MACD line crosses above the signal line, it may indicate a potential upward price movement.
    When the MACD line crosses below the signal line, it may indicate a potential downward price movement. 
    The larger the difference between the MACD line and the signal line, the stronger the signal.    
    
    The MACD histogram provides signals for buying and selling:
        1. Buy when the MACD line crosses above the signal line.
        2. Sell when the MACD line crosses below the signal line.
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Args:
            fast_period (int): The fast period for the MACD
            slow_period (int): The slow period for the MACD
            signal_period (int): The signal period for the MACD
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def is_action_strength_normalized(self) -> bool:
        return True

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements MACD strategy.
        Buy when the MACD line crosses above the signal line.
        Sell when the MACD line crosses below the signal line.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        macd = self.get_indicator(coin_data)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= self.slow_period:
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i + 1], macd.iloc[:i + 1])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date, 
            data={
                Actions.ACTION: actions,
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )

    MACD = 'MACD'
    SIGNAL = 'SIGNAL'

    def get_indicator(self, coin_data: DataFrame) -> DataFrame:
        return self._get_macd(coin_data, self.fast_period, self.slow_period, self.signal_period)

    def _get_macd(self, coin_data: DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> DataFrame:
        """
        Function calculates the MACD.

        Args:
            coin_data (DataFrame): The coin data
            fast_period (int): The fast period for the MACD
            slow_period (int): The slow period for the MACD
            signal_period (int): The signal period for the MACD

        Returns:
            DataFrame: The MACD
        """
        macd = TA.MACD(coin_data, period_fast=fast_period, period_slow=slow_period, signal=signal_period)
        return DataFrame(
            index=macd.index,
            data={
                self.MACD: macd['MACD'].values,
                self.SIGNAL: macd['SIGNAL'].values
            }
        )

    def _get_simple_action(self, coin_data: DataFrame, macd: DataFrame) -> Tuple[ActionSimple, int]:
        """
        Function gets the action to take based on the MACD.

        Args:
            coin_data (DataFrame): The coin data
            macd (DataFrame): The MACD

        Returns:
            ActionSimple: The action to take
            int: The indicator strength
        """
        action = ActionSimple.HOLD
        # if macd line is above the signal
        if macd.iloc[-1][self.MACD] > macd.iloc[-1][self.SIGNAL]:
            indicator_strength = 1
            # and it previously was not
            if macd.iloc[-2][self.MACD] < macd.iloc[-2][self.SIGNAL]:
                action = ActionSimple.BUY
        # if macd line is below the signal
        elif macd.iloc[-1][self.MACD] < macd.iloc[-1][self.SIGNAL]:
            indicator_strength = -1
            # and it previously was not
            if macd.iloc[-2][self.MACD] > macd.iloc[-2][self.SIGNAL]:
                action = ActionSimple.SELL
        return action, indicator_strength
