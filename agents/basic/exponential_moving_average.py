from actions.actions import Actions, ActionSimple
from agents.agent import Indicator
from finta import TA
from pandas.core.frame import DataFrame
from typing import Tuple

class EmaAgent(Indicator):
    """
    Agent that implements *exponential moving average* (EMA) strategy.
    EMA is a trend-following momentum lagging indicator that places a greater weight and significance on the most recent data points.
    EMA reacts faster to recent price changes than SMA.

    The EMA is calculated using the following formula:
        EMA = Price(t) * k + EMA(y) * (1 - k)
        t = today, y = yesterday, N = window size, k = 2/(N+1)

    In general, the 50 and 200 EMA are used to determine the long-term trend.

    When the price crosses above the EMA, it may indicate a potential upward price movement.
    When the price crosses below the EMA, it may indicate a potential downward price movement.

    The (long-term) EMA provides signals for buying and selling:
        1. Buy when the price crosses above the EMA.
        2. Sell when the price crosses below the EMA.
    """

    def __init__(self, window: int = 50):
        """
        Args:
            window (int): The window size for the EMA
        """
        self.window = window

    def is_action_strength_normalized(self) -> bool:
        return False
    
    def get_initial_intervals(self) -> int:
        return self.window

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements EMA strategy.
        Buy when the price crosses above the EMA.
        Sell when the price crosses below the EMA.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        ema = self.get_indicator(coin_data)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= self.get_initial_intervals():
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i + 1], ema.iloc[:i + 1])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date, 
            data={
                Actions.ACTION: actions,
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )

    EMA = 'EMA'

    def get_indicator(self, coin_data: DataFrame) -> DataFrame:
        """
        Method that returns the EMA indicator.

        Args:
            coin_data (DataFrame): The coin data

        Returns:
            DataFrame: The EMA
        """
        return self._get_ema(coin_data, self.window)

    def _get_ema(self, coin_data: DataFrame, window: int) -> DataFrame:
        """
        Function calculates the EMA.

        Args:
            coin_data (DataFrame): The coin data
            window (int): The window size for the EMA

        Returns:
            DataFrame: The EMA
        """
        ema = TA.EMA(coin_data, window)
        return DataFrame(
            index=ema.index,
            data = {
                self.EMA: ema.values,
            }
        )

    def _get_simple_action(self, coin_data: DataFrame, ema: DataFrame) -> Tuple[ActionSimple, int]:
        """
        Function gets the EMA simple action. 
        Buy when the price crosses above the EMA.
        Sell when the price crosses below the EMA.

        Args:
            coin_data (DataFrame): The coin data
            ema (DataFrame): The EMA

        Returns:
            (ActionSimple, int): The action and indicator strength
        """
        if coin_data.iloc[-1]['Close'] > ema.iloc[-1][self.EMA]:
            action = ActionSimple.BUY
        elif coin_data.iloc[-1]['Close'] < ema.iloc[-1][self.EMA]:
            action = ActionSimple.SELL
        else:
            action = ActionSimple.HOLD
        return action, ema.iloc[-1][self.EMA]
