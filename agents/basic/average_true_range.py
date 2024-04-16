from actions.actions import Actions, ActionSimple
from agents.agent import Indicator
from finta import TA
from pandas import DataFrame
from typing import Tuple

class AtrAgent(Indicator):
    """
    Agent that implements *average true range* (ATR) strategy.
    ATR is a market volatility indicator that shows the average range of an asset's price over a specified time period.

    ATR is calculated using the following steps:
        1. Calculate the true range (TR) over the window size.
            TR = max(high - low, abs(high - close_prev), abs(low - close_prev))
            high = current high
            low = current low
            close_prev = previous close
        2. Calculate the ATR:
            ATR = (ATR_prev * (window - 1) + TR) / window

            If previous ATR is not available, then ATR = 1/window * \sum_{i=1}^{window} TR_i

    ATR is not normalized, its value is greater than 0.
    ATR is typically used as a stop-loss indicator, and to determine the size of a position.
    """

    def __init__(self, window: int = 14):
        """
        Args:
            window (int): The window size for the ATR
        """
        self.window = window

    def is_action_strength_normalized(self) -> bool:
        """
        Method that returns whether the action strength is normalized, having values between [-1, 1].

        Returns:
            bool: Whether the action strength is normalized
        """
        return False
    
    def get_initial_intervals(self) -> int:
        return self.window
    
    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements ATR strategy.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        atr = self._get_atr(coin_data, self.window)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= self.get_initial_intervals():
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i + 1], atr.iloc[:i + 1])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date, 
            data={
                Actions.ACTION: actions,
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )
    
    ATR = 'ATR'

    def get_indicator(self, coin_data: DataFrame) -> DataFrame:
        """
        Function gets the ATR.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            DataFrame: The ATR
        """
        return self._get_atr(coin_data, self.window)

    def _get_atr(self, coin_data: DataFrame, window) -> DataFrame:
        """
        Function calculates the ATR.

        Args:
            coin_data (DataFrame): The coin data
            window (int): The window size for the ATR
        
        Returns:
            DataFrame: The ATR
            The first (window) values are 0.
        """
        atr = TA.ATR(coin_data, window)
        return DataFrame(
            index=atr.index,
            data={
                self.ATR: atr.values
            }
        )
    
    def _get_simple_action(self, coin_data: DataFrame, atr: DataFrame) -> Tuple[ActionSimple, float]:
        """
        Function gets the simple action based on the indicator.

        Args:
            coin_data (DataFrame): The coin data
            atr (DataFrame): The ATR

        Returns:
            ActionSimple: The action to take. In this case, the action is always HOLD.
            float: The indicator strength
        """
        return ActionSimple.HOLD, atr.iloc[-1][self.ATR]
