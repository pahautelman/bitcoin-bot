from actions.actions import Actions, ActionSimple
from agents.agent import Indicator
from finta import TA
from pandas import DataFrame
from typing import Tuple

class ObvAgent(Indicator):
    """
    Agent that implements *on balance volume* (OBV) strategy.
    OBV is a leading momentum indicator that uses volume flow to predict changes in price.
    
    OBV is calculated using the following steps:
        1. Calculate the OBV:
            OBV = OBV_prev + sign(close - close_prev) * volume
            close = current close
            close_prev = previous close
            if close = close_prev, then OBV = OBV_prev

            volume = current volume
        2. Calculate the OBV moving average over the window size.

    OBV measures positive and negative volume flow, offering insights into (smart/institutional) market players' intent.
    It produces predictions without specific information on what happened.

    Leading indicator prone to false signals; best used with lagging indicators.

    The signal strength is not bound between [-1, 1], but rather it accumulates over time.
    """

    def __init__(self, window: int = 5):
        """
        Args:
            window (int): The window size for the OBV simple moving average
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
        Function implements OBV strategy.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        obv = self.get_indicator(coin_data)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= self.get_initial_intervals():
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i + 1], obv.iloc[:i + 1])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date,
            data={
                Actions.ACTION: actions,
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )
    
    OBV = 'OBV'

    def get_indicator(self, coin_data: DataFrame) -> DataFrame:
        """
        Function returns the OBV for the given coin data.

        Args:
            coin_data (DataFrame): The coin data

        Returns:
            DataFrame: The OBV
        """
        return self._get_obv(coin_data)
    
    def _get_obv(self, coin_data: DataFrame) -> DataFrame:
        """
        Function calculates the OBV for the given coin data.

        Args:
            coin_data (DataFrame): The coin data

        Returns:
            DataFrame: The OBV
        """
        obv = TA.OBV(coin_data)
        return DataFrame(
            index=obv.index,
            data={
                self.OBV: obv.values
            }
        )
    
    def _get_simple_action(self, coin_data: DataFrame, obv_sma: DataFrame) -> Tuple[ActionSimple, int]:
        """
        Function gets the action to take based on the OBV.

        Args:
            coin_data (DataFrame): The coin data
            obv_sma (DataFrame): The OBV SMA

        Returns:
            ActionSimple: Always HOLD
            int: The OBV SMA value
        """
        return ActionSimple.HOLD, obv_sma.iloc[-1]
    