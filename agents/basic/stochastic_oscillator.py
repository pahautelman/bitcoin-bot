from actions.actions import Actions, ActionSimple
from agents.agent import Indicator
from finta import TA
from pandas import DataFrame
from typing import Tuple

class SoAgent(Indicator):
    """
    Agent implements the *stochastic oscillator* (SO) strategy. 
    SO is a leading momentum indicator that compares the closing price of an asset to its price range over a given period of time.
    It is used to identify overbought and oversold conditions.
    
    SO is more useful than RSI during sideways movements.

    SO is calculated as follows:
        1. Calculate the highest high and lowest low over the window size.
        2. Calculate the %K using the formula 
            %K = 100 * (close - lowest low) / (highest high - lowest low)
        3. Calculate the %D using the formula
            %D = 100 * SMA(%K, window=smoothing_window)
    
    An asset is considered overbought when the SO is above the overbought threshold.
    An asset is considered oversold when the SO is below the oversold threshold.
        * The overbought threshold is typically set to 80.
        * The oversold threshold is typically set to 20.

    SO provides signals for buying and selling:
        1. Buy when the SO crosses below the oversold threshold and then rises above it.
        2. Sell when the SO crosses above the overbought threshold and then falls below it.
    """

    def __init__(
            self, 
            window: int = 14, 
            smoothing_window: int = 3, 
            oversold: int = 20, 
            overbought: int = 80
        ):
        """
        Args:
            window (int): The window size for the SO
            smoothing_window (int): The smoothing window for the SO
            oversold (int): The oversold threshold for the SO
            overbought (int): The overbought threshold for the SO
        """
        assert oversold < overbought, "oversold must be less than overbought threshold"
        assert smoothing_window < window, "smoothing_window must be less than window"
        self.window = window
        self.smoothing_window = smoothing_window
        self.oversold = oversold
        self.overbought = overbought

    def is_action_strength_normalized(self) -> bool:
        """
        Method that returns whether the action strength is normalized, having values between [-1, 1].

        Returns:
            bool: Whether the action strength is normalized
        """
        return True
    
    def get_initial_intervals(self) -> int:
        return self.window

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements SO strategy.
        Buy when the SO crosses below the oversold threshold and then rises above it.
        Sell when the SO crosses above the overbought threshold and then falls below it.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        so = self.get_indicator(coin_data)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= self.get_initial_intervals():
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i + 1], so.iloc[:i + 1])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date,
            data={
                Actions.ACTION: actions,
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )
    
    SO = 'SO'

    def get_indicator(self, coin_data: DataFrame) -> DataFrame:
        """
        Function returns the SO for the given coin data.

        Args:
            coin_data (DataFrame): The coin data

        Returns:
            DataFrame: The SO
        """
        return self._get_so(coin_data, self.window, self.smoothing_window)
    
    def _get_so(self, coin_data: DataFrame, window: int, smoothing_window: int) -> DataFrame:
        """
        Function calculates the slow stochastic oscillator (SO) for the coin data.

        Args:
            coin_data (DataFrame): The coin data
            window (int): The window size for the SO
            smoothing_window (int): The smoothing window for the SO
        
        Returns:
            DataFrame: The SO for the coin data
        """
        so = TA.STOCHD(coin_data, period=smoothing_window, stoch_period=window)
        return DataFrame(
            index=so.index,
            data={
                self.SO: so.values
            }
        )
    
    def _get_simple_action(self, coin_data: DataFrame, so: DataFrame) -> Tuple[ActionSimple, int]:
        """
        Function returns the action to take based on the SO.

        Args:
            coin_data (DataFrame): The coin data
            so (DataFrame): The SO
        
        Returns:
            (ActionSimple, int): The action to take and the strength of the indicator
        """
        action = ActionSimple.HOLD
        # If the SO is below the oversold threshold and then rises above it, buy
        if so.iloc[-2][self.SO] < self.oversold and so.iloc[-1][self.SO] > self.oversold:
            action = ActionSimple.BUY
        # If the SO is above the overbought threshold and then falls below it, sell
        elif so.iloc[-2][self.SO] > self.overbought and so.iloc[-1][self.SO] < self.overbought:
            action = ActionSimple.SELL
        
        indicator_strength = (so.iloc[-1][self.SO] - 50) / 50
        return action, indicator_strength
