from actions.actions import Actions, ActionSimple
from agents.agent import Indicator
from finta import TA
from pandas.core.frame import DataFrame
from typing import Tuple

class BbAgent(Indicator):
    """
    Agent that implements Bollinger bands (BB) strategy.
    BB is lagging mean reversion indicator. It shows overbought/oversold conditions relative to past price action, 
    taking volatility into account.
    
    BB is calculated using the following formula:
        1. Middle Band = bb_window simple moving average (SMA)
        2. Upper Band = Middle Band + bb_std standard deviations
        3. Lower Band = Middle Band - bb_std standard deviations

    The upper and lower bands are based on the standard deviation, which is a measure of volatility.
    The bands widen when volatility increases and narrow when volatility decreases.
    When the price moves closer to the upper band, the asset is becoming overbought.
    When the price moves closer to the lower band, the asset is becoming oversold.
    The price tends to return to the middle band after touching the upper or lower band.
    
    The bands provide signals for buying and selling:
        1. Buy when the price touches or falls below the lower BB and then rises back inside the bands.
        2. Sell when the price touches or exceeds the upper BB and then falls back inside the bands.
    """

    def __init__(self, bb_window: int = 20, bb_std: int = 2):
        """
        Args:
            bb_window (int): The window size for the BB
            bb_std (int): The number of standard deviations for the BB
        """
        self.bb_window = bb_window
        self.bb_std = bb_std

    def is_action_strength_normalized(self) -> bool:
        """
        Method that returns whether the action strength is normalized, having values between [-1, 1].

        Returns:
            bool: Whether the action strength is normalized
        """
        return True
    
    def get_initial_intervals(self) -> int:
        return self.bb_window

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements Bollinger Bands strategy.
        Buy when the price touches or falls below the lower BB and then rises back inside the bands.
        Sell when the price touches or exceeds the upper BB and then falls back inside the bands.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        bands = self.get_indicator(coin_data)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(0, len(coin_data)):
            if i < self.get_initial_intervals():
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i + 1], bands.iloc[:i + 1])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date, 
            data={
                Actions.ACTION: actions, 
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )

    UPPER_BAND = 'upper_band'
    LOWER_BAND = 'lower_band'
    ROLLING_MEAN = 'rolling_mean'

    def get_indicator(self, coin_data: DataFrame) -> DataFrame:
        """
        Method that returns the Bollinger bands indicator.

        Args:
            coin_data (DataFrame): The coin data

        Returns:
            DataFrame: The Bollinger bands
        """
        return self._get_bollinger_bands(coin_data, window=self.bb_window, std=self.bb_std)

    def _get_bollinger_bands(self, coin_data: DataFrame, window: int, std: int) -> DataFrame:
        """
        Function calculates the bollinger bands.

        Args:
            coin_data (DataFrame): The coin data
            window (int): The window size for the BB
            std (int): The number of standard deviations for the BB

        Returns:
            DataFrame: The bollinger bands
        """
        bbands = TA.BBANDS(coin_data, period=window, std_multiplier=std)
        return DataFrame(
            index=bbands.index,
            data={
                self.UPPER_BAND: bbands['BB_UPPER'].values,
                self.LOWER_BAND: bbands['BB_LOWER'].values,
                self.ROLLING_MEAN: bbands['BB_MIDDLE'].values,
            }
        )

    def _get_simple_action(self, coin_data: DataFrame, bands: DataFrame) -> Tuple[ActionSimple, int]:
        """
        Function return instantaneous Bollinger bands strategy.
        Buy when the price touches or falls below the lower BB and then rises back inside the bands.
        Sell when the price touches or exceeds the upper BB and then falls back inside the bands.

        Args:
            coin_data (DataFrame): The coin data
            bands (DataFrame): The bollinger bands

        Returns:
            ActionSimple: The action to take
            int: The indicator strength
        """
        current_price = coin_data.iloc[-1]['Close']
        upper_band = bands.iloc[-1][self.UPPER_BAND]
        lower_band = bands.iloc[-1][self.LOWER_BAND]
        mean_band = bands.iloc[-1][self.ROLLING_MEAN]
        previous_price = coin_data.iloc[-2]['Close']

        if lower_band <= current_price <= upper_band:
            if previous_price > upper_band:
                action = ActionSimple.SELL
            elif previous_price < lower_band:
                action = ActionSimple.BUY
            else:
                action = ActionSimple.HOLD

            if current_price > mean_band:
                indicator_strength = -(current_price - mean_band) / (upper_band - mean_band)
            else:
                indicator_strength = (current_price - mean_band) / (lower_band - mean_band)
        elif current_price >= upper_band:
            action = ActionSimple.SELL
            indicator_strength = -1
        else:
            action = ActionSimple.BUY
            indicator_strength = 1

        return action, indicator_strength 
