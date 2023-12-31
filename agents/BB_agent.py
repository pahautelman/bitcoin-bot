from agents.agent import Agent
from pandas.core.frame import DataFrame
from actions.actions import Actions, ActionSimple

class BB_agent(Agent):
    """
    Agent that implements Bollinger bands strategy.

    Buy when the price touches or falls below the lower BB and then rises back inside the bands.
    Sell when the price touches or exceeds the upper BB and then falls back inside the bands.
    """

    def __init__(self, bb_window: int, bb_std: int):
        """
        Args:
            bb_window (int): The window size for the BB
            bb_std (int): The number of standard deviations for the BB
        """
        self.bb_window = bb_window
        self.bb_std = bb_std

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
        bands = self._get_bollinger_bands(coin_data, window=self.bb_window, std=self.bb_std)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(0, len(coin_data)):
            if i < self.bb_window:
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i], bands.iloc[:i])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(index=action_date, data={Actions.ACTION: actions, Actions.INDICATOR_STRENGTH: indicator_values})

    UPPER_BAND = 'upper_band'
    LOWER_BAND = 'lower_band'
    ROLLING_MEAN = 'rolling_mean'

    def _get_bollinger_bands(self, coin_data: DataFrame, window: int=20, std: int=2) -> DataFrame:
        """
        Function calculates the bollinger bands.

        Args:
            coin_data (DataFrame): The coin data
            window (int): The window size for the BB
            std (int): The number of standard deviations for the BB

        Returns:
            DataFrame: The bollinger bands
        """
        rolling_mean = coin_data['Close'].rolling(window=window).mean()
        rolling_std = coin_data['Close'].rolling(window=window).std()

        upper_band = rolling_mean + (rolling_std * std)
        lower_band = rolling_mean - (rolling_std * std)

        return DataFrame(data={
            self.UPPER_BAND: upper_band,
            self.LOWER_BAND: lower_band,
            self.ROLLING_MEAN: rolling_mean
        })

    def _get_simple_action(self, coin_data: DataFrame, bands: DataFrame) -> (ActionSimple, int):
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
        action = ActionSimple.HOLD
        # if price is inside the bands
        if bands.iloc[-1][self.LOWER_BAND] < coin_data.iloc[-1]['Close'] < bands.iloc[-1][self.UPPER_BAND]:
            # and it previously was above the upper band
            if coin_data.iloc[-2]['Close'] > bands.iloc[-2][self.UPPER_BAND]:
                # then sell
                action = ActionSimple.SELL
            # if it previously was below the lower band
            elif coin_data.iloc[-2]['Close'] < bands.iloc[-2][self.LOWER_BAND]:
                # then buy
                action = ActionSimple.BUY
            # calculate indicator strength
            if coin_data.iloc[-1]['Close'] > bands.iloc[-1][self.ROLLING_MEAN]:
                indicator_strength = -(coin_data.iloc[-1]['Close'] - bands.iloc[-1][self.ROLLING_MEAN]) / (bands.iloc[-1][self.UPPER_BAND] - bands.iloc[-1][self.ROLLING_MEAN]) 
            else:
                indicator_strength = (coin_data.iloc[-1]['Close'] - bands.iloc[-1][self.ROLLING_MEAN]) / (bands.iloc[-1][self.LOWER_BAND] - bands.iloc[-1][self.ROLLING_MEAN])
        # if price is above the upper band
        elif coin_data.iloc[-1]['Close'] >= bands.iloc[-1][self.UPPER_BAND]:
            indicator_strength = -1
        else:
            indicator_strength = 1

        return action, indicator_strength     
