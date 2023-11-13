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

        action_date = []
        actions = []
        for i in range(1, len(coin_data)):
            if i < self.bb_window:
                continue

            action = self._get_simple_action(coin_data.iloc[:i], bands.iloc[:i])
            action_date.append(coin_data.index[i])
            actions.append(action)

        return Actions(index=action_date, data={Actions.ACTION: actions})

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

    def _get_simple_action(self, coin_data: DataFrame, bands: DataFrame) -> ActionSimple:
        """
        Function return instantaneous Bollinger bands strategy.
        Buy when the price touches or falls below the lower BB and then rises back inside the bands.
        Sell when the price touches or exceeds the upper BB and then falls back inside the bands.

        Args:
            coin_data (DataFrame): The coin data
            bands (DataFrame): The bollinger bands

        Returns:
            ActionSimple: The action to take
        """
        # if price is inside the bands
        if bands.iloc[-1][self.LOWER_BAND] < coin_data.iloc[-1]['Close'] < bands.iloc[-1][self.UPPER_BAND]:
            # and it previously was above the upper band
            if coin_data.iloc[-2]['Close'] > bands.iloc[-2][self.UPPER_BAND]:
                # then sell
                return ActionSimple.SELL
            # if it previously was below the lower band
            elif coin_data.iloc[-2]['Close'] < bands.iloc[-2][self.LOWER_BAND]:
                # then buy
                return ActionSimple.BUY
        return ActionSimple.HOLD            
