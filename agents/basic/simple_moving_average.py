from pandas import DataFrame
from actions.actions import Actions, ActionSimple
from agents.agent import Agent


class SmaAgent(Agent):
    """
    Agent that implements *simple moving average* (SMA) strategy.

    Buy when the price crosses above the SMA.
    Sell when the price crosses below the SMA.
    """

    def __init__(self, window: int = 50):
        """
        Args:
            window (int): The window size for the SMA
        """
        self.window = window

    def is_action_strength_normalized(self) -> bool:
        """
        Method that returns whether the action strength is normalized, having values between [-1, 1].

        Returns:
            bool: Whether the action strength is normalized
        """
        return False

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements SMA strategy.
        Buy when the price crosses above the SMA.
        Sell when the price crosses below the SMA.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        sma = self._get_sma(coin_data, self.window)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= self.window:
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i + 1], sma.iloc[:i + 1])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date, 
            data={
                Actions.ACTION: actions,
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )

    SMA = 'sma'

    def get_indicator(self, coin_data: DataFrame) -> DataFrame:
        """
        Function returns the SMA for the given coin data.

        Args:
            coin_data (DataFrame): The coin data

        Returns:
            DataFrame: The SMA
        """
        return self._get_sma(coin_data, self.window)

    def _get_sma(self, coin_data: DataFrame, window: int=14) -> DataFrame:
        """
        Function calculates the SMA.

        Args:
            coin_data (DataFrame): The coin data
            window (int): The window size for the SMA
        
        Returns:
            DataFrame: The SMA
        """
        return DataFrame(
            index=coin_data.index,
            data={
                self.SMA: coin_data['Close'].rolling(window=window).mean()
            }
        )

    def _get_simple_action(self, coin_data: DataFrame, sma: DataFrame) -> (ActionSimple, int):
        """
        Function calculates the action to take based on the SMA.

        Args:
            coin_data (DataFrame): The coin data
            sma (DataFrame): The SMA
        
        Returns:
            ActionSimple: The action to take
        """
        if coin_data.iloc[-1]['Close'] >= sma.iloc[-1][self.SMA]:
            return ActionSimple.BUY, 1
        else:
            return ActionSimple.SELL, -1
            