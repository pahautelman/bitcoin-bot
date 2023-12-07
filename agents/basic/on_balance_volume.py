from pandas import DataFrame
from actions.actions import Actions, ActionSimple
from agents.agent import Agent

class ObvAgent(Agent):
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

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements OBV strategy.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        obv = self._get_obv(coin_data, self.window)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= self.window:
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
    
    OBV = 'obv'

    def get_indicator(self, coin_data: DataFrame) -> DataFrame:
        """
        Function returns the OBV for the given coin data.

        Args:
            coin_data (DataFrame): The coin data

        Returns:
            DataFrame: The OBV
        """
        obv = self._get_obv(coin_data, sma_window=self.window)
        # apply pct_change to the OBV SMA
        obv[self.OBV] = obv[self.OBV].pct_change()
        # limit the values to [-1, 1]
        obv[self.OBV] = obv[self.OBV].clip(lower=-1, upper=1) 
        return obv
    
    def _get_obv(self, coin_data: DataFrame, sma_window: int) -> DataFrame:
        """
        Function calculates the OBV for the given coin data.

        Args:
            coin_data (DataFrame): The coin data
            sma_window (int): The window size for the OBV SMA

        Returns:
            DataFrame: The OBV
        """
        obv = coin_data['Close'].diff()
        obv[obv > 0] = coin_data['Volume']
        obv[obv < 0] = -coin_data['Volume']
        obv[obv == 0] = 0
        obv = obv.cumsum()
        obv = self._get_sma(obv, window=sma_window)

        return DataFrame(
            index=coin_data.index,
            data={
                self.OBV: obv
            }
        )
    
    def _get_sma(self, obv: DataFrame, window: int) -> DataFrame:
        """
        Function calculates the simple moving average for the given OBV.

        Args:
            obv (DataFrame): The OBV
            window (int): The window size for the SMA

        Returns:
            DataFrame: The SMA
        """
        return obv.rolling(window=window).mean()
    
    def _get_simple_action(self, coin_data: DataFrame, obv_sma: DataFrame) -> (ActionSimple, int):
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