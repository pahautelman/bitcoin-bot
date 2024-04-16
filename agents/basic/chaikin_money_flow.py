from actions.actions import Actions, ActionSimple
from agents.agent import Indicator
from pandas import DataFrame
from typing import Tuple

class CmfAgent(Indicator):
    """
    Agent that implements *Chaikin money flow* (CMF) strategy.

    CMF is a leading momentum oscillator that combines price and volume to assess the flow of money into or out of the security.
    CMF is particularly useful for identifying potential trend reversals and confirming the strength of a trend.

    CMF is calculated using the following steps:
        1. Calculate the money flow multiplier (MFM) as follows:
            MFM = ((close - low) - (high - close)) / (high - low)
        2. Calculate the money flow volume (MFV) as follows:
            MFV = MFM * volume
        3. Calculate the CMF as follows:
            CMF = sum(MFV) / sum(volume)
            sum is calculated over the window size.

    CMF oscillates between -1 and 1.
    A CMF value above 0 indicates that the security is under accumulation.
    """

    def __init__(self, window: int = 20):
        """
        Args:
            window (int): The window size for the CMF
        """
        self.window = window

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
        Function implements CMF strategy.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        cmf = self.get_indicator(coin_data)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= self.get_initial_intervals():
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i + 1], cmf.iloc[:i + 1])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date,
            data={
                Actions.ACTION: actions,
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )
    
    CMF = 'CMF'

    def get_indicator(self, coin_data: DataFrame) -> DataFrame:
        """
        Method that returns the CMF indicator.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            DataFrame: The CMF indicator
        """
        return self._get_cmf(coin_data, self.window)
    
    def _get_cmf(self, coin_data: DataFrame, window: int) -> DataFrame:
        """
        Function calculates the CMF for the given coin data.

        Args:
            coin_data (DataFrame): The coin data
            window (int): The window size for the CMF
        """
        mfm = ((coin_data['Close'] - coin_data['Low']) - (coin_data['High'] - coin_data['Close'])) / (coin_data['High'] - coin_data['Low'])
        mfv = mfm * coin_data['Volume']
        cmf = mfv.rolling(window).sum() / coin_data['Volume'].rolling(window).sum()
        
        return DataFrame(
            index=coin_data.index,
            data={
                self.CMF: cmf
            }
        )
    
    def _get_simple_action(self, coin_data: DataFrame, cmf: DataFrame) -> Tuple[ActionSimple, float]:
        """
        Function returns the action and the indicator strength for the given coin data and indicator.

        Args:
            coin_data (DataFrame): The coin data
            indicator (DataFrame): The indicator
        
        Returns:
            ActionSimple: Always HOLD
            int: The CMF value
        """
        return ActionSimple.HOLD, cmf.iloc[-1][self.CMF]
    