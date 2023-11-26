from pandas import DataFrame
from actions.actions import Actions, ActionSimple
from agents.agent import Agent

class CmfAgent(Agent):
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

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements CMF strategy.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        cmf = self._get_cmf(coin_data, self.window)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= self.window:
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i + 1], cmf.iloc[:i + 1])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            action_date=action_date,
            actions=actions,
            indicator_values=indicator_values
        )
    
    def _get_cmf(self, coin_data: DataFrame, window: int) -> DataFrame:
        """
        Function calculates the CMF for the given coin data.

        Args:
            coin_data (DataFrame): The coin data
            window (int): The window size for the CMF
        """
        mfm = ((coin_data['close'] - coin_data['low']) - (coin_data['high'] - coin_data['close'])) / (coin_data['high'] - coin_data['low'])
        mfv = mfm * coin_data['volume']
        cmf = mfv.rolling(window).sum() / coin_data['volume'].rolling(window).sum()
        return cmf
    
    def _get_simple_action(self, coin_data: DataFrame, cmf: DataFrame) -> (ActionSimple, float):
        """
        Function returns the action and the indicator strength for the given coin data and indicator.

        Args:
            coin_data (DataFrame): The coin data
            indicator (DataFrame): The indicator
        
        Returns:
            ActionSimple: Always HOLD
            int: The CMF value
        """
        return ActionSimple.HOLD, cmf.iloc[-1]
    