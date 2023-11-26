from pandas import DataFrame
from actions.actions import Actions, ActionSimple
from agents.agent import Agent

class AtrAgent(Agent):
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

    ATR is typically used as a stop-loss indicator, and to determine the size of a position.
    """

    def __init__(self, window: int = 14):
        """
        Args:
            window (int): The window size for the ATR
        """
        self.window = window
    
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
            if i <= self.window:
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i + 1], atr.iloc[:i + 1])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            action_date=action_date,
            actions=actions,
            indicator_values=indicator_values,
            indicator_name='ATR'
        )
    
    ATR = 'atr'

    def _get_atr(self, coin_data: DataFrame, window: int = 14) -> DataFrame:
        """
        Function calculates the ATR.

        Args:
            coin_data (DataFrame): The coin data
            window (int): The window size for the ATR
        
        Returns:
            DataFrame: The ATR
            The first (window) values are 0.
        """
        # calculate true range for the first window
        tr = []
        atr = []
        for i in range(window):
            high = coin_data['high'].iloc[i]
            low = coin_data['low'].iloc[i]
            if i == 0:
                close_prev = coin_data['close'].iloc[i]
            else:
                close_prev = coin_data['close'].iloc[i-1]
            tr.append(max(high-low, abs(high-close_prev), abs(low-close_prev)))
            atr.append(0)

        # calculate ATR
        atr.append(sum(tr)/window)
        for i in range(window+1, len(coin_data)):
            high = coin_data['high'].iloc[i]
            low = coin_data['low'].iloc[i]
            close_prev = coin_data['close'].iloc[i-1]
            tr.append(max(high-low, abs(high-close_prev), abs(low-close_prev)))
            atr.append((atr[i-1]*(window-1) + tr[i]) / window)

        return DataFrame(
            data={
                self.ATR: atr
            }
        )
    
    def _get_simple_action(self, coin_data: DataFrame, atr: DataFrame) -> (ActionSimple, float):
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
