from pandas import DataFrame
from actions.actions import Actions, ActionSimple
from agents.basic.simple_moving_average import SmaAgent

# TODO: implement as difference between the fast moving exponential average and the slow moving exponential average
# TODO: probably don't need indicator for trading agent, simply pass the 50 and 200 EMAs
class DmacAgent(SmaAgent):
    """
    Agent that implements *dual moving average crossover* (DMAC) strategy.

    Buy when the short-term SMA crosses above the long-term SMA.
    Sell when the short-term SMA crosses below the long-term SMA.
    """

    DMAC = 'dmac'

    def __init__(self, fast_period: int, slow_period: int):
        """
        Args:
            fast_period (int): The fast period for the SMA
            slow_period (int): The slow period for the SMA
        """
        self.fast_period = fast_period
        self.slow_period = slow_period

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements DMAC strategy.
        Buy when the short-term SMA crosses above the long-term SMA.
        Sell when the short-term SMA crosses below the long-term SMA.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        dmac = self._get_dmac(coin_data, self.fast_period, self.slow_period)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= self.slow_period:
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(dmac.iloc[:i + 1])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date, 
            data={
                Actions.ACTION: actions,
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )
    
    def _get_dmac(self, coin_data: DataFrame, fast_period: int, slow_period: int) -> DataFrame:
        """
        Function calculates the DMAC.

        Args:
            coin_data (DataFrame): The coin data
            fast_period (int): The fast period for the SMA
            slow_period (int): The slow period for the SMA
        
        Returns:
            DataFrame: The DMAC
        """
        fast_sma = self._get_sma(coin_data, fast_period)
        slow_sma = self._get_sma(coin_data, slow_period)
        
        return DataFrame(
            data={
                self.DMAC: fast_sma[self.SMA] - slow_sma[self.SMA]
            }
        )

    def _get_simple_action(self, coin_data: DataFrame, dmac: DataFrame) -> (ActionSimple, int):
        """
        Function calculates the action to take based on the DMAC.

        Args:
            coin_data (DataFrame): The coin data
            dmac (DataFrame): The DMAC

        Returns:
            ActionSimple: The action to take
            int: The indicator strength
        """
        action = ActionSimple.HOLD
        # buy when the short-term SMA crosses above the long-term SMA
        if dmac.iloc[-2][self.DMAC] < 0 and dmac.iloc[-1][self.DMAC] > 0:
            action = ActionSimple.BUY
        # sell when the short-term SMA crosses below the long-term SMA
        elif dmac.iloc[-2][self.DMAC] > 0 and dmac.iloc[-1][self.DMAC] < 0:
            action = ActionSimple.SELL
        
        indicator_strength = 1 if dmac.iloc[-1][self.DMAC] > 0 else -1
        return action, indicator_strength
