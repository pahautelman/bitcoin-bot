from pandas import DataFrame
from actions.actions import Actions, ActionSimple
from agents.basic.dual_moving_average_crossover import DmacAgent
from agents.basic.relative_strength_index import RsiAgent

class DmacRsiAgent(DmacAgent, RsiAgent):
    """
    Class implements a SMA and RSI strategy.

    Buy when the short-term SMA crosses above the long-term SMA and the RSI crosses below the oversold threshold and then rises above it.
    Sell when the short-term SMA crosses below the long-term SMA and the RSI crosses above the overbought threshold and then falls below it.
    """

    def __init__(self, fast_period: int, slow_period: int, window: int, oversold: float, overbought: float):
        """
        Args:
            fast_period (int): The fast period for the SMA
            slow_period (int): The slow period for the SMA
            window (int): The window for the RSI agent
            oversold (float): The oversold threshold for the RSI
            overbought (float): The overbought threshold for the RSI
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.window = window
        self.oversold = oversold
        self.overbought = overbought

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements SMA and RSI strategy.
        Buy when the price crosses above the SMA and the RSI crosses below the oversold threshold and then rises above it.
        Sell when the price crosses below the SMA and the RSI crosses above the overbought threshold and then falls below it.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        dmac = self._get_dmac(coin_data, self.fast_period, self.slow_period)
        rsi = self._get_rsi(coin_data, self.window)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= self.window or i <= self.slow_period:
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i], dmac.iloc[:i], rsi.iloc[:i])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date, 
            data={
                Actions.ACTION: actions,
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )
    
    def _get_simple_action(self, coin_data: DataFrame, dmac: DataFrame, rsi: DataFrame)-> (ActionSimple, int):
        """
        Function gets the action to take based on the DMAC and RSI.

        Args:
            coin_data (DataFrame): The coin data
            dmac (DataFrame): The DMAC
            rsi (DataFrame): The RSI
        
        Returns:
            (ActionSimple, int): The action to take and the indicator strength
        """
        _, dmac_strength = DmacAgent._get_simple_action(self, coin_data, dmac)
        rsi_action, _ = RsiAgent._get_simple_action(self, coin_data, rsi)

        if (dmac_strength == 1 and rsi.iloc[-1][self.RSI] < self.oversold) or rsi_action == ActionSimple.BUY:
            return ActionSimple.BUY, 1
        elif (dmac_strength == -1 and rsi.iloc[-1][self.RSI] > self.overbought) or rsi_action == ActionSimple.SELL:
            return ActionSimple.SELL, -1
        else:
            return ActionSimple.HOLD, 0
