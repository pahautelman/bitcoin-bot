from pandas import DataFrame
from actions.actions import Actions, ActionSimple
from agents.basic.moving_average_convergence_divergence import MacdAgent
from agents.basic.relative_strength_index import RsiAgent

class MacdRsiAgent(MacdAgent, RsiAgent):
    """
    Agent that implements *moving average convergence divergence* (MACD) and *relative strength index* (RSI) strategy.

    Buy when the MACD line crosses above the signal line and the RSI crosses below the oversold threshold and then rises above it.
    Sell when the MACD line crosses below the signal line and the RSI crosses above the overbought threshold and then falls below it.
    """

    def __init__(self, fast_period: int, slow_period: int, signal_period: int, window: int, oversold: int, overbought: int):
        """
        Args:
            fast_period (int): The fast period for the MACD
            slow_period (int): The slow period for the MACD
            signal_period (int): The signal period for the MACD
            window (int): The window size for the RSI
            oversold (int): The oversold threshold for the RSI
            overbought (int): The overbought threshold for the RSI
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.window = window
        self.oversold = oversold
        self.overbought = overbought

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements DMAC and RSI strategy.
        Buy when the MACD line crosses above the signal line and the RSI crosses below the oversold threshold and then rises above it.
        Sell when the MACD line crosses below the signal line and the RSI crosses above the overbought threshold and then falls below it.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        macd = self._get_macd(coin_data, self.fast_period, self.slow_period, self.signal_period)
        rsi = self._get_rsi(coin_data, self.window)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= self.slow_period or i <= self.window:
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i], macd.iloc[:i], rsi.iloc[:i])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date, 
            data={
                Actions.ACTION: actions,
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )
    
    def _get_simple_action(self, coin_data: DataFrame, macd: DataFrame, rsi: DataFrame) -> ActionSimple:
        """
        Function calculates the action to take based on the MACD and RSI.

        Args:
            coin_data (DataFrame): The coin data
            macd (DataFrame): The MACD
            rsi (DataFrame): The RSI

        Returns:
            ActionSimple: The action to take
        """
        macd_action, macd_strength = MacdAgent._get_simple_action(self, coin_data, macd)
        rsi_action, rsi_strength = RsiAgent._get_simple_action(self, coin_data, rsi)

        action = ActionSimple.HOLD
        indicator_strength = 0
        if (macd_action == ActionSimple.BUY and rsi_strength > 0.15) or rsi_action == ActionSimple.BUY:
            action = ActionSimple.BUY
            indicator_strength = 1
        elif (macd_strength == ActionSimple.SELL and rsi_strength < -0.15) or rsi_action == ActionSimple.SELL:
            action = ActionSimple.SELL
            indicator_strength = -1
        
        return action, indicator_strength
