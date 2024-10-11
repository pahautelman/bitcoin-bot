from actions.actions import Actions, ActionSimple
from agents.basic.moving_average_convergence_divergence import MacdAgent
from agents.basic.relative_strength_index import RsiAgent
from pandas import DataFrame
from typing import Tuple

class MacdRsiAgent(MacdAgent, RsiAgent):
    """
    Agent that implements *moving average convergence divergence* (MACD) and *relative strength index* (RSI) strategy.

    Buy when the MACD line crosses above the signal line and the RSI crosses below the oversold threshold and then rises above it.
    Sell when the MACD line crosses below the signal line and the RSI crosses above the overbought threshold and then falls below it.
    """

    def __init__(self, macd_fast_period: int = 12, macd_slow_period: int = 26, macd_signal_period: int = 9, rsi_window: int = 14, rsi_smoothing_window: int = 3, rsi_oversold: int = 30, rsi_overbought: int = 70):
        """
        Args:
            macd_fast_period (int): The fast period for the MACD
            macd_slow_period (int): The slow period for the MACD
            macd_signal_period (int): The signal period for the MACD
            rsi_window (int): The window size for the RSI
            rsi_smoothing_window (int): The smoothing window for the RSI
            rsi_oversold (int): The oversold threshold for the RSI
            rsi_overbought (int): The overbought threshold for the RSI
        """
        MacdAgent.__init__(self, macd_fast_period, macd_slow_period, macd_signal_period)
        RsiAgent.__init__(self, rsi_window, rsi_smoothing_window, rsi_oversold, rsi_overbought)

    def get_initial_intervals(self) -> int:
        return max(MacdAgent.get_initial_intervals(self), RsiAgent.get_initial_intervals(self))

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements MACD and RSI strategy.

        Buy when the MACD line crosses above the signal line and the RSI crosses below the oversold threshold and then rises above it.
        Sell when the MACD line crosses below the signal line and the RSI crosses above the overbought threshold and then falls below it.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        macd = MacdAgent.get_indicator(self, coin_data)
        rsi = RsiAgent.get_indicator(self, coin_data)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= self.get_initial_intervals():
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i + 1], macd.iloc[:i + 1], rsi.iloc[:i + 1])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date, 
            data={
                Actions.ACTION: actions,
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )
    
    def _get_simple_action(self, coin_data: DataFrame, macd: DataFrame, rsi: DataFrame) -> Tuple[ActionSimple, int]:
        """
        Function calculates the action to take based on the MACD and RSI

        Args:
            coin_data (DataFrame): The coin data
            macd (DataFrame): The MACD data
            rsi (DataFrame): The RSI data

        Returns:
            Tuple[ActionSimple, int]: The action to take and the strength of the indicator
        """
        macd_action, macd_strength = MacdAgent._get_simple_action(self, coin_data, macd)
        rsi_action, _ = RsiAgent._get_simple_action(self, coin_data, rsi)

        if macd_strength > 0 and rsi_action == ActionSimple.BUY:
            return ActionSimple.BUY, 1
        elif macd_strength < 0 and rsi_action == ActionSimple.SELL:
            return ActionSimple.SELL, -1
        return ActionSimple.HOLD, 0
