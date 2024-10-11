from actions.actions import Actions, ActionSimple
from agents.basic.bollinger_bands import BbAgent
from agents.basic.relative_strength_index import RsiAgent
from pandas import DataFrame
from typing import Tuple

class BbRsiAgent(BbAgent, RsiAgent):
    """
    Agent that implements *Bollinger Bands* (BB) and *relative strength index* (RSI) strategy.

    Buy when the price crosses below the lower band of the Bollinger Bands and the RSI crosses below the oversold threshold and then rises above it.
    Sell when the price crosses above the upper band of the Bollinger Bands and the RSI crosses above the overbought threshold and then falls below it.
    """

    def __init__(self, bb_window: int = 20, bb_std: int = 2, rsi_window: int = 14, rsi_smoothing_window: int = 3, rsi_oversold: int = 30, rsi_overbought: int = 70):
        """
        Args:
            bb_window (int): The window size for the BB
            bb_std (int): The number of standard deviations for the BB
            rsi_window (int): The window size for the RSI
            rsi_smoothing_window (int): The smoothing window for the RSI
            rsi_oversold (int): The oversold threshold for the RSI
            rsi_overbought (int): The overbought threshold for the RSI
        """
        BbAgent.__init__(self, bb_window, bb_std)
        RsiAgent.__init__(self, rsi_window, rsi_smoothing_window, rsi_oversold, rsi_overbought)

    def get_initial_intervals(self) -> int:
        return max(BbAgent.get_initial_intervals(self), RsiAgent.get_initial_intervals(self))

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements Bollinger Bands and RSI strategy.

        Buy when the price crosses below the lower band of the Bollinger Bands and the RSI crosses below the oversold threshold and then rises above it.
        Sell when the price crosses above the upper band of the Bollinger Bands and the RSI crosses above the overbought threshold and then falls below it.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        bb = BbAgent.get_indicator(self, coin_data)
        rsi = RsiAgent.get_indicator(self, coin_data)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= self.get_initial_intervals():
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i + 1], bb.iloc[:i + 1], rsi.iloc[:i + 1])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date, 
            data={
                Actions.ACTION: actions, 
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )
    
    def _get_simple_action(self, coin_data: DataFrame, bb: DataFrame, rsi: DataFrame) -> Tuple[ActionSimple, int]:
        """
        Function calculates the action to take based on the Bollinger Bands and RSI

        Args:
            coin_data (DataFrame): The coin data
            bb (DataFrame): The Bollinger Bands
            rsi (DataFrame): The RSI

        Returns:
            Tuple[ActionSimple, int]: The action to take and the indicator strength
        """
        bb_action, bb_strength = BbAgent._get_simple_action(self, coin_data, bb)
        rsi_action, rsi_strength = RsiAgent._get_simple_action(self, coin_data, rsi)

        action = ActionSimple.HOLD
        if bb_strength == -1 and rsi_action == ActionSimple.SELL:
            action = ActionSimple.SELL
        elif bb_strength == 1 and rsi_action == ActionSimple.BUY:
            action = ActionSimple.BUY
        indicator_strength = (bb_strength + rsi_strength) / 2
        return action, indicator_strength
    