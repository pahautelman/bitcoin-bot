from pandas import DataFrame
from actions.actions import Actions, ActionSimple
from agents.agent import Agent

class RSI_agent(Agent):
    """
    Agent that implements *relative strength index* (RSI) strategy.

    Buy when the RSI crosses below the oversold threshold and then rises above it.
    Sell when the RSI crosses above the overbought threshold and then falls below it.
    """

    def __init__(self, window: int, oversold: int, overbought: int):
        """
        Args:
            window (int): The window size for the RSI
            oversold (int): The oversold threshold for the RSI
            overbought (int): The overbought threshold for the RSI
        """
        self.window = window
        self.oversold = oversold
        self.overbought = overbought

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements RSI strategy.
        Buy when the RSI crosses below the oversold threshold and then rises above it.
        Sell when the RSI crosses above the overbought threshold and then falls below it.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        rsi = self._get_rsi(coin_data, self.window)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= self.window:
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue
            action, indicator_strength = self._get_simple_action(rsi.iloc[:i])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date, 
            data={
                Actions.ACTION: actions,
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )

    RSI = 'rsi'

    def _get_rsi(self, coin_data: DataFrame, window: int=14) -> DataFrame:
        """
        Function calculates the RSI.

        Args:
            coin_data (DataFrame): The coin data
            window (int): The window size for the RSI

        Returns:
            DataFrame: The RSI
        """
        delta = coin_data['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=window - 1, adjust=True, min_periods=window).mean()
        ema_down = down.ewm(com=window - 1, adjust=True, min_periods=window).mean()
        rs = ema_up / ema_down
        rsi = 100 - (100 / (1 + rs))
        return DataFrame(data={
            self.RSI: rsi
        })

    def _get_simple_action(self, rsi: DataFrame) -> (ActionSimple, int):
        """
        Function returns the simple action based on the RSI.

        Args:
            rsi (DataFrame): The RSI

        Returns:
            ActionSimple: The action to take
            int: The indicator strength            
        """
        action = ActionSimple.HOLD
        indicator_strength = 0
        
        # if rsi line is above overbought threshold
        if rsi.iloc[-1][self.RSI] > self.overbought:
            action = ActionSimple.SELL
            indicator_strength = -1
        # if its below oversold threshold
        elif rsi.iloc[-1][self.RSI] < self.oversold:
            action = ActionSimple.BUY
            indicator_strength = 1
        else:
            action = ActionSimple.HOLD
            # calculate indicator strength based on how far away from the threshold it is
            if rsi.iloc[-1][self.RSI] > 50:
                indicator_strength = -(rsi.iloc[-1][self.RSI] - 50) / (self.overbought - 50)
            else:
                indicator_strength = (rsi.iloc[-1][self.RSI] - 50) / (self.oversold - 50)
        return action, indicator_strength
