from pandas import DataFrame
from actions.actions import Actions, ActionSimple
from agents.agent import Agent

class RsiAgent(Agent):
    """
    Agent that implements *relative strength index* (RSI) strategy.
    RSI is a leading momentum oscillator that measures the speed and change of price movements.

    RSI is calculated as follows:
        1. Calculate average gain and average loss over the window size.
        2. Calculate the relative strength (RS) as the ratio of the average gain to the average loss.
        3. Calculate the RSI using the formula RSI = 100 - (100 / (1 + RS))
    
    RSI oscillates between 0 and 100.
    An asset is considered overbought when the RSI is above the overbought threshold.
    An asset is considered oversold when the RSI is below the oversold threshold.
        * The overbought threshold is typically set to 70.
        * The oversold threshold is typically set to 30.

    RSI provides signals for buying and selling:
        1. RSI values above the overbought threshold may suggest that the asset is overbought, and a potential reversal may occur
        2. RSI values below the oversold threshold may suggest that the asset is oversold.
    
        Buy when the RSI crosses below the oversold threshold and then rises above it.
        Sell when the RSI crosses above the overbought threshold and then falls below it.
    """

    def __init__(
            self, 
            window: int = 14, 
            oversold: int = 30, 
            overbought: int = 70
        ):
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
            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i + 1], rsi.iloc[:i + 1])
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

    def _get_simple_action(self, coin_data: DataFrame, rsi: DataFrame) -> (ActionSimple, int):
        """
        Function returns the simple action based on the RSI.

        Args:
            coin_data (DataFrame): The coin data
            rsi (DataFrame): The RSI

        Returns:
            ActionSimple: The action to take
            int: The indicator strength            
        """
        action = ActionSimple.HOLD        
        # if rsi line is above overbought threshold
        if rsi.iloc[-1][self.RSI] > self.overbought:
            action = ActionSimple.SELL
        # if its below oversold threshold
        elif rsi.iloc[-1][self.RSI] < self.oversold:
            action = ActionSimple.BUY
        else:
            action = ActionSimple.HOLD

        indicator_strength = (rsi.iloc[-1][self.RSI] - 50) / 50
        return action, indicator_strength
