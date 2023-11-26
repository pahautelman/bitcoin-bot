from pandas import DataFrame
from actions.actions import Actions, ActionSimple
from agents.basic.exponential_moving_average import EmaAgent

class AdxAgent(EmaAgent):
    """
    Agent that implement the *average directional index* (ADX) strategy.
    ADX is lagging trend indicator that measures the direction and strength of a trend.

    ADX is calculated using the following formula:
        1. Calculate the directional movement, +DM and -DM
            +DM is found by calculating the difference between the current high and the previous high.
            -DM is found by calculating the difference between the previous low and the current low.
            If +DM > -DM, then +DM is the directional movement.
        2. Calculate the directional indicators, +DI and -DI
            +DI = 100 * EMA(+DM) / ATR
            -DI = 100 * EMA(-DM) / ATR
            ATR is the average true range over the window size.
        3. Calculate the average directional index, ADX
            ADX = 100 * EMA(|+DI - -DI| / (+DI + -DI)) * sign(+DI - -DI)
            Sign is added to determine if the trend is bullish (>0) or bearish (<0). 

    ADX oscillates between (+/-)0 and (+/-)100.
    A value above 25 is typically considered to indicate a trend.
    A value above 50 is typically considered to indicate a strong trend.

    ADX provides signals for buying and selling:
        1. Buy when the ADX crosses above the threshold.
        2. Sell when the ADX crosses below the threshold.
    """

    def __init__(self, window: int = 14, threshold: int = 25):
        """
        Args:
            window (int): The window size for the ADX
            threshold (int): The threshold for the ADX
        """
        super().__init__(window)
        self.threshold = threshold

    def is_action_strength_normalized(self) -> bool:
        """
        Method that returns whether the action strength is normalized, having values between [-1, 1].

        Returns:
            bool: Whether the action strength is normalized
        """
        return True

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements ADX strategy.

        Buy when the ADX crosses above the threshold.
        Sell when the ADX crosses below the threshold.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        adx = self.get_indicator(coin_data)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= self.window:
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i + 1], adx.iloc[:i + 1])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date, 
            data={
                'adx': adx,
                'indicator_values': indicator_values,
                'actions': actions
            }
        )

    ADX = 'adx'

    def get_indicator(self, coin_data: DataFrame) -> DataFrame:
        """
        Method that returns the ADX indicator.

        Args:
            coin_data (DataFrame): The coin data

        Returns:
            DataFrame: The ADX
        """
        return self._get_adx(coin_data, self.window)

    def _get_adx(self, coin_data: DataFrame, window: int) -> DataFrame:
        """
        Function calculates the ADX for the given coin data.

        Args:
            coin_data (DataFrame): The coin data
            window (int): The window size for the ADX
        
        Returns:
            DataFrame: The ADX
        """
        # Calculate the directional movement, +DM and -DM
        dm = coin_data['High'].diff()
        dm = dm.apply(lambda x: 0 if x < 0 else x)
        dm = dm.rename('dm')
        dm_neg = coin_data['Low'].diff()
        dm_neg = dm_neg.apply(lambda x: 0 if x > 0 else abs(x))
        dm_neg = dm_neg.rename('dm_neg')
        dm = dm.to_frame().join(dm_neg.to_frame())

        # Calculate the directional indicators, +DI and -DI
        di_pos = self._get_ema(dm['dm'], window)
        di_neg = self._get_ema(dm['dm_neg'], window)
        di_pos.rename(columns={self.EMA: 'di_pos'}, inplace=True)
        di_neg.rename(columns={self.EMA: 'di_neg'}, inplace=True)
        di = di_pos.join(di_neg)

        # TODO: check math
        # Calculate the average directional index, ADX
        adx = (di['di_pos'] - di['di_neg']).abs()
        adx = adx / (di['di_pos'] + di['di_neg'])
        adx = self._get_ema(adx, window)
        adx = adx.apply(lambda x: abs(x))
        adx = adx.apply(lambda x: 100 * x)

        return DataFrame(
            index=adx.index,
            data={
                self.ADX: adx[self.EMA]
            }
        )

    def _get_simple_action(self, coin_data: DataFrame, adx: DataFrame) -> (ActionSimple, int):
        """
        Function gets the ADX simple action.

        Args:
            coin_data (DataFrame): The coin data
            adx (DataFrame): The ADX

        Returns:
            (ActionSimple, int): The action and indicator strength
        """
        action = ActionSimple.HOLD
        # if ADX crosses above threshold, buy
        if adx.iloc[-1] > self.threshold and adx.iloc[-2] < self.threshold:
            action = ActionSimple.BUY
        elif adx.iloc[-1] < -self.threshold and adx.iloc[-2] > -self.threshold:
            action = ActionSimple.SELL
        
        indicator_strength = adx.iloc[-1] / 100 
        return action, indicator_strength