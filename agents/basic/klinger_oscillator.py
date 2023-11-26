from pandas import DataFrame
from actions.actions import Actions, ActionSimple
from agents.agent import Agent

class KoAgent(Agent):
    """
    Agent that implements *Klinger oscillator* (KO) strategy.
    KO is a leading momentum oscillator and is used to measure the long-term trends of money flow while also considering short-term price fluctuations.

    KO is calculated using the following steps:
        1. Calculate the volume force (VF) as follows:
            VF = volume * (2 * ((dm / cm) - 1)) * trend * 100
            
            trend = 1 if (high + low + close) > (high_prev + low_prev + close_prev)
                  = -1 otherwise

            dm = directional movement
            dm = high - low

            cm = cumulative directional movement
            cm = cm_prev + dm if trend == trend_prev
            cm = dm_prev + dm if trend != trend_prev

        2. Calculate the KO as follows:
            KO = EMA_34(VF) - EMA_55(VF)

        3. Calculate the 13-period simple moving average of the KO.

    KO oscillates between (-inf, inf).
    """

    def __init__(self):
        super().__init__()

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements KO strategy.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        ko = self._get_ko(coin_data)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= 55:
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i + 1], ko.iloc[:i + 1])
            actions.append(action)
            indicator_values.append(indicator_strength)
        
        return Actions(
            action_date=action_date,
            actions=actions,
            indicator_values=indicator_values
        )
    
    def _get_ko(self, coin_data: DataFrame) -> DataFrame:
        """
        Function calculates the KO for the given coin data.

        Args:
            coin_data (DataFrame): The coin data

        Returns:
            DataFrame: The KO
        """
        # Calculate the volume force, VF
        dm = coin_data['high'] - coin_data['low']
        trend = coin_data['high'] + coin_data['low'] + coin_data['close'] > coin_data['high'].shift(1) + coin_data['low'].shift(1) + coin_data['close'].shift(1) 
        trend = trend.apply(lambda x: 1 if x else -1)
        cm = dm.copy(deep=True)
        cm[0] = dm[0]
        for i in range(1, len(cm)):
            cm[i] = cm[i-1] + dm[i] if trend[i] == trend[i-1] else dm[i-1] + dm[i]
        # TODO: check dimensions
        vf = coin_data['Volume'] * (2 * ((dm / cm) - 1)) * trend * 100

        # Calculate the KO
        ko = vf.ewm(span=34).mean() - vf.ewm(span=55).mean()

        # Calculate the 13-period simple moving average of the KO
        ko_sma = ko.rolling(window=13).mean()

        return ko_sma
    
    def _get_simple_action(self, coin_data: DataFrame, ko_sma: DataFrame) -> (ActionSimple, float):
        """
        Function returns the simple action and indicator strength for the given indicator.

        Args:
            coin_data (DataFrame): The coin data
            ko_sma (DataFrame): The KO SMA

        Returns:
            ActionSimple: Always hold
            float: The KO SMA value
        """
        return ActionSimple.HOLD, ko_sma.iloc[-1]
