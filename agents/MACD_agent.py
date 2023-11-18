from pandas import DataFrame
from actions.actions import Actions, ActionSimple
from agents.agent import Agent




class MACD_agent(Agent):
    """
    Agent that implements *moving average convergence divergence* (MACD) strategy.

    Buy when the MACD line crosses above the signal line.
    Sell when the MACD line crosses below the signal line.
    """

    def __init__(self, fast_period: int, slow_period: int, signal_period: int):
        """
        Args:
            fast_period (int): The fast period for the MACD
            slow_period (int): The slow period for the MACD
            signal_period (int): The signal period for the MACD
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements MACD strategy.
        Buy when the MACD line crosses above the signal line.
        Sell when the MACD line crosses below the signal line.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        macd = self._get_macd(coin_data, self.fast_period, self.slow_period, self.signal_period)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= self.slow_period:
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(macd.iloc[:i])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date, 
            data={
                Actions.ACTION: actions,
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )

    MACD = 'macd'
    SIGNAL = 'signal'

    def _get_macd(self, coin_data: DataFrame, fast_period: int=12, slow_period: int=26, signal_period: int=9) -> DataFrame:
        """
        Function calculates the MACD.

        Args:
            coin_data (DataFrame): The coin data
            fast_period (int): The fast period for the MACD
            slow_period (int): The slow period for the MACD
            signal_period (int): The signal period for the MACD

        Returns:
            DataFrame: The MACD
        """
        macd = coin_data[coin_data.columns[0]].ewm(span=fast_period, adjust=False).mean() - \
            coin_data[coin_data.columns[0]].ewm(span=slow_period, adjust=False).mean()
        
        signal = macd.ewm(span=signal_period, adjust=False).mean()

        return DataFrame({MACD_agent.MACD: macd, MACD_agent.SIGNAL: signal})


    def _get_simple_action(self, macd: DataFrame) -> (ActionSimple, int):
        """
        Function gets the action to take based on the MACD.

        Args:
            macd (DataFrame): The MACD

        Returns:
            ActionSimple: The action to take
            int: The indicator strength
        """
        action = ActionSimple.HOLD
        indicator_strength = 0
        # if macd line is above the signal
        if macd.iloc[-1][MACD_agent.MACD] > macd.iloc[-1][MACD_agent.SIGNAL]:
            indicator_strength = 1
            # and it previously was not
            if macd.iloc[-2][MACD_agent.MACD] < macd.iloc[-2][MACD_agent.SIGNAL]:
                action = ActionSimple.BUY
        # if macd line is below the signal
        elif macd.iloc[-1][MACD_agent.MACD] < macd.iloc[-1][MACD_agent.SIGNAL]:
            indicator_strength = -1
            # and it previously was not
            if macd.iloc[-2][MACD_agent.MACD] > macd.iloc[-2][MACD_agent.SIGNAL]:
                action = ActionSimple.SELL
        return action, indicator_strength
    
        # # if macd line is above signal line and it previously was not, buy
        # if macd.iloc[-1][MACD_agent.MACD] > macd.iloc[-1][MACD_agent.SIGNAL] and macd.iloc[-2][MACD_agent.MACD] < macd.iloc[-2][MACD_agent.SIGNAL]:
        #     return ActionSimple.BUY
        # # if macd line is below signal line and it previously was not, sell
        # elif macd.iloc[-1][MACD_agent.MACD] < macd.iloc[-1][MACD_agent.SIGNAL] and macd.iloc[-2][MACD_agent.MACD] > macd.iloc[-2][MACD_agent.SIGNAL]:
        #     return ActionSimple.SELL
        # else:
        #     return ActionSimple.HOLD

        # if macd.iloc[-1][MACD_agent.MACD] > macd.iloc[-1][MACD_agent.SIGNAL]:
        #     return ActionSimple.BUY
        # elif macd.iloc[-1][MACD_agent.MACD] < macd.iloc[-1][MACD_agent.SIGNAL]:
        #     return ActionSimple.SELL
        # else:
        #     return ActionSimple.HOLD
