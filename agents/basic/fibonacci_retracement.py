from actions.actions import Actions, ActionSimple
from agents.agent import Indicator
from finta import TA
from pandas import DataFrame
from typing import Tuple

class FibonacciRetracementAgent(Indicator):
    """
    Agent that implements Fibonacci retracement strategy.

    TODO:


    Uses 1.382, 1, 0.618, 0.382 as the retracement levels.
    """

    def __init__(self):
        super().__init__()

    def is_action_strength_normalized(self) -> bool:
        return False
    
    def get_initial_intervals(self) -> int:
        return 1
    
    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements Fibonacci retracement strategy.

        Buy when the price crosses above the R4 level.
        Sell when the price crosses below the S4 level.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        pivot_fib = self.get_indicator(coin_data)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            if i <= self.get_initial_intervals():
                actions.append(ActionSimple.HOLD)
                indicator_values.append(0)
                continue

            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i + 1], pivot_fib.iloc[:i + 1])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date, 
            data={
                Actions.ACTION: actions,
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )
    
    # Rezistances and supports
    R4 = "R4"
    R3 = "R3"
    R2 = "R2"
    R1 = "R1"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"

    def get_indicator(self, coin_data: DataFrame) -> DataFrame:
        """
        Method that returns the Fibonacci retracement indicator.

        Args:
            coin_data (DataFrame): The coin data

        Returns:
            DataFrame: The Fibonacci retracement
        """
        return self._get_pivot_fib(coin_data)

    def _get_pivot_fib(self, coin_data: DataFrame) -> DataFrame:
        """
        Method that returns the Fibonacci retracement indicator.

        Args:
            coin_data (DataFrame): The coin data

        Returns:
            DataFrame: The Fibonacci retracement
        """
        pivot_fib = TA.PIVOT_FIB(coin_data)
        return DataFrame(
            index=pivot_fib.index,
            data={
                self.R4: pivot_fib['r4'].values,
                self.R3: pivot_fib['r3'].values,
                self.R2: pivot_fib['r2'].values,
                self.R1: pivot_fib['r1'].values,
                self.S1: pivot_fib['s1'].values,
                self.S2: pivot_fib['s2'].values,
                self.S3: pivot_fib['s3'].values,
                self.S4: pivot_fib['s4'].values,
            }
        )
    
    def _get_simple_action(self, coin_data: DataFrame, pivot_fib: DataFrame) -> Tuple[ActionSimple, int]:
        """
        Function calculates the action to take based on the Fibonacci retracement.
        Buy when the strike price crosses above the R2 level.
        Sell when the strike price crosses below the S2 level.

        Args:
            coin_data (DataFrame): The coin data
            pivot_fib (DataFrame): The Fibonacci retracement

        Returns:
            Tuple[ActionSimple, int]: The action to take and the indicator strength
        """
        action = ActionSimple.HOLD
        if coin_data.iloc[-1]['Close'] > pivot_fib.iloc[-1][self.R2] and coin_data.iloc[-2]['Close'] <= pivot_fib.iloc[-2][self.R2]:
            action = ActionSimple.BUY
        elif coin_data.iloc[-1]['Close'] < pivot_fib.iloc[-1][self.S2] and coin_data.iloc[-2]['Close'] >= pivot_fib.iloc[-2][self.S2]:
            action = ActionSimple.SELL
        
        return action, pivot_fib.iloc[-1]
    