from abc import ABC, abstractmethod
from typing import Tuple
from pandas.core.frame import DataFrame
from actions.actions import Actions, Investments

class Investor(ABC):
    """
    Class for investor agent.
    Here you can specify how much USD or coin to invest, based on the actions 
      (ie: buy/sell/hold) made by the agent.
    More complex classes can be implemented:

    Method @invest returns all investments made with the passed action. The portfolio value is updated with the investments.
    Method @make_investment makes an investment based on one passed action. The portfolio value is updated with the investment.
    Method @get_portfolio_value returns the current portfolio value.

    Use method @set_env_parameters to set the environment parameters.
    """

    def __init__(self, portfolio_size: float):
        """
        Args:
            portfolio_size (float): The initial size of the portfolio.
        """
        self.portfolio_size = portfolio_size

    def set_env_parameters(self, **kwargs):
        """
        Method to set the environment parameters.

        Args:
            **kwargs: The environment parameters
        """
        pass

    @abstractmethod
    def invest(self, coin_data: DataFrame, actions: Actions) -> Investments:
        """
        Method to make investments based on the actions.

        Args:
            coin_data (DataFrame): The coin data
            actions (Actions): The actions to take

        Returns:
            Investments: The investments made
        """
        pass

    @abstractmethod
    def make_investment(self, coin_data: DataFrame, action: Actions, return_final_portfolio_value: bool) -> Tuple[Investments, float]:
        """
        Method to make an investment based on the action.

        Args:
            coin_data (DataFrame): The coin data
            action (Actions): The action to take. If Actions DataFrame contains more than one action, the investment will be made based on the last action.
            return_final_portfolio_value (bool): If True, the final portfolio value will be returned. If False, the portfolio value after the investment will be returned.

        Returns:
            Investments: The investment made
            float: The portfolio value after the investment
        """
        pass

    @abstractmethod
    def get_portfolio_value(self) -> float:
        """
        Method to get the current portfolio value.

        Returns:
            float: The portfolio value
        """
        pass

    @abstractmethod
    def get_portfolio_value_final(self) -> float:
        """
        Method to get the final portfolio value. All accrued investments are sold at the last coin data time.

        Returns:
            float: The final portfolio value
        """
        pass 
