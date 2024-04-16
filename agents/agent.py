from abc import ABC, abstractmethod
from pandas.core.frame import DataFrame
from actions.actions import Actions

class Agent(ABC):
    """
    Class for all agents. 
    
    Method @act should return an Actions object.
    """

    def __init__(self):
        """
        Here pass all the parameters that are needed for the agent.
        """
        pass

    @abstractmethod
    def act(coin_data: DataFrame) -> Actions:
        """
        Method that returns an Actions object.

        Args:
            coin_data (DataFrame): The coin data

        Returns:
            Actions: The actions to take
        """
        pass

    @abstractmethod
    def get_initial_intervals(self) -> int:
        """
        Method that returns the number of initial intervals where the agent does not take any action.

        Returns:
            int: The number of initial intervals where the agent does not take any action
        """
        pass

class Indicator(Agent):
    """
    Class for all indicator agents.

    Method @get_indicator should return a DataFrame object containing the indicator values.
    """

    def __init__(self):
        """
        Here pass all the parameters that are needed for the agent.
        """
        pass

    @abstractmethod
    def is_action_strength_normalized(self) -> bool:
        """
        Method that returns whether the action strength is normalized, having values between [-1, 1].

        Returns:
            bool: Whether the action strength is normalized
        """
        pass

    @abstractmethod
    def get_indicator(self, coin_data: DataFrame) -> DataFrame:
        """
        Method that returns the indicator.

        Args:
            coin_data (DataFrame): The coin data

        Returns:
            DataFrame: The indicator
        """
        pass
