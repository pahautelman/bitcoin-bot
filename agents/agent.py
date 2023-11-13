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