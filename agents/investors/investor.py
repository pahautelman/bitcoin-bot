from abc import ABC, abstractmethod
from pandas.core.frame import DataFrame
from actions.actions import Actions, Investments

class Investor(ABC):
    """
    Class for investor agent.
    Here you can specify how much USD or coin to invest, based on the actions 
      (ie: buy/sell/hold) made by the agent.
    More complex classes can be implemented:

    Method @get_investments should return Investment object.
    """

    def __init__(self):
        """
        Here pass all the parameters that are needed for the agent.
        """
        pass

    @abstractmethod
    def get_investments(coin_data: DataFrame) -> Investments:
        """
        Method return Investments.

        Args:
            coin_data (DataFrame): The coin data

        Returns:
            Investments: The investments to make
        """
        pass