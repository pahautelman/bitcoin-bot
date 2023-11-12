from agents.agent import Agent
from pandas.core.frame import DataFrame
from actions.actions import Actions, ActionSimple, Investments

class DCA_agent(Agent):
    """
    Dollar Cost Averaging agent.
    """

    def __init__(self, investment_interval: int, investment_amount: float):
        """
        Args:
            investment_interval (int): The interval between investments
            investment_amount (float): The amount to invest
        """
        self.investment_interval = investment_interval
        self.investment_amount = investment_amount

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Method that returns an Actions object.

        Args:
            coin_data (DataFrame): The coin data

        Returns:
            Actions: The actions to take
        """
        action_date = []
        actions = []
        for i in range(0, len(coin_data), self.investment_interval):
            action_date.append(coin_data.index[i])
            actions.append(ActionSimple.BUY)
        
        return Actions(index=action_date, data={Actions.ACTION: actions})

    def get_investments(self, coin_data: DataFrame) -> Investments:
        """
        Method return Investments.

        Args:
            coin_data (DataFrame): The coin data

        Returns:
            Investments: The investments to make
        """
        actions = self.act(coin_data)
        total_invested = len(actions) * self.investment_amount
        usd_invested = [total_invested - self.investment_amount * i  for i in range(1, len(actions) + 1)]
        num_coins_bought = []
        for i in range(len(actions)):
            # TODO: fix this
            num_coins_bought.append(self.investment_amount / coin_data.loc[actions.index[i]]['Close'])

        investments_data = {
            Investments.USD_AMOUNT_INVESTED: usd_invested,
            Investments.COIN_AMOUNT_INVESTED: num_coins_bought
        }
        return Investments(actions.index, investments_data)
