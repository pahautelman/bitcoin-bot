from agents.agent import Agent
from pandas.core.frame import DataFrame
from actions.actions import Actions, ActionSimple, Investments

class DcaAgent(Agent):
    """
    Dollar Cost Averaging agent.

    Invests a fixed amount of money at regular intervals.
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
        action_date = coin_data.index
        actions = [ActionSimple.HOLD for _ in range(len(coin_data))]
        indicator_values = [0 for _ in range(len(coin_data))]
        for i in range(0, len(coin_data), self.investment_interval):
            actions[i] = ActionSimple.BUY
            indicator_values[i] = 1

        return Actions(
            index=action_date, 
            data={
                Actions.ACTION: actions, 
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )

    def get_investments(self, coin_data: DataFrame) -> Investments:
        """
        Method return Investments.

        Args:
            coin_data (DataFrame): The coin data

        Returns:
            Investments: The investments to make
        """
        actions = self.act(coin_data)
        portfolio_size = self.investment_amount * len(actions)

        investments = None
        for i in range(len(actions)):
            if i == 0:
                investments = Investments(
                    index=[coin_data.index[0]],
                    data={
                        Investments.USD_AMOUNT_INVESTED: [portfolio_size - self.investment_amount],
                        Investments.COIN_AMOUNT_INVESTED: [self.investment_amount / coin_data.loc[actions.index[0]]['Close']]
                    }
                )
                continue
            investments.buy_asset(actions.index[i], self.investment_amount, coin_data.loc[actions.index[i]]['Close'])
        return investments
