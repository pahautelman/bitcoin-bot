from typing import Tuple
import pandas as pd
from pandas.core.frame import DataFrame
from actions.actions import ActionSimple, Actions, Investments
from agents.investors.investor import Investor


class SpotInvestor(Investor):
    """
    Class for spot investor agent.

    On each action, the agent buys/sells the same amount of USD.
    """

    def __init__(self, portfolio_size: float, investment_size: float):
        """
        Args:
            portfolio_size (float): The initial size of the portfolio.
            investment_size (float): The size of the investment in USD.
        """
        super().__init__(portfolio_size)
        self.accumulated_coins = 0.0
        self.investment_size = investment_size
        self.trading_fee = 0.001  # 0.1% trading fee on each trade

        self.last_close = None

    def set_env_parameters(self, environment_parameters: dict):
        """
        Method to set the environment parameters.

        Args:
            environment_parameters (dict): The environment parameters. Should contain the following keys:
                - trading_fee (float): The trading fee percentage
        """
        if 'trading_fee' in environment_parameters:
            self.trading_fee = environment_parameters['trading_fee']


    def invest(self, coin_data: DataFrame, actions: Actions) -> Investments:
        """
        Method to make investments based on the actions.

        Args:
            coin_data (DataFrame): The coin data
            actions (Actions): The actions to take

        Returns:
            Investments: The investments made
        """
        investments = Investments()

        for i in range(len(coin_data)):
            date = coin_data.index[i]
            if date not in actions.index:
                continue

            action = Actions(
                index = [date], 
                data = {
                    Actions.ACTION: [actions.loc[date][Actions.ACTION]],
                    Actions.INDICATOR_STRENGTH: [actions.loc[date][Actions.INDICATOR_STRENGTH]],
                }
            )
            investment, _ = self.make_investment(coin_data.iloc[:i + 1], action)
            
            # to avoid warnings
            if not investments.empty and not investment.empty:
                investments = pd.concat([investments, investment])
            elif investments.empty:
                investments = investment

        return investments
    
    def make_investment(self, coin_data: DataFrame, action: Actions, return_final_portfolio_value: bool = False) -> Tuple[Investments, float]:
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
        self.last_close = coin_data.iloc[-1]['Close']
        investment = self._make_investment(coin_data, action)

        if return_final_portfolio_value:
            return investment, self.get_portfolio_value_final()
        else:
            return investment, self.get_portfolio_value()
    
    def _make_investment(self, coin_data: DataFrame, action: Actions) -> Investments:
        action_type = action.iloc[-1][Actions.ACTION]
        if action_type == ActionSimple.HOLD:
            return Investments()
        elif action_type == ActionSimple.BUY:
            return self._make_buy_investment(coin_data)
        elif action_type == ActionSimple.SELL:
            return self._make_sell_investment(coin_data)
        raise ValueError(f'Invalid action type: {action_type}')
    
    def _make_buy_investment(self, coin_data: DataFrame) -> Investments:
        if self.portfolio_size < self.investment_size:
            print('WARNING: Not enough USD to buy')
            return Investments()
        
        self.portfolio_size -= self.investment_size
        trading_fee = self.investment_size * self.trading_fee

        coin_amount = (self.investment_size - trading_fee) / self.last_close
        self.accumulated_coins += coin_amount

        return Investments(
            index=[coin_data.index[-1]],
            data={
                Investments.FIAT_AMOUNT_INVESTED: [-self.investment_size],
                Investments.COIN_AMOUNT: [coin_amount],
            }
        )
    
    def _make_sell_investment(self, coin_data: DataFrame) -> Investments:
        coin_amount = self.investment_size / self.last_close
        if coin_amount > self.accumulated_coins:
            print('WARNING: Not enough coins to sell')
            return Investments()
        
        self.accumulated_coins -= coin_amount

        trading_fee = self.investment_size * self.trading_fee
        self.portfolio_size += self.investment_size - trading_fee

        return Investments(
            index=[coin_data.index[-1]],
            data={
                Investments.FIAT_AMOUNT_INVESTED: [self.investment_size],
                Investments.COIN_AMOUNT: [-coin_amount],
            }
        )

    def get_portfolio_value(self) -> float:
        """
        Method to get the value of the portfolio.

        Returns:
            float: The value of the portfolio at the last 'Close' value in the coin data
        """
        return self.portfolio_size + self.accumulated_coins * self.last_close
    
    def get_portfolio_value_final(self) -> float:
        """
        Method to get the final value of the portfolio.

        Returns:
            float: The final value of the portfolio
        """
        coin_value = self.accumulated_coins * self.last_close
        trading_fee = coin_value * self.trading_fee

        return self.portfolio_size + coin_value - trading_fee
