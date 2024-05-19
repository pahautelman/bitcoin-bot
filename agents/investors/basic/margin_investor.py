import pandas as pd
from pandas.core.frame import DataFrame
from pandas import  Timestamp
from actions.actions import ActionSimple, Actions, Investments
from agents.investors.investor import Investor


class MarginInvestor(Investor):
    """
    Class for margin investor agent.

    Makes long/short investments based on the actions. Action strength is used as leverage. Leverage should be positive value for both LONG and SHORT investments.
    The passed asset interest rate, and fiat interest rate are used to calculate the margin interest rate. The interest rate should be specified to the time interval of the coin data.
    
    If borrowed assets are more valuable than the initial portfolio size, the agent will stop trading.
    """

    class MarginInvestments(Investments):
        """
        Class represents a list of margin investments.
        Index is date of investment.

        For each investment, store the type of investment (SHORT/LONG), the amount of USD invested, the asset price when acquired, the leverage, and whether the investment is active.
        """
        INVESTMENT_TYPE = 'investment_type'
        ASSET_PRICE = 'asset_price'
        LEVERAGE = 'leverage'
        IS_ACTIVE = 'is_active'
        COLUMNS = [INVESTMENT_TYPE, Investments.FIAT_AMOUNT_INVESTED, ASSET_PRICE, LEVERAGE, IS_ACTIVE]

        def __init__(self):
            super().__init__()
        
        def add_investment(self, date: Timestamp, investment_type: str, usd_invested: float, asset_price: float, leverage: float, is_active: bool) -> DataFrame:
            """
            Method to add an investment.

            Args:
                date (Timestamp): The date of the investment
                investment_type (str): The type of investment (SHORT/LONG)
                usd_invested (float): The amount of USD invested
                coins_acquired (float): The amount of coins acquired
                asset_price (float): The price of the asset at the time of investment
                borrowed_fiat (float): The amount of borrowed fiat
                borrowed_assets (float): The amount of borrowed assets
                is_active (bool): Whether the investment is active

            Returns:
                MarginInvestments: The added investment.
            """
            self.loc[date] = [investment_type, usd_invested, asset_price, leverage, is_active]
            return self.loc[[date]]

    def __init__(self, portfolio_size: float, investment_size: float):
        """
        Args:
            portfolio_size (float): The initial size of the portfolio.
            investment_size (float): The size of the investment in USD.
        """
        # portfolio size is now the available USD for trading
        super().__init__(portfolio_size)
        self.investment_size = investment_size

        self.investments = MarginInvestor.MarginInvestments()

        self.asset_interest_rate = 0.002 / 100 # 0.002% per hour
        self.fiat_interest_rate = 0.03 / 100 # 0.03% per hour
    
        self.take_profit_percentage = 1.05
        self.stop_loss_percentage = 0.97

        self.last_close_price = None
        self.can_trade = True

    def set_env_parameters(self, environment_parameters: dict):
        """
        Method to set the environment parameters.

        Args:
            environment_parameters (dict): The environment parameters. Should contain the following keys:
                - asset_interest_rate (float): The interest rate for the asset.
                - fiat_interest_rate (float): The interest rate for the fiat.
                - take_profit_percentage (float): The take profit percentage.
                - stop_loss_percentage (float): The stop loss percentage.
        """
        self.asset_interest_rate = environment_parameters.get('asset_interest_rate', self.asset_interest_rate)
        self.fiat_interest_rate = environment_parameters.get('fiat_interest_rate', self.fiat_interest_rate)
        self.take_profit_percentage = environment_parameters.get('take_profit_percentage', self.take_profit_percentage)
        self.stop_loss_percentage = environment_parameters.get('stop_loss_percentage', self.stop_loss_percentage)

    def invest(self, coin_data: DataFrame, actions: Actions) -> Investments:
        """
        Method to make investments based on the actions.

        Args:
            coin_data (DataFrame): The coin data
            actions (Actions): The actions to take

        Returns:
            Investments: The investments made
        """
        for i in range(len(coin_data)):
            date = coin_data.index[i]
            if date not in actions.index:
                self.last_close_price = coin_data.iloc[i]['Close']
                self._update_portfolio_value()
                continue

            self.make_investment(coin_data.loc[:date], actions.loc[:date])

        return self.investments
    
    def make_investment(self, coin_data: DataFrame, actions: Actions, return_final_portfolio_value: bool = False) -> Investments:
        """
        Method to make an investment based on the action.

        Args:
            coin_data (DataFrame): The coin data
            action (Actions): The action to take. If Actions DataFrame contains more than one action, the investment will be made based on the last action.
            return_final_portfolio_value (bool): If True, the final portfolio value will be returned. If False, the portfolio value after the investment will be returned.

        Returns:
            Investments: The investment made
        """
        self.last_close_price = coin_data.iloc[-1]['Close']
        self._update_portfolio_value()

        investment = self._make_investment(coin_data, actions)

        if return_final_portfolio_value:
            return investment, self.get_portfolio_value_final()
        else:
            return investment, self.get_portfolio_value()

    def _make_investment(self, coin_data: DataFrame, actions: Actions) -> Investments:
        """
        Method to make an investment based on the action.

        Args:
            coin_data (DataFrame): The coin data
            action (Actions): The action to take. If Actions DataFrame contains more than one action, the investment will be made based on the last action.

        Returns:
            Investments: The investment made
        """
        investments = MarginInvestor.MarginInvestments()
        action = actions.iloc[-1]

        if action[Actions.ACTION] == ActionSimple.BUY:
            investments = self._make_long_investment(coin_data, actions)
        elif action[Actions.ACTION] == ActionSimple.SELL:
            investments = self._make_short_investment(coin_data, actions)

        return investments
    
    def _make_long_investment(self, coin_data: DataFrame, actions: Actions) -> Investments:
        action = actions.iloc[-1]
        leverage = action[Actions.INDICATOR_STRENGTH]

        # check if the agent can afford the investment
        investment_amount = self.investment_size * leverage
        if self.portfolio_size < investment_amount:
            return MarginInvestor.MarginInvestments()
        
        # make the investment
        self.portfolio_size -= investment_amount

        return self.investments.add_investment(
            date = coin_data.index[-1],
            investment_type = 'LONG',
            usd_invested = self.investment_size,
            asset_price = self.last_close_price,
            leverage = leverage,
            is_active = True
        )
    
    def _make_short_investment(self, coin_data: DataFrame, actions: Actions) -> Investments:
        action = actions.iloc[-1]
        leverage = action[Actions.INDICATOR_STRENGTH]

        # check if the agent can afford the investment
        investment_amount = self.investment_size * leverage
        if self.portfolio_size < investment_amount:
            return MarginInvestor.MarginInvestments()
        
        # make the investment
        self.portfolio_size -= investment_amount

        return self.investments.add_investment(
            date = coin_data.index[-1],
            investment_type = 'SHORT',
            usd_invested = self.investment_size,
            asset_price = self.last_close_price,
            leverage = leverage,
            is_active = True
        )
    
    def _update_portfolio_value(self):
        """
        Method to update the portfolio value.
        """
        # subtract the borrowed interests, and check TP and SL
        for index, investment in self.investments.iterrows():
            if not investment[MarginInvestor.MarginInvestments.IS_ACTIVE]:
                continue

            if investment[MarginInvestor.MarginInvestments.INVESTMENT_TYPE] == 'LONG':
                interest = investment[MarginInvestor.MarginInvestments.FIAT_AMOUNT_INVESTED] * investment[MarginInvestor.MarginInvestments.LEVERAGE] * self.fiat_interest_rate
            else:
                interest = investment[MarginInvestor.MarginInvestments.FIAT_AMOUNT_INVESTED] * investment[MarginInvestor.MarginInvestments.LEVERAGE] * self.asset_interest_rate
            self.portfolio_size -= interest

            # check take profit and stop loss
            if investment[MarginInvestor.MarginInvestments.INVESTMENT_TYPE] == 'LONG':
                tp_price = investment[MarginInvestor.MarginInvestments.ASSET_PRICE] * self.take_profit_percentage
                sl_price = investment[MarginInvestor.MarginInvestments.ASSET_PRICE] * self.stop_loss_percentage

                if self.last_close_price >= tp_price:
                    # sell at take profit price
                    self.portfolio_size += investment[MarginInvestor.MarginInvestments.FIAT_AMOUNT_INVESTED] * investment[MarginInvestor.MarginInvestments.LEVERAGE] * (self.take_profit_percentage - 1)
                    # add borrowed amount
                    self.portfolio_size += investment[MarginInvestor.MarginInvestments.FIAT_AMOUNT_INVESTED] * investment[MarginInvestor.MarginInvestments.LEVERAGE]
                    self.investments.at[index, MarginInvestor.MarginInvestments.IS_ACTIVE] = False
                elif self.last_close_price <= sl_price:
                    # sell at stop loss price
                    self.portfolio_size += investment[MarginInvestor.MarginInvestments.FIAT_AMOUNT_INVESTED] * investment[MarginInvestor.MarginInvestments.LEVERAGE] * (self.stop_loss_percentage - 1)
                    self.portfolio_size += investment[MarginInvestor.MarginInvestments.FIAT_AMOUNT_INVESTED] * investment[MarginInvestor.MarginInvestments.LEVERAGE]
                    self.investments.at[index, MarginInvestor.MarginInvestments.IS_ACTIVE] = False
            else:
                tp_price = investment[MarginInvestor.MarginInvestments.ASSET_PRICE] * (2 - self.take_profit_percentage)
                sl_price = investment[MarginInvestor.MarginInvestments.ASSET_PRICE] * (2 - self.stop_loss_percentage)

                if self.last_close_price <= tp_price:
                    # sell at take profit price
                    self.portfolio_size += investment[MarginInvestor.MarginInvestments.FIAT_AMOUNT_INVESTED] * investment[MarginInvestor.MarginInvestments.LEVERAGE] * (self.take_profit_percentage - 1)
                    self.portfolio_size += investment[MarginInvestor.MarginInvestments.FIAT_AMOUNT_INVESTED] * investment[MarginInvestor.MarginInvestments.LEVERAGE]
                    self.investments.at[index, MarginInvestor.MarginInvestments.IS_ACTIVE] = False
                elif self.last_close_price >= sl_price:
                    # sell at stop loss price
                    self.portfolio_size += investment[MarginInvestor.MarginInvestments.FIAT_AMOUNT_INVESTED] * investment[MarginInvestor.MarginInvestments.LEVERAGE] * (self.stop_loss_percentage - 1)
                    self.portfolio_size += investment[MarginInvestor.MarginInvestments.FIAT_AMOUNT_INVESTED] * investment[MarginInvestor.MarginInvestments.LEVERAGE]
                    self.investments.at[index, MarginInvestor.MarginInvestments.IS_ACTIVE] = False

        # check borrowed amount does not exceed the portfolio size
        if self.portfolio_size <= 0:
            self.can_trade = False

    def get_portfolio_value(self) -> float:
        """
        Method to get the portfolio value.

        Returns:
            float: The portfolio value
        """
        return self.get_portfolio_value_final()


    def get_portfolio_value_final(self) -> float:
        """
        Method to get the final portfolio value.

        Returns:
            float: The final portfolio value
        """
        if not self.can_trade:
            return 0.0
        
        portfolio_value = self.portfolio_size

        # sell active investments
        for _, investment in self.investments.iterrows():
            if not investment[MarginInvestor.MarginInvestments.IS_ACTIVE]:
                continue
            
            # add borrowed amount
            portfolio_value += investment[MarginInvestor.MarginInvestments.FIAT_AMOUNT_INVESTED] * investment[MarginInvestor.MarginInvestments.LEVERAGE]

            pct_change = self.last_close_price / investment[MarginInvestor.MarginInvestments.ASSET_PRICE]
            if investment[MarginInvestor.MarginInvestments.INVESTMENT_TYPE] == 'LONG':
                portfolio_value += investment[MarginInvestor.MarginInvestments.FIAT_AMOUNT_INVESTED] * investment[MarginInvestor.MarginInvestments.LEVERAGE] * (pct_change - 1)
            else:
                portfolio_value += investment[MarginInvestor.MarginInvestments.FIAT_AMOUNT_INVESTED] * investment[MarginInvestor.MarginInvestments.LEVERAGE] * (1 - pct_change)
        return portfolio_value
