from agents.investors.investor import Investor
from actions.actions import ActionSimple, Actions, Investments
from pandas.core.frame import DataFrame
from pandas._libs.tslibs.timestamps import Timestamp
from typing import List

# TODO: test
# TODO: selling strategy where you sell depending on the average increase in price over the previous buys.
#   ie: if the average increase in price over the previous buys is 10%, then sell x * 10% of the coins acquired so far.
class SimpleStopLossInvestor(Investor):
    """
    Class for simple stop loss investor agent

    Invest a fixed allocation of the current portfolio size in each BUY investment (portfolio_allocation * current_portfolio_size).
        - If the price drops by stop_loss percentage, sell all coins acquired in the transaction
    
    Sell a fixed allocation of the current portfolio size in each SELL investment (assets_allocation * current_assets_owned).
    """

    class Buys(DataFrame):
        """
        Class represents a list of buys.
        Index is date of buy.

        For each buy, store the amount of USD invested, the asset price, and the (remaining) amount of coins acquired.

        Each sell should update the amount of coins acquired.
        A stop-loss should sell all remaining coins acquired in the transaction.
        """
        USD_AMOUNT_INVESTED = 'USD_AMOUNT_INVESTED'
        INITIAL_ASSET_PRICE = 'INITIAL_ASSET_PRICE'
        REMAINING_COIN_AMOUNT = 'REMAINING_COIN_AMOUNT'
        COLUMNS = [USD_AMOUNT_INVESTED, INITIAL_ASSET_PRICE, REMAINING_COIN_AMOUNT]

        def __init__(self):
            super().__init__(index=[], data={}, columns=self.COLUMNS)

        def add_buy(self, date: Timestamp, usd_amount: float, asset_price: float):
            """
            Method to add a buy.

            Args:
                date (Timestamp): The date of the investment
                usd_amount (float): The amount of USD invested
                asset_price (float): The price of the asset at the time of investment
            """
            self.loc[date] = [usd_amount, asset_price, usd_amount / asset_price]

        def sell(self, coin_percentage: float) -> float:
            """
            Method to sell coins. coin_percentage of all coins acquired in the transaction are sold.

            Args:
                coin_percentage (float): The amount of coins to sell

            Returns:
                float: The amount of coins sold
            """
            assert 0 <= coin_percentage <= 1

            coins_sold = (self[self.REMAINING_COIN_AMOUNT] * coin_percentage).sum()
            self[self.REMAINING_COIN_AMOUNT] -= self[self.REMAINING_COIN_AMOUNT] * coin_percentage

            return coins_sold


        def stop_loss(self, date: Timestamp) -> float:
            """
            Method to sell all coins acquired in the transaction at the given date.

            Args:
                date (Timestamp): The date of the investment

            Returns:
                float: The amount of coins sold
            """
            assert date in self.index

            coin_amount = self.loc[date][self.REMAINING_COIN_AMOUNT]
            self.drop(date, inplace=True)

            return coin_amount


    def __init__(
        self, 
        portfolio_size: int, 
        stop_loss: float, 
        portfolio_allocation: float, 
        assets_allocation: float
    ):
        """
        Args:
            portfolio_size (int): The size of the portfolio, USD value
            stop_loss (float): The stop loss percentage
            portfolio_allocation (float): The percentage of the portfolio to allocate to each investment
            assets_allocation (float): The percentage of the assets to sell in each sell investment
        """
        self.portfolio_size = portfolio_size
        self.stop_loss = stop_loss
        self.portfolio_allocation = portfolio_allocation
        self.assets_allocation = assets_allocation

    def get_investments(self, coin_data: DataFrame, actions: Actions) -> Investments:
        """
        Method return Investments.

        Args:
            coin_data (DataFrame): The coin data
            actions (Actions): The actions to take

        Returns:
            Investments: The investments to make
        """
        assert not coin_data.index.empty

        buys = self.Buys()
        # TODO: init with start date, portfolio size
        investments = Investments(
            index=[coin_data.index[0]], 
            data={
                Investments.USD_AMOUNT_INVESTED: [self.portfolio_size],
                Investments.COIN_AMOUNT_INVESTED: [0]
            }
        )

        for i in range(len(coin_data)):
            # check stop-loss
            stop_loss_transactions = self._check_stop_loss(buys, coin_data.iloc[i]['Close'], self.stop_loss)
            if stop_loss_transactions:
                coins_sold = 0
                for date in stop_loss_transactions:
                    coins_sold += buys.stop_loss(date)

                investments.sell_asset(coin_data.index[i], coins_sold, coin_data.iloc[i]['Close'])
                self.portfolio_size = investments.get_usd_amount_invested()

            # check date is in actions
            date = coin_data.index[i]
            if date not in actions.index:
                continue

            # TODO: update portfolio size
            if actions.loc[date][Actions.ACTION] == ActionSimple.BUY:
                usd_amount = self.portfolio_allocation * self.portfolio_size
                # self.portfolio_size -= usd_amount
                
                asset_price = coin_data.iloc[i]['Close']

                buys.add_buy(coin_data.index[i], usd_amount, asset_price)
                investments.buy_asset(coin_data.index[i], usd_amount, asset_price)
            elif actions.loc[date][Actions.ACTION] == ActionSimple.SELL:
                coins_sold = buys.sell(self.assets_allocation)
                asset_price = coin_data.iloc[i]['Close']

                investments.sell_asset(coin_data.index[i], coins_sold, asset_price)
            
            self.portfolio_size = investments.get_usd_amount_invested()

        return investments

    def _check_stop_loss(self, buys: Buys, current_asset_price: float, stop_loss: float) -> List[Timestamp]:
        """
        Method to check if the stop-loss has been triggered for any of the previous buys.  

        Args:
            buys (Buys): The buys
            current_asset_price (float): The current asset price
            stop_loss (float): The stop loss percentage

        Returns:
            List[Timestamp]: The dates of the stop-losses
        """
        stop_loss_dates = []
        for date in buys.index:
            if buys.loc[date][self.Buys.INITIAL_ASSET_PRICE] * (1 - stop_loss) > current_asset_price:
                stop_loss_dates.append(date)

        return stop_loss_dates
