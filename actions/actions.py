from pandas.core.frame import DataFrame
from pandas._libs.tslibs.timestamps import Timestamp
from typing import Dict, List
from math import isclose

tolerance = 1e-6

class ActionSimple():
    """
    Class represents a simple action: Buy, Sell, or Hold.
    """

    BUY = 'BUY'
    SELL = 'SELL'
    HOLD = 'HOLD'

# TODO: have int deciding the strength of the action
class Actions(DataFrame):
    """
    Class represents a list of actions.
    Index is date of action.
    """

    ACTION = 'ACTION'

    def __init__(self, index: List[Timestamp], data: Dict[str, List]):
        if (list(data.keys()) != [Actions.ACTION]):
            raise Exception(f'Invalid columns. Expected {[self.ACTION]} but received {list(data.keys())}')

        super().__init__(index=index, data=data, columns=[Actions.ACTION])

class Investments(DataFrame):
    """
    Class represents a list of investments.
    Index is date of investment.
    """

    USD_AMOUNT_INVESTED = 'USD_AMOUNT_INVESTED'
    COIN_AMOUNT_INVESTED = 'COIN_AMOUNT_INVESTED'
    COLUMNS = [USD_AMOUNT_INVESTED, COIN_AMOUNT_INVESTED]

    # TODO: initialize with only start date and amount of USD to invest
    def __init__(self, index: List[Timestamp] = [], data: Dict[str, List] = []):
        if data and (list(data.keys()) != self.COLUMNS):
            raise Exception(f'Invalid columns. Expected {self.COLUMNS} but received {list(data.keys())}')

        super().__init__(index=index, data=data, columns=self.COLUMNS)

        # get last value of investments
        self.usd_amount_invested = self.iloc[-1][self.USD_AMOUNT_INVESTED]
        self.coin_amount_invested = self.iloc[-1][self.COIN_AMOUNT_INVESTED]

    def get_usd_amount_invested(self) -> float:
        """
        Method returns the amount of USD invested.

        Returns:
            float: The amount of USD invested
        """
        return self.usd_amount_invested

    # TODO: take commission into account 
    def buy_asset(self, date: Timestamp, usd_amount: float, asset_price: float):
        """
        Method to buy an asset.

        Args:
            date (Timestamp): The date of the investment
            usd_amount (float): The amount of USD to invest
            asset_price (float): The price of the asset at the time of investment
        """
        self.usd_amount_invested -= usd_amount
        self.coin_amount_invested += usd_amount / asset_price

        self._validate(self.usd_amount_invested, self.coin_amount_invested)

        self.loc[date] = [self.usd_amount_invested, self.coin_amount_invested]

    def sell_asset(self, date: Timestamp, coin_amount: float, asset_price: float):
        """
        Method to sell an asset.

        Args:
            date (Timestamp): The date of the investment
            coin_amount (float): The amount of coins to sell
            asset_price (float): The price of the asset at the time of investment
        """
        self.usd_amount_invested += coin_amount * asset_price
        self.coin_amount_invested -= coin_amount

        self._validate(self.usd_amount_invested, self.coin_amount_invested)

        self.loc[date] = [self.usd_amount_invested, self.coin_amount_invested]

    def _validate(self, usd_amount_invested: float, coin_amount_invested: float):
        """
        Method to validate the investments.

        Args:
            usd_amount_invested (float): The amount of USD invested
            coin_amount_invested (float): The amount of coins invested
        """
        assert usd_amount_invested >= 0 or isclose(usd_amount_invested, 0, abs_tol=tolerance)
        assert coin_amount_invested >= 0 or isclose(coin_amount_invested, 0, abs_tol=tolerance)
