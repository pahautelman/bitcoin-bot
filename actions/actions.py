from pandas.core.frame import DataFrame
from pandas._libs.tslibs.timestamps import Timestamp
from typing import Dict, List

tolerance = 1e-6

class ActionSimple():
    """
    Class represents a simple action: Buy, Sell, or Hold.
    """

    BUY = 'BUY'
    SELL = 'SELL'
    HOLD = 'HOLD'

class Actions(DataFrame):
    """
    Class represents a list of actions.
    Index is date of action.

    Columns:
        ACTION: The action to take
        INDICATOR_STRENGTH: The strength of the indicator. 1 represents a bullish sentiment, -1 represents a bearish sentiment.
    """

    ACTION = 'ACTION'
    INDICATOR_STRENGTH = 'INDICATOR_STRENGTH'

    def __init__(self, index: List[Timestamp], data: Dict[str, List]):
        if (list(data.keys()) != [Actions.ACTION, Actions.INDICATOR_STRENGTH]):
            raise Exception(f'Invalid columns. Expected {[self.ACTION, Actions.INDICATOR_STRENGTH]} but received {list(data.keys())}')

        super().__init__(index=index, data=data, columns=[Actions.ACTION, Actions.INDICATOR_STRENGTH])


class Investments(DataFrame):
    """
    Class represents a list of investments.
    Index is date of investment.
    """

    FIAT_AMOUNT_INVESTED = 'FIAT_AMOUNT_INVESTED'
    COIN_AMOUNT = 'COIN_AMOUNT'
    COLUMNS = [FIAT_AMOUNT_INVESTED, COIN_AMOUNT]

    def __init__(self, index: List[Timestamp] = None, data: Dict[str, List] = None):
        if index is None:
            index = []
        if data is None:
            data = {column: [] for column in self.COLUMNS}

        if list(data.keys()) != self.COLUMNS:
            raise Exception(f'Invalid columns. Expected {self.COLUMNS} but received {list(data.keys())}')

        super().__init__(index=index, data=data, columns=self.COLUMNS)
