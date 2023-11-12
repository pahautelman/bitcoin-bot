from pandas.core.frame import DataFrame

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

    def __init__(self, index, data):
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

    def __init__(self, index, data):
        if (list(data.keys()) != self.COLUMNS):
            raise Exception(f'Invalid columns. Expected {self.COLUMNS} but received {list(data.keys())}')

        super().__init__(index=index, data=data, columns=self.COLUMNS)

    # TODO: helper classes buy_coin, sell_coin