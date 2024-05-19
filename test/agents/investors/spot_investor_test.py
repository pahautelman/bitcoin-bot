import pandas as pd
import unittest
from actions.actions import ActionSimple
from agents.investors.basic.spot_investor import SpotInvestor, Actions, Investments

class TestSpotInvestor(unittest.TestCase):
    def setUp(self):
        self.coin_data = pd.DataFrame({
            'Close': [100, 110, 120, 130, 140],
        }, index=pd.date_range(start='2022-01-01', periods=5))

    def test_make_buy_investment(self):
        investor = SpotInvestor(portfolio_size=1000, investment_size=100)
        investor.last_close = self.coin_data.iloc[-1]['Close']
        investment = investor._make_buy_investment(self.coin_data)
        self.assertEqual(investment.index[0], pd.Timestamp('2022-01-05'))
        self.assertEqual(investment.iloc[0][Investments.FIAT_AMOUNT_INVESTED], -100)
        self.assertAlmostEqual(investment.iloc[0][Investments.COIN_AMOUNT], 0.713571429, places=6)

    def test_make_sell_investment(self):
        investor = SpotInvestor(portfolio_size=1000, investment_size=100)
        investor.last_close = self.coin_data.iloc[-1]['Close']
        investor.accumulated_coins = 1.0
        investment = investor._make_sell_investment(self.coin_data)
        self.assertEqual(investment.index[0], pd.Timestamp('2022-01-05'))
        self.assertEqual(investment.iloc[0][Investments.FIAT_AMOUNT_INVESTED], 100)
        self.assertAlmostEqual(investment.iloc[0][Investments.COIN_AMOUNT], -0.71428571, places=6)

    def test_invest(self):
        investor = SpotInvestor(portfolio_size=1000, investment_size=100)
        actions = Actions(
            index = pd.date_range(start='2022-01-01', periods=5),
            data = {
                Actions.ACTION: [ActionSimple.HOLD, ActionSimple.BUY, ActionSimple.SELL, ActionSimple.HOLD, ActionSimple.BUY],
                Actions.INDICATOR_STRENGTH: [0, 0, 0, 0, 0],
            }
        )

        investments = investor.invest(self.coin_data, actions)
        self.assertEqual(len(investments), 3)
        self.assertEqual(investments.index[0], pd.Timestamp('2022-01-02'))
        self.assertEqual(investments.index[1], pd.Timestamp('2022-01-03'))
        self.assertEqual(investments.index[2], pd.Timestamp('2022-01-05'))

        self.assertEqual(investments.iloc[0][Investments.FIAT_AMOUNT_INVESTED], -100)
        self.assertAlmostEqual(investments.iloc[0][Investments.COIN_AMOUNT], 0.908181818, places=6)

        self.assertEqual(investments.iloc[1][Investments.FIAT_AMOUNT_INVESTED], 100)
        self.assertAlmostEqual(investments.iloc[1][Investments.COIN_AMOUNT], -0.83333333, places=6)

        self.assertEqual(investments.iloc[2][Investments.FIAT_AMOUNT_INVESTED], -100)
        self.assertAlmostEqual(investments.iloc[2][Investments.COIN_AMOUNT], 0.713571429, places=6)

    def test_get_portfolio_value(self):
        investor = SpotInvestor(portfolio_size=1000, investment_size=100)
        investor.last_close = self.coin_data.iloc[-1]['Close']
        investor.accumulated_coins = 1.0
        value = investor.get_portfolio_value()
        self.assertAlmostEqual(value, 1140, places=6)

    def test_get_portfolio_value_final(self):
        investor = SpotInvestor(portfolio_size=1000, investment_size=100)
        investor.last_close = self.coin_data.iloc[-1]['Close']
        investor.accumulated_coins = 1.0
        value = investor.get_portfolio_value_final()
        self.assertAlmostEqual(value, 1139.86, places=6)

    def test_get_portfolio_value_final_no_coins(self):
        investor = SpotInvestor(portfolio_size=1000, investment_size=100)
        investor.last_close = self.coin_data.iloc[-1]['Close']
        value = investor.get_portfolio_value_final()
        self.assertAlmostEqual(value, 1000, places=6)

    def test_buy_when_not_enough_value(self):
        investor = SpotInvestor(portfolio_size=0, investment_size=100)
        investor.last_close = self.coin_data.iloc[-1]['Close']
        investment = investor._make_buy_investment(self.coin_data)
        
        self.assertEqual(len(investment.index), 0)
        
    def sell_when_no_coins(self):
        investor = SpotInvestor(portfolio_size=1000, investment_size=100)
        investor.last_close = self.coin_data.iloc[-1]['Close']
        investment = investor._make_sell_investment(self.coin_data)
        
        self.assertEqual(len(investment.index), 0)

if __name__ == '__main__':
    unittest.main()
