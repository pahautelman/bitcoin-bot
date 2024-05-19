import unittest
import pandas as pd
from pandas import DataFrame
from actions.actions import Actions, ActionSimple
from agents.investors.basic.margin_investor import MarginInvestor

class TestMarginInvestor(unittest.TestCase):
    def setUp(self):
        self.portfolio_size = 1000
        self.investment_size = 100
        self.time_frame = pd.date_range(start='2022-01-01', periods=5)
        self.coin_data = DataFrame({
            'Close': [100, 110, 120, 130, 140]
        }, index=self.time_frame)

        self.investor = MarginInvestor(portfolio_size=self.portfolio_size, investment_size=self.investment_size)

        
    def test_initialization(self):
        self.assertEqual(self.investor.portfolio_size, self.portfolio_size)
        self.assertEqual(self.investor.investment_size, self.investment_size)
        self.assertEqual(self.investor.last_close_price, None)
        self.assertTrue(self.investor.investments.empty)

    def test_set_env_parameters(self):
        env_params = {
            'asset_interest_rate': 0.005 / 100,
            'fiat_interest_rate': 0.04 / 100,
            'take_profit_percentage': 1.10,
            'stop_loss_percentage': 0.95
        }
        self.investor.set_env_parameters(env_params)
        self.assertEqual(self.investor.asset_interest_rate, env_params['asset_interest_rate'])
        self.assertEqual(self.investor.fiat_interest_rate, env_params['fiat_interest_rate'])
        self.assertEqual(self.investor.take_profit_percentage, env_params['take_profit_percentage'])
        self.assertEqual(self.investor.stop_loss_percentage, env_params['stop_loss_percentage'])

    def test_invest_no_action(self):
        actions = Actions(
            index=[],
            data={
                Actions.ACTION: [],
                Actions.INDICATOR_STRENGTH: []
            }
        )
        investments = self.investor.invest(self.coin_data, actions)
        self.assertTrue(investments.empty)
        self.assertTrue(self.investor.investments.empty)

    def test_make_long_investment(self):
        self.investor.last_close_price = self.coin_data.iloc[-1]['Close']
        actions = Actions(
            index=pd.date_range(start='2022-01-01', periods=5),
            data={
                Actions.ACTION: [ActionSimple.HOLD, ActionSimple.BUY, ActionSimple.SELL, ActionSimple.HOLD, ActionSimple.BUY],
                Actions.INDICATOR_STRENGTH: [0, 1.0, 1.0, 0, 2.0],
            }
        )
        investment = self.investor._make_long_investment(self.coin_data, actions)
        self.assertEqual(len(investment), 1)
        self.assertEqual(investment.index[0], pd.Timestamp('2022-01-05'))
        self.assertEqual(investment.iloc[0][MarginInvestor.MarginInvestments.INVESTMENT_TYPE], 'LONG')
        self.assertEqual(investment.iloc[0][MarginInvestor.MarginInvestments.FIAT_AMOUNT_INVESTED], 100)
        self.assertEqual(investment.iloc[0][MarginInvestor.MarginInvestments.ASSET_PRICE], 140)
        self.assertEqual(investment.iloc[0][MarginInvestor.MarginInvestments.LEVERAGE], 2.0)
        self.assertEqual(investment.iloc[0][MarginInvestor.MarginInvestments.IS_ACTIVE], True)


    def test_make_short_investment(self):
        self.investor.last_close_price = self.coin_data.iloc[-1]['Close']
        actions = Actions(
            index=pd.date_range(start='2022-01-01', periods=5),
            data={
                Actions.ACTION: [ActionSimple.HOLD, ActionSimple.HOLD, ActionSimple.HOLD, ActionSimple.HOLD, ActionSimple.SELL],
                Actions.INDICATOR_STRENGTH: [0, 0, 0, 0, 3.0],
            }
        )
        investment = self.investor._make_short_investment(self.coin_data, actions)
        self.assertEqual(len(investment), 1)
        self.assertEqual(investment.index[0], pd.Timestamp('2022-01-05'))
        self.assertEqual(investment.iloc[0][MarginInvestor.MarginInvestments.INVESTMENT_TYPE], 'SHORT')
        self.assertEqual(investment.iloc[0][MarginInvestor.MarginInvestments.FIAT_AMOUNT_INVESTED], 100)
        self.assertEqual(investment.iloc[0][MarginInvestor.MarginInvestments.ASSET_PRICE], 140)
        self.assertEqual(investment.iloc[0][MarginInvestor.MarginInvestments.LEVERAGE], 3.0)
        self.assertEqual(investment.iloc[0][MarginInvestor.MarginInvestments.IS_ACTIVE], True)

    def test_update_portfolio_value_fiat_interest(self):
        self.assertTrue(self.investor.investments.empty)
        self.assertEqual(self.investor.portfolio_size, self.portfolio_size)
        
        leverage = 2.0
        fiat_interest_rate = 0.04 / 100
        self.investor.fiat_interest_rate = fiat_interest_rate
        
        self.investor.last_close_price = self.coin_data.iloc[-1]['Close']
        self.investor._make_investment(
            coin_data=self.coin_data,
            actions=Actions(
                index=[pd.Timestamp('2022-01-05')],
                data={
                    Actions.ACTION: [ActionSimple.BUY],
                    Actions.INDICATOR_STRENGTH: [leverage]
                }
            )
        )
        self.investor._update_portfolio_value()
        
        self.assertFalse(self.investor.investments.empty)
        self.assertTrue(self.investor.investments.iloc[-1][MarginInvestor.MarginInvestments.IS_ACTIVE])

        # subtract the interest rate
        self.assertAlmostEqual(
            self.investor.get_portfolio_value(), 
            self.portfolio_size - (self.investment_size * leverage * fiat_interest_rate),
            places=6
        )

    def test_update_portfolio_value_asset_interest(self):
        self.assertTrue(self.investor.investments.empty)
        self.assertEqual(self.investor.portfolio_size, self.portfolio_size)
        
        leverage = 2.0
        asset_interest_rate = 0.005 / 100
        self.investor.asset_interest_rate = asset_interest_rate
        
        self.investor.last_close_price = self.coin_data.iloc[-1]['Close']
        self.investor._make_investment(
            coin_data=self.coin_data,
            actions=Actions(
                index=[pd.Timestamp('2022-01-05')],
                data={
                    Actions.ACTION: [ActionSimple.SELL],
                    Actions.INDICATOR_STRENGTH: [leverage]
                }
            )
        )
        self.investor._update_portfolio_value()
        
        self.assertFalse(self.investor.investments.empty)
        self.assertTrue(self.investor.investments.iloc[-1][MarginInvestor.MarginInvestments.IS_ACTIVE])
        # subtract the interest rate
        self.assertAlmostEqual(
            self.investor.get_portfolio_value(), 
            self.portfolio_size - (self.investment_size * leverage * asset_interest_rate),
            places=6
        )

    def test_update_portfolio_value_long_take_profit(self):
        self.assertTrue(self.investor.investments.empty)
        self.assertEqual(self.investor.portfolio_size, self.portfolio_size)
        
        leverage = 2.0
        fiat_interest_rate = 0.04 / 100
        take_profit_percentage = 1.1
        self.investor.fiat_interest_rate = fiat_interest_rate
        self.investor.take_profit_percentage = take_profit_percentage
        
        self.investor.last_close_price = 100
        self.investor._make_investment(
            coin_data=self.coin_data,
            actions=Actions(
                index=[pd.Timestamp('2022-01-05')],
                data={
                    Actions.ACTION: [ActionSimple.BUY],
                    Actions.INDICATOR_STRENGTH: [leverage]
                }
            )
        )
        self.investor.last_close_price = 110.01
        self.investor._update_portfolio_value()
        
        # subtract the interest rate
        portfolio_value = self.portfolio_size - (self.investment_size * leverage * fiat_interest_rate)
        # add the profit
        portfolio_value += self.investment_size * leverage * (take_profit_percentage - 1)

        self.assertFalse(self.investor.investments.empty)
        self.assertFalse(self.investor.investments.iloc[-1][MarginInvestor.MarginInvestments.IS_ACTIVE])
        self.assertAlmostEqual(
            self.investor.get_portfolio_value(), 
            portfolio_value,
            places=6
        )

    def test_update_portfolio_value_short_take_profit(self):
        self.assertTrue(self.investor.investments.empty)
        self.assertEqual(self.investor.portfolio_size, self.portfolio_size)
        
        leverage = 2.0
        asset_interest_rate = 0.005 / 100
        take_profit_percentage = 1.10
        self.investor.asset_interest_rate = asset_interest_rate
        self.investor.take_profit_percentage = take_profit_percentage
        
        self.investor.last_close_price = 100
        self.investor._make_investment(
            coin_data=self.coin_data,
            actions=Actions(
                index=[pd.Timestamp('2022-01-05')],
                data={
                    Actions.ACTION: [ActionSimple.SELL],
                    Actions.INDICATOR_STRENGTH: [leverage]
                }
            )
        )
        self.investor.last_close_price = 89.99
        self.investor._update_portfolio_value()
        
        # subtract the interest rate
        portfolio_value = self.portfolio_size - (self.investment_size * leverage * asset_interest_rate)
        # add the profit
        portfolio_value += self.investment_size * leverage * (take_profit_percentage - 1)
        
        self.assertFalse(self.investor.investments.empty)
        self.assertFalse(self.investor.investments.iloc[-1][MarginInvestor.MarginInvestments.IS_ACTIVE])
        self.assertAlmostEqual(
            self.investor.get_portfolio_value(), 
            portfolio_value,
            places=6
        )

    def test_update_portfolio_value_long_stop_loss(self):
        self.assertTrue(self.investor.investments.empty)
        self.assertEqual(self.investor.portfolio_size, self.portfolio_size)
        
        leverage = 2.0
        fiat_interest_rate = 0.04 / 100
        stop_loss_percentage = 0.95
        self.investor.fiat_interest_rate = fiat_interest_rate
        self.investor.stop_loss_percentage = stop_loss_percentage
        
        self.investor.last_close_price = 100
        self.investor._make_investment(
            coin_data=self.coin_data,
            actions=Actions(
                index=[pd.Timestamp('2022-01-05')],
                data={
                    Actions.ACTION: [ActionSimple.BUY],
                    Actions.INDICATOR_STRENGTH: [leverage]
                }
            )
        )
        self.investor.last_close_price = 94.99
        self.investor._update_portfolio_value()
        
        # subtract the interest rate
        portfolio_value = self.portfolio_size - (self.investment_size * leverage * fiat_interest_rate)
        # subtract the loss
        portfolio_value -= self.investment_size * leverage * (1 - stop_loss_percentage)
        
        self.assertFalse(self.investor.investments.empty)
        self.assertFalse(self.investor.investments.iloc[-1][MarginInvestor.MarginInvestments.IS_ACTIVE])
        self.assertAlmostEqual(
            self.investor.get_portfolio_value(), 
            portfolio_value,
            places=6
        )

    def test_update_portfolio_value_short_stop_loss(self):
        self.assertTrue(self.investor.investments.empty)
        self.assertEqual(self.investor.portfolio_size, self.portfolio_size)
        
        leverage = 2.0
        asset_interest_rate = 0.005 / 100
        stop_loss_percentage = 0.95
        self.investor.asset_interest_rate = asset_interest_rate
        self.investor.stop_loss_percentage = stop_loss_percentage
        
        self.investor.last_close_price = 100
        self.investor._make_investment(
            coin_data=self.coin_data,
            actions=Actions(
                index=[pd.Timestamp('2022-01-05')],
                data={
                    Actions.ACTION: [ActionSimple.SELL],
                    Actions.INDICATOR_STRENGTH: [leverage]
                }
            )
        )
        self.investor.last_close_price = 105.01
        self.investor._update_portfolio_value()
        
        # subtract the interest rate
        portfolio_value = self.portfolio_size - (self.investment_size * leverage * asset_interest_rate)
        # subtract the loss
        portfolio_value -= self.investment_size * leverage * (1 - stop_loss_percentage)

        self.assertFalse(self.investor.investments.empty)
        self.assertFalse(self.investor.investments.iloc[-1][MarginInvestor.MarginInvestments.IS_ACTIVE])
        self.assertAlmostEqual(
            self.investor.get_portfolio_value(), 
            portfolio_value,
            places=6
        )
    
    def test_update_portfolio_value_inactive(self):
        self.assertTrue(self.investor.investments.empty)
        self.assertEqual(self.investor.portfolio_size, self.portfolio_size)
        
        leverage = 2.0
        fiat_interest_rate = 0.04 / 100
        self.investor.fiat_interest_rate = fiat_interest_rate
        
        self.investor.last_close_price = 110.01
        self.investor.investments.add_investment(
            date=pd.Timestamp('2022-01-01'),
            investment_type='LONG',
            usd_invested=self.investment_size,
            asset_price=self.investment_size,
            leverage=leverage,
            is_active=False
        )
        self.investor.investments.add_investment(
            date=pd.Timestamp('2022-01-01'),
            investment_type='SHORT',
            usd_invested=self.investment_size,
            asset_price=self.investment_size,
            leverage=leverage,
            is_active=False
        )
        self.investor._update_portfolio_value()
        
        self.assertFalse(self.investor.investments.empty)
        self.assertFalse(self.investor.investments.iloc[-1][MarginInvestor.MarginInvestments.IS_ACTIVE])
        self.assertAlmostEqual(
            self.investor.get_portfolio_value(), 
            self.portfolio_size,
            places=6
        )

    def test_available_fiat_after_investment(self):
        leverage = 2.0
        self.investor._make_investment(
            coin_data=self.coin_data,
            actions=Actions(
                index=[pd.Timestamp('2022-01-05')],
                data={
                    Actions.ACTION: [ActionSimple.BUY],
                    Actions.INDICATOR_STRENGTH: [leverage]
                }
            )
        )
        self.assertAlmostEqual(
            self.investor.portfolio_size, 
            self.portfolio_size - self.investment_size * leverage,
            places=6    
        )
        
    def test_invest(self):
        fiat_interest_rate = 0.04 / 100
        asset_interest_rate = 0.005 / 100
        tp_percentage = 1.10
        sl_percentage = 0.95
        self.investor.set_env_parameters({
            'fiat_interest_rate': fiat_interest_rate,
            'asset_interest_rate': asset_interest_rate,
            'take_profit_percentage': tp_percentage,
            'stop_loss_percentage': sl_percentage
        })

        actions = Actions(
            index=self.time_frame,
            data={
                Actions.ACTION: [ActionSimple.HOLD, ActionSimple.BUY, ActionSimple.SELL, ActionSimple.HOLD, ActionSimple.BUY],
                Actions.INDICATOR_STRENGTH: [0, 2.0, 1.0, 0, 2.0],
            }
        )
        investments = self.investor.invest(self.coin_data, actions)
        self.assertEqual(len(investments), 3)
        
        self.assertEqual(investments.index[0], pd.Timestamp('2022-01-02'))
        investment = investments.iloc[0]
        self.assertEqual(investment[MarginInvestor.MarginInvestments.INVESTMENT_TYPE], 'LONG')
        self.assertEqual(investment[MarginInvestor.MarginInvestments.FIAT_AMOUNT_INVESTED], 100)
        self.assertEqual(investment[MarginInvestor.MarginInvestments.ASSET_PRICE], 110)
        self.assertEqual(investment[MarginInvestor.MarginInvestments.LEVERAGE], 2.0)
        self.assertEqual(investment[MarginInvestor.MarginInvestments.IS_ACTIVE], False) # false since sold due to TP

        self.assertEqual(investments.index[1], pd.Timestamp('2022-01-03'))
        investment = investments.iloc[1]
        self.assertEqual(investment[MarginInvestor.MarginInvestments.INVESTMENT_TYPE], 'SHORT')
        self.assertEqual(investment[MarginInvestor.MarginInvestments.FIAT_AMOUNT_INVESTED], 100)
        self.assertEqual(investment[MarginInvestor.MarginInvestments.ASSET_PRICE], 120)
        self.assertEqual(investment[MarginInvestor.MarginInvestments.LEVERAGE], 1.0)
        self.assertEqual(investment[MarginInvestor.MarginInvestments.IS_ACTIVE], False) # false since sold due to SL

        self.assertEqual(investments.index[2], pd.Timestamp('2022-01-05'))
        investment = investments.iloc[2]
        self.assertEqual(investment[MarginInvestor.MarginInvestments.INVESTMENT_TYPE], 'LONG')
        self.assertEqual(investment[MarginInvestor.MarginInvestments.FIAT_AMOUNT_INVESTED], 100)
        self.assertEqual(investment[MarginInvestor.MarginInvestments.ASSET_PRICE], 140)
        self.assertEqual(investment[MarginInvestor.MarginInvestments.LEVERAGE], 2.0)
        self.assertEqual(investment[MarginInvestor.MarginInvestments.IS_ACTIVE], True)

        # calculate portfolio value
        portfolio_value = self.portfolio_size
        # two interest rates on 1st LONG position
        portfolio_value -= 2 * self.investment_size * 2.0 * fiat_interest_rate
        # one interest rate on 1st SHORT position
        portfolio_value -= self.investment_size * 1.0 * asset_interest_rate

        # profit from 1st LONG position
        portfolio_value += self.investment_size * 2.0 * (tp_percentage - 1)
        # loss from 1st SHORT position
        portfolio_value -= self.investment_size * 1.0 * (1 - sl_percentage)

        self.assertAlmostEqual(self.investor.get_portfolio_value_final(), portfolio_value, places=6)
    

    def test_invest_cannot_trade(self):
        self.investor.take_profit_percentage = 2.0
        self.investor.fiat_interest_rate = 0.1
        actions = Actions(
            index=self.time_frame,
            data={
                Actions.ACTION: [ActionSimple.HOLD, ActionSimple.BUY, ActionSimple.BUY, ActionSimple.HOLD, ActionSimple.BUY],
                Actions.INDICATOR_STRENGTH: [0, 5.0, 4.0, 0, 1.0],
            }
        )
        investments = self.investor.invest(self.coin_data, actions)
        self.assertFalse(self.investor.can_trade)
        self.assertEqual(len(investments), 2)
        self.assertAlmostEqual(self.investor.get_portfolio_value(), 0.0, places=6)
        
        
if __name__ == '__main__':
    unittest.main()
