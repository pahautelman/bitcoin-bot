import ccxt

class KucoinClient:
    def __init__(self, api_key, api_secret, passphrase):
        """
        Kucoin API client constructor.

        https://www.kucoin.com/docs/beginners/introduction

        Args:
            api_key (str): The Kucoin API token id
            api_secret (str): The Kucoin API secret
            passphrase (str): The Kucoin passphrase used to create API
        """
        authentication = {
            'apiKey': api_key,
            'secret': api_secret,
            'password': passphrase
        }
        self.client = ccxt.kucoin(authentication)
        
    def create_margin_order(self, coin, side, amount, leverage, take_profit_percentage, stop_loss_percentage, test=True):
        # TODO: kucoin Access denied, require more permission.
        """
        Method to create a margin order.

        https://www.kucoin.com/docs/rest/margin-trading/orders/place-margin-order

        Args:
            coin (str): The coin to buy
            side (str): The side of the order (buy or sell)
            amount (float): The amount of currency to trade
            leverage (int): The leverage to use
            take_profit_percentage (float): The take profit percentage, relative to the current price
            stop_loss_percentage (float): The stop loss percentage, relative to the current price
            test (bool, optional): Whether to test the order. Defaults to False.
        """
        if not test:
            raise Exception('Not implemented yet')
        
        assert side in ['buy', 'sell'], f'Invalid side {side}. Expected buy or sell'

        ticker = self.client.fetch_ticker(coin)
        price = (ticker['bid'] + ticker['ask']) / 2
        stop_loss_price = price * stop_loss_percentage
        take_profit_price = price * take_profit_percentage
        assert stop_loss_price < take_profit_price if side == 'buy' else stop_loss_price > take_profit_price, \
            f'Invalid stop loss and take profit prices. Expected stop loss price {stop_loss_price} to be lower than take profit price {take_profit_price}'
        assert take_profit_price > price if side == 'buy' else take_profit_price < price, \
            f'Invalid take profit price {take_profit_price}. Expected take profit price to be higher than current price {price}'

        main_order, stop_loss_order, take_profit_order = None, None, None
        try:
            # place order
            main_order = self.client.create_order(
                coin,
                type='market',
                side=side,
                amount=amount,
                params={
                    'test': True,
                    'leverage': leverage,
                }
            )

            # create stop-loss order
            stop_loss_price = main_order['price'] * stop_loss_percentage
            stop_loss_order = self.client.create_order(
                coin,
                type='market',
                side='sell' if side == 'buy' else 'buy',
                amount=amount,
                price=stop_loss_price,
                params={
                    'test': True,
                }
            )

            # create take-profit order
            take_profit_price = main_order['price'] * take_profit_percentage
            take_profit_order = self.client.create_order(
                coin,
                type='market',
                side='sell' if side == 'buy' else 'buy',
                amount=amount,
                price=take_profit_price,
                params={
                    'test': True,
                }
            )

        except Exception as e:
            # cancel the orders
            if main_order:
                self.client.cancel_order(main_order['id'])
            if stop_loss_order:
                self.client.cancel_order(stop_loss_order['id'])
            if take_profit_order:
                self.client.cancel_order(take_profit_order['id'])

            print(e)            

        return main_order, stop_loss_order, take_profit_order