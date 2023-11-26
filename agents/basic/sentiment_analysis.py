import requests

from pandas import DataFrame
from pandas._libs.tslibs.timestamps import Timestamp
from actions.actions import Actions, ActionSimple
from agents.agent import Agent
from datetime import datetime


class SentimentAnalysisAgent(Agent):
    """
    Agent that implements sentiment analysis strategy.

    Currently retrives the BTC fear-and-greed index from alternative.me.

    Indicator strength is equal to the sentiment on a given day. Value between [-1, 1]
    """

    def __init__(self, coin: str = 'BTC'):
        """
        Args:
            coin (str): The coin
        """
        if coin != 'BTC':
            raise Exception(f'Coin {coin} not supported.')
        self.coin = coin

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements sentiment analysis strategy.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        fear_and_greed = self._get_fear_and_greed_index(coin_data)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i + 1], fear_and_greed.iloc[:i + 1])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date,
            data={
                Actions.ACTION: actions,
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )
    
    FEAR_AND_GREED = 'fear_and_greed'

    def _get_fear_and_greed_index(self, coin_data : DataFrame) -> DataFrame:
        """
        Function gets the fear and greed index from alternative.me.
        
        Args:
            coin_data (DataFrame): The coin data

        Returns:
            DataFrame: The fear and greed index
        """
        first_date = coin_data.index[0]
        last_date = coin_data.index[-1]
        date_format = '%Y-%m-%d %H:%M:%S'

        # number of days between current time and first date
        limit = (datetime.now() - first_date).days + 2  # add 2 to account for first date and the time before first date

        api_url = f'https://api.alternative.me/fng/?limit={limit}&format=json'
        response = requests.get(api_url)
        data = response.json()

        if 'data' not in data:
            raise Exception('No data in response. Error: ' + data.get('metadata', {}).get('error', ''))

        index_data = []
        for entry in data['data']:
            timestamp = entry.get('timestamp')
            if timestamp is None:
                raise Exception('Timestamp is None')
            
            date = Timestamp.fromtimestamp(int(timestamp)).strftime(date_format)
            date = Timestamp(date)

            if date > last_date:
                break

            value = entry.get('value')
            if value is None:
                raise Exception('Value is None at timestamp ' + timestamp)
            
            index_data.append({'Timestamp': date, 'Value': int(value)})

        index_data.reverse()    # put oldest date first

        # set value for dates inside coin_data
        # get fear and greed value for first date before coin_data.index[0]
        # TODO: if coin_data.index[0] is before the first date in index_data, then value will be None
        first_date = coin_data.index[0]
        index_data_index = -1
        index_data_value = None
        # while index_data_index + 1 < len(index_data) and index_data[index_data_index + 1]['Timestamp'] <= first_date:
        #     index_data_index += 1
        # index_data_value = index_data[index_data_index]['Value']

        fear_and_greed_dates = []
        fear_and_greed_values = []
        
        index_data_index = -1
        index_data_value = None
        for date in coin_data.index:
            while index_data_index + 1 < len(index_data) and index_data[index_data_index + 1]['Timestamp'] <= date:
                index_data_index += 1
                index_data_value = index_data[index_data_index]['Value']

            fear_and_greed_dates.append(date)
            value = (index_data_value - 50) / 50 if index_data_value is not None else None
            fear_and_greed_values.append(value)
            
        df = DataFrame(
            index=fear_and_greed_dates, 
            data={
                self.FEAR_AND_GREED: fear_and_greed_values
            }
        )
        return df
    
    def _get_simple_action(self, coin_data: DataFrame, fear_and_greed: DataFrame) -> (ActionSimple, float):
        """
        Function returns the action and the indicator strength for the given coin data and indicator.

        Args:
            coin_data (DataFrame): The coin data
            fear_and_greed (DataFrame): The indicator

        Returns:
            ActionSimple: Always HOLD
            int: The indicator value
        """
        return ActionSimple.HOLD, fear_and_greed.iloc[-1][self.FEAR_AND_GREED]
    

    
