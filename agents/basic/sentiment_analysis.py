import requests

from actions.actions import Actions, ActionSimple
from agents.agent import Indicator
from datetime import datetime
from pandas import DataFrame
from pandas._libs.tslibs.timestamps import Timestamp
from typing import Tuple


class SentimentAnalysisAgent(Indicator):
    """
    Agent that implements sentiment analysis strategy.

    Currently retrieves the BTC fear-and-greed index from alternative.me.

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

    def is_action_strength_normalized(self) -> bool:
        """
        Method that returns whether the action strength is normalized, having values between [-1, 1].

        Returns:
            bool: Whether the action strength is normalized
        """
        return True
    
    def is_action_strength_normalized(self) -> bool:
        """
        Method that returns whether the action strength is normalized, having values between [-1, 1].

        Returns:
            bool: Whether the action strength is normalized
        """
        return True
    
    def get_initial_intervals(self) -> int:
        return 0

    def act(self, coin_data: DataFrame) -> Actions:
        """
        Function implements sentiment analysis strategy.

        Args:
            coin_data (DataFrame): The coin data
        
        Returns:
            Actions: The actions to take
        """
        sentiment_analysis = self.get_indicator(coin_data)

        action_date = coin_data.index
        actions = []
        indicator_values = []
        for i in range(len(coin_data)):
            action, indicator_strength = self._get_simple_action(coin_data.iloc[:i + 1], sentiment_analysis.iloc[:i + 1])
            actions.append(action)
            indicator_values.append(indicator_strength)

        return Actions(
            index=action_date,
            data={
                Actions.ACTION: actions,
                Actions.INDICATOR_STRENGTH: indicator_values
            }
        )
    
    SENTIMENT_ANALYSIS = 'sentiment_analysis'

    def get_indicator(self, coin_data: DataFrame) -> DataFrame:
        """
        Function returns the fear and greed index for the given coin data.

        Args:
            coin_data (DataFrame): The coin data

        Returns:
            DataFrame: The fear and greed index
        """
        ind = self._get_sentiment_analysis(coin_data)
        return ind

    def _get_sentiment_analysis(self, coin_data: DataFrame) -> DataFrame:
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

        # Number of days between current time and first date
        limit = (datetime.now() - first_date).days + 2  # Add 2 to account for first date and the time before first date

        api_url = f'https://api.alternative.me/fng/?limit={limit}&format=json'
        response = requests.get(api_url)
        data = response.json()

        if 'data' not in data:
            raise Exception('No data in response. Error: ' + data.get('metadata', {}).get('error', ''))

        data['data'].reverse()  # Put oldest date first

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

        sa_dates = []
        sa_values = []
        
        index_data_index = -1
        index_data_value = None
        for date in coin_data.index:
            while index_data_index + 1 < len(index_data) and index_data[index_data_index + 1]['Timestamp'] <= date:
                index_data_index += 1
                index_data_value = index_data[index_data_index]['Value']

            sa_dates.append(date)
            value = (index_data_value - 50) / 50 if index_data_value is not None else None
            sa_values.append(value)
            
        df = DataFrame(
            index=sa_dates, 
            data={
                self.SENTIMENT_ANALYSIS: sa_values
            }
        )
        return df
    
    def _get_simple_action(self, coin_data: DataFrame, fear_and_greed: DataFrame) -> Tuple[ActionSimple, float]:
        """
        Function returns the action and the indicator strength for the given coin data and indicator.

        Args:
            coin_data (DataFrame): The coin data
            fear_and_greed (DataFrame): The indicator

        Returns:
            ActionSimple: Always HOLD
            int: The indicator value
        """
        return ActionSimple.HOLD, fear_and_greed.iloc[-1][self.SENTIMENT_ANALYSIS]
    

    
