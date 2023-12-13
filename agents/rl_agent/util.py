import os
import gymnasium as gym
import numpy as np

from gym.core import Env
from gym_trading_env.environments import TradingEnv
from gym_trading_env.renderer import Renderer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from pandas._libs.tslibs.timestamps import Timestamp
from pandas.core.frame import DataFrame

from coin_data import get_coin_data

# TODO: record weights and biases https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ 
def preprocess_data(data: DataFrame, window_size: int=60) -> DataFrame:
    """
    Method that pre-processes the coin data for the gym-trading environment.
        * renames columns
        * creates normalized features

    Args:
        data (DataFrame): The coin data
        window_size (int, optional): The window size for the volume feature. Defaults to 60.

    Returns:
        DataFrame: The pre-processed coin data
    """
    coin_data = data.copy()
    coin_data.rename(
        columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
    coin_data.sort_index(inplace=True)
    coin_data = _normalize_features(coin_data, window_size=window_size)

    return coin_data

def _normalize_features(coin_data: DataFrame, window_size: int) -> DataFrame:
    """
    Method that normalizes the features of the coin data.

    Args:
        coin_data (DataFrame): The coin data
        window_size (int, optional): The window size for the volume feature. Defaults to 60.

    Returns:
        DataFrame: The coin data with normalized features
    """
    coin_data['feature_Close'] = coin_data['close'].pct_change()
    coin_data['feature_High'] = coin_data['high'] / coin_data['close']
    coin_data['feature_Low'] = coin_data['low'] / coin_data['close']
    coin_data['feature_Open'] = coin_data['open'] / coin_data['close']
    coin_data['feature_Volume'] = coin_data['volume'] / coin_data['volume'].shift(window_size).max()

    coin_data.dropna(inplace=True)
    return coin_data

def get_env(
        coin: str,
        coin_data: DataFrame,
        window_size: int=60,
        positions: [float]=[-1, 0, 1],
        trading_fees=0.001,
        borrow_interest_rate=0.0003/100,
        portfolio_initial_value=10000,
        ) -> Env:
    """
    Method that returns the gym-trading environment.

    Args:
        coin (str): The coin name
        coin_data (DataFrame): The processed coin data
        window (int, optional): The window size for the volume feature. Defaults to 60.
        positions ([float], optional): The positions. Defaults to [-1, 0, 1].
        trading_fees (float, optional): The trading fees. Defaults to 0.001.
        borrow_interest_rate (float, optional): The borrow interest rate. Defaults to 0.0003/100.
        portfolio_initial_value (int, optional): The initial portfolio value. Defaults to 10000.

    Returns:
        gym.core.Env: The gym-trading environment
    """
    env = gym.make(
        "TradingEnv",
        name=coin,
        df=coin_data,
        windows=window_size,
        positions=positions,
        trading_fees=trading_fees,
        borrow_interest_rate=borrow_interest_rate,
        portfolio_initial_value=portfolio_initial_value,
        reward_function=_reward_function
    )

    env.unwrapped.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0) )
    env.unwrapped.add_metric('Episode Length', lambda history : len(history['position']) )

    return env

def _reward_function(history):
    return np.log(history['portfolio_valuation'][-1] / history['portfolio_valuation'][-2])

def train_model(
        model_name: str,
        model: BaseAlgorithm,
        env: gym.core.Env,
        total_timesteps: int,
        ):
    env.reset()

    eval_env = Monitor(env)
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=f'./log/models/{model_name}',
        log_path=f'./log/models/{model_name}',
        eval_freq=1000,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
    )

def eval_model(
        model_name: str,
        model: BaseAlgorithm,
        env: Env,
        coin_data: DataFrame = None,
        load_best: bool=True,
        render: bool = False,
        ):
    if load_best is None:
        best_model_path = f'./log/models/{model_name}/best_model.zip'
        if not os.path.exists(best_model_path):
            raise Exception(f'Best model does not exist for {model_name}')
        model = model.load(best_model_path)
    if coin_data is None:
        data = get_coin_data('BTC/USDT', '1h', start_date=Timestamp('2021-01-01'))
        coin_data = preprocess_data(data)
    
    observation, info = env.reset()
    print(info)
    done, truncated = False, False
    while not done and not truncated:
        action, _states = model.predict(observation)
        observation, reward, done, truncated, info = env.step(action)
        if done or truncated:
            print(info)

    env.render()
    # check file exists
    if not os.path.exists(f'./log/models/{model_name}/render/BTC'):
        os.makedirs(f'./log/models/{model_name}/render/BTC')

    env.unwrapped.save_for_render(dir = f'./log/models/{model_name}/render')
    if render:
        renderer = Renderer(render_logs_dir=f'./log/models/{model_name}/render/BTC')
        renderer.run()
        print("!!! Server running http://127.0.0.1:5000 !!!")
    