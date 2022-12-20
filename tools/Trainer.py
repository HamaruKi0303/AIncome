import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from tqdm import tqdm
#from tqdm.notebook import tqdm
#from tqdm import tqdm_notebook as tqdm

import warnings
import typing
from typing import Union, List, Dict, Any, Optional

import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL

from stable_baselines.bench import Monitor

from stable_baselines.common.vec_env import VecEnv, sync_envs_normalization, DummyVecEnv
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
# from stable_baselines import PPO2
from tools.AIncomePPO2 import PPO2

from stable_baselines import ACKTR
from stable_baselines import A2C
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, EventCallback
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.evaluation import evaluate_policy

import quantstats as qs
import mplfinance as mpf

import talib
from yahoo_finance_api2 import share
import yfinance
from Historic_Crypto import Cryptocurrencies
from Historic_Crypto import HistoricalData

import tensorflow as tf

import glob
import os
import sys
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.aincome_config_demo import Exp
from tools.Callback import CheckpointCallback2, EvalCallback2
from tools.reward_logger import total_episode_reward_logger2




class Trainer():
    def __init__(self, param):
        self.param  = param
        
        # ---------------------
        # datasets
        #
        self.df_train_ti = pd.read_csv(self.param.dataset_train_name, index_col=0)
        self.df_test_ti = pd.read_csv(self.param.dataset_valid_name, index_col=0)
        
        
    def make_env(self, MyBTCEnv):
        #####################################
        # 環境の生成
        #
        env_marker_train = lambda:  MyBTCEnv(df=self.df_train_ti, window_size=self.param.window_size, frame_bound=(self.param.window_size, len(self.df_train_ti)))
        env_marker_train.trade_fee = self.param.trade_fee

        env_marker_test = lambda:  MyBTCEnv(df=self.df_test_ti, window_size=self.param.window_size, frame_bound=(self.param.window_size, len(self.df_test_ti)))
        env_marker_test.trade_fee = self.param.trade_fee

        env_marker_test2 = env_marker_test()
        env_marker_test2.trade_fee = self.param.trade_fee

        env = DummyVecEnv([env_marker_train for _ in range(self.param.env_num)])
        #env = SubprocVecEnv([env_marker_train for i in range(env_num)])
        # env_test = DummyVecEnv([env_marker_test for _ in range(1)])
        
        return env, env_marker_test2
        
    def make_model(self, env, policy_kwargs):
        #model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=0, tensorboard_log=log_dir, full_tensorboard_log=True)
        model = PPO2('MlpLstmPolicy', env, verbose=1, policy_kwargs=policy_kwargs, nminibatches=self.param.env_num, tensorboard_log=self.param.log_dir, n_steps=self.param.n_steps, noptepochs=4, train_num=self.param.train_num)
        return model
    
    def make_callback(self, i, env_marker_test2):
        # ==========================
        # callback
        #
        # -------
        # eval callback
        #
        eval_callback = EvalCallback2(env_marker_test2, best_model_save_path='./logs/best_model',
                                    log_path='./logs/results', eval_freq=self.param.eval_freq, verbose=2, name_prefix='PPO2_{:09d}_'.format(i))
        # checkpoint callback
        #
        checkpoint_callback = CheckpointCallback2(save_freq=self.param.save_freq, save_path='./logs/', name_prefix='PPO2_{:09d}'.format(i), verbose=2)

        # -------
        # merge callback
        #
        #callback = CallbackList([checkpoint_callback, eval_callback, ProgressBarCallback(total_timesteps)])
        callback = CallbackList([checkpoint_callback, eval_callback, ])
        
        return callback
    
    def get_split_datasets(self):
        df_train_ti = pd.read_csv(self.param.dataset_train_name, index_col=0)
        df_test_ti = pd.read_csv(self.param.dataset_valid_name, index_col=0)
        
    