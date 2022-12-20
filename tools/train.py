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
from stable_baselines import PPO2
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
from tools.Trainer import Trainer

#####################################
# データ読み込み部分
#
def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Low'].to_numpy()[start:end]

    # -----------------------------
    # 特徴量生成
    #
    df_features = env.df.copy()
    # df_features.drop("")
    ohlc_features = df_features.loc[:, :].to_numpy()[start:end]

    #print(ohlc_features.shape)
    #print(np.diff(ohlc_features, axis=0).shape)
    diff1 = np.insert(np.diff(ohlc_features, axis=0), 0, 0, axis=0)
    diff2 = np.insert(np.diff(diff1, axis=0), 0, 0, axis=0)
    #print(diff)
    #signal_features = env.df.loc[:, ['Close', 'Open', 'High', 'Low', 'Vol']].to_numpy()[start:end]

    #signal_features = np.column_stack((ohlc_features, diff1, diff2))
    signal_features = np.column_stack((ohlc_features, ))

    print(">>> signal_features.shape")
    print( signal_features.shape)
    #print( signal_features.head(5))

    return prices, signal_features

#####################################
# 環境クラス
#
class MyBTCEnv(ForexEnv):
    _process_data = my_process_data


def get_latest_model_param():
  # ---------
  # load save model
  model_list = sorted(glob.glob("logs/*.zip"))
  resume_FLAG = False

  if(len(model_list)>0):
    model_latest = model_list[-1].split("/")[-1].split(".zip")[0].split("_step")[0]
    resume_FLAG = True
    return model_latest.split("_")[0], int(model_latest.split("_")[1]), int(model_latest.split("_")[2]), resume_FLAG
  
  return None, None, None, resume_FLAG



if __name__ == '__main__':
    
    # ---------------------------------------
    # param setting
    #
    Param = Exp()
    Param.preview_param()
    
    TrainerX = Trainer(param=Param)
    
    # ---------------------------------------
    # model setting
    #
    # ------
    # first train
    #
    policy_kwargs = dict(net_arch=[64*2, 'lstm', dict(vf=[128, 128, 128], pi=[64*2, 64*2])])
    #model = PPO2('MlpLstmPolicy', env, verbose=0, policy_kwargs=policy_kwargs, nminibatches=env_num, tensorboard_log=log_dir, n_steps=n_steps)

    # ------
    # resume
    #
    #model = PPO2.load("/content/drive/MyDrive/AIncome/AutoTrade13/logs/rl_model_90000_steps")
    #model.set_env(env)

    # ---------------------------------------
    # Learn model
    #    
    # ==========================
    # Create Env & Model
    #
    # env = gym.make('LunarLander-v2')
    env, env_marker_test2 = TrainerX.make_env(MyBTCEnv=MyBTCEnv)
    
    model = TrainerX.make_model(env, policy_kwargs)
    

    i = 0

    callback = TrainerX.make_callback(i, env_marker_test2)

    # ==========================
    # resume
    #
    _model_name, _resume_idx, _train_num, _resume_FLAG = get_latest_model_param()
    if(_resume_FLAG):
        # ====================================================
        # Model setting
        #
        model = PPO2.load("./logs/{}_{:09d}_{:09d}_steps".format(_model_name, int(_resume_idx), int(_train_num)))
        model.set_env(env)
        model.train_num = _train_num

        # -------
        # tensorboard
        #
        model.tensorboard_log = "logs"
        model.num_timesteps = _train_num

    #tb_log_name = "sampleE{}v".format(i)
    # tb_log_name = "AutoLoopTrain_no_feat"
    # print("tb_log_name : {}".format(tb_log_name))

    model.learn(total_timesteps=Param.total_timesteps , log_interval=100, tb_log_name=Param.tb_log_name, reset_num_timesteps=False, callback=callback)

    # ====================================================
    # Save model
    #
    model.save("./logs/PPO2_{:09d}_{:09d}_steps".format(i, Param.total_timesteps*(i+1)*Param.env_num))