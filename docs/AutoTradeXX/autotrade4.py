# -*- coding: utf-8 -*-
"""AutoTrade5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13FCOrj2zyfGWxNw7t6Oq0XaQqFlrRLtE

## Google Drive setting
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/AIncome

!pwd

"""### 作業フォルダ"""

!mkdir AutoTrade04

# Commented out IPython magic to ensure Python compatibility.
# %cd  AutoTrade04

!pwd

"""## 必要パッケージのインストール"""

#!pip install gym[all] -U
!pip install "gym==0.19.0"
!pip install stable-baselines[mpi]
!pip install tensorflow==1.14.0
!pip install pyqt5
!pip install imageio
!pip install gym-anytrading

!pip uninstall tensorboard-plugin-wit --yes

"""## インポート

"""

import os, gym
import gym_anytrading
import matplotlib.pyplot as plt
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import ACKTR
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds

"""## 設定"""

# ログフォルダの生成
log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)


# train data
idx1 = 10
idx2 = 5000

# test data
idx3 = 6000

"""## 環境の生成"""

# 環境の生成
env = gym.make('forex-v0', frame_bound=(idx1, idx2), window_size=10)
env.trade_fee = 0

env = Monitor(env, log_dir, allow_early_resets=True)



# シードの指定
env.seed(0)
set_global_seeds(0)

# ベクトル化環境の生成
env = DummyVecEnv([lambda: env])

# モデルの生成
model = PPO2('MlpPolicy', env, tensorboard_log=log_dir)
#model = ACKTR('MlpPolicy', env, verbose=1)

# モデルの読み込み
# model = PPO2.load('trading_model')

"""## 学習"""

# モデルの学習
model.learn(total_timesteps=128000*10)

# モデルの保存
model.save('trading_model4.0')

"""## モデルのテスト"""

env = gym.make('forex-v0', frame_bound=(idx2, idx3), window_size=100)
env.seed(0)
state = env.reset()
while True:
    # 行動の取得
    action, _ = model.predict(state)
    # 1ステップ実行
    state, reward, done, info = env.step(action)
    # エピソード完了
    if done:
        print('info:', info)
        break

"""## グラフのプロット"""

#plt.cla()
plt.figure(figsize=(30, 10))
env.render_all()
plt.show()

# Commented out IPython magic to ensure Python compatibility.
# Load the TensorBoard notebook extension
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs/