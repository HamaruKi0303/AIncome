## Google Drive setting


```python
from google.colab import drive
drive.mount('/content/drive')
```


```python
%cd /content/drive/MyDrive/AIncome
```


```python
!pwd
```

### 作業フォルダ


```python
!mkdir AutoTrade06_multi
```


```python
%cd  AutoTrade06_multi
```


```python
!pwd
```

## 必要パッケージのインストール


```python
"""
#!pip install gym[all] -U
!pip install "gym==0.19.0"
!pip install stable-baselines[mpi]
#!pip install tensorflow==1.14.0
!pip install tensorflow-gpu==1.14.0
!pip install pyqt5
!pip install imageio
!pip install gym-anytrading
"""
!pip install "gym==0.19.0"
!pip install stable-baselines[mpi]
!pip uninstall -y tensorflow-gpu
!pip install tensorflow-gpu==1.14.0

```


```python
#!pip uninstall tensorboard-plugin-wit --yes
```

## インポート



```python
"""

import gym_anytrading

from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import ACKTR
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
"""
import os, gym
import matplotlib.pyplot as plt
```

## 設定


```python

# ログフォルダの生成
log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)


# train data
idx1 = 100
idx2 = 5000

# test data
idx3 = 6000

window_size = 100

trade_fee = 0

```

## 環境の生成


```python
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

# multiprocess environment
env = make_vec_env('CartPole-v1', n_envs=400)

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=2500000)
model.save("ppo2_cartpole")

del model # remove to demonstrate saving and loading

model = PPO2.load("ppo2_cartpole")


```


```python
# Enjoy trained agent

plt.figure(figsize=(30, 10))
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # env.render()
    # print('info:', info)

    # エピソード完了
    if dones.any():
        print('info:', info)
        break
```


```python
# Load the TensorBoard notebook extension
%load_ext tensorboard
```


```python
%tensorboard --logdir logs/
```
