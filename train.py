import time
import gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.utils import sync, str2bool

#### Check the environment's spaces ########################
env = gym.make("takeoff-aviary-v0")      # 起飛
#env = gym.make("hover-aviary-v0")       # 旋停
#env = gym.make("flythrugate-aviary-v0") # 穿越
#env = gym.make("tune-aviary-v0")        # 調整
print("[INFO] Action space:", env.action_space)
print("[INFO] Observation space:", env.observation_space)
check_env(env, warn=True, skip_render_check=True)

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1000000)
model.save('drone_a2c')

