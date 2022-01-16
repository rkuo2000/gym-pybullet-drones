import time
import gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.utils.utils import sync, str2bool

model = A2C.load('drone_a2c')

env = TakeoffAviary(gui=True, record=False)
#env = HoverAviary(gui=True, record=False)
#env = FlyThruGateAviary(gui=True, record=False)

logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS), num_drones=1)

obs = env.reset()
start = time.time()
for i in range(3*env.SIM_FREQ):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    logger.log(drone=0, timestamp=i/env.SIM_FREQ,
               state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
               control=np.zeros(12)
               )
    if i%env.SIM_FREQ == 0:
        env.render()
        print(done)
    sync(i, start, env.TIMESTEP)
    if done:
        obs = env.reset()
env.close()
logger.plot()
