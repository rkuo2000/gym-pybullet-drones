import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary

#### Check the environment's spaces ########################
env = gym.make("takeoff-aviary-v0")     # 起飛
#env = gym.make("hover-aviary-v0")       # 旋停
#env = gym.make("flythrugate-aviary-v0") # 穿越

print("[INFO] Action space:", env.action_space)
print("[INFO] Observation space:", env.observation_space)
check_env(env, warn=True, skip_render_check=True)

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
model.save('drone_a2c')

