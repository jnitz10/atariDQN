from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3 import DQN

env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=4, seed=0)

model = DQN('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
