import gymnasium as gym
from agent import Agent

GAMMA = 0.99
EPSILON = 1.0
LEARNING_RATE = 0.00025
MEM_SIZE = 100000
BATCH_SIZE = 32
EPS_MIN = 0.1
EPS_DEC = 5e-7
CURRENT_GAME = 'ALE/Breakout-v5'
num_games = 100

env = gym.make('ALE/Breakout-v5', frameskip=1, render_mode='human')
env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, terminal_on_life_loss=True, scale_obs=True)
env = gym.wrappers.FrameStack(env, 4)
obs, _ = env.reset()
n_actions = env.action_space.n

agent = Agent(current_game=CURRENT_GAME, epsilon=EPSILON, lr=LEARNING_RATE, n_actions=n_actions,
              input_dims=obs.shape, mem_size=MEM_SIZE, batch_size=BATCH_SIZE,
              eps_min=EPS_MIN, eps_dec=EPS_DEC, gamma=GAMMA, chkpt_dir='tmp/')
agent.load_checkpoint()

for i in range(num_games + 1):
    done = False
    score = 0
    obs, _ = env.reset()
    while not done:
        action = agent.play(obs)
        obs_, reward, done, info, _ = env.step(action)
        score += reward
        obs = obs_
    print('episode', i, 'score %.1f' % score)