import gymnasium as gym
import sys
from agent import Agent
import os

import numpy as np

GAMMA = 0.99
EPSILON = 1.0
LEARNING_RATE = 0.00025
MEM_SIZE = 100000
BATCH_SIZE = 32
EPS_MIN = 0.1
EPS_DEC = 5e-7
CURRENT_GAME = 'ALE/Breakout-v5'


if __name__ == '__main__':
    env = gym.make(CURRENT_GAME, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True,
                                          terminal_on_life_loss=True)
    env = gym.wrappers.FrameStack(env, 4, lz4_compress=True)
    obs, _ = env.reset()
    n_actions = env.action_space.n
    n_games = 1000
    scores, eps_history = [], []

    agent = Agent(current_game=CURRENT_GAME, epsilon=EPSILON, lr=LEARNING_RATE, n_actions=n_actions,
                  input_dims=obs.shape, mem_size=MEM_SIZE, batch_size=BATCH_SIZE,
                  eps_min=EPS_MIN, eps_dec=EPS_DEC, gamma=GAMMA, chkpt_dir='tmp/')

    for i in range(n_games + 1):
        done = False
        score = 0
        obs, _ = env.reset()
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info, _ = env.step(action)
            score += reward
            agent.store_transition(obs, action, reward, obs_, int(done))
            agent.learn()
            obs = obs_
        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            agent.save_checkpoint()

        avg_score = np.mean(scores[-100:])
        print('episode', agent.episode_cntr, 'score %.1f' % score,
              'average score %.1f' % avg_score,
              'epsilon %.7f' % agent.epsilon)

        agent.increment_episode()

    agent.close_writer()





