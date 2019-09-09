import gym
import numpy as np


class Brownian(gym.Env):
    def __init__(self):
        self.dt = 0.1
        self.observation_space = gym.spaces.Box(low=np.array([-np.inf]), high=np.array([np.inf]))
        self.action_space = gym.spaces.Box(low=np.array([-np.inf]), high=np.array([np.inf]))

    def reset(self):
        self.x = 0
        return self.x

    def step(self, action):
        self.x += self.dt * action
        return self.x, np.linalg.norm(self.x), False, {}
