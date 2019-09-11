import gym
import numpy as np

# gamma is the probability to end at each step
class Brownian(gym.Env):
    def __init__(self, gamma=1.0):
        self.dt = 0.1
        self.observation_space = gym.spaces.Box(low=np.array([-np.inf]), high=np.array([np.inf]))
        self.action_space = gym.spaces.Box(low=np.array([-np.inf]), high=np.array([np.inf]))
        self.gamma = gamma

    def reset(self):
        self.x = 0
        return self.x

    def step(self, action):
        self.x += self.dt * action
        done = np.random.rand() > self.gamma
        return self.x, np.linalg.norm(self.x), done, {}
