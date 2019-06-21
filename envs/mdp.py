import numpy as np
import gym
from gym import spaces

### this is not tested, since we only study continuous control

# this will return a dense transition to ensure it has stationary distribution
def get_random_mdp(n_states, n_actions):
    transition = np.random.rand(n_states, n_actions, n_states)
    transition /= transition.sum(2, keepdims=True)
    reward = np.random.rand(n_states, n_actions)
    reward /= reward.sum(keepdims=True)
    init_dist = np.random.rand(n_states)
    init_dist /= init_dist.sum(keepdims=True)
    return MDP(transition, reward, init_dist)

# transition matrix (SxAxS) and init state distribution (S)
class MDP(gym.Env):
    def __init__(self, transition, reward, init_dist, seed=0):
        super().__init__()
        self.n_states, self.n_actions = transition.shape[:2]
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)
        self.transition = transition
        self.reward = reward # state only
        self.init_dist = init_dist
        self.rng = np.random.RandomState(seed)
        assert self.init_dist.shape[0] == self.transition.shape[0]
        assert self.init_dist.shape[0] == self.transition.shape[2]
        assert self.init_dist.shape[0] == self.reward.shape[0]
        assert np.allclose(self.init_dist.sum(), 1.0)
        assert np.allclose(self.transition.sum(2), 1.0)

    def reset(self):
        self.state = self.rng.choice(self.n_states, p=self.init_dist)
        return self.state

    def step(self, action):
        reward = self.reward[self.state][action]
        self.state = self.rng.choice(self.n_states, p=self.transition[self.state][action])
        return self.state, self.reward[self.state][action], False, {}

    def expected_return(self, policy):
        mc_transition = (self.transition * policy[:,:,np.newaxis]).sum(1)
        mc_reward = (self.reward * policy).sum(1)
        A = np.concatenate([mc_transition.T - np.eye(self.n_states), np.ones(self.n_states)])
        b = np.zeros(self.n_states + 1)
        b[-1] = 1
        stat_dist = np.linalg.inv(A).dot(b) # stationary distribution
        return np.dot(stat_dist, mc_reward)

    def seed(self, seed=None):
        self.rng.seed(seed)

