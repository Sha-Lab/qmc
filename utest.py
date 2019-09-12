import unittest
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from envs import *

class TestControl(unittest.TestCase):
    def test_optimal_policy(self):
        env = InvertedPendulum(init_scale=1.0, max_steps=100)
        K = env.optimal_controller()
        state = env.reset()
        done = False
        print(state)
        while not done:
            state, r, done, _ = env.step(K.dot(state))
            print(state)
            print(np.linalg.norm(state), r)

class TestDistribution(unittest.TestCase):
    def test_dist(self):
        from rqmc_distributions.dist_rqmc import Normal_RQMC, Uniform_RQMC
        dist = Normal_RQMC(5)
        print(dist.sample(10))

class TestLQR(unittest.TestCase):
    def test_expected_cost(self):
        env = LQR(max_steps=100)
        K = env.optimal_controller()
        Sigma_a = np.diag(np.ones(5))
        cost_gt = env.expected_cost(K, Sigma_a)
        cost_f = env.expected_cost_state_func(K, Sigma_a)()
        print(cost_gt - cost_f)
        x0 = np.random.rand(env.N)
        cost_gt = env.expected_cost(K, Sigma_a, x0=x0)
        cost_f = env.expected_cost_state_func(K, Sigma_a)(x0)
        print(cost_gt - cost_f)

if __name__ == "__main__":
    unittest.main()
