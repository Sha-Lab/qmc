import unittest
import numpy as np
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


if __name__ == "__main__":
    unittest.main()
