import unittest
from lqr import LQR
from envs import *

class TestControl(unittest.TestCase):
    def test_optimal_policy(self):
        env = InvertedPendulum(init_scale=1.0, max_steps=100)
        K = env.optimal_controller()
        state = env.reset()
        done = False
        while not done:
            state, r, done, _ = env.step(K.dot(state))
            print(r)

if __name__ == "__main__":
    unittest.main()
