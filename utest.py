import unittest
from lqr import LQR

class TestControl(unittest.TestCase):
    def test_optimal_policy(self):
        import control
        env = LQR()
        A, B, Q, R = env.A, env.B, env.Q, env.P
        print(env.optimal_controller())
        K, S, E = control.lqr(A, B, Q, R)
        print(K)
        print(S)
        print(E)

if __name__ == "__main__":
    unittest.main()
