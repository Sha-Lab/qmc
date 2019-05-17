import numpy as np
from lqr import LQR

# cartpole: https://github.com/neka-nat/ilqr-gym/blob/master/env/cartpole_continuous.py
# It seems that in this example the A, B depends on state in a non-linear way

# https://github.com/wiany11/lqr-wip
class WIP(LQR):
    def __init__(
        self,
        init_scale,
        max_steps,
        Sigma_s_scale=0.0,
        seed=0,
    ):
        M = 25.0 # ball mass
        m = 5.0 # pole mass
        l = 2.0 # pole length
        g = 9.80665 # gravity
        A = np.asarray(
            [[0,                1,  0,  0   ],
             [((M+m)*g)/(M*l),  0,  0,  0   ],
             [0,                0,  0,  1   ],
             [-m*g/M,           0,  0,  0   ]])
        B = np.asarray(
            [[0,    0,          0,  0       ],
             [0,    -1.0/(M*l), 0,  0       ],
             [0,    0,          0,  0       ],
             [0,    0,          0,  1.0/M   ]])
        P = np.asarray(
            [[1.0,  0,      0,      0   ],
             [0,    1.0,    0,      0   ],
             [0,    0,      1.0,    0   ],
             [0,    0,      0,      1.0 ]]) / 250
        Q = np.asarray(
            [[1.0,  0,      0,      0       ],
             [0,    1.0,    0,      0       ],
             [0,    0,      250.0,   0       ],
             [0,    0,      0,      100.0    ]]) / 250
        print('A:', np.linalg.norm(A), np.linalg.cond(A))
        print('B:', np.linalg.norm(B), np.linalg.cond(B))
        print('P:', np.linalg.norm(P), np.linalg.cond(P))
        print('Q:', np.linalg.norm(Q), np.linalg.cond(Q))
        super().__init__(*B.shape, A=A, B=B, P=P, Q=Q,
            init_scale=init_scale, max_steps=max_steps, Sigma_s_scale=Sigma_s_scale, seed=seed)

# inverted pendulum: https://github.com/Nikkhil16/Inverted_Pendulum/blob/master/inverted_pendulum.py
class InvertedPendulum(LQR):
     def __init__(
        self,
        init_scale,
        max_steps,
        Sigma_s_scale=0.0,
        seed=0,
    ):
        g = 9.8
        cart_mass = 10.0
        ball_mass = 1.0
        pendulum_length = 1.0
        A = np.array([
            [0,1,0,0],
            [0,0,g*ball_mass/cart_mass,0],
            [0,0,0,1],
            [0,0,(cart_mass+ball_mass)*g/(pendulum_length*cart_mass),0]
        ])      
        B = np.array([
            [0],
            [1/cart_mass],
            [0],
            [1/(pendulum_length*cart_mass)]
        ])
        P = np.array([[500]]) / 10000
        Q = np.array([
            [10,0,0,0],
            [0,1,0,0],
            [0,0,10000,0],
            [0,0,0,100]
        ]) / 10000
        print('A:', np.linalg.norm(A), np.linalg.cond(A))
        print('B:', np.linalg.norm(B), np.linalg.cond(B))
        print('P:', np.linalg.norm(P), np.linalg.cond(P))
        print('Q:', np.linalg.norm(Q), np.linalg.cond(Q))
        super().__init__(*B.shape, A=A, B=B, P=P, Q=Q,
            init_scale=init_scale, max_steps=max_steps, Sigma_s_scale=Sigma_s_scale, seed=seed)
   
