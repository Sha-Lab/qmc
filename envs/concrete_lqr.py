from lqr import LQR

# cartpole: https://github.com/neka-nat/ilqr-gym/blob/master/env/cartpole_continuous.py
# It seems that in this example the A, B depends on state in a non-linear way
# inverted pendulum: https://github.com/Nikkhil16/Inverted_Pendulum/blob/master/inverted_pendulum.py



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
        A = [[0,                1,  0,  0   ],
             [((M+m)*g)/(M*l),  0,  0,  0   ],
             [0,                0,  0,  1   ],
             [-m*g/M,           0,  0,  0   ]]
        B = [[0,    0,          0,  0       ],
             [0,    -1.0/(M*l), 0,  0       ],
             [0,    0,          0,  0       ],
             [0,    0,          0,  1.0/M   ]]
        P = [[1.0,  0,      0,      0   ],
             [0,    1.0,    0,      0   ],
             [0,    0,      1.0,    0   ],
             [0,    0,      0,      1.0 ]]
        Q = [[1.0,  0,      0,      0       ],
             [0,    1.0,    0,      0       ],
             [0,    0,      250.0,   0       ],
             [0,    0,      0,      100.0    ]]
        super().__init__(4, 4, A, B, P, Q, init_scale=init_scale, max_steps=max_steps, Sigma_s_sclae=Sigma_s_scale, seed=seed)
