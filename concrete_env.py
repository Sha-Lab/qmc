from lqr import LQR

# cartpole: https://github.com/neka-nat/ilqr-gym/blob/master/env/cartpole_continuous.py
# It seems that in this example the A, B depends on state in a non-linear way
# inverted pendulum: https://github.com/Nikkhil16/Inverted_Pendulum/blob/master/inverted_pendulum.py

# https://github.com/wiany11/lqr-wip
class WIP(LQR):
    def __init__(self):
        super().__init__()
