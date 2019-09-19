#!/usr/bin/python3

import numpy as np
import gym
from gym import spaces
from scipy.linalg import solve_discrete_are

EPS = 1e-8


def random_psd(N, kappa=None, semi=True, rng=None):
    """
    Returns a NxN random positive semi-definite matrix.

    Arguments:
        * semi (bool) If False, returns positive definite matrix.
          (Default: True)
        * kappa (float or None) Condition number of the matrix, if specified.
    """
    if rng is None: rng = np.random
    matrix = rng.randn(N, N)
    Q, R = np.linalg.qr(matrix)
    if kappa is not None:
        evalues = np.linspace(1, kappa, N)
    else:
        if semi:
            min_eig = 0.0
        else:
            min_eig = EPS
        evalues = rng.uniform(low=min_eig, high=1, size=N)
    evalues = np.diag(evalues)
    matrix = Q.dot(evalues).dot(Q.transpose())
    return matrix


class LQR(gym.Env):

    """
    Instanciates a LQR problem, with state transition noise.
    The environment terminates when its boundaries are crossed,
    or when a maximum number of steps have been executed.

    Arguments:
        * N (int) Dimensionality of state space.
        * M (int) Dimensionality of action space.
        * lims (float) Boundary value for any dimension of action/state space.
        * A, B, Q, P, or Sigma_s (ndarray) Matrices of the environment.
        * A_norm (float) L2-norm of the A matrix.
        * B_norm (float) L2-norm of the B matrix.
        * Q_kappa (float) Condition number of matrix Q.
        * P_kappa (float) Condition number of matrix P.
        * Sigma_s_kappa (float) Condition number of matrix Sigma_s.
    """

    def __init__(
        self,
        N=5,
        M=5,
        lims=1000000000.0,
        init_scale=100.0,
        max_steps=10000,
        A=None,
        A_norm=None,
        B=None,
        B_norm=None,
        P=None,
        P_kappa=1.0,
        Q=None,
        Q_kappa=1.0,
        Sigma_s=None,
        Sigma_s_kappa=1.0,
        Sigma_s_scale=1.0,
        random_init=False,
        gamma=1.0,
        seed=0,
    ):
        super(LQR, self).__init__()
        self.rng = np.random.RandomState(seed)
        state_lims = lims * np.ones(N)
        action_lims = lims * np.ones(M)
        self.N = N
        self.M = M
        self.max_steps = max_steps
        self.lims = lims
        self.random_init = random_init
        self.init_state = np.ones(N) * init_scale
        self.observation_space = spaces.Box(low=-state_lims,
                                            high=state_lims,
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-action_lims,
                                       high=action_lims,
                                       dtype=np.float32)
        self.A = A if A is not None else self.rng.randn(N, N) / np.sqrt(N)
        if A_norm is not None:
            self.A *= A_norm / np.linalg.norm(self.A)
        self.B = B if B is not None else self.rng.randn(N, M) / np.sqrt(N)
        if B_norm is not None:
            self.B *= B_norm / np.linalg.norm(self.B)
        self.P = P if P is not None else random_psd(M, kappa=P_kappa, semi=False, rng=self.rng) / np.sqrt(M)
        self.Q = Q if Q is not None else random_psd(N, kappa=Q_kappa, semi=True, rng=self.rng) / np.sqrt(N)
        self.Sigma_s = Sigma_s if Sigma_s is not None else 0.1 * random_psd(N, kappa=Sigma_s_kappa, semi=True, rng=self.rng)
        if Sigma_s_scale:
            self.Sigma_s *= Sigma_s_scale ** 2
            self.Sigma_s_L = np.linalg.cholesky(self.Sigma_s)
        else:
            self.Sigma_s = np.zeros((N, N))
            self.Sigma_s_L = np.zeros((N, N))
        self.gamma = gamma

    def seed(self, seed=None):
        self.rng.seed(seed)

    # does not depend on noise, optimal for infinite horizon
    def optimal_controller(self):
        K = solve_discrete_are(self.A, self.B, self.Q, self.P)
        L = -np.linalg.inv(self.B.T @ K @ self.B + self.P) @ self.B.T @ K @ self.A
        return L

    def reset(self):
        self.num_steps = 0
        if self.random_init:
            self.state = self.rng.randn(self.N)
            self.state /= np.linalg.norm(self.state)
            self.state *= self.init_state[0] # hack!
        else:
            self.state = self.init_state
        return self.state

    # build a function that calculate expected cost, with initial state as input
    def expected_cost_state_func(self, K, Sigma_a, T=None):
        if T is None:
            T = self.max_steps
        const = 0.0
        init_q = 0.0
        m_list = [self.Q + K.T.dot(self.P).dot(K)]
        C = self.A + self.B @ K
        for i in range(T-1):
            m_list.append(C.T.dot(m_list[-1]).dot(C))
        discount = 1.0
        for t in range(T):
            const_t = np.trace(self.P @ Sigma_a)
            for i in range(t):
                const_t += np.trace(self.B.T @ m_list[t-1-i] @ self.B @ Sigma_a)
                const_t += np.trace(m_list[t-1-i] @ self.Sigma_s) # environmental noise
            const += discount * const_t
            init_q += discount * m_list[t]
            discount *= self.gamma
        def f(x0=None):
            if x0 is None:
                x0 = self.init_state
            return const + x0.dot(init_q).dot(x0)
        return f

    # you need to input a stable control matrix K
    # only support independent action noise, need to input the covariance
    def expected_cost(self, K, Sigma_a, x0=None, T=None):
        return self.expected_cost_state_func(K, Sigma_a, T=T)(x0)

    # expected state covariance at time t, used for expected policy gradient
    def expected_state_cov(self, t, K, Sigma_a):
        assert self.gamma == 1.0
        abk = self.A + self.B @ K
        abk_list = [np.identity(self.N)]
        for _ in range(t):
            abk_list.append(abk @ abk_list[-1])
        cov = abk_list[t] @ np.outer(self.init_state, self.init_state) @ abk_list[t].T
        for i in range(t):
            c_s = abk_list[t-1-i] @ self.Sigma_s @ abk_list[t-1-i].T
            c_a = abk_list[t-1-i] @ self.B @ Sigma_a @ self.B.T @ abk_list[t-1-i].T
            cov += c_s + c_a
        return cov

    def expected_policy_gradient(self, K, Sigma_a):
        assert self.gamma == 1.0
        assert not self.random_init, 'policy gradient for random init has not been implemented'
        conv = [self.expected_state_cov(t, K, Sigma_a) for t in range(self.max_steps)]
        abk = self.A + self.B @ K
        qkp = self.Q + K.T @ self.P @ K
        abk_list = [np.identity(self.N)]
        for _ in range(self.max_steps):
            abk_list.append(abk @ abk_list[-1])
        grad = 0.0
        for t in range(self.max_steps):
            grad += 2 * Sigma_a @ self.P @ K @ conv[t]
            for tt in range(t+1, self.max_steps):
                grad += 2 * Sigma_a @ self.B.T @ abk_list[tt-1-t].T @ qkp @ abk_list[tt-t] @ conv[t]
        return -np.linalg.inv(Sigma_a) @ grad # in derivation I use formula for cost

    def step(self, action):
        noise = self.Sigma_s_L.dot(self.rng.randn(self.N))
        next_state = self.A.dot(self.state) + self.B.dot(action) + noise
        state_cost = self.state.dot(self.Q).dot(self.state)
        action_cost = action.dot(self.P).dot(action)
        reward = - state_cost - action_cost
        self.state = next_state
        done = np.sum(self.state > self.lims) > 0
        done += np.sum(self.state < -self.lims) > 0
        if done: reward -= 1000.0 # incur high ending penalty
        info = {}
        self.num_steps += 1
        if self.num_steps >= self.max_steps or np.random.rand() > self.gamma:
            done = True
        return next_state, reward, done, info
