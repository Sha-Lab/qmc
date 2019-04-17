#!/usr/bin/python3

import numpy as np
import gym
from gym import spaces
from scipy.linalg import solve_discrete_are

EPS = 1e-8


def random_psd(N, kappa=None, semi=True):
    """
    Returns a NxN random positive semi-definite matrix.

    Arguments:
        * semi (bool) If False, returns positive definite matrix.
          (Default: True)
        * kappa (float or None) Condition number of the matrix, if specified.
    """
    matrix = np.random.randn(N, N)
    Q, R = np.linalg.qr(matrix)
    if kappa is not None:
        evalues = np.linspace(1, kappa, N)
    else:
        if semi:
            min_eig = 0.0
        else:
            min_eig = EPS
        evalues = np.random.uniform(low=min_eig, high=1, size=N)
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
    ):
        super(LQR, self).__init__()
        state_lims = lims * np.ones(N)
        action_lims = lims * np.ones(M)
        self.N = N
        self.M = M
        self.max_steps = max_steps
        self.lims = lims
        self.init_state = np.ones(N) * init_scale
        self.observation_space = spaces.Box(low=-state_lims,
                                            high=state_lims,
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-action_lims,
                                       high=action_lims,
                                       dtype=np.float32)
        self.A = A if A is not None else np.random.randn(N, N) / np.sqrt(N)
        if A_norm is not None:
            self.A *= A_norm / np.linalg.norm(self.A)
        self.B = B if B is not None else np.random.randn(N, M) / np.sqrt(N)
        if B_norm is not None:
            self.B *= B_norm / np.linalg.norm(self.B)
        self.P = P if P is not None else random_psd(M, kappa=P_kappa, semi=False) / np.sqrt(M)
        self.Q = Q if Q is not None else random_psd(N, kappa=Q_kappa, semi=True) / np.sqrt(N)
        self.Sigma_s = Sigma_s if Sigma_s is not None else 0.1 * random_psd(N, kappa=Sigma_s_kappa, semi=True)
        if Sigma_s_scale:
            self.Sigma_s *= Sigma_s_scale ** 2
            self.Sigma_s_L = np.linalg.cholesky(self.Sigma_s)
        else:
            self.Sigma_s = np.zeros((N, N))
            self.Sigma_s_L = np.zeros((N, N))

    def optimal_controller(self): # does not depend on noise...
        K = solve_discrete_are(self.A, self.B, self.Q, self.P)
        L = -np.linalg.inv(self.B.T @ K @ self.B + self.P) @ self.B.T @ K @ self.A
        return L

    def reset(self):
        self.num_steps = 0
        #self.state = np.random.random(size=self.N) * self.lims # minus 1/2?
        self.state = self.init_state
        return self.state

    # you need to input a stable control matrix K
    # only suppose independent action noise, need to input the covariance
    def expected_cost(self, K, Sigma_a):
        x0 = self.init_state
        cost =  0.0
        m_list = [self.Q + K.T.dot(self.P).dot(K)]
        C = self.A + self.B @ K
        for i in range(self.max_steps-1):
            m_list.append(C.T.dot(m_list[-1]).dot(C))
        for t in range(self.max_steps):
            cost += x0.dot(m_list[t]).dot(x0) + np.trace(self.P @ Sigma_a)
            for i in range(t):
                cost += np.trace(self.B.T @ m_list[t-1-i] @ self.B @ Sigma_a)
                cost += np.trace(m_list[t-1-i] @ self.Sigma_s) # environmental noise
        return cost

    # expected state covariance at time t, used for expected policy gradient
    def expected_state_cov(self, t, K, Sigma_a):
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
        noise = self.Sigma_s_L.dot(np.random.randn(self.N))
        next_state = self.A.dot(self.state) + self.B.dot(action) + noise
        state_cost = self.state.dot(self.Q).dot(self.state)
        action_cost = action.dot(self.P).dot(action)
        reward = - state_cost - action_cost
        self.state = next_state
        done = np.sum(self.state > self.lims) > 0
        done += np.sum(self.state < -self.lims) > 0
        info = {}
        self.num_steps += 1
        if self.num_steps >= self.max_steps:
            done = True
        return next_state, reward, done, info
