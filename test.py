import argparse
import torch
import numpy as np
from envs import Brownian, LQR
#from rqmc_distributions.dist_rqmc import Uniform_RQMC, Normal_RQMC
from rqmc_distributions import Normal_RQMC, Uniform_RQMC
from scipy.stats import norm

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', choices=['brownian', 'lqr'], default='lqr')
    parser.add_argument('--n_trajs', type=int, default=2 ** 5)
    parser.add_argument('--horizon', type=int, defualt=10)
    parser.add_argument('--n_runs', type=int, default=10)
    return parser.parse_args(args)


def brownian(args):
    ground_truth = 0.2 / np.sqrt(2 * np.pi) * np.sum(np.sqrt(np.arange(1, args.horizon + 1)))

    env = Brownian()

    # mc
    returns = []
    for i in range(args.n_trajs):
        rs = 0.0
        env.reset()
        for j in range(args.horizon):
            _, r, _, _ = env.step(np.random.randn())
            rs += r
        returns.append(rs)
    res = np.mean(returns)
    print('mc: {}, error: {}'.format(res, np.abs(ground_truth - res)))

    # rqmc
    returns = []
    
    loc = torch.zeros(args.horizon)
    scale = torch.ones(args.horizon)
    actions = Normal_RQMC(loc, scale, scrambled=True).sample(torch.Size([args.n_trajs])).data.numpy()
    for i in range(args.n_trajs):
        rs = 0.0
        env.reset()
        for j in range(args.horizon):
            _, r, _, _ = env.step(actions[i][j])
            rs += r
        returns.append(rs)
    res = np.mean(returns)
    print('rqmc: {}, error: {}'.format(res, np.abs(ground_truth - res)))

    # array rqmc
    returns = 0.0
    loc = torch.zeros(1)
    scale = torch.ones(1)
    noises = Uniform_RQMC(loc, scale, scrambled=False).sample(torch.Size([args.n_trajs])).data.numpy()
    envs = [Brownian() for _ in range(args.n_trajs)]
    states = [env.reset() for env in envs]
    for j in range(args.horizon):
        envs, states = zip(*sorted(zip(envs, states), key=lambda x: x[1]))
        states = list(states)
        bias = np.random.rand()
        for i, env in enumerate(envs):
            state, r, _, _ = env.step(norm.ppf((noises[i] + bias) % 1.0))
            states[i] = state
            returns += r
    res = returns / args.n_trajs
    print('array rqmc: {}, error: {}'.format(res, np.abs(ground_truth - res)))

def lqr(args):
    def get_action(state, K, noise):
        return np.matmul(K, state) + noise

    def get_env():
        return LQR(
            N=6,
            M=5,
            init_scale=1.0,
            max_steps=args.horizon,
            Sigma_s_kappa=1.0,
            Q_kappa=1.0,
            P_kappa=1.0,
            A_norm=1.0,
            B_norm=1.0,
            Sigma_s_scale=0.0,
        )

    env = get_env()
    K = env.optimal_controller()
    sigma_a = np.diag(np.ones(env.M))
    ground_truth = -env.expected_cost(K, sigma_a)

    # mc
    errors = []
    for _ in range(args.n_runs):
        returns = []
        for i in range(args.n_trajs):
            rs = 0.0
            state = env.reset()
            for j in range(args.horizon):
                state, r, _, _ = env.step(get_action(state, K, np.random.randn(env.M)))
                rs += r
            returns.append(rs)
        res = np.mean(returns)
        errors.append(np.abs(ground_truth - res))
    print('mc error: {}'.format(np.mean(errors)))

    # rqmc
    errors = []
    for _ in range(args.n_runs):
        returns = []
        loc = torch.zeros(env.M * args.horizon)
        scale = torch.ones(env.M * args.horizon)
        noises = Normal_RQMC(loc, scale, scrambled=True).sample(torch.Size([args.n_trajs])).numpy().reshape(args.n_trajs, args.horizon, env.M)
        for i in range(args.n_trajs):
            rs = 0.0
            state = env.reset()
            for j in range(args.horizon):
                state, r, _, _ = env.step(get_action(state, K, noises[i][j]))
                rs += r
            returns.append(rs)
        res = np.mean(returns)
        errors.append(np.abs(ground_truth - res))
    print('rqmc error: {}'.format(np.mean(errors)))

    # array rqmc
    loc = torch.zeros(env.M)
    scale = torch.ones(env.M)
    noises = Uniform_RQMC(loc, scale, scrambled=False).sample(torch.Size([args.n_trajs])).data.numpy()
    envs = [get_env() for _ in range(args.n_trajs)]
    errors = []
    for _ in range(args.n_runs):
        returns = 0.0
        states = [env.reset() for env in envs]
        for j in range(args.horizon):
            envs, states = zip(*sorted(zip(envs, states), key=lambda x: env.expected_cost(K, sigma_a, x0=x[1]))) # whether swap them
            states = list(states)
            bias = np.random.rand(env.M)
            for i, env in enumerate(envs):
                noise = norm.ppf((noises[i] + bias) % 1.0)
                state, r, _, _ = env.step(get_action(states[i], K, noise))
                states[i] = state
                returns += r
        res = returns / args.n_trajs
        errors.append(np.abs(ground_truth - res))
    print('array rqmc error: {}'.format(np.mean(errors)))

def main(args=None):
    args = parse_args(args)
    if args.exp == 'brownian':
        brownian(args)
    elif args.exp == 'lqr':
        lqr(args)
    else:
        raise Exception('unknown exp type')

if __name__ == "__main__":
    main()
    #brownian()
    #lqr()
