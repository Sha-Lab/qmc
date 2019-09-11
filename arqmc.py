import sys
import argparse
import random
import torch
import numpy as np
from pathlib import Path

from envs import Brownian, LQR
#from rqmc_distributions.dist_rqmc import Uniform_RQMC, Normal_RQMC
from rqmc_distributions import Normal_RQMC, Uniform_RQMC
from scipy.stats import norm
from utils import logger

# TODO:
# try arqmc without full trajectory

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['brownian', 'lqr'], default='lqr')
    parser.add_argument('--n_trajs', type=int, default=2 ** 5)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--n_runs', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--algos', type=str, nargs='+', default=['mc', 'rqmc', 'arqmc'])
    parser.add_argument('--exp_name', type=str, default=None)
    return parser.parse_args(args)

def brownian(args):
    ground_truth = 0.2 / np.sqrt(2 * np.pi) * np.sum(np.sqrt(np.arange(1, args.horizon + 1)) * (np.cumprod(args.gamma * np.ones(args.horizon))/ args.gamma))
    env = Brownian(gamma=args.gamma)
    # mc
    if 'mc' in args.algos:
        errors = []
        for _ in range(args.n_runs):
            returns = []
            for i in range(args.n_trajs):
                rs = 0.0
                env.reset()
                done = False
                for _ in range(args.horizon):
                    if done: break
                    _, r, done, _ = env.step(np.random.randn())
                    rs += r
                returns.append(rs)
            res = np.mean(returns)
            errors.append(np.abs(ground_truth - res))
        print('mc: {}, error: {}'.format(res, np.mean(errors)))
    # rqmc
    if 'rqmc' in args.algos:
        errors = []
        for _ in range(args.n_runs):
            returns = [] 
            loc = torch.zeros(args.horizon)
            scale = torch.ones(args.horizon)
            actions = Normal_RQMC(loc, scale, scrambled=True).sample(torch.Size([args.n_trajs])).data.numpy()
            for i in range(args.n_trajs):
                rs = 0.0
                env.reset()
                done = False
                for j in range(args.horizon):
                    if done: break
                    _, r, done, _ = env.step(actions[i][j])
                    rs += r
                returns.append(rs)
            res = np.mean(returns)
            errors.append(np.abs(ground_truth - res))
        print('rqmc: {}, error: {}'.format(res, np.mean(errors)))
    # array rqmc
    if 'arqmc' in args.algos:
        errors = []
        for _ in range(args.n_runs):
            returns = [0.0 for _ in range(args.n_trajs)]
            loc = torch.zeros(1)
            scale = torch.ones(1)
            noises = Uniform_RQMC(loc, scale, scrambled=False).sample(torch.Size([args.n_trajs])).data.numpy()
            envs = [Brownian(args.gamma) for _ in range(args.n_trajs)]
            states = [env.reset() for env in envs]
            dones = [False for _ in range(args.n_trajs)]
            for _ in range(args.horizon):
                if np.all(dones): break
                envs, states, dones, returns = zip(*sorted(zip(envs, states, dones, returns), key=lambda x: np.inf if x[2] else x[1]))
                states, dones, returns = list(states), list(dones), list(returns)
                bias = np.random.rand()
                for i, env in enumerate(envs):
                    if dones[i]: break
                    state, r, done, _ = env.step(norm.ppf((noises[i] + bias) % 1.0))
                    states[i] = state
                    dones[i] = done
                    returns[i] += r
            errors.append(np.abs(ground_truth - np.mean(returns)))
        print('array rqmc: {}, error: {}'.format(res, np.mean(errors)))

def lqr(args):
    if args.exp_name is not None:
        Path('log', args.exp_name).mkdir(parents=True, exist_ok=True)
        logger.get_logger('out', [sys.stdout, open(Path('log', args.exp_name, 'log.txt'), 'w')])
    else:
        logger.get_logger('out', [sys.stdout])
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
            gamma=args.gamma,
        )
    env = get_env()
    K = env.optimal_controller()
    #K = np.random.randn(env.M, env.N)
    sigma_a = np.diag(np.ones(env.M))
    ground_truth = -env.expected_cost(K, sigma_a)
    # mc
    if 'mc' in args.algos:
        errors = []
        for _ in range(args.n_runs):
            returns = []
            for i in range(args.n_trajs):
                rs = 0.0
                done = False
                state = env.reset()
                while not done:
                    state, r, done, _ = env.step(get_action(state, K, np.random.randn(env.M)))
                    rs += r
                returns.append(rs)
            res = np.mean(returns)
            errors.append(np.abs(ground_truth - res))
        logger.log('mc error: {}'.format(np.mean(errors)), name='out')
    # rqmc
    if 'rqmc' in args.algos:
        errors = []
        for _ in range(args.n_runs):
            returns = []
            loc = torch.zeros(env.M * args.horizon)
            scale = torch.ones(env.M * args.horizon)
            noises = Normal_RQMC(loc, scale, scrambled=True).sample(torch.Size([args.n_trajs])).numpy().reshape(args.n_trajs, args.horizon, env.M)
            for i in range(args.n_trajs):
                rs = 0.0
                state = env.reset()
                done = False
                for j in range(args.horizon):
                    if done: break
                    state, r, done, _ = env.step(get_action(state, K, noises[i][j]))
                    rs += r
                returns.append(rs)
            res = np.mean(returns)
            errors.append(np.abs(ground_truth - res))
        logger.log('rqmc error: {}'.format(np.mean(errors)), name='out')
    # array rqmc
    if 'arqmc' in args.algos:
        errors = []
        for _ in range(args.n_runs):
            loc = torch.zeros(1)
            scale = torch.ones(1)
            noises = Uniform_RQMC(loc, scale, scrambled=False).sample(torch.Size([args.n_trajs])).data.numpy()
            envs = [get_env() for _ in range(args.n_trajs)]
            states = [env.reset() for env in envs]
            dones = [False for _ in range(args.n_trajs)]
            returns = [0.0 for _ in range(args.n_trajs)]
            for _ in range(args.horizon):
                if np.all(dones): break
                envs, states, dones, returns = zip(*sorted(zip(envs, states, dones, returns), key=lambda x: np.inf if x[2] else env.expected_cost(K, sigma_a, x0=x[1])))
                states, dones, returns = list(states), list(dones), list(returns)
                bias = np.random.rand()
                for i, env in enumerate(envs):
                    if dones[i]: break
                    noise = norm.ppf((noises[i] + bias) % 1.0)
                    state, r, done, _ = env.step(get_action(states[i], K, noise))
                    states[i] = state
                    dones[i] = done
                    returns[i] += r
            errors.append(np.abs(ground_truth - np.mean(returns)))
        logger.log('array rqmc error: {}'.format(np.mean(errors)), name='out')

        '''
        loc = torch.zeros(env.M)
        scale = torch.ones(env.M)
        noises = Uniform_RQMC(loc, scale, scrambled=False).sample(torch.Size([args.n_trajs])).data.numpy()
        envs = [get_env() for _ in range(args.n_trajs)]
        errors = []
        for _ in range(args.n_runs):
            returns = 0.0
            states = [env.reset() for env in envs]
            for j in range(args.horizon):
                envs, states = zip(*sorted(zip(envs, states), key=lambda x: env.expected_cost(K, sigma_a, x0=x[1])))
                states = list(states)
                bias = np.random.rand(env.M)
                for i, env in enumerate(envs):
                    noise = norm.ppf((noises[i] + bias) % 1.0)
                    state, r, _, _ = env.step(get_action(states[i], K, noise))
                    states[i] = state
                    returns += r
            res = returns / args.n_trajs
            errors.append(np.abs(ground_truth - res))
        logger.log('array rqmc error: {}'.format(np.mean(errors)), name='out')
        '''

def main(args=None):
    args = parse_args(args)
    if args.task == 'brownian':
        brownian(args)
    elif args.task == 'lqr':
        lqr(args)
    else:
        raise Exception('unknown exp type')

if __name__ == "__main__":
    main()
