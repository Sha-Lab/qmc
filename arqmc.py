import sys
import argparse
import random
import torch
import chaospy
import numpy as np
from pathlib import Path
from ipdb import launch_ipdb_on_exception

import exps.utils.run
from envs import Brownian, LQR
#from rqmc_distributions.dist_rqmc import Uniform_RQMC, Normal_RQMC
from rqmc_distributions import Normal_RQMC, Uniform_RQMC
from scipy.stats import norm
from models import GaussianPolicy
from utils import logger, set_seed, ssj_uniform, uniform2normal, random_shift
from utils import sort_by_optimal_value, sort_by_norm, multdim_sort, no_sort, random_permute

# TODO:
# try arqmc without full trajectory

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    #parser.add_argument('--task', choices=['estimate_cost', 'learn'], default='estimate_cost')
    parser.add_argument('--env', choices=['brownian', 'lqr'], default='lqr')
    parser.add_argument('--n_trajs', type=int, default=2 ** 5)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--n_runs', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--sorter', nargs='+', default=['value'], choices=['value', 'norm', 'none', 'permute', 'group'])
    parser.add_argument('--algos', type=str, nargs='+', default=['mc', 'rqmc', 'arqmc'])
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    return exps.utils.run.parse_args(parser, args, exp_name_attr='exp_name')

### tasks ### (estimate cost, learn)

### envs ### (see args.env)

### sampler ###
        
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
        logger.log('mc error: {}({})'.format(np.mean(errors), np.std(errors)), name='out')
    # rqmc
    if 'rqmc' in args.algos:
        errors = []
        for _ in range(args.n_runs):
            returns = [] 
            #actions = Normal_RQMC(torch.zeros(args.horizon), torch.ones(args.horizon), scrambled=True).sample(torch.Size([args.n_trajs])).data.numpy()
            actions = uniform2normal(ssj_uniform(args.n_trajs, args.horizon))
            
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
        logger.log('rqmc error: {}({})'.format(np.mean(errors), np.std(errors)), name='out')
    # array rqmc
    if 'arqmc' in args.algos:
        errors = []
        for _ in range(args.n_runs):
            returns = [0.0 for _ in range(args.n_trajs)]
            envs = [Brownian(args.gamma) for _ in range(args.n_trajs)]
            states = [env.reset() for env in envs]
            dones = [False for _ in range(args.n_trajs)]
            uniform_noises = ssj_uniform(args.n_trajs, 1) # n_trajs , action_dim
            noises = uniform2normal(random_shift(np.expand_dims(uniform_noises, 1).repeat(args.horizon, 1), 0)) # n_trajs, horizon, action_dim
            import ipdb; ipdb.set_trace()
            for j in range(args.horizon):
                if np.all(dones): break
                envs, states, dones, returns = zip(*sorted(zip(envs, states, dones, returns), key=lambda x: np.inf if x[2] else x[1]))
                states, dones, returns = list(states), list(dones), list(returns)
                #noises = Normal_RQMC(torch.zeros(1), torch.ones(1)).sample(torch.Size([args.n_trajs])).data.numpy()
                #noises = uniform2normal(random_shift(uniform_noises))
                for i, env in enumerate(envs):
                    if dones[i]: break
                    state, r, done, _ = env.step(noises[i][j])
                    states[i] = state
                    dones[i] = done
                    returns[i] += r
            errors.append(np.abs(ground_truth - np.mean(returns)))
        logger.log('array rqmc error: {}({})'.format(np.mean(errors), np.std(errors)), name='out')

def get_sorter(sorter, env):
    if sorter == 'value':
        return lambda pairs: sorted(pairs, key=sort_by_optimal_value(env))
    elif sorter == 'norm':
        return lambda pairs: sorted(pairs, key=sort_by_norm(env))
    elif sorter == 'permute':
        return random_permute
    elif sorter == 'none':
        return no_sort
    elif sorter == 'group':
        return multdim_sort
    else:
        raise Exception('unknown sorter')

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
        logger.log('mc error: {}({})'.format(np.mean(errors), np.std(errors)), name='out')
    # rqmc
    if 'rqmc' in args.algos:
        errors = []
        for _ in range(args.n_runs):
            returns = []
            #noises = Normal_RQMC(
                #torch.zeros(env.M * args.horizon), 
                #torch.ones(env.M * args.horizon), scrambled=True).sample(torch.Size([args.n_trajs])).numpy().reshape(args.n_trajs, args.horizon, env.M)
            noises = uniform2normal(ssj_uniform(args.n_trajs, args.horizon * env.M)).reshape(args.n_trajs, args.horizon, env.M)
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
        logger.log('rqmc error: {}({})'.format(np.mean(errors), np.std(errors)), name='out')
    # array rqmc
    if 'arqmc' in args.algos:
        for sorter_type in args.sorter:
            sorter = get_sorter(sorter_type, env)
            errors = []
            for _ in range(args.n_runs):
                envs = [get_env() for _ in range(args.n_trajs)]
                states = [env.reset() for env in envs]
                dones = [False for _ in range(args.n_trajs)]
                returns = [0.0 for _ in range(args.n_trajs)]
                uniform_noises = ssj_uniform(args.n_trajs, env.M) # it is also ok to use the same set of uniform noises for different runs
                noises = uniform2normal(random_shift(np.expand_dims(uniform_noises, 1).repeat(args.horizon, 1), 0))
                for j in range(args.horizon):
                    if np.all(dones): break
                    pairs = list(zip(envs, states, dones, returns))
                    pairs_to_sort = [p for p in pairs if not p[2]]
                    pairs_done = [p for p in pairs if p[2]]
                    envs, states, dones, returns = zip(*( sorter(pairs_to_sort) + pairs_done ))
                    states, dones, returns = list(states), list(dones), list(returns)
                    #noises = Normal_RQMC(torch.zeros(env.M), torch.ones(env.M), scrambled=True).sample(torch.Size([args.n_trajs])).numpy()
                    #noises = uniform2normal(random_shift(uniform_noises))
                    for i, env in enumerate(envs):
                        if dones[i]: break 
                        state, r, done, _ = env.step(get_action(states[i], K, noises[i][j]))
                        states[i] = state
                        dones[i] = done
                        returns[i] += r
                errors.append(np.abs(ground_truth - np.mean(returns)))
            logger.log('array rqmc error ({}): {}({})'.format(sorter_type, np.mean(errors), np.std(errors)), name='out')

def main(args=None):
    args = parse_args(args)
    if args.seed is None:
        args.seed = np.random.randint(0, 10000000)
        logger.prog('randomly selected seed: {}'.format(args.seed))
    set_seed(args.seed)
    if args.exp_name is not None:
        Path('log', args.exp_name).mkdir(parents=True, exist_ok=True)
        logger.get_logger('out', [sys.stdout, open(Path('log', args.exp_name, 'log.txt'), 'w')])
    else:
        logger.get_logger('out', [sys.stdout])
    if args.env == 'brownian':
        brownian(args)
    elif args.env == 'lqr':
        lqr(args)
    else:
        raise Exception('unknown exp type')

if __name__ == "__main__":
    with launch_ipdb_on_exception():
        main()
