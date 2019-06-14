import sys
import gym
import torch
import copy
import dill
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm, trange
from ipdb import slaunch_ipdb_on_exception
from pathlib import Path

from envs import *
from models import GaussianPolicy, get_mlp
from utils import set_seed, rollout, mse, cummean, Sampler, select_device, tensor, reinforce_loss, variance_reduced_loss, no_loss
from torch.distributions import Uniform, Normal
from rqmc_distributions import Uniform_RQMC, Normal_RQMC

# TODO: 
# check infinite horizon value estimation in MDP
# value estimation with critic
# run on general environment (zerobaseline, then actor critic)
# read LQR paper to learn the proof
# check torch's multiprocessing, it might cost problems for sampler
# make vectorized sampler to support gpu samping (multiprocessing with one gpu is not efficient)
# how to quickly cut unpromising configuration?

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task', 
        choices=['cost', 'grad', 'learn', 'inf'], 
        default='learn')
    parser.add_argument('--env', choices=['lqr', 'WIP', 'IP', 'mdp'], default='lqr')
    parser.add_argument('--xu_dim', type=int, nargs=2, default=(20, 12))
    parser.add_argument('--init_scale', type=float, default=3.0)
    parser.add_argument('--PQ_kappa', type=float, default=3.0)
    parser.add_argument('--AB_norm', type=float, default=1.0)
    parser.add_argument('-H', type=int, default=10, help='horizon')
    parser.add_argument('--noise', type=float, default=0.0, help='noise scale')
    parser.add_argument('--n_trajs', type=int, default=800, help='number of trajectories used')
    parser.add_argument('--n_iters', type=int, default=200, help='number of iterations of training')
    parser.add_argument('-lr', type=float, default=5e-5)
    parser.add_argument('--init_policy', choices=['optimal', 'linear', 'linear_bias', 'mlp'], default='linear')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_fig', action='store_true')
    parser.add_argument('--save_fig', type=str, default=None)
    parser.add_argument('--mode', choices=['single', 'over', 'collect'], default='single')
    parser.add_argument('--n_seeds', type=int, default=200)
    parser.add_argument('--max_seed', type=int, default=100)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--save_fn', type=str, default=None)
    return parser.parse_args(args)

def get_env(args):
    if args.env == 'lqr':
        env = LQR(
            N=args.xu_dim[0],
            M=args.xu_dim[1],
            init_scale=args.init_scale,
            max_steps=args.H,
            Sigma_s_kappa=1.0,
            Q_kappa=args.PQ_kappa,
            P_kappa=args.PQ_kappa,
            A_norm=args.AB_norm,
            B_norm=args.AB_norm,
            Sigma_s_scale=args.noise,
            #random_init=True,
        )
    elif args.env == 'WIP':
        env = WIP(
            init_scale=args.init_scale,
            max_steps=args.H,
            Sigma_s_scale=args.noise,
        )
    elif args.env == 'IP':
        env = InvertedPendulum(
            init_scale=args.init_scale,
            max_steps=args.H,
            Sigma_s_scale=args.noise,
        )
    elif args.env == 'mdp':
        # 3 states, 2 actions
        transition = np.array([
            [[0.1, 0.9, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.2, 0.8], [0.5, 0.0, 0.5]],
            [[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]],
        ])
        reward = np.array([
            [0.0, 0.5],
            [1.0, -1.0],
            [0.5, 0.0],
        ])
        init_dist = np.array([1.0, 0.0, 0.0])
        env = MDP(transition, reward, init_dist)
    else:
        raise Exception('unsupported lqr env')
    return env

def get_policy(args, env):    
    if args.init_policy == 'optimal':
        K = env.optimal_controller()
        mean_network = nn.Linear(*K.shape[::-1], bias=False)
        mean_network.weight.data = tensor(K)
    elif args.init_policy == 'linear':
        K = np.random.randn(env.M, env.N)
        mean_network = nn.Linear(*K.shape[::-1], bias=False)
        mean_network.weight.data = tensor(K)
    elif args.init_policy == 'linear_bias':
        K = np.random.randn(env.M, env.N)
        mean_network = nn.Linear(*K.shape[::-1], bias=True)
        mean_network.weight.data = tensor(K)
    elif args.init_policy == 'mlp':
        mean_network = get_mlp((env.N, 16, env.M), gate=nn.ReLU)
    else:
        raise Exception('unsupported policy type')
    return GaussianPolicy(env.N, env.M, mean_network)

# error bar: https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars
#def compare_cost(horizon=100, num_trajs=1000, noise_scale=0.0, seed=0, save_dir=None, show_fig=False):
def compare_cost(args):
    set_seed(args.seed)
    env = LQR(
        N=20,
        M=12,
        init_scale=1.0,
        #max_steps=100,
        max_steps=10,
        Sigma_s_kappa=1.0,
        Q_kappa=1.0,
        P_kappa=1.0,
        A_norm=1.0,
        B_norm=1.0,
        Sigma_s_scale=0.0,
    )
    K = env.optimal_controller()
    mean_network = nn.Linear(*K.shape[::-1], bias=False)
    mean_network.weight.data = tensor(K)
    policy = GaussianPolicy(*K.shape[::-1], mean_network)

    mc_costs = [] # individual
    mc_means = [] # cumulative
    for i in tqdm(range(args.n_trajs), 'mc'):
        noises = np.random.randn(env.max_steps, env.M)
        _, _, rewards = rollout(env, policy, noises)
        mc_costs.append(-rewards.sum())
        mc_means.append(np.mean(mc_costs))

    rqmc_costs = []
    rqmc_means = []
    loc = torch.zeros(env.max_steps * env.M)
    scale = torch.ones(env.max_steps * env.M)
    rqmc_noises = Normal_RQMC(loc, scale).sample(torch.Size([args.n_trajs])).data.numpy()
    for i in tqdm(range(args.n_trajs), 'rqmc'):
        _, _, rewards = rollout(env, policy, rqmc_noises[i].reshape(env.max_steps, env.M))
        rqmc_costs.append(-rewards.sum())
        rqmc_means.append(np.mean(rqmc_costs))

    expected_cost = env.expected_cost(K, np.diag(np.ones(env.M)))

    mc_errors = np.abs(mc_means - expected_cost)
    rqmc_errors = np.abs(rqmc_means - expected_cost)
    info = {**vars(args), 'mc_costs': mc_costs, 'rqmc_costs': rqmc_costs}
    if args.save_fn is not None:
        with open(args.save_fn, 'wb') as f:
            dill.dump(dict(mc_errors=mc_errors, rqmc_errors=rqmc_errors, info=info), f)
    if args.show_fig:
        data = pd.concat([
            pd.DataFrame({
                'name': 'mc',
                'x': np.arange(len(mc_errors)),
                'error': mc_errors,
            }),
            pd.DataFrame({
                'name': 'rqmc',
                'x': np.arange(len(rqmc_errors)),
                'error': rqmc_errors,
            }),
        ])
        plot = sns.lineplot(x='x', y='error', hue='name', data=data)
        plot.set(yscale='log')
        plt.show()
    return mc_errors, rqmc_errors, info

#def compare_grad(horizon, num_trajs, noise_scale=0.0, seed=0, save_dir=None, show_fig=False):
def compare_grad(args):
    set_seed(args.seed)
    env = LQR(
        lims=100,
        init_scale=1.0,
        max_steps=args.H,
        Sigma_s_kappa=1.0,
        Q_kappa=1.0,
        P_kappa=1.0,
        A_norm=1.0,
        B_norm=1.0,
        Sigma_s_scale=args.noise,
    )
    #K = env.optimal_controller()
    K = np.random.randn(env.M, env.N) # debug, this one seems to work worse, by 1 magnitude
    mean_network = nn.Linear(*K.shape[::-1], bias=False)
    mean_network.weight.data = tensor(K)
    policy = GaussianPolicy(*K.shape[::-1], mean_network)

    Sigma_a = np.diag(np.ones(env.M))
    Sigma_a_inv = np.linalg.inv(Sigma_a)
    print(env.Sigma_s)
    mc_grads = []
    for i in tqdm(range(args.n_trajs), 'mc'):
        noises = np.random.randn(env.max_steps, env.M)
        states, actions, rewards = rollout(env, policy, noises)
        mc_grads.append(policy_gradient(states, actions, rewards, policy))
        origin_grad = Sigma_a_inv @ (actions - states @ K.T).T @ states * rewards.sum()
        #print(mc_grads[-1] - origin_grad)
        #exit()
        #mc_grads.append(Sigma_a_inv @ (actions - states @ K.T).T @ states * rewards.sum()) # need minus since I use cost formula in derivation
    mc_grads = np.asarray(mc_grads)
    mc_means = np.cumsum(mc_grads, axis=0) / np.arange(1, len(mc_grads) + 1)[:, np.newaxis, np.newaxis]
    
    rqmc_grads = []
    loc = torch.zeros(env.max_steps * env.M)
    scale = torch.ones(env.max_steps * env.M)
    rqmc_noises = Normal_RQMC(loc, scale).sample(torch.Size([args.n_trajs])).data.numpy()
    for i in tqdm(range(args.n_trajs), 'rqmc'):
        states, actions, rewards = rollout(env, policy, rqmc_noises[i].reshape(env.max_steps, env.M))
        rqmc_grads.append(policy_gradient(states, actions, rewards, policy))
        #rqmc_grads.append(Sigma_a_inv @ (actions - states @ K.T).T @ states * rewards.sum())
    rqmc_grads = np.asarray(rqmc_grads)
    rqmc_means = np.cumsum(rqmc_grads, axis=0) / np.arange(1, len(rqmc_grads) + 1)[:, np.newaxis, np.newaxis]

    expected_grad = env.expected_policy_gradient(K, Sigma_a)

    mc_errors = ((mc_means - expected_grad) ** 2).reshape(mc_means.shape[0], -1).mean(1) # why the sign is reversed?
    rqmc_errors = ((rqmc_means - expected_grad) ** 2).reshape(rqmc_means.shape[0], -1).mean(1)
    info = {**vars(args)}
    if args.save_fn is not None:
        with open(save_fn, 'wb') as f:
            dill.dump(dict(mc_errors=mc_errors, rqmc_errors=rqmc_errors, info=info), f)
    if args.show_fig:
        mc_data = pd.DataFrame({
            'name': 'mc',
            'x': np.arange(len(mc_errors)),
            'error': mc_errors,
        })
        rqmc_data = pd.DataFrame({
            'name': 'rqmc',
            'x': np.arange(len(rqmc_errors)),
            'error': rqmc_errors,
        })
        plot = sns.lineplot(x='x', y='error', hue='name', data=pd.concat([mc_data, rqmc_data]))
        plot.set(yscale='log')
        plt.show()
    return mc_errors, rqmc_errors, info

def learning(args):
    set_seed(args.seed)
    env = get_env(args)
    sampler = Sampler(env, args.n_workers) # mp
    init_policy = get_policy(args, env)
    print(init_policy)
    out_set = set()
    def train(name, loss_fn, init_policy, use_rqmc=False, n_iters=None):
        if n_iters is None: n_iters = args.n_iters
        policy = copy.deepcopy(init_policy)
        optim = torch.optim.SGD(policy.parameters(), args.lr)
        all_returns = []
        prog = trange(n_iters, desc=name)
        for _ in prog:
            if name in out_set or (name == 'full' and len(out_set) == 2): # fast skip
                all_returns.append(np.nan)
                continue
            returns = []
            loss = [] # policy gradient loss
            if use_rqmc:
                loc = torch.zeros(env.max_steps * env.M)
                scale = torch.ones(env.max_steps * env.M) 
                noises = Normal_RQMC(loc, scale).sample(torch.Size([args.n_trajs])).data.numpy().reshape(args.n_trajs, env.max_steps, env.M)
            else:
                noises = np.random.randn(args.n_trajs, env.max_steps, env.M)
            data = sampler.sample(policy, noises) # mp
            for states, actions, rewards in data: 
                loss.append(loss_fn(states, actions, rewards, policy))
                returns.append(rewards.sum())
                if len(states) != args.H: 
                    out_set.add(name)
            optim.zero_grad()
            loss = -torch.mean(torch.stack(loss))
            loss.backward()
            optim.step()
            all_returns.append(np.mean(returns))
            prog.set_postfix(ret=all_returns[-1])
        return np.asarray(all_returns)
    results = dict(
        mc=train('mc', variance_reduced_loss, init_policy),
        rqmc=train('rqmc', variance_reduced_loss, init_policy, use_rqmc=True),
        #full=train('full', init_K, full_grad), # this is only available for linear policy
        optimal=train('optimal', no_loss, get_policy(argparse.Namespace(init_policy='optimal'), env), n_iters=1).repeat(args.n_iters),
    )
    if args.show_fig or args.save_fig is not None:
        valid_results = {k: v for k, v in results.items() if k not in out_set}
        costs = pd.concat([pd.DataFrame({'name': name, 'x': np.arange(len(rs)), 'cost': -rs}) for name, rs in valid_results.items()])  
        plot = sns.lineplot(x='x', y='cost', hue='name', data=costs)
        plt.yscale('log')
        if args.save_fig:
            plt.savefig(args.save_fig)
        if args.show_fig:
            plt.show()
    info = {**vars(args), 'out': out_set}
    return results, info

def comparing_over_seeds(save_fn, sample_f, sample_args, num_seeds=200):
    results = []
    sample_args.save_fn = None # overwrite
    for seed in range(num_seeds):
        print('running seed {}/{}'.format(seed, num_seeds))
        sample_args.seed = seed
        result = sample_f(sample_args)
        results.append(result)
    with open(save_fn, 'wb') as f:
        dill.dump(results, f)

# run until a number of success seed is collected
def collect_seeds(save_fn, sample_f, sample_args, success_f, n_seeds=50, max_seed=200):
    results = []
    sample_args.save_fn = None # overwrite, do not save
    n_success = 0
    for seed in range(max_seed):
        print('running seed {}/{}, collecting seed {}/{}'.format(seed, max_seed, n_success, n_seeds))
        sample_args.seed = seed
        result = sample_f(sample_args)
        if success_f(result):
            print('success seed, appended')
            n_success += 1
        else:
            print('fail seed, discarded')
        results.append(result)
        if n_success == n_seeds: break
    save_fn = Path(save_fn) 
    save_fn.parent.mkdir(parents=True, exist_ok=True) 
    with open(save_fn, 'wb') as f:
        dill.dump(results, f)

def main(args=None):
    #select_device(0 if torch.cuda.is_available() else -1)
    select_device(-1)
    args = parse_args(args)
    if args.task == 'learn':
        exp_f = learning
    elif args.task == 'cost':
        exp_f = compare_cost
    elif args.task == 'grad':
        exp_f = compare_grad
    else:
        raise Exception('unsupported task')
    if args.mode == 'single':
        exp_f(args)
    elif args.mode == 'over':
        comparing_over_seeds(args.save_fn, exp_f, argparse.Namespace(**vars(args)), args.n_seeds) # why namespace on args???
    elif args.mode == 'collect':
        assert args.task == 'learn'
        success_f = lambda result: len(result[1]['out']) == 0
        collect_seeds(args.save_fn, exp_f, args, success_f=success_f, n_seeds=args.n_seeds, max_seed=args.max_seed)

if __name__ == "__main__":
    with slaunch_ipdb_on_exception():
        main()
