import sys
import gym
import torch
import dill
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from ipdb import slaunch_ipdb_on_exception

from envs import *
from utils import set_seed, rollout, mse, cummean, Sampler
from torch.distributions import Uniform, Normal
from rqmc_distributions import Uniform_RQMC, Normal_RQMC

# TODO: make sure compare_cost produce the same result

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task', 
        choices=['cost', 'grad', 'learn'], 
        default='learn')
    parser.add_argument('--env', choices=['custom', 'WIP', 'IP'], default='custom')
    parser.add_argument('--xu_dim', type=int, nargs=2, default=(20, 12))
    parser.add_argument('--init_scale', type=float, default=3.0)
    parser.add_argument('--PQ_kappa', type=float, default=3.0)
    parser.add_argument('--AB_norm', type=float, default=1.0)
    parser.add_argument('-H', type=int, default=10, help='horizon')
    parser.add_argument('--noise', type=float, default=0.0, help='noise scale')
    parser.add_argument('--n_trajs', type=int, default=800, help='number of trajectories used')
    parser.add_argument('--n_iters', type=int, default=200, help='number of iterations of training')
    parser.add_argument('-lr', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_fig', action='store_true')
    parser.add_argument('--save_fig', type=str, default=None)
    parser.add_argument('--mode', choices=['single', 'over', 'collect'], default='single')
    parser.add_argument('--n_seeds', type=int, default=200)
    parser.add_argument('--max_seed', type=int, default=200)
    parser.add_argument('--save_fn', type=str, default=None)
    return parser.parse_args(args)

def get_env(args):
    if args.env == 'custom':
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
    else:
        raise Exception('unsupported lqr env')
    return env

# error bar: https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars
#def compare_cost(horizon=100, num_trajs=1000, noise_scale=0.0, seed=0, save_dir=None, show_fig=False):
def compare_cost(args):
    set_seed(args.seed)
    env = LQR(
        init_scale=1.0,
        max_steps=100,
        Sigma_s_kappa=1.0,
        Q_kappa=1.0,
        P_kappa=1.0,
        A_norm=1.0,
        B_norm=1.0,
        Sigma_s_scale=0.0,
    )
    K = env.optimal_controller()

    mc_costs = [] # individual
    mc_means = [] # cumulative
    for i in tqdm(range(args.n_trajs), 'mc'):
        noises = np.random.randn(env.max_steps, env.M)
        _, _, rewards = rollout(env, K, noises)
        mc_costs.append(-rewards.sum())
        mc_means.append(np.mean(mc_costs))

    rqmc_costs = []
    rqmc_means = []
    loc = torch.zeros(env.max_steps * env.M)
    scale = torch.ones(env.max_steps * env.M)
    rqmc_noises = Normal_RQMC(loc, scale).sample(torch.Size([args.n_trajs])).data.numpy()
    for i in tqdm(range(args.n_trajs), 'rqmc'):
        _, _, rewards = rollout(env, K, rqmc_noises[i].reshape(env.max_steps, env.M))
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
    K = env.optimal_controller()
    Sigma_a = np.diag(np.ones(env.M))
    Sigma_a_inv = np.linalg.inv(Sigma_a)
    print(env.Sigma_s)
    mc_grads = []
    for i in tqdm(range(args.n_trajs), 'mc'):
        noises = np.random.randn(env.max_steps, env.M)
        states, actions, rewards = rollout(env, K, noises)
        mc_grads.append(Sigma_a_inv @ (actions - states @ K.T).T @ states * rewards.sum()) # need minus since I use cost formula in derivation
    mc_grads = np.asarray(mc_grads)
    mc_means = np.cumsum(mc_grads, axis=0) / np.arange(1, len(mc_grads) + 1)[:, np.newaxis, np.newaxis]
    
    rqmc_grads = []
    loc = torch.zeros(env.max_steps * env.M)
    scale = torch.ones(env.max_steps * env.M)
    rqmc_noises = Normal_RQMC(loc, scale).sample(torch.Size([args.n_trajs])).data.numpy()
    for i in tqdm(range(args.n_trajs), 'rqmc'):
        states, actions, rewards = rollout(env, K, rqmc_noises[i].reshape(env.max_steps, env.M))
        rqmc_grads.append(Sigma_a_inv @ (actions - states @ K.T).T @ states * rewards.sum())
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
    sampler = Sampler(env, 1) # mp
    Sigma_a = np.diag(np.ones(env.M))
    Sigma_a_inv = np.linalg.inv(Sigma_a)
    init_K = np.random.randn(env.M, env.N)
    out_set = set()
    def reinforce_grad(states, actions, rewards, K):
        return Sigma_a_inv @ (actions - states @ K.T).T @ states * rewards.sum()
    def variance_reduced_grad(states, actions, rewards, K):
        Rs = rewards[::-1].cumsum()[::-1]
        return Sigma_a_inv @ (actions - states @ K.T).T @ (states * Rs[:,None])
    def full_grad(states, actions, rewards, K):
        return env.expected_policy_gradient(K, Sigma_a)
    def no_grad(states, actions, rewards, K):
        return np.zeros_like(K)
    def train(name, init_K, grad_fn, use_rqmc=False, n_iters=None):
        if n_iters is None: n_iters = args.n_iters
        K = np.copy(init_K)
        all_returns = []
        grad_errors = []
        grad_norms = []
        prog = trange(n_iters, desc=name)
        for i in prog:
            if name in out_set or (name == 'full' and len(out_set) == 2): # fast skip
                all_returns.append(np.nan)
                grad_errors.append(np.nan)
                grad_norms.append(np.nan)
                continue
            grad = []
            returns = []
            if use_rqmc:
                loc = torch.zeros(env.max_steps * env.M)
                scale = torch.ones(env.max_steps * env.M) 
                noises = Normal_RQMC(loc, scale).sample(torch.Size([args.n_trajs])).data.numpy().reshape(args.n_trajs, env.max_steps, env.M)
            else:
                noises = np.random.randn(args.n_trajs, env.max_steps, env.M)
            data = sampler.sample(K, noises) # mp
            for states, actions, rewards in data: 
                grad.append(grad_fn(states, actions, rewards, K))
                returns.append(rewards.sum())
                if len(states) != args.H: 
                    out_set.add(name)
            grad = np.mean(grad, axis=0)
            grad_norm = np.linalg.norm(grad) #mse(grad, env.expected_policy_gradient(K, Sigma_a))
            grad_error = mse(grad, env.expected_policy_gradient(K, Sigma_a))
            grad_norms.append(grad_norm)
            grad_errors.append(grad_error)
            #K += lr / np.maximum(1.0, np.linalg.norm(grad)) * grad # constant norm of gradient
            #K += args.lr / (i+1) * grad # decreasing learning rate!
            K += args.lr * grad # constant learning rate
            all_returns.append(np.mean(returns))
            prog.set_postfix(ret=all_returns[-1], grad_norm=grad_norm, grad_err=grad_error)
        return np.asarray(all_returns), np.asarray(grad_errors), np.asarray(grad_norms)
    results = dict(
        mc=train('mc', init_K, variance_reduced_grad),
        rqmc=train('rqmc', init_K, variance_reduced_grad, use_rqmc=True),
        #full=train('full', init_K, full_grad),        
        optimal=tuple(map(lambda x: x.repeat(args.n_iters), train('optimal', env.optimal_controller(), no_grad, n_iters=1))),
    )
    if args.show_fig or args.save_fig is not None:
        valid_results = {k: v for k, v in results.items() if k not in out_set}
        fig, axs = plt.subplots(ncols=3)
        costs = pd.concat([pd.DataFrame({'name': name, 'x': np.arange(len(rs[0])), 'cost': -rs[0]}) for name, rs in valid_results.items()])  
        grad_errors = pd.concat([pd.DataFrame({'name': name, 'x': np.arange(len(rs[1])), 'grad_error': rs[1]}) for name, rs in valid_results.items()]) 
        grad_norms = pd.concat([pd.DataFrame({'name': name, 'x': np.arange(len(rs[2])), 'grad_norm': rs[2]}) for name, rs in valid_results.items()]) 
        cost_plot = sns.lineplot(x='x', y='cost', hue='name', data=costs, ax=axs[0])
        grad_error_plot = sns.lineplot(x='x', y='grad_error', hue='name', data=grad_errors, ax=axs[1]) 
        grad_norm_plot = sns.lineplot(x='x', y='grad_norm', hue='name', data=grad_norms, ax=axs[2])
        plt.yscale('log')
        if args.save_fig:
            fig.savefig(args.save_fig)
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
    with open(save_fn, 'wb') as f:
        dill.dump(results, f)

def main(args=None):
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