import sys
import torch
import dill
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from ipdb import slaunch_ipdb_on_exception

from lqr import LQR
from utils import set_seed, rollout, mse
from torch.distributions import Uniform, Normal
from rqmc_distributions import Uniform_RQMC, Normal_RQMC

# TODO: make sure compare_cost produce the same result

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task', 
        choices=['cost', 'cov', 'grad', 'learn'], 
        default='learn')
    parser.add_argument('-H', type=int, default=10, help='horizon')
    parser.add_argument('--noise', type=float, default=0.0, help='noise scale')
    parser.add_argument('--n_trajs', type=int, default=800, help='number of trajectories used')
    parser.add_argument('--n_iters', type=int, default=600, help='number of iterations of training')
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fig', action='store_true') # whether to show the figure
    # for over seed
    parser.add_argument('--over_seed', action='store_true')
    parser.add_argument('--n_seeds', type=int, default=200) # for over
    parser.add_argument('--save_fn', type=str, default=None)
    return parser.parse_args(args)

# error bar: https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars
def compare_cost(horizon=100, num_trajs=1000, noise_scale=0.0, seed=0, save_dir=None, show_fig=False):
    set_seed(seed)
    env = LQR(
        lims=100,
        init_scale=1.0,
        max_steps=horizon,
        Sigma_s_kappa=1.0,
        Q_kappa=1.0,
        P_kappa=1.0,
        A_norm=1.0,
        B_norm=1.0,
        Sigma_s_scale=noise_scale,
    )
    K = env.optimal_controller()
    print(env.Sigma_s)

    mc_costs = []
    mc_means = []
    for i in range(num_trajs):
        noises = np.random.randn(env.max_steps, env.M)
        #cost, total_steps = rollout(env, K, noises)
        _, _, rewards = rollout(env, K, noises)
        mc_costs.append(-rewards.sum())
        mc_means.append(np.mean(mc_costs))

    rqmc_costs = []
    rqmc_means = []
    loc = torch.zeros(env.max_steps * env.M)
    scale = torch.ones(env.max_steps * env.M)
    rqmc_noises = Normal_RQMC(loc, scale).sample(torch.Size([num_trajs])).data.numpy()
    for i in range(num_trajs):
        #cost, total_steps = rollout(env, K, rqmc_noises[i].reshape(env.max_steps, env.N))
        _, _, rewards = rollout(env, K, rqmc_noises[i].reshape(env.max_steps, env.M))
        rqmc_costs.append(-rewards.sum())
        rqmc_means.append(np.mean(rqmc_costs))

    expected_cost = env.expected_cost(K, np.diag(np.ones(env.M)))

    mc_errors = (mc_means - expected_cost) ** 2
    rqmc_errors = (rqmc_means - expected_cost) ** 2
    save_fn = 'H-{}.num_traj-{}.ns-{}.{}.pkl'.format(horizon, num_trajs, noise_scale, seed)
    info = dict(mc_costs=mc_costs, rqmc_costs=rqmc_costs, save_fn=save_fn)
    if save_dir is not None:
        with open(Path(save_dir, save_fn), 'wb') as f:
            dill.dump(dict(mc_errors=mc_errors, rqmc_errors=rqmc_errors, info=info), f)
    if show_fig:
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
        plot = sns.relplot(x='x', y='error', hue='name', kind='line', data=pd.concat([mc_data, rqmc_data]))
        plot.set(yscale='log')
    return mc_errors, rqmc_errors, info

def compare_cov(horizon, num_trajs, noise_scale=0.0, seed=0, save_dir=None, show_fig=False):
    set_seed(seed)
    env = LQR(
        lims=100,
        init_scale=1.0,
        max_steps=horizon,
        Sigma_s_kappa=1.0,
        Q_kappa=1.0,
        P_kappa=1.0,
        A_norm=1.0,
        B_norm=1.0,
        Sigma_s_scale=noise_scale,
    )
    K = env.optimal_controller()
    Sigma_a = np.diag(np.ones(env.M))
    print(env.Sigma_s)
    mc_states = []
    for i in tqdm(range(num_trajs), 'mc'):
        noises = np.random.randn(env.max_steps, env.M)
        states, _, _ = rollout(env, K, noises)
        mc_states.append(states)
    mc_states = np.asarray(mc_states)
    mc_sc = np.matmul(np.expand_dims(mc_states, 3), np.expand_dims(mc_states, 2))
    mc_means = np.cumsum(mc_sc, axis=0) / np.arange(1, len(mc_sc) + 1)[:, np.newaxis, np.newaxis, np.newaxis]
    
    rqmc_states = []
    loc = torch.zeros(env.max_steps * env.M)
    scale = torch.ones(env.max_steps * env.M)
    rqmc_noises = Normal_RQMC(loc, scale).sample(torch.Size([num_trajs])).data.numpy()
    for i in tqdm(range(num_trajs), 'rqmc'):
        states, _, _ = rollout(env, K, rqmc_noises[i].reshape(env.max_steps, env.M))
        rqmc_states.append(states)
    rqmc_sc = np.matmul(np.expand_dims(rqmc_states, 3), np.expand_dims(rqmc_states, 2))
    rqmc_means = np.cumsum(rqmc_sc, axis=0) / np.arange(1, len(rqmc_sc) + 1)[:, np.newaxis, np.newaxis, np.newaxis]

    expected_sc = np.asarray([env.expected_state_cov(t, K, Sigma_a) for t in range(env.max_steps)])

    mc_errors = ((mc_means - expected_sc) ** 2).reshape(*mc_sc.shape[:2], -1).mean(2)
    rqmc_errors = ((rqmc_means - expected_sc) ** 2).reshape(*rqmc_sc.shape[:2], -1).mean(2)
    if save_dir is not None:
        with open(Path(save_dir, save_fn), 'wb') as f:
            info = dict(horizon=horizon, num_trajs=num_trajs, seed=seed)
            dill.dump(dict(mc_errors=mc_errors, rqmc_errors=rqmc_errors, info=info), f)
    if show_fig:
        mc_data = pd.DataFrame({
            'name': 'mc',
            'x': np.arange(len(mc_errors)),
            'error': mc_errors[:, -1],
        })
        rqmc_data = pd.DataFrame({
            'name': 'rqmc',
            'x': np.arange(len(rqmc_errors)),
            'error': rqmc_errors[:, -1],
        })
        plot = sns.lineplot(x='x', y='error', hue='name', data=pd.concat([mc_data, rqmc_data]))
        plot.set(yscale='log')
        plt.show()

def compare_grad(horizon, num_trajs, noise_scale=0.0, seed=0, save_dir=None, show_fig=False):
    set_seed(seed)
    env = LQR(
        lims=100,
        init_scale=1.0,
        max_steps=horizon,
        Sigma_s_kappa=1.0,
        Q_kappa=1.0,
        P_kappa=1.0,
        A_norm=1.0,
        B_norm=1.0,
        Sigma_s_scale=noise_scale,
    )
    K = env.optimal_controller()
    Sigma_a = np.diag(np.ones(env.M))
    Sigma_a_inv = np.linalg.inv(Sigma_a)
    print(env.Sigma_s)
    mc_grads = []
    #grad = [] # debug
    for i in tqdm(range(num_trajs), 'mc'):
        noises = np.random.randn(env.max_steps, env.M)
        states, actions, rewards = rollout(env, K, noises)
        mc_grads.append(Sigma_a_inv @ (actions - states @ K.T).T @ states * rewards.sum()) # need minus since I use cost formula in derivation
        #grad.append(Sigma_a_inv @ np.outer(actions[0] - K @ states[0], states[0]) * rewards[0]) # debug
    mc_grads = np.asarray(mc_grads)
    mc_means = np.cumsum(mc_grads, axis=0) / np.arange(1, len(mc_grads) + 1)[:, np.newaxis, np.newaxis]
    
    rqmc_grads = []
    loc = torch.zeros(env.max_steps * env.M)
    scale = torch.ones(env.max_steps * env.M)
    rqmc_noises = Normal_RQMC(loc, scale).sample(torch.Size([num_trajs])).data.numpy()
    for i in tqdm(range(num_trajs), 'rqmc'):
        states, actions, rewards = rollout(env, K, rqmc_noises[i].reshape(env.max_steps, env.M))
        rqmc_grads.append(Sigma_a_inv @ (actions - states @ K.T).T @ states * rewards.sum())
    rqmc_grads = np.asarray(rqmc_grads)
    rqmc_means = np.cumsum(rqmc_grads, axis=0) / np.arange(1, len(rqmc_grads) + 1)[:, np.newaxis, np.newaxis]

    expected_grad = env.expected_policy_gradient(K, Sigma_a)
    #import ipdb; ipdb.set_trace()
    #assert np.all(expected_grad == 2 * env.P @ K @ np.outer(env.init_state, env.init_state))

    mc_errors = ((mc_means - expected_grad) ** 2).reshape(mc_means.shape[0], -1).mean(1) # why the sign is reversed?
    rqmc_errors = ((rqmc_means - expected_grad) ** 2).reshape(rqmc_means.shape[0], -1).mean(1)
    info = dict(horizon=horizon, num_trajs=num_trajs, seed=seed)
    if save_dir is not None:
        with open(Path(save_dir, save_fn), 'wb') as f:
            dill.dump(dict(mc_errors=mc_errors, rqmc_errors=rqmc_errors, info=info), f)
    if show_fig:
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

#def learning(n_iters, n_trajs, lr=0.001, horizon=10, noise_scale=0.0, seed=0, show_fig=False):
def learning(args):
    set_seed(args.seed)
    env = LQR(
        init_scale=1.0,
        max_steps=args.H,
        Sigma_s_kappa=1.0,
        Q_kappa=1.0,
        P_kappa=1.0,
        A_norm=1.0,
        B_norm=1.0,
        Sigma_s_scale=args.noise,
    )
    Sigma_a = np.diag(np.ones(env.M))
    Sigma_a_inv = np.linalg.inv(Sigma_a)
    init_K = np.random.randn(env.N, env.M)
    out_set = set()
    # mc
    K = np.copy(init_K)
    mc_returns = []
    prog = trange(args.n_iters, desc='mc')
    for i in prog:
        mc_grad = []
        returns = []
        for _ in range(args.n_trajs):
            noises = np.random.randn(env.max_steps, env.M)
            states, actions, rewards = rollout(env, K, noises)
            mc_grad.append(Sigma_a_inv @ (actions - states @ K.T).T @ states * rewards.sum()) # need minus since I use cost formula in derivation
            returns.append(rewards.sum())
            if len(states) != args.H: 
                out_set.add('mc')
                #print('the length of trajecoty is wrong!')
                #return None
        mc_grad = np.mean(mc_grad, axis=0)
        grad_error = mse(mc_grad, env.expected_policy_gradient(K, Sigma_a))
        #K += lr / np.maximum(1.0, np.linalg.norm(mc_grad)) * mc_grad
        K += args.lr * mc_grad
        mc_returns.append(np.mean(returns))
        prog.set_postfix(ret=mc_returns[-1])
    # rqmc
    K = np.copy(init_K)
    rqmc_returns = []
    loc = torch.zeros(env.max_steps * env.M)
    scale = torch.ones(env.max_steps * env.M) 
    prog = trange(args.n_iters, desc='rqmc')
    for i in prog:
        rqmc_grad = []
        returns = []
        rqmc_noises = Normal_RQMC(loc, scale).sample(torch.Size([args.n_trajs])).data.numpy()
        for j in range(args.n_trajs):
            states, actions, rewards = rollout(env, K, rqmc_noises[j].reshape(env.max_steps, env.M))
            rqmc_grad.append(Sigma_a_inv @ (actions - states @ K.T).T @ states * rewards.sum()) # need minus since I use cost formula in derivation
            returns.append(rewards.sum())
            if len(states) != args.H: 
                out_set.add('rqmc')
                #print('the length of trajecoty is wrong!')
                #return None
        rqmc_grad = np.mean(rqmc_grad, axis=0)
        grad_error = mse(rqmc_grad, env.expected_policy_gradient(K, Sigma_a))
        #K += lr / np.maximum(1.0, np.linalg.norm(rqmc_grad)) * rqmc_grad
        K += args.lr * rqmc_grad
        rqmc_returns.append(np.mean(returns))
        prog.set_postfix(ret=rqmc_returns[-1])
    # full
    K = np.copy(init_K)
    full_returns = []
    prog = trange(args.n_iters, desc='full')
    for i in prog:
        returns = []
        for j in range(args.n_trajs):
            noises = np.random.randn(env.max_steps, env.M)
            states, actions, rewards = rollout(env, K, noises)
            returns.append(rewards.sum())
            if len(states) != args.H: 
                out_set.add('full')
                #print('the length of trajecoty is wrong!')
                #return None
        full_grad = env.expected_policy_gradient(K, Sigma_a)
        #K += lr / np.maximum(1.0, np.linalg.norm(full_grad)) * full_grad
        K += args.lr * full_grad
        full_returns.append(np.mean(returns))
        prog.set_postfix(ret=full_returns[-1])
    # optimal
    K = env.optimal_controller()
    optimal_returns = []
    prog = trange(args.n_iters, desc='optimal')
    for i in prog:
        returns = []
        for j in range(args.n_trajs):
            noises = np.random.randn(env.max_steps, env.M)
            states, actions, rewards = rollout(env, K, noises)
            returns.append(rewards.sum())
            if len(states) != args.H: 
                out_set.add('optimal')
                #print('the length of trajecoty is wrong!')
                #return None
        optimal_returns.append(np.mean(returns)) 
        prog.set_postfix(ret=optimal_returns[-1])
    if args.fig:
        mc_data = pd.DataFrame({
            'name': 'mc',
            'x': np.arange(len(mc_returns)),
            'return': mc_returns,
        }) 
        rqmc_data = pd.DataFrame({
            'name': 'rqmc',
            'x': np.arange(len(rqmc_returns)),
            'return': rqmc_returns,
        }) 
        full_data = pd.DataFrame({
            'name': 'full',
            'x': np.arange(len(full_returns)),
            'return': full_returns,
        })
        optimal_data = pd.DataFrame({
            'name': 'optimal',
            'x': np.arange(len(optimal_returns)),
            'return': optimal_returns,
        })
        plot = sns.lineplot(x='x', y='return', hue='name', data=pd.concat([mc_data, rqmc_data, full_data, optimal_data]))
        plt.show()
    info = {**vars(args), 'out': out_set}
    return mc_returns, rqmc_returns, full_returns, optimal_returns, info

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

def main(args=None):
    args = parse_args(args)
    if args.task == 'learn':
        exp_f = learning
    else:
        raise Exception('unsupported task')
    if args.over_seed:
        comparing_over_seeds(args.save_fn, exp_f, argparse.Namespace(**vars(args)), args.n_seeds)
    else:
        exp_f(args)

if __name__ == "__main__":
    main()
