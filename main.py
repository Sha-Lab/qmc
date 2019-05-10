import sys
import torch
import dill
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm, trange
from ipdb import slaunch_ipdb_on_exception

from lqr import LQR
from utils import set_seed, rollout, mse
from torch.distributions import Uniform, Normal
from rqmc_distributions import Uniform_RQMC, Normal_RQMC

from data.rllqr.rllqr import LQRProblem, random_K_eigvalmax

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

def learning(args):
    set_seed(args.seed)
    env = LQR(
        N=20, # compared with rllqr
        M=12,
        init_scale=3.0,
        max_steps=args.H,
        Sigma_s_kappa=1.0,
        Q_kappa=1.0,
        P_kappa=1.0,
        A_norm=1.0,
        B_norm=1.0,
        Sigma_s_scale=args.noise,
        #random_init=True,
    )

    #npr = np.random.RandomState()

    #xdim = 6
    #udim = 4
    #prob = LQRProblem.random(xdim, udim, npr)
    #A, B, Q, R = prob.ABQR()

    #eigmax = 0.9
    #K = random_K_eigvalmax(npr, A, B, eigmax)

    #init_K = K # P action, Q state
    #env.A, env.B, env.P, env.Q = A, B, R, Q

    Sigma_a = np.diag(np.ones(env.M))
    Sigma_a_inv = np.linalg.inv(Sigma_a)
    init_K = np.random.randn(env.M, env.N) ### swap, here it has problem!!! fixed by a smaller learning rate
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
    def train(name, init_K, grad_fn, use_rqmc=False):
        K = np.copy(init_K)
        all_returns = []
        prog = trange(args.n_iters, desc=name)
        for i in prog:
            grad = []
            returns = []
            if use_rqmc:
                loc = torch.zeros(env.max_steps * env.M)
                scale = torch.ones(env.max_steps * env.M) 
                noises = Normal_RQMC(loc, scale).sample(torch.Size([args.n_trajs])).data.numpy().reshape(args.n_trajs, env.max_steps, env.M)
            else:
                noises = np.random.randn(args.n_trajs, env.max_steps, env.M)
            for j in range(args.n_trajs):
                states, actions, rewards = rollout(env, K, noises[j])
                grad.append(grad_fn(states, actions, rewards, K))
                returns.append(rewards.sum())
                if len(states) != args.H: 
                    out_set.add(name)
            grad = np.mean(grad, axis=0)
            grad_norm = np.linalg.norm(grad) #mse(grad, env.expected_policy_gradient(K, Sigma_a))
            #K += lr / np.maximum(1.0, np.linalg.norm(grad)) * grad
            K += args.lr / (i+1) * grad # decreasing learning rate!
            all_returns.append(np.mean(returns))
            prog.set_postfix(ret=all_returns[-1], grad_norm=grad_norm)
        return np.asarray(all_returns)

    #pool = multiprocessing.Pool(processes=6)
    #names = ['rqmc', 'mc']
    #train_args = [(name, init_K, reinforce_grad, name=='rqmc') for name in names]
    #results = pool.starmap(train, train_args)
    #returns = {k: v for k, v in zip(names, results)}

    returns = dict(
        rqmc=train('rqmc', init_K, variance_reduced_grad, use_rqmc=True),
        mc=train('mc', init_K, variance_reduced_grad),
        #vrmc=train('vrmc', init_K, variance_reduced_grad),
        full=train('full', init_K, full_grad),        
        optimal=train('optimal', env.optimal_controller(), no_grad), # should only be calculated once
    )
    if args.fig:
        #data = pd.concat([pd.DataFrame({'name': name, 'x': np.arange(len(rs)), 'return': rs}) for name, rs in returns.items()])
        #plot = sns.lineplot(x='x', y='return', hue='name', data=data)
        data = pd.concat([pd.DataFrame({'name': name, 'x': np.arange(len(rs)), 'cost': -rs}) for name, rs in returns.items()])
        plot = sns.lineplot(x='x', y='cost', hue='name', data=data)
        plt.yscale('log')
        plt.show()
    info = {**vars(args), 'out': out_set}
    return returns, info

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
