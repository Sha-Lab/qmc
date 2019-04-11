import sys
import torch
import dill
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ipdb import slaunch_ipdb_on_exception

from lqr import LQR
from utils import set_seed, cmd_run, cmd
from torch.distributions import Uniform, Normal
from rqmc_distributions import Uniform_RQMC, Normal_RQMC

# TODO: make sure compare_cost produce the same result
# TODO: (done) write compare_cov to verify the calculation
# TODO: write compare_grad to verify calculation

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.0)
    return parser.parse_args()

def rollout(env, K, noises):
    states = []
    actions = []
    rewards = []
    done = False
    s = env.reset()
    cur_step = 0
    while not done:
        a = K.dot(s) + noises[cur_step]
        next_s, r, done, _ = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = next_s
        cur_step += 1
    return np.asarray(states), np.asarray(actions), np.asarray(rewards)

# error bar: https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars
def compare_cost(horizon=100, num_trajs=1000, noise_scale=0.0, seed=0, save_dir=None, show_fig=False):
    set_seed(seed)
    env = LQR(
        lims=100,
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
    if save_dir is not None:
        with open(Path(save_dir, save_fn), 'wb') as f:
            info = dict(horizon=horizon, num_trajs=num_trajs, seed=seed)
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

### procedures ###
def comparing_over_seeds(save_fn, sample_config, num_seeds=200):
    results = []
    for seed in range(num_seeds):
        print('running seed {}/{}'.format(seed, num_seeds))
        result = compare_cost(seed=seed, **sample_config)
        results.append(result)
    #print('mc has larger variance in {}/{}'.format(sum([mc[-1] > rqmc[-1] for mc, rqmc in results], 0), num_seeds))
    with open(save_fn, 'wb') as f:
        dill.dump(results, f)

if __name__ == "__main__":
    args = get_args()
    with slaunch_ipdb_on_exception():
        #compare_cov(100, 5000, show_fig=True)
        compare_grad(20, 500000, show_fig=True)
        #for seed in range(100):
            #print('running the {}-th seed'.format(seed))
            #compare_cost(args.H, 100000, seed=seed, save=True)
        #comparing_over_seeds(
            #'comparing_over_noises/{}.pkl'.format(args.noise), 
            #sample_config=dict(
                #horizon=50,
                #num_trajs=5000,
                #noise_scale=args.noise,
            #),
            #num_seeds=100,
        #)
        #for seed in range(20):
            #print('running the {}-th seed'.format(seed))
            #compare_cost(args.H, 100000, seed=seed, save=True)


