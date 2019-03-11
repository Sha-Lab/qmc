import torch
import dill
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipdb import slaunch_ipdb_on_exception

from lqr import LQR
from utils import set_seed
from torch.distributions import Uniform, Normal
from rqmc_distributions import Uniform_RQMC, Normal_RQMC

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.0)
    return parser.parse_args()

def rollout(env, K, noises):
    cost = 0.0
    total_steps = 0
    done = False
    s = env.reset()
    while not done:
        a = K.dot(s) + noises[total_steps]
        s, r, done, _ = env.step(a)
        #if not np.all(s == s) or not np.all(r == r):
            #import ipdb; ipdb.set_trace()
        cost -= r
        total_steps += 1
    return cost, total_steps

# error bar: https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars
def compare_samples(horizon=100, num_trajs=1000, noise_scale=0.0, seed=0, save_dir=None, show_fig=False):
    set_seed(seed)
    env = LQR(
        lims=10,
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
    mc_stds = []
    for i in range(num_trajs):
        noises = np.random.randn(env.max_steps, env.N)
        cost, total_steps = rollout(env, K, noises)
        mc_costs.append(cost)
        mc_means.append(np.mean(mc_costs))
        mc_stds.append(np.std(mc_costs))

    rqmc_costs = []
    rqmc_means = []
    rqmc_stds = []
    loc = torch.zeros(env.max_steps * env.N)
    scale = torch.ones(env.max_steps * env.N)
    rqmc_noises = Normal_RQMC(loc, scale).sample(torch.Size([num_trajs])).data.numpy()
    for i in range(num_trajs):
        cost, total_steps = rollout(env, K, rqmc_noises[i].reshape(env.max_steps, env.N))
        rqmc_costs.append(cost)
        rqmc_means.append(np.mean(rqmc_costs))
        rqmc_stds.append(np.std(rqmc_costs))

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

### procedures ###
def comparing_over_seeds(save_fn, sample_config, num_seeds=200):
    results = []
    for seed in range(num_seeds):
        print('running seed {}/{}'.format(seed, num_seeds))
        result = compare_samples(seed=seed, **sample_config)
        results.append(result)
    #print('mc has larger variance in {}/{}'.format(sum([mc[-1] > rqmc[-1] for mc, rqmc in results], 0), num_seeds))
    with open(save_fn, 'wb') as f:
        dill.dump(results, f)

if __name__ == "__main__":
    args = get_args()
    with slaunch_ipdb_on_exception():
        #for seed in range(100):
            #print('running the {}-th seed'.format(seed))
            #compare_samples(args.H, 100000, seed=seed, save=True)
        comparing_over_seeds(
            'comparing_over_noises/{}.pkl'.format(args.noise), 
            sample_config=dict(
                horizon=50,
                num_trajs=5000,
                noise_scale=args.noise,
            ),
            num_seeds=100,
        )
        #for seed in range(20):
            #print('running the {}-th seed'.format(seed))
            #compare_samples(args.H, 100000, seed=seed, save=True)


