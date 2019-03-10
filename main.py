import numpy as np
import torch
import dill
import matplotlib.pyplot as plt
from lqr import LQR
from utils import set_seed
from torch.distributions import Uniform, Normal
from rqmc_distributions import Uniform_RQMC, Normal_RQMC

H = 100
L = 5
n_trajs = 2500

def rollout(env, K, noises):
    cost = 0.0
    total_steps = 0
    done = False
    s = env.reset()
    while not done:
        s, r, done, _ = env.step(K.dot(s) + noises[total_steps])
        cost -= r
        total_steps += 1
    return cost, total_steps

# gather the result of function f over a large number of seeds
# return a list of all results over all seeds
def over_seeds(f, num_seeds=100):
    results = []
    for i in range(num_seeds):
        print('[over_seeds] running the {} seed'.format(i))
        set_seed(i)
        results.append(f())
    return results

# error bar: https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars
def compare_samples(num_trajs=1000, savefig=False):
    env = LQR(
        lims=10,
        max_steps=H,
        Sigma_s_kappa=1.0,
        Q_kappa=1.0,
        P_kappa=1.0,
        A_norm=1.0,
        B_norm=1.0,
        Sigma_s_scale=0.0,
    )
    K = env.optimal_controller()

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
    if savefig:
        plt.figure(figsize=(1 * L, 1 * L))
        plt.semilogy((mc_means - expected_cost)**2, label='MC')
        plt.semilogy((rqmc_means - expected_cost)**2, label='RQMC')
        plt.legend()
        plt.title("Var(J) of K_opt, H={}, num_trajs={}".format(H, num_trajs))
        plt.savefig("./comparing_samples.png")
    return mc_errors, rqmc_errors

### procedures ###
def comparing_over_seeds(num_seeds=200):
    results = over_seeds(lambda: compare_samples(2500), num_seeds)
    print('mc has larger variance in {}/{}'.format(sum([mc[-1] > rqmc[-1] for mc, rqmc in results], 0), num_seeds))
    with open('comparing_over_seeds.pkl', 'wb') as f:
        dill.dump(results, f)

if __name__ == "__main__":
    #set_seed(0)
    compare_samples(20000, savefig=False)
    #comparing_over_seeds()
