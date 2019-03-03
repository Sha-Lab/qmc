import numpy as np
import torch
import matplotlib.pyplot as plt
from lqr import LQR
from utils import set_seed
from torch.distributions import Uniform, Normal
from rqmc_distributions import Uniform_RQMC, Normal_RQMC

H = 100
L = 5

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

def compare_samples(num_trajs=50):
    env = LQR(
        lims=10,
        max_steps=H,
        Sigma_a_kappa=1.0,
        Q_kappa=1.0,
        P_kappa=1.0,
        A_norm=1.0,
        B_norm=1.0,
    )
    K = env.optimal_controller()

    mc_costs = []
    mc_means = []
    for i in range(num_trajs):
        noises = np.random.randn(env.max_steps, env.N)
        cost, total_steps = rollout(env, K, noises)
        mc_costs.append(cost)
        mc_means.append(np.mean(mc_costs))

    rqmc_costs = []
    rqmc_means = []
    loc = torch.zeros(env.max_steps * env.N)
    scale = torch.ones(env.max_steps * env.N)
    rqmc_noises = Normal_RQMC(loc, scale).sample(torch.Size([num_trajs])).data.numpy()
    for i in range(num_trajs):
        cost, total_steps = rollout(env, K, rqmc_noises[i].reshape(env.max_steps, env.N))
        rqmc_costs.append(cost)
        rqmc_means.append(np.mean(rqmc_costs))

    expected_cost = env.expected_cost(K)

    plt.figure(figsize=(1 * L, 2 * L))
    plt.semilogy((mc_means - expected_cost)**2, label='error mean MC')
    plt.semilogy((rqmc_means - expected_cost)**2, label='error mean RQMC')
    plt.legend()
    plt.title("Error of empirical mean of normal distribution")

    plt.savefig("./comparing_samples.png")

    #print('mean cost:', np.mean(costs))
    #print('MC ground truth:', env.expected_cost(K))


if __name__ == "__main__":
    set_seed(0)
    compare_samples(5000)
