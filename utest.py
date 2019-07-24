import unittest
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from envs import *

class TestControl(unittest.TestCase):
    def test_optimal_policy(self):
        env = InvertedPendulum(init_scale=1.0, max_steps=100)
        K = env.optimal_controller()
        state = env.reset()
        done = False
        print(state)
        while not done:
            state, r, done, _ = env.step(K.dot(state))
            print(state)
            print(np.linalg.norm(state), r)

class TestDistribution(unittest.TestCase):
    def test_dist(self):
        from rqmc_distributions.dist_rqmc import Normal_RQMC, Uniform_RQMC
        dist = Normal_RQMC(5)
        print(dist.sample(10))

class TestArray(unittest.TestCase):
    def test(self):
        from envs import Brownian
        from models import NoisePolicy
        from utils import ArrayRQMCSampler, sort_by_norm, sort_by_val, SortableWrapper, HorizonWrapper, rollout
        from ipdb import launch_ipdb_on_exception

        n_trajs = 1024
        horizon = 20
        with launch_ipdb_on_exception():
            array_costs = []
            array_means = []
            env = HorizonWrapper(Brownian(), horizon)
            policy = NoisePolicy()
            data = ArrayRQMCSampler(SortableWrapper(env), n_envs=n_trajs, sort_f=sort_by_val).sample(policy, 1)
            for traj in data:
                _, _, rewards = traj
                rewards = np.asarray(rewards)
                array_costs.append(rewards[-1])
                array_means.append(np.mean(array_costs))
            array_errors = np.abs(array_means)
            print(array_errors[-1])
            
            mc_costs = [] # individual
            mc_means = [] # cumulative
            for i in tqdm(range(n_trajs), 'mc'):
                noises = np.random.randn(horizon, 1)
                _, _, rewards = rollout(env, policy, noises)
                mc_costs.append(rewards[-1])
                mc_means.append(np.mean(mc_costs))
            mc_errors = np.abs(mc_means)
            print(mc_errors[-1])

            '''
            data = pd.DataFrame({
                'name': 'array',
                'x': np.arange(len(array_errors)),
                'error': array_errors,
            })
            plot = sns.lineplot(x='x', y='error', hue='name', data=data)
            plot.set(yscale='log')
            plt.show()
            '''

if __name__ == "__main__":
    unittest.main()
