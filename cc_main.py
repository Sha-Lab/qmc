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


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task', 
        choices=['learn'], 
        default='learn')
    parser.add_argument('--env', choices=['cartpole'], default='cartpole')
    parser.add_argument('-H', type=int, default=10, help='horizon')
    parser.add_argument('--n_trajs', type=int, default=800, help='number of trajectories used')
    parser.add_argument('--n_iters', type=int, default=600, help='number of iterations of training')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_fig', action='store_true')
    parser.add_argument('--save_fig', type=str, default=None)
    parser.add_argument('--save_fn', type=str, default=None)
    return parser.parse_args(args)

def get_env(args):
    if args.env == 'cartpole':
        env = CartPoleContinuousEnv()
    else:
        raise Exception('unsupported environment')
    return env

def learning(args):
    set_seed(args.seed)
    env = get_env(args)
    sampler = Sampler(env, 4)
    def train(name, init_K, grad_fn, use_rqmc=False, n_iters=None):
        if n_iters is None: n_iters = args.n_iters
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
            K += args.lr * grad # constant learning rate
            all_returns.append(np.mean(returns))
            prog.set_postfix(ret=all_returns[-1], grad_norm=grad_norm, grad_err=grad_error)
        return np.asarray(all_returns), np.asarray(grad_errors), np.asarray(grad_norms)

def main(args=None):
    args = parse_args(args)
    if args.task == 'learn':
        exp_f = learning
    else:
        raise Exception('unsupported task')
    exp_f(args)

if __name__ == "__main__":
    with slaunch_ipdb_on_exception():
        main()
