import gym
import copy
import json
import torch
import argparse
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from ipdb import launch_ipdb_on_exception
from pathlib import Path
from tqdm import tqdm, trange

from envs import *
from models import get_mlp, GaussianPolicy
from utils import tensor, set_seed, MPSampler, SeqRunner, VecSampler, Config, select_device, HorizonWrapper, running_seeds
from rqmc_distributions import Normal_RQMC

# network is tanh, 64x64 with RMSProp

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['cartpole', 'ant'], default='cartpole')
    parser.add_argument('--n_trajs', type=int, default=500)
    parser.add_argument('--n_iters', type=int, default=6000)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[8])
    parser.add_argument('--save_fn', type=str, default=None)
    parser.add_argument('--show_fig', action='store_true')
    parser.add_argument('--save_fig', type=str, default=None)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--mode', choices=['single', 'seeds'], default='single')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args(args)

def get_env(args):
    if args.env == 'cartpole':
        env = HorizonWrapper(CartPoleContinuousEnv(), args.horizon)
    elif args.env == 'ant':
        env = HorizonWrapper(gym.make('Ant-v2'), args.horizon)
    else:
        raise Exception('unknown env')
    return env

def reinforce_loss(states, actions, returns, policy):
    log_probs = policy.distribution(states).log_prob(tensor(actions)).sum(-1)
    return -(log_probs * tensor(returns)).mean()

def train(args, name, env, init_policy, use_rqmc=False):
    iter_returns = []
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0] 
    policy = copy.deepcopy(init_policy)
    optim = torch.optim.SGD(policy.parameters(), args.lr)
    if Config.DEVICE.type == 'cpu': 
        sampler = MPSampler(env, args.n_workers) # mp
    else:
        sampler = VecSampler(env, args.n_workers)
    prog = trange(args.n_iters, desc=name)
    for _ in prog:
        if use_rqmc:
            loc = torch.zeros(args.horizon * action_dim)
            scale = torch.ones(args.horizon * action_dim)
            noises = Normal_RQMC(loc, scale).sample(torch.Size([args.n_trajs])).data.numpy().reshape(args.n_trajs, args.horizon, action_dim)
        else:
            noises = np.random.randn(args.n_trajs, args.horizon, action_dim)
        paths = []
        data = sampler.sample(policy, noises)
        for observations, actions, rewards in data:
            returns = []
            return_so_far = 0
            for t in range(len(rewards) - 1, -1, -1):
                return_so_far = rewards[t] + args.discount * return_so_far
                returns.append(return_so_far)
            returns = returns[::-1]
            paths.append(dict(
                observations=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
                returns=np.array(returns)
            ))
        observations = np.concatenate([p["observations"] for p in paths])
        actions = np.concatenate([p["actions"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        # post processing
        returns = (returns - np.mean(returns)) / (returns.std() + 1e-8)
        optim.zero_grad()
        loss = reinforce_loss(observations, actions, returns, policy)
        loss.backward()
        optim.step()
        iter_returns.append(np.mean([sum(p["rewards"]) for p in paths]))
        prog.set_postfix(ret=iter_returns[-1])
    return iter_returns

def learning(args):
    set_seed(args.seed)
    #env = HorizonWrapper(CartPoleContinuousEnv(), args.horizon)
    env = get_env(args)
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0] 
    mean_network = get_mlp((state_dim,)+tuple(args.hidden_sizes)+(action_dim,), gate=nn.Tanh)
    init_policy = GaussianPolicy(state_dim, action_dim, mean_network)
    info = dict(
        rqmc=train(args, 'rqmc', env, init_policy, use_rqmc=True),
        mc=train(args, 'mc', env, init_policy),
    )
    if args.save_fn:
        Path(args.save_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_fn, 'w') as f:
            json.dump(info, f)
    if args.show_fig or args.save_fig is not None:
        returns = pd.concat([pd.DataFrame({'name': name, 'x': np.arange(len(rs)), 'return': rs}) for name, rs in info.items()])  
        plot = sns.lineplot(x='x', y='return', hue='name', data=returns)
        if args.save_fig:
            plt.savefig(args.save_fig)
        if args.show_fig:
            plt.show()
    return info

def main(args=None):
    args = parse_args(args)
    select_device(0 if torch.cuda.is_available() else -1)
    if args.mode == 'single':
        learning(args)
    elif args.mode == 'seeds':
        running_seeds(args.save_fn, learning, argparse.Namespace(**vars(args)), args.n_seeds) # namespace on args to get a deepcopy
    else:
        raise Exception('unknown mode')

if __name__ == "__main__":
    with launch_ipdb_on_exception():
        main()
