import torch
import argparse
import numpy as np
from torch import nn
from ipdb import launch_ipdb_on_exception
from pathlib import Path

from envs import *
from models import get_mlp, GaussianPolicy
from utils import tensor, set_seed


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trajs', type=int, default=500)
    parser.add_argument('--n_iters', type=int, default=6000)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[8])
    parser.add_argument('--save_fn', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args(args)

def reinforce_loss(states, actions, returns, policy):
    log_probs = policy.distribution(states).log_prob(tensor(actions)).sum(-1)
    return -(log_probs * tensor(returns)).mean()

def main(args=None):
    args = parse_args(args)
    iter_returns = []
    set_seed(args.seed)
    env = CartPoleContinuousEnv()
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0] 
    mean_network = get_mlp((state_dim,)+tuple(args.hidden_sizes)+(action_dim,), gate=nn.ReLU, output_gate=nn.Tanh)
    policy = GaussianPolicy(state_dim, action_dim, mean_network)
    optim = torch.optim.SGD(policy.parameters(), args.lr)
    for _ in range(args.n_iters):
        noises = np.random.randn(args.n_trajs, args.horizon, action_dim)
        paths = []
        for n in range(args.n_trajs):
            observations = []
            actions = []
            rewards = []
            observation = env.reset()
            for t in range(args.horizon):
                action = policy(observation, noises[n][t])
                next_observation, reward, terminal, _ = env.step(action)
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                observation = next_observation
                if terminal:
                    break
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
        optim.zero_grad()
        loss = reinforce_loss(observations, actions, returns, policy)
        loss.backward()
        optim.step()
        iter_returns.append(np.mean([sum(p["rewards"]) for p in paths]))
        print('Average Return:', iter_returns[-1])
    if args.save_fn:
        Path(args.save_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_fn, 'w') as f:
            for ret in iter_returns:
                f.write('{}\n'.format(ret))

if __name__ == "__main__":
    with launch_ipdb_on_exception():
        main()
