import sys
import gym
import torch
import copy
import dill
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm, trange
from ipdb import slaunch_ipdb_on_exception
from pathlib import Path

from envs import *
from models import GaussianPolicy, get_mlp, PGPELinearPolicy, MovingAverageCritic
from utils import set_seed, rollout, mse, cummean, MPSampler, SeqRunner, select_device, tensor, reinforce_loss, variance_reduced_loss, no_loss, running_seeds, collect_seeds, get_gradient, sort_by_norm, sort_by_optimal_value, ArrayRQMCSampler, SortableWrapper, critic_loss, PGPE_loss
from torch.distributions import Uniform, Normal
from rqmc_distributions import Uniform_RQMC, Normal_RQMC, dist_rqmc

# TODO:
# (done) implement discount
# check why stepwise does not work
# value estimation with critic
# how to quickly cut unpromising configuration?
# implement thread to generate sobol sequence to acclerate training

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        choices=['cost', 'grad', 'learn', 'critic', 'learn_PGPE'],
        default='learn')
    parser.add_argument('--env', choices=['lqr', 'WIP', 'IP', 'cartpole', 'ant'], default='lqr')
    parser.add_argument('--xu_dim', type=int, nargs=2, default=(20, 12))
    parser.add_argument('--init_scale', type=float, default=3.0)
    parser.add_argument('--PQ_kappa', type=float, default=3.0)
    parser.add_argument('--AB_norm', type=float, default=1.0)
    parser.add_argument('-H', type=int, default=10, help='horizon')
    parser.add_argument('--noise', type=float, default=0.0, help='noise scale')
    parser.add_argument('--rqmc_type', choices=['stepwise', 'trajwise', 'array'], default='trajwise')
    parser.add_argument('--n_trajs', type=int, default=800, help='number of trajectories used')
    parser.add_argument('--n_iters', type=int, default=200, help='number of iterations of training')
    parser.add_argument('-lr', type=float, default=5e-5)
    parser.add_argument('--init_policy', choices=['optimal', 'linear', 'linear_bias', 'mlp'], default='linear')
    parser.add_argument('--use_critic', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_fig', action='store_true')
    parser.add_argument('--save_fig', type=str, default=None)
    parser.add_argument('--mode', choices=['single', 'seeds', 'collect'], default='single')
    parser.add_argument('--n_seeds', type=int, default=200)
    parser.add_argument('--max_seed', type=int, default=100)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--save_fn', type=str, default=None)
    return parser.parse_args(args)

LQR_ENVS = ['lqr', 'WIP', 'IP']

def get_env(args):
    if args.env == 'lqr':
        env = LQR(
            N=args.xu_dim[0],
            M=args.xu_dim[1],
            init_scale=args.init_scale,
            max_steps=args.H,
            Sigma_s_kappa=1.0,
            Q_kappa=args.PQ_kappa,
            P_kappa=args.PQ_kappa,
            A_norm=args.AB_norm,
            B_norm=args.AB_norm,
            Sigma_s_scale=args.noise,
            #random_init=True,
        )
    elif args.env == 'WIP':
        env = WIP(
            init_scale=args.init_scale,
            max_steps=args.H,
            Sigma_s_scale=args.noise,
        )
    elif args.env == 'IP':
        env = InvertedPendulum(
            init_scale=args.init_scale,
            max_steps=args.H,
            Sigma_s_scale=args.noise,
        )
    elif args.env == 'cartpole':
        env = CartPoleContinuousEnv()
    elif args.env == 'gym':
        env = gym.make('Ant-v2')
    else:
        raise Exception('unsupported lqr env')
    return env

def get_policy(args, env):
    N = env.observation_space.shape[0]
    M = env.action_space.shape[0]
    if args.init_policy == 'optimal':
        K = env.optimal_controller()
        mean_network = nn.Linear(*K.shape[::-1], bias=False)
        mean_network.weight.data = tensor(K)
    elif args.init_policy == 'linear':
        K = np.random.randn(M, N)
        mean_network = nn.Linear(*K.shape[::-1], bias=False)
        mean_network.weight.data = tensor(K)
    elif args.init_policy == 'linear_bias':
        K = np.random.randn(M, N)
        mean_network = nn.Linear(*K.shape[::-1], bias=True)
        mean_network.weight.data = tensor(K)
    elif args.init_policy == 'mlp':
        mean_network = get_mlp((N, 16, M), gate=nn.Tanh)
    else:
        raise Exception('unsupported policy type')
    return GaussianPolicy(N, M, mean_network)

def get_rqmc_noises(n_trajs, n_steps, action_dim, noise_type):
    if noise_type == 'stepwise':
        #loc = torch.zeros(action_dim)
        #scale = torch.ones(action_dim)
        #noises = []
        #for i in range(0, n_trajs, 100):
            #sub_noises = Normal_RQMC(loc, scale).sample(torch.Size([100, n_steps])).data.numpy()
            #noises.append(sub_noises)
        #noises = np.concatenate(noises, 0)
        #noises = Normal_RQMC(loc, scale).sample(torch.Size([n_trajs, n_steps])).data.numpy()
        from scipy.stats import norm
        noises = dist_rqmc.Uniform_RQMC(dim=action_dim).sample(n_steps)
        noises = norm.ppf((np.random.randn(n_trajs, n_steps, action_dim) + noises) % 1.0)
    elif noise_type == 'trajwise':
        loc = torch.zeros(n_steps * action_dim)
        scale = torch.ones(n_steps * action_dim)
        noises = Normal_RQMC(loc, scale).sample(torch.Size([n_trajs])).data.numpy().reshape((n_trajs, n_steps, action_dim))
    elif noise_type == 'array':
        from scipy.stats import norm
        noises = dist_rqmc.Uniform_RQMC(dim=action_dim).sample(n_trajs).reshape((n_trajs, 1, action_dim))
        noises = norm.ppf((noises + np.random.randn(n_trajs, n_steps, action_dim)) % 1.0)
    else:
        raise Exception('unknown rqmc type')
    return noises

def compare_cost(args):
    set_seed(args.seed)
    env = LQR(
        #N=20,
        #M=12,
        init_scale=1.0,
        max_steps=10,
        #max_steps=20,
        Sigma_s_kappa=1.0,
        Q_kappa=1.0,
        P_kappa=1.0,
        A_norm=1.0,
        B_norm=1.0,
        Sigma_s_scale=0.0,
    )
    K = env.optimal_controller()
    mean_network = nn.Linear(*K.shape[::-1], bias=False)
    mean_network.weight.data = tensor(K)
    policy = GaussianPolicy(*K.shape[::-1], mean_network, learn_std=False, gate_output=False)

    # mc
    mc_costs = [] # individual
    mc_means = [] # cumulative
    for i in tqdm(range(args.n_trajs), 'mc'):
        noises = np.random.randn(env.max_steps, env.M)
        _, _, rewards, _, _ = rollout(env, policy, noises)
        mc_costs.append(-rewards.sum())
        mc_means.append(np.mean(mc_costs))

    # rqmc
    rqmc_costs = []
    rqmc_means = []
    rqmc_noises = get_rqmc_noises(args.n_trajs, env.max_steps, env.M, args.rqmc_type)
    for i in tqdm(range(args.n_trajs), 'rqmc'):
        _, _, rewards, _, _ = rollout(env, policy, rqmc_noises[i].reshape(env.max_steps, env.M))
        rqmc_costs.append(-rewards.sum())
        rqmc_means.append(np.mean(rqmc_costs))

    # array rqmc
    array_costs = []
    array_means = []
    data = ArrayRQMCSampler(SortableWrapper(env), n_envs=args.n_trajs, sort_f=sort_by_optimal_value).sample(policy, 1)
    for traj in data:
        _, _, rewards = traj
        rewards = np.asarray(rewards)
        array_costs.append(-rewards.sum())
        array_means.append(np.mean(array_costs))

    expected_cost = env.expected_cost(K, np.diag(np.ones(env.M)))

    mc_errors = np.abs(mc_means - expected_cost)
    rqmc_errors = np.abs(rqmc_means - expected_cost)
    array_errors = np.abs(array_means - expected_cost)
    info = {**vars(args), 'mc_costs': mc_costs, 'rqmc_costs': rqmc_costs, 'array_costs': array_costs}
    if args.save_fn is not None:
        with open(args.save_fn, 'wb') as f:
            dill.dump(dict(mc_errors=mc_errors, rqmc_errors=rqmc_errors, array_errors=array_errors, info=info), f)
    if args.show_fig:
        data = pd.concat([
            pd.DataFrame({
                'name': 'mc',
                'x': np.arange(len(mc_errors)),
                'error': mc_errors,
            }),
            pd.DataFrame({
                'name': 'rqmc',
                'x': np.arange(len(rqmc_errors)),
                'error': rqmc_errors,
            }),
            pd.DataFrame({
                'name': 'array',
                'x': np.arange(len(array_errors)),
                'error': array_errors,
            }),
        ])
        plot = sns.lineplot(x='x', y='error', hue='name', data=data)
        plot.set(yscale='log')
        plt.show()
    return mc_errors, rqmc_errors, array_errors, info

def compare_grad(args):
    set_seed(args.seed)
    env = LQR(
        lims=100,
        init_scale=1.0,
        max_steps=args.H,
        Sigma_s_kappa=1.0,
        Q_kappa=1.0,
        P_kappa=1.0,
        A_norm=1.0,
        B_norm=1.0,
        Sigma_s_scale=args.noise,
    )
    K = env.optimal_controller()
    #K = np.random.randn(env.M, env.N) # debug, this one seems to work worse, by 1 magnitude
    mean_network = nn.Linear(*K.shape[::-1], bias=False)
    mean_network.weight.data = tensor(K)
    policy = GaussianPolicy(*K.shape[::-1], mean_network, learn_std=False, gate_output=False)

    Sigma_a = np.diag(np.ones(env.M))
    Sigma_a_inv = np.linalg.inv(Sigma_a)
    print(env.Sigma_s)
    mc_grads = []
    for i in tqdm(range(args.n_trajs), 'mc'):
        noises = np.random.randn(env.max_steps, env.M)
        states, actions, rewards = rollout(env, policy, noises)
        mc_grads.append(get_gradient(states, actions, rewards, policy, reinforce_loss))
    mc_grads = np.asarray(mc_grads)
    mc_means = np.cumsum(mc_grads, axis=0) / np.arange(1, len(mc_grads) + 1)[:, np.newaxis, np.newaxis]

    rqmc_grads = []
    loc = torch.zeros(env.max_steps * env.M)
    scale = torch.ones(env.max_steps * env.M)
    rqmc_noises = Normal_RQMC(loc, scale).sample(torch.Size([args.n_trajs])).data.numpy()
    for i in tqdm(range(args.n_trajs), 'rqmc'):
        states, actions, rewards = rollout(env, policy, rqmc_noises[i].reshape(env.max_steps, env.M))
        rqmc_grads.append(get_gradient(states, actions, rewards, policy, reinforce_loss))
    rqmc_grads = np.asarray(rqmc_grads)
    rqmc_means = np.cumsum(rqmc_grads, axis=0) / np.arange(1, len(rqmc_grads) + 1)[:, np.newaxis, np.newaxis]

    expected_grad = env.expected_policy_gradient(K, Sigma_a)

    mc_errors = ((mc_means - expected_grad) ** 2).reshape(mc_means.shape[0], -1).mean(1) # why the sign is reversed?
    rqmc_errors = ((rqmc_means - expected_grad) ** 2).reshape(rqmc_means.shape[0], -1).mean(1)
    info = {**vars(args)}
    if args.save_fn is not None:
        with open(save_fn, 'wb') as f:
            dill.dump(dict(mc_errors=mc_errors, rqmc_errors=rqmc_errors, info=info), f)
    if args.show_fig:
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

def PGPE(args):
    set_seed(args.seed)
    env = get_env(args)
    #sampler = MPSampler(env, args.n_workers, deterministic=True) # mp
    sampler = SeqRunner(env, deterministic=True) # sequential
    init_policy = PGPELinearPolicy(env.N, env.M)
    print(init_policy)
    out_set = set()
    def train(name, loss_fn, init_policy, use_rqmc=False, n_iters=None):
        if n_iters is None: n_iters = args.n_iters
        policy = copy.deepcopy(init_policy)
        critic = MovingAverageCritic()
        optim = torch.optim.SGD(policy.parameters(), args.lr)
        all_returns = []
        prog = trange(n_iters, desc=name)
        N = env.observation_space.shape[0]
        M = env.action_space.shape[0]
        for _ in prog:
            if name in out_set or (name == 'full' and len(out_set) == 2): # fast skip
                all_returns.append(np.nan)
                continue
            returns = []
            loss = []
            if use_rqmc:
                loc = torch.zeros(policy.parameter_dim)
                scale = torch.ones(policy.parameter_dim)
                noises = Normal_RQMC(loc, scale).sample(torch.Size([args.n_trajs])).data.numpy()
            else:
                noises = np.random.randn(args.n_trajs, policy.parameter_dim)
            data = sampler.sample(policy, noises) # mp
            for traj, noise in zip(data, noises):
                states, actions, rewards = traj[:3]
                #loss.append(loss_fn(rewards, policy, noise, critic=critic))
                loss.append(loss_fn(rewards, policy, noise))
                critic.update(rewards.sum()) # critic
                returns.append(rewards.sum())
                if len(states) != args.H and args.env in LQR_ENVS:
                    out_set.add(name)
            optim.zero_grad()
            loss = -torch.mean(torch.stack(loss))
            loss.backward()
            optim.step()
            all_returns.append(np.mean(returns))
            prog.set_postfix(ret=all_returns[-1])
        return np.asarray(all_returns)
    results = dict(
        mc=train('mc', PGPE_loss, init_policy),
        rqmc=train('rqmc', PGPE_loss, init_policy, use_rqmc=True),
        #full=train('full', init_K, full_grad), # this is only available for linear policy
    )
    #if args.env in LQR_ENVS: # later
        #results['optimal'] = train('optimal', no_loss, get_policy(argparse.Namespace(init_policy='optimal'), env), n_iters=1).repeat(args.n_iters)
    if args.show_fig or args.save_fig is not None:
        valid_results = {k: v for k, v in results.items() if k not in out_set}
        costs = pd.concat([pd.DataFrame({'name': name, 'x': np.arange(len(rs)), 'cost': -rs}) for name, rs in valid_results.items()])
        plot = sns.lineplot(x='x', y='cost', hue='name', data=costs)
        plt.yscale('log')
        if args.save_fig:
            plt.savefig(args.save_fig)
        if args.show_fig:
            plt.show()
    info = {**vars(args), 'out': out_set}
    return results, info


def learning(args):
    set_seed(args.seed)
    env = get_env(args)
    sampler = MPSampler(env, args.n_workers) # mp
    #sampler = SeqRunner(env) # sequential
    init_policy = get_policy(args, env)
    print(init_policy)
    out_set = set()
    def train(name, loss_fn, init_policy, use_rqmc=False, n_iters=None):
        if n_iters is None: n_iters = args.n_iters
        policy = copy.deepcopy(init_policy)
        optim = torch.optim.SGD(policy.parameters(), args.lr)
        all_returns = []
        prog = trange(n_iters, desc=name)
        N = env.observation_space.shape[0]
        M = env.action_space.shape[0]
        for _ in prog:
            if name in out_set or (name == 'full' and len(out_set) == 2): # fast skip
                all_returns.append(np.nan)
                continue
            returns = []
            loss = [] # policy gradient loss
            if use_rqmc:
                noises = get_rqmc_noises(args.n_trajs, env.max_steps, M, args.rqmc_type)
                #loc = torch.zeros(env.max_steps * M)
                #scale = torch.ones(env.max_steps * M)
                #noises = Normal_RQMC(loc, scale).sample(torch.Size([args.n_trajs])).data.numpy().reshape(args.n_trajs, env.max_steps, M)
            else:
                noises = np.random.randn(args.n_trajs, env.max_steps, M)
            data = sampler.sample(policy, noises) # mp
            for states, actions, rewards, _, _ in data:
                loss.append(loss_fn(states, actions, rewards, policy))
                returns.append(rewards.sum())
                if len(states) != args.H and args.env in LQR_ENVS:
                    out_set.add(name)
            optim.zero_grad()
            loss = -torch.mean(torch.stack(loss))
            ### another way to calculate loss ###
            #states, actions, rewards = zip(*data)
            #states, actions, rewards = sum(states, []), sum(actions, []), sum(rewards, [])
            #loss = -torch.mean(loss_fn(states, actions, rewards, ))
            ### end of another way ###
            loss.backward()
            optim.step()
            all_returns.append(np.mean(returns))
            prog.set_postfix(ret=all_returns[-1])
        return np.asarray(all_returns)
    results = dict(
        mc=train('mc', variance_reduced_loss, init_policy),
        rqmc=train('rqmc', variance_reduced_loss, init_policy, use_rqmc=True),
        #full=train('full', init_K, full_grad), # this is only available for linear policy
    )
    if args.env in LQR_ENVS:
        results['optimal'] = train('optimal', no_loss, get_policy(argparse.Namespace(init_policy='optimal'), env), n_iters=1).repeat(args.n_iters)
    if args.show_fig or args.save_fig is not None:
        valid_results = {k: v for k, v in results.items() if k not in out_set}
        costs = pd.concat([pd.DataFrame({'name': name, 'x': np.arange(len(rs)), 'cost': -rs}) for name, rs in valid_results.items()])
        plot = sns.lineplot(x='x', y='cost', hue='name', data=costs)
        plt.yscale('log')
        if args.save_fig:
            plt.savefig(args.save_fig)
        if args.show_fig:
            plt.show()
    info = {**vars(args), 'out': out_set}
    return results, info

def learn_critic(args):
    set_seed(args.seed)
    env = get_env(args)
    sampler = MPSampler(env, args.n_workers) # mp
    #sampler = SeqRunner(env) # sequential
    init_policy = get_policy(args, env)
    init_critic = get_mlp((env.N, 32, 1), gate=nn.ReLU, output_gate=None)
    def train(name, loss_fn, init_policy, init_critic, use_rqmc=False, n_iters=None):
        if n_iters is None: n_iters = args.n_iters
        policy = copy.deepcopy(init_policy)
        critic = copy.deepcopy(init_critic)
        optim = torch.optim.SGD(critic.parameters(), args.lr)
        all_loss = []
        prog = trange(n_iters, desc=name)
        N = env.observation_space.shape[0]
        M = env.action_space.shape[0]
        for _ in prog:
            returns = []
            loss = []
            if use_rqmc:
                loc = torch.zeros(env.max_steps * M)
                scale = torch.ones(env.max_steps * M)
                noises = Normal_RQMC(loc, scale).sample(torch.Size([args.n_trajs])).data.numpy().reshape(args.n_trajs, env.max_steps, M)
            else:
                noises = np.random.randn(args.n_trajs, env.max_steps, M)
            data = sampler.sample(policy, noises)
            for states, actions, rewards, next_states, terminals in data:
                returns.append(np.cumsum(rewards[::-1])[::-1] + (1 - terminals) * critic(tensor(next_states)).squeeze(1).detach().numpy())
            loss = loss_fn(states, returns, critic)
            optim.zero_grad()
            loss.backward()
            optim.step()
            all_loss.append(loss.data.numpy())
            prog.set_postfix(loss=all_loss[-1])
        return all_loss
    results = dict(
        mc=train('mc', critic_loss, init_policy, init_critic),
        rqmc=train('rqmc', critic_loss, init_policy, init_critic, use_rqmc=True),
    )
    if args.show_fig or args.save_fig is not None:
        errors = pd.concat([pd.DataFrame({'name': name, 'x': np.arange(len(rs)), 'error': errs}) for name, errs in results.items()])
        plot = sns.lineplot(x='x', y='error', hue='name', data=errors)
        #plt.yscale('log')
        if args.save_fig:
            plt.savefig(args.save_fig)
        if args.show_fig:
            plt.show()
    info = {**vars(args)}
    return results, info


def main(args=None):
    #select_device(0 if torch.cuda.is_available() else -1)
    select_device(-1)
    args = parse_args(args)
    if args.task == 'learn':
        exp_f = learning
    elif args.task == 'cost':
        exp_f = compare_cost
    elif args.task == 'grad':
        exp_f = compare_grad
    elif args.task == 'critic':
        exp_f = learn_critic
    elif args.task == 'learn_PGPE':
        exp_f = PGPE
    else:
        raise Exception('unsupported task')
    if args.mode == 'single':
        exp_f(args)
    elif args.mode == 'seeds':
        running_seeds(args.save_fn, exp_f, argparse.Namespace(**vars(args)), args.n_seeds)
    elif args.mode == 'collect':
        assert args.task in ['learn', 'learn_PGPE']
        success_f = lambda result: len(result[1]['out']) == 0
        collect_seeds(args.save_fn, exp_f, args, success_f=success_f, n_seeds=args.n_seeds, max_seed=args.max_seed)

if __name__ == "__main__":
    with slaunch_ipdb_on_exception():
        main()
