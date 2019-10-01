import sys
import gym
import torch
import copy
import dill
import argparse
import importlib
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm, trange
from ipdb import slaunch_ipdb_on_exception
from pathlib import Path

import exps
import postprocess
from envs import *
from models import GaussianPolicy, get_mlp
from utils import MPSampler, SeqSampler, ArrayRQMCSampler, VecSampler, rollout
from utils import set_seed, select_device, tensor, reinforce_loss, variance_reduced_loss, no_loss, running_seeds, collect_seeds, get_gradient, sort_by_optimal_value, sort_by_norm, multdim_sort, no_sort, random_permute, logger, debug, Config, HorizonWrapper
from torch.distributions import Uniform, Normal
from rqmc_distributions import Uniform_RQMC, Normal_RQMC

# DEBUG FLAG
Config.DEBUG = True

# TODO:
# (done) implement discount
# async vec env, batch action

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        choices=['cost', 'grad', 'learn'],
        default='learn')
    parser.add_argument('--algos', default=['mc', 'rqmc', 'arqmc'], nargs='+', choices=['mc', 'rqmc', 'arqmc']) # learning use it
    parser.add_argument('--env', choices=['lqr', 'cartpole', 'swimmer', 'ant', 'pointmass'], default='lqr')
    parser.add_argument('--map_name', type=str, default='8x8') # for pointmass only
    parser.add_argument('--xu_dim', type=int, nargs=2, default=(20, 12))
    parser.add_argument('--init_scale', type=float, default=3.0)
    parser.add_argument('--PQ_kappa', type=float, default=3.0)
    parser.add_argument('--AB_norm', type=float, default=1.0)
    parser.add_argument('-H', type=int, default=10, help='horizon')
    parser.add_argument('--noise', type=float, default=0.0, help='noise scale')
    parser.add_argument('--sorter', nargs='+', choices=['value', 'norm', 'none', 'permute', 'group'], default=['value'])
    parser.add_argument('--n_trajs', type=int, default=800, help='number of trajectories used')
    parser.add_argument('--n_iters', type=int, default=200, help='number of iterations of training')
    parser.add_argument('-lr', type=float, default=5e-5)
    parser.add_argument('--init_policy', choices=['optimal', 'linear', 'linear_bias', 'mlp'], default='linear')
    parser.add_argument('--fix_std', action='store_true')
    parser.add_argument('--gate_output', action='store_true')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[16])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_fig', action='store_true')
    parser.add_argument('--save_fig', type=str, default=None)
    parser.add_argument('--mode', choices=['single', 'seeds', 'collect'], default='single')
    parser.add_argument('--n_seeds', type=int, default=200)
    parser.add_argument('--max_seed', type=int, default=100)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--save_fn', type=str, default=None)
    parser.add_argument('--post_f', type=str, default=None) # post processing function
    parser.add_argument('--cpu', action='store_true')
    args = exps.parse_args(parser, args, exp_name_attr='save_fn')
    return args

LQR_ENVS = ['lqr']

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
            lims=100,
        )
    elif args.env == 'cartpole':
        env = HorizonWrapper(CartPoleContinuousEnv(), args.H)
        #env = CartPoleContinuousEnv()
    elif args.env == 'ant':
        env = HorizonWrapper(gym.make('Ant-v2'), args.H)
    elif args.env == 'swimmer':
        env = HorizonWrapper(gym.make('Swimmer-v2'), args.H)
    elif args.env == 'pointmass':
        env = HorizonWrapper(PointMass(args.map_name, goal=(2, 2), init_pos=(8, 8)), args.H)
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
        mean_network = get_mlp((N,) + tuple(args.hidden_sizes) + (M,), gate=nn.Tanh)
    else:
        raise Exception('unsupported policy type')
    return GaussianPolicy(N, M, mean_network, learn_std=not args.fix_std, gate_output=args.gate_output)

def get_rqmc_noises(n_trajs, n_steps, action_dim, noise_type):
    if noise_type == 'trajwise':
        loc = torch.zeros(n_steps * action_dim)
        scale = torch.ones(n_steps * action_dim)
        noises = Normal_RQMC(loc, scale).sample(torch.Size([n_trajs])).data.numpy().reshape((n_trajs, n_steps, action_dim))
    elif noise_type == 'array':
        from scipy.stats import norm
        loc = torch.zeros(action_dim)
        scale = torch.ones(action_dim)
        noises = np.asarray([Normal_RQMC(loc, scale).sample(torch.Size([n_trajs])).data.numpy() for _ in range(n_steps)]).reshape(n_steps, n_trajs, action_dim).transpose(1, 0, 2)
    else:
        raise Exception('unknown rqmc type')
    return noises

def get_sorter(sorter, env):
    if sorter == 'value':
        return lambda pairs: sorted(pairs, key=sort_by_optimal_value(env))
    elif sorter == 'norm':
        return lambda pairs: sorted(pairs, key=sort_by_norm(env))
    elif sorter == 'permute':
        return random_permute
    elif sorter == 'none':
        return no_sort
    elif sorter == 'group':
        return multdim_sort
    else:
        raise Exception('unknown sorter')

# it does not make sense to compare array RQMC in cumulative case, since it treated all trajectories together, but let's see what happen
def compare_cost(args):
    set_seed(args.seed)
    env = LQR(
        #N=20,
        #M=12,
        init_scale=1.0,
        max_steps=args.H, # 10, 20
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
    rqmc_noises = get_rqmc_noises(args.n_trajs, env.max_steps, env.M, 'trajwise')
    for i in tqdm(range(args.n_trajs), 'rqmc'):
        _, _, rewards, _, _ = rollout(env, policy, rqmc_noises[i])
        rqmc_costs.append(-rewards.sum())
        rqmc_means.append(np.mean(rqmc_costs))

    # array rqmc
    arqmc_costs_dict = {}
    arqmc_means_dict = {}
    arqmc_noises = get_rqmc_noises(args.n_trajs, env.max_steps, env.M, 'array')

    for sorter in args.sorter:
        arqmc_costs = []
        arqmc_means = []
        sort_f = get_sorter(sorter, env)

        data = ArrayRQMCSampler(env, args.n_trajs, sort_f=sort_f).sample(policy, arqmc_noises)
        for traj in data:
            rewards = np.asarray(traj['rewards'])
            arqmc_costs.append(-rewards.sum())
            arqmc_means.append(np.mean(arqmc_costs))
        arqmc_costs_dict[sorter] = arqmc_costs
        arqmc_means_dict[sorter] = arqmc_means

    expected_cost = env.expected_cost(K, np.diag(np.ones(env.M)))

    mc_errors = np.abs(mc_means - expected_cost)
    rqmc_errors = np.abs(rqmc_means - expected_cost)
    arqmc_errors_dict = {sorter: np.abs(arqmc_means - expected_cost) for sorter, arqmc_means in arqmc_means_dict.items()}
    logger.info('mc: {}, rqmc: {} '.format(mc_errors[-1], rqmc_errors[-1]) + \
        ' '.join(['arqmc ({}): {}'.format(sorter, arqmc_errors[-1]) for sorter, arqmc_errors in arqmc_errors_dict.items()]))
    info = {**vars(args), 'mc_costs': mc_costs, 'rqmc_costs': rqmc_costs, 'arqmc_costs': arqmc_costs}
    if args.save_fn is not None:
        with open(args.save_fn, 'wb') as f:
            dill.dump(dict(mc_errors=mc_errors, rqmc_errors=rqmc_errors, arqmc_errors_dict=arqmc_errors_dict, info=info), f)
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
            pd.concat([
                pd.DataFrame({
                    'name': 'arqmc_{}'.format(sorter),
                    'x': np.arange(len(arqmc_errors)),
                    'error': arqmc_errors,
                })
                for sorter, arqmc_errors in arqmc_errors_dict.items()
            ]),
        ])
        plot = sns.lineplot(x='x', y='error', hue='name', data=data)
        plot.set(yscale='log')
        plt.show()
    return mc_errors, rqmc_errors, arqmc_errors_dict, info

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
    mc_grads = []
    for i in tqdm(range(args.n_trajs), 'mc'):
        noises = np.random.randn(env.max_steps, env.M)
        states, actions, rewards, _, _ = rollout(env, policy, noises)
        #mc_grads.append(get_gradient(states, actions, rewards, policy, reinforce_loss))
        mc_grads.append(get_gradient(states, actions, rewards, policy, variance_reduced_loss))
    mc_grads = np.asarray(mc_grads)
    mc_means = np.cumsum(mc_grads, axis=0) / np.arange(1, len(mc_grads) + 1)[:, np.newaxis, np.newaxis]

    rqmc_grads = []
    loc = torch.zeros(env.max_steps * env.M)
    scale = torch.ones(env.max_steps * env.M)
    rqmc_noises = Normal_RQMC(loc, scale).sample(torch.Size([args.n_trajs])).data.numpy()
    for i in tqdm(range(args.n_trajs), 'rqmc'):
        states, actions, rewards, _, _ = rollout(env, policy, rqmc_noises[i].reshape(env.max_steps, env.M))
        #rqmc_grads.append(get_gradient(states, actions, rewards, policy, reinforce_loss))
        rqmc_grads.append(get_gradient(states, actions, rewards, policy, variance_reduced_loss))
    rqmc_grads = np.asarray(rqmc_grads)
    rqmc_means = np.cumsum(rqmc_grads, axis=0) / np.arange(1, len(rqmc_grads) + 1)[:, np.newaxis, np.newaxis]

    arqmc_grads = []
    arqmc_noises = get_rqmc_noises(args.n_trajs, args.H, env.M, 'array')
    sort_f = get_sorter(args, env)
    data = ArrayRQMCSampler(env, args.n_trajs, sort_f=sort_f).sample(policy, arqmc_noises)
    for traj in data:
        states, actions, rewards = np.asarray(traj['states']), np.asarray(traj['actions']), np.asarray(traj['rewards'])
        #arqmc_grads.append(get_gradient(states, actions, rewards, policy, reinforce_loss))
        arqmc_grads.append(get_gradient(states, actions, rewards, policy, variance_reduced_loss))
    arqmc_grads = np.asarray(arqmc_grads)
    arqmc_means = np.cumsum(arqmc_grads, axis=0) / np.arange(1, len(arqmc_grads) + 1)[:, np.newaxis, np.newaxis]

    expected_grad = env.expected_policy_gradient(K, Sigma_a)

    mc_errors = ((mc_means - expected_grad) ** 2).reshape(mc_means.shape[0], -1).mean(1) # why the sign is reversed?
    rqmc_errors = ((rqmc_means - expected_grad) ** 2).reshape(rqmc_means.shape[0], -1).mean(1)
    arqmc_errors = ((arqmc_means - expected_grad) ** 2).reshape(arqmc_means.shape[0], -1).mean(1)
    info = {**vars(args)}
    if args.save_fn is not None:
        with open(save_fn, 'wb') as f:
            dill.dump(dict(mc_errors=mc_errors, rqmc_errors=rqmc_errors, arqmc_errors=arqmc_errors, info=info), f)
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
        arqmc_data = pd.DataFrame({
            'name': 'arqmc',
            'x': np.arange(len(arqmc_errors)),
            'error': arqmc_errors,
        })
        plot = sns.lineplot(x='x', y='error', hue='name', data=pd.concat([mc_data, rqmc_data, arqmc_data]))
        plot.set(yscale='log')
        plt.show()
    return mc_errors, rqmc_errors, arqmc_errors, info

def learning(args):
    set_seed(args.seed)
    env = get_env(args)
    if Config.DEVICE.type == 'cpu': 
        #sampler = MPSampler(env, args.n_workers) # mp
        sampler = SeqSampler(env) # sequential
    else:
        sampler = VecSampler(env, args.n_trajs) # a simplied version where the number of workers == the number of trajs
    sort_f = get_sorter(args.sorter[0], env)
    arqmc_sampler = ArrayRQMCSampler(env, args.n_trajs, sort_f=sort_f)
    init_policy = get_policy(args, env)
    out_set = set()
    def train(name, loss_fn, init_policy, noise_type='mc', n_iters=None):
        if n_iters is None: n_iters = args.n_iters
        policy = copy.deepcopy(init_policy)
        optim = torch.optim.SGD(policy.parameters(), args.lr)
        all_returns = []
        prog = trange(n_iters, desc=name)
        N = env.observation_space.shape[0]
        M = env.action_space.shape[0]
        for _ in prog:
            if name in out_set:
                all_returns.append(np.nan)
                continue
            returns = []
            loss = [] # policy gradient loss
            if noise_type == 'mc': # mc, rqmc, arqmc
                noises = np.random.randn(args.n_trajs, env.max_steps, M)
            elif noise_type == 'rqmc':
                noises = get_rqmc_noises(args.n_trajs, env.max_steps, M, 'trajwise')
            elif noise_type == 'arqmc':
                noises = get_rqmc_noises(args.n_trajs, env.max_steps, M, 'array')
            else:
                raise Exception('unknown sequence type')
            if noise_type in ['mc', 'rqmc']:
                data = sampler.sample(policy, noises)
            else:
                data = arqmc_sampler.sample(policy, noises)
            if isinstance(data[0], dict): # from arrayrqmcsampler
                data = [(np.asarray(d['states']), np.asarray(d['actions']), np.asarray(d['rewards'])) for d in data]
            for traj in data:
                states, actions, rewards = traj[:3]
                loss.append(loss_fn(states, actions, rewards, policy))
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
    results = {}
    if 'mc' in args.algos: results['mc'] = train('mc', variance_reduced_loss, init_policy, noise_type='mc')
    if 'rqmc' in args.algos: results['rqmc'] = train('rqmc', variance_reduced_loss, init_policy, noise_type='rqmc')
    if 'arqmc' in args.algos: results['arqmc'] = train('arqmc', variance_reduced_loss, init_policy, noise_type='arqmc')
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

def main(args=None):
    args = parse_args(args)

    select_device(0 if torch.cuda.is_available() and not args.cpu else -1)
    #select_device(-1)
    logger.prog('device: {}'.format(Config.DEVICE))

    if args.task == 'learn':
        exp_f = learning
    elif args.task == 'cost':
        exp_f = compare_cost
    elif args.task == 'grad':
        exp_f = compare_grad
    else:
        raise Exception('unsupported task')
    if args.post_f is not None:
        post_f = lambda results: getattr(postprocess, args.post_f)(args, results)
    else: post_f = None
    if args.mode == 'single':
        exp_f(args)
    elif args.mode == 'seeds':
        running_seeds(args.save_fn, exp_f, argparse.Namespace(**vars(args)), args.n_seeds, post_f=post_f)
    elif args.mode == 'collect':
        assert args.task in ['learn']
        success_f = lambda result: len(result[1]['out']) == 0
        collect_seeds(args.save_fn, exp_f, args, success_f=success_f, n_seeds=args.n_seeds, max_seed=args.max_seed, post_f=post_f)


if __name__ == '__main__':
    exps.run_one_exp(main)
