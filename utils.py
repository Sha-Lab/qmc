import gym
import sys
import copy
import dill
import json
import numpy as np
import random
import torch
import shlex
import argparse
import inspect
import filelock
import traceback
import subprocess
import torch.multiprocessing as mp
import torch.nn.functional as F
from inspect import signature
from pathlib import Path
from termcolor import colored
from scipy.stats import norm
from contextlib import contextmanager

from rqmc_distributions import dist_rqmc

class Config:
    DEVICE = torch.device('cpu')
    SEED_RANGE = 100000000
    DEBUG = False # for debug usage

class Logger:
    def __init__(self):
        self.logger = dict()
        self.default_logger = None

    def reset(self):
        for name in self.logger: # may not be save
            self.close_logger(name)
        self.logger = dict()
        self.default_logger = None

    def get_logger(self, name, outs=[sys.stdout]):
        self.logger[name] = outs

    def close_logger(self, name, exclude=[sys.stdout]):
        if name not in self.logger: return
        for f in self.logger[name]:
            if f in exclude: continue
            f.close()

    def set_default(self, name):
         self.default_logger = name

    @contextmanager
    def as_default(self, name):
        pre_logger = self.default_logger
        self.default_logger = name
        try:
            yield self
        finally:
            self.default_logger = pre_logger

    def log(self, info, name=None, outs=None):
        if name is None:
            name = self.default_logger
        if name is not None: outs = self.logger[name]
        for out in outs:
            #print >> out, info # py2.7 only
            print(info, file=out)
            out.flush()

    # passed in a list of info
    def logs(self, infos, name=None, outs=None):
        if name is None:
            name = self.default_logger
        if name is not None: outs = self.logger[name]
        for info in infos:
            for out in outs:
                #print >> out, info
                print(info, file=out)
                out.flush()

    # these use sys.stdout
    def info(self, msg):
        print(msg)

    def prog(self, msg):
        cprint(msg, 'green')

    def error(self, msg):
        cprint(msg, 'red')

    def debug(self, msg):
        cprint(msg, 'blue')

    def log_experiment_info(self, args, name=None, outs=None):
        self.log(get_git_sha(), name=name, outs=outs)
        self.log(args, name=name, outs=outs)

logger = Logger()

def select_device(gpu_id=-1):
    if gpu_id >= 0:
        Config.DEVICE = torch.device('cuda:%d' % (gpu_id))
    else:
        Config.DEVICE = torch.device('cpu')

def tensor(x, dtype=torch.float32, requires_grad=False): # for better precision!
    if torch.is_tensor(x):
        return x.to(dtype=dtype, device=Config.DEVICE)
    x = torch.tensor(x, device=Config.DEVICE, dtype=dtype, requires_grad=requires_grad)
    return x

def is_git_diff():
    return bool(subprocess.check_output(['git', 'diff']))

# commandr
_cmd_dict = {}  

def cmd(name=None):
    def f(g):
        nonlocal name
        if name is None:
            name = g.__name__
        _cmd_dict[name] = g 
        return g
    return f

def parse_args_as_func(argv):
    args = []
    kwargs = {}
    i = 0 
    while i < len(argv):
        if argv[i].startswith('-'):
            kwargs[argv[i].lstrip('-')] = argv[i+1]
            i += 2
        else:
            args.append(argv[i])
            i += 1
    return args, kwargs

def annotate(arg, p):
    if isinstance(p.annotation, inspect._empty):
        return arg
    return p.annotation(arg)

def cmd_frun(name, *args, **kwargs):
    f = _cmd_dict[name]
    sig = signature(f)
    args = [annotate(arg, p) for arg, p in zip(args, sig.parameters.values())]
    kwargs = {k: annotate(v, sig.parameters[k]) for k, v in kwargs.items()}
    return f(*args, **kwargs)

def cmd_run(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args, kwargs = parse_args_as_func(argv)
    cmd_frun(args[0], *args[1:], **kwargs)

class with_null:
    def __enter__(self):
        return None

    def __exit__(self, type, value, traceback):
        return None

# for bach experiments, but combined with argparse and put this into your main.py
def read_args(args_path, timeout=30):
    args_path = Path(args_path)
    lock_dir = Path(args_path.parent, '.lock')
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_fn = Path(lock_dir, args_path.stem)
    lock_fn.touch(exist_ok=True)
    with filelock.FileLock(lock_fn).acquire(timeout=timeout):
        with open(args_path) as f:
            jobs = f.read().splitlines(True)
        while jobs:
            job = jobs[0].strip()
            if not job or job.startswith('#'):
                jobs = jobs[1:]
            else:
                break
        if jobs:
            # skip empty line and comments
            args = shlex.split(jobs[0])
            with open(args_path, 'w') as f:
                f.writelines(jobs[1:])
        else:
            args = None
    return args

def push_args(args_str, args_path, timeout=30):
    args_path = Path(args_path)
    lock_dir = Path(args_path.parent, '.lock')
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_fn = Path(lock_dir, args_path.stem)
    lock_fn.touch(exist_ok=True) # disadvantages: this will not be cleaned up
    with filelock.FileLock(lock_fn).acquire(timeout=timeout):
        with open(args_path) as f:
            jobs = f.read().splitlines(True)
        jobs.insert(0, args_str + '\n')
        with open(args_path, 'w') as f:
            f.writelines(jobs)

def batch_args(exp_path, exp_f, config=None):
    if config is not None and not config.d and is_git_diff():
        print(colored('please commit your changes before running new experiments!', 'red', attrs=['bold']))
        return
    while True:
        args = read_args(exp_path)
        if args is None: break
        args_str = ' '.join(args)
        exp_finished = False
        try:
            print(args)
            config = argparse.Namespace() if config is None else config
            exp_finished = exp_f(args, config) 
        except Exception as e:
            traceback.print_exc() # if traceback is not import, no error will be shown
        finally:
            if not exp_finished:
                push_args(args_str, exp_path)
                break

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if 'torch' in sys.modules:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

# environment might has different random seed
def rollout(env, policy, noises, deterministic=False):
    states = []
    actions = []
    rewards = []
    next_states = []
    terminals = []
    done = False
    s = env.reset()
    cur_step = 0
    if deterministic: policy.set_noise(noises)
    while not done:
        if deterministic:
            a = policy(s)
        else:
            a = policy(s, noises[cur_step])
        next_s, r, done, _ = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(next_s)
        terminals.append(done)
        s = next_s
        cur_step += 1
    return np.asarray(states), np.asarray(actions), np.asarray(rewards), np.asarray(next_states), np.asarray(terminals)
  
# sampler helper function
def sample_init(env, init_seeds):
    global sample_env
    sample_env = env
    seed = init_seeds.get()
    env.seed(seed)

def stochastic_policy_rollout(policy, noises):
    global sample_env
    return rollout(sample_env, policy, noises)

def deterministic_policy_rollout(policy, noises):
    global sample_env
    return rollout(sample_env, policy, noises, deterministic=True)

# initializer take init_queue as input
# This is just for rollout
class MPSampler:
    def __init__(self, env, n_processes=0, deterministic=False):
        if n_processes <= 0: n_processes = mp.cpu_count()
        init_seeds = mp.Queue()
        for seed in np.random.randint(Config.SEED_RANGE, size=n_processes): init_seeds.put(int(seed)) # initseeds
        self.pool = mp.Pool(n_processes, sample_init, (env, init_seeds))
        if deterministic:
            self.rollout_f = deterministic_policy_rollout
        else:
            self.rollout_f = stochastic_policy_rollout
        
    def sample(self, policy, noises): # might cost problems
        return self.pool.starmap_async(self.rollout_f, [(policy, noise) for noise in noises]).get()

    def __del__(self):
        self.pool.close()
        self.pool.join()

class HorizonWrapper(gym.Wrapper):
    def __init__(self, env, horizon):
        super().__init__(env)
        self.horizon = horizon

    def reset(self):
        self.t = 0
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.t += 1
        if self.t == self.horizon: done = True
        return next_state, reward, done, info

class LastWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def __getattribute__(self, attr):
        if attr == 'env':
            return object.__getattribute__(self, attr)
        env = self.env
        while True:
            if hasattr(env, attr):
                return getattr(env, attr)
            if env.unwrapped == env: break
            env = env.env 
        raise Exception('attribute error: {}'.format(attr))

class EnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    @property
    def last(self):
        return LastWrapper(self)

# sort_f takes (env, state, done, data)
class ArrayRQMCSampler:
    def __init__(self, env, n_envs, sort_f):
        envs = [copy.deepcopy(env) for _ in range(n_envs)]
        for env, seed in zip(envs, np.random.randint(Config.SEED_RANGE, size=len(envs))):
            env.seed(int(seed))
        self.envs = envs
        self.n_envs = n_envs
        self.sort_f = sort_f
        
    def sample(self, policy, noises):
        assert noises.shape[0] == self.n_envs and noises.shape[2] == self.envs[0].action_space.shape[0]
        n_trajs, horizon, action_dim = noises.shape
        data = [{'states': [], 'actions': [], 'rewards': []} for _ in range(n_trajs)]
        envs = list(self.envs) # shallow copy, copy the order only
        states = [env.reset() for env in self.envs]
        dones = [False for _ in range(n_trajs)]
        for j in range(horizon):
            if np.all(dones): break
            envs, states, dones, data = zip(*sorted(zip(envs, states, dones, data), key=self.sort_f))
            #lambda env, state, done, data: np.inf if done else env.expected_cost(K, sigma_a, x0=x[1])))
            states, dones, data = list(states), list(dones), list(data)
            for i, env in enumerate(envs):
                if dones[i]: break
                action = policy(states[i], noises[i][j])
                state, r, done, _ = env.step(action)
                states[i] = state
                dones[i] = done
                data[i]['states'].append(state)
                data[i]['actions'].append(action)
                data[i]['rewards'].append(r)
        return data

class SeqRunner:
    def __init__(self, env, deterministic=False):
        self.env = env
        env.seed(int(np.random.randint(Config.SEED_RANGE)))
        self.deterministic = deterministic

    def sample(self, policy, noises):
        return [rollout(self.env, policy, noise, deterministic=self.deterministic) for noise in noises]

def cumulative_return(rewards, discount):
    returns = []
    cur_return = 0.0
    for r in rewards[::-1]:
        cur_return = discount * cur_return + r
        returns.append(cur_return)
    return returns[::-1]

def reinforce_loss(states, actions, rewards, policy):
    log_probs = policy.distribution(states).log_prob(tensor(actions)).sum(-1)
    return log_probs.sum() * tensor(rewards).sum()

def variance_reduced_loss(states, actions, rewards, policy):
    log_probs = policy.distribution(states).log_prob(tensor(actions)).sum(-1)
    returns = rewards[::-1].cumsum()[::-1].copy()
    return (log_probs * tensor(returns)).sum()

def no_loss(states, actions, rewards, policy):
    return tensor(0.0, requires_grad=True)

# this is a tricky function, since it will affect the gradient of the policy
def get_gradient(states, actions, rewards, policy, loss_fn):
    policy.zero_grad()
    loss_fn(states, actions, rewards, policy).backward() 
    return np.array(policy.mean.weight.grad.cpu().numpy())

def running_seeds(save_fn, sample_f, sample_args, num_seeds=200):
    results = []
    sample_args.save_fn = None # overwrite
    for seed in range(num_seeds):
        print('running seed {}/{}'.format(seed, num_seeds))
        sample_args.seed = seed
        result = sample_f(sample_args)
        results.append(result)
    Path(save_fn).parent.mkdir(parents=True, exist_ok=True)
    with open(save_fn, 'wb') as f:
        dill.dump(results, f)

# run until a number of success seed is collected
def collect_seeds(save_fn, sample_f, sample_args, success_f, n_seeds=50, max_seed=200):
    results = []
    sample_args.save_fn = None # overwrite, do not save
    n_success = 0
    for seed in range(max_seed):
        print('running seed {}/{}, collecting seed {}/{}'.format(seed, max_seed, n_success, n_seeds))
        sample_args.seed = seed
        result = sample_f(sample_args)
        if success_f(result):
            print('success seed, appended')
            n_success += 1
        else:
            print('fail seed, discarded')
        results.append(result)
        if n_success == n_seeds: break
    save_fn = Path(save_fn) 
    save_fn.parent.mkdir(parents=True, exist_ok=True) 
    with open(save_fn, 'wb') as f:
        dill.dump(results, f)

# only support LQR
def sort_by_optimal_value(env):
    K = env.optimal_controller()
    sigma_a = np.diag(np.ones(env.M))
    cost_f = env.expected_cost_state_func(K, Sigma_a)
    def f(args):
        env, state, done, data = args
        return np.inf if done else cost_f(state)
    return f

def sort_by_norm(env):
    def f(args):
        env, state, done, data = args
        return np.inf if done else np.linalg.norm(state)
    return f
