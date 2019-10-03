import time
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
from multiprocessing.pool import ThreadPool

from rqmc_distributions import dist_rqmc

class Config:
    DEVICE = torch.device('cpu')
    SEED_RANGE = 100000000
    DEBUG = False # for debug usage

# put your debug statement inside this, when the debug flag is disabled, the code will report error
@contextmanager
def debug(desc=None):
    assert Config.DEBUG
    try:
        yield None
    finally:
        pass

def cprint(msg, color, attrs=[]):
    print(colored(msg, color, attrs=attrs))

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
            assert self.default_logger is not None, 'no default logger, please specify the logger when using logger.log'
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

def np_check_numerics(*args):
    return all([np.all(np.isfinite(x)) for x in args])

def cosine_similarity(a, b):
    return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)

# environment might has different random seed
def rollout(env, policy, noises):
    states = []
    actions = []
    rewards = []
    next_states = []
    terminals = []
    done = False
    s = env.reset()
    cur_step = 0
    while not done:
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

class HorizonWrapper(gym.Wrapper):
    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps = max_steps

    def reset(self):
        self.t = 0
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.t += 1
        if self.t == self.max_steps: done = True
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

def _one_step(env, action):
    return env.step(action)

# sort_f takes (env, state, done, data)
class ArrayRQMCSampler:
    def __init__(self, env, n_envs, sort_f):
        envs = [copy.deepcopy(env) for _ in range(n_envs)]
        for env, seed in zip(envs, np.random.randint(Config.SEED_RANGE, size=len(envs))):
            env.seed(int(seed))
        self.envs = envs
        self.n_envs = n_envs
        self.sort_f = sort_f
        #self.pool = mp.Pool(8)
        #self.pool = ThreadPool(8)
        
    def sample(self, policy, noises):
        assert noises.shape[0] == self.n_envs and noises.shape[2] == self.envs[0].action_space.shape[0]
        n_trajs, horizon, action_dim = noises.shape
        data = [{'states': [], 'actions': [], 'rewards': []} for _ in range(n_trajs)]
        envs = list(self.envs) # shallow copy, copy the order only
        states = [env.reset() for env in self.envs]
        dones = [False for _ in range(n_trajs)]
        for j in range(horizon):
            if np.all(dones): break
            pairs = list(zip(envs, states, dones, data))
            pairs_to_sort = [p for p in pairs if not p[2]]
            pairs_done = [p for p in pairs if p[2]]
            #n_valid = len(pairs_to_sort)
            envs, states, dones, data = zip(*( self.sort_f(pairs_to_sort) + pairs_done ))
            states, dones, data = list(states), list(dones), list(data)

            ''' # a multithreading version, turn out to be slower
            actions = policy(states[:n_valid], noises[:n_valid, j])
            valid_states, valid_rewards, valid_dones, _ = zip(*self.pool.starmap(_one_step, zip(envs[:n_valid], actions)))
            for i in range(n_valid):
                data[i]['states'].append(states[i])
                data[i]['actions'].append(actions[i])
                data[i]['rewards'].append(valid_rewards[i])
            states[:n_valid] = valid_states
            dones[:n_valid] = valid_dones
            '''

            actions = policy(states, noises[:, j])
            for i, env in enumerate(envs):
                if dones[i]: break
                state, r, done, _ = env.step(actions[i])
                data[i]['states'].append(states[i])
                data[i]['actions'].append(actions[i])
                data[i]['rewards'].append(r)
                states[i] = state
                dones[i] = done
        return data

class VecSampler(ArrayRQMCSampler):
    def __init__(self, env, n_envs):
        super().__init__(env, n_envs, no_sort)

# sampler helper function
def mp_sampler_init(env, init_seeds):
    global sample_env
    sample_env = env
    seed = init_seeds.get()
    env.seed(seed)

def stochastic_policy_rollout(policy, noises):
    global sample_env
    return rollout(sample_env, policy, noises)

# initializer take init_queue as input
# This is just for rollout
class MPSampler:
    def __init__(self, env, n_processes=0):
        if n_processes <= 0: n_processes = mp.cpu_count()
        init_seeds = mp.Queue()
        for seed in np.random.randint(Config.SEED_RANGE, size=n_processes): init_seeds.put(int(seed)) # initseeds
        self.pool = mp.Pool(n_processes, mp_sampler_init, (env, init_seeds))
        self.rollout_f = stochastic_policy_rollout
        
    def sample(self, policy, noises): # might cost problems
        return self.pool.starmap_async(self.rollout_f, [(policy, noise) for noise in noises]).get()

    def __del__(self):
        self.pool.close()
        self.pool.join()

class SeqSampler:
    def __init__(self, env):
        self.env = env
        env.seed(int(np.random.randint(Config.SEED_RANGE)))

    def sample(self, policy, noises):
        return [rollout(self.env, policy, noise) for noise in noises]

def cumulative_return(rewards, discount):
    returns = []
    cur_return = 0.0
    for r in rewards[::-1]:
        cur_return = discount * cur_return + r
        returns.append(cur_return)
    return returns[::-1]

def reinforce_loss(states, actions, qs, policy):
    log_probs = policy.distribution(states).log_prob(tensor(actions)).sum(-1)
    return -(log_probs * tensor(qs)).mean()

def variance_reduced_loss(states, actions, qs, policy):
    log_probs = policy.distribution(states).log_prob(tensor(actions)).sum(-1)
    assert np.all(np.isfinite(qs)), 'invalid return'
    assert not torch.isnan(log_probs).any(), 'invalid logprobs'
    return -(log_probs * tensor(qs)).mean()

def no_loss(states, actions, rewards, policy):
    return tensor(0.0, requires_grad=True)

def lqr_gt_loss(env):
    # it only works for Gaussian Policy with linear layer, no bias, fix_std, does not gate output
    def f(states, actions, qs, policy):
        K = policy._mean.weight
        grad = env.expected_policy_gradient(K.detach().cpu().numpy(), torch.diag(policy.std).detach().cpu().numpy())
        return -torch.trace(torch.matmul(tensor(grad.T), K)) / env.max_steps # just to make the scale same as REINFORCE loss
    return f

# this is a tricky function, since it will affect the gradient of the policy, and only works for Gaussian policy
def get_gaussian_policy_gradient(states, actions, rewards, policy, loss_fn):
    policy.zero_grad()
    loss_fn(states, actions, rewards, policy).backward() 
    return np.array(policy._mean.weight.grad.cpu().numpy())

def running_seeds(save_fn, sample_f, sample_args, num_seeds=200, post_f=None):
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
    if post_f is not None:
        post_f(results)

# run until a number of success seed is collected
def collect_seeds(save_fn, sample_f, sample_args, success_f, n_seeds=50, max_seed=200, post_f=None):
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
    if post_f is not None:
        post_f(results)

# only works for LQR
def sort_by_policy_value(env, K):
    Sigma_a = np.diag(np.ones(env.M))
    cost_f = env.expected_cost_state_func(K, Sigma_a)
    def f(args):
        env, state, done, data = args
        return np.inf if done else cost_f(state)
    return f

# only support LQR
def sort_by_optimal_value(env):
    K = env.optimal_controller()
    return sort_by_policy_value(env, K)

def sort_by_norm(env):
    def f(args):
        env, state, done, data = args
        return np.inf if done else np.linalg.norm(state)
    return f

# pair: env, state, done, data
def multdim_sort(pairs, dim=0):
    if len(pairs) == 1: return pairs
    if dim == pairs[0][1].shape[0] - 1:
        return sorted(pairs, key=lambda p: p[1][dim])
    else:
        mid = len(pairs) // 2 
        return multdim_sort(sorted(pairs[:mid], key=lambda p: p[1][dim]), dim+1) + multdim_sort(sorted(pairs[mid:], key=lambda p: p[1][dim]), dim+1)

def random_permute(pairs):
    pairs = list(pairs) # shallow copy
    random.shuffle(pairs)
    return pairs

def no_sort(pairs):
    return pairs
