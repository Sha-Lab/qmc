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
  
def mse(a, b):
    return ((a - b) ** 2).mean()

def cummean(x, axis=0):
    return np.cumsum(x, axis=axis) / np.cumsum(np.ones_like(x), axis=axis)

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

# use state as the value to sort
class SortableWrapper(EnvWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        self.val = self.env.reset()
        return self.val

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.val = next_state
        return next_state, reward, done, info

class MonitorWrapper(EnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.traj = None
        self.monitor = False

    def start_monitor(self, trajs=None):
        assert not self.monitor, 'already start'
        self.trajs = trajs
        self.monitor = True

    def close_monitor(self):
        assert self.monitor, 'already close'
        self.traj = None
        self.monitor = False

    def reset(self):
        if self.monitor:
            if self.traj is not None and self.trajs is not None:
                self.trajs.append(self.traj)
            self.traj = dict(states=[], actions=[], rewards=[])
        self.cur_state = self.env.reset()
        return self.cur_state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if self.monitor:
            self.traj['states'].append(self.cur_state)
            self.traj['actions'].append(action)
            self.traj['rewards'].append(reward)
        self.cur_state = next_state
        return next_state, reward, done, info

    def get_traj(self):
        return self.traj

class VecEnv:
    def __init__(self, envs):
        self.envs = envs
        self.n_envs = len(self.envs)

    def step(self, actions):
        data = []
        for i in range(self.n_envs):
            obs, rew, done, info = self.envs[i].step(actions[i])
            if done:
                obs = self.envs[i].reset()
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        return obs, np.asarray(rew), np.asarray(done), info

    def reset(self):
        return [env.reset() for env in self.envs]

class VecSampler:
    def __init__(self, env, n_envs=1):
        envs = [copy.deepcopy(env) for _ in range(n_envs)]
        for env, seed in zip(envs, np.random.randint(Config.SEED_RANGE, size=len(envs))):
            env.seed(int(seed))
        self.env = VecEnv(envs)
        self.n_envs = n_envs
        
    def sample(self, policy, noises): # might cost problems
        data = []
        cur_rollout = [dict(states=[], actions=[], rewards=[]) for _ in range(self.n_envs)]
        noise_indices = list([(i if i < len(noises) else -1) for i in range(self.n_envs)])
        idx = self.n_envs
        states = self.env.reset() # compare to seqrunner, call one more reset in each sample
        cur_t = [0 for _ in range(self.n_envs)]
        while True:
            actions = policy(states, noises[noise_indices, cur_t])
            next_states, rewards, terminals, infos = self.env.step(actions)
            for i, step in enumerate(zip(states, actions, rewards, terminals)):
                state, action, reward, terminal = step
                cur_rollout[i]['states'].append(state)
                cur_rollout[i]['actions'].append(action)
                cur_rollout[i]['rewards'].append(reward)
                cur_t[i] += 1
                if terminal:
                    data.append([cur_rollout[i][k] for k in ['states', 'actions', 'rewards']])
                    cur_rollout[i] = dict(states=[], actions=[], rewards=[])
                    noise_indices[i] = idx if idx < len(noises) else -1
                    cur_t[i] = 0
                    idx += 1
                if len(data) == len(noises): break
            if len(data) == len(noises): break
            states = next_states
        return data

def sort_by_norm(envs):
    return sorted(envs, key=lambda env: np.linalg.norm(env.last.val))

def sort_by_val(envs):
    return sorted(envs, key=lambda env: env.last.val)

def sort_by_optimal_value(envs):
    K = envs[0].last.optimal_controller()
    Sigma_a = np.eye(envs[0].last.M)
    return sorted(envs, key=lambda env: env.last.expected_cost(K, Sigma_a, x0=env.last.val, T=env.last.max_steps-env.last.num_steps))

def sort_envs(envs, sort_f, stopped):
    return sort_f([env for env, done in zip(envs, stopped) if not done]) + \
        [env for env, done in zip(envs, stopped) if done]

def scramble_points(points):
    return (points + np.random.randn(*points.shape)) % 1.0

# sort f should take pair as input!
# I have to say using monitor is very ugly...
class ArrayRQMCSampler:
    def __init__(self, env, n_envs, sort_f=None):
        envs = [copy.deepcopy(MonitorWrapper(env)) for _ in range(n_envs)]
        for env, seed in zip(envs, np.random.randint(Config.SEED_RANGE, size=len(envs))):
            env.seed(int(seed))
        self.env = VecEnv(envs)
        self.n_envs = n_envs
        self.sort_f = sort_f
        # sobol does not need to sort the first dimension
        self.points = dist_rqmc.Uniform_RQMC(env.action_space.shape[0]).sample(n_envs) 
        
    def sample(self, policy, times):
        data = []
        for _ in range(times):
            rollouts = []
            for env in self.env.envs: env.start_monitor(rollouts)
            states = self.env.reset()
            terminals = [False for _ in range(self.n_envs)]
            stopped = np.zeros(len(states), dtype=np.bool) # this should also be sorted...
            while not np.all(stopped):
                self.env.envs = sort_envs(self.env.envs, self.sort_f, stopped)
                noises = norm.ppf(scramble_points(self.points))
                actions = policy(states, noises)
                next_states, rewards, terminals, _ = self.env.step(actions)
                for i, terminal in enumerate(terminals):
                    if stopped[i]: continue
                    if terminal:
                        self.env.envs[i].close_monitor()
                        stopped[i] = True
                states = next_states
            data.extend([[rollout[k] for k in ['states', 'actions', 'rewards']] for rollout in rollouts])
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

def PGPE_loss(rewards, policy, noises, critic=None):
    returns = rewards.sum(-1)
    if critic is not None: returns = returns - critic.avg
    noises = policy.reshape_n(tensor(noises))
    log_probs = policy.distribution().log_prob((policy.mean + noises * policy.std).detach()).sum(-1)
    return (log_probs * tensor(rewards).sum(-1)).sum()

def no_loss(states, actions, rewards, policy):
    return tensor(0.0, requires_grad=True)

def critic_loss(states, returns, critic):
    return F.mse_loss(critic(tensor(states)).squeeze(1), tensor(returns))

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

### for arqmc
class MCSampler:
    def __init__(self, env, policy):
