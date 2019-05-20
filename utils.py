import sys
import numpy as np
import random
import torch
import shlex
import argparse
import inspect
import filelock
import traceback
import subprocess
import multiprocessing as mp
from inspect import signature
from pathlib import Path
from termcolor import colored

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

def set_seed(t, r=None, p=None, c=None):
    if r is None:
        r = t
    if p is None:
        p = r
    torch.manual_seed(t)
    random.seed(r)
    np.random.seed(p)
    if c is not None:
      torch.cuda.manual_seed(c)

# environment might has different random seed
def rollout(env, K, noises):
    states = []
    actions = []
    rewards = []
    done = False
    s = env.reset()
    cur_step = 0
    while not done:
        a = K.dot(s) + noises[cur_step]
        next_s, r, done, _ = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = next_s
        cur_step += 1
    return np.asarray(states), np.asarray(actions), np.asarray(rewards)
  
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

def _rollout(K, noises):
    global sample_env
    return rollout(sample_env, K, noises)

# initializer take init_queue as input
# This is just for rollout
class Sampler:
    def __init__(self, env, n_processes=0):
        if n_processes <= 0: n_processes = mp.cpu_count()
        init_seeds = mp.Queue()
        for seed in np.random.randint(100000000, size=n_processes): init_seeds.put(seed) # initseeds
        self.pool = mp.Pool(n_processes, sample_init, (env, init_seeds))
        
    def sample(self, K, noises):
        return self.pool.starmap_async(_rollout, [(K, noise) for noise in noises]).get()

    def __del__(self):
        self.pool.close()
        self.pool.join()

def cumulative_return(rewards, discount):
    returns = []
    cur_return = 0.0
    for r in rewards[::-1]:
        cur_return = discount * cur_return + r
        returns.append(cur_return)
    return returns[::-1]
