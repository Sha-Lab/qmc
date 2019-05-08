import random
from itertools import product
from utils import cmd, cmd_run
from pathlib import Path

def generate_cmd(args=None, kwargs=None):
    cmds = ''
    if args:
        cmds += ' '.join(args)
    cmds += ' ' 
    if kwargs:
        cmds += ' '.join(['{} {}'.format(k, v) for k, v in kwargs.items()])
    return cmds

def dump_args(f, args=None, kwargs=None):
    f.write(generate_cmd(args, kwargs) + '\n')

def generate_args(exp_path, args, kwargs, variants, touch=True, shuffle=False):
    keys, values = zip(*variants.items())
    variants = [dict(zip(keys, v)) for v in product(*values)]
    if shuffle: random.shuffle(variants)
    if touch: open(exp_path, 'w').close()
    with open(exp_path, 'a+') as f:
        for variant in variants:
            dump_args(f, args, {**kwargs, **variant})

@cmd()
def search_learn(touch: int=1, shuffle: int=0):
    variants = {
        '-H': [5, 7, 10, 13, 15, 17, 20],
        '--noise': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
        '--n_trajs': [50, 80, 100, 120, 140, 160, 180, 200, 300, 400],
        '-lr': [0.0005, 0.001, 0.003, 0.006, 0.01],
    }
    args = ['--over_seed']
    kwargs = {
        '--task': 'learn',
        '--n_seeds': 50,
        '--n_iters': 1000,
    }
    generate_args('exps/search_learn', args, kwargs, variants, shuffle=shuffle)

if __name__ == "__main__":
    cmd_run()
