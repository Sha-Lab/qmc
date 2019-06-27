import random
from itertools import product
from utils import cmd, cmd_run
from pathlib import Path
from ipdb import launch_ipdb_on_exception

def generate_cmd(args=None, kwargs=None):
    cmds = ''
    if args:
        cmds += ' '.join(map(process_arg, args))
    cmds += ' ' 
    if kwargs:
        cmds += ' '.join(['{} {}'.format(k, process_arg(v)) for k, v in kwargs.items()])
    return cmds

def dump_args(f, args=None, kwargs=None):
    f.write(generate_cmd(args, kwargs) + '\n')

def process_arg(arg):
    if isinstance(arg, (list, tuple)):
        return ' '.join([str(v) for v in arg])
    return arg

def generate_args(exp_path, args, kwargs, variants, post_variant=None, touch=True, shuffle=False):
    keys, values = zip(*variants.items())
    variants = [dict(zip(keys, v)) for v in product(*values)]
    if shuffle: random.shuffle(variants)
    if touch: open(exp_path, 'w').close()
    with open(exp_path, 'a+') as f:
        for variant in variants:
            if post_variant: variant = post_variant(variant)
            dump_args(f, args, {**kwargs, **variant})

@cmd()
def search_learn(touch: int=1, shuffle: int=0):
    variants = {
        '--n_trajs': [80, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000],
        '-lr': [0.00001, 0.0001, 0.0003, 0.0006],
    }
    args = []
    kwargs = {
        '--env': 'WIP', # IP
        '-H': 8, # 6
        '--init_scale': 0.1,
        '--task': 'learn',
        '--n_seeds': 50,
        '--n_iters': 1000,
        '--mode': 'collect',
    }
    def post_variant(variant):
        variant['--save_fn'] = 'data/search_learn/{}-{}-{}'.format(kwargs['--env'], variant['--n_trajs'], variant['-lr'])
        return variant
    generate_args('exps/search_learn_{}'.format(kwargs['--env']), args, kwargs, variants, post_variant=post_variant, shuffle=shuffle)

@cmd()
def search_network_std(touch: int=1, shuffle: int=0):
    variants = {
        '--n_trajs': [60, 100, 150, 200, 300],
        '-lr': [0.0001, 0.0005, 0.001],
        '-H': [5, 10, 15],
        '--init_scale': [1.0, 3.0, 5.0],
    }
    args = []
    kwargs = {
        '--task': 'learn',
        '--n_iters': 300,
        '--n_seeds': 10,
        '--max_seed': 30,
        '--mode': 'collect',
        '--init_policy': 'mlp',
        '--n_workers': 8,
    }
    def post_variant(variant):
        variant['--save_fn'] = 'data/search_network_std/{}-{}-{}-{}'.format(*[variant[k] for k in ['--n_trajs', '-lr', '-H', '--init_scale']])
        return variant
    generate_args('exps/search_network_std', args, kwargs, variants, post_variant=post_variant, shuffle=shuffle)

@cmd()
def search_vpg():
    variants = {
        '--n_trajs': [500, 1000, 1500, 2000],
        '--hidden_sizes': [(8,), (16,), (16, 8), (16, 16), (32, 16), (32 ,32)],
    }
    args = []
    kwargs = {'--n_workers': 250}
    def post_variant(variant):
        variant['--save_fn'] = 'data/search_vpg/{}-{}'.format(variant['--n_trajs'], '-'.join([str(x) for x in variant['--hidden_sizes']]))
        return variant
    generate_args('exps/search_vpg', args, kwargs, variants, post_variant=post_variant, shuffle=True)

@cmd()
def search_vpg_trajs():
    variants = {
        '--n_trajs': [50, 60, 70, 80, 90, 100, 150, 200],
    }
    args = []
    kwargs = {
        '--n_iters': 8000,
        '--n_workers': 50,
        '--hidden_sizes': (32, 32),
        '--mode': 'seeds',
        '--n_seeds': 8,
    }
    def post_variant(variant):
        variant['--save_fn'] = 'data/search_vpg_trajs_32/{}'.format(variant['--n_trajs'])
        return variant
    generate_args('exps/search_vpg_trajs_32', args, kwargs, variants, post_variant=post_variant, shuffle=False)

if __name__ == "__main__":
    with launch_ipdb_on_exception():
        cmd_run()
