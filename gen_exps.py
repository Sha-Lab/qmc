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

def generate_args(exp_path, args, kwargs, variants, post_variant=None, touch=True, shuffle=False):
    keys, values = zip(*variants.items())
    variants = [dict(zip(keys, v)) for v in product(*values)]
    if shuffle: random.shuffle(variants)
    if touch: open(exp_path, 'w').close()
    with open(exp_path, 'a+') as f:
        for variant in variants:
            if post_variant: variant = post_variant(variant)
            dump_args(f, args, {**kwargs, **variant})

# TODO: combine change in sa!!!
def generate_args(exp_path, args, kwargs, variants, post_variants=None, touch=True, shuffle=False):
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
        '-xu_dim': [(20, 12), ()], 
        '-H': [5, 7, 10, 13],
        '--noise': [0.0, 0.1, 0.3, 0.5],
        '--n_trajs': [5, 10, 20, 40, 60, 80, 100, 200, 300, 500],
        '-lr': [0.00001, 0.00005, 0.0001, 0.0005],
    }
    args = []
    kwargs = {
        '--task': 'learn',
        '--n_seeds': 50,
        '--n_iters': 1000,
        '--mode': 'collect',
    }
    def post_variant(variant):
        variant['--save_fn'] = 'data/search_learn/{}-{}-{}-{}'.format(variant['-H'], variant['--noise'], variant['--n_trajs'], variant['-lr']) 
        variant['--save_fig'] = 'data/search_learn/{}-{}-{}-{}'.format(variant['-H'], variant['--noise'], variant['--n_trajs'], variant['-lr'])
        return variant
    generate_args('exps/search_learn', args, kwargs, variants, shuffle=shuffle)

if __name__ == "__main__":
    cmd_run()
