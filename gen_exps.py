import random
from itertools import product
from utils import cmd, cmd_run
from pathlib import Path

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
    }
    args = []
    kwargs = {
        '--env': 'WIP', # IP
        '-lr': 0.00005,
        '-H': 8, # 6
        '--init_scale': 0.1,
        '--task': 'learn',
        '--n_seeds': 50,
        '--n_iters': 1000,
        '--mode': 'collect',
    }
    def post_variant(variant):
        variant['--save_fn'] = 'data/search_learn/{}-{}'.format(kwargs['--env'], variant['--n_trajs'])
        return variant
    generate_args('exps/search_learn_{}'.format(kwargs['--env']), args, kwargs, variants, post_variant=post_variant, shuffle=shuffle)


if __name__ == "__main__":
    cmd_run()
