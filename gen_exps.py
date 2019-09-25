from ipdb import launch_ipdb_on_exception
from exps import cmd, cmd_run, generate_args


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

@cmd()
def search_PGPE():
    variants = {
        '--n_trajs': [100, 300, 500, 700, 1000, 1500],
        '--xu_dim': [(20, 12), (10, 8), (5, 5)],
        '--init_scale': [1.0, 3.0, 5.0],
        '-H': [5, 10, 15],
    }
    args = []
    kwargs = {
        '--task': 'learn_PGPE',
        '--mode': 'collect',
        '--n_iters': 2000,
    }
    def post_variant(variant):
        variant['--save_fn'] = 'data/search_PGPE/{}_{}-{}-{}'.format(variant['--xu_dim'][0], variant['--xu_dim'][1], variant['--init_scale'], variant['-H'])
        return variant
    generate_args('exps/search_PGPE', args, kwargs, variants, post_variant=post_variant, shuffle=False)

@cmd()
def search_arqmc_linear(touch: int=1, shuffle: int=0):
    variants = {
        '--n_trajs': [64, 128, 256, 512],
        '-lr': [0.0001, 0.0005, 0.001],
        '-H': [5, 10, 20, 40],
        '--init_scale': [1.0, 3.0, 5.0],
    }
    args = []
    kwargs = {
        '--task': 'learn',
        '--n_iters': 300,
        '--n_seeds': 10,
        '--max_seed': 30,
        '--mode': 'collect',
        '--init_policy': 'linear',
        '--n_workers': 8,
    }
    def post_variant(variant):
        variant['--save_fn'] = 'data/search_arqmc_linear/traj_{}-lr_{}-H_{}-init_{}'.format(*[variant[k] for k in ['--n_trajs', '-lr', '-H', '--init_scale']])
        return variant
    generate_args('exps/search_arqmc_linear', args, kwargs, variants, post_variant=post_variant, shuffle=shuffle)

@cmd()
def compare_arqmc_sorter_on_cost():
    name = 'compare_arqmc_sorter_on_cost'
    args = []
    kwargs = {
        '--task': 'cost',
        '--mode': 'seeds',
        '--sorter': 'value norm none permute group',
        '--n_seeds': 100,
        '--n_trajs': 8192,
    }
    toggles = []
    variants = {
        '--xu_dim': [(20, 12), (10, 8), (5, 5)],
        '-H': [5, 10, 20, 40],
        '--init_scale': [1.0, 3.0, 5.0],
    }
    def post_option(toggle, variant):
        variant['--save_fn'] = 'log/{}/{}_{}-{}-{}'.format(name, variant['--xu_dim'][0], variant['--xu_dim'][1], variant['-H'], variant['--init_scale'])
        return toggle, variant
    generate_args('exps/{}'.format(name), args, kwargs, toggles, variants, post_option=post_option, shuffle=False)

@cmd()
def compare_on_cartpole():
    args = []
    kwargs = {
        '--env': 'cartpole',
        '--n_iters': 500,
        '--sorter': 'norm',
        '--n_workers': 8,
        '--hidden_sizes': (32, 32),
        '--mode': 'seeds',
        '--n_seeds': 3,
        '-H': 100,
    }
    toggles = []
    variants = {
        '--n_trajs': [64, 128, 256],
    }
    def post_option(toggle, variant):
        variant['--save_fn'] = 'log/compare_on_cartpole/traj_{}'.format(variant['--n_trajs'])
        return toggle, variant
    generate_args('exps/compare_on_cartpole', args, kwargs, toggles, variants, post_option=post_option, shuffle=False)

@cmd()
def compare_on_swimmer():
    exp_name = 'compare_on_swimmer'
    args = []
    kwargs = {
        '--env': 'swimmer',
        '--n_iters': 2000,
        '--sorter': 'group',
        '--n_workers': 8,
        '--hidden_sizes': (32, 32),
        '--mode': 'seeds',
        '--n_seeds': 5,
        '-H': 100,
    }
    toggles = []
    variants = {
        '--n_trajs': [64, 128, 256],
    }
    def post_option(toggle, variant):
        variant['--save_fn'] = 'log/{}/traj_{}'.format(exp_name, variant['--n_trajs'])
        return toggle, variant
    generate_args('exps/{}'.format(exp_name), args, kwargs, toggles, variants, post_option=post_option, shuffle=False)


if __name__ == "__main__":
    with launch_ipdb_on_exception():
        cmd_run()
