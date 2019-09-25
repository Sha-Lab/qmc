from ipdb import launch_ipdb_on_exception
from exps import cmd, cmd_run, generate_args


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
def search_network_std_tanh(touch: int=1, shuffle: int=0):
    variants = {
        '--n_trajs': [60, 100, 150, 200, 300],
        '-lr': [0.0005, 0.001],
        '-H': [10, 15],
        '--init_scale': [3.0, 5.0],
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
        variant['--save_fn'] = 'data/search_network_std_tanh/{}-{}-{}-{}'.format(*[variant[k] for k in ['--n_trajs', '-lr', '-H', '--init_scale']])
        return variant
    generate_args('exps/search_network_std_tanh', args, kwargs, variants, post_variant=post_variant, shuffle=shuffle)


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

@cmd()
def search_vpg_ant_trajs():
    variants = {
        '--n_trajs': [80, 100, 150],
    }
    args = []
    kwargs = {
        '--env': 'ant',
        '--n_iters': 7000,
        '--n_workers': 4,
        '--hidden_sizes': (64, 64),
        '--mode': 'seeds',
        '--n_seeds': 5,
    }
    def post_variant(variant):
        variant['--save_fn'] = 'data/search_vpg_ant_trajs/{}'.format(variant['--n_trajs'])
        return variant
    generate_args('exps/search_vpg_ant_trajs', args, kwargs, variants, post_variant=post_variant, shuffle=False)

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
def search_arqmc():
    args = ['--no_swap']
    kwargs = {
        '--task': 'lqr',
        '--n_runs': 20,
    }
    variants = {
        '--n_trajs': [2 ** i for i in (5, 7, 9)],
        '--horizon': [10, 20, 40, 60, 80, 100, 100],
    }
    def post_option(toggle, variant):
        variant['--exp_name'] = 'search_arqmc/{}-{}'.format(variant['--n_trajs'], variant['--horizon'])
        return toggle, variant
    generate_args('exps/search_arqmc', args, kwargs, toggles, variants, post_option=post_option, shuffle=False)

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
    }
    toggles = []
    variants = {
        '--n_trajs': [64, 128, 256],
    }
    def post_option(toggle, variant):
        variant['--save_fn'] = 'log/compare_on_cartpole/traj_{}'.format(variant['--n_trajs'])
        return toggle, variant
    generate_args('exps/compare_on_cartpole', args, kwargs, toggles, variants, post_option=post_option, shuffle=False)


if __name__ == "__main__":
    with launch_ipdb_on_exception():
        cmd_run()
