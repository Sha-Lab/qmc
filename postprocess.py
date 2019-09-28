import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import logger



def plot_cost(args, results):
    mc_data = pd.concat([
        pd.DataFrame(
            {'name': 'mc',
             'error': res[0],
             'x': np.arange(len(res[0]))}) for res in results
    ])

    rqmc_data = pd.concat([
        pd.DataFrame(
            {'name': 'rqmc',
             'error': res[1],
             'x': np.arange(len(res[1]))}) for res in results
    ])

    arqmc_data = pd.concat([
        pd.concat([
            pd.DataFrame(
                {'name': name,
                 'error': error,
                 'x': np.arange(len(error))}) 
            for name, error in res[2].items()
        ])
        for res in results
    ])

    data = pd.concat([mc_data, rqmc_data, arqmc_data])
    plot = sns.relplot(x='x', y='error', kind='line', hue='name', data=data)
    plot.set(yscale='log')
    plt.savefig(args.save_fig)

def plot_learn(args, full_results):
    mc_discard_threshold = 3
    Path(args.save_fig).parent.mkdir(parents=True, exist_ok=True)
    logger.info('ploting {}'.format(args.save_fig))
    results = [res for res, info in full_results if len(info['out']) == 0]
    if len(results) < mc_discard_threshold:
        results = [res for res, info in full_results if len(info['out']) == 1]
    if len(results) == 0: return
    data = pd.concat([
        pd.concat([
            pd.DataFrame({
                'name': name,
                'cost': -val,
                'x': np.arange(len(val)),
            })
            for name, val in res.items()
        ])
        for res in results
    ])
    plot = sns.relplot(x='x', y='cost', kind='line', hue='name', data=data)
    plt.savefig(args.save_fig)
