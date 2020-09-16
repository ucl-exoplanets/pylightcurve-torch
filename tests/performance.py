import argparse
import os
import timeit
import warnings

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import torch

import pylightcurve_torch
from pylightcurve_torch import TransitModule

if hasattr(pylightcurve_torch, "__version__"):
    from pylightcurve_torch import __version__ as version
else:
    version = 'no-version'

pars = {'e': 0.01, 'i': 90., 'w': 0., 'rp': 0.05, 'method': 'linear', 'ldc': [0.1],
        'P': 4., 't0': 5., 'a': 5., 'fp': 0.0001}

map_dict = lambda d, f: {k: (f(v) if k != 'method' else v) for k, v in d.items()}
get_pars = lambda n: map_dict(pars, lambda x: np.array(x)[None, None].repeat(n))


def compute_perf(n_vect=None, p_vect=None, dur_vect=None, plot=True, save=True, device=None):
    n_cpus = int(os.cpu_count())
    n_gpus = int(torch.cuda.device_count() * device == 'cuda')

    # Batch size
    if n_vect is None:
        n_vect = [1, 10, 20, 40]
    if len(n_vect):
        n_vect = np.array(n_vect, dtype=int)
        T = 200
        colnames = ['version'] + [str(n) for n in n_vect] + ['number', 'time_length', 'cpus', 'gpus']
        fname = os.path.join('tests', 'batch_size_perf.csv')
        if os.path.exists(fname):
            df = pd.read_csv(fname, header=0, index_col=0,
                             dtype={'version': str, 'number': int, 'time_length': int, 'cpus': int, 'gpus': int})
            df = df.rename(columns=lambda x: x.strip())
            if version in df.version.values:
                warnings.warn(f'Version label {version} already exists.')
        else:
            df = pd.DataFrame(columns=colnames)
        df = df.append({'version': version}, ignore_index=True)

        number = 200
        times = []
        tm = TransitModule(torch.linspace(5, 7, T)).to(device)
        for n in n_vect:
            tm.reset_params()
            tm.set_params(**get_pars(n))
            tm = tm.to(device)
            times += [timeit.timeit(tm, number=number) / number]
        df.iloc[-1, range(1, 1 + len(n_vect))] = times
        df.loc[df.index[-1], 'cpus'] = int(n_cpus)
        df.loc[df.index[-1], 'gpus'] = int(n_gpus)
        df.loc[df.index[-1], 'number'] = int(number)
        df.loc[df.index[-1], ['time_length']] = int(T)
        if save:
            df.to_csv(fname)
        if plot:
            plt.figure()
            for i in range(len(df)):
                if i < len(df) - 1:
                    kwargs = {'alpha': 0.7, 'linewidth': 0.7}
                else:
                    kwargs = {'color': 'red', 'linewidth': 1.5}

                p = plt.plot(n_vect, df.iloc[i, range(1, 1 + len(n_vect))],
                             label=str(df.version.iloc[i]) + '-' + str(df.index[i]) + f'({device})',
                             **kwargs)
                plt.scatter(n_vect, df.iloc[i, range(1, 1 + len(n_vect))], s=7, c=p[0].get_color())
            plt.xlabel('batch size ')
            plt.ylabel('exec time [s]')
            plt.legend()
            if save:
                plt.savefig(os.path.join('tests', 'batch_size_perf.png'))
            else:
                plt.show()

    # Transit duration
    if p_vect is None:
        p_vect = np.arange(2, 20, 7)
    if len(p_vect):
        p_vect = np.array(p_vect, dtype=int)
        number = 50
        batch_size = 32
        colnames = ['version'] + [str(P) for P in p_vect] + ["batch_size", "number", 'cpus', 'gpus']
        fname = os.path.join('tests', 'transit_dur_perf.csv')
        if os.path.exists(fname):
            df = pd.read_csv(fname, header=0, index_col=0,
                             dtype={'version': str, 'cpus': int, 'gpus': int, 'batch_size': int, 'number': int})
            df = df.rename(columns=lambda x: x.strip())
            if version in df.version.values:
                warnings.warn(f'Version label {version} already exists.')
        else:
            df = pd.DataFrame(columns=colnames)
        df = df.append({'version': version}, ignore_index=True)

        times = []
        durations = []

        tm = TransitModule(torch.linspace(4, 6, 1000)).to(device)
        for P in p_vect:
            tm.reset_params()
            tm.set_params(**get_pars(batch_size))
            tm.set_param('P', P)
            tm = tm.to(device)
            durations += [tm.duration[0]]
            times += [timeit.timeit(lambda: tm(),
                                    number=number) / number]

        df.iloc[df.index[-1], range(1, 1 + len(p_vect))] = times
        df.loc[df.index[-1], 'cpus'] = int(n_cpus)
        df.loc[df.index[-1], 'gpus'] = int(n_gpus)
        df.loc[df.index[-1], 'number'] = int(number)
        df.loc[df.index[-1], 'batch_size'] = int(batch_size)
        if save:
            df.to_csv(fname)

        if plot:
            plt.figure()
            for i in range(len(df)):
                if i < len(df) - 1:
                    kwargs = {'alpha': 0.7, 'linewidth': 0.7}
                else:
                    kwargs = {'color': 'red', 'linewidth': 1.5}

                p = plt.plot(p_vect, df.iloc[i, range(1, 1 + len(p_vect))],
                             label=str(df.version.iloc[i]) + '-' + str(df.index[i]) + f'({device})',
                             **kwargs)
                plt.scatter(p_vect, df.iloc[i, range(1, 1 + len(p_vect))], s=7, c=p[0].get_color())
            plt.xlabel('Transit duration [d]')
            # plt.yscale('log')
            plt.ylabel('exec time [s]')
            plt.legend()
            if save:
                plt.savefig(os.path.join('tests', 'transit_dur_perf.png'))
            else:
                plt.show()

    # Time vector length

    if dur_vect is None:
        dur_vect = np.int_(10 ** torch.arange(6))
    if len(dur_vect):
        dur_vect = np.array(dur_vect, dtype=int)
        number = 10
        batch_size = 4
        colnames = ['version'] + [str(int(T)) for T in dur_vect] + ["batch_size", "number", 'cpus', 'gpus']

        fname = os.path.join('tests', 'time_length_perf.csv')
        if os.path.exists(fname):
            df = pd.read_csv(fname, header=0, index_col=0,
                             dtype={'version': str, 'cpus': int, 'gpus': int, 'batch_size': int, 'number': int})
            df = df.rename(columns=lambda x: x.strip())
            if version in df.version.values:
                warnings.warn(f'Version label {version} already exists.')
        else:
            df = pd.DataFrame(columns=colnames)
        df = df.append({'version': version}, ignore_index=True)
        times = []
        tm = TransitModule()
        for T in dur_vect:
            tm.reset_time()
            tm.set_time(torch.linspace(4, 6, T))
            tm.reset_params()
            tm.set_params(**get_pars(batch_size))
            tm = tm.to(device)
            times += [timeit.timeit(lambda: tm(),
                                    number=number) / number]
        df.iloc[-1, range(1, 1 + len(dur_vect))] = times
        df.loc[df.index[-1], 'cpus'] = int(n_cpus)
        df.loc[df.index[-1], 'gpus'] = int(n_gpus)
        df.loc[df.index[-1], 'number'] = int(number)
        df.loc[df.index[-1], 'batch_size'] = int(batch_size)
        if save:
            df.to_csv(fname)

        if plot:
            plt.figure()
            for i in range(len(df)):
                if i < len(df) - 1:
                    kwargs = {'alpha': 0.7, 'linewidth': 0.7}
                else:
                    kwargs = {'color': 'red', 'linewidth': 1.5}

                p = plt.plot(dur_vect, df.iloc[i, range(1, 1 + len(dur_vect))],
                             label=str(df.version.iloc[i]) + '-' + str(df.index[i]) + f'({device})',
                             **kwargs)
                plt.scatter(dur_vect, df.iloc[i, range(1, 1 + len(dur_vect))], s=7, c=p[0].get_color())
            plt.legend()
            plt.yscale('log')
            plt.xscale('log')
            plt.ylabel('exec time [s]')
            plt.xlabel('Time vector length [time steps]')
            if save:
                plt.savefig(os.path.join('tests', 'time_length_perf.png'))
            else:
                plt.show()


def process_perf():
    parser = argparse.ArgumentParser(description='Executing performance with optional saving and plotting')
    parser.add_argument('-p', '--plot', action='store_true', help="whether to prepare figures or not")
    parser.add_argument('-s', '--save', action='store_true', help="whether to save data and plots")
    parser.add_argument('-g', '--gpu', action='store_true', help="whether to use GPU if available")
    args = parser.parse_args()
    device = 'cpu'
    if args.gpu:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            warnings.warn('GPU asked but not available')

    print(f'testing performance on device {device}')
    compute_perf(plot=args.plot, save=args.save, device=device)


if __name__ == '__main__':
    process_perf()
