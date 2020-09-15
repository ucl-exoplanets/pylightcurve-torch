import numpy as np
import pytest
import torch

import pylightcurve_torch as pt
from pylightcurve_torch import TransitModule

parnames = {'transit': ('rp_over_rs', 'period', 'sma_over_rs', 'eccentricity', 'inclination', 'periastron',
                        'mid_time', 'limb_darkening_coefficients', 'method'),
            'eclipse': (
                'rp_over_rs', 'fp_over_fs', 'period', 'sma_over_rs', 'eccentricity', 'inclination', 'periastron',
                'mid_time')
            }

time_array = np.linspace(3, 7, 1000)
time_tensor = torch.linspace(3, 7, 1000)[None, :]


def sample_pars(function):
    pars_scalars = {"rp_over_rs": np.random.rand() * (1 - 1.e-5) + 1e-5,
                    "fp_over_fs": np.random.rand() / 100,
                    "period": 1 + np.random.rand() * 10,
                    "sma_over_rs": 2 + np.random.rand() * 5,
                    "eccentricity": np.random.rand() / 2,
                    "inclination": 90 + (np.random.rand() - 0.5) * 2,
                    "periastron": np.random.rand() * 180,
                    "mid_time": 4.95 + np.random.rand() / 10,
                    "method": str(np.random.choice(['linear', 'sqrt', 'quad', 'claret']))
                    }
    if function == "eclipse":
        pars_scalars['mid_time'] += pars_scalars['period'] / 2
        pars_scalars['eccentricity'] / 5

    out = pars_scalars
    if function == 'transit':
        pars_ldc = {'linear': np.random.rand(1),
                    'sqrt': np.random.rand(2),
                    'quad': np.random.rand(2),
                    'claret': np.random.rand(4)}
        out.update({'limb_darkening_coefficients': pars_ldc[pars_scalars['method']]})
    out_np = {k: v for k, v in out.items() if k in parnames[function]}
    out_torch = {k: v if k == "method" else torch.tensor(v, dtype=torch.float64).view(1, -1) for k, v in out_np.items()}
    return out_np, out_torch


def test_plc_compat():
    try:
        import pylightcurve as plc
        ...
    except ImportError:
        pytest.skip("pylightcurve package can't be found."
                    + "Skipping the comparison tests with original pylightcurve functions.")
    """Tests the precision and compatibility with original pylightcurve transit function"""

    for i in range(10):
        pars_np, pars_torch = sample_pars('transit')
        flux_plc_np = plc.transit(time_array=time_array, **pars_np)
        flux_plc_torch = pt.functional.transit(time_array=time_tensor, **pars_torch, n_pars=1)
        tm = TransitModule(time_array, primary=True, **pars_torch)
        assert 1 - flux_plc_np.min() > (pars_np['rp_over_rs'] ** 2) / 2
        assert np.allclose(flux_plc_np, tm(), atol=1e-6, rtol=1e-2)
        assert np.allclose(flux_plc_torch, flux_plc_np, atol=1e-6, rtol=1e-2)


def test_eclipse_compat():
    try:
        import pylightcurve as plc
    except ImportError:
        pytest.skip("pylightcurve package can't be found."
                    + "Skipping the comparison tests with original pylightcurve functions.")
    """Tests the precision and compatibility with original pylightcurve eclipse function"""

    for i in range(10):
        pars_np, pars_torch = sample_pars('eclipse')
        flux_plc_np = plc.eclipse(time_array=time_array, **pars_np)
        flux_plc_torch = pt.functional.eclipse(time_array=time_tensor, **pars_torch, n_pars=1)
        tm = TransitModule(time_array, secondary=True, primary=False, epoch_type='primary', **pars_torch)
        try:
            assert 1 - flux_plc_np.min() > (pars_np['fp_over_fs'] ** 2) / 2
        except:
            print('eclipse probably just out of time array...')
            continue
        passed = True
        try:
            assert np.allclose(flux_plc_np, tm(), atol=1e-6, rtol=1e-2)
        except AssertionError:
            mae = np.abs(flux_plc_np - tm().numpy()).max()
            raise RuntimeError(f'Eclipse precision unacceptable (MaxAE={mae})')
        try:
            assert np.allclose(flux_plc_torch, flux_plc_np, atol=1e-6, rtol=1e-2)
        except AssertionError:
            mae = np.abs(flux_plc_np - flux_plc_torch.numpy()).max()
            import matplotlib.pylab as plt
            plt.plot(time_array, flux_plc_np)
            plt.plot(time_array, flux_plc_torch[0])
            plt.show()
            raise RuntimeError(f'Eclipse precision unacceptable (MaxAE={mae})')
    if not passed:
        raise ValueError('first step never passed')
