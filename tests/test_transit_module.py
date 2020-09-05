import numpy as np
import torch

from pylightcurve_torch.nn import TransitModule

pars = {'method': "linear", 'rp': 0.0241, 'fp': 0.00001, 'P': 7.8440, 'a': 5.4069, 'e': 0.3485, 'i': 91.8170,
        'w': 77.9203, 't0': 5.1814, 'ldc': 0.1}

map_dict = lambda d, f: {k: (f(v) if k!= 'method' else v) for k, v in d.items()}
params_dicts = {'scalar': pars,
                'with_int': map_dict(pars, int),
                'np_scalar': map_dict(pars, np.array),
                'torch_scalar': map_dict(pars, torch.tensor),
                'np_array_1': map_dict(pars, lambda x: np.array(x)[None]),
                'np_array_2': map_dict(pars, lambda x: np.array(x)[None, None]),
                'np_array_3': map_dict(pars, lambda x: np.array(x)[None, None, None]),
                'tensor_1': map_dict(pars, lambda x: torch.tensor(x)[None]),
                'tensor_2': map_dict(pars, lambda x: torch.tensor(x)[None, None]),
                'tensor_3': map_dict(pars, lambda x: torch.tensor(x)[None, None, None]),
                'plc': {'method': "linear", 'rp_over_rs': 0.0241, 'period': 7.8440, 'fp_over_fs': 0.00001,
                        'sma_over_rs': 5.4069, 'eccentricity': 0.3485, 'inclination': 91.8170,
                        'periastron': 77.9203, 'mid_time': 5.1814, 'limb_darkening_coefficients': 0.5}
                }

time_array = np.linspace(0, 10, 100)
time_tensor = torch.tensor(time_array)


def test_transit_type():
    assert TransitModule().primary == True
    assert TransitModule().secondary == False
    assert TransitModule(primary=False, secondary=True).primary == False
    assert TransitModule(secondary=True).primary == True
    assert TransitModule(secondary=True).secondary == True

    try:
        TransitModule(primary=False, secondary=False)
    except RuntimeError:
        pass  # rightly caught error

    # epoch_type
    for tm in [TransitModule(),
               TransitModule(primary=True),
               TransitModule(secondary=True),
               TransitModule(primary=False, secondary=True, epoch_type='primary')
               ]:
        assert tm.epoch_type == 'primary'

    for tm in [TransitModule(primary=False, secondary=True),
               TransitModule(epoch_type='secondary')]:
        assert tm.epoch_type == 'secondary'


def test_pytorch_inherited_attr():
    tm = TransitModule()
    tm.set_param(e=0.)

    assert tm.float().e.dtype == torch.float32
    assert tm.double().e.dtype == torch.float64

    assert tm.to('cpu').e.device.type == 'cpu'
    if torch.cuda.is_available():
        assert tm.to('cuda').e.device.type in ['gpu', 'cuda']

    tm.train()
    tm.eval()
    tm.named_modules()
    tm.named_parameters()
    assert 'e' in tm._parameters
    tm.zero_grad()


def test_transit_params():
    for k, d in params_dicts.items():
        tm = TransitModule(**d)
        tm.set_time(time_array)

        for p in tm._parnames:
            assert p == 'method' or getattr(tm, p).data.dtype == torch.float64
        assert tm.time.dtype == torch.float64

        tm.clear_params()
        attr = np.random.choice(list(tm._parnames))
        assert getattr(tm, attr) is None
        tm.set_param(**d)

        tm.position
        for i, x in enumerate([tm.proj_dist,  tm.flux_drop, tm.forward(), tm()]):
            assert isinstance(x, torch.Tensor)
            assert not torch.isnan(x).any()

        flux_1 = tm()
        # External arguments
        flux_2 = tm(**d)
        assert torch.isclose(flux_1, flux_2).all()

def test_time_tensor():
    tm = TransitModule(**params_dicts['scalar'])
    tm.set_time(torch.linspace(0, 10, 100))
    tm()

    tm.set_time(torch.linspace(0, 10, 100)[None, :].repeat(5, 1))
    tm()

    tm = TransitModule(**map_dict(pars, lambda x: torch.tensor(x)[None, None].repeat(5, 1)))
    tm.set_time(torch.linspace(0, 10, 100)[None, :].repeat(5, 1))
    tm()

    tm = TransitModule(**map_dict(pars, lambda x: torch.tensor(x)[None, None].repeat(5, 1)))
    try:
        tm.set_time(torch.linspace(0, 10, 100)[None, :].repeat(6, 1))
    except RuntimeError:
        ...  # Caught error


def test_gradients():
    tm = TransitModule(time=time_array, **params_dicts['scalar'], secondary=True)
    for param in tm._parameters:
        tm.zero_grad()
        tm.fit_param(param)
        tm().sum().backward()
        g = getattr(tm, param).grad
        assert not torch.isnan(g) and g.item() != 0.
        tm.freeze_param(param)
