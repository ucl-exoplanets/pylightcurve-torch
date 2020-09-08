import warnings
from typing import Any, Optional, Union

import torch
from torch import nn, dtype, device

from .constants import MAX_RATIO_RADII, PLC_ALIASES
from .functional import exoplanet_orbit, transit_duration, transit_flux_drop


class TransitModule(nn.Module):
    _parnames = {'method', 'P', 'i', 'e', 'a', 'rp', 'fp', 't0', 'w', 'ldc'}
    _authorised_parnames = _parnames.union(set(PLC_ALIASES.keys()))
    _methods_dim = {'linear': 1, 'sqrt': 2, 'quad': 2, 'claret': 4}
    _pars_of_fun = {'position': {'P', 'i', 'e', 'a', 't0', 'w'},
                    'duration': {'i', 'rp', 'P', 'a', 'i', 'e', 'w'},
                    'drop_p': {'method', 'ldc', 'rp'},
                    'drop_s': {'fp', 'rp'}}

    def __init__(self, time=None, primary=True, secondary=False, epoch_type=None, precision=3, dtype=torch.float64,
                 cache_pos=True, cache_flux=True, cache_dur=True, **kwargs):
        """ Creates a pytorch transit model class, inheriting torch.nn.Module

        The model computes kepler positions, primary and secondary transits flux_drop drops for N different sets of
         parameters and T time steps

        :param time: array-like time series of time values. Shape: (T,), (1, T), (N, T)
        :param primary: boolean indicating whether to compute the primary transit or not (type: bool ; default: True)
        :param secondary: Whether to compute the secondary transit or not (type: bool ; default: False)
        :param epoch_type: str indicating whether the t0 parameter refers to mid primary or secondary transit times.
            It can take the str values 'primary' or 'secondary'. If the primary param is set to True, it will be
            defaulted to 'primary', and otherwise to 'secondary'.
        :param precision:  (type: int ; default: 3)
        :param dtype: tensors dtype (default: torch.float64) If None provided, the environment default torch dtype will
            be used.
        :param kwargs: additional optional parameters. If given these must be named like transit params
        """
        super().__init__()

        self.primary = bool(primary)
        self.secondary = bool(secondary)
        if not self.primary and not self.secondary:
            raise RuntimeError("transit can't be neither primary nor secondary")

        if epoch_type is None:
            self.epoch_type = ('primary' if self.primary else 'secondary')
        else:
            try:
                self.epoch_type = epoch_type.strip().lower()
                assert self.epoch_type in ('primary', 'secondary')
            except AttributeError:
                raise TypeError("epoch_type must be a str, one of 'primary' and 'secondary' ")
            except AssertionError:
                raise ValueError("epoch_type should be one of: 'primary', 'secondary' ")
        self.precision = precision
        if dtype is None:
            self.dtype = torch.get_default_dtype()
        else:
            self.dtype = dtype
        self.cache_pos = cache_pos
        self.cache_flux = cache_flux
        self.cache_dur = cache_dur

        self.__shape = [None, None]
        self.method = None
        for name in self._parnames:
            if name != 'method':
                self.__setattr__(name, None)
        if kwargs:
            self.set_param(**kwargs)

        self.time = None
        self.time_unit = None
        if time is not None:
            self.set_time(time)

        self.__pos = None
        self.__dur = None
        self.__flux_p = None
        self.__flux_s = None

    def __repr__(self):
        return (f"TransitModule({'primary, ' if self.primary else ''}"
                + f"{'secondary, ' if self.secondary else ''}shape={self.shape})")

    @property
    def _pos_factor(self):
        return float((self.epoch_type == 'primary') - (self.epoch_type == 'secondary'))

    @property
    def shape(self):
        """ returns the shape of the model
        :return: tuple of dimensions (N, T) where N is the batch size and T the number of time steps
        """
        return tuple(self.__shape)

    def get_ldc_dim(self, data=None):
        """ dimensionality of the limb-darkening coefs

        :return: int between 1 and 4 (None if no method defined)
        """
        if self.method is not None:
            return self._methods_dim[self.method]
        elif isinstance(data, torch.Tensor):
            if len(data.shape) > 1:
                dim = data.shape[-1]
                if dim not in list(self._methods_dim.values()):
                    raise RuntimeError("could not retrieve a correct ldc dimensionality given param's shape")
            else:
                warnings.warn("Neither of method nor ldc shape seem to provide a clear ldc dimensionality."
                              "It is advised to provide either method str arg or 2D-shaped ldc inputs")
                dim = 1
            return dim

    ldc_dim = property(get_ldc_dim)

    @property
    def rp_over_rs(self):
        """ Alias for 'rp'"""
        return self.rp

    @property
    def fp_over_fs(self):
        """ Alias for 'fp' """
        return self.fp

    @property
    def inclination(self):
        """ Alias for 'i' """
        return self.i

    @property
    def eccentricity(self):
        """ Alias for 'e' """
        return self.e

    @property
    def periastron(self):
        """ Alias for 'w' """
        return self.w

    @property
    def limb_darkening_coefficients(self):
        """ Alias for 'ldc' """
        return self.ldc

    @property
    def sma_over_rs(self):
        """ Alias for 'a' """
        return self.a

    @property
    def mid_time(self):
        """ Alias for 't0' """
        return self.t0

    @property
    def period(self):
        """ Alias for 'P' """
        return self.P

    def __setattr__(self, key, value):
        if key in PLC_ALIASES.keys():
            key = PLC_ALIASES[key]
        super().__setattr__(key, value)

    def set_time(self, data, time_unit=None):
        """ Sets the tensor of time values

        the input time vector will be converted to a detached tensor

        :param data: array-like time series of time values. Shape: (T,) or (N, T)
        :param time_unit: time unit for record (optional)
        :return: None
        """
        if data is None:
            raise ValueError("data musn't be None. For clearing the data array use clear_time method")
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        if len(data.shape) == 0:
            raise ValueError('data input must be a sized iterable object')
        if len(data.shape) == 1:
            data = data[None, :]
        elif len(data.shape) > 2:
            raise ValueError('data array shape must be one of: (T,), (N, T), (1, T)')
        setattr(self, 'time', nn.Parameter(data.detach().to(self.dtype), requires_grad=False))
        # Updating shape
        self.__shape[1] = self.time.shape[1]
        if self.shape[0] in [None, 1]:
            self.__shape[0] = self.time.shape[0]
        elif self.time.shape[0] > 1 and self.time.shape[0] != self.shape[0]:
            raise RuntimeError("incompatible batch dimensions between data and parameters"
                               + f" (module's ({self.shape[0]}) != data's ({self.time.shape[0]}))")

        self.time_unit = time_unit

    def clear_time(self):
        """ Clears the time tensor

        :return:
        """
        self.time = None
        self.__shape[1] = None
        self.__flux_p = None
        self.__flux_s = None
        self.__pos = None

    def clear_cache(self):
        self.__pos = None
        self.__flux_p = None
        self.__flux_s = None
        self.__dur = None

    def set_param(self, **kwargs):
        """ sets or updates transit parameters values
        Parameters are accessible as class attributes.
        Except method, all params are torch tensor parameters

        :param kwargs: dict of (name, value) of parameters
        :return:
        """
        if not kwargs:
            warnings.warn('no parameter provided')
        for name, value in kwargs.items():
            if name not in self._authorised_parnames:
                raise RuntimeError(f"parameter {name} not in authorized model's list")

            if name == "method":
                self.set_method(value)
                continue
            data = self.prepare_value(name, value)

            if getattr(self, name) is None:
                self.__setattr__(name, nn.Parameter(data))
            else:
                getattr(self, name).data = data
            self.fit_param(name)

    def prepare_value(self, name, value, update_shape=True):
        if name == "method":
            self.check_method(value)
            return value

        # Conversion to Tensor
        if isinstance(value, nn.Parameter):
            data = value.data.to(self.dtype)
        elif isinstance(value, torch.Tensor):
            data = value.to(self.dtype)
        else:
            data = torch.tensor(value, dtype=self.dtype)

        # Dimensionality
        if name == 'ldc':
            dim = self.get_ldc_dim(data)
        else:
            dim = 1

        data = data.view(-1, dim)
        if update_shape:
            if self.shape[0] in [None, 1]:
                self.__shape[0] = data.shape[0]
            elif data.shape[0] > 1 and data.shape[0] != self.shape[0]:
                raise RuntimeError("incompatible batch dimensions between parameters"
                                   + f" (module's ({self.shape[0]}) != {name}'s ({data.shape[0]}))")
        return data

    def fit_param(self, *args):
        for name in args:
            if name not in self._authorised_parnames:
                raise RuntimeError(f"parameter {name} not in authorized model's list")
            param = getattr(self, name)
            if param is None:
                warnings.warn("param is None, its grad can't be activated")
            else:
                param.requires_grad = True
                if name in self._pars_of_fun['duration']:
                    self.__dur = None
                    if self.cache_dur:
                        warnings.warn('duration caching deactivated because of its inclusion in the DCG')
                    self.cache_dur = False
                if name in self._pars_of_fun['position']:
                    if self.cache_pos:
                        warnings.warn('position caching deactivated because of its inclusion in the DCG')
                    self.__pos = None
                    self.cache_pos = False
                if name in self._pars_of_fun['drop_p'].union(self._pars_of_fun['drop_s']):
                    self.__flux_p = None
                    self.__flux_s = None
                    if self.cache_flux:
                        warnings.warn('flux drop caching deactivated because of its inclusion in the DCG')
                    self.cache_flux = False

    def freeze_param(self, *args):
        for name in args:
            if name not in self._authorised_parnames:
                raise RuntimeError(f"parameter {name} not in authorized model's list")
            param = getattr(self, name)
            if param is None:
                warnings.warn("param is None, its grad can't be activated")
            else:
                param.requires_grad = False

    def clear_param(self, name):
        """ Resets a param to None value

        :param name: parameter name (str)
        :return:
        """
        if name not in self._authorised_parnames:
            raise RuntimeError(f"parameter {name} not in authorized model's list")
        setattr(self, name, None)

        if self.cache_flux:
            if name in self._pars_of_fun['drop_p']:
                self.__flux_p = None
            if name in self._pars_of_fun['drop_s']:
                self.__flux_s = None
        if self.cache_dur and name in self._pars_of_fun['duration']:
            self.__dur = None
        if self.cache_pos and name in self._pars_of_fun['position']:
            self.__pos = None

    def clear_params(self, *args):
        """ Resets several parameters to None value

        :param args: list of parameters names. If None is provided, all the parameters will be reset
        :return:
        """
        if not args:
            args = self._parnames
            self.__shape[0] = None
        for name in args:
            self.clear_param(name)

    def check_method(self, value):
        """ Checks the limb-darkening method

        :param value: one of [None, 'linear', 'sqrt', 'quad', 'claret']
        :return:
        """
        if not (value is None or value in self._methods_dim):
            raise ValueError(f'if stated limb darkening method must be in {tuple(self._methods_dim.keys())}')
        if self.ldc is not None and self.ldc.shape[-1] != self._methods_dim[value]:
            self.set_param(ldc=None)
            warnings.warn('ldc method incompatible with ldc tensor dimension. ldc coefs have been reset.')

    def set_method(self, value):
        """ Sets the limb-darkening method

        :param value: one of [None, 'linear', 'sqrt', 'quad', 'claret']
        :return:
        """
        self.check_method(value)
        setattr(self, 'method', value)

    def get_input_params(self, function='none', prepare_args=True, allow_none=False, **kwargs):
        """ Returns a dict of model parameters

        :param function: which function to provide the params of - 'none', 'position', 'duration', 'drop_p', 'drop_s'
        :param prepare_args: whether to prepare or not the external arguments to correct type and shape
        :param allow_none: when False, will raise an error if a param is missing
        :param kwargs: parameter names to be given. if None is povided
        :return:
        """
        parlist = self._pars_of_fun[function]
        batch_size = 1
        out = dict()
        if function in ['dop_p', 'drop_s', 'none']:
            batch_size = self.time.shape[0]
        for k in parlist:
            if k in kwargs:
                v = kwargs[k]
                if prepare_args:
                    v = self.prepare_value(k, v, update_shape=False)
            else:
                v = getattr(self, k)
            if not allow_none and v is None:
                raise RuntimeError(f"Parameter '{k}' should not be missing")
            if k != 'method' and v.shape[0] > 1:
                if batch_size == 1:
                    batch_size = v.shape[0]
                    # print("batch_size just updated")
                else:
                    assert batch_size == v.shape[0]
            out[k] = v
        return out, batch_size

    def get_position(self, **kwargs):
        """ Computes the 3D cartesian positions of the planet following a Keplerian orbit
        if any model parameter is provided while calling this method, no caching will take place
        :param kwargs:
        :return: tuple of position arrays x, y, z, each of shape (N, T)
        # """
        if self.cache_pos and self.__pos is not None and not {k for k in kwargs if k in self._pars_of_fun['position']}:
            return self.__pos
        if self.time is None:
            raise RuntimeError('time attribute needs to be defined')
        d, batch_size = self.get_input_params(**kwargs, function='position')
        x, y, z = exoplanet_orbit(d['P'], d['a'], d['e'], d['i'], d['w'], d['t0'], self.time,
                                  ww=torch.zeros(1, 1, dtype=self.dtype), n_pars=batch_size, dtype=self.dtype)

        out = x * self._pos_factor, y * self._pos_factor, z * self._pos_factor
        if self.cache_pos:
            self.__pos = out
        return out

    position = property(get_position)

    def get_proj_dist(self, restrict_orbit=None, **kwargs):
        x, y, z = self.get_position(**kwargs)
        proj_dist = torch.sqrt(y ** 2 + z ** 2)
        if restrict_orbit is None:
            return proj_dist
        elif restrict_orbit == 'primary':
            return torch.where(x < 0., torch.ones_like(x, dtype=self.dtype) * MAX_RATIO_RADII, proj_dist)
        elif restrict_orbit == 'secondary':
            return torch.where(x > 0., torch.ones_like(x, dtype=self.dtype) * MAX_RATIO_RADII, proj_dist)

    proj_dist = property(get_proj_dist)

    def get_duration(self, **kwargs):
        """ Returns the cached or computed duration of transit(s)
        if any model parameter is provided while calling this method, no caching will take place
        :return:
        """
        ext_provided = {k for k in kwargs if k in self._pars_of_fun['duration']}
        if self.cache_dur and self.__dur is not None and not ext_provided:
            return self.__dur
        d, batch_size = self.get_input_params(**kwargs, function='duration')
        out = transit_duration(d['rp'], d['P'], d['a'], d['i'], d['e'], d['w'])
        if self.cache_dur:
            self.__dur = out
        return out

    duration = property(get_duration)

    def get_flux_p(self, **kwargs):
        """ Returns the cached or computed flux_drop drop of transit(s), relative to the star

        :param precision: precision of computation (int between 1 and 6)
        :param kwargs: additional functions argument to replace the class' one, disabling caching where necessary
        :return: (N, T)-shaped array of flux_drop drop values
        """
        ext_provided = bool([k for k in kwargs if k in self._pars_of_fun['drop_p']])
        if self.cache_flux and self.__flux_p is not None and not ext_provided:
            return self.__flux_p
        proj_dist = self.get_proj_dist(**kwargs, restrict_orbit='primary')
        d, batch_size = self.get_input_params(**kwargs, function='drop_p')
        batch_size = max(batch_size, proj_dist.shape[0])
        out = transit_flux_drop(d['method'], d['ldc'], d['rp'], proj_dist,
                                precision=self.precision, n_pars=batch_size)
        if self.cache_flux:
            self.__flux_p = out
        return out

    flux_p = property(get_flux_p)

    def get_flux_s(self, **kwargs):
        """ Returns the cached or computed flux_drop drop of eclipse(s), relative to the star

        :param precision: precision of computation (int between 1 and 6)
        :param kwargs: additional functions argument to replace the class' one, disabling caching where necessary
        :return: (N, T)-shaped array of flux drop values
        """
        ext_provided = bool([k for k in kwargs if k in self._pars_of_fun['drop_s']])
        if self.cache_flux and self.__flux_s is not None and not ext_provided:
            return self.__flux_s
        d, batch_size = self.get_input_params(**kwargs, function='drop_s')
        proj_dist = self.get_proj_dist(**kwargs, restrict_orbit='secondary') / d['rp']
        batch_size = max(batch_size, proj_dist.shape[0])
        out = transit_flux_drop('linear', torch.zeros(batch_size, 1), 1 / d['rp'], proj_dist,
                                precision=self.precision, n_pars=batch_size)
        out = (1. + d['fp'] * out) / (1. + d['fp'])
        if self.cache_flux:
            self.__flux_s = out
        return out

    flux_s = property(get_flux_s)

    def get_flux_drop(self, **kwargs):
        """ Returns the combined flux drop of primary/secondary transits (if activated) relative to the star

        :param kwargs: transit parameters to substitute to the model's.

        Be wary, no shape modification implemented for these additional arguments
        :return: (N, T)-shaped array of flux drop values
        """
        out = 1.
        if self.primary:
            out *= self.get_flux_p(**kwargs)
        if self.secondary:
            out *= self.get_flux_s(**kwargs)
        return out

    flux_drop = property(get_flux_drop)

    def forward(self, **kwargs: Any):
        """ Alias for get_flux_drop function, overriding nn.Module.forward method

        :return:
        """
        return self.get_flux_drop(**kwargs)

