import warnings

import torch
from torch import nn

from ._constants import MAX_RATIO_RADII, PLC_ALIASES
from .functional import exoplanet_orbit, transit_duration, transit_flux_drop


class TransitModule(nn.Module):
    _parnames = ['method', 'P', 'i', 'e', 'a', 'rp', 'fp', 't0', 'w', 'ldc', 'time']
    _authorised_parnames = _parnames + list(PLC_ALIASES.keys())
    _methods_dim = {'linear': 1, 'sqrt': 2, 'quad': 2, 'claret': 4}
    _pars_of_fun = {'position': {'time', 'P', 'i', 'e', 'a', 't0', 'w'},
                    'duration': {'i', 'rp', 'P', 'a', 'i', 'e', 'w'},
                    'drop_p': {'method', 'ldc', 'rp'},
                    'drop_s': {'fp', 'rp'}}

    def __init__(self, time=None, primary=True, secondary=False, epoch_type=None, precision=3, dtype=torch.float64,
                 cache_pos=False, cache_flux=False, cache_dur=False, **kwargs):
        """Create a pytorch transit module instance

        The model computes kepler positions, primary and secondary transits flux_drop drops for N different sets of
         parameters and T time steps.

         Paameters:
        -------
        :param time: array-like time series of time values. Shape: (T,), (1, T), (N, T)
        :param primary: boolean indicating whether to compute the primary transit or not (type: bool ; default: True)
        :param secondary: Whether to compute the secondary transit or not (type: bool ; default: False)
        :param epoch_type: str indicating whether the t0 parameter refers to mid primary or secondary transit times.
            It can take the str values 'primary' or 'secondary'. If the primary param is set to True, it will be
            defaulted to 'primary', and otherwise to 'secondary'.
        :param precision: numerical precision between 1 and 6 (type: int ; default: 3)
        :param dtype: tensors dtype (default: torch.float64) If None provided, the environment default torch dtype will
            be used.
        :param cache_pos: whether or not to save the position tensors for efficiency. Note that this requires it to be
            outside of the Dynamic Computational Graph, and will be set to False with a warning as soon as a dependable
            parameter has its gradient activated.
        :param cache_flux: whether or not to save the flux decrements tensors for efficiency. Note that this requires it
        to be outside of the Dynamic Computational Graph, and will be set to False with a warning as soon as a dependent
            parameter has its gradient activated.
        :param cache_dur: whether or not to save the duration tensors for efficiency. Note that this requires it to be
            outside of the Dynamic Computational Graph, and will be set to False with a warning as soon as a dependable
            parameter has its gradient activated.
        :param kwargs: additional optional parameters. If given these must be authorised transit parameters, as listed
        below:
            #Transit parameters
            -------
            rp: ratio of planetary by stellar radii  - unitless - alias: rp_ove_rs
            fp: ratio of planetary by stellar fluxes - unitless - alias: fp_over_rs (specific to secondary transit)
            a: ratio of semi-major axis by the stellar radius - unitless - alias: sma_over_rs
            P: orbital period - days - alias: period
            e: orbital eccentricity - unitless - alias eccentricity
            i: orbital inclination - degrees - alias: inclination
            p: orbital argument of periastron - degrees - alias: periastron
            t0: transit mid-time - days - alias: mid_time
            method: str indicating the limb-darkening law (available methods: 'claret', 'quad', 'sqrt' or 'linear')
                (specific to primary transit)
            ldc: A list containing the limb darkening coefficients. The list should contain 1 element if the method used
                is the 'linear', 2 if the method used is the 'quad' or teh 'sqrt', and 4 if the method used is the
                'claret'. (specific to primaru transit) - alias: limb_darkening_coefficients

        """
        super().__init__()

        self.primary = bool(primary)
        self.secondary = bool(secondary)
        if not self.primary and not self.secondary:
            raise RuntimeError(
                "transit can't be neither primary nor secondary")

        if epoch_type is None:
            self.epoch_type = ('primary' if self.primary else 'secondary')
        else:
            try:
                self.epoch_type = epoch_type.strip().lower()
                assert self.epoch_type in ('primary', 'secondary')
            except AttributeError:
                raise TypeError(
                    "epoch_type must be a str, one of 'primary' and 'secondary' ")
            except AssertionError:
                raise ValueError(
                    "epoch_type should be one of: 'primary', 'secondary' ")
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
            self.set_params(**kwargs)

        self.time = None
        self.time_unit = None
        if time is not None:
            self.set_time(time)

        self.__pos = None
        self.__dur = None
        self.__drop_p = None
        self.__drop_s = None

    def __setattr__(self, key, value):
        if key in PLC_ALIASES.keys():
            key = PLC_ALIASES[key]
        super().__setattr__(key, value)

    def __repr__(self):
        """Class str representation"""
        return (f"TransitModule({'primary=True, ' if self.primary else ''}"
                + f"{'secondary=True, ' if self.secondary else ''}shape={self.shape})")

    def forward(self, **kwargs):
        """Return the combined flux drop of primary/secondary transits relative to the star

        Overrides torch.nn.Module.forward method, itself aliased by __call__

        :param kwargs: transit parameters to substitute to the model's.
        :return: (N, T)-shaped tensor of flux drop values
        """
        out = 1.
        if self.primary:
            out *= self.get_drop_p(**kwargs)
        if self.secondary:
            out *= self.get_drop_s(**kwargs)
        return out

    def set_param(self, name, value):
        """Set or updates a transit parameter by name and value

        Parameter will be defined as a class Attribute, most specifically a nn.Parameter except for 'method' which
        is just a str.
        :param name: nme of the param (or alias)
        :param value: value of the param
        :return:
        """
        if name not in self._authorised_parnames:
            raise RuntimeError(
                f"parameter {name} not in authorized model's list")

        if name == "method":
            self.set_method(value)
            return
        data = self._prepare_value(name, value)

        if getattr(self, name) is None:
            self.__setattr__(name, nn.Parameter(data, requires_grad=False))
        else:
            getattr(self, name).data = data
        # update cache
        if self.cache_pos or self.cache_dur or self.cache_pos:
            self.reset_cache(name)
        # update gradient
        if data.requires_grad:
            self.activate_grad(name)

    def set_params(self, **kwargs):
        """Set or updates transit parameters values as key/value pairs

        Parameters are accessible as class attributes.
        Except method, all params are torch tensor parameters

        :param kwargs: dict of (name, value) of parameters
        :return:
        """
        if not kwargs:
            warnings.warn('no parameter provided')
        for name, value in kwargs.items():
            self.set_param(name, value)

    def set_time(self, data, time_unit=None):
        """Set the tensor of time values

        the input time vector will be converted to a detached tensor

        :param data: array-like time series of time values. Shape: (T,) or (N, T)
        :param time_unit: time unit for record (optional)
        :return: None
        """
        if data is None:
            raise ValueError(
                "time data shouldn't be None. For resetting it use reset_time method")
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        if len(data.shape) == 0:
            raise ValueError('data input must be a sized iterable object')
        if len(data.shape) == 1:
            data = data[None, :]
        elif len(data.shape) > 2:
            raise ValueError(
                'time input shape must be one of: (T,), (N, T), (1, T)')
        setattr(self, 'time', nn.Parameter(
            data.detach().to(self.dtype), requires_grad=False))
        # Updating shape
        self.__shape[1] = self.time.shape[1]
        if self.shape[0] in [None, 1]:
            self.__shape[0] = self.time.shape[0]
        elif self.time.shape[0] > 1 and self.time.shape[0] != self.shape[0]:
            raise RuntimeError("incompatible batch dimensions between data and parameters"
                               + f" (module's ({self.shape[0]}) != data's ({self.time.shape[0]}))")
        self.time_unit = time_unit

    def set_method(self, value):
        """Set the limb-darkening method

        :param value: one of [None, 'linear', 'sqrt', 'quad', 'claret']
        :return:
        """
        self._check_method(value)
        setattr(self, 'method', value)

    def activate_grad(self, *args):
        """Activate the gradient for each of the parameters provided

        This will also deactivate and reset the cache for the dependent tensors, with a possible warning if applicable.

        :param args: list of paranamer names
        :return:
        """
        for name in args:
            if name not in self._authorised_parnames:
                raise RuntimeError(
                    f"parameter {name} not in authorized model's list")
            param = getattr(self, name)
            if param is None:
                warnings.warn("param is None, its grad can't be activated")
            else:
                param.requires_grad = True
                self.reset_cache(name, deactivate=True)

    def deactivate_grad(self, *args):
        """Deactivate the gradient for each of the parameters provided

        :param args: list of parameters names
        :return:
        """
        for name in args:
            if name not in self._authorised_parnames:
                raise RuntimeError(
                    f"parameter {name} not in authorized model's list")
            param = getattr(self, name)
            if param is None:
                warnings.warn("param is None, its grad can't be activated")
            else:
                param.requires_grad = False

    def reset_param(self, name):
        """Reset a param to None value

        :param name: parameter name (str)
        :return:
        """
        if name not in self._authorised_parnames:
            raise RuntimeError(
                f"parameter {name} not in authorized model's list")
        setattr(self, name, None)
        self.reset_cache(name)

    def reset_params(self, *args):
        """Reset several parameters to None value

        :param args: list of parameters names. If None is provided, all the parameters will be reset
        :return:
        """
        if not args:
            args = self._parnames
            self.__shape[0] = None
        for name in args:
            self.reset_param(name)

    def reset_time(self):
        """Reset the time tensor

        :return:
        """
        self.time = None
        self.__shape[1] = None
        self.__drop_p = None
        self.__drop_s = None
        self.__pos = None

    def reset_cache(self, name=None, deactivate=False):
        """Reset the appropriate cached tensors.

        Cached tensors are reset to None value, selectively if a parameter name is given.
        :param name: parameter name from which to selectively infer the dependent cached tensors. If no name provided
            ('default'), all the cached tensors will be reset (to None).
        :return:
        """
        if name is None:
            self.__pos = None
            self.__drop_p = None
            self.__drop_s = None
            self.__dur = None
            return
        if name in self._pars_of_fun['duration']:
            self.__dur = None
            if deactivate and self.cache_dur:
                warnings.warn(
                    'duration caching deactivated because of its inclusion in the DCG')
                self.cache_dur = False
        if name in self._pars_of_fun['position']:
            self.__pos = None
            if deactivate and self.cache_pos:
                warnings.warn(
                    'position caching deactivated because of its inclusion in the DCG')
                self.cache_pos = False
        if name in self._pars_of_fun['drop_p'].union(self._pars_of_fun['drop_s']):
            self.__drop_p = None
            self.__drop_s = None
            if deactivate and self.cache_flux:
                warnings.warn(
                    'flux drop caching deactivated because of its inclusion in the DCG')
                self.cache_flux = False

    def get_ldc_dim(self, data=None):
        """Return the dimensionality of the limb-darkening coefs

        Consistency with the method attribute is important and should be checked beforehand whenever possible.
        :data: optional tensor from which to infer the dimensionality
        :return: int between 1 and 4 (None if no method defined)
        """
        if self.method is not None:
            return self._methods_dim[self.method]
        elif isinstance(data, torch.Tensor):
            if len(data.shape) > 1:
                dim = data.shape[-1]
                if dim not in list(self._methods_dim.values()):
                    raise RuntimeError(
                        "could not retrieve a correct ldc dimensionality given param's shape")
            else:
                warnings.warn("Neither of method nor ldc shape seem to provide a clear ldc dimensionality."
                              "It is advised to provide either method str arg or 2D-shaped ldc inputs")
                dim = 1
            return dim

    def get_input_params(self, function='none', prepare_args=True, allow_none=False, **kwargs):
        """Return a dict of model parameters

        :param function: which function to provide the params of - 'none', 'position', 'duration', 'drop_p', 'drop_s'
        :param prepare_args: whether to prepare or not the external arguments to correct type and shape
        :param allow_none: when False, will raise an error if a param is missing
        :param kwargs: parameter names to be given. if None is povided
        :return: dict of parameters
        """
        parlist = self._pars_of_fun[function]
        batch_size = 1
        out = dict()
        ext_args = dict()
        # External arguments aliases
        for k in kwargs:
            if k in PLC_ALIASES:
                parname = PLC_ALIASES[k]
            else:
                parname = k
            if not parname in self._parnames:
                raise RuntimeError(
                    f"parameter {parname} not in authorized model's list")
            ext_args[parname] = kwargs[k]
        for k in parlist:
            if k in ext_args:
                v = ext_args[k]
                if prepare_args:
                    v = self._prepare_value(k, v, update_shape=False)
            else:
                v = getattr(self, k)
            if not allow_none and v is None:
                raise RuntimeError(f"Parameter '{k}' should not be missing")
            if k != 'method' and v.shape[0] > 1:
                if batch_size == 1:
                    batch_size = v.shape[0]
                else:
                    assert batch_size == v.shape[0]
            out[k] = v
        return out, batch_size

    def get_position(self, **kwargs):
        """Compute the cached or computed 3D cartesian positions of the planet following a Keplerian orbit.

        if any model parameter is provided while calling this method, no caching will take place

        :param kwargs: optional external parameters to be provided as key/values pairs and to replace module's pars
        :return: tuple of position tensors x, y, z, each of shape (N, T)
        # """
        runtime_mode = bool(
            {k for k in kwargs if k in self._pars_of_fun['position']})
        if self.cache_pos and self.__pos is not None and not runtime_mode:
            return self.__pos
        d, batch_size = self.get_input_params(**kwargs, function='position')
        x, y, z = exoplanet_orbit(d['P'], d['a'], d['e'], d['i'], d['w'], d['t0'], d['time'],
                                  ww=d['time'].new_zeros(1, 1, dtype=self.dtype), n_pars=batch_size, dtype=self.dtype)

        out = x * self._pos_factor(), y * self._pos_factor(), z * self._pos_factor()
        if self.cache_pos and not runtime_mode:
            self.__pos = out
        return out

    def get_proj_dist(self, orbit_type=None, **kwargs):
        """Return the star-planet projected distance

        Distances are normalised with respect to the stellar radius. and projected onto the viewing plane.
        :param orbit_type: if set to 'primary' or 'secondary', will set a high float constant outside of
            the concerned transit times
        :param kwargs: optional external parameters to be provided as key/values pairs and to replace module's pars
        :return: tensor of shape (N, T)
        """
        x, y, z = self.get_position(**kwargs)
        proj_dist = torch.sqrt(y ** 2 + z ** 2)
        if orbit_type is None:
            return proj_dist
        elif orbit_type == 'primary':
            return torch.where(x < 0., torch.ones_like(x, dtype=self.dtype) * MAX_RATIO_RADII, proj_dist)
        elif orbit_type == 'secondary':
            return torch.where(x > 0., torch.ones_like(x, dtype=self.dtype) * MAX_RATIO_RADII, proj_dist)

    def get_duration(self, **kwargs):
        """Return the cached or computed duration of transit(s)

        if any model parameter is provided while calling this method, no caching will take place
        :return: (N,1)-shaped tensor of durations
        """
        runtime_mode = {
            k for k in kwargs if k in self._pars_of_fun['duration']}
        if self.cache_dur and self.__dur is not None and not runtime_mode:
            return self.__dur
        d, batch_size = self.get_input_params(**kwargs, function='duration')
        out = transit_duration(d['rp'], d['P'], d['a'], d['i'], d['e'], d['w'])
        if self.cache_dur and not runtime_mode:
            self.__dur = out
        return out

    def get_drop_p(self, **kwargs):
        """Return the cached or computed flux_drop drop of transit(s), relative to the star

        :param precision: precision of computation (int between 1 and 6)
        :param kwargs: additional functions argument to replace the class' one, disabling caching where necessary
        :return: (N, T)-shaped tensor of flux_drop drop values
        """
        runtime_mode = bool(
            [k for k in kwargs if k in self._pars_of_fun['drop_p']])
        if self.cache_flux and self.__drop_p is not None and not runtime_mode:
            return self.__drop_p
        proj_dist = self.get_proj_dist(**kwargs, orbit_type='primary')
        d, batch_size = self.get_input_params(**kwargs, function='drop_p')
        batch_size = max(batch_size, proj_dist.shape[0])
        out = transit_flux_drop(d['method'], d['ldc'], d['rp'], proj_dist,
                                precision=self.precision, n_pars=batch_size)
        if self.cache_flux and not runtime_mode:
            self.__drop_p = out
        return out

    def get_drop_s(self, **kwargs):
        """Return the cached or computed flux_drop drop of eclipse(s), relative to the star

        :param precision: precision of computation (int between 1 and 6)
        :param kwargs: additional functions argument to replace the class' one, disabling caching where necessary
        :return: (N, T)-shaped tensor of flux drop values
        """
        runtime_mode = bool(
            [k for k in kwargs if k in self._pars_of_fun['drop_s']])
        if self.cache_flux and self.__drop_s is not None and not runtime_mode:
            return self.__drop_s
        d, batch_size = self.get_input_params(**kwargs, function='drop_s')
        proj_dist = self.get_proj_dist(
            **kwargs, orbit_type='secondary') / d['rp']
        batch_size = max(batch_size, proj_dist.shape[0])
        out = transit_flux_drop('linear', torch.zeros(batch_size, 1, dtype=d['rp'].dtype, device=d['rp'].device),
                                1 / d['rp'], proj_dist, precision=self.precision, n_pars=batch_size)
        out = (1. + d['fp'] * out) / (1. + d['fp'])
        if self.cache_flux and not runtime_mode:
            self.__drop_s = out
        return out

    @property
    def shape(self):
        """Return the shape of the model

        :return: tuple of dimensions (N, T) where N is the batch size and T the number of time steps
        """
        return tuple(self.__shape)

    @property
    def rp_over_rs(self):
        """Alias for 'rp'"""
        return self.rp

    @property
    def fp_over_fs(self):
        """Alias for 'fp' """
        return self.fp

    @property
    def inclination(self):
        """Alias for 'i' """
        return self.i

    @property
    def eccentricity(self):
        """Alias for 'e' """
        return self.e

    @property
    def periastron(self):
        """Alias for 'w' """
        return self.w

    @property
    def limb_darkening_coefficients(self):
        """Alias for 'ldc' """
        return self.ldc

    @property
    def sma_over_rs(self):
        """Alias for 'a' """
        return self.a

    @property
    def mid_time(self):
        """Alias for 't0' """
        return self.t0

    @property
    def period(self):
        """Alias for 'P' """
        return self.P

    ldc_dim = property(get_ldc_dim)
    position = property(get_position)
    proj_dist = property(get_proj_dist)
    duration = property(get_duration)
    drop_p = property(get_drop_p)
    drop_s = property(get_drop_s)

    def _pos_factor(self):
        return float((self.epoch_type == 'primary') - (self.epoch_type == 'secondary'))

    def _prepare_value(self, name, value, update_shape=True):
        if name == "method":
            self._check_method(value)
            return value

        # Conversion to Tensor
        if isinstance(value, nn.Parameter):
            data = value.data.to(self.dtype)
        elif isinstance(value, torch.Tensor):
            data = value.to(self.dtype)
        else:
            data = torch.tensor(value, dtype=self.dtype, requires_grad=False)

        # Dimensionality
        if name == 'time':
            data.requires_grad = False
            if len(data.shape) == 0:
                raise ValueError('data input must be a sized iterable object')
            if len(data.shape) == 1:
                data = data[None, :]
            elif len(data.shape) > 2:
                raise ValueError(
                    'time input shape must be one of: (T,), (N, T), (1, T)')

            if update_shape:
                self.__shape[1] = data.shape[1]
                if self.shape[0] in [None, 1]:
                    self.__shape[0] = data.shape[0]
                elif data.shape[0] > 1 and data.shape[0] != self.shape[0]:
                    raise RuntimeError("incompatible batch dimensions between data and parameters"
                                       + f" (module's ({self.shape[0]}) != data's ({data.shape[0]}))")
        else:
            if name == 'ldc' or (name in PLC_ALIASES and PLC_ALIASES[name] == 'ldc'):
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

    def _check_method(self, value):
        """ Checks the limb-darkening method

        ldc parameters attribute will be reset if inconsistent with the new method value.
        :param value: one of [None, 'linear', 'sqrt', 'quad', 'claret']
        :return:
        """
        if not (value is None or value in self._methods_dim):
            raise ValueError(
                f'if stated limb darkening method must be in {tuple(self._methods_dim.keys())}')
        if self.ldc is not None and self.ldc.shape[-1] != self._methods_dim[value]:
            warnings.warn(f'ldc method ({value}) incompatible with ldc tensor dimension ({self.ldc.shape[-1]}). '
                          + 'Ressetting ldc coefs.')
            self.reset_param('ldc')
