import warnings
import torch
from torch import nn
from typing import Any


class TransitModule(nn.Module):
    _parnames = {'method', 'P', 'i', 'e', 'a', 'rp', 'fp', 't0', 'w', 'ldc'}
    _methods_dim = {'linear': 1, 'sqrt': 2, 'quad': 2, 'claret': 4}

    def __init__(self, time=None, primary=True, secondary=False, epoch_type=None, precision=3, **kwargs):
        super().__init__()
        self.time = None
        self.time_unit = None
        if time is not None:
            self.set_time(time)

        self.primary = bool(primary)
        self.secondary = bool(secondary)
        if not self.primary and not self.secondary:
            raise RuntimeError("transit can't be neither primary nor secondary")

        if epoch_type is None:
            self.epoch_type = 'primary' if self.primary else 'secondary'
        else:
            try:
                self.epoch_type = epoch_type.strip().lower()
                assert self.epoch_type in ('primary', 'secondary')
            except AttributeError:
                raise TypeError("epoch_type must be a str, one of 'primary' and 'secondary' ")
            except AssertionError:
                raise ValueError("epoch_type should be one of: 'primary', 'secondary' ")
        self.epoch_type = epoch_type
        self._pos_factor = float((self.epoch_type == 'primary') - (self.epoch_type == 'secondary'))
        self.precision = precision

        for name in self._parnames:
            setattr(self, name, None)
        if kwargs:
            self.set_param(**kwargs)

        self.__shape = [None, None]
        self.__pos = None
        self.__dur = None
        self.__flux_p = None
        self.__flux_s = None

    def __repr__(self):
        return (f"TransitModule({'primary, ' if self.primary else ''}"
                + f"{'secondary, ' if self.secondary else ''}shape={self.shape})")

    @property
    def shape(self):
        """ returns the shape of the model
        :return: tuple of dimensions (N, T) where N is the batch size and T the number of time steps
        """
        return tuple(self.__shape)

    @property
    def ldc_dim(self):
        if self.method is not None:
            return self._methods_dim[self.method]

    def set_time(self, time, time_unit=None):
        """ Sets the array of time values

        the input time vector wiill beconverted to a detached tensor

        :param time: array-like time series of time values. Shape: (T,) or (N, T)
        :param time_unit: time unit for record
        :return: None
        """
        if time is None:
            raise ValueError("time musn't be None. For clearing the time array use clear_time method")
        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time, dtype=float)
        self.time = time.detach()
        if len(self.time.shape) == 0:
            raise ValueError('time input must be a sized iterable object')
        if len(self.time.shape) == 1:
            self.time = self.time[None, :]
        elif len(self.time.shape) > 2:
            raise ValueError('time array shape must be one of: (T,), (N, T), (1, T)')
        self.__shape[1] = self.time.shape[1]
        self.time_unit = time_unit

    def clear_time(self):
        self.time = None
        self.__shape[1] = None

    def set_param(self, **kwargs):
        """ sets Module attributes as trainable Module Parameters when possible - and as simple attributes otherwise
        :param name: param name
        :param value: param value
        (default = False). This option overwrites param's setup when input is a nn.Parameter or a Tensor
        :return:
        """
        if not kwargs:
            warnings.warn('no parameter provided')
        for name, value in kwargs.items():
            if name not in self._parnames:
                raise RuntimeError(f"parameter {name} not in authorized model's list")

            if name == "method":
                self.set_method(value)
                return None
            if isinstance(value, nn.Parameter):
                data = value.data
            elif isinstance(value, torch.Tensor):
                data = value
            else:
                data = torch.tensor(value)

            # reshaping
            dim = 1
            if name == 'ldc':
                dim = self.ldc_dim
                if dim is None:
                    if len(data.shape) > 1:
                        dim = data.shape[-1]
                        if not dim in list(self._methods_dim.values()):
                            raise RuntimeError("could not retrieve a correct ldc dimensionality given param's shape")
                    else:
                        warnings.warn("Neither of method nor ldc shape seem to provide a clear ldc dimensionality."
                                      "It is advised to provide either method str arg or 2D-shaped ldc inputs")
                        dim = 1
            data = data.view(-1, dim)
            if self.__shape[0] in [None, 1]:
                self.__shape[0] = data.shape[0]
            elif data.shape[0] > 1 and data.shape[0] != self.__shape[0]:
                raise RuntimeError("incompatible batch dimensions between parameters"
                                   + f" (module's ({self.shape[0]}) != {name}'s ({data.shape[0]}))")

            param = nn.Parameter(data, requires_grad=data.requires_grad)
            if getattr(self, name) is None:
                setattr(self, name, param)
            else:
                getattr(self, name).data = param.data

    def fit_param(self, *args):
        for name in args:
            if name not in self._parnames:
                raise RuntimeError(f"parameter {name} not in authorized model's list")
            param = getattr(self, name)
            if param is None:
                warnings.warn("param is None, its grad can't be activated")
            else:
                param.requires_grad = True

    def freeze_param(self, *args):
        for name in args:
            if name not in self._parnames:
                raise RuntimeError(f"parameter {name} not in authorized model's list")
            param = getattr(self, name)
            if param is None:
                warnings.warn("param is None, its grad can't be activated")
            else:
                param.requires_grad = False

    def clear_param(self, name):
        setattr(self, name, None)

    def clear_params(self, *args):
        if not args:
            args = self._parnames
        for name in args:
            self.clear_params(name)

    def set_method(self, value):
        if not (value is None or value in self._methods_dim):
            raise ValueError(f'if stated limb darkening method must be in {tuple(self._methods_dim.keys())}')
        setattr(self, 'method', value)
        if self.ldc is not None and self.ldc.shape[-1] != self._methods_dim[value]:
            self.set_param(ldc=None)
            warnings.warn('ldc method incompatible with ldc tensor dimension. ldc coefs have been reset.')

    def forward(self, *input: Any, **kwargs: Any):
        raise NotImplementedError
