""" A package for modelling exoplanets transits in Pytorch


The package pylightcurve-torch provides functions adapted from pylightcurve package to pytorch. A base level
TransitModule class simplifies the use of these functions, computing the transit flux drops of primary and/or seconday
eclipses. The main advantages of this new transit modelling package is its scalability on GPUs, automatic reverse-mode
differentiability and embedding in a deep learning framework for easier combination with artificial neural network
architectures and pipelines.

Available modules:
-------
functional
    functions for transit computation
nn
    models implemented as neural network modules

Classes:
-------
TransitModule
    convenience class for transit computation in pytorch

Import example
-------

    >>> import pylightcurve_torch as pt

"""

import pkg_resources

from .nn import TransitModule

__version__ = pkg_resources.get_distribution('pylightcurve-torch').version
