# PyLightcurve-torch

An exoplanet transit modelling package for deep learning applications in Pytorch.

The code for orbit and flux drop computation is largely adapted from https://github.com/ucl-exoplanets/pylightcurve/ (under a MIT license). 

The module ```pylightcurve_torch.functional.py``` contains the functions implemented in Pytorch and computing the orbital positions, 
transit durations and flux drops. (see [PyLightcurve repository](https://github.com/ucl-exoplanets/pylightcurve/) 
for more information about the numerical models used).

A ```TransitModule``` class is implemented in ```pylightcurve_torch.nn.py``` with the following features:
- Computes time series of planetary positions and primary/secondary flux drops
- it inherits ```torch.nn.Module``` class to benefit from its parameters  optimisation and management capabilities and facilitated combination with neural networks
- native GPU compatibility  


### Installation
```bash
$ pip install pylightcurve-torch
```

### Basic use
```python
from pylightcurve_torch import TransitModule

tm = TransitModule(time, **transit_params)

flux_drop = tm.forward()

```
If needs be, the returned ```torch.Tensor``` can be converted to a ```numpy.ndarrray``` using ``` flux_drop.numpy()``` torch method or 
```flux.cpu().numpy()``` if the computation took place on a gpu.


### Differentiation

One of the main benefits of having a pytorch implementation for modelling transits is offered by its 
automatic differentiation feature with torch.autograd, stemming from autograd library. 

Here is an example of basic usage:
```  
...
tm.fit_param('rp')                  # activates the gradient computation for parameter 'rp'
err = loss(flux, **data)            # loss computation in pytorch 
err.backward()                      # gradients computation 
tm.rp.grad                          # access to computed gradient for parameter 'rp'
```
