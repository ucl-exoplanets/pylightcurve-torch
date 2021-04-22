# PyLightcurve-torch

An exoplanet transit modelling package for deep learning applications in Pytorch.

See [this open publication in the Publications of the Astronomical Society of the Pacific](https://iopscience.iop.org/article/10.1088/1538-3873/abe6e8) for more details and official citation and [jupyter notebook tutorials here.](https://github.com/mariomorvan/pylightcurve-torch-tutorials)

The code for orbit and flux drop computation is adapted from [Pylightcurve](https://github.com/ucl-exoplanets/pylightcurve/).

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

flux_drop = tm()

```
If needs be, the returned ```torch.Tensor``` can be converted to a ```numpy.ndarrray``` using ``` flux_drop.numpy()``` torch method or 
```flux.cpu().numpy()``` if the computation took place on a gpu.



### Transit parameters

Below is a summary table of the planetary orbital and transit parameters use in PyLightcurve-torch: 

| Name         | Pylightcurve alias                | Description                                    | Python type | Unit     | Transit type      |
|--------------|-----------------------------------|------------------------------------------------|-------------|----------|-------------------|
| ```a```      | ```sma_over_rs```                 | ratio of semi-major axis by the stellar radius | float       | unitless | primary/secondary |
| ```P```      | ```period```                      | orbital period                                 | float       | days     | primary/secondary |
| ```e```      | ```eccentricity```                | orbital eccentricity                           | float       | unitless | primary/secondary |
| ```i```      | ```inclination```                 | orbital inclination                            | float       | degrees  | primary/secondary |
| ```p```      | ```periastron```                  | orbital argument of periastron                 | float       | degrees  | primary/secondary |
| ```t0```     | ```mid_time```                    | transit mid-time epoch                         | float       | days     | primary/secondary |
| ```rp```     | ```rp_over_rs```                  | ratio of planetary by stellar radii            | float       | unitless | primary/secondary |
| ```method``` | ```method```                      | limb-darkening law                             | str         |          | primary           |
| ```ldc```    | ```limb_darkening_coefficients``` | limb-darkening coefficients                    | list        | unitless | primary           |
| ```fp```     | ```fp_over_fs```                  | ratio of planetary by stellar fluxes           | float       | unitless | secondary         |

A short version of each parameter has been introduced, while maintaining a compatibility with origin PyLightcurve 
parameter names. All the parameters except method are converted to ```torch.Parameters ``` when passed to 
a ``TransitModule```, with double dtype. 



### Differentiation

One of the main benefits of having a pytorch implementation for modelling transits is offered by its 
automatic differentiation feature with torch.autograd, stemming from autograd library. 

Here is an example of basic usage:
```python  
...
tm.fit_param('rp')                  # activates the gradient computation for parameter 'rp'
err = loss(flux, **data)            # loss computation in pytorch 
err.backward()                      # gradients computation 
tm.rp.grad                          # access to computed gradient for parameter 'rp'
```


### More Pytorch support

Several utility methods inherited from PyTorch modules are listed below, simplifying operations on all module's 
defined tensor parameters. 

```python  
tm = TransitModule()

# Parameters access (iterators)
tm.parameters()
tm.named_parameters()

# dtype conversions
tm.float()
tm.double()

# Gradient local deactivation
with torch.no_grad():
    flux_no_grad = tm()

# device conversion
tm.cpu()
tm.cuda()

```

### Running performance tests

In addition to traditional unit tests, computation performance tests can be executed this way:
```python
 python tests/performance.py --plot
 
```
This will measure the computation time for computing forward transits as a function of transit duration, time vector 
length or batch size. If data have been savec previously, these will be plotted to with the name of the corresponding
tag.