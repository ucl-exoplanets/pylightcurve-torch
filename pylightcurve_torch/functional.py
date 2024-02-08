"""
This module contains functions for efficiently computing transit light curves in Pytorch.

The main functions in this module are `transit` and `eclipse`, which compute the light 
curves for primary transit and secondary eclipse events, respectively. 
It can also help to compute the duration of the transit and eclipse events or exoplanet orbit.


This module builds on analogous numpy functions from the open source PyLightcurve project 
(https://github.com/ucl-exoplanets/pylightcurve), which is licensed under the MIT license.
"""


from typing import Optional, Tuple

import torch
from torch import Tensor

from ._constants import gauss_table, PI, EPS, MAX_RATIO_RADII, MAX_ITERATIONS, ORBIT_PRECISION


# ==============================
# Public Functions
# ==============================


def transit(
    method: str,
    limb_darkening_coefficients: Tensor,
    rp_over_rs: Tensor,
    period: Tensor,
    sma_over_rs: Tensor,
    eccentricity: Tensor,
    inclination: Tensor,
    periastron: Tensor,
    mid_time: Tensor,
    time_array: Tensor,
    precision=3,
    n_pars=None,
    dtype=torch.float64,
):
    """Compute the light curve of a primary transit event.

    The function computes the light curve of a primary transit event for N different sets of
         parameters and T time steps.

    Args:
        method (str): limb-darkening law (available methods: 'claret', 'quad', 'sqrt' or 'linear')
        limb_darkening_coefficients (Tensor): A 2D tensor of shape (N, M) where 'M' is the number
            of limb darkening coefficients. Each row represents a different set of limb darkening
            coefficients.
        rp_over_rs (Tensor): (1,1) or (N, 1) shape tensor of Rp/Rs values - unitless
        period (Tensor): (1,1) or (N, 1) shape tensor of period values - unit = days
        sma_over_rs (Tensor): (1,1) or (N, 1) shape tensor of semi-major-axis values - unitless
        eccentricity (Tensor): (1,1) or (N, 1) shape tensor of eccentricity values - unitless
        inclination (Tensor): (1,1) or (N, 1) shape tensor of inclination values - unit = degrees
        periastron (Tensor): (1,1) or (N, 1) shape tensor of periastron values - unit = degrees
        mid_time (Tensor): (1,1) or (N, 1) shape tensor of mid-transit time values - unit = days
        time_array (Tensor): Tensor of time values with shape  (1, T) or (N, T) - unit = days
        precision (int, optional):integer between 1 and 6. Defaults to 3.
        n_pars (int, optional): integer specifying the batch size to resolve ambiguous cases
        dtype (dtype, optional): _description_. Defaults to torch.float64.

    Returns:
        light_curve (Tensor): Tensor of shape (N, T) of stellar flux for the provided time and 
            transit parameters.
    """

    x, y, z = exoplanet_orbit(
        period,
        sma_over_rs,
        eccentricity,
        inclination,
        periastron,
        mid_time,
        time_array,
        n_pars=n_pars,
        dtype=dtype,
    )
    projected_distance = torch.where(
        x < 0.0,
        torch.ones_like(x, device=x.device, dtype=dtype) * MAX_RATIO_RADII,
        torch.sqrt(y**2 + z**2),
    )

    return _transit_flux_drop(
        method,
        limb_darkening_coefficients,
        rp_over_rs,
        projected_distance,
        precision=precision,
        n_pars=n_pars,
    )


def eclipse(
    fp_over_fs: Tensor,
    rp_over_rs: Tensor,
    period: Tensor,
    sma_over_rs: Tensor,
    eccentricity: Tensor,
    inclination: Tensor,
    periastron: Tensor,
    mid_time: Tensor,
    time_array: Tensor,
    precision: int = 3,
    n_pars: int = None,
    dtype=torch.float64,
):
    """Compute the flux increase during a secondary eclipse event.

    The function computes the light curve of a secondary eclipse event for N different sets of
    parameters and T time steps.
    Attention, the mid_time parameter corresponds to the primary transit. You can use 
    'eclipse_centered' function to compute the light curve with the mid time properly set for 
    the secondary eclipse.

    Args:
        fp_over_fs (Tensor): (1,1) or (N, 1) shape tensor of Fp/Fs values - unitless
        rp_over_rs (Tensor): (1,1) or (N, 1) shape tensor of Rp/Rs values - unitless
        period (Tensor): (1,1) or (N, 1) shape tensor of period values - unit = days
        sma_over_rs (Tensor): (1,1) or (N, 1) shape tensor of semi-major-axis values - unitless
        eccentricity (Tensor): (1,1) or (N, 1) shape tensor of eccentricity values - unitless
        inclination (Tensor): (1,1) or (N, 1) shape tensor of inclination values - unit = degrees
        periastron (Tensor): (1,1) or (N, 1) shape tensor of periastron values - unit = degrees
        mid_time (Tensor): (1,1) or (N, 1) shape tensor of mid-transit time values - unit = days
        time_array (Tensor): Tensor of time values with shape  (1, T) or (N, T) - unit = days
        precision (int, optional): integer between 1 and 6. Defaults to 3.
        n_pars (int, optional): integer specifying the batch size to resolve ambiguous cases
        dtype (dtype, optional): _description_. Defaults to torch.float64.

    Returns:
        light_curve (Tensor): Tensor of shape (N, T) of stellar flux for the provided time and 
            eclipse parameters.
    """
    x, y, z = exoplanet_orbit(
        period,
        -sma_over_rs / rp_over_rs,
        eccentricity,
        inclination,
        periastron,
        mid_time,
        time_array,
        n_pars=n_pars,
        dtype=dtype,
    )
    projected_distance = torch.where(
        x < 0,
        torch.ones_like(x, dtype=dtype, device=x.device) * MAX_RATIO_RADII,
        torch.sqrt(y**2 + z**2),
    )
    n_pars = max(
        n_pars,
        projected_distance.shape[0],
        fp_over_fs.shape[0] if isinstance(fp_over_fs, Tensor) else 1,
    )

    return (
        1.0
        + fp_over_fs
        * _transit_flux_drop(
            "linear",
            time_array.new_zeros(n_pars, 1),
            1.0 / rp_over_rs,
            projected_distance,
            precision=precision,
            n_pars=n_pars,
        )
    ) / (1.0 + fp_over_fs)


def eclipse_centered(
    fp_over_fs: Tensor,
    rp_over_rs: Tensor,
    period: Tensor,
    sma_over_rs: Tensor,
    eccentricity: Tensor,
    inclination: Tensor,
    periastron: Tensor,
    mid_time: Tensor,
    time_array: Tensor,
    precision: int = 3,
    n_pars: int = None,
    dtype=torch.float64,
):
    """Compute the flux increase during a secondary transit event.

    The function computes the light curve of a secondary eclipse event for N different sets of
    parameters and T time steps.
    Attention, the mid_time parameter corresponds to the secondary transit. You can use 
    'eclipse_centered' function to compute the light curve with the mid time set for the primary 
    transit.

    Args:
        fp_over_fs (Tensor): (1,1) or (N, 1) shape tensor of Fp/Fs values - unitless
        rp_over_rs (Tensor): (1,1) or (N, 1) shape tensor of Rp/Rs values - unitless
        period (Tensor): (1,1) or (N, 1) shape tensor of period values - unit = days
        sma_over_rs (Tensor): (1,1) or (N, 1) shape tensor of semi-major-axis values - unitless
        eccentricity (Tensor): (1,1) or (N, 1) shape tensor of eccentricity values - unitless
        inclination (Tensor): (1,1) or (N, 1) shape tensor of inclination values - unit = degrees
        periastron (Tensor): (1,1) or (N, 1) shape tensor of periastron values - unit = degrees
        mid_time (Tensor): (1,1) or (N, 1) shape tensor of mid-transit time values - unit = days
        time_array (Tensor): Tensor of time values with shape  (1, T) or (N, T) - unit = days
        precision (int, optional): integer between 1 and 6. Defaults to 3.
        n_pars (int, optional): integer specifying the batch size to resolve ambiguous cases
        dtype (dtype, optional): _description_. Defaults to torch.float64.

    Returns:
        light_curve (Tensor): Tensor of shape (N, T) of stellar flux for the provided time and 
            eclipse parameters.
    """
    return eclipse(
        fp_over_fs,
        rp_over_rs,
        period,
        -sma_over_rs,
        eccentricity,
        inclination,
        periastron + 180.0,
        mid_time,
        time_array,
        precision,
        n_pars,
        dtype,
    )


def exoplanet_orbit(
    period: Tensor,
    sma_over_rs: Tensor,
    eccentricity: Tensor,
    inclination: Tensor,
    periastron: Tensor,
    mid_time: Tensor,
    time_array: Tensor,
    ww: Optional[Tensor] = None,
    n_pars: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute the orbit of an exoplanet.

    Args:
        period (Tensor): (1,1) or (N, 1) shape tensor of period values - unit = days
        sma_over_rs (Tensor): (1,1) or (N, 1) shape tensor of semi-major-axis values - unitless
        eccentricity (Tensor): (1,1) or (N, 1) shape tensor of eccentricity values - unitless
        inclination (Tensor): (1,1) or (N, 1) shape tensor of inclination values - unit = degrees
        periastron (Tensor): (1,1) or (N, 1) shape tensor of periastron values - unit = degrees
        mid_time (Tensor): (1,1) or (N, 1) shape tensor of mid-transit time values - unit = days
        time_array (Tensor): Tensor of time values with shape  (1, T) or (N, T) - unit = days
        ww (Tensor, optional): Tensor of argument of periastron values in degrees. Default = None
        n_pars (int, optional): integer specifying the batch size to resolve ambiguous cases
        dtype (dtype, optional): _description_. Defaults to None.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Tensors of x, y, z coordinates of the exoplanet's orbit.
    """
    # generate docstring

    inclination = inclination * PI / 180.0
    periastron = periastron * PI / 180.0
    if ww is None:
        # same device and dtype as time_array
        ww = time_array.new_zeros(n_pars, 1, requires_grad=False, dtype=dtype)
    ww = ww * PI / 180.0

    aa = torch.where(periastron < PI / 2.0, PI / 2.0 - periastron, 5.0 * PI / 2.0 - periastron)
    bb = 2 * torch.atan(
        torch.sqrt((1.0 - eccentricity) / (1.0 + eccentricity)) * torch.tan(aa / 2.0)
    )
    if isinstance(bb, Tensor):
        bb[bb < 0.0] += 2.0 * PI

    mid_time = mid_time - (period / 2.0 / PI) * (bb - eccentricity * torch.sin(bb))
    m = (
        (time_array - mid_time - ((time_array - mid_time) / period).int() * period)
        * 2.0
        * PI
        / period
    )
    u0 = m
    stop = False
    u1 = 0.0
    for _ in range(MAX_ITERATIONS):
        u1 = u0 - (u0 - eccentricity * torch.sin(u0) - m) / (1.0 - eccentricity * torch.cos(u0))
        stop = (torch.abs(u1 - u0) < ORBIT_PRECISION).all()
        if stop:
            break
        u0 = u1.clone()
    if not stop:
        u1 = u0 - (u0 - eccentricity * torch.sin(u0) - m) / (1.0 - eccentricity * torch.cos(u0))
        raise RuntimeError(
            f"Failed to find a solution in {MAX_ITERATIONS} loops. \n"
            + f"mean precision = {torch.abs(u1 - u0).mean().item() / ORBIT_PRECISION} "
            + f"(req={ORBIT_PRECISION}) \n"
            + f"{torch.isnan(u0).sum().item()} nan values in u0"
        )

    vv = 2.0 * torch.atan(
        torch.sqrt((1.0 + eccentricity) / (1.0 - eccentricity)) * torch.tan(u1 / 2.0)
    )
    #
    rr = (
        sma_over_rs
        * (1.0 - (eccentricity**2.0))
        / (torch.ones_like(vv) + eccentricity * torch.cos(vv))
    )
    aa = torch.cos(vv + periastron)
    bb = torch.sin(vv + periastron)
    x = rr * bb * torch.sin(inclination)
    y = rr * (-aa * torch.cos(ww) + bb * torch.sin(ww) * torch.cos(inclination))
    z = rr * (-aa * torch.sin(ww) - bb * torch.cos(ww) * torch.cos(inclination))
    return [x, y, z]


def transit_duration(
    rp_over_rs: Tensor,
    period: Tensor,
    sma_over_rs: Tensor,
    inclination: Tensor,
    eccentricity: Tensor,
    periastron: Tensor,
    **kwargs,
) -> Tensor:
    """Compute the duration of a primary transit.

    Args:
        rp_over_rs (Tensor): Planet radius over star radius - unitless
        period (Tensor): Orbital period - unit = days
        sma_over_rs (Tensor): Semi-major axis over star radius - unitless
        inclination (Tensor): Orbital inclination - unit = degrees
        eccentricity (Tensor): Orbital eccentricity - unitless
        periastron (Tensor): Argument of periastron - unit = degrees
        **kwargs: Additional arguments

    Returns:
        Tensor: Tensor of shape (N, 1) of transit duration for the provided parameters.
    """
    ww = periastron * PI / 180.0
    ii = inclination * PI / 180.0
    ee = eccentricity
    aa = sma_over_rs
    ro_pt = (1.0 - ee**2) / (1.0 + ee * torch.sin(ww))
    b_pt = aa * ro_pt * torch.cos(ii)
    if isinstance(b_pt, Tensor):
        b_pt[b_pt > 1.0] = 0.5
    s_ps = 1.0 + rp_over_rs
    df = torch.asin(torch.sqrt((s_ps**2 - b_pt**2) / ((aa**2) * (ro_pt**2) - b_pt**2)))
    abs_value = (period * (ro_pt**2.0)) / (PI * torch.sqrt(1.0 - ee**2)) * df
    return abs_value


def eclipse_duration(
    rp_over_rs: Tensor,
    period: Tensor,
    sma_over_rs: Tensor,
    inclination: Tensor,
    eccentricity: Tensor,
    periastron: Tensor,
    **kwargs,
) -> Tensor:
    """Compute the duration of a secondary eclipse.

    Args:
        rp_over_rs (Tensor): Planet radius over star radius - unitless
        period (Tensor): Orbital period - unit = days
        sma_over_rs (Tensor): Semi-major axis over star radius - unitless
        inclination (Tensor): Orbital inclination - unit = degrees
        eccentricity (Tensor): Orbital eccentricity - unitless
        periastron (Tensor): Argument of periastron - unit = degrees
        **kwargs: Additional arguments

    Returns:
        Tensor: Tensor of shape (N, 1) of eclipse duration for the provided parameters.
    """
    return transit_duration(rp_over_rs, period, -sma_over_rs, inclination, eccentricity,
                            periastron)


def eclipse_centered_duration(
    rp_over_rs: Tensor,
    period: Tensor,
    sma_over_rs: Tensor,
    inclination: Tensor,
    eccentricity: Tensor,
    periastron: Tensor,
    **kwargs,
) -> Tensor:
    """Compute the duration of the eclipse centered.

    Args:
        rp_over_rs (Tensor): Planet radius over star radius - unitless
        period (Tensor): Orbital period - unit = days
        sma_over_rs (Tensor): Semi-major axis over star radius - unitless
        inclination (Tensor): Orbital inclination - unit = degrees
        eccentricity (Tensor): Orbital eccentricity - unitless
        periastron (Tensor): Argument of periastron - unit = degrees
        **kwargs: Additional arguments

    Returns:
        Tensor: Tensor of shape (N, 1) of eclipse centered duration for the provided parameters.
    """
    return transit_duration(
        rp_over_rs, period, sma_over_rs, inclination, eccentricity, periastron + 180.0
    )


# ==============================
# Internal Functions
# ==============================


def _integral_r_claret(limb_darkening_coefficients, r):
    a1, a2, a3, a4 = limb_darkening_coefficients.transpose(1, 0)
    mu44 = 1.0 - r * r
    mu24 = torch.sqrt(mu44)
    mu14 = torch.sqrt(mu24)
    return (
        -(2.0 * (1.0 - a1 - a2 - a3 - a4) / 4.0) * mu44
        - (2.0 * a1 / 5.0) * mu44 * mu14
        - (2.0 * a2 / 6.0) * mu44 * mu24
        - (2.0 * a3 / 7.0) * mu44 * mu24 * mu14
        - (2.0 * a4 / 8.0) * mu44**2
    )


def _num_claret(r, limb_darkening_coefficients, rprs, z):
    a1, a2, a3, a4 = limb_darkening_coefficients.transpose(1, 0)
    rsq = r**2
    mu44 = 1.0 - rsq
    mu24 = torch.sqrt(mu44)
    mu14 = torch.sqrt(mu24)
    return (
        ((1.0 - a1 - a2 - a3 - a4) + a1 * mu14 + a2 * mu24 + a3 * mu24 * mu14 + a4 * mu44)
        * r
        * torch.acos(
            torch.clamp((-(rprs**2) + z * z + rsq) / (2.0 * z * r), min=-1.0 + EPS, max=1.0 - EPS)
        )
    )


def _integral_r_f_claret(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    return _gauss_numerical_integration(
        _num_claret, r1, r2, precision, limb_darkening_coefficients, rprs, z
    )


# integral definitions for linear method


def _integral_r_linear(limb_darkening_coefficients, r):
    a1 = limb_darkening_coefficients[:, 0]
    musq = 1.0 - r**2
    return (-1.0 / 6.0) * musq * (3.0 + a1 * (-3.0 + 2.0 * torch.sqrt(musq)))


def _num_linear(r, limb_darkening_coefficients, rprs, z):
    a1 = limb_darkening_coefficients[:, 0]
    rsq = r**2
    return (
        (1.0 - a1 * (1.0 - torch.sqrt(1.0 - rsq)))
        * r
        * torch.acos(
            torch.clamp((-(rprs**2) + z**2 + rsq) / (2.0 * z * r), min=-1 + EPS, max=1.0 - EPS)
        )
    )


def _integral_r_f_linear(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    return _gauss_numerical_integration(
        _num_linear, r1, r2, precision, limb_darkening_coefficients, rprs, z
    )


# integral definitions for quadratic method


def _integral_r_quad(limb_darkening_coefficients, r):
    a1, a2 = limb_darkening_coefficients.transpose(1, 0)
    musq = 1.0 - r**2
    mu = torch.sqrt(musq)
    return (1.0 / 12.0) * (
        -4.0 * (a1 + 2.0 * a2) * mu * musq + 6.0 * (-1.0 + a1 + a2) * musq + 3.0 * a2 * musq * musq
    )


def _num_quad(r, limb_darkening_coefficients, rprs, z):
    a1, a2 = limb_darkening_coefficients.transpose(1, 0)
    rsq = r**2
    cc = 1.0 - torch.sqrt(1.0 - rsq)
    return (
        (1.0 - a1 * cc - a2 * cc * cc)
        * r
        * torch.acos(
            torch.clamp((-(rprs**2) + z**2 + rsq) / (2.0 * z * r), min=-1.0 + EPS, max=1.0 - EPS)
        )
    )


def _integral_r_f_quad(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    return _gauss_numerical_integration(
        _num_quad, r1, r2, precision, limb_darkening_coefficients, rprs, z
    )


# integral definitions for square root method


def _integral_r_sqrt(limb_darkening_coefficients, r):
    a1, a2 = limb_darkening_coefficients.transpose(1, 0)
    musq = 1.0 - r**2
    mu = torch.sqrt(musq)
    return (
        (-2.0 / 5.0) * a2 * torch.sqrt(mu) - (1.0 / 3.0) * a1 * mu + (1.0 / 2.0) * (-1 + a1 + a2)
    ) * musq


def _num_sqrt_torch(r, limb_darkening_coefficients, rprs, z):
    a1, a2 = limb_darkening_coefficients.transpose(1, 0)
    rsq = r**2
    mu = torch.sqrt(1.0 - rsq)
    return (
        (1.0 - a1 * (1.0 - mu) - a2 * (1.0 - torch.sqrt(mu)))
        * r
        * torch.acos(
            torch.clamp((-(rprs**2) + z**2 + rsq) / (2.0 * z * r), min=-1.0 + EPS, max=1.0 - EPS)
        )
    )


def _integral_r_f_sqrt_torch(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    return _gauss_numerical_integration(
        _num_sqrt_torch, r1, r2, precision, limb_darkening_coefficients, rprs, z
    )


def _integral_centred(method, limb_darkening_coefficients, rprs, ww1, ww2):
    return (
        _integral_r[method](limb_darkening_coefficients, rprs)
        - _integral_r[method](limb_darkening_coefficients, rprs.new_zeros(1))
    ) * torch.abs(ww2 - ww1)


def _integral_plus_core(method, limb_darkening_coefficients, rprs, z, ww1, ww2, precision=3):
    if len(z) == 0:
        return z

    rr1 = z * torch.cos(ww1) + torch.sqrt(torch.clamp(rprs**2 - (z * torch.sin(ww1)) ** 2, EPS))
    rr1 = torch.clamp(rr1, EPS, 1.0 - EPS)
    rr2 = z * torch.cos(ww2) + torch.sqrt(torch.clamp(rprs**2 - (z * torch.sin(ww2)) ** 2, EPS))
    rr2 = torch.clamp(rr2, EPS, 1.0 - EPS)
    w1 = torch.min(ww1, ww2)
    r1 = torch.min(rr1, rr2)
    w2 = torch.max(ww1, ww2)
    r2 = torch.max(rr1, rr2)
    parta = _integral_r[method](limb_darkening_coefficients, rprs.new_zeros(1)) * (w1 - w2)
    partb = _integral_r[method](limb_darkening_coefficients, r1) * w2
    partc = _integral_r[method](limb_darkening_coefficients, r2) * (-w1)
    partd = _integral_r_f[method](limb_darkening_coefficients, rprs, z, r1, r2, precision=precision)
    return parta + partb + partc + partd


def _integral_minus_core(method, limb_darkening_coefficients, rprs, z, ww1, ww2, precision=3):
    if len(z) == 0:
        return z
    rr1 = z * torch.cos(ww1) - torch.sqrt(torch.clamp(rprs**2 - (z * torch.sin(ww1)) ** 2, EPS))
    rr1 = torch.clamp(rr1, EPS, 1 - EPS)
    rr2 = z * torch.cos(ww2) - torch.sqrt(torch.clamp(rprs**2 - (z * torch.sin(ww2)) ** 2, EPS))
    rr2 = torch.clamp(rr2, EPS, 1 - EPS)
    w1 = torch.min(ww1, ww2)
    r1 = torch.min(rr1, rr2)
    w2 = torch.max(ww1, ww2)
    r2 = torch.max(rr1, rr2)
    parta = _integral_r[method](limb_darkening_coefficients, rprs.new_zeros(1)) * (w1 - w2)
    partb = _integral_r[method](limb_darkening_coefficients, r1) * (-w1)
    partc = _integral_r[method](limb_darkening_coefficients, r2) * w2
    partd = _integral_r_f[method](limb_darkening_coefficients, rprs, z, r1, r2, precision=precision)
    return parta + partb + partc - partd


def _gauss_numerical_integration(f, x1, x2, precision, *f_args):
    x1, x2 = (x2 - x1) / 2, (x2 + x1) / 2

    return x1 * torch.sum(
        gauss_table[precision][0][:, None].to(device=x1.device)
        * f(
            x1[None, :] * gauss_table[precision][1][:, None].to(device=x1.device) + x2[None, :],
            *f_args,
        ),
        0,
    )

def _transit_flux_drop(
    method, limb_darkening_coefficients, rp_over_rs, z_over_rs, precision=3, n_pars=None
):
    """

    :param method: one of ('linear', 'sqrt', 'quad', 'claret')
    :param limb_darkening_coefficients: (N, d)-shape tensor where N is 1 or the batch_size and d 
        the ldc dimensionality
    :param rp_over_rs: (1,1) or (N, 1) shape tensor of Rp/Rs values
    :param z_over_rs: (1, T) or (N, T) shape tensor of projected distances
    :param precision: integer between 1 and 6
    :param n_pars: (optional), integer specifying the batch size to resolve ambiguous cases
    :return:
    """
    if n_pars is None:
        n_pars = max(rp_over_rs.shape[0], len(z_over_rs), limb_darkening_coefficients.shape[0])

    if n_pars > 1 and len(z_over_rs) == 1:
        z_over_rs = z_over_rs.repeat(n_pars, 1)

    n_pts = z_over_rs.shape[-1]

    # cases
    zsq = z_over_rs**2  # n,
    sum_z_rprs = z_over_rs + rp_over_rs  # n, T
    dif_z_rprs = rp_over_rs - z_over_rs  # n, T
    sqr_dif_z_rprs = zsq - rp_over_rs**2  # n, T
    case0 = (z_over_rs == 0) & (rp_over_rs <= 1)
    case1 = (z_over_rs < rp_over_rs) & (sum_z_rprs <= 1)
    casea = (z_over_rs < rp_over_rs) & (sum_z_rprs > 1) & (dif_z_rprs < 1)
    caseb = (z_over_rs < rp_over_rs) & (sum_z_rprs > 1) & (dif_z_rprs > 1)
    case2 = (z_over_rs == rp_over_rs) & (sum_z_rprs <= 1)
    casec = (z_over_rs == rp_over_rs) & (sum_z_rprs > 1)
    case3 = (z_over_rs > rp_over_rs) & (sum_z_rprs < 1)
    case4 = (z_over_rs > rp_over_rs) & (sum_z_rprs == 1)
    case5 = (z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs < 1)
    case6 = (z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs == 1)
    case7 = (z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs > 1) & (-1 < dif_z_rprs)
    plus_case = case1 + case2 + case3 + case4 + case5 + casea + casec
    minus_case = case3 + case4 + case5 + case6 + case7
    star_case = case5 + case6 + case7 + casea + casec

    # cross points
    ph = torch.acos(
        torch.clamp((1.0 - rp_over_rs**2 + zsq) / (2.0 * z_over_rs), min=-(1 - EPS), max=1 - EPS)
    )
    theta_1 = z_over_rs.new_zeros((n_pars, n_pts))
    ph_case = case5 + casea + casec
    theta_1[ph_case] = ph[ph_case]
    theta_2 = torch.asin(torch.min(rp_over_rs / z_over_rs, torch.ones_like(z_over_rs)))
    theta_2[case1] = PI
    theta_2[case2] = PI / 2.0
    theta_2[casea] = PI
    theta_2[casec] = PI / 2.0
    theta_2[case7] = ph[case7]

    # flux_upper

    plusflux = z_over_rs.new_zeros(n_pars, n_pts)
    indplus = plus_case.nonzero(as_tuple=False)

    def flex_ind(x, ind):
        return x[ind] if len(x) > 1 else x

    if len(indplus):
        plusflux[indplus[:, 0], indplus[:, 1]] = _integral_plus_core(
            method,
            flex_ind(limb_darkening_coefficients, indplus[:, 0]),
            flex_ind(rp_over_rs, indplus[:, 0])[:, 0],
            z_over_rs[plus_case],
            theta_1[plus_case],
            theta_2[plus_case],
            precision=precision,
        )

    ind0 = case0.nonzero(as_tuple=False)
    if len(ind0):
        plusflux[ind0[:, 0], ind0[:, 1]] = _integral_centred(
            method,
            flex_ind(limb_darkening_coefficients, ind0[:, 0]),
            flex_ind(rp_over_rs, ind0[:, 0])[:, 0],
            rp_over_rs.new_zeros(1),
            PI,
        )

    indb = caseb.nonzero(as_tuple=False)
    if len(indb):
        plusflux[indb[:, 0], indb[:, 1]] = _integral_centred(
            method,
            flex_ind(limb_darkening_coefficients, indb[:, 0]),
            rp_over_rs.new_ones(1),
            rp_over_rs.new_zeros(1),
            PI,
        )

    # flux_lower

    minsflux = z_over_rs.new_zeros(n_pars, n_pts)
    indmins = minus_case.nonzero(as_tuple=False)
    minsflux[indmins[:, 0], indmins[:, 1]] = _integral_minus_core(
        method,
        flex_ind(limb_darkening_coefficients, indmins[:, 0]),
        flex_ind(rp_over_rs, indmins[:, 0])[:, 0],
        z_over_rs[minus_case],
        rp_over_rs.new_zeros(1),
        theta_2[minus_case],
        precision=precision,
    )

    # flux_star
    starflux = z_over_rs.new_zeros(n_pars, n_pts)
    indstar = star_case.nonzero(as_tuple=False)
    starflux[indstar[:, 0], indstar[:, 1]] = _integral_centred(
        method,
        flex_ind(limb_darkening_coefficients, indstar[:, 0]),
        rp_over_rs.new_ones(1),
        z_over_rs.new_zeros(1),
        ph[star_case],
    )

    # flux_total
    total_flux = _integral_centred(
        method,
        limb_darkening_coefficients,
        rp_over_rs.new_ones(1),
        rp_over_rs.new_zeros(n_pars),
        2.0 * PI,
    )[:, None]
    return 1 - (2.0 / total_flux) * (plusflux + starflux - minsflux)


# dictionaries containing the different methods,
# if you define a new method, include the functions in the dictionary as well

_integral_r = {
    "claret": _integral_r_claret,
    "linear": _integral_r_linear,
    "quad": _integral_r_quad,
    "sqrt": _integral_r_sqrt,
}

_integral_r_f = {
    "claret": _integral_r_f_claret,
    "linear": _integral_r_f_linear,
    "quad": _integral_r_f_quad,
    "sqrt": _integral_r_f_sqrt_torch,
}
