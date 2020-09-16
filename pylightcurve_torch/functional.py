""""
Adaptated functions from OSI code repository:
-name: PyLightcurve
-url: https://github.com/ucl-exoplanets/pylightcurve/ with following public license:
-license notice:
MIT License

Copyright (c) 2016-2019 Angelos Tsiaras and Konstantinos Karpouzas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import torch

from ._constants import gauss_table, PI, EPS, MAX_RATIO_RADII, MAX_ITERATIONS, ORBIT_PRECISION


def exoplanet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array,
                    ww=None, n_pars=None, dtype=None):
    inclination = inclination * PI / 180.
    periastron = periastron * PI / 180.
    if ww is None:
        ww = time_array.new_zeros(n_pars, 1, requires_grad=False, dtype=dtype)  # same device and dtype as time_array
    ww = ww * PI / 180.

    aa = torch.where(periastron < PI / 2., PI / 2. - periastron, 5. * PI / 2. - periastron)
    bb = 2 * torch.atan(torch.sqrt((1. - eccentricity) / (1. + eccentricity)) * torch.tan(aa / 2.))
    if isinstance(bb, torch.Tensor):
        bb[bb < 0.] += 2. * PI

    mid_time = mid_time - (period / 2. / PI) * (bb - eccentricity * torch.sin(bb))
    m = (time_array - mid_time - ((time_array - mid_time) / period).int() * period) * 2. * PI / period
    u0 = m
    stop = False
    u1 = 0.
    for ii in range(MAX_ITERATIONS):
        u1 = u0 - (u0 - eccentricity * torch.sin(u0) - m) / (1. - eccentricity * torch.cos(u0))
        stop = (torch.abs(u1 - u0) < ORBIT_PRECISION).all()
        if stop:
            break
        else:
            u0 = u1.clone()
    if not stop:
        u1 = u0 - (u0 - eccentricity * torch.sin(u0) - m) / (1. - eccentricity * torch.cos(u0))
        raise RuntimeError(f'Failed to find a solution in {MAX_ITERATIONS} loops. \n'
                           + f"mean precision = {torch.abs(u1 - u0).mean().item() / ORBIT_PRECISION} (req={ORBIT_PRECISION}) \n"
                           + f"{torch.isnan(u0).sum().item()} nan values in u0")

    vv = 2. * torch.atan(torch.sqrt((1. + eccentricity) / (1. - eccentricity)) * torch.tan(u1 / 2.))
    #
    rr = sma_over_rs * (1. - (eccentricity ** 2.)) / (torch.ones_like(vv) + eccentricity * torch.cos(vv))
    aa = torch.cos(vv + periastron)
    bb = torch.sin(vv + periastron)
    x = rr * bb * torch.sin(inclination)
    y = rr * (-aa * torch.cos(ww) + bb * torch.sin(ww) * torch.cos(inclination))
    z = rr * (-aa * torch.sin(ww) - bb * torch.cos(ww) * torch.cos(inclination))
    return [x, y, z]


def transit_duration(rp_over_rs, period, sma_over_rs, inclination, eccentricity, periastron, **kwargs):
    ww = periastron * PI / 180.
    ii = inclination * PI / 180.
    ee = eccentricity
    aa = sma_over_rs
    ro_pt = (1. - ee ** 2) / (1. + ee * torch.sin(ww))
    b_pt = aa * ro_pt * torch.cos(ii)
    if isinstance(b_pt, torch.Tensor):
        b_pt[b_pt > 1.] = 0.5
    s_ps = 1. + rp_over_rs
    df = torch.asin(torch.sqrt((s_ps ** 2 - b_pt ** 2) / ((aa ** 2) * (ro_pt ** 2) - b_pt ** 2)))
    abs_value = (period * (ro_pt ** 2.)) / (PI * torch.sqrt(1. - ee ** 2)) * df
    return abs_value


def eclipse_duration(rp_over_rs, period, sma_over_rs, inclination, eccentricity, periastron, **kwargs):
    return transit_duration(rp_over_rs, period, -sma_over_rs, inclination, eccentricity, periastron)


def eclipse_centered_duration(rp_over_rs, period, sma_over_rs, inclination, eccentricity, periastron, **kwargs):
    return transit_duration(rp_over_rs, period, sma_over_rs, inclination, eccentricity, periastron + 180.)


def integral_r_claret(limb_darkening_coefficients, r):
    a1, a2, a3, a4 = limb_darkening_coefficients.transpose(1, 0)
    mu44 = 1. - r * r
    mu24 = torch.sqrt(mu44)
    mu14 = torch.sqrt(mu24)
    return - (2. * (1. - a1 - a2 - a3 - a4) / 4.) * mu44 \
           - (2. * a1 / 5.) * mu44 * mu14 \
           - (2. * a2 / 6.) * mu44 * mu24 \
           - (2. * a3 / 7.) * mu44 * mu24 * mu14 \
           - (2. * a4 / 8.) * mu44 ** 2


def num_claret(r, limb_darkening_coefficients, rprs, z):
    a1, a2, a3, a4 = limb_darkening_coefficients.transpose(1, 0)
    rsq = r ** 2
    mu44 = 1. - rsq
    mu24 = torch.sqrt(mu44)
    mu14 = torch.sqrt(mu24)
    return ((1. - a1 - a2 - a3 - a4) + a1 * mu14 + a2 * mu24 + a3 * mu24 * mu14 + a4 * mu44) \
           * r * torch.acos(torch.clamp((-rprs ** 2 + z * z + rsq) / (2. * z * r), min=-1. + EPS, max=1. - EPS))


def integral_r_f_claret(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    return gauss_numerical_integration(num_claret, r1, r2, precision, limb_darkening_coefficients, rprs, z)


# integral definitions for linear method


def integral_r_linear(limb_darkening_coefficients, r):
    a1 = limb_darkening_coefficients[:, 0]
    musq = 1. - r ** 2
    return (-1. / 6.) * musq * (3. + a1 * (-3. + 2. * torch.sqrt(musq)))


def num_linear(r, limb_darkening_coefficients, rprs, z):
    a1 = limb_darkening_coefficients[:, 0]
    rsq = r ** 2
    return (1. - a1 * (1. - torch.sqrt(1. - rsq))) \
           * r * torch.acos(torch.clamp((-rprs ** 2 + z ** 2 + rsq) / (2. * z * r), min=-1 + EPS, max=1. - EPS))


def integral_r_f_linear(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    return gauss_numerical_integration(num_linear, r1, r2, precision, limb_darkening_coefficients, rprs, z)


# integral definitions for quadratic method


def integral_r_quad(limb_darkening_coefficients, r):
    a1, a2 = limb_darkening_coefficients.transpose(1, 0)
    musq = 1. - r ** 2
    mu = torch.sqrt(musq)
    return (1. / 12.) * (-4. * (a1 + 2. * a2) * mu * musq + 6. * (-1. + a1 + a2) * musq + 3. * a2 * musq * musq)


def num_quad(r, limb_darkening_coefficients, rprs, z):
    a1, a2 = limb_darkening_coefficients.transpose(1, 0)
    rsq = r ** 2
    cc = 1. - torch.sqrt(1. - rsq)
    return (1. - a1 * cc - a2 * cc * cc) \
           * r * torch.acos(torch.clamp((-rprs ** 2 + z ** 2 + rsq) / (2. * z * r), min=-1. + EPS, max=1. - EPS))


def integral_r_f_quad(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    return gauss_numerical_integration(num_quad, r1, r2, precision, limb_darkening_coefficients, rprs, z)


# integral definitions for square root method


def integral_r_sqrt(limb_darkening_coefficients, r):
    a1, a2 = limb_darkening_coefficients.transpose(1, 0)
    musq = 1. - r ** 2
    mu = torch.sqrt(musq)
    return ((-2. / 5.) * a2 * torch.sqrt(mu) - (1. / 3.) * a1 * mu + (1. / 2.) * (-1 + a1 + a2)) * musq


def num_sqrt_torch(r, limb_darkening_coefficients, rprs, z):
    a1, a2 = limb_darkening_coefficients.transpose(1, 0)
    rsq = r ** 2
    mu = torch.sqrt(1. - rsq)
    return ((1. - a1 * (1. - mu) - a2 * (1. - torch.sqrt(mu)))
            * r * torch.acos(torch.clamp((-rprs ** 2 + z ** 2 + rsq) / (2. * z * r), min=-1. + EPS, max=1. - EPS)))


def integral_r_f_sqrt_torch(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    return gauss_numerical_integration(num_sqrt_torch, r1, r2, precision, limb_darkening_coefficients, rprs, z)


# dictionaries containing the different methods,
# if you define a new method, include the functions in the dictionary as well

integral_r = {
    'claret': integral_r_claret,
    'linear': integral_r_linear,
    'quad': integral_r_quad,
    'sqrt': integral_r_sqrt
}

integral_r_f = {
    'claret': integral_r_f_claret,
    'linear': integral_r_f_linear,
    'quad': integral_r_f_quad,
    'sqrt': integral_r_f_sqrt_torch,
}

num = {
    'claret': num_claret,
    'linear': num_linear,
    'quad': num_quad,
    'sqrt': num_sqrt_torch
}


def integral_centred(method, limb_darkening_coefficients, rprs, ww1, ww2):
    return (integral_r[method](limb_darkening_coefficients, rprs)
            - integral_r[method](limb_darkening_coefficients, rprs.new_zeros(1))) * torch.abs(ww2 - ww1)


def integral_plus_core(method, limb_darkening_coefficients, rprs, z, ww1, ww2, precision=3):
    if len(z) == 0:
        return z

    rr1 = z * torch.cos(ww1) + torch.sqrt(torch.clamp(rprs ** 2 - (z * torch.sin(ww1)) ** 2, EPS))
    rr1 = torch.clamp(rr1, EPS, 1. - EPS)
    rr2 = z * torch.cos(ww2) + torch.sqrt(torch.clamp(rprs ** 2 - (z * torch.sin(ww2)) ** 2, EPS))
    rr2 = torch.clamp(rr2, EPS, 1. - EPS)
    w1 = torch.min(ww1, ww2)
    r1 = torch.min(rr1, rr2)
    w2 = torch.max(ww1, ww2)
    r2 = torch.max(rr1, rr2)
    parta = integral_r[method](limb_darkening_coefficients, rprs.new_zeros(1)) * (w1 - w2)
    partb = integral_r[method](limb_darkening_coefficients, r1) * w2
    partc = integral_r[method](limb_darkening_coefficients, r2) * (-w1)
    partd = integral_r_f[method](limb_darkening_coefficients, rprs, z, r1, r2, precision=precision)
    return parta + partb + partc + partd


def integral_minus_core(method, limb_darkening_coefficients, rprs, z, ww1, ww2, precision=3):
    if len(z) == 0:
        return z
    rr1 = z * torch.cos(ww1) - torch.sqrt(torch.clamp(rprs ** 2 - (z * torch.sin(ww1)) ** 2, EPS))
    rr1 = torch.clamp(rr1, EPS, 1 - EPS)
    rr2 = z * torch.cos(ww2) - torch.sqrt(torch.clamp(rprs ** 2 - (z * torch.sin(ww2)) ** 2, EPS))
    rr2 = torch.clamp(rr2, EPS, 1 - EPS)
    w1 = torch.min(ww1, ww2)
    r1 = torch.min(rr1, rr2)
    w2 = torch.max(ww1, ww2)
    r2 = torch.max(rr1, rr2)
    parta = integral_r[method](limb_darkening_coefficients, rprs.new_zeros(1)) * (w1 - w2)
    partb = integral_r[method](limb_darkening_coefficients, r1) * (-w1)
    partc = integral_r[method](limb_darkening_coefficients, r2) * w2
    partd = integral_r_f[method](limb_darkening_coefficients, rprs, z, r1, r2, precision=precision)
    return parta + partb + partc - partd


def gauss_numerical_integration(f, x1, x2, precision, *f_args):
    x1, x2 = (x2 - x1) / 2, (x2 + x1) / 2

    return x1 * torch.sum(gauss_table[precision][0][:, None].to(device=x1.device) *
                          f(x1[None, :] * gauss_table[precision][1][:, None].to(device=x1.device) + x2[None, :],
                            *f_args), 0)  # TODO: maybe better to avoid conversion?


def transit_flux_drop(method, limb_darkening_coefficients, rp_over_rs, z_over_rs, precision=3, n_pars=None):
    """

    :param method: one of ('linear', 'sqrt', 'quad', 'claret')
    :param limb_darkening_coefficients: (N, d)-shape tensor where N is 1 or the batch_size and d the ldc dimensionality
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
    zsq = z_over_rs ** 2  # n,
    sum_z_rprs = z_over_rs + rp_over_rs  # n, T
    dif_z_rprs = rp_over_rs - z_over_rs  # n, T
    sqr_dif_z_rprs = zsq - rp_over_rs ** 2  # n, T
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
    ph = torch.acos(torch.clamp((1. - rp_over_rs ** 2 + zsq) / (2. * z_over_rs), min=-(1 - EPS), max=1 - EPS))
    theta_1 = z_over_rs.new_zeros((n_pars, n_pts))
    ph_case = case5 + casea + casec
    theta_1[ph_case] = ph[ph_case]
    theta_2 = torch.asin(torch.min(rp_over_rs / z_over_rs, torch.ones_like(z_over_rs)))
    theta_2[case1] = PI
    theta_2[case2] = PI / 2.
    theta_2[casea] = PI
    theta_2[casec] = PI / 2.
    theta_2[case7] = ph[case7]

    # flux_upper

    plusflux = z_over_rs.new_zeros(n_pars, n_pts)
    indplus = plus_case.nonzero(as_tuple=False)
    flex_ind = lambda x, ind: x[ind] if len(x) > 1 else x
    if len(indplus):
        plusflux[indplus[:, 0], indplus[:, 1]] = integral_plus_core(method,
                                                                    flex_ind(limb_darkening_coefficients,
                                                                             indplus[:, 0]),
                                                                    flex_ind(rp_over_rs, indplus[:, 0])[:, 0],
                                                                    z_over_rs[plus_case],
                                                                    theta_1[plus_case], theta_2[plus_case],
                                                                    precision=precision)

    ind0 = case0.nonzero(as_tuple=False)
    if len(ind0):
        plusflux[ind0[:, 0], ind0[:, 1]] = integral_centred(method,
                                                            flex_ind(limb_darkening_coefficients, ind0[:, 0]),
                                                            flex_ind(rp_over_rs, ind0[:, 0])[:, 0],
                                                            rp_over_rs.new_zeros(1), PI)

    indb = caseb.nonzero(as_tuple=False)
    if len(indb):
        plusflux[indb[:, 0], indb[:, 1]] = integral_centred(method,
                                                            flex_ind(limb_darkening_coefficients, indb[:, 0]),
                                                            rp_over_rs.new_ones(1),
                                                            rp_over_rs.new_zeros(1), PI)

    # flux_lower

    minsflux = z_over_rs.new_zeros(n_pars, n_pts)
    indmins = minus_case.nonzero(as_tuple=False)
    minsflux[indmins[:, 0], indmins[:, 1]] = integral_minus_core(method,
                                                                 flex_ind(limb_darkening_coefficients, indmins[:, 0]),
                                                                 flex_ind(rp_over_rs, indmins[:, 0])[:, 0],
                                                                 z_over_rs[minus_case],
                                                                 rp_over_rs.new_zeros(1),
                                                                 theta_2[minus_case],
                                                                 precision=precision)

    # flux_star
    starflux = z_over_rs.new_zeros(n_pars, n_pts)
    indstar = star_case.nonzero(as_tuple=False)
    starflux[indstar[:, 0], indstar[:, 1]] = integral_centred(method,
                                                              flex_ind(limb_darkening_coefficients, indstar[:, 0]),
                                                              rp_over_rs.new_ones(1),
                                                              z_over_rs.new_zeros(1),
                                                              ph[star_case])

    # flux_total
    total_flux = integral_centred(method, limb_darkening_coefficients, rp_over_rs.new_ones(1),
                                  rp_over_rs.new_zeros(n_pars), 2. * PI)[:, None]
    return 1 - (2. / total_flux) * (plusflux + starflux - minsflux)


def transit(method, limb_darkening_coefficients, rp_over_rs, period, sma_over_rs, eccentricity, inclination, periastron,
            mid_time, time_array, precision=3, n_pars=None, dtype=torch.float64):
    x, y, z = exoplanet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array,
                              n_pars=n_pars, dtype=dtype)
    projected_distance = torch.where(x < 0., torch.ones_like(x, device=x.device, dtype=dtype) * MAX_RATIO_RADII,
                                     torch.sqrt(y ** 2 + z ** 2))

    return transit_flux_drop(method, limb_darkening_coefficients, rp_over_rs, projected_distance, precision=precision,
                             n_pars=n_pars)


def eclipse(fp_over_fs, rp_over_rs, period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array,
            precision=3, n_pars=None, dtype=torch.float64):
    x, y, z = exoplanet_orbit(period, - sma_over_rs / rp_over_rs, eccentricity, inclination, periastron,
                              mid_time, time_array, n_pars=n_pars, dtype=dtype)
    projected_distance = torch.where(x < 0, torch.ones_like(x, dtype=dtype, device=x.device) * MAX_RATIO_RADII,
                                     torch.sqrt(y ** 2 + z ** 2))
    n_pars = max(n_pars, projected_distance.shape[0],
                 fp_over_fs.shape[0] if isinstance(fp_over_fs, torch.Tensor) else 1)

    return (1. + fp_over_fs * transit_flux_drop('linear', time_array.new_zeros(n_pars, 1), 1. / rp_over_rs,
                                                projected_distance, precision=precision,
                                                n_pars=n_pars)) / (1. + fp_over_fs)


def eclipse_centered(fp_over_fs, rp_over_rs, period, sma_over_rs, eccentricity, inclination, periastron, mid_time,
                     time_array, precision=3):
    return eclipse(fp_over_fs, rp_over_rs, period, -sma_over_rs, eccentricity, inclination, periastron + 180.,
                   mid_time, time_array, precision)
