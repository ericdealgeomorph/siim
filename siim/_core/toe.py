"""The toe slope correction's law pieces, one source for siim1d and siim2d.

Near a terminus the steady surface is singular, S ~ Q^-phi_t, and the upwind
step across it pins the terminus to its cell. :func:`thin_ice_phi` gives the
exponent phi_t, :func:`toe_flux_factor` the exact-slope factor c(q).
"""
import math

import numba
import numpy as np


def thin_ice_phi(sliding_law, mu_over_nu):
    """phi_t, the exponent of the singular toe surface S ~ s^-phi_t (s the
    distance from the terminus): the law's phi = (dlnE/dlnQ)/(dlnE/dlnS) in its
    thin-ice limit, the limit every terminus reaches (Q ~ s -> 0 takes H -> 0).
    Power: H << lambda_p gives Q = cg*lambda_p^2*S^3*H^4 and
    E ~ Q^(ell/2)*S^(3*ell/2), so 1/3. Coulomb: H << R gives
    Q = cg*R*S^3*H^5 and E ~ R^(2*ell/5)*Q^(3*ell/5)*S^(6*ell/5), so 1/2
    (tau = rho*g*H*S tends to a constant at the toe, so R does too).
    Effective exponent ('eff-exp'): phi = mu/nu at any thickness, read from
    ``mu_over_nu``."""
    if sliding_law == 'power':
        return 1.0 / 3.0
    if sliding_law == 'coulomb':
        return 0.5
    return mu_over_nu


@numba.njit(cache=True)
def toe_flux_factor(q, phi):
    """c(q) = (1 - phi)(1 - q)/(1 - q^(1 - phi)), 0 < q < 1 (scalar): the
    factor on the upwind slope of a node whose receiver carries the flux
    fraction q of its own that gives the exact slope at the node for
    S ~ Q^-phi with Q linear across the cell. It rises from 1 - phi at q -> 0
    to 1 at q -> 1, as 1 - phi (1 - q)/2 near 1. Written with expm1 so it stays
    accurate as q -> 1. Needs phi < 1."""
    a = 1.0 - phi
    L = math.log(q)
    return a * math.expm1(L) / math.expm1(a * L)


@numba.njit(cache=True)
def toe_flux_factors(q, phi):
    """:func:`toe_flux_factor` over a 1D array ``q`` (a new array)."""
    out = np.empty(q.shape[0])
    for i in range(q.shape[0]):
        out[i] = toe_flux_factor(q[i], phi)
    return out
