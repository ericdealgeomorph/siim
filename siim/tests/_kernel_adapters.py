"""Test-only adapters with the historical per-law kernel signatures.

The pre-v1.0 API refactor dropped the per-law wrapper entry points from the
package: the model now calls the ``siim._core.skeleton`` law_code skeletons
directly with a :class:`siim._core.params.GlacialParams` record. These thin
adapters reproduce the old positional signatures (building the params record +
mapping the law) so the existing boundary/parity regression tests keep
exercising the kernels unchanged. Production code does NOT use these.
"""
from siim._core.params import GlacialParams
from siim._core.solvers import LAW_EFFEXP, LAW_POWER, LAW_COULOMB
from siim._core.skeleton import (
    _diag_walk, _glac_fast_solve_modeB_sfr, _glac_fast_solve_modeB_dinf,
)
# The historical hard-coded waterline datum was 0; pass it explicitly so the
# base-level-BC signature (bl) leaves these shims — and the BC/parity tests
# routed through them — meaning-identical. The flotation gate defaults on
# (constants.FLOTATION_GATE), matching production.
_BL0 = 0.0


def _gp_effexp(Ko, Co, n, nu, m, mu, cg, alpha_g, lambda_p, hc, D_H):
    return GlacialParams(Ko, Co, 0.0, n, nu, m, mu, cg, alpha_g, lambda_p,
                         0.0, 0.0, 0.0, 0.0, hc, D_H)


def _gp_power(Ko, ce, n, nu, m, cg, alpha_g, lambda_p, hc, D_H):
    return GlacialParams(Ko, 0.0, ce, n, nu, m, 0.0, cg, alpha_g, lambda_p,
                         0.0, 0.0, 0.0, 0.0, hc, D_H)


def _gp_coulomb(Ko, ce, n, nu, m, cg, alpha_g, lambda_c, tau_c, clamp, rho_g_g,
                hc, D_H):
    return GlacialParams(Ko, 0.0, ce, n, nu, m, 0.0, cg, alpha_g, 0.0,
                         lambda_c, tau_c, clamp, rho_g_g, hc, D_H)


# --- 1D mode-B joint walk ---------------------------------------------------

def _diag_walk_eff_exp(zb, Qg, H_out, cg, lambda_p, dx, didx_l, didx_r, nx, hc):
    p = _gp_effexp(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, cg, 0.0, lambda_p, hc, 0.0)
    _diag_walk(zb, Qg, H_out, LAW_EFFEXP, p, dx, didx_l, didx_r, nx)


def _diag_walk_power(zb, Qg, H_out, cg, lambda_p, dx, didx_l, didx_r, nx, hc):
    p = _gp_power(0.0, 0.0, 0.0, 0.0, 0.0, cg, 0.0, lambda_p, hc, 0.0)
    _diag_walk(zb, Qg, H_out, LAW_POWER, p, dx, didx_l, didx_r, nx)


def _diag_walk_coulomb(zb, Qg, H_out, cg, lambda_c, tau_c, rho_g_g, dx, clamp,
                       didx_l, didx_r, nx, hc):
    p = _gp_coulomb(0.0, 0.0, 0.0, 0.0, 0.0, cg, 0.0, lambda_c, tau_c, clamp,
                    rho_g_g, hc, 0.0)
    _diag_walk(zb, Qg, H_out, LAW_COULOMB, p, dx, didx_l, didx_r, nx)


# --- 2D mode-B, SFR ---------------------------------------------------------

def glac_fast_solve_modeB(zb_flat, ice_flux, water_flux, H_flat, surface_out,
                          Ko, Co, n, nu, m, mu, cg, alpha_g, lambda_p,
                          dt, lengths, stack, rec, D_H, ny, nx, dx_cell, dy_cell,
                          border_bed_uplift, hc):
    p = _gp_effexp(Ko, Co, n, nu, m, mu, cg, alpha_g, lambda_p, hc, D_H)
    _glac_fast_solve_modeB_sfr(zb_flat, ice_flux, water_flux,
                               H_flat, surface_out,
                               LAW_EFFEXP, p, dt, lengths, stack, rec,
                               ny, nx, dx_cell, dy_cell, border_bed_uplift, _BL0)


def glac_fast_solve_power_modeB(zb_flat, ice_flux, water_flux, H_flat, surface_out,
                                Ko, ce, n, nu, m, cg, alpha_g, lambda_p,
                                dt, lengths, stack, rec, D_H, ny, nx, dx_cell,
                                dy_cell, border_bed_uplift, hc):
    p = _gp_power(Ko, ce, n, nu, m, cg, alpha_g, lambda_p, hc, D_H)
    _glac_fast_solve_modeB_sfr(zb_flat, ice_flux, water_flux,
                               H_flat, surface_out,
                               LAW_POWER, p, dt, lengths, stack, rec,
                               ny, nx, dx_cell, dy_cell, border_bed_uplift, _BL0)


def glac_fast_solve_coulomb_modeB(zb_flat, ice_flux, water_flux, H_flat,
                                  surface_out, Ko, ce, n, nu, m, cg, alpha_g,
                                  lambda_c, tau_c, coulomb_clamp, rho_g_g,
                                  dt, lengths, stack, rec, D_H, ny, nx, dx_cell,
                                  dy_cell, border_bed_uplift, hc):
    p = _gp_coulomb(Ko, ce, n, nu, m, cg, alpha_g, lambda_c, tau_c,
                    coulomb_clamp, rho_g_g, hc, D_H)
    _glac_fast_solve_modeB_sfr(zb_flat, ice_flux, water_flux,
                               H_flat, surface_out,
                               LAW_COULOMB, p, dt, lengths, stack, rec,
                               ny, nx, dx_cell, dy_cell, border_bed_uplift, _BL0)


# --- 2D mode-B, D-inf -------------------------------------------------------

def glac_fast_solve_modeB_dinf(zb_flat, ice_flux, water_flux, H_flat, surface_out,
                               Ko, Co, n, nu, m, mu, cg, alpha_g, lambda_p,
                               dt, stack, nb_receivers, receivers, weights,
                               lengths, D_H, ny, nx, dx_cell, dy_cell,
                               border_bed_uplift, hc, wrap_y, wrap_x):
    p = _gp_effexp(Ko, Co, n, nu, m, mu, cg, alpha_g, lambda_p, hc, D_H)
    _glac_fast_solve_modeB_dinf(zb_flat, ice_flux, water_flux,
                                H_flat, surface_out,
                                LAW_EFFEXP, p, dt, stack, nb_receivers, receivers,
                                weights, lengths, ny, nx, dx_cell, dy_cell,
                                border_bed_uplift, wrap_y, wrap_x, _BL0)


def glac_fast_solve_power_modeB_dinf(zb_flat, ice_flux, water_flux, H_flat,
                                     surface_out, Ko, ce, n, nu, m, cg, alpha_g,
                                     lambda_p, dt, stack, nb_receivers, receivers,
                                     weights, lengths, D_H, ny, nx, dx_cell,
                                     dy_cell, border_bed_uplift, hc, wrap_y, wrap_x):
    p = _gp_power(Ko, ce, n, nu, m, cg, alpha_g, lambda_p, hc, D_H)
    _glac_fast_solve_modeB_dinf(zb_flat, ice_flux, water_flux,
                                H_flat, surface_out,
                                LAW_POWER, p, dt, stack, nb_receivers, receivers,
                                weights, lengths, ny, nx, dx_cell, dy_cell,
                                border_bed_uplift, wrap_y, wrap_x, _BL0)


def glac_fast_solve_coulomb_modeB_dinf(zb_flat, ice_flux, water_flux, H_flat,
                                       surface_out, Ko, ce, n, nu, m, cg, alpha_g,
                                       lambda_c, tau_c, coulomb_clamp, rho_g_g,
                                       dt, stack, nb_receivers, receivers, weights,
                                       lengths, D_H, ny, nx, dx_cell, dy_cell,
                                       border_bed_uplift, hc, wrap_y, wrap_x):
    p = _gp_coulomb(Ko, ce, n, nu, m, cg, alpha_g, lambda_c, tau_c,
                    coulomb_clamp, rho_g_g, hc, D_H)
    _glac_fast_solve_modeB_dinf(zb_flat, ice_flux, water_flux,
                                H_flat, surface_out,
                                LAW_COULOMB, p, dt, stack, nb_receivers, receivers,
                                weights, lengths, ny, nx, dx_cell, dy_cell,
                                border_bed_uplift, wrap_y, wrap_x, _BL0)
