"""Per-law 2D erosion loops (SFR + D-inf) for the siim numerical core.

Walk the flow graph and apply the implicit erosion step per node, delegating
the per-node closures to the shared solvers in ``siim._core.solvers``.
Consolidated into the numerical core in the pre-v1.0 rewrite. numpy/numba
only -- no model/fastscape imports.
"""
import numpy as np
import numba

from .solvers import (
    LAW_EFFEXP, LAW_POWER, _solve_ice_thickness_coulomb,
    _solver_fluvial, _solver_glacial, _solver_glacial_power,
    _solver_glacial_coulomb, _solver_nonlinear_dinf,
)

# =============================================================================
# 2D erosion loops — walk directed graph instead of 1D array indices.
# Kf and Kg are per-node arrays because dx = lengths[i] varies per edge.
# =============================================================================

@numba.njit(cache=True)
def _linear_erode_2d(z, zo, Qf, Qg, Kf, Kg, m, mu, stack, rec):
    """Linear (n=nu=1) implicit erosion on a directed graph."""
    for i in stack:
        r = rec[i]
        if r != i:
            if Qg[i] > 0.0:   # under ice (align w/ the other 6 eroders; m13)
                Gi = Kg[i] * Qg[i] ** mu
                z[i] = (zo[i] + Gi * z[r]) / (1.0 + Gi)
            else:
                Fi = Kf[i] * Qf[i] ** m
                z[i] = (zo[i] + Fi * z[r]) / (1.0 + Fi)


@numba.njit(cache=True)
def _nonlinear_erode_2d(z, zo, Qf, Qg, Kf, Kg, m, mu, n, nu, stack, rec):
    """Nonlinear implicit erosion on a directed graph."""
    for i in stack:
        r = rec[i]
        if r != i:
            if Qg[i] > 0.0:   # under ice (align w/ the other 6 eroders; m13)
                Gi = Kg[i] * Qg[i] ** mu
                z[i] = _solver_glacial(zo[i], z[r], Gi, nu)
            else:
                Fi = Kf[i] * Qf[i] ** m
                z[i] = _solver_fluvial(zo[i], z[r], Fi, n)


@numba.njit(cache=True)
def _power_erode_2d(z, zo, Qf, Qg, H, Kf, Kg_prefactor, m, n, t, lambda_p, stack, rec):
    """Power sliding law erosion on a directed graph."""
    ell_half = t / 3.0  # ell/2 = t/3  (since t = 3*ell/2)
    for i in stack:
        r = rec[i]
        if r != i:
            if Qg[i] > 0.0:
                # rheology_factor = 1 when H=0 (well-defined, finite glacial
                # erosion); falling back to fluvial here gave 'long shallow glaciers'
                # because H-solver returns 0 at S=0 nodes, and Qf << Qg there.
                # Matches siim1d's _power_erode.
                rheology_factor = 1.0 + (H[i] / lambda_p) ** 2
                Gi = Kg_prefactor[i] * (Qg[i] / rheology_factor) ** ell_half
                z[i] = _solver_glacial_power(zo[i], z[r], Gi, t)
            else:
                Fi = Kf[i] * Qf[i] ** m
                z[i] = _solver_fluvial(zo[i], z[r], Fi, n)


# =============================================================================
# D-inf implicit erosion solvers.
#
# Same form as the SFR versions but with the implicit equation summed over
# the cell's receivers:
#     F(z) = z - zo + Σ_k w_k · (dt/L_k^ν) · Co · Q^μ · max(0, z - z_rk)^ν
# Each term is monotonic non-decreasing in z (for z > z_rk), so F is monotonic
# and Newton converges from above.
#
# Iteration order: D-inf's stack is donor-first, so to march receivers-first
# (so each cell's receivers have already been updated when we hit it) we
# iterate stack[::-1].
# =============================================================================

@numba.njit(cache=True)
def _linear_erode_2d_dinf(z, zo, Qf, Qg, dt, Ko, Co, m, mu,
                         stack, nb_receivers, receivers, weights, lengths):
    """Linear (n=nu=1) implicit erosion on a multi-flow graph.

    For ν=n=1, the implicit equation collapses to::

        z (1 + K) = zo + Σ_k w_k · Gi_k · z_rk

    where K = Σ_k w_k · Gi_k. Closed-form, no Newton needed.
    """
    for idx in range(stack.shape[0] - 1, -1, -1):
        i = stack[idx]
        n_rec = nb_receivers[i]
        if n_rec == 0:
            continue
        if n_rec == 1 and receivers[i, 0] == i:
            continue
        is_glacial = Qg[i] > 0.0
        K_sum = 0.0
        Kz_sum = 0.0
        for k in range(n_rec):
            r = receivers[i, k]
            L = lengths[i, k]
            if L <= 0.0:
                continue
            if is_glacial:
                Gi = (dt / L) * Co * Qg[i] ** mu
            else:
                Gi = (dt / L) * Ko * Qf[i] ** m
            wG = weights[i, k] * Gi
            K_sum += wG
            Kz_sum += wG * z[r]
        if K_sum > 0.0:
            z[i] = (zo[i] + Kz_sum) / (1.0 + K_sum)


@numba.njit(cache=True)
def _nonlinear_erode_2d_dinf(z, zo, Qf, Qg, dt, Ko, Co, m, mu, n, nu,
                            stack, nb_receivers, receivers, weights, lengths,
                            epsilon=1e-3, max_iter=50):
    """Multi-receiver implicit Newton on z for the nonlinear case.

    F(z) = z - zo + Σ_k w_k · (dt/L_k^p) · A · max(0, z - z_rk)^p
    dF/dz = 1 + Σ_k w_k · p · (dt/L_k^p) · A · max(0, z - z_rk)^(p-1)
    where (p, A) = (ν, Co·Q_g^μ) when glacial, else (n, Ko·Q_f^m).
    """
    for idx in range(stack.shape[0] - 1, -1, -1):
        i = stack[idx]
        n_rec = nb_receivers[i]
        if n_rec == 0:
            continue
        if n_rec == 1 and receivers[i, 0] == i:
            continue
        if Qg[i] > 0.0:
            p = nu
            A = Co * Qg[i] ** mu
        else:
            p = n
            A = Ko * Qf[i] ** m
        z[i] = _solver_nonlinear_dinf(zo[i], n_rec, receivers[i], weights[i],
                                      lengths[i], z, A, p, dt, epsilon, max_iter)


@numba.njit(cache=True)
def _power_erode_2d_dinf(z, zo, Qf, Qg, H, dt, Ko, ce, m, n, t, lambda_p,
                        cg, alpha_g,
                        stack, nb_receivers, receivers, weights, lengths,
                        epsilon=1e-3, max_iter=50):
    """Power sliding law D-inf erosion. Same structure as _nonlinear_erode_2d_dinf
    but with the per-cell H-dependent G_o prefactor for the glacial branch."""
    ell_half = t / 3.0
    base = ce * (cg * lambda_p ** 2 / alpha_g ** 2) ** ell_half
    for idx in range(stack.shape[0] - 1, -1, -1):
        i = stack[idx]
        n_rec = nb_receivers[i]
        if n_rec == 0:
            continue
        if n_rec == 1 and receivers[i, 0] == i:
            continue
        if Qg[i] > 0.0:
            rheology_factor = 1.0 + (H[i] / lambda_p) ** 2
            G_o = base * (Qg[i] / rheology_factor) ** ell_half
            p = t
        else:
            G_o = Ko * Qf[i] ** m
            p = n
        z[i] = _solver_nonlinear_dinf(zo[i], n_rec, receivers[i], weights[i],
                                      lengths[i], z, G_o, p, dt, epsilon, max_iter)


# =============================================================================
# Coulomb (regularized Coulomb) sliding law.
# Structurally identical to the 1D version in siim1d.py; only the outer loop
# differs (walks fastscape's directed graph via stack/receivers instead of 1D
# indices). See docs/dev/core_rewrite_plan.md for the derivation and history.
# =============================================================================







@numba.njit(cache=True)
def _coulomb_erode_2d(z, zo, Qf, Qg, Kf, A_const_nodal, m, n, ell, t,
                      cg, rho_g_g, tau_c, lambda_c, clamp,
                      lengths, stack, rec):
    """Regularized Coulomb sliding law erosion on a directed graph (with kt
    absorbed in cg).
    A_const_nodal[i] = (dt / lengths[i]^t) * ce * (cg^(2/5)/alpha_g)^ell;
    the per-node Qg^(3*ell/5) and (H, R)-dependent mass factor are applied inside.
    """
    exp_Q = 3.0 * ell / 5.0
    for i in stack:
        r = rec[i]
        if r != i and lengths[i] > 0.0:
            if Qg[i] > 0.0:
                A_pre = A_const_nodal[i] * Qg[i] ** exp_Q
                z[i] = _solver_glacial_coulomb(
                    zo[i], z[r], Qg[i], A_pre, ell, t,
                    cg, rho_g_g, tau_c, lambda_c, lengths[i], clamp)
            else:
                Fi = Kf[i] * Qf[i] ** m
                z[i] = _solver_fluvial(zo[i], z[r], Fi, n)


@numba.njit(cache=True)
def _F_coulomb_dinf_residual(zik, zo_i, n_rec, receivers_i, weights_i, lengths_i,
                            z_flat, Qg, base_A, ell, t, exp_Q,
                            cg, rho_g_g, tau_c, lambda_c, clamp):
    """Compute the multi-receiver coulomb residual F(zik) and dF/dz.

    Mass factor M and H are computed from the cell's weighted-mean slope
    S_eff = Σ_k w_k · max(0, (zik - z_rk)/L_k); the derivative is *lagged* in S
    (we drop the dM/dS coupling and rely on Newton line-search for robustness).
    Returns (F, dF). If no downhill receiver exists, returns (zik-zo, 1).
    """
    # Weighted-mean slope across active downhill receivers
    S_eff = 0.0
    for k in range(n_rec):
        L = lengths_i[k]
        if L <= 0.0:
            continue
        r = receivers_i[k]
        dz_k = zik - z_flat[r]
        if dz_k > 0.0:
            S_eff += weights_i[k] * (dz_k / L)
    if S_eff <= 0.0:
        return zik - zo_i, 1.0

    # H from cell-scale D, then mass-conservation substitution
    D = (Qg / (cg * S_eff ** 3)) ** 0.2
    a = (rho_g_g * S_eff / tau_c) ** 3
    H = _solve_ice_thickness_coulomb(D, a, lambda_c, clamp)
    if H <= 0.0:
        return zik - zo_i, 1.0
    H_over_D = H / D
    D_over_H = 1.0 / H_over_D
    D_H_2 = D_over_H * D_over_H
    H6_D5 = H * H_over_D ** 5
    M = D_H_2 * (1.0 - H6_D5)
    if M <= 0.0:
        return zik - zo_i, 1.0

    # Per-cell prefactor (no L dependence yet)
    G_cell = base_A * Qg ** exp_Q * M ** ell

    # F, dF — receiver-wise weighted contribution, lagged dG/dz
    F = zik - zo_i
    dF = 1.0
    for k in range(n_rec):
        L = lengths_i[k]
        if L <= 0.0:
            continue
        r = receivers_i[k]
        dz_k = zik - z_flat[r]
        if dz_k <= 0.0:
            continue
        c_k = weights_i[k] * G_cell / L ** t
        dz_t = dz_k ** t
        F  += c_k * dz_t
        dF += c_k * t * dz_t / dz_k
    if dF < 1.0:
        dF = 1.0
    return F, dF


@numba.njit(cache=True)
def _solver_glacial_coulomb_dinf(zo_i, n_rec, receivers_i, weights_i, lengths_i,
                                z_flat, Qg, base_A_with_dt, ell, t, exp_Q,
                                cg, rho_g_g, tau_c, lambda_c, clamp,
                                epsilon=1e-8, max_iter=50):
    """Multi-receiver coulomb Newton with Armijo ``|F|``-decrease
    backtracking."""
    zik = zo_i
    F, dF = _F_coulomb_dinf_residual(
        zik, zo_i, n_rec, receivers_i, weights_i, lengths_i,
        z_flat, Qg, base_A_with_dt, ell, t, exp_Q,
        cg, rho_g_g, tau_c, lambda_c, clamp)
    if abs(F) < epsilon * (1.0 + abs(zik)):
        return zik
    for _ in range(max_iter):
        step = F / dF
        alpha = 1.0
        accepted = False
        zik_new = zik
        F_new = F
        dF_new = dF
        while alpha > 1e-15:
            zik_try = zik - alpha * step
            F_try, dF_try = _F_coulomb_dinf_residual(
                zik_try, zo_i, n_rec, receivers_i, weights_i, lengths_i,
                z_flat, Qg, base_A_with_dt, ell, t, exp_Q,
                cg, rho_g_g, tau_c, lambda_c, clamp)
            if abs(F_try) < abs(F):
                zik_new = zik_try
                F_new = F_try
                dF_new = dF_try
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            return zik
        if abs(zik_new - zik) < epsilon * (1.0 + abs(zik)):
            return zik_new
        zik = zik_new
        F = F_new
        dF = dF_new
    return zik


@numba.njit(cache=True)
def _coulomb_erode_2d_dinf(z, zo, Qf, Qg, dt, Ko, ce, m, n, ell, t,
                          cg, rho_g_g, tau_c, lambda_c, clamp, alpha_g,
                          stack, nb_receivers, receivers, weights, lengths):
    """Coulomb sliding law erosion on a multi-receiver graph.

    Per-cell H is computed from the weighted-mean slope; erosion is solved by
    multi-receiver Newton with lagged-S Jacobian and line-search backtracking.
    Fluvial fallback shares the bracketed ``_solver_nonlinear_dinf`` with the
    eff-exp/power D-inf eroders.
    """
    exp_Q = 3.0 * ell / 5.0
    base_A = ce * (cg ** 0.4 / alpha_g) ** ell
    base_A_with_dt = dt * base_A

    # Iterate stack in reverse (donor-first D-inf convention → receivers-first here)
    for idx in range(stack.shape[0] - 1, -1, -1):
        i = stack[idx]
        n_rec = nb_receivers[i]
        if n_rec == 0:
            continue
        if n_rec == 1 and receivers[i, 0] == i:
            continue
        if Qg[i] > 0.0:
            z[i] = _solver_glacial_coulomb_dinf(
                zo[i], n_rec, receivers[i], weights[i], lengths[i],
                z, Qg[i], base_A_with_dt, ell, t, exp_Q,
                cg, rho_g_g, tau_c, lambda_c, clamp)
        elif Qf[i] > 0.0:
            # Fluvial multi-receiver Newton — shared bracketed solver, identical
            # to the eff-exp/power D-inf eroders. Restores the bracket this
            # branch previously dropped (plain Newton 2-cycles for n < 1,
            # silently returning zo = zero erosion).
            z[i] = _solver_nonlinear_dinf(zo[i], n_rec, receivers[i], weights[i],
                                          lengths[i], z, Ko * Qf[i] ** m, n, dt)


# =============================================================================
# Mode-B border-bed glacial erosion rate, per law. The eroder loops skip
# self-receiving outlet cells, so the border-bed budget needs the glacial
# erosion rate computed explicitly here (consumed by the closed-form IMPLICIT
# border step in the skeletons — docs/dev/outflow_implicit_budget.md; E is
# frozen per step, the implicit solve handles the flotation ramp). Each
# returns 0 when its slope/thickness guard fails, so the law-agnostic
# skeleton calls them unconditionally.
# =============================================================================

@numba.njit(cache=True)
def _border_erosion_effexp(qi, slope, Co, mu, nu):
    if slope <= 0.0:
        return 0.0
    return Co * qi ** mu * slope ** nu


@numba.njit(cache=True)
def _border_erosion_power(qi, slope, Hi, ce, nu, cg, alpha_g, lambda_p):
    if slope <= 0.0:
        return 0.0
    ell = 3.0 * nu / 5.0
    ell_half = ell / 2.0
    t = 3.0 * ell / 2.0
    base = ce * (cg * lambda_p ** 2 / alpha_g ** 2) ** ell_half
    rheology = 1.0 + (Hi / lambda_p) ** 2
    return base * (qi / rheology) ** ell_half * slope ** t


@numba.njit(cache=True)
def _border_erosion_coulomb(qi, slope, Hi, ce, nu, cg, alpha_g):
    if slope <= 0.0 or Hi <= 0.0:
        return 0.0
    ell = nu / 2.0
    t = 6.0 * ell / 5.0
    base_A = ce * (cg ** 0.4 / alpha_g) ** ell
    Gamma = (qi / (cg * slope ** 3)) ** 0.2
    ratio6 = Hi ** 6 / Gamma ** 5
    fac = (Gamma / Hi) ** 2 * max(0.0, 1.0 - ratio6)
    return base_A * fac ** ell * qi ** (3.0 * ell / 5.0) * slope ** t


@numba.njit(cache=True)
def _modeb_border_erosion(law_code, qi, slope, Hi,
                          Co, mu, nu, ce, cg, alpha_g, lambda_p):
    """Dispatch the mode-B border-bed glacial erosion rate on ``law_code``.

    Inactive-law constants are passed as 0.0 by the thin wrappers and reach
    only the unused branch.
    """
    if law_code == LAW_EFFEXP:
        return _border_erosion_effexp(qi, slope, Co, mu, nu)
    elif law_code == LAW_POWER:
        return _border_erosion_power(qi, slope, Hi, ce, nu, cg, alpha_g, lambda_p)
    else:
        return _border_erosion_coulomb(qi, slope, Hi, ce, nu, cg, alpha_g)


# =============================================================================
# Mode-B SFR erosion step (step 4 of the joint walk): build the per-node Kf/Kg
# rate arrays and erode the lake-filled view in place. Lifted verbatim from the
# three SFR mode-B kernels; the law-agnostic skeleton calls the dispatch and
# the inactive-law constants reach only the unused branch.
# =============================================================================

@numba.njit(cache=True)
def _erode_modeb_sfr_effexp(z_filled, z_pre, water_flux, ice_flux,
                            Ko, Co, n, nu, m, mu, dt, lengths, stack, rec):
    nn = z_filled.shape[0]
    Kf = np.zeros(nn)
    Kg = np.zeros(nn)
    for i in range(nn):
        if lengths[i] > 0.0:
            Kf[i] = dt / lengths[i] ** n * Ko
            Kg[i] = dt / lengths[i] ** nu * Co
    if n == 1.0 and nu == 1.0:
        _linear_erode_2d(z_filled, z_pre, water_flux, ice_flux, Kf, Kg, m, mu, stack, rec)
    else:
        _nonlinear_erode_2d(z_filled, z_pre, water_flux, ice_flux, Kf, Kg, m, mu, n, nu, stack, rec)


@numba.njit(cache=True)
def _erode_modeb_sfr_power(z_filled, z_pre, water_flux, ice_flux, H_flat,
                           Ko, ce, n, nu, m, cg, alpha_g, lambda_p,
                           dt, lengths, stack, rec):
    nn = z_filled.shape[0]
    ell = 3.0 * nu / 5.0
    t = 3.0 * ell / 2.0
    ell_half = ell / 2.0
    Kf = np.zeros(nn)
    Kg_prefactor = np.zeros(nn)
    base = ce * (cg * lambda_p ** 2 / alpha_g ** 2) ** ell_half
    for i in range(nn):
        if lengths[i] > 0.0:
            Kf[i] = dt / lengths[i] ** n * Ko
            Kg_prefactor[i] = dt / lengths[i] ** t * base
    _power_erode_2d(z_filled, z_pre, water_flux, ice_flux, H_flat,
                    Kf, Kg_prefactor, m, n, t, lambda_p, stack, rec)


@numba.njit(cache=True)
def _erode_modeb_sfr_coulomb(z_filled, z_pre, water_flux, ice_flux,
                             Ko, ce, n, nu, m, cg, alpha_g,
                             lambda_c, tau_c, coulomb_clamp, rho_g_g,
                             dt, lengths, stack, rec):
    nn = z_filled.shape[0]
    ell = nu / 2.0
    t = 6.0 * ell / 5.0
    Kf = np.zeros(nn)
    A_const_nodal = np.zeros(nn)
    base_A = ce * (cg ** 0.4 / alpha_g) ** ell
    for i in range(nn):
        if lengths[i] > 0.0:
            Kf[i] = dt / lengths[i] ** n * Ko
            A_const_nodal[i] = dt / lengths[i] ** t * base_A
    _coulomb_erode_2d(z_filled, z_pre, water_flux, ice_flux,
                      Kf, A_const_nodal, m, n, ell, t,
                      cg, rho_g_g, tau_c, lambda_c, coulomb_clamp,
                      lengths, stack, rec)


@numba.njit(cache=True)
def _erode_modeb_sfr(law_code, z_filled, z_pre, water_flux, ice_flux, H_flat,
                     Ko, Co, ce, n, nu, m, mu, cg, alpha_g, lambda_p,
                     lambda_c, tau_c, coulomb_clamp, rho_g_g,
                     dt, lengths, stack, rec):
    """Dispatch the mode-B SFR erosion step on ``law_code``."""
    if law_code == LAW_EFFEXP:
        _erode_modeb_sfr_effexp(z_filled, z_pre, water_flux, ice_flux,
                                Ko, Co, n, nu, m, mu, dt, lengths, stack, rec)
    elif law_code == LAW_POWER:
        _erode_modeb_sfr_power(z_filled, z_pre, water_flux, ice_flux, H_flat,
                               Ko, ce, n, nu, m, cg, alpha_g, lambda_p,
                               dt, lengths, stack, rec)
    else:
        _erode_modeb_sfr_coulomb(z_filled, z_pre, water_flux, ice_flux,
                                 Ko, ce, n, nu, m, cg, alpha_g,
                                 lambda_c, tau_c, coulomb_clamp, rho_g_g,
                                 dt, lengths, stack, rec)


# =============================================================================
# Mode-B D-inf erosion step (step 4): erode the working view in place via the
# multi-receiver D-inf eroders. Sibling of _erode_modeb_sfr — the D-inf eroders
# take dt + the rate constants directly (no precomputed Kf/Kg) plus the
# (nb_receivers, receivers, weights, lengths) graph, so the signatures differ.
# Lifted verbatim from the six D-inf kernels; shared by the mode-A (on z_flat,
# zo) and mode-B (on the filled view) D-inf skeletons.
# =============================================================================

@numba.njit(cache=True)
def _erode_modeb_dinf_effexp(z, zo, water_flux, ice_flux,
                             Ko, Co, n, nu, m, mu, dt,
                             stack, nb_receivers, receivers, weights, lengths):
    if n == 1.0 and nu == 1.0:
        _linear_erode_2d_dinf(z, zo, water_flux, ice_flux, dt, Ko, Co, m, mu,
                              stack, nb_receivers, receivers, weights, lengths)
    else:
        _nonlinear_erode_2d_dinf(z, zo, water_flux, ice_flux,
                                 dt, Ko, Co, m, mu, n, nu,
                                 stack, nb_receivers, receivers, weights, lengths)


@numba.njit(cache=True)
def _erode_modeb_dinf_power(z, zo, water_flux, ice_flux, H,
                            Ko, ce, n, nu, m, cg, alpha_g, lambda_p, dt,
                            stack, nb_receivers, receivers, weights, lengths):
    ell = 3.0 * nu / 5.0
    t = 3.0 * ell / 2.0
    _power_erode_2d_dinf(z, zo, water_flux, ice_flux, H,
                         dt, Ko, ce, m, n, t, lambda_p, cg, alpha_g,
                         stack, nb_receivers, receivers, weights, lengths)


@numba.njit(cache=True)
def _erode_modeb_dinf_coulomb(z, zo, water_flux, ice_flux,
                              Ko, ce, n, nu, m, cg, alpha_g,
                              lambda_c, tau_c, coulomb_clamp, rho_g_g, dt,
                              stack, nb_receivers, receivers, weights, lengths):
    ell = nu / 2.0
    t = 6.0 * ell / 5.0
    _coulomb_erode_2d_dinf(z, zo, water_flux, ice_flux,
                           dt, Ko, ce, m, n, ell, t,
                           cg, rho_g_g, tau_c, lambda_c, coulomb_clamp, alpha_g,
                           stack, nb_receivers, receivers, weights, lengths)


@numba.njit(cache=True)
def _erode_modeb_dinf(law_code, z, zo, water_flux, ice_flux, H,
                      Ko, Co, ce, n, nu, m, mu, cg, alpha_g, lambda_p,
                      lambda_c, tau_c, coulomb_clamp, rho_g_g, dt,
                      stack, nb_receivers, receivers, weights, lengths):
    """Dispatch the mode-B D-inf erosion step on ``law_code``."""
    if law_code == LAW_EFFEXP:
        _erode_modeb_dinf_effexp(z, zo, water_flux, ice_flux,
                                 Ko, Co, n, nu, m, mu, dt,
                                 stack, nb_receivers, receivers, weights, lengths)
    elif law_code == LAW_POWER:
        _erode_modeb_dinf_power(z, zo, water_flux, ice_flux, H,
                                Ko, ce, n, nu, m, cg, alpha_g, lambda_p, dt,
                                stack, nb_receivers, receivers, weights, lengths)
    else:
        _erode_modeb_dinf_coulomb(z, zo, water_flux, ice_flux,
                                  Ko, ce, n, nu, m, cg, alpha_g,
                                  lambda_c, tau_c, coulomb_clamp, rho_g_g, dt,
                                  stack, nb_receivers, receivers, weights, lengths)

# =============================================================================
# Level-scheduled parallel mode-B erosion (the ``parallel_erode`` toggle).
#
# The serial eroders walk receivers-first because each node's implicit solve
# reads its receiver's UPDATED elevation. That dependency is a DAG depth, not
# a serial chain: bucket the nodes by topological level (level = 1 + max over
# receivers' levels; see _core.routing._levels_sfr/_levels_dinf) and every
# node within a level is independent — its receivers all live in strictly
# lower, already-finalized levels. Levels run in order; nodes within a level
# run under ``prange``. Writes are disjoint (z[i] only) and the per-node
# arithmetic replicates the serial eroders expression-for-expression (the
# prefactors are computed inline from the same formulas — including the two
# distinct ell_half rounding paths the serial SFR/D-inf power eroders use),
# so the result is BIT-FOR-BIT identical to the serial walk at any thread
# count — pinned by test_parallel_erode.
# =============================================================================

@numba.njit(cache=True, parallel=True)
def _erode_modeb_sfr_levels(law_code, z, zo, water_flux, ice_flux, H_flat,
                            Ko, Co, ce, n, nu, m, mu, cg, alpha_g, lambda_p,
                            lambda_c, tau_c, coulomb_clamp, rho_g_g,
                            dt, lengths, rec, order, offsets, nlev):
    """Level-parallel twin of :func:`_erode_modeb_sfr` (same dispatch, same
    per-node arithmetic; ``order``/``offsets``/``nlev`` from
    :func:`siim._core.routing._levels_sfr`)."""
    # eff-exp linear test (matches _erode_modeb_sfr_effexp's branch)
    linear = n == 1.0 and nu == 1.0
    # power constants (exactly _erode_modeb_sfr_power + _power_erode_2d)
    ell_p = 3.0 * nu / 5.0
    t_p = 3.0 * ell_p / 2.0
    base_p = ce * (cg * lambda_p ** 2 / alpha_g ** 2) ** (ell_p / 2.0)
    ell_half_p = t_p / 3.0
    # coulomb constants (exactly _erode_modeb_sfr_coulomb + _coulomb_erode_2d)
    ell_c = nu / 2.0
    t_c = 6.0 * ell_c / 5.0
    base_c = ce * (cg ** 0.4 / alpha_g) ** ell_c
    exp_Q = 3.0 * ell_c / 5.0
    for lev in range(nlev):
        for k in numba.prange(offsets[lev], offsets[lev + 1]):
            i = order[k]
            r = rec[i]
            if r == i:
                continue
            if law_code == LAW_EFFEXP:
                Kf_i = dt / lengths[i] ** n * Ko if lengths[i] > 0.0 else 0.0
                Kg_i = dt / lengths[i] ** nu * Co if lengths[i] > 0.0 else 0.0
                if ice_flux[i] > 0.0:
                    Gi = Kg_i * ice_flux[i] ** mu
                    if linear:
                        z[i] = (zo[i] + Gi * z[r]) / (1.0 + Gi)
                    else:
                        z[i] = _solver_glacial(zo[i], z[r], Gi, nu)
                else:
                    Fi = Kf_i * water_flux[i] ** m
                    if linear:
                        z[i] = (zo[i] + Fi * z[r]) / (1.0 + Fi)
                    else:
                        z[i] = _solver_fluvial(zo[i], z[r], Fi, n)
            elif law_code == LAW_POWER:
                if ice_flux[i] > 0.0:
                    Kg_pref = (dt / lengths[i] ** t_p * base_p
                               if lengths[i] > 0.0 else 0.0)
                    rheology_factor = 1.0 + (H_flat[i] / lambda_p) ** 2
                    Gi = Kg_pref * (ice_flux[i] / rheology_factor) ** ell_half_p
                    z[i] = _solver_glacial_power(zo[i], z[r], Gi, t_p)
                else:
                    Kf_i = dt / lengths[i] ** n * Ko if lengths[i] > 0.0 else 0.0
                    Fi = Kf_i * water_flux[i] ** m
                    z[i] = _solver_fluvial(zo[i], z[r], Fi, n)
            else:
                if lengths[i] <= 0.0:
                    continue
                if ice_flux[i] > 0.0:
                    A_pre = (dt / lengths[i] ** t_c * base_c) \
                        * ice_flux[i] ** exp_Q
                    z[i] = _solver_glacial_coulomb(
                        zo[i], z[r], ice_flux[i], A_pre, ell_c, t_c,
                        cg, rho_g_g, tau_c, lambda_c, lengths[i], coulomb_clamp)
                else:
                    Kf_i = dt / lengths[i] ** n * Ko
                    Fi = Kf_i * water_flux[i] ** m
                    z[i] = _solver_fluvial(zo[i], z[r], Fi, n)


@numba.njit(cache=True, parallel=True)
def _erode_modeb_dinf_levels(law_code, z, zo, water_flux, ice_flux, H,
                             Ko, Co, ce, n, nu, m, mu, cg, alpha_g, lambda_p,
                             lambda_c, tau_c, coulomb_clamp, rho_g_g, dt,
                             nb_receivers, receivers, weights, lengths,
                             order, offsets, nlev):
    """Level-parallel twin of :func:`_erode_modeb_dinf` (same dispatch, same
    per-node arithmetic; ``order``/``offsets``/``nlev`` from
    :func:`siim._core.routing._levels_dinf`)."""
    linear = n == 1.0 and nu == 1.0
    # power constants (exactly _erode_modeb_dinf_power + _power_erode_2d_dinf:
    # base uses ell_half = t/3.0 — the D-inf rounding path, not SFR's ell/2)
    ell_pw = 3.0 * nu / 5.0
    t_pw = 3.0 * ell_pw / 2.0
    ell_half_pw = t_pw / 3.0
    base_pw = ce * (cg * lambda_p ** 2 / alpha_g ** 2) ** ell_half_pw
    # coulomb constants (exactly _erode_modeb_dinf_coulomb + _coulomb_erode_2d_dinf)
    ell_c = nu / 2.0
    t_c = 6.0 * ell_c / 5.0
    exp_Q = 3.0 * ell_c / 5.0
    base_A_with_dt = dt * (ce * (cg ** 0.4 / alpha_g) ** ell_c)
    for lev in range(nlev):
        for k in numba.prange(offsets[lev], offsets[lev + 1]):
            i = order[k]
            n_rec = nb_receivers[i]
            if n_rec == 0:
                continue
            if n_rec == 1 and receivers[i, 0] == i:
                continue
            if law_code == LAW_EFFEXP:
                if linear:
                    # closed form (matches _linear_erode_2d_dinf)
                    is_glacial = ice_flux[i] > 0.0
                    K_sum = 0.0
                    Kz_sum = 0.0
                    for kk in range(n_rec):
                        r = receivers[i, kk]
                        L = lengths[i, kk]
                        if L <= 0.0:
                            continue
                        if is_glacial:
                            Gi = (dt / L) * Co * ice_flux[i] ** mu
                        else:
                            Gi = (dt / L) * Ko * water_flux[i] ** m
                        wG = weights[i, kk] * Gi
                        K_sum += wG
                        Kz_sum += wG * z[r]
                    if K_sum > 0.0:
                        z[i] = (zo[i] + Kz_sum) / (1.0 + K_sum)
                else:
                    if ice_flux[i] > 0.0:
                        p = nu
                        A = Co * ice_flux[i] ** mu
                    else:
                        p = n
                        A = Ko * water_flux[i] ** m
                    z[i] = _solver_nonlinear_dinf(zo[i], n_rec, receivers[i],
                                                  weights[i], lengths[i], z,
                                                  A, p, dt)
            elif law_code == LAW_POWER:
                if ice_flux[i] > 0.0:
                    rheology_factor = 1.0 + (H[i] / lambda_p) ** 2
                    G_o = base_pw * (ice_flux[i] / rheology_factor) ** ell_half_pw
                    p = t_pw
                else:
                    G_o = Ko * water_flux[i] ** m
                    p = n
                z[i] = _solver_nonlinear_dinf(zo[i], n_rec, receivers[i],
                                              weights[i], lengths[i], z,
                                              G_o, p, dt)
            else:
                if ice_flux[i] > 0.0:
                    z[i] = _solver_glacial_coulomb_dinf(
                        zo[i], n_rec, receivers[i], weights[i], lengths[i],
                        z, ice_flux[i], base_A_with_dt, ell_c, t_c, exp_Q,
                        cg, rho_g_g, tau_c, lambda_c, coulomb_clamp)
                elif water_flux[i] > 0.0:
                    z[i] = _solver_nonlinear_dinf(zo[i], n_rec, receivers[i],
                                                  weights[i], lengths[i], z,
                                                  Ko * water_flux[i] ** m, n, dt)
