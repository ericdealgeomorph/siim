"""Shared scalar Newton solvers and ice-thickness closures for the siim
numerical core.

Single canonical home for the per-law scalar kernels that the 1D profile model
(:mod:`siim.siim1d`) and the 2D step skeletons (:mod:`siim._core.skeleton`)
both consume. These were previously maintained as duplicate copies in the 1D
and 2D modules; they are verified identical in executable logic and unified
here so a fix lands once. numpy/numba only -- no model, fastscape or matplotlib imports,
so the numerical core stays importable without the fastscape stack.

All public entry points are ``@numba.njit(cache=True)`` and topology-agnostic:
scalar (or small fixed-size) inputs, called per-node by the skeleton kernels.
The ``_*_func`` / ``_dz*_func`` helpers are the per-law residuals and their
derivatives used by the bracketed Newton iterations.
"""
import math

import numba

# Sliding-law dispatch codes for the law_code skeleton kernels (one real-file
# @njit kernel per mode x routing branches on these; per-law physics lives in
# the helpers below). The integer switch keeps cache=True working where a
# closure factory / exec templating cannot (see core_rewrite_plan.md step 0).
LAW_EFFEXP = 0
LAW_POWER = 1
LAW_COULOMB = 2


@numba.njit(cache=True)
def _solve_ice_thickness_power_analytical(D, lambda_p):
    """Closed-form root of x^3 + lambda_p^2 x^2 - lambda_p^2 D^4 = 0 with x = H^2.
    Equivalent to solving H*(1 + (H/lambda_p)^2)^(1/4) = D, but with no iteration.
    Picks the unique positive real root via Cardano (single-root regime) or trig
    form (three-real-roots regime)."""
    if D <= 0.0 or not math.isfinite(D):
        return 0.0
    lambda_p2 = lambda_p * lambda_p
    a  = lambda_p2 / 3.0
    b  = 0.5 * lambda_p2 * D**4
    a3 = a * a * a
    disc = b * (b - 2.0 * a3)         # factored form, stable near b ≈ 2a^3
    if disc >= 0.0:                   # single real root — Cardano
        D1 = math.sqrt(disc)
        u  = (b - a3 + D1) ** (1.0 / 3.0)
        y  = u + a * a / u            # = cbrt(b - a^3 + D1) + cbrt(b - a^3 - D1), no cancellation
    else:                             # three real roots — trig form, k=0 is the positive one
        arg = b / a3 - 1.0
        if arg < -1.0: arg = -1.0
        if arg >  1.0: arg =  1.0
        theta = math.acos(arg) / 3.0
        y = 2.0 * a * math.cos(theta)
    # m10 (WON'T-FIX): x = y - a cancels catastrophically for D -> 0 in the trig
    # branch (y -> a), but the returned H is then capped at ~lambda_p*sqrt(eps/3)
    # ~ 2.6 um at defaults, where the rheology factor 1+(H/lambda_p)^2 differs
    # from 1 by ~1e-17 — no physical consequence at any plausible parameter.
    x = y - a
    if x < 0.0:                       # guard FP undershoot when H is tiny
        x = 0.0
    return math.sqrt(x)

@numba.njit(cache=True)
def _solve_ice_thickness_coulomb(D, a, lambda_c, clamp, tol=1e-12, max_iter=100):
    """Solve H*(H + lambda_c/(1 - a*H^3))^(1/5) = D for H in (0, H_max).
    a = (rho_g*g*S / tau_c)^3, pole at H = H_max = a^(-1/3) (tau = tau_c).

    Uses Newton + line-search bisection with a *residual-based* exit criterion
    (``|f|`` < tol * D). A step-based criterion is unsafe here because near the
    pole fp diverges, so dH -> 0 even when f is still large, and Newton will
    exit prematurely with a spurious H. Residual-based guarantees we actually
    satisfy the equation.

    clamp: minimum relative gap from the pole (1e-12 default) that the line
    search maintains. Prevents 1-a*H^3 underflowing to 0 in double precision.
    """
    if D <= 0.0 or a <= 0.0:
        return 0.0

    H_max  = (1.0 / a) ** (1.0 / 3.0)
    H_safe = (1.0 - clamp) * H_max            # strict upper bound on H

    # Initial guess: no-sliding-correction estimate, clipped into (0, H_safe).
    H = D ** (5.0 / 6.0)
    if H >= H_safe or H <= 0.0:
        H = 0.5 * H_max

    f_tol = tol * D if D > 1.0 else tol
    for _ in range(max_iter):
        H2    = H * H
        H3    = H2 * H
        denom = 1.0 - a * H3                  # > 0 by line-search invariant

        g  = H + lambda_c / denom
        gp = 1.0 + 3.0 * a * lambda_c * H2 / (denom * denom)

        g_pow = g ** 0.2                      # g^(1/5)
        f  = H * g_pow - D
        if abs(f) < f_tol:
            return H                          # residual-based convergence
        fp = g_pow * (1.0 + 0.2 * H * gp / g)

        dH = f / fp
        # Line search: halve until H_new in (0, H_safe). Near the pole, fp
        # diverges and dH naturally shrinks, so this rarely fires after the
        # first few iterations.
        alpha = 1.0
        H_new = H - dH
        while H_new >= H_safe or H_new <= 0.0:
            alpha *= 0.5
            if alpha < 1e-15:
                return H                      # below floating-point resolution
            H_new = H - alpha * dH

        H = H_new

    return H

@numba.njit(cache=True)
def _F_func(zik, zio, zr, Fi, n):
    if zik >= zr:
        return zik - zio + Fi * (zik - zr) ** n
    return zik - zio

@numba.njit(cache=True)
def _dzF_func(zik, zr, Fi, n):
    if zik > zr:
        return 1 + n * Fi * (zik - zr) ** (n - 1)
    return 1.0

@numba.njit(cache=True)
def _solver_fluvial(zio, zr, Fi, n, epsilon=1e-3, max_iter=50):
    # f(z) = z - zio + Fi*(z - zr)^n is strictly increasing with its root in
    # [zr, zio]. Plain Newton 2-cycles for n < 1 once Fi*n*(1-n)*dz^(n-1) is
    # large enough, silently returning zio (zero erosion) — keep a bracket
    # and bisect whenever the Newton step leaves it.
    if zio <= zr:
        return zio
    lo = zr
    hi = zio
    zik = zio
    for _ in range(max_iter):
        f = _F_func(zik, zio, zr, Fi, n)
        if f > 0.0:
            hi = zik
        elif f < 0.0:
            lo = zik
        else:
            return zik
        zik_new = zik - f / _dzF_func(zik, zr, Fi, n)
        if not (lo < zik_new < hi):
            zik_new = 0.5 * (lo + hi)
        if abs(zik_new - zik) < epsilon:
            return zik_new
        zik = zik_new
    return zik

@numba.njit(cache=True)
def _solver_nonlinear_dinf(zo_i, n_rec, receivers_i, weights_i, lengths_i,
                           z_flat, A, p, dt, epsilon=1e-3, max_iter=50):
    """Multi-receiver D-inf counterpart of ``_solver_fluvial``: one bracketed
    implicit erosion Newton summed over a cell's receivers.

    ``F(z) = z - zo_i + sum_k w_k * (dt / L_k^p) * A * max(0, z - z_rk)^p``

    F is strictly increasing with its root in ``[min_k z_rk, zo_i]``; the step
    bisects whenever Newton leaves the bracket. Plain (unbracketed) Newton
    2-cycles for ``p < 1`` and silently returns ``zo_i`` (zero erosion) -- the
    same failure the scalar ``_solver_fluvial`` guards against, one
    receiver-dimension up. Shared by all three ``*_erode_2d_dinf`` eroders:
    ``(A, p) = (Ko*Qf^m, n)`` for the fluvial branch of every law, and the
    eff-exp / power glacial branches pass their own ``(A, p)``. Returns the
    updated node elevation ``z_i``.

    Allocation-free (audit N31): the D-inf pack carries at most 2 receivers
    per cell (``receivers`` is (n, 2)), so the per-receiver prefactors live in
    scalars instead of per-call heap arrays -- same arithmetic, same order,
    bit-for-bit with the array form it replaced.
    """
    G0 = 0.0
    G1 = 0.0
    zr0 = 0.0
    zr1 = 0.0
    n_active = 0
    for k in range(n_rec):
        L = lengths_i[k]
        if L <= 0.0:
            continue
        g = weights_i[k] * (dt / L ** p) * A
        zr = z_flat[receivers_i[k]]
        if n_active == 0:
            G0 = g
            zr0 = zr
        else:
            G1 = g
            zr1 = zr
        n_active += 1
    if n_active == 0:
        return zo_i
    lo = zr0
    if n_active > 1 and zr1 < lo:
        lo = zr1
    if zo_i <= lo:
        return zo_i
    hi = zo_i
    zik = zo_i
    for _ in range(max_iter):
        F = zik - zo_i
        dF = 1.0
        dz = zik - zr0
        if dz > 0.0:
            F  += G0 * dz ** p
            dF += p * G0 * dz ** (p - 1.0)
        if n_active > 1:
            dz = zik - zr1
            if dz > 0.0:
                F  += G1 * dz ** p
                dF += p * G1 * dz ** (p - 1.0)
        if F > 0.0:
            hi = zik
        elif F < 0.0:
            lo = zik
        else:
            break
        zik_new = zik - F / dF
        if not (lo < zik_new < hi):
            zik_new = 0.5 * (lo + hi)
        if abs(zik_new - zik) < epsilon:
            zik = zik_new
            break
        zik = zik_new
    return zik

@numba.njit(cache=True)
def _G_func(zik, zio, zr, Gi, nu):
    if zik >= zr:
        return zik - zio + Gi * (zik - zr) ** nu
    return zik - zio

@numba.njit(cache=True)
def _dzG_func(zik, zr, Gi, nu):
    if zik > zr:   # strict, matching _dzF_func / _dzG_func_power (0**neg at zik==zr; N8)
        return 1 + nu * Gi * (zik - zr) ** (nu - 1)
    return 1.0

@numba.njit(cache=True)
def _solver_glacial(zio, zr, Gi, nu, epsilon=1e-3, max_iter=50):
    # Bracketed Newton (root in [zr, zio]; see _solver_fluvial).
    if zio <= zr:
        return zio
    lo = zr
    hi = zio
    zik = zio
    for _ in range(max_iter):
        f = _G_func(zik, zio, zr, Gi, nu)
        if f > 0.0:
            hi = zik
        elif f < 0.0:
            lo = zik
        else:
            return zik
        zik_new = zik - f / _dzG_func(zik, zr, Gi, nu)
        if not (lo < zik_new < hi):
            zik_new = 0.5 * (lo + hi)
        if abs(zik_new - zik) < epsilon:
            return zik_new
        zik = zik_new
    return zik

@numba.njit(cache=True)
def _G_func_power(zik, zio, zr, Gi, t):
    if zik >= zr:
        return zik - zio + Gi * (zik - zr) ** t
    return zik - zio

@numba.njit(cache=True)
def _dzG_func_power(zik, zr, Gi, t):
    if zik > zr:
        return 1.0 + t * Gi * (zik - zr) ** (t - 1.0)
    return 1.0

@numba.njit(cache=True)
def _solver_glacial_power(zio, zr, Gi, t, epsilon=1e-3, max_iter=50):
    # Bracketed Newton (root in [zr, zio]; see _solver_fluvial). Bracketing
    # guards a sub-linear slope exponent t < 1 (which needs nu < 10/9); the
    # current default nu = 2 gives t = 9*nu/10 = 1.8.
    if zio <= zr:
        return zio
    lo = zr
    hi = zio
    zik = zio
    for _ in range(max_iter):
        f = _G_func_power(zik, zio, zr, Gi, t)
        if f > 0.0:
            hi = zik
        elif f < 0.0:
            lo = zik
        else:
            return zik
        zik_new = zik - f / _dzG_func_power(zik, zr, Gi, t)
        if not (lo < zik_new < hi):
            zik_new = 0.5 * (lo + hi)
        if abs(zik_new - zik) < epsilon:
            return zik_new
        zik = zik_new
    return zik

@numba.njit(cache=True)
def _F_and_dF_coulomb(zik, zio, zr, Qg, A_pre, ell, t,
                       cg, rho_g_g, tau_c, lambda_c, dx, clamp):
    """Residual F(zik) and analytical total derivative dF/dz for the
    regularized Coulomb erosion law.

    The rheology factor R*(H+R)^(-3/5) is evaluated via the H-eq substitution::

        H + R = D^5/H^5   =>   R*(H+R)^(-3/5) = (D/H)^2 * (1 - H^6/D^5)

    which bypasses the catastrophic cancellation in 1 - (rho_g*g*H*S/tau_c)^3.
    The H^6/D^5 intermediate also gives R = H*(1-H^6/D^5)/(H^6/D^5) without
    recomputing 1-y, so the derivative stays precision-clean.

    Derivation: differentiate H^5(H+R) = D^5 implicitly, using dD/dS = -3D/(5S).
    The log-slope gamma = (S/H) dH/dS comes out as::

        gamma = -3(q+1) / (6q + 5 - 2y),   q = H*lambda_c/R^2,  y = a*H^3

    with gamma -> -1 at the pole (H -> H_max) and gamma -> -1/2 in the
    zero-sliding regime. Then dM/dS = (dM/dD)(dD/dS) + (dM/dH)(dH/dS) using
    the expanded form M = D^2/H^2 - H^4/D^3 for the partials.
    """
    dz = zik - zr
    if dz <= 0.0:
        return zik - zio, 1.0
    S = dz / dx
    # D = (Qg/(cg*S^3))^(1/5). With kt inside cg there is no explicit kt
    # factor; Qg in m^3/yr, cg in m^-3/yr.
    D = (Qg / (cg * S ** 3)) ** 0.2
    a = (rho_g_g * S / tau_c) ** 3
    H = _solve_ice_thickness_coulomb(D, a, lambda_c, clamp)
    if H <= 0.0:
        return zik - zio, 1.0

    # Rheology factor M = (D/H)^2 * (1 - H^6/D^5).
    H_over_D = H / D
    D_over_H = 1.0 / H_over_D
    D_H_2 = D_over_H * D_over_H                 # (D/H)^2 = D^2/H^2
    H_D_3 = H_over_D * H_over_D * H_over_D      # (H/D)^3
    H6_D5 = H * H_over_D ** 5                   # H^6/D^5 = H/(H+R)
    one_minus_H6_D5 = 1.0 - H6_D5
    M = D_H_2 * one_minus_H6_D5
    if M <= 0.0:                                # match the D-inf twin's guard
        return zik - zio, 1.0                  # (M**ell NaN / dG_dz 1/M pole; m12)
    G = A_pre * M ** ell
    dz_t = dz ** t
    F = zik - zio + G * dz_t

    # --- analytical dF/dz ---
    # R via mass conservation (no 1-y cancellation): H+R = H/H6_D5; R = (H+R)-H.
    R = H * one_minus_H6_D5 / H6_D5
    y = a * H * H * H                           # (rho_g*g*H*S/tau_c)^3
    q = H * lambda_c / (R * R)
    gamma = -3.0 * (q + 1.0) / (6.0 * q + 5.0 - 2.0 * y)

    # dM/dS = -(1/S) [(6/5 + 2 gamma) D^2/H^2 + (9/5 + 4 gamma) H^4/D^3]
    A2 = H * H_D_3                              # H^4/D^3 = H * (H/D)^3
    dM_dS = -(1.0 / S) * ((1.2 + 2.0 * gamma) * D_H_2
                          + (1.8 + 4.0 * gamma) * A2)
    # dG/dz = (G * ell / M) * (dM/dS) * (dS/dz),  dS/dz = 1/dx
    dG_dz = (G * ell / M) * dM_dS / dx
    # dF/dz = 1 + (dG/dz) dz^t + G * t * dz^(t-1);  dz^(t-1) = dz^t / dz
    dF = 1.0 + dG_dz * dz_t + G * t * dz_t / dz
    if dF < 1.0:
        dF = 1.0                                # physical lower bound

    return F, dF

@numba.njit(cache=True)
def _solver_glacial_coulomb(zio, zr, Qg, A_pre, ell, t,
                                cg, rho_g_g, tau_c, lambda_c, dx, clamp,
                                epsilon=1e-8, max_iter=50):
    """Scalar Newton on zik, using an analytical total
    derivative (see _F_and_dF_coulomb) plus Armijo ``|F|``-decrease backtracking.
    One H-solve per accepted Newton step; the Armijo call also returns dF,
    which is carried into the next iteration.
    """
    if zio <= zr:
        return zio
    zik = zio
    F, dF = _F_and_dF_coulomb(zik, zio, zr, Qg, A_pre, ell, t,
                               cg, rho_g_g, tau_c, lambda_c, dx, clamp)
    if abs(F) < epsilon * (1.0 + abs(zik)):
        return zik                                      # already at the root

    for _ in range(max_iter):
        dz = zik - zr
        if dz <= 0.0:
            return zik

        step = F / dF

        # Armijo backtracking: halve until zik_new > zr AND |F_new| < |F|.
        alpha = 1.0
        zik_new = zik
        F_new = F
        dF_new = dF
        accepted = False
        while alpha > 1e-15:
            zik_try = zik - alpha * step
            if zik_try <= zr:
                alpha *= 0.5
                continue
            F_try, dF_try = _F_and_dF_coulomb(zik_try, zio, zr, Qg, A_pre, ell, t,
                                               cg, rho_g_g, tau_c, lambda_c, dx, clamp)
            if abs(F_try) < abs(F):
                zik_new = zik_try
                F_new = F_try
                dF_new = dF_try
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            return zik                                  # stuck — bail

        if abs(zik_new - zik) < epsilon * (1.0 + abs(zik)):
            return zik_new
        zik = zik_new
        F = F_new
        dF = dF_new
    return zik

@numba.njit(cache=True)
def _diag_solve_eff_exp_H_newton(a, C, tol=1e-12, max_iter=60):
    if C <= 0.0:
        return 0.0
    # initial guess
    if a > 0.0:
        # try both asymptotic branches and take the smaller positive one
        H1 = C ** (3.0 / 5.0)
        H2 = C / (a ** (2.0 / 3.0))
        H = H1 if H1 < H2 else H2
        if H <= 0.0:
            H = H1
    else:
        # need H > -a; start above the singular wall
        H = -a + max(1e-6, C ** (3.0 / 5.0))

    for _ in range(max_iter):
        ha = H + a
        if ha <= 0.0:
            # bisect back into H + a > 0
            H = 0.5 * (H + (-a + 1e-12))
            continue
        ha13 = ha ** (1.0 / 3.0)
        ha23 = ha13 * ha13
        f = H * ha23 - C
        # f'(H) = (ha + (2/3)*H) / ha^(1/3)
        fprime = (ha + (2.0 / 3.0) * H) / ha13
        if fprime <= 0.0:
            break
        dH = f / fprime
        H_new = H - dH
        if H_new + a <= 0.0:
            H_new = 0.5 * (H + (-a + 1e-12))
        if H_new < 0.0:
            H_new = 0.5 * H
        H = H_new
        if abs(dH) < tol * (1.0 + abs(H)):
            break
    if H < 0.0:
        H = 0.0
    return H

@numba.njit(cache=True)
def _diag_solve_power_H_newton(a, K_p, lambda_p2, tol=1e-12, max_iter=80):
    """Solve H^4 * (H^2 + lambda_p^2) * (H + a)^3 = K_p for H > max(0, -a).
    K_p = Q_g * dx^3 / cg, lambda_p2 = lambda_p^2. f is strictly increasing on
    the admissible interval, so Newton + bisect-on-bad-step converges."""
    if K_p <= 0.0:
        return 0.0
    if a > 0.0:
        # small-a (H^9 ≈ K_p) vs large-a (H^6 a^3 ≈ K_p, lambda<<H) asymptotes
        H1 = K_p ** (1.0 / 9.0)
        H2 = (K_p / (a * a * a)) ** (1.0 / 6.0)
        H = H1 if H1 < H2 else H2
        if H <= 0.0:
            H = H1
    else:
        H = -a + max(1e-6, K_p ** (1.0 / 9.0))

    for _ in range(max_iter):
        ha = H + a
        if ha <= 0.0:
            H = 0.5 * (H + (-a + 1e-12))
            continue
        H_sq = H * H
        H3 = H_sq * H
        H4 = H_sq * H_sq
        ha2 = ha * ha
        ha3 = ha2 * ha
        g = H4 * (H_sq + lambda_p2)                       # H^6 + lambda_p^2 H^4
        f = g * ha3 - K_p
        gp = 2.0 * H3 * (3.0 * H_sq + 2.0 * lambda_p2)    # g'(H)
        fp = gp * ha3 + g * 3.0 * ha2
        if fp <= 0.0:
            break
        dH = f / fp
        H_new = H - dH
        if H_new + a <= 0.0:
            H_new = 0.5 * (H + (-a + 1e-12))
        if H_new < 0.0:
            H_new = 0.5 * H
        H = H_new
        if abs(dH) < tol * (1.0 + abs(H)):
            break
    if H < 0.0:
        H = 0.0
    return H

@numba.njit(cache=True)
def _diag_solve_coulomb_H_newton(a, K_c, lambda_c, beta, clamp, tol=1e-12, max_iter=80):
    """Solve H^5 * (H+a)^3 * (H + lambda_c/(1-phi^3)) = K_c for H in admissible domain.
    K_c = Q_g * dx^3 / cg, beta = rho_g*g/(tau_c*dx), phi = beta*H*(H+a),
    pole at phi=1 (tau = tau_c). Domain: H > max(0, -a), phi < 1-clamp."""
    if K_c <= 0.0:
        return 0.0
    H_min = -a if a < 0.0 else 0.0
    target = 1.0 - clamp
    # Pole-bound H: phi = target → beta*H*(H+a) = target → H^2 + a*H - target/beta = 0
    disc = a * a + 4.0 * target / beta
    if disc < 0.0:
        return 0.0
    H_safe = 0.5 * (-a + math.sqrt(disc))
    if H_safe <= H_min:
        return 0.0

    # Initial guess: pole-far asymptote H^5*(H+a)^3*(H+lambda_c) = K_c.
    # In H << lambda_c, a≈0 limit: H^5*lambda_c = K_c → H ~ (K_c/lambda_c)^(1/5).
    # In H << lambda_c, a >> H limit: H^5*a^3*lambda_c = K_c → H ~ (K_c/(a^3*lambda_c))^(1/5).
    if lambda_c > 0.0:
        H = (K_c / lambda_c) ** 0.2
        if a > 0.0:
            H_alt = (K_c / (a * a * a * lambda_c)) ** 0.2
            if H_alt < H:
                H = H_alt
    else:
        H = K_c ** (1.0 / 9.0)
    # For an adverse bed step (a < 0) the pole-far asymptote can fall below the
    # H_min = -a wall; start just above the wall instead (mirrors the eff-exp /
    # power siblings). Otherwise the first Newton step from the clipped wall is
    # astronomical (fp ~ (H+a)^2 -> 0) and the halving line search dies at the
    # wall, returning a flat surface with zero erosion for the step (audit B7).
    if a < 0.0:
        H = -a + max(1e-6, H)
    # Clip to safe interior of admissible interval.
    H_lo = H_min + 1e-12 * max(1.0, abs(H_min))
    H_hi = 0.9 * H_safe + 0.1 * H_lo
    if H < H_lo:
        H = H_lo
    if H > H_hi:
        H = H_hi

    f_tol = tol * K_c if K_c > 1.0 else tol
    for _ in range(max_iter):
        ha = H + a
        if ha <= 0.0:
            H = 0.5 * (H + H_lo)
            continue
        H_sq = H * H
        H4 = H_sq * H_sq
        H5 = H4 * H
        ha2 = ha * ha
        ha3 = ha2 * ha
        P = beta * H * ha
        if P >= target:
            H = 0.5 * (H + H_lo)
            continue
        P2 = P * P
        P3 = P2 * P
        Q = 1.0 - P3
        T = H + lambda_c / Q
        f = H5 * ha3 * T - K_c
        if abs(f) < f_tol:
            return H
        # T'(H) = 1 + 3*lambda_c*P^2*(2H+a) / Q^2
        Tp = 1.0 + 3.0 * lambda_c * P2 * beta * (2.0 * H + a) / (Q * Q)
        fp = 5.0 * H4 * ha3 * T + H5 * 3.0 * ha2 * T + H5 * ha3 * Tp
        if fp <= 0.0:
            break
        dH = f / fp
        H_new = H - dH
        # Line search: stay in (H_lo, H_safe) and phi < target.
        alpha = 1.0
        valid = False
        for _ls in range(60):
            if H_lo < H_new < H_safe:
                P_new = beta * H_new * (H_new + a)
                if P_new < target:
                    valid = True
                    break
            alpha *= 0.5
            if alpha < 1e-15:
                break
            H_new = H - alpha * dH
        if not valid:
            return H
        H = H_new

    if H < 0.0:
        H = 0.0
    return H


# =============================================================================
# Mode-B joint-walk H-closures (drive -> H), per law. The skeleton calls these
# at the closure sub-sites; ``from_slope`` picks the local-slope analytic form
# (H from a given surface slope — mode A, and the mode-A-style outlet) over the
# from-a joint Newton form (mode-B interior, a = zb_i - zs_receiver). The
# hc_over_H column substitution G = hc_over_H*H is applied here. The surrounding
# BC logic is law-agnostic and lives in the skeleton; only these closures and
# the per-law erosion vary by law.
# =============================================================================

@numba.njit(cache=True)
def _modeb_closure_effexp(from_slope, x, qi, L, hc_over_H, cg, lambda_p):
    """Eff-exp H-closure. from_slope: x = surface slope (L unused); else
    x = a with cell length L."""
    K = cg * lambda_p ** 1.5
    exp_C = 2.0 / 9.0
    if from_slope:
        # N7 (WON'T-FIX): no x <= 0 guard here (unlike the power isfinite(D) and
        # coulomb a <= 0 branches) — unreachable because every skeleton call site
        # gates S > 0 and the post-solve scrubs any zero-NaN H. Left for parity
        # with the historical form; add `if x <= 0.0: return 0.0` if that ever
        # changes.
        return (qi / (K * x ** 3)) ** exp_C
    C = (qi / K) ** exp_C * L ** (2.0 / 3.0)
    return _diag_solve_eff_exp_H_newton(x, hc_over_H * C) / hc_over_H


@numba.njit(cache=True)
def _modeb_closure_power(from_slope, x, qi, L, hc_over_H, cg, lambda_p):
    """Power H-closure. from_slope: x = surface slope (L unused); else x = a
    with cell length L."""
    if from_slope:
        lambda_p2 = lambda_p * lambda_p
        D0 = (qi / (cg * lambda_p2 * x ** 3)) ** 0.25
        return _solve_ice_thickness_power_analytical(D0, lambda_p)
    hc6 = hc_over_H ** 6
    lpc = hc_over_H * lambda_p
    lambda_p2c = lpc * lpc
    K_p = qi * L * L * L / cg
    return _diag_solve_power_H_newton(x, hc6 * K_p, lambda_p2c) / hc_over_H


@numba.njit(cache=True)
def _modeb_closure_coulomb(from_slope, x, qi, L, hc_over_H, cg, lambda_c, tau_c,
                           rho_g_g, clamp):
    """Coulomb H-closure. from_slope: x = surface slope (L unused); else
    x = a with cell length L."""
    if from_slope:
        D0 = (qi / (cg * x ** 3)) ** 0.2
        a0 = (rho_g_g * x / tau_c) ** 3
        return _solve_ice_thickness_coulomb(D0, a0, lambda_c, clamp)
    hc6 = hc_over_H ** 6
    lambda_cc = hc_over_H * lambda_c
    K_c = qi * L * L * L / cg
    beta = rho_g_g / (tau_c * L)
    return _diag_solve_coulomb_H_newton(
        x, hc6 * K_c, lambda_cc, beta / hc_over_H, clamp) / hc_over_H


@numba.njit(cache=True)
def _modeb_closure(law_code, from_slope, x, qi, L, hc_over_H,
                   cg, lambda_p, lambda_c, tau_c, rho_g_g, clamp):
    """Dispatch the mode-B joint-walk H-closure on ``law_code``.

    The skeleton calls this at the three closure sub-sites (interior,
    border-normal, border-degenerate); the inactive law's constants are passed
    as 0.0 by the thin wrappers and reach only the unused branch.
    """
    if law_code == LAW_EFFEXP:
        return _modeb_closure_effexp(from_slope, x, qi, L, hc_over_H, cg, lambda_p)
    elif law_code == LAW_POWER:
        return _modeb_closure_power(from_slope, x, qi, L, hc_over_H, cg, lambda_p)
    else:
        return _modeb_closure_coulomb(from_slope, x, qi, L, hc_over_H,
                                      cg, lambda_c, tau_c, rho_g_g, clamp)
