"""Mode-A/B step skeletons (law_code switch) for the siim numerical core.

One ``@njit(cache=True)`` skeleton per (mode x routing) for 2D, plus the 1D
joint-walk skeleton. Each takes a small-int ``law_code`` and reaches the per-law
physics through the shared dispatchers in :mod:`siim._core.solvers` /
:mod:`siim._core.eroders`. The 2D model (:mod:`siim.fastscape.processes`)
and the 1D model (:mod:`siim.siim1d`) call these skeletons directly with a
``GlacialParams`` record + an integer ``law_code``. numpy/numba only -- no
model/fastscape imports, so the numerical core stays importable without the
fastscape stack.
"""
import numpy as np
import numba

from ..constants import (BL as _BL, FLOTATION_GATE as _FLOT_GATE,
                         FLOTATION_RAMP as _FLOT_RAMP, S_FLOOR_BC as _S_FLOOR)
from .solvers import _modeb_closure
from .eroders import (_erode_modeb_sfr, _erode_modeb_dinf,
                      _erode_modeb_sfr_levels, _erode_modeb_dinf_levels,
                      _modeb_border_erosion)
from .diffusion import _diffuse_H_2d
from .routing import _priority_flood_eps, _levels_sfr, _levels_dinf


@numba.njit(cache=True)
def _flot_factor(zs, H, bl, hc_over_H, ramp):
    """Waterline-flotation factor f in [0, 1] multiplying glacial erosion
    (the effective-pressure ramp described in ``docs/guides/concepts.md``)::

        f = clip((zs - bl) / (ramp*hc_over_H*H), 0, 1)

    with f = 0 EXACTLY for zs <= bl (the anti-runaway backstop: a fully
    afloat column does no glacial erosion and, via E_c <= 0, carves
    nothing). ``ramp`` = gamma, the ramp width in ice-column heights
    (delta = gamma*hc*H). ramp <= 0 — or a degenerate delta (thin ice) —
    falls back to the hard binary gate: f = 1 for zs >= bl, 0 below,
    bit-for-bit the pre-ramp behavior. Callers apply it only where the
    gate is on and the cell is icy."""
    d = ramp * hc_over_H * H
    if d <= 0.0:
        return 1.0 if zs >= bl else 0.0
    if zs <= bl:
        return 0.0
    f = (zs - bl) / d
    return 1.0 if f > 1.0 else f


@numba.njit(cache=True)
def _implicit_border_step(zb0, U, E, dt, H, hc_over_H, bl, ramp):
    """Closed-form backward-Euler step of the border-bed budget
    dzb/dt = U - f(zb)*E, with f the flotation ramp evaluated at the NEW bed::

        g(z) = z - zb0 - (U - f(z)*E)*dt
        f(z) = clip((z + hc*H - bl) / (ramp*hc*H), 0, 1)

    E (the arrival-slope border erosion rate), H and the slope are frozen per
    step, so f is piecewise linear in z and g is strictly increasing
    (g' = 1 + f'*E*dt >= 1): exactly one of three branches is consistent —
    fully grounded (f = 1), fully afloat (f = 0), or the linear ramp. The
    iterate approaches the flotation-draft equilibrium
    zb* = bl - hc*H + delta*U/E (delta = ramp*hc*H) MONOTONICALLY — it cannot
    overshoot at any dt (the reason the border budget is dt-robust where the
    explicit form dug km/step). ramp <= 0 is the binary gate, whose implicit
    solution is the Filippov sliding mode: the bed sticks at the flotation
    manifold z = bl - hc*H (no chatter possible). Verified closed-form-exact:
    residual < 2e-11 over 20 000 random draws (probe + build-time check)."""
    z1 = zb0 + (U - E) * dt          # f = 1 (fully grounded all step)
    z0 = zb0 + U * dt                # f = 0 (fully afloat all step)
    hcH = hc_over_H * H
    d = ramp * hcH
    if d <= 0.0:                     # binary gate: implicit = sliding mode
        if z1 + hcH >= bl:
            return z1
        if z0 + hcH <= bl:
            return z0
        return bl - hcH
    if z1 + hcH - bl >= d:
        return z1
    if z0 + hcH - bl <= 0.0:
        return z0
    a = E * dt / d                   # ramp branch (linear in z)
    return (zb0 + U * dt - a * (hcH - bl)) / (1.0 + a)


@numba.njit(cache=True)
def _lake_fill_sfr_2d(z_flat, stack, rec):
    """In-place monotone fill: walk stack outlet-first; any cell below its
    receiver gets raised to receiver level. The 2D analog of lake_fill_1d —
    relies on the basin-corrected receivers from fastscape's flowrouting
    (lake-interior cells point toward the spillway), so this single pass
    fills closed depressions to the spillway elevation. Boundary cells
    (rec == self) are untouched.
    """
    for inode in stack:
        r = rec[inode]
        if r != inode and z_flat[inode] < z_flat[r]:
            z_flat[inode] = z_flat[r]


@numba.njit(cache=True)
def _lake_fill_1d(zb, didx_l, didx_r, nx):
    """In-place monotone fill on the 1D bed: walk each flank from the divide
    outward and raise any node below its downstream (already-filled) receiver
    to it — the 1D analog of :func:`_lake_fill_sfr_2d`. Every closed basin
    spills at its downstream rim. ``zb`` is MODIFIED IN PLACE. njit'd because
    it is on the default mode-B 1D hot path (~33% of the step; audit m55)."""
    if didx_l >= 0:
        for i in range(1, didx_l + 1):
            if zb[i] < zb[i - 1]:
                zb[i] = zb[i - 1]
    if didx_r < nx:
        for i in range(nx - 2, didx_r - 1, -1):
            if zb[i] < zb[i + 1]:
                zb[i] = zb[i + 1]


@numba.njit(cache=True)
def _glac_fast_solve_modeA_sfr(z_flat, ice_flux, water_flux, H_flat, law_code, p,
                               dt, lengths, stack, rec):
    """SFR + mode A (z-tracking), law-agnostic skeleton.

    One ``@njit(cache=True)`` kernel for all three sliding laws (see the mode-B
    skeleton :func:`_glac_fast_solve_modeB_sfr` for the dispatch design). Mode A
    tracks the ice surface ``z_flat`` directly, so H is a per-node local solve
    from the surface slope ``S = (z_i - z_r)/L`` — exactly the ``from_slope``
    branch of :func:`_modeb_closure` (hc_over_H is irrelevant when the surface is
    tracked, so 1.0 is passed) — and there is no lake-fill, border-bed budget,
    ``surface_out`` or ``hc_over_H``. The law enters only at the H-closure and the
    erosion step, both dispatched on ``law_code``; the thin wrappers pass their
    ``LAW_*`` code and 0.0 for the inactive-law constants (which reach only the
    unused branch). ``z_flat`` and ``H_flat`` are MODIFIED IN PLACE.

    Parameters
    ----------
    z_flat : ndarray
        Ice surface elevation, flattened — MODIFIED IN PLACE.
    ice_flux, water_flux : ndarray
        Accumulated ice / water flux (m^3/yr), flattened.
    H_flat : ndarray
        Ice thickness (m), flattened — MODIFIED IN PLACE.
    law_code : int
        ``LAW_EFFEXP`` / ``LAW_POWER`` / ``LAW_COULOMB`` (see _core.solvers).
    p : GlacialParams
        Packed law/physics constants (see :mod:`siim._core.params`). Notable
        field: ``cg`` = ``alpha_g * kt * (2*Ac/5) * (rho_g*g)^3`` [m^-3 yr^-1]
        (kt absorbed).
    lengths : ndarray
        Distance from each node to its receiver (m).
    stack : ndarray
        Topological sort (upstream → downstream).
    rec : ndarray
        Receiver indices.
    """
    Ko, Co, ce, n, nu, m, mu, cg, alpha_g, lambda_p, lambda_c, tau_c, coulomb_clamp, rho_g_g, hc_over_H, D_H = p
    nn = z_flat.shape[0]
    zo = z_flat.copy()  # pre-erosion ice surface

    # --- Ice thickness: per-node H from the local surface slope (the
    # from_slope branch of the mode-B closure). ---
    for i in range(nn):
        r = rec[i]
        if r != i and ice_flux[i] > 0.0 and lengths[i] > 0.0:
            S = (z_flat[i] - z_flat[r]) / lengths[i]
            if S > 0.0:
                H_flat[i] = _modeb_closure(
                    law_code, True, S, ice_flux[i], 0.0, 1.0,
                    cg, lambda_p, lambda_c, tau_c, rho_g_g, coulomb_clamp)
            else:
                H_flat[i] = 0.0
        else:
            H_flat[i] = 0.0

    # Clean up H: remove negatives and NaNs
    for i in range(nn):
        if H_flat[i] < 0.0 or H_flat[i] != H_flat[i]:
            H_flat[i] = 0.0

    # --- Erosion: erode the surface in place (per-law dispatch; mode A erodes
    # z_flat directly against its pre-erosion copy zo, no lake-fill). ---
    _erode_modeb_sfr(law_code, z_flat, zo, water_flux, ice_flux, H_flat,
                     Ko, Co, ce, n, nu, m, mu, cg, alpha_g, lambda_p,
                     lambda_c, tau_c, coulomb_clamp, rho_g_g,
                     dt, lengths, stack, rec)

@numba.njit(cache=True)
def _glac_fast_solve_modeA_dinf(z_flat, ice_flux, water_flux, H_flat, law_code, p,
                                dt, stack, nb_receivers, receivers, weights,
                                lengths):
    """D-inf + mode A (z-tracking), law-agnostic skeleton.

    The D-inf twin of :func:`_glac_fast_solve_modeA_sfr`: H is solved per node
    from the weighted-mean per-cell slope ``S = Σ_k w_k·max(0, (z_i-z_rk)/L_k)``
    over the cell's D-inf receivers — that single effective slope feeds the same
    ``from_slope`` branch of :func:`_modeb_closure` (hc_over_H irrelevant, 1.0 passed).
    Erosion uses the D-inf eroders via :func:`_erode_modeb_dinf` on
    ``(z_flat, zo)``. ``z_flat`` and ``H_flat`` are MODIFIED IN PLACE.
    """
    Ko, Co, ce, n, nu, m, mu, cg, alpha_g, lambda_p, lambda_c, tau_c, coulomb_clamp, rho_g_g, hc_over_H, D_H = p
    nn = z_flat.shape[0]
    zo = z_flat.copy()

    # --- Ice thickness: H from the weighted-mean per-cell slope ---
    for i in range(nn):
        if ice_flux[i] <= 0.0:
            H_flat[i] = 0.0
            continue
        n_rec = nb_receivers[i]
        if n_rec == 0 or (n_rec == 1 and receivers[i, 0] == i):
            H_flat[i] = 0.0
            continue
        S = 0.0
        for k in range(n_rec):
            L = lengths[i, k]
            if L <= 0.0:
                continue
            r = receivers[i, k]
            dz = z_flat[i] - z_flat[r]
            if dz > 0.0:
                S += weights[i, k] * (dz / L)
        if S > 0.0:
            H_flat[i] = _modeb_closure(
                law_code, True, S, ice_flux[i], 0.0, 1.0,
                cg, lambda_p, lambda_c, tau_c, rho_g_g, coulomb_clamp)
        else:
            H_flat[i] = 0.0
    for i in range(nn):
        if H_flat[i] < 0.0 or H_flat[i] != H_flat[i]:
            H_flat[i] = 0.0

    # --- Erosion: D-inf eroders on the surface in place (per-law dispatch) ---
    _erode_modeb_dinf(law_code, z_flat, zo, water_flux, ice_flux, H_flat,
                      Ko, Co, ce, n, nu, m, mu, cg, alpha_g, lambda_p,
                      lambda_c, tau_c, coulomb_clamp, rho_g_g, dt,
                      stack, nb_receivers, receivers, weights, lengths)

@numba.njit(cache=True)
def _glac_fast_solve_modeB_sfr(zb_flat, ice_flux, water_flux,
                               H_flat, surface_out,
                               law_code, p,
                               dt, lengths, stack, rec,
                               ny, nx, dx_cell, dy_cell, border_bed_uplift,
                               bl=_BL, gate=_FLOT_GATE, ramp=_FLOT_RAMP,
                               wrap_y=False, wrap_x=False,
                               parallel_erode=False):
    """SFR + mode B (zb-tracking joint walk + lake-fill), law-agnostic skeleton.

    One ``@njit(cache=True)`` kernel for all three sliding laws; the per-law
    physics is reached through the ``law_code`` dispatch at the two sites that
    vary — the joint-walk H-closure (:func:`_modeb_closure`) and the erosion
    step (:func:`_erode_modeb_sfr`). Everything else (the outflow base-level
    ice BC, the two-view waterline lake-fill and the interior flotation ramp)
    is law-agnostic. The thin public wrappers pass their law's ``LAW_*`` code
    and 0.0 for the inactive-law constants, which reach only the unused
    dispatch branch.

    Base-level ice border = OUTFLOW: the domain
    edge is an arbitrary cut through a continuing glacier. A through-flowing
    border gets zero-gradient thickness ``H_border = H_dominant_donor`` (the
    max-ice-flux interior donor), and its bed keeps eroding by the IMPLICIT
    BORDER BUDGET ``dzb/dt = U - f*E`` — E the glacial law on the interior
    ARRIVAL slope, f the flotation ramp — integrated by the closed-form
    backward-Euler step :func:`_implicit_border_step`, so the bed approaches
    the flotation-draft equilibrium ``zb* = bl - hc*H + delta*U/E``
    monotonically at any dt. The waterline-flotation gate (``gate``/``ramp``)
    is the rho_i = rho_w effective-pressure law, applied to the interior
    erosion delta via :func:`_flot_factor` and inside the border step
    (``ramp = 0`` is the hard binary gate / its Filippov sliding mode).

    Parameters
    ----------
    zb_flat : ndarray
        Bedrock, flat — uplift already applied; MODIFIED IN PLACE.
    ice_flux, water_flux : ndarray
        Accumulated ice / water flux (m^3/yr), flattened. The interior
        flotation ramp, H-closure, erosion, the dominant-donor pick and the
        border icy/ice-free switch all read ``ice_flux``.
    H_flat : ndarray
        Ice thickness; previous-step value on input, MODIFIED IN PLACE.
    surface_out : ndarray
        Filled with zb + hc_over_H*H_new (the new ice surface) on return.
    law_code : int
        ``LAW_EFFEXP`` / ``LAW_POWER`` / ``LAW_COULOMB`` (see _core.solvers).
    p : GlacialParams
        Packed law/physics constants (see :mod:`siim._core.params`). Notable
        fields: ``D_H`` (H diffusivity [m^2/yr]; 0 disables, sub-stepped CFL)
        and ``hc_over_H`` (centerline-to-mean depth ratio; zs = zb + hc_over_H*H).
    border_bed_uplift : ndarray
        Flat (nn,) uplift rate (m/yr); only border (self-receiving) cells are
        read — the U in the icy border budget and the rate of the ice-free
        post-glacial recovery toward bl.
    bl : float
        Base level: the per-step water-line (Dirichlet) datum. Replaces the
        literal 0 at every waterline site — the ice-free-border erosion-view
        floor + lake-fill seed, the border recovery threshold, and the
        flotation reference. Default 0.0 (bit-for-bit with the historical
        hard-coded datum).
    gate : bool
        Waterline-flotation gate (default constants.FLOTATION_GATE = True):
        the rho_i = rho_w effective-pressure law. Interior: scales the erosion
        delta. Border: the physical bound inside the implicit budget. Off is
        for diagnostics only — it un-bounds the border (f == 1, measured
        runaway).
    ramp : float
        Flotation-ramp width gamma (default constants.FLOTATION_RAMP = 0.1):
        glacial erosion is scaled by
        ``f = clip((zs - bl)/(gamma*hc*H), 0, 1)`` instead of the hard on/off
        switch. 0 = the hard binary gate (interior bit-for-bit; at the border
        its implicit solution is the flotation sliding mode). Safe ceiling 0.2
        (see constants.FLOTATION_RAMP). Only active when ``gate`` is on.
    parallel_erode : bool
        Run the erosion step level-scheduled in parallel (topological levels
        of the flow graph; _core.routing._levels_sfr +
        eroders._erode_modeb_sfr_levels). BIT-FOR-BIT with the serial eroder
        at any thread count (disjoint writes, identical per-node arithmetic;
        pinned by test_parallel_erode). Default False.
    """
    Ko, Co, ce, n, nu, m, mu, cg, alpha_g, lambda_p, lambda_c, tau_c, coulomb_clamp, rho_g_g, hc_over_H, D_H = p
    nn = zb_flat.shape[0]

    # --- 0. Dominant donor per cell (the max-ice-flux cell draining into it).
    # A border reads dom for its zero-gradient thickness (step 1) and the
    # arrival slope of its implicit bed budget (step 5b). ---
    dom = np.full(nn, -1, dtype=np.int64)
    any_icy_border = False
    for i in range(nn):
        r = rec[i]
        if r == i:
            if ice_flux[i] > 0.0:
                any_icy_border = True
            continue
        if ice_flux[i] > 0.0 and (dom[r] < 0 or ice_flux[i] > ice_flux[dom[r]]):
            dom[r] = i

    # --- 1. Joint walk for H on zb (outlet-first; receiver visited before cell).
    # OUTFLOW border = zero-gradient thickness H_border = H_dominant_donor,
    # resolved by a 2-pass walk when any border is icy (the donor is visited
    # AFTER the border, so pass 1 uses the lagged border H as a bounded
    # provisional and pass 2 the corrected one — the 2D twin of siim1d's 2-pass
    # _diag_walk). Ice-free borders get H = 0. ---
    n_pass = 2 if any_icy_border else 1
    for _pass in range(n_pass):
        for inode in stack:
            r = rec[inode]
            if r == inode:
                # border: ice-free -> H = 0; icy -> leave provisional here and
                # set the zero-gradient thickness after the interior is solved.
                if ice_flux[inode] <= 0.0:
                    H_flat[inode] = 0.0
                continue
            if ice_flux[inode] <= 0.0:
                H_flat[inode] = 0.0
                continue
            L = lengths[inode]
            if L <= 0.0:
                H_flat[inode] = 0.0
                continue
            zs_r = zb_flat[r] + hc_over_H * H_flat[r]
            a = zb_flat[inode] - zs_r
            H_flat[inode] = _modeb_closure(
                law_code, False, a, ice_flux[inode], L, hc_over_H,
                cg, lambda_p, lambda_c, tau_c, rho_g_g, coulomb_clamp)
        if any_icy_border:
            for i in range(nn):
                if rec[i] == i and ice_flux[i] > 0.0 and dom[i] >= 0:
                    H_flat[i] = H_flat[dom[i]]      # zero-gradient thickness

    # --- 1b. Pre-diffusion H scrub: heal negative/NaN H from Newton edge cases
    # BEFORE diffusion, so one bad cell can't spread through the 5-point stencil
    # (the per-substep clamp catches negatives but not NaN; audit m18). ---
    for i in range(nn):
        if H_flat[i] < 0.0 or H_flat[i] != H_flat[i]:
            H_flat[i] = 0.0

    # --- 2. Optional H diffusion (seam-aware on looped axes; audit m16) ---
    _diffuse_H_2d(H_flat, ny, nx, dx_cell, dy_cell, D_H, dt, wrap_y, wrap_x)

    # --- 2b. Post-diffusion scrub: re-zero H at ice-free cells so diffusion
    # can't leave a phantom ice apron past the terminus (1D mode-B parity;
    # audit m15), and heal any residual negative/NaN before the filled view,
    # the width carve, or the output surface. ---
    for i in range(nn):
        if ice_flux[i] <= 0.0 or H_flat[i] < 0.0 or H_flat[i] != H_flat[i]:
            H_flat[i] = 0.0

    # --- 3. Build z' = zb + hc_over_H*H and lake-fill in place via stack walk.
    # An ICE-FREE border presents the WATER LINE max(.., bl) BEFORE the fill
    # (rho_i = rho_w: a still-water border below the datum is open water), so
    # the fill propagates the waterline across the submerged connected reach —
    # those nodes see delta = 0 and the rock problem grades to the datum. An
    # ICY (outflow) border is a free outflow: it keeps its true surface (the
    # through-flowing ice is not still base water), so the interior grades to
    # the real border surface, not to bl. ---
    z_filled = np.empty(nn)
    for i in range(nn):
        z_filled[i] = zb_flat[i] + hc_over_H * H_flat[i]
        if rec[i] == i and H_flat[i] <= 0.0 and z_filled[i] < bl:
            z_filled[i] = bl
    _lake_fill_sfr_2d(z_filled, stack, rec)
    z_pre = z_filled.copy()

    # --- 4. Erode the lake-filled view in place (per-law dispatch; serial
    # walk, or level-scheduled parallel under the parallel_erode toggle —
    # bit-for-bit twins). ---
    if parallel_erode:
        order, offsets, nlev = _levels_sfr(stack, rec)
        _erode_modeb_sfr_levels(law_code, z_filled, z_pre, water_flux,
                                ice_flux, H_flat,
                                Ko, Co, ce, n, nu, m, mu, cg, alpha_g, lambda_p,
                                lambda_c, tau_c, coulomb_clamp, rho_g_g,
                                dt, lengths, rec, order, offsets, nlev)
    else:
        _erode_modeb_sfr(law_code, z_filled, z_pre, water_flux, ice_flux,
                         H_flat,
                         Ko, Co, ce, n, nu, m, mu, cg, alpha_g, lambda_p,
                         lambda_c, tau_c, coulomb_clamp, rho_g_g,
                         dt, lengths, stack, rec)

    # --- 5. Erosion delta to zb (delta ≈ 0 inside lakes; carves persist).
    # Flotation gate: scale the glacial-erosion delta at an icy interior cell by
    # the effective-pressure ramp f (_flot_factor: f = 0 exactly for zs <= bl —
    # no erosion beneath the waterline; ramp = 0 = the hard binary gate). The
    # carve is throttled for free downstream: the scaled bed drop IS the source
    # E_c the carve reads, and an afloat source has E_c = 0 and carves nothing
    # (siim/_core/carve.py). ---
    for i in range(nn):
        delta = z_pre[i] - z_filled[i]
        if delta > 0.0:
            if gate and H_flat[i] > 0.0:
                f = _flot_factor(zb_flat[i] + hc_over_H * H_flat[i],
                                 H_flat[i], bl, hc_over_H, ramp)
                if f <= 0.0:
                    continue
                delta *= f
            zb_flat[i] -= delta

    # --- 5b. Border bed at base-level outlets (OUTFLOW BC): the IMPLICIT
    # BORDER BUDGET. An icy border bed
    # keeps ERODING — dzb/dt = U − f·E with E the glacial law on the ARRIVAL
    # slope (the interior flow slope one cell inside: the dominant donor's own
    # upstream surface slope, floored at S_FLOOR_BC — the ~0 local slope of
    # zero-gradient-H would starve it into a sill) and f the flotation ramp —
    # integrated by the CLOSED-FORM backward-Euler step (_implicit_border_step),
    # so the bed approaches the flotation-draft equilibrium
    # zb* = bl − hc·H + δ·U/E monotonically at any dt (no explicit overshoot,
    # no single-cell pits, no sill: the digging border always hands the donor a
    # real receiver drop). gate off → f ≡ 1, the unbounded control (diagnostics
    # only). Ice-free, a bed below the datum rises tectonically toward bl
    # (post-glacial recovery). ---
    for i in range(nn):
        if rec[i] != i:
            continue
        if ice_flux[i] > 0.0:
            d = dom[i]
            S_arr = _S_FLOOR
            if d >= 0:
                d2 = dom[d]
                if d2 >= 0 and lengths[d2] > 0.0:
                    s = (zb_flat[d2] + hc_over_H * H_flat[d2]
                         - zb_flat[d] - hc_over_H * H_flat[d]) / lengths[d2]
                    if s > S_arr:
                        S_arr = s
            E = _modeb_border_erosion(law_code, ice_flux[i], S_arr,
                                      H_flat[i], Co, mu, nu, ce, cg,
                                      alpha_g, lambda_p)
            if gate:
                zb_flat[i] = _implicit_border_step(
                    zb_flat[i], border_bed_uplift[i], E, dt,
                    H_flat[i], hc_over_H, bl, ramp)
            else:                    # unbounded control (diagnostics only)
                zb_flat[i] += (border_bed_uplift[i] - E) * dt
        elif zb_flat[i] < bl:
            zb_flat[i] = min(zb_flat[i] + border_bed_uplift[i] * dt, bl)

    # --- 6. Output surface = zb + hc_over_H*H — the TRUE state everywhere: a relict
    # drowned border bed shows through below bl, an icy through-flowing (outflow)
    # border stands at its true surface, and interior trough cells present their
    # drowned bed (mass balance melts at the real deep elevation, ice flux dies
    # crossing an empty trough). The ice-free-border floor + lake fill live ONLY
    # in the erosion working view (steps 3-4); water is a display layer at bl. ---
    for i in range(nn):
        surface_out[i] = zb_flat[i] + hc_over_H * H_flat[i]

@numba.njit(cache=True)
def _dinf_modeB_recv(zb_flat, H_flat, inode, nb_receivers, receivers,
                     weights, lengths, hc_over_H):
    """Effective receiver for the joint walk: a = zb_i − Z̄ and L̄ from the
    weighted receiver surfaces (receivers already solved — stack order).
    Returns (a, L_eff, ok)."""
    A = 0.0
    B = 0.0
    for k in range(nb_receivers[inode]):
        r = receivers[inode, k]
        L = lengths[inode, k]
        w = weights[inode, k]
        if L <= 0.0 or w <= 0.0 or r == inode:
            continue
        zs_r = zb_flat[r] + hc_over_H * H_flat[r]
        A += w / L
        B += w * zs_r / L
    if A <= 0.0:
        return 0.0, 0.0, False
    L_eff = 1.0 / A
    return zb_flat[inode] - B * L_eff, L_eff, True

@numba.njit(cache=True)
def _dinf_modeB_filled_view(zb_flat, H_flat, receivers, hc_over_H, ny, nx,
                            wrap_y, wrap_x, bl=_BL):
    """Erosion working view: z' = zb + hc_over_H*H with ICE-FREE borders floored
    at the water line bl, depression-filled by the flat (eps = 0) priority flood
    seeded at the borders — the 2D generalization of the SFR lake-fill stack
    walk. An ICY (outflow) border keeps its true surface (a free outflow, not
    still base water). Returns the filled view."""
    nn = zb_flat.shape[0]
    z_raw = np.empty(nn)
    interior = np.empty(nn, dtype=np.int8)
    for i in range(nn):
        z_raw[i] = zb_flat[i] + hc_over_H * H_flat[i]
        if receivers[i, 0] == i:
            interior[i] = 0
            if H_flat[i] <= 0.0 and z_raw[i] < bl:
                z_raw[i] = bl
        else:
            interior[i] = 1
    z_filled = np.empty(nn)
    _priority_flood_eps(z_raw, ny, nx, interior, 0.0, wrap_y, wrap_x,
                        z_filled)
    return z_filled

@numba.njit(cache=True)
def _glac_fast_solve_modeB_dinf(zb_flat, ice_flux, water_flux, H_flat,
                                surface_out, law_code, p,
                                dt, stack, nb_receivers, receivers, weights,
                                lengths,
                                ny, nx, dx_cell, dy_cell,
                                border_bed_uplift, wrap_y, wrap_x,
                                bl=_BL, gate=_FLOT_GATE, ramp=_FLOT_RAMP,
                                parallel_erode=False):
    """D-inf + mode B, law-agnostic skeleton.

    The D-inf twin of :func:`_glac_fast_solve_modeB_sfr` and the routing twin of
    :func:`_glac_fast_solve_modeA_dinf`: same six-step structure (joint walk,
    diffuse, filled view, erode, implicit border budget, output surface) over the
    D-inf graph. The joint walk visits the donor-first stack in REVERSE
    (receivers-first) and collapses the weighted receiver surfaces to a single
    effective receiver (:func:`_dinf_modeB_recv`, ``a = zb_i - Z̄`` and
    ``L̄``), so the per-law H-closure is the SAME :func:`_modeb_closure`
    used by SFR; erosion is the D-inf dispatch :func:`_erode_modeb_dinf`. See
    the D-inf mode-B block comment above. Thin wrappers pass their ``LAW_*``
    code and 0.0 for the inactive-law constants. ``zb_flat`` / ``H_flat`` are
    MODIFIED IN PLACE and ``surface_out`` is filled.

    OUTFLOW base-level ice BC (see the SFR twin): a through-flowing border
    gets zero-gradient thickness ``H_border = H_dominant_donor`` (max ice flux
    into it) and its bed erodes by the IMPLICIT BORDER BUDGET on the interior
    arrival slope (:func:`_implicit_border_step`, ramp-bounded at the
    flotation draft). The flotation ``gate``/``ramp`` is the rho_i = rho_w
    effective-pressure law (``ramp`` = gamma; 0 = the hard binary gate /
    sliding mode). ``bl`` is the per-step water-line datum (default 0.0 =
    bit-for-bit). ``parallel_erode``: level-scheduled parallel erosion step,
    bit-for-bit with the serial eroder (see the SFR twin). Default False.
    """
    Ko, Co, ce, n, nu, m, mu, cg, alpha_g, lambda_p, lambda_c, tau_c, coulomb_clamp, rho_g_g, hc_over_H, D_H = p
    nn = zb_flat.shape[0]

    # --- 0. Dominant donor per cell (max-ice-flux donor over its D-inf
    # receivers); borders read it for zero-gradient thickness (step 1) and
    # the arrival slope of the implicit bed budget (step 5b; see the SFR
    # twin). ---
    dom = np.full(nn, -1, dtype=np.int64)
    any_icy_border = False
    for i in range(nn):
        if ice_flux[i] <= 0.0:
            continue
        if receivers[i, 0] == i:
            any_icy_border = True
            continue
        for k in range(nb_receivers[i]):
            r = receivers[i, k]
            if r == i:
                continue
            if dom[r] < 0 or ice_flux[i] > ice_flux[dom[r]]:
                dom[r] = i

    # --- 1. joint walk (receivers-first = reversed donor-first stack). OUTFLOW
    # border = zero-gradient thickness from the dominant donor, resolved by the
    # same 2-pass scheme as the SFR twin (donor visited after the border). ---
    n_pass = 2 if any_icy_border else 1
    for _pass in range(n_pass):
        for idx in range(stack.shape[0] - 1, -1, -1):
            inode = stack[idx]
            if receivers[inode, 0] == inode:
                # border: ice-free -> H = 0; icy -> zero-gradient set below.
                if ice_flux[inode] <= 0.0:
                    H_flat[inode] = 0.0
                continue
            if ice_flux[inode] <= 0.0:
                H_flat[inode] = 0.0
                continue
            a, L_eff, ok = _dinf_modeB_recv(zb_flat, H_flat, inode, nb_receivers,
                                            receivers, weights, lengths, hc_over_H)
            if not ok:
                H_flat[inode] = 0.0
                continue
            H_flat[inode] = _modeb_closure(
                law_code, False, a, ice_flux[inode], L_eff, hc_over_H,
                cg, lambda_p, lambda_c, tau_c, rho_g_g, coulomb_clamp)
        if any_icy_border:
            for i in range(nn):
                if receivers[i, 0] == i and ice_flux[i] > 0.0 and dom[i] >= 0:
                    H_flat[i] = H_flat[dom[i]]      # zero-gradient thickness

    # --- 1b. pre-diffusion scrub: heal negative/NaN H before the stencil can
    # spread a bad cell (audit m18). ---
    for i in range(nn):
        if H_flat[i] < 0.0 or H_flat[i] != H_flat[i]:
            H_flat[i] = 0.0

    # --- 2. optional H diffusion (seam-aware on looped axes; m16) + post-scrub:
    # re-zero ice-free cells (no phantom apron past the terminus; 1D parity,
    # m15) and heal any residual negative/NaN. ---
    _diffuse_H_2d(H_flat, ny, nx, dx_cell, dy_cell, D_H, dt, wrap_y, wrap_x)
    for i in range(nn):
        if ice_flux[i] <= 0.0 or H_flat[i] < 0.0 or H_flat[i] != H_flat[i]:
            H_flat[i] = 0.0

    # --- 3. filled erosion view ---
    z_filled = _dinf_modeB_filled_view(zb_flat, H_flat, receivers, hc_over_H,
                                       ny, nx, wrap_y, wrap_x, bl)
    z_pre = z_filled.copy()

    # --- 4. erode the filled view in place (per-law D-inf dispatch; serial
    # walk, or level-scheduled parallel under the parallel_erode toggle —
    # bit-for-bit twins). ---
    if parallel_erode:
        order, offsets, nlev = _levels_dinf(stack, nb_receivers, receivers)
        _erode_modeb_dinf_levels(law_code, z_filled, z_pre, water_flux,
                                 ice_flux, H_flat,
                                 Ko, Co, ce, n, nu, m, mu, cg, alpha_g,
                                 lambda_p, lambda_c, tau_c, coulomb_clamp,
                                 rho_g_g, dt,
                                 nb_receivers, receivers, weights, lengths,
                                 order, offsets, nlev)
    else:
        _erode_modeb_dinf(law_code, z_filled, z_pre, water_flux, ice_flux,
                          H_flat,
                          Ko, Co, ce, n, nu, m, mu, cg, alpha_g, lambda_p,
                          lambda_c, tau_c, coulomb_clamp, rho_g_g, dt,
                          stack, nb_receivers, receivers, weights, lengths)

    # --- 5. erosion delta to zb (flotation gate: scale icy interior cells by
    # the effective-pressure ramp f — f = 0 exactly for zs <= bl, no glacial
    # erosion below the waterline; ramp = 0 = binary; see the SFR twin). ---
    for i in range(nn):
        delta = z_pre[i] - z_filled[i]
        if delta > 0.0:
            if gate and H_flat[i] > 0.0:
                f = _flot_factor(zb_flat[i] + hc_over_H * H_flat[i],
                                 H_flat[i], bl, hc_over_H, ramp)
                if f <= 0.0:
                    continue
                delta *= f
            zb_flat[i] -= delta

    # --- 5b. border bed: the IMPLICIT BORDER BUDGET on the arrival slope
    # (OUTFLOW BC; see _glac_fast_solve_modeB_sfr). The arrival slope is the
    # dominant donor's own upstream surface slope (one cell inside the border);
    # in D-inf the donor-to-dominant-donor edge length is looked up over the
    # inner donor's receivers. Closed-form implicit step, ramp-bounded at the
    # flotation draft. ---
    for i in range(nn):
        if receivers[i, 0] != i:
            continue
        if ice_flux[i] > 0.0:
            d = dom[i]
            S_arr = _S_FLOOR
            if d >= 0:
                d2 = dom[d]
                if d2 >= 0:
                    L2 = 0.0
                    for k in range(nb_receivers[d2]):
                        if receivers[d2, k] == d:
                            L2 = lengths[d2, k]
                            break
                    if L2 > 0.0:
                        s = (zb_flat[d2] + hc_over_H * H_flat[d2]
                             - zb_flat[d] - hc_over_H * H_flat[d]) / L2
                        if s > S_arr:
                            S_arr = s
            E = _modeb_border_erosion(law_code, ice_flux[i], S_arr,
                                      H_flat[i], Co, mu, nu, ce, cg,
                                      alpha_g, lambda_p)
            if gate:
                zb_flat[i] = _implicit_border_step(
                    zb_flat[i], border_bed_uplift[i], E, dt,
                    H_flat[i], hc_over_H, bl, ramp)
            else:                    # unbounded control (diagnostics only)
                zb_flat[i] += (border_bed_uplift[i] - E) * dt
        elif zb_flat[i] < bl:
            zb_flat[i] = min(zb_flat[i] + border_bed_uplift[i] * dt, bl)

    # --- 6. output surface = zb + hc_over_H*H, the TRUE state everywhere
    # (true-state output convention; see the SFR twin's step 6). ---
    for i in range(nn):
        surface_out[i] = zb_flat[i] + hc_over_H * H_flat[i]


@numba.njit(cache=True)
def _diag_walk(zb, Qg, H_out, law_code, p, dx, didx_l, didx_r, nx):
    """Mode-B 1D joint walk (H + surface), law-agnostic skeleton.

    The 1D twin of the 2D mode-B skeletons. Walk both sides of the divide from
    the base-level outlets inward, solving H and z_s = zb + hc_over_H*H jointly
    per node. The per-law H-closure is the SAME ``_modeb_closure`` the 2D kernels
    use — the from-a branch with cell length L = dx at interior nodes. A
    through-flowing base-level outlet is an OUTFLOW border with zero-gradient
    thickness ``H_outlet = H_interior_neighbour``, resolved by a 2-pass walk (the
    neighbour is solved after the outlet, so pass 1 uses the lagged outlet H as a
    bounded provisional and pass 2 the corrected one). Qg = 0 keeps H = 0. Only
    the closure dispatch varies by law; the outflow outlet BC and the Qg<=0
    terminus are law-agnostic and live here. ``H_out`` is MODIFIED IN PLACE (its
    incoming value is the previous-step provisional); the thin wrappers pass
    their ``LAW_*`` code and 0.0 for the inactive-law constants.

    The 1D border bed (the implicit arrival-slope budget) lives in
    ``siim.siim1d._erode_border_bed_1d`` — this walk solves only H.
    """
    Ko, Co, ce, n, nu, m, mu, cg, alpha_g, lambda_p, lambda_c, tau_c, clamp, rho_g_g, hc_over_H, D_H = p
    # left side: outlet at i=0; walk i=1..didx_l, receiver is i-1.
    if didx_l >= 0:
        icy = Qg[0] > 0.0
        if not icy:
            H_out[0] = 0.0
        for _pass in range(2 if icy else 1):
            for i in range(1, didx_l + 1):
                if Qg[i] <= 0.0:
                    H_out[i] = 0.0
                    continue
                zs_r = zb[i - 1] + hc_over_H * H_out[i - 1]
                a = zb[i] - zs_r
                H_out[i] = _modeb_closure(law_code, False, a, Qg[i], dx,
                                          hc_over_H, cg, lambda_p, lambda_c,
                                          tau_c, rho_g_g, clamp)
            if icy:
                H_out[0] = H_out[1]                  # zero-gradient thickness
    # right side: outlet at i=nx-1; walk i=nx-2..didx_r, receiver is i+1
    if didx_r < nx:
        icy = Qg[nx - 1] > 0.0
        if not icy:
            H_out[nx - 1] = 0.0
        for _pass in range(2 if icy else 1):
            for i in range(nx - 2, didx_r - 1, -1):
                if Qg[i] <= 0.0:
                    H_out[i] = 0.0
                    continue
                zs_r = zb[i + 1] + hc_over_H * H_out[i + 1]
                a = zb[i] - zs_r
                H_out[i] = _modeb_closure(law_code, False, a, Qg[i], dx,
                                          hc_over_H, cg, lambda_p, lambda_c,
                                          tau_c, rho_g_g, clamp)
            if icy:
                H_out[nx - 1] = H_out[nx - 2]        # zero-gradient thickness
