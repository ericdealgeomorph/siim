"""Framework-free composition-chain step functions for the siim 2D model.

Each ``run_step`` body of siim's own ``@xs.process`` classes
(:mod:`siim.fastscape.processes`) is extracted here as a ``self``-free
module-level function taking state + params explicitly. The xs.process classes
become thin shells that unpack ``self.*``, call the function, and assign the
outputs; the standalone in-house driver (:mod:`siim._core.driver`) calls the
same functions. One implementation, two front ends — no divergence.

This module stays numpy/numba/scipy-only (like the rest of :mod:`siim._core`):
it imports NOTHING from xsimlab / fastscape / siim.fastscape. The flexure plate
solve (the in-house spectral solve of :mod:`siim._core.flexure`) is injected as a
callable (:func:`glacial_flexure_step`'s ``flexure_solve`` argument).

The step-order and state-separation subtleties that must remain bit-for-bit
stable are summarized below:

* the ``ice_thickness`` one-step lag (router/accumulator see ``H(t-1)`` — an
  ordering artifact reproduced by reading H into the routing surface before the
  kernel overwrites it; at ``inner_passes > 1`` the staged graph/fluxes come from
  the final pass, routed on the last rehearsal iterate, so the lag is shortened,
  not removed);
* size-1 clock-sliced inputs de-squeezed inside the extracted fns
  (``bl = float(asarray(bl).ravel()[-1])``, ``bbu[0]`` on a ``(1, ny, nx)`` slice);
* the FIREWALL (measured twice): the mode-B kernel reconstructs its OWN
  ``zs = zb + hc*H`` from raw H, so no routing surface built here enters a
  closure. FIREWALL: see ``siim._core.driver``.
"""

import functools
import math

import numba
import numpy as np

from ..constants import GRAVITY, RHO_ICE, MFD_GAMMA_FLOOR as _GAMMA_FLOOR
from .. import constants as _constants
from .params import GlacialParams
from .routing import (
    _flow_accumulate_sd, _flow_accumulate_sd_2,
    _flow_accumulate_dinf, _flow_accumulate_dinf_2,
    _priority_flood_eps, _dinf_route, _dinf_topo_stack, _dinf_pack,
    _DINF_E1_DJ, _DINF_E1_DI, _DINF_E2_DJ, _DINF_E2_DI,
    _d8_receivers, _d8_stack, _d8_basin, d8_interior_mask,
    _recv_mean_direction, gamma_scaled_lengths,
    _mfd_route, _mfd_stack, _MFD_DJ, _MFD_DI,
)
from .skeleton import (
    _glac_fast_solve_modeA_sfr, _glac_fast_solve_modeA_dinf,
    _glac_fast_solve_modeB_sfr, _glac_fast_solve_modeB_dinf,
)
from .solvers import (
    LAW_EFFEXP, LAW_POWER, LAW_COULOMB,
    _modeb_closure,
)
from .hillslope import _fixed_flags
from .threads import level_threads
from .carve import (
    _power_dt_2d, _power_dt_2d_periodic, _carve_offsets, _carve_subgrid_width,
)


# ---------------------------------------------------------------------------
# 1. Law record (GlacialLaw.initialize + _glacial_params_and_code)
# ---------------------------------------------------------------------------
def build_glacial_params(*, sliding_law, Ko, ce, n, nu, m, mu, Ac, alpha_g,
                         lambda_p, lambda_c, tau_c, coulomb_clamp, hc_over_H,
                         H_diffusivity):
    """(law_code, GlacialParams) for the given sliding law — the frozen per-run
    scalars the law_code step skeletons consume. Body of ``GlacialLaw.initialize``
    + ``_glacial_params_and_code``. Validates ``sliding_law`` and ``hc_over_H``;
    ``m`` defaults to ``n/2`` and ``mu`` to the per-law ``constants.derive_*``."""
    if sliding_law not in ('eff-exp', 'power', 'coulomb'):
        raise ValueError(
            f"Unknown sliding_law: '{sliding_law}'. "
            f"Options: 'eff-exp', 'power', 'coulomb'")
    lambda_p = float(lambda_p)
    lambda_c = float(lambda_c) if lambda_c is not None else _constants.LAMBDA_C
    tau_c = float(tau_c)
    coulomb_clamp = float(coulomb_clamp)
    rho_g = RHO_ICE
    g = GRAVITY
    rho_g_g = rho_g * g
    # cg via the single-source rheology helper (kt absorbed; audit m36).
    cg = _constants.cg_prefactor(float(alpha_g), float(Ac), rho_g, g)
    # mu: the siim2d wrapper always passes a per-law derived value; standalone
    # use falls back to the SAME per-law relations via constants.derive_*
    # (single-sourced; audit m35). Co (eff-exp solver) via Co_power with the
    # effective mu — an explicit override wins, and Co tracks it (B5).
    if mu is not None:
        mu_val = float(mu)
    elif sliding_law == 'coulomb':
        mu_val = _constants.derive_coulomb(
            float(ce), float(alpha_g), tau_c, rho_g, g, nu=float(nu)).mu
    else:  # power, eff-exp
        mu_val = _constants.derive_power(
            float(ce), cg, lambda_p, float(alpha_g), nu=float(nu)).mu
    Co = _constants.Co_power(float(ce), cg, lambda_p, float(alpha_g), mu_val)
    D_H = 0.0 if H_diffusivity is None else float(H_diffusivity)
    hc = float(hc_over_H)
    if not hc > 0.0:
        raise ValueError(f"hc_over_H must be > 0, got {hc_over_H!r}")

    Ko = float(Ko)
    n = float(n)
    nu = float(nu)
    m_val = float(m) if m is not None else n / 2.0   # default n/2 (matches siim1d/2d)
    ag = float(alpha_g)
    if sliding_law == 'power':
        return LAW_POWER, GlacialParams(
            Ko=Ko, ce=float(ce), n=n, nu=nu, m=m_val, cg=cg,
            alpha_g=ag, lambda_p=lambda_p,
            hc_over_H=hc, D_H=D_H)
    elif sliding_law == 'coulomb':
        return LAW_COULOMB, GlacialParams(
            Ko=Ko, ce=float(ce), n=n, nu=nu, m=m_val, cg=cg,
            alpha_g=ag, lambda_c=lambda_c, tau_c=tau_c,
            coulomb_clamp=coulomb_clamp, rho_g_g=rho_g_g,
            hc_over_H=hc, D_H=D_H)
    return LAW_EFFEXP, GlacialParams(
        Ko=Ko, Co=Co, n=n, nu=nu, m=m_val, mu=mu_val, cg=cg,
        alpha_g=ag, lambda_p=lambda_p,
        hc_over_H=hc, D_H=D_H)


# ---------------------------------------------------------------------------
# 2. Initial topography (InitialTopography.initialize)
# ---------------------------------------------------------------------------
def initial_topography(elevation_init, shape, border_status, seed,
                       noise_amplitude):
    """Initial surface = ``elevation_init`` + uniform tie-breaking noise, zeroed
    on 'fixed_value' edges. RNG seeded here (init-only; no per-step RNG)."""
    if seed is not None:
        seed = None if np.isnan(float(seed)) else int(seed)
    else:
        seed = None
    rs = np.random.RandomState(seed=seed)
    if noise_amplitude is None:
        noise_scale = 0.1 * np.max(elevation_init)
    else:
        noise_scale = float(noise_amplitude)
    noise = noise_scale * rs.rand(*shape)
    bs = list(np.broadcast_to(border_status, 4))   # a scalar names all four sides
    if bs[0] == "fixed_value": noise[:,  0] = 0.0
    if bs[1] == "fixed_value": noise[:, -1] = 0.0
    if bs[2] == "fixed_value": noise[0,  :] = 0.0
    if bs[3] == "fixed_value": noise[-1, :] = 0.0
    return elevation_init + noise


# ---------------------------------------------------------------------------
# 2b. Escarpment forcing (framework-free bodies of siim.fastscape.forcing;
#     OQ-4 — the moving-Gaussian WaveUplift + the arctan PlateauSurface. The
#     xs.process classes in forcing.py become shells over these, so escarpment's
#     three override seams work under BOTH the xsimlab and the in-house drivers.)
# ---------------------------------------------------------------------------
def plateau_profile(x, plateau_zo, plateau_dz, plateau_frac, plateau_w):
    """1-D arctan-smoothed plateau profile on the coordinate ``x``: exactly 0 at
    ``x[0]``, rising across an escarpment centred at
    ``x_esc = (1 - plateau_frac)*Lx`` with transition width ``plateau_w``, then
    dropping ``plateau_dz`` linearly to exactly ``plateau_zo - plateau_dz`` at
    ``x[-1]``. The raw arctan ramp is rescaled so the domain ends land on 0 / 1
    (the raw ramp only reaches them asymptotically — with the default
    frac/w on a 50 km domain the low edge sat at 0.25*zo, a permanent sill
    above the border's water datum)."""
    x = np.asarray(x, dtype=float)
    Lx = float(x[-1] - x[0])
    x_esc = (1.0 - plateau_frac) * Lx
    x_or = x - float(x[0])
    ramp = (1.0 + 2.0 / np.pi * np.arctan((x_or - x_esc) / plateau_w)) / 2.0
    ramp = (ramp - ramp[0]) / (ramp[-1] - ramp[0])
    slope = plateau_dz * np.clip(x_or - x_esc, 0.0, None) / max(Lx - x_esc, 1e-9)
    return ramp * (plateau_zo - slope)


def plateau_surface(x, y, shape, border_status, seed, noise_amplitude,
                    plateau_zo, plateau_dz, plateau_frac, plateau_w):
    """Arctan-smoothed plateau initial topography (:func:`plateau_profile`
    tiled along y) + tie-breaking noise (zeroed on 'fixed_value' edges, so a
    fixed x-border starts exactly on 0 / ``plateau_zo - plateau_dz``). Body of
    ``PlateauSurface.initialize``."""
    topo = np.tile(plateau_profile(x, plateau_zo, plateau_dz, plateau_frac,
                                   plateau_w), (len(y), 1))

    if seed is not None and not (isinstance(seed, float) and np.isnan(seed)):
        seed = int(seed)
    else:
        seed = None
    rs = np.random.RandomState(seed=seed)
    if noise_amplitude is None:
        noise_scale = 0.1 * np.max(topo)
    else:
        noise_scale = float(noise_amplitude)
    noise = noise_scale * rs.rand(*shape)
    bs = list(np.broadcast_to(border_status, 4))   # a scalar names all four sides
    if bs[0] == "fixed_value": noise[:,  0] = 0.0
    if bs[1] == "fixed_value": noise[:, -1] = 0.0
    if bs[2] == "fixed_value": noise[0,  :] = 0.0
    if bs[3] == "fixed_value": noise[-1, :] = 0.0
    return topo + noise


def wave_rate(x, y, dt, t, delta_h, wave_width, wave_velocity, x_escarpment,
              wave_calibration, U_inf):
    """The moving-Gaussian wave's local uplift RATE (m/yr, every cell),
    midpoint-sampled (``t + dt/2``) with calibration 1.0 = exact ``delta_h``
    deposition over the passage. ``t`` is the time since the run's start, so
    ``x_escarpment`` is the wave's initial position. Also the default border-bed
    uplift under the wave (the border follows the wave; audit #25)."""
    time = t + 0.5 * dt
    wave_center = float(x[0]) + x_escarpment + time * wave_velocity
    X, _Y = np.meshgrid(x, y)
    U0 = (wave_calibration * delta_h * wave_velocity
          / (wave_width * np.sqrt(np.pi)))
    return U_inf + U0 * np.exp(-(X - wave_center) ** 2 / wave_width ** 2)


# ---------------------------------------------------------------------------
# 3. Block uplift (BlockUplift.initialize mask + GlacialBlockUplift.run_step)
# ---------------------------------------------------------------------------
#: Border ring of each domain edge, indexed in ``border_status`` order
#: (:data:`siim._core.outputs.SIDES` = left, right, bottom, top): index 2 is
#: row 0 (y = 0), index 3 is row ny-1.
SIDE_SLICES = ((slice(None), 0), (slice(None), -1),
               (0, slice(None)), (-1, slice(None)))


def uplift_mask(border_status, shape):
    """Binary uplift mask: 0 on 'fixed_value' border rings, 1 elsewhere. Body of
    fastscape ``BlockUplift.initialize`` (the mask half)."""
    mask = np.ones(shape)
    for status, border in zip(np.broadcast_to(border_status, 4), SIDE_SLICES):
        if status == "fixed_value":
            mask[border] = 0.0
    return mask


def block_rate(rate, shape):
    """The block uplift rate on the grid, unmasked (m/yr): drops the leading
    size-1 tstep dim xsimlab leaves on a ``(nt, y, x)`` slice and broadcasts.
    Also the default border-bed uplift (the local tectonic rate)."""
    rate = np.asarray(rate)
    if rate.ndim == 3:
        rate = rate[0]
    return np.broadcast_to(rate, shape)


def block_uplift(rate, dt, mask, shape):
    """``uplift = rate * dt``, zeroed on fixed borders by ``mask``. Body of
    ``GlacialBlockUplift.run_step``."""
    return block_rate(rate, shape) * mask * dt


# ---------------------------------------------------------------------------
# 4. Routing surface (GlacialSurfaceToErode: hc reconstruction)
# ---------------------------------------------------------------------------
def routing_surface(post_uplift_surface, hc_over_H, H):
    """The mode-B routing/mass-balance surface ``zs = post_uplift_bed + hc*H``
    (the reconstructed ice column on the post-uplift bed). ``H`` is the lagged
    thickness (the driver's pass iterate);
    ``hc_over_H`` the scalar datum or, in mode D (valley <-> sheet map), the
    per-cell ``hc_cell`` field. It feeds the router, the mass balance and the
    hillslope diffusion, never a closure. FIREWALL: see ``siim._core.driver``."""
    hc = hc_over_H if np.ndim(hc_over_H) else float(hc_over_H)
    return post_uplift_surface + hc * H


def _triangular_axis(g, m, axis, wrap=False):
    """``sum_k (m+1-|k-i|) g(k)``, k in ``[max(i-m, 0), min(i+m, n-1)]``, along
    ``axis``: the 2m+1 weighted shifts, added term by term (O(m) whole-grid adds
    per level, every term of one sign). Truncation at the domain edge is just
    the clipped slice. On a looped axis (``wrap``) the shifts are periodic
    rolls instead, in the same order, so a seam shift commutes with the sum
    exactly; there the half-width is capped at ``(n-1)//2`` so no cell enters
    the window twice. Prefix sums of g and of k*g close the same
    weights in O(1) per cell, but differencing them cancels ~1e-7 m out of a
    3 km surface on a 1000^2 grid — 100x the router's fill increment, enough to
    flip receiver rows on a FLAT surface — so the straight sum is the one kept.
    With :func:`_triangular_level` it is now the tests' reference: the filter
    ships the per-cell kernel :func:`_triangular_mean_at`, pinned to these sums
    to the last bit (test_sheet_knobs.py::test_filter_is_the_level_sums)."""
    g = np.moveaxis(np.asarray(g, dtype=np.float64), axis, 0)
    n, out = g.shape[0], np.zeros(g.shape)
    if wrap:
        m = min(m, (n - 1) // 2)
        for d in range(-m, m + 1):
            out += (m + 1 - abs(d)) * np.roll(g, -d, axis=0)   # sources (i + d) mod n
        return np.moveaxis(out, 0, axis)
    for d in range(-m, m + 1):
        lo, hi = max(d, 0), min(n + d, n)          # sources k = i + d, clipped
        if hi <= lo:                                # |d| >= n: no sources at all
            continue
        out[lo - d:hi - d] += (m + 1 - abs(d)) * g[lo:hi]
    return np.moveaxis(out, 0, axis)


def _triangular_level(field, H, H_field, m, wrap_y=False, wrap_x=False):
    """Numerator and denominator grids of the thickness-weighted triangular mean
    of ``field`` at integer half-width m: weights
    ``(1 - |di|/(m+1))*(1 - |dj|/(m+1))*H[i+di, j+dj]``, with ``H_field`` the
    precomputed ``H*field``. Separable, so the truncated 2-D sum factorises into
    two 1-D passes (the common factor ``(m+1)^2`` cancels in the ratio). The
    caller divides at the cells it needs; ``m = 0`` is the cell itself, so the
    numerator is ``field`` and there is nothing to divide by."""
    if m == 0:
        return field, None
    return (_triangular_axis(_triangular_axis(H_field, m, 0, wrap_y), m, 1, wrap_x),
            _triangular_axis(_triangular_axis(H, m, 0, wrap_y), m, 1, wrap_x))


def filter_routing_surface(zs, H, dx, factor, wrap_y=False, wrap_x=False):
    """The ROUTER's surface: a thickness-weighted TRIANGULAR spatial mean of the
    routing surface ``zs``, over a window of half-width
    ``max(factor*H/dx - 1/2, 0)`` cells set by the LOCAL thickness (about
    ``factor`` ice thicknesses). A fractional half-width interpolates linearly
    between the two bracketing integer levels; dry cells and cells thinner than
    ``dx/(2*factor)`` keep their raw ``zs``. Windows wrap on looped axes
    (``wrap_y`` / ``wrap_x``, half-width capped at ``(n-1)//2`` there) and
    truncate at the other edges. Kernel, target and ``factor = 2`` are the
    settled choices of ``docs/dev/sheet_smoother_choice.md``.

    The returned surface constructs the ROUTER INPUT alone (FIREWALL: see
    ``siim._core.driver``). Smoothing the routing surface redistributes Q and
    hence the diagnosed H, so a filtered run is a different model, not a
    filtered view of the same one;
    the graph can send flux along an edge that is uphill on the unfiltered
    surface (accumulation follows the graph, so nothing is lost)."""
    # One window per filtered cell (:func:`_filter_kernel`): (2m + 1)^2 terms at
    # half-width m, rows in parallel, the floats of the whole-grid level sums
    # (:func:`_triangular_level`) to the last bit.
    # dx/factor are validated at construction (router.check_routing_filter_config)
    # and H is the driver's own state; nothing is re-checked per call.
    zs = np.ascontiguousarray(zs, dtype=np.float64)
    H = np.ascontiguousarray(H, dtype=np.float64)
    out = np.empty(zs.shape)
    # Average the DEVIATION from a constant reference and add it back: the sums
    # then carry relief, not elevation, so a flat surface comes back EXACTLY
    # (same weights either way — this is where the last ULPs go).
    _filter_kernel(zs, H, float(dx), float(factor), float(max(H.shape)),
                   zs.min(), bool(wrap_y), bool(wrap_x),
                   np.empty(zs.shape), np.empty(zs.shape), out)
    return out


@numba.njit(cache=True, inline='always')
def _triangular_mean_at(dev, H, weighted, i, j, m, ny, nx, wrap_y, wrap_x):
    """:func:`_triangular_level` at ONE cell, numerator over denominator: the
    same terms added in the same order (each column's y sum first, from zero in
    d = -m..m, then the x sum of those), so the two agree to the last bit. A
    half-width is capped at ``(n-1)//2`` on a looped axis, clipped at the edge
    of the others; ``m = 0`` is the cell itself."""
    if m == 0:
        return dev[i, j]
    my = min(m, (ny - 1) // 2) if wrap_y else m
    mx = min(m, (nx - 1) // 2) if wrap_x else m
    num = 0.0
    den = 0.0
    for b in range(-mx, mx + 1):
        jj = j + b
        if wrap_x:
            jj = jj % nx
        elif jj < 0 or jj >= nx:
            continue
        col_num = 0.0
        col_den = 0.0
        for a in range(-my, my + 1):
            ii = i + a
            if wrap_y:
                ii = ii % ny
            elif ii < 0 or ii >= ny:
                continue
            w = my + 1 - abs(a)
            col_num += w * weighted[ii, jj]
            col_den += w * H[ii, jj]
        w = mx + 1 - abs(b)
        num += w * col_num
        den += w * col_den
    # den > 0: the cell's own H is > dx/(2*factor) > 0.
    return num / den


@numba.njit(cache=True, parallel=True)
def _filter_kernel(zs, H, dx, factor, radius_cap, zs0, wrap_y, wrap_x,
                   dev, weighted, out):
    """Fill ``out`` with :func:`filter_routing_surface`'s surface. ``dev``
    (``zs - zs0``) and ``weighted`` (``H*dev``) are (ny, nx) scratch. Rows in
    parallel: a cell reads ``H``, ``dev`` and ``weighted`` and writes its own
    ``out`` alone, so the result does not depend on the thread count."""
    ny, nx = zs.shape
    for i in numba.prange(ny):
        for j in range(nx):
            dev[i, j] = zs[i, j] - zs0
            weighted[i, j] = H[i, j] * dev[i, j]
            out[i, j] = zs[i, j]
    for i in numba.prange(ny):
        for j in range(nx):
            radius = min(max(factor * H[i, j] / dx - 0.5, 0.0), radius_cap)
            if not radius > 0.0:
                continue
            lower = int(math.floor(radius))
            a = _triangular_mean_at(dev, H, weighted, i, j, lower, ny, nx,
                                    wrap_y, wrap_x)
            b = _triangular_mean_at(dev, H, weighted, i, j, lower + 1, ny, nx,
                                    wrap_y, wrap_x)
            out[i, j] = zs0 + (a + (radius - lower) * (b - a))


# ---------------------------------------------------------------------------
# 5. Fabricated trunk routing surface (relocated verbatim; already pure)
# ---------------------------------------------------------------------------
def _fabricate_trunk_surface(zs_dyn, zb, H_lag, border, alpha_g, dx, dy,
                             k_dip, floor, offsets, D, SRC, wrap_y, wrap_x,
                             s_conf=None):
    """Build the fabricated trunk routing surface described in
    ``docs/dev/trunk_surface_routing.md``. Pure function (no process state) so
    the process and the channel-persistence test share one implementation.

    ``zs_dyn`` (ny, nx) is the dynamic ice surface ``zb + hc*H_lag``; ``zb`` the
    matching bed; ``H_lag`` the lagged thickness; ``border`` (ny, nx bool) the
    base-level edges (excluded as seeds + never fabricated). ``offsets/D/SRC``
    are (ny, nx) scratch buffers. Returns a fresh (ny, nx) elevation: the
    V-dipped trunk surface at footprint cells (``max(zs_geo, zb)``), ``zs_dyn``
    elsewhere. ``s_conf`` (``(ny, nx)``, mode D's valley <-> sheet
    confinedness, ``constants.HC_SHEET``) blends each footprint cell by
    its SOURCE's s — ``V*s + zs_dyn*(1 - s)`` — so a trunk (s = 1) keeps its
    whole V, flanks included, and a sheet source (s = 0) fabricates nothing;
    ``None`` is today's V.
    """
    ny, nx = zs_dyn.shape
    nn = ny * nx
    cell_scale = min(dx, dy)
    zs_flat = zs_dyn.ravel()
    zb_flat = zb.ravel()
    border_flat = border.ravel()
    idx = np.arange(nn)

    # Attribution on lagged H (thickest disc wins): seeds = icy ∧ non-border.
    rec_marker = np.where(border_flat, idx, -1)
    seed_mask = (~border_flat).astype(np.int8)
    n_seed = _carve_offsets(H_lag.ravel(), rec_marker, float(alpha_g),
                            offsets.ravel(), seed_mask)
    if n_seed == 0:
        return zs_dyn.copy()
    if wrap_x or wrap_y:
        _power_dt_2d_periodic(offsets, dy, dx, D, SRC, wrap_y, wrap_x)
    else:
        _power_dt_2d(offsets, dy, dx, D, SRC)
    Df = D.ravel()
    SRCf = SRC.ravel()

    # Cross-slope per source: S_c = k_dip * max(|grad zs_dyn|, floor). |grad| at a
    # trunk-bottom source ~= the down-valley slope (cross-valley ~0 there);
    # over-estimating S_c only aids convergence, so grad magnitude is the safe
    # estimator.
    gy, gx = np.gradient(zs_dyn, dy, dx)
    # np.gradient is one-sided at the edges; on a looped axis the edge rows /
    # columns have neighbours across the seam, so take the same central
    # difference there (the V was seam-position dependent, audit #0)
    if wrap_y:
        gy = (np.roll(zs_dyn, -1, 0) - np.roll(zs_dyn, 1, 0)) / (2.0 * dy)
    if wrap_x:
        gx = (np.roll(zs_dyn, -1, 1) - np.roll(zs_dyn, 1, 1)) / (2.0 * dx)
    S_c = float(k_dip) * np.maximum(np.hypot(gy, gx).ravel(), float(floor))

    member = (Df < 0.0) & (~border_flat)
    mem_idx = np.nonzero(member)[0]
    elev = zs_flat.copy()
    if mem_idx.size == 0:
        return elev.reshape(ny, nx)
    s = SRCf[mem_idx]
    R2s = -offsets.ravel()[s]                     # R_s^2 at the source
    Rs = np.sqrt(R2s)
    gate = (s >= 0) & (Rs > cell_scale)           # sub-cell sources: no trunk
    mem_idx = mem_idx[gate]
    if mem_idx.size == 0:
        return elev.reshape(ny, nx)
    s = s[gate]
    d = np.sqrt(np.maximum(Df[mem_idx] + R2s[gate], 0.0))
    zs_geo = zs_flat[s] + S_c[s] * (d - Rs[gate])
    geo = np.maximum(zs_geo, zb_flat[mem_idx])
    if s_conf is None:
        elev[mem_idx] = geo
    else:
        s_src = np.asarray(s_conf, dtype=np.float64).ravel()[s]
        elev[mem_idx] = geo * s_src + zs_flat[mem_idx] * (1.0 - s_src)
    return elev.reshape(ny, nx)


# ---------------------------------------------------------------------------
# 6. Glacial flow accumulation (GlacialFlowAccumulator.run_step)
# ---------------------------------------------------------------------------
def _ice_source(z_clim, zELA, beta, area, field):
    """The ice accumulation source ``beta*(z_clim - zELA)*area``, capped at the
    cell's precipitation flux ``field``: one formula for
    :func:`accumulate_glacial_flow` and :func:`accumulate_sheet_ice_flow`."""
    field_ice = beta * (z_clim - zELA) * area
    return np.where(field_ice > field, field, field_ice)


def accumulate_glacial_flow(surface, surface_upward, zELA, beta, runoff,
                            cell_area, width_hack_k, width_hack_p, shape,
                            stack, receivers, nb_receivers, weights, lengths,
                            basin, s_conf=None):
    """Accumulate water flux, ice flux, drainage area + the routing-topology
    outputs (basin_ids, receivers_2d, stack_2d; all three None when ``basin``
    is None, which the in-house driver passes on a step no frame saves). Body of
    ``GlacialFlowAccumulator.run_step``. ``surface`` is the post-uplift routing
    surface (``zs_route``); the ELA-relative mass balance ``b(z)`` is evaluated
    on the PRE-uplift climate surface ``z_clim = surface - surface_upward`` (no
    O(U*dt) bias). ``s_conf`` (``(ny, nx)``, the per-cell confinedness of
    mode D's valley <-> sheet map, ``constants.HC_SHEET``) fades the below-ELA
    ablation footprint from the Hack width (s = 1) to the cell (s = 0); ``None``
    is today's footprint. Returns ``(ice_flux, water_flux, area, basin_ids,
    receivers_2d, stack_2d)`` (caller also sets ``flowacc = water_flux``)."""
    zELA = np.broadcast_to(zELA, shape)
    field = np.broadcast_to(runoff * cell_area, shape)
    z_clim = surface - np.broadcast_to(surface_upward, shape)

    # D-inf outputs are 2D — receivers, lengths, weights are all
    # (n_nodes, nb_rec_max). SFR outputs are 1D. Dispatch on ndim.
    is_dinf = receivers.ndim == 2
    if is_dinf:
        nb_rec = np.asarray(nb_receivers, dtype=np.int64)
        recs   = np.asarray(receivers,   dtype=np.int64)
        wts    = np.asarray(weights,     dtype=np.float64)

    # 1. Drainage area — accumulated once and reused for both glacier_width
    #    and the area output.
    field_area = np.broadcast_to(cell_area, shape).flatten().copy()
    if is_dinf:
        _flow_accumulate_dinf(field_area, stack, nb_rec, recs, wts)
    else:
        _flow_accumulate_sd(field_area, stack, receivers)

    # 2. Sub-grid glacier plan-view area for the width-aware ablation.
    if is_dinf:
        lengths_2d = (lengths * weights).sum(axis=1).reshape(shape)
    else:
        lengths_2d = lengths.reshape(shape)
    glacier_width = float(width_hack_k) * field_area.reshape(shape) ** float(width_hack_p)
    glacier_area  = glacier_width * lengths_2d
    wide_area     = np.maximum(glacier_area, cell_area)
    # Above ELA, snow falls on the cell, not the glacier's footprint. In mode D
    # the below-ELA footprint is the convex blend wide*s + cell*(1 - s)
    # (exact at both ends): a trunk ablates over its Hack width, a sheet cell
    # over its own cell.
    below = (wide_area if s_conf is None
             else wide_area * s_conf + cell_area * (1.0 - s_conf))
    accum_area = np.where(z_clim < zELA, below, cell_area)

    # 3. Source field for ice accumulation, capped at the cell's precip flux.
    field_ice = _ice_source(z_clim, zELA, beta, accum_area, field)

    field     = field.flatten()
    field_ice = field_ice.flatten()

    # 4. Accumulate water + ice through the flow graph.
    if is_dinf:
        _flow_accumulate_dinf_2(field, field_ice, stack, nb_rec, recs, wts)
    else:
        _flow_accumulate_sd_2(field, field_ice, stack, receivers)

    # Clamp ice flux (nodes below ELA with no upstream contribution can go neg).
    np.maximum(field_ice, 0.0, out=field_ice)

    ice_flux = field_ice.reshape(shape)
    water_flux = field.reshape(shape) - ice_flux
    area = field_area.reshape(shape)
    if basin is None:
        return ice_flux, water_flux, area, None, None, None
    basin_ids = basin.astype(np.int32, copy=False)
    # For D-inf, output the "primary" receiver per cell (largest-weight one) so
    # downstream code keeps a single-receiver view; cells with no active
    # receivers get receiver = self (SFR self-receiving convention).
    if is_dinf:
        recs_2d = np.asarray(receivers)
        wts_2d  = np.asarray(weights)
        n_nodes = recs_2d.shape[0]
        primary_k = np.argmax(wts_2d, axis=1)
        primary_rec = recs_2d[np.arange(n_nodes), primary_k].astype(np.int32)
        self_idx = np.arange(n_nodes, dtype=np.int32)
        no_flow = (wts_2d.sum(axis=1) <= 0.0) | (primary_rec < 0)
        primary_rec = np.where(no_flow, self_idx, primary_rec)
        receivers_2d = primary_rec.reshape(shape)
    else:
        receivers_2d = receivers.reshape(shape).astype(np.int32, copy=False)
    stack_2d = stack.reshape(shape).astype(np.int32, copy=False)
    return ice_flux, water_flux, area, basin_ids, receivers_2d, stack_2d


def accumulate_sheet_ice_flow(surface, surface_upward, zELA, beta, runoff,
                              cell_area, shape, stack, receivers, nb_receivers,
                              weights):
    """The ice flux of :func:`accumulate_glacial_flow` at ``s_conf = 0`` (every
    cell a sheet cell), alone and to the last bit. There the below-ELA
    footprint ``wide*0 + cell_area*(1 - 0)`` is the cell exactly, so the
    drainage area, the Hack width and the water flux do not enter: one source
    field, one accumulation. Mode D's diagnostic pass
    (``siim._core.driver._diagnostic_discharge``) reads nothing else."""
    zELA = np.broadcast_to(zELA, shape)
    field = np.broadcast_to(runoff * cell_area, shape)
    z_clim = surface - np.broadcast_to(surface_upward, shape)
    field_ice = _ice_source(z_clim, zELA, beta, cell_area, field).flatten()
    if receivers.ndim == 2:
        _flow_accumulate_dinf(field_ice, stack,
                              np.asarray(nb_receivers, dtype=np.int64),
                              np.asarray(receivers, dtype=np.int64),
                              np.asarray(weights, dtype=np.float64))
    else:
        _flow_accumulate_sd(field_ice, stack, receivers)
    np.maximum(field_ice, 0.0, out=field_ice)
    return field_ice.reshape(shape)


# ---------------------------------------------------------------------------
# 7. Mode A step (GlacialSPLModeA: kernel + border-H closure + commit)
# ---------------------------------------------------------------------------
@numba.njit(cache=True)
def _strip_columns(lo, hi, g, c, s_lo, s_hi, d):
    """The column offsets o in [lo, hi] whose ``o*d*g + c`` can lie in
    [s_lo, s_hi], widened by one column each way for rounding and clipped to
    [lo, hi]; empty (lo > hi) when none can. Conservative wherever exact
    arithmetic is not: g == 0 keeps the row or empties it, a NaN keeps it, and
    a huge bound (tiny g) is compared before it is cast."""
    if g == 0.0:
        if c < s_lo or c > s_hi:
            return 1, 0
        return lo, hi
    a = (s_lo - c) / g / d
    b = (s_hi - c) / g / d
    if a > b:
        a, b = b, a
    if a > hi or b < lo:
        return 1, 0
    if a > lo:
        lo = max(lo, int(math.floor(a)) - 1)
    if b < hi:
        hi = min(hi, int(math.ceil(b)) + 1)
    return lo, hi


@numba.njit(cache=True, parallel=True)
def _transect_flux_kernel(Q, H_lag, receivers, weights, nb_receivers, ny, nx,
                          dx, dy, alpha_g, wrap_y, wrap_x, min_rule, Qt):
    """Per-cell transect flux (EXPERIMENTAL closure_flux='transect').

    For every icy cell i (Q[i] > 0): sum the flux of every routing link that
    crosses, in the flow direction, the line through i's centre perpendicular
    to i's own flow direction, within a half-length R = alpha_g*H_lag/2 (the
    closure's channel half-width). The line is offset by eps downstream of the
    centre so i's OUTGOING links cross it and its incoming links do not. Link
    weights ramp linearly from 1 at ``|t| <= R - w/2`` to 0 at ``|t| >= R + w/2``
    (w = min(dx, dy)); on a uniform cardinal sheet that ramp sums to exactly
    2R for any R. Result floored at Q[i] (thin ice: the cell alone).

    ``receivers`` / ``weights`` are (nn, k); ``nb_receivers`` (nn,). Flat
    arrays; Qt (nn,) is written in place (0 where Q <= 0). ``min_rule``: a
    neighbour j != i contributes min(Q[j], Q[i]) instead of Q[j].
    """
    nn = ny * nx
    w = dx if dx < dy else dy
    eps = 0.25 * w
    ldiag = math.sqrt(dx * dx + dy * dy)
    Lx = nx * dx
    Ly = ny * dy
    for i in numba.prange(nn):
        Qi = Q[i]
        if Qi <= 0.0:
            Qt[i] = 0.0
            continue
        iy = i // nx
        ix = i - iy * nx
        # --- flow direction of i: weighted mean of its receiver unit vectors.
        # ``fn`` is gamma, the same scalar that normalises the effective slope
        # (one definition, in _recv_mean_direction; see constants.MFD_GAMMA_FLOOR).
        fx, fy, fn = _recv_mean_direction(i, receivers, weights, nb_receivers,
                                          ny, nx, dx, dy, wrap_y, wrap_x)
        # Self-receiving (border), degenerate, or an ill-conditioned fan: a
        # direction built from a receiver set spanning more than a half-plane
        # carries no usable orientation, so fall back to the cell's own flux
        # rather than normalising an arbitrarily small vector. A D-inf cell
        # cannot reach the floor (its receivers span 45 deg => gamma >= 0.9239),
        # so this is inert on every shipped router.
        if fn < _GAMMA_FLOOR:
            Qt[i] = Qi
            continue
        fx /= fn
        fy /= fn
        tx = -fy                           # transverse unit vector
        ty = fx
        R = 0.5 * alpha_g * H_lag[i]
        reach = R + 0.5 * w
        # bounding box: crossing point within |t| < reach on the line, link
        # endpoints within one diagonal of it -> conservative axis-aligned box
        bx = int(math.ceil((reach + 2.0 * ldiag) / dx))
        by = int(math.ceil((reach + 2.0 * ldiag) / dy))
        # Each row of the box is narrowed to the strip of columns that can
        # hold a contributing j: a link (at most ldiag long; every router links
        # 8-neighbours) crossing the line puts j within ldiag upstream of it
        # (-ldiag <= s_j < 0) and within reach + ldiag of i along it. Only j
        # that would add nothing are skipped, in the same ascending order, so
        # the sum is bit-for-bit the full box's. ``m`` dominates the rounding
        # of s and t (relative 1e-9 of the box against ~1e-15).
        m = 1e-9 * (bx * dx + by * dy)
        acc = 0.0
        for oy in range(-by, by + 1):
            py = oy * dy
            lo, hi = _strip_columns(-bx, bx, fx, py * fy - eps,
                                    -ldiag - m, m, dx)
            if lo > hi:
                continue
            lo, hi = _strip_columns(lo, hi, tx, py * ty,
                                    -(reach + ldiag + m), reach + ldiag + m, dx)
            if lo > hi:
                continue
            jy = iy + oy
            if jy < 0 or jy >= ny:
                if wrap_y:
                    jy = jy % ny
                else:
                    continue
            for ox in range(lo, hi + 1):
                jx = ix + ox
                if jx < 0 or jx >= nx:
                    if wrap_x:
                        jx = jx % nx
                    else:
                        continue
                j = jy * nx + jx
                Qj = Q[j]
                if Qj <= 0.0:
                    continue
                px = ox * dx                # j's centre relative to i's
                sj = px * fx + py * fy - eps
                if sj >= 0.0:               # j downstream of the line: no forward crossing
                    continue
                if min_rule and j != i and Qj > Qi:
                    Qj = Qi
                for k in range(nb_receivers[j]):
                    r = receivers[j, k]
                    if r == j:
                        continue
                    ry = r // nx
                    rx = r - ry * nx
                    rdx = (rx - jx) * dx
                    rdy = (ry - jy) * dy
                    if wrap_x:
                        if rdx > 0.5 * Lx:
                            rdx -= Lx
                        elif rdx < -0.5 * Lx:
                            rdx += Lx
                    if wrap_y:
                        if rdy > 0.5 * Ly:
                            rdy -= Ly
                        elif rdy < -0.5 * Ly:
                            rdy += Ly
                    sr = (px + rdx) * fx + (py + rdy) * fy - eps
                    if sr < 0.0:            # link does not reach the line
                        continue
                    f = sj / (sj - sr)      # crossing parameter in [0, 1)
                    cx = px + f * rdx
                    cy = py + f * rdy
                    t = cx * tx + cy * ty
                    at = -t if t < 0.0 else t
                    if at >= reach:
                        continue
                    om = (reach - at) / w
                    if om > 1.0:
                        om = 1.0
                    acc += om * Qj * weights[j, k]
        if acc < Qi:
            acc = Qi
        Qt[i] = acc


def transect_flux(ice_flux, H_lag, receivers, weights, nb_receivers, shape,
                  dx, dy, alpha_g, wrap_y, wrap_x, min_rule=False):
    """Closure flux for ``closure_flux='transect'`` / ``'transect_min'``: the
    routed ice flux crossing a line one closure-width (``alpha_g*H_lag``) long
    through each icy cell, perpendicular to its flow. Takes the router bundle
    in either layout (SFR ``(n,)`` receivers or D-inf ``(n, k)``) and the RAW
    lagged thickness. Returns ``(ny, nx)``;
    equals ``ice_flux`` at a trunk with flux-free flanks and
    ``ice_flux*alpha_g*H/w_eff`` inside a uniform sheet. See
    :func:`_transect_flux_kernel` and ``constants.CLOSURE_FLUX``."""
    ny, nx = int(shape[0]), int(shape[1])
    Q = np.ascontiguousarray(np.asarray(ice_flux, dtype=np.float64).ravel())
    Hl = np.ascontiguousarray(np.asarray(H_lag, dtype=np.float64).ravel())
    rec = np.asarray(receivers, dtype=np.int64)
    if rec.ndim == 1:
        rec = rec.reshape(-1, 1)
        wts = np.ones_like(rec, dtype=np.float64)
        nbr = np.ones(rec.shape[0], dtype=np.int64)
    else:
        wts = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
        nbr = np.ascontiguousarray(np.asarray(nb_receivers, dtype=np.int64))
    rec = np.ascontiguousarray(rec)
    Qt = np.empty(ny * nx, dtype=np.float64)
    _transect_flux_kernel(Q, Hl, rec, wts, nbr, ny, nx, float(dx), float(dy),
                          float(alpha_g), bool(wrap_y), bool(wrap_x),
                          bool(min_rule), Qt)
    return Qt.reshape((ny, nx))


def _H_from_QS_modeA(law_code, gp, Qg, S):
    """Per-law point closure H(Qg, S): the from_slope branch of the shared
    :func:`_modeb_closure` (single-sourced; audit m34)."""
    p = gp
    return float(_modeb_closure(
        law_code, True, S, Qg, 0.0, 1.0,
        p.cg, p.lambda_p, p.lambda_c, p.tau_c, p.rho_g_g, p.coulomb_clamp))


def _solve_border_H_modeA(z_flat, H_flat, ice_flux, receivers, nb_receivers,
                          lengths, law_code, gp):
    """Mode A: a self-receiving border node with through-flowing ice gets its
    thickness from the per-law H(Q, S) closure with S the steepest upwind
    (donor-side) surface slope — matching siim1d's outlet treatment. Mutates
    ``H_flat`` in place at border cells; the interior is untouched. ``nb_receivers``
    is read only on the D-inf branch (may be None under SFR).

    Deliberately RAW lengths, even under the gamma effective-slope normalisation
    (``constants.MFD_GAMMA_FLOOR``): this maxes over single donor EDGE slopes,
    not a multi-receiver reduction, and gamma is not defined for one edge. Same
    call for the mode-B outflow border's arrival slope (step 5b of
    :func:`siim._core.skeleton._glac_fast_solve_modeB_dinf`)."""
    nn = z_flat.shape[0]
    idx = np.arange(nn)
    ice = ice_flux.ravel()
    s_up = np.zeros(nn)
    if receivers.ndim == 2:  # D-inf
        rec = np.asarray(receivers, dtype=np.int64)
        lens = np.asarray(lengths, dtype=np.float64)
        nb = np.asarray(nb_receivers, dtype=np.int64)
        self_rec = (nb == 1) & (rec[:, 0] == idx)
        for k in range(rec.shape[1]):
            m = (~self_rec) & (lens[:, k] > 0.0) & (rec[:, k] != idx)
            # Slot k is active only where the cell actually has k+1 receivers.
            # (Was hardwired ``nb == 2`` at k == 1 -- exact for the 2-slot D-inf
            # pack, but it silenced slots 2..7 of a k-ary pack entirely.
            # ``nb > k`` is identical at k <= 1 and a no-op at k == 0, where
            # every cell has nb >= 1.)
            m &= nb > k
            np.maximum.at(s_up, rec[m, k],
                          (z_flat[m] - z_flat[rec[m, k]]) / lens[m, k])
    else:  # SFR
        rec = np.asarray(receivers, dtype=np.int64)
        lens = np.asarray(lengths, dtype=np.float64)
        self_rec = rec == idx
        m = (~self_rec) & (lens > 0.0)
        np.maximum.at(s_up, rec[m], (z_flat[m] - z_flat[rec[m]]) / lens[m])
    for b in np.where(self_rec & (ice > 0.0) & (s_up > 0.0))[0]:
        H_flat[b] = _H_from_QS_modeA(law_code, gp, float(ice[b]), float(s_up[b]))


def run_modeA_step(surface, H, ice_flux, water_flux, law_code, gp, dt,
                   stack, receivers, nb_receivers, weights, lengths,
                   hc_over_H, shape, slope_normalise=False,
                   dx=None, dy=None, wrap_y=False, wrap_x=False):
    """Mode A erosion step (ice-surface state). Body of ``GlacialSPLModeA``:
    dispatch to the routing-specific mode-A skeleton (mutating a fresh copy of
    the surface + H), solve the border-H closure, then commit. Returns
    ``(z_eroded, H_new, bedrock_surface, erosion, denudation)`` — ``erosion ==
    denudation`` (mode A erodes the ice surface as its single hc-invariant
    state).

    ``slope_normalise`` (kernel default False; the facades pass the
    router-derived value, see
    ``siim._core.router.SLOPE_NORMALISED_ROUTERS``) turns on the gamma
    effective-slope normalisation on the D-inf branch, and then ``dx``/``dy``
    are REQUIRED (gamma is a grid-geometry quantity). It is applied by handing
    the kernel gamma-scaled link lengths: the mode-A kernel reads ``lengths``
    only as the denominator of its weighted-mean slope
    ``S = sum_k w_k*(z_i - z_rk)/L_k`` and of the per-receiver eroder residuals,
    so the thickness closure sees exactly ``S /= gamma``. The eroders see it
    exactly only at residual exponent 1: at ``p != 1`` their sum
    ``gamma**-p * sum_k w_k*S_k**p`` (``S_k`` the raw link slope) is not
    ``(S/gamma)**p`` (Jensen), so on a plane the erosion runs up to 0.6 % high
    at p = 2 and 1.7 % at p = 3, depending on the flow angle within the facet.
    Mode B avoids this with the effective-receiver eroders
    (``eroders._erode_modeb_eff``); mode A keeps the per-receiver ones. The
    border-H closure keeps the RAW lengths
    -- it maxes single upwind EDGE slopes, not a multi-receiver reduction. SFR
    ignores the flag (one receiver => gamma == 1). See
    ``constants.MFD_GAMMA_FLOOR``."""
    z_flat = surface.flatten()               # flatten() already returns a fresh copy
    H_flat = H.flatten()                     # ditto
    ice = ice_flux.ravel()
    water = water_flux.ravel()
    # --- kernel: mode A's H solver reads the pre-erosion slope of z_flat, so
    #     the H written is consistent with the pre-erosion geometry.
    if receivers.ndim == 2:   # D-inf
        nb_rec = np.asarray(nb_receivers, dtype=np.int64)
        recs   = np.asarray(receivers,   dtype=np.int64)
        wts    = np.asarray(weights,     dtype=np.float64)
        lens   = np.asarray(lengths,     dtype=np.float64)
        if slope_normalise:
            if dx is None or dy is None:
                raise ValueError(
                    "run_modeA_step(slope_normalise=True) needs dx and dy "
                    "(gamma is a grid-geometry quantity).")
            lens = gamma_scaled_lengths(
                recs, wts, nb_rec, lens, int(shape[0]), int(shape[1]),
                float(dx), float(dy), bool(wrap_y), bool(wrap_x))
        _glac_fast_solve_modeA_dinf(
            z_flat, ice, water, H_flat,
            law_code, gp, dt, stack, nb_rec, recs, wts, lens)
    else:                     # SFR
        _glac_fast_solve_modeA_sfr(
            z_flat, ice, water, H_flat,
            law_code, gp, dt, lengths, stack, receivers)
    # --- border-H closure (mutates H_flat at self-receiving border cells).
    _solve_border_H_modeA(z_flat, H_flat, ice_flux, receivers, nb_receivers,
                          lengths, law_code, gp)
    # --- commit.
    ice_thickness = H_flat.reshape(shape)
    z_eroded = z_flat.reshape(shape)
    bedrock_surface = z_eroded - hc_over_H * ice_thickness
    erosion = surface - z_eroded
    denudation = erosion
    return z_eroded, ice_thickness, bedrock_surface, erosion, denudation


# ---------------------------------------------------------------------------
# 8. Mode B kernel (GlacialSPLModeB._run_modeB_kernel_nocarve)
# ---------------------------------------------------------------------------
def run_modeB_kernel(zb_flat, H_flat, ice_flux, water_flux, law_code, gp, dt,
                     stack, receivers, nb_receivers, weights, lengths,
                     shape, dx, dy, border_bed_uplift, bl, gate, ramp,
                     parallel, wrap_y, wrap_x, slope_normalise=False,
                     hc_cell=None):
    """No-carve mode-B kernel dispatch (SFR or D-inf). Mutates ``zb_flat`` and
    ``H_flat`` in place; returns the kernel's ``surface_out`` (= zb + hc*H).
    ``hc_cell`` (flat, ``ny*nx``) is the per-cell channel-floor datum of
    mode D's valley <-> sheet map (``constants.HC_SHEET``) — every surface the
    kernel reconstructs becomes ``zb + hc_cell*H``; ``None`` is the run's scalar.

    The kernel reconstructs its OWN ``zs = zb + hc*H`` from the raw H for every
    closure and erosion slope — it never sees a routing surface (FIREWALL: see
    ``siim._core.driver``). De-squeezes the size-1 clock slices of
    ``border_bed_uplift`` and ``bl`` inside, so the shell and the driver share
    the treatment. ``bl``
    may also be the driver's flat ``(ny*nx,)`` per-side border datum (see
    :func:`siim._core.driver._bl_field`).

    ``slope_normalise`` (kernel default False; the facades pass the
    router-derived value, see
    ``siim._core.router.SLOPE_NORMALISED_ROUTERS``) turns on the gamma
    effective-slope normalisation inside the D-inf kernel; SFR ignores it (a
    single receiver has gamma == 1 identically). See
    ``constants.MFD_GAMMA_FLOOR``."""
    ny, nx = int(shape[0]), int(shape[1])
    dx_cell = float(dx)
    dy_cell = float(dy)
    surface_out = np.empty_like(zb_flat)
    # Drop the leading size-1 tstep dim left by xsimlab on a (nt, y, x) slice.
    bbu = np.asarray(border_bed_uplift, dtype=np.float64)
    if bbu.ndim == 3:
        bbu = bbu[0]
    bbu_flat = np.broadcast_to(bbu, (ny, nx)).ravel()
    # Per-step water-line datum + the global flotation gate and its ramp width.
    # ``bl`` arrives either as a scalar (the clock strips the 'tstep' dim, so
    # the adapter's bl is scalar each step) — broadcast to the uniform field —
    # or as the driver's flat (ny*nx,) PER-SIDE border datum; the kernel gives
    # each interior node the datum of the outlet its basin drains to.
    bl_arr = np.asarray(bl, dtype=np.float64)
    if bl_arr.size == ny * nx:
        bl = np.ascontiguousarray(bl_arr.ravel())
    else:
        bl = np.full(ny * nx, float(bl_arr.ravel()[-1]))
    gate = bool(gate)
    ramp = float(ramp)
    par = bool(parallel)
    # The kernel's only parallel regions are the level eroder's (par), which
    # run at the performance-core thread count.
    with level_threads(par):
        if receivers.ndim == 2:   # D-inf
            nb_rec = np.asarray(nb_receivers, dtype=np.int64)
            recs = np.asarray(receivers, dtype=np.int64)
            wts = np.asarray(weights, dtype=np.float64)
            lens = np.asarray(lengths, dtype=np.float64)
            _glac_fast_solve_modeB_dinf(
                zb_flat, ice_flux.ravel(), water_flux.ravel(),
                H_flat, surface_out, law_code, gp,
                dt, stack, nb_rec, recs, wts, lens,
                ny, nx, dx_cell, dy_cell, bbu_flat, wrap_y, wrap_x,
                bl, gate, ramp, par, bool(slope_normalise), hc_cell=hc_cell)
        else:
            _glac_fast_solve_modeB_sfr(
                zb_flat, ice_flux.ravel(), water_flux.ravel(),
                H_flat, surface_out, law_code, gp,
                dt, lengths, stack, receivers,
                ny, nx, dx_cell, dy_cell, bbu_flat, bl, gate, ramp,
                wrap_y, wrap_x, par, hc_cell=hc_cell)
    return surface_out


# ---------------------------------------------------------------------------
# 9. Sub-grid width carve (GlacialSPLModeC._carve_bed)
# ---------------------------------------------------------------------------
def carve_bed(zb_flat, H_flat, surface_out, zb_pre, receivers, alpha_g,
              hc_over_H, widening_factor, shape, dx, dy, wrap_y, wrap_x,
              offsets=None, D=None, SRC=None, zb_kern=None, s_conf=None):
    """Apply the sub-grid width carve (see :mod:`siim._core.carve`) to the
    post-kernel bed ``zb_flat`` IN PLACE. ``zb_pre`` is the pre-kernel bed (the
    denudation datum + descent-cap origin); ``surface_out`` the kernel's
    reconstructed ice surface (updated for carved cells). Routing-agnostic:
    receivers enter only through the border marker ``rec[i] == i`` (self-receiving
    = base-level border). ``offsets/D/SRC/zb_kern`` are optional reusable scratch
    buffers (an optimization, not semantics — allocated fresh if omitted).
    ``s_conf`` (``(ny, nx)``, mode D's valley <-> sheet confinedness,
    ``constants.HC_SHEET``) scales each SOURCE's widening factor by its
    own s: a sheet source (s = 0) carves nothing, a trunk (s = 1) carves exactly
    as without it; ``None`` is today's carve."""
    ny, nx = int(shape[0]), int(shape[1])
    dx_cell = float(dx)
    dy_cell = float(dy)
    nn = ny * nx
    if D is None:
        offsets = np.empty((ny, nx))
        D = np.empty((ny, nx))
        SRC = np.empty((ny, nx), dtype=np.int64)
        zb_kern = np.empty(nn)
    if receivers.ndim == 2:            # D-inf: 1-D self-at-border marker
        idx = np.arange(nn, dtype=np.int64)
        recs = np.asarray(receivers, dtype=np.int64)
        rec_marker = np.where(recs[:, 0] == idx, idx, -1)
    else:
        rec_marker = receivers
    seed_mask = np.ones(nn, dtype=np.int8)   # seed every icy interior cell
    n_seed = _carve_offsets(H_flat, rec_marker, float(alpha_g),
                            offsets.ravel(), seed_mask)
    if n_seed == 0:
        return
    if wrap_x or wrap_y:
        _power_dt_2d_periodic(offsets, dy_cell, dx_cell, D, SRC, wrap_y, wrap_x)
    else:
        _power_dt_2d(offsets, dy_cell, dx_cell, D, SRC)
    np.copyto(zb_kern, zb_flat)              # post-kernel bed (source anchors)
    # lazy: a fresh ones array per call outside mode D (nn floats, far
    # below the two power-transform passes), upgrade path: a per-run scratch.
    s_src = (np.ones(nn) if s_conf is None
             else np.ascontiguousarray(s_conf, dtype=np.float64).ravel())
    _carve_subgrid_width(zb_flat, zb_kern, zb_pre, H_flat,
                         surface_out, rec_marker, D, SRC,
                         offsets, widening_factor, hc_over_H, s_src)


# ---------------------------------------------------------------------------
# 10. Flow routing (fill-then-route). Three in-house producers on the SAME
#     eps-filled surface: the D8 single-flow router (route_d8, the framework-free
#     replacement for fastscape's fortran SingleFlowRouter), the D-inf
#     Tarboton router (route_dinf) and the MFD router (route_mfd). All derive the
#     interior/boundary mask directly from border_status (Map 3 §4) — no fortran
#     call — and the two multi-receiver producers share the D8 basin labeling.
# ---------------------------------------------------------------------------
def _filled_surface_and_mask(elevation, shape, border_status):
    """The fill-then-route preamble shared by :func:`route_d8` / :func:`route_dinf`:
    the ``border_status`` interior mask (Map 3 §4 — the in-house replacement for
    the fortran ``sfr_rec != i`` mask), the looped-axis wrap flags, and the
    eps-filled surface (depression floors filled to spill + eps, so every
    interior cell has a strictly-lower 8-neighbour and lakes drain toward their
    spills). Returns ``(z_route, interior, wrap_y, wrap_x, ny, nx, nn)``. The
    kernels/physics keep consuming the TRUE elevation; only routing sees the fill."""
    ny, nx = int(shape[0]), int(shape[1])
    nn = ny * nx
    interior = d8_interior_mask(border_status, ny, nx)
    bs = list(np.broadcast_to(border_status, 4))
    wrap_x = bs[0] == 'looped'
    wrap_y = bs[2] == 'looped'
    z_flat = np.ascontiguousarray(elevation.ravel().astype(np.float64))
    z_route = np.empty(nn, dtype=np.float64)
    _priority_flood_eps(z_flat, ny, nx, interior, 1e-6, wrap_y, wrap_x, z_route)
    return z_route, interior, wrap_y, wrap_x, ny, nx, nn


def route_d8(elevation, shape, dx, dy, border_status):
    """In-house D8 single-flow router (fill-then-route) — the framework-free
    replacement for fastscape's fortran ``SingleFlowRouter``
    (``fs.flowroutingsingleflowdirection``). Returns the SFR router bundle
    ``(receivers, weights, lengths, nb_receivers, stack, basin)``: ``receivers``
    / ``lengths`` 1D ``(n,)``, ``weights`` / ``nb_receivers`` all-ones (SFR is
    single-receiver), ``stack`` outlet-first, ``basin`` ``(ny, nx)`` labeled by
    outlet index. The receiver scan replicates fortran ``find_receiver`` on the
    eps-filled surface (Map 3 §3); the routing delta vs fortran is confined to
    depression/tie cells (behavioral gate)."""
    z_route, interior, wrap_y, wrap_x, ny, nx, nn = _filled_surface_and_mask(
        elevation, shape, border_status)
    receivers = np.empty(nn, dtype=np.int64)
    lengths = np.empty(nn, dtype=np.float64)
    _d8_receivers(z_route, ny, nx, float(dx), float(dy), interior,
                  receivers, lengths, wrap_y, wrap_x)
    stack = np.empty(nn, dtype=np.int64)
    _d8_stack(receivers, nn, stack)
    basin = np.empty(nn, dtype=np.int64)
    _d8_basin(receivers, stack, nn, basin)
    weights = np.ones(nn, dtype=np.float64)
    nb_receivers = np.ones(nn, dtype=np.int64)
    return receivers, weights, lengths, nb_receivers, stack, basin.reshape((ny, nx))


def route_dinf(elevation, shape, dx, dy, border_status, lazy_basin=False):
    """D-infinity flow directions on the eps-filled surface. Body of
    ``DinfFlowRouter.run_step`` — fully fortran-free (S4): the interior/boundary
    mask comes from ``border_status`` directly (Map 3 §4, provably identical to
    the old fortran ``sfr_rec != i`` mask), and ``basin`` from the in-house
    outlet labeling. Returns ``(receivers, weights, lengths, nb_receivers,
    stack, basin)`` — ``receivers``/``weights``/``lengths`` ``(n, 2)``,
    ``basin`` ``(ny, nx)``, or with ``lazy_basin`` a zero-argument callable
    returning it (a whole D8 pass, which the in-house driver runs only on the
    steps a frame saves; :class:`siim._core.router.RoutingArrays`)."""
    z_route, interior, wrap_y, wrap_x, ny, nx, nn = _filled_surface_and_mask(
        elevation, shape, border_status)
    # D-inf on the eps-filled surface: every interior cell — including
    # closed-basin floors — gets a strictly downhill facet, so flux crosses
    # depressions toward their spills and the topological sort is valid by
    # construction. The kernels/physics keep consuming the true elevation.
    rec1 = np.zeros(nn, dtype=np.int64)
    rec2 = np.zeros(nn, dtype=np.int64)
    w1 = np.zeros(nn, dtype=np.float64)
    w2 = np.zeros(nn, dtype=np.float64)
    len1 = np.zeros(nn, dtype=np.float64)
    len2 = np.zeros(nn, dtype=np.float64)
    # slope_out = the steepest-facet slope on the FILLED surface; the router
    # itself does not consume it (audit N14), a cheap routing diagnostic.
    slope_out = np.zeros(nn, dtype=np.float64)
    _dinf_route(z_route, ny, nx, float(dx), float(dy), interior,
                rec1, rec2, w1, w2, len1, len2, slope_out,
                _DINF_E1_DJ, _DINF_E1_DI, _DINF_E2_DJ, _DINF_E2_DI,
                wrap_y, wrap_x)

    # Pack into (n, 2) receiver/weight arrays.
    receivers = np.zeros((nn, 2), dtype=np.int64)
    weights = np.zeros((nn, 2), dtype=np.float64)
    lengths = np.zeros((nn, 2), dtype=np.float64)
    nb_receivers = np.zeros(nn, dtype=np.int64)
    _dinf_pack(rec1, rec2, w1, w2, len1, len2, nn,
               receivers, weights, lengths, nb_receivers)

    # Topological sort (receivers-first), then reverse for fastscape's
    # donor-first convention. Rebuilt each step (topology changes every step).
    stack_rec_first = np.zeros(nn, dtype=np.int64)
    _dinf_topo_stack(rec1, rec2, w1, w2, nn, stack_rec_first)
    stack = stack_rec_first[::-1].copy()

    # basin_ids (diagnostic-only): the D8 single-receiver chain on the SAME
    # eps-filled surface, labeled by outlet index — deterministic, replacing
    # the fortran ``catch`` (unseeded random labels; the last D-inf fortran
    # dependency). Excluded from every equality gate (Map 4 §4).
    basin = functools.partial(_d8_basin_labels, z_route, interior, ny, nx,
                              dx, dy, wrap_y, wrap_x)
    return (receivers, weights, lengths, nb_receivers, stack,
            basin if lazy_basin else basin())


def _d8_basin_labels(z_route, interior, ny, nx, dx, dy, wrap_y, wrap_x):
    """The diagnostic ``basin`` labeling both multi-receiver producers reuse: the
    D8 single-receiver chain on the SAME eps-filled surface, labeled by outlet
    index (deterministic, replacing the fortran ``catch``). Excluded from every
    equality gate (Map 4 §4), and — under a multi-receiver router — documented as
    possibly disagreeing with the argmax-weight graph."""
    nn = ny * nx
    d8_rec = np.empty(nn, dtype=np.int64)
    d8_len = np.empty(nn, dtype=np.float64)
    _d8_receivers(z_route, ny, nx, float(dx), float(dy), interior,
                  d8_rec, d8_len, wrap_y, wrap_x)
    d8_stack = np.empty(nn, dtype=np.int64)
    _d8_stack(d8_rec, nn, d8_stack)
    basin = np.empty(nn, dtype=np.int64)
    _d8_basin(d8_rec, d8_stack, nn, basin)
    return basin.reshape((ny, nx))


def route_mfd(elevation, shape, dx, dy, border_status,
              mfd_exponent=_constants.MFD_EXPONENT, lazy_basin=False):
    """Multiple-flow-direction router on the eps-filled surface — body of
    ``MfdFlowRouter.run_step`` and of ``InhouseRouter``'s ``routing='mfd'``
    branch. Flux is split over ALL strictly downhill 8-neighbours with weights
    ``(S_k/S_max)^p``, ``p = mfd_exponent``; at ``p = inf`` it follows the
    steepest alone (see ``constants.MFD_EXPONENT`` and
    :func:`siim._core.routing._mfd_route` for the producer contract).

    Same fill-then-route preamble as :func:`route_d8` / :func:`route_dinf`, an
    ``(nn, 8)`` pack, the k-ary donor-first Kahn stack
    (:func:`siim._core.routing._mfd_stack`) and the D-inf convention for
    ``basin`` (the D8 chain on the same filled surface, and ``lazy_basin`` as
    there). Returns
    ``(receivers, weights, lengths, nb_receivers, stack, basin)`` —
    ``receivers``/``weights``/``lengths`` ``(nn, 8)``, ``basin`` ``(ny, nx)``.

    The exponent goes through ``constants.resolve_mfd_exponent`` (a finite
    value above ``MFD_EXPONENT_MAX`` is routed as ``inf``, the steepest link
    alone, with a warning; a non-positive or NaN one raises). Raises for a
    PERIODIC axis shorter than 3 cells: there the ``-1`` and ``+1`` offsets alias onto the same
    node, which would put a duplicate link in the pack (the producer does not
    de-duplicate, and every consumer sums over the prefix)."""
    p = _constants.resolve_mfd_exponent(mfd_exponent)
    z_route, interior, wrap_y, wrap_x, ny, nx, nn = _filled_surface_and_mask(
        elevation, shape, border_status)
    if (wrap_x and nx < 3) or (wrap_y and ny < 3):
        raise ValueError(
            "flow_routing='mfd' needs at least 3 cells along a 'looped' axis "
            f"(got ny={ny}, nx={nx} with wrap_y={wrap_y}, wrap_x={wrap_x}): "
            "on a shorter periodic axis the -1 and +1 neighbour offsets alias "
            "onto the same node, duplicating a link in the receiver pack.")
    receivers = np.empty((nn, 8), dtype=np.int64)     # _mfd_route writes every slot
    weights = np.empty((nn, 8), dtype=np.float64)
    lengths = np.empty((nn, 8), dtype=np.float64)
    nb_receivers = np.empty(nn, dtype=np.int64)
    link_mask = np.empty(nn, dtype=np.uint8)          # the live links, for the stack
    out_degree = np.empty(nn, dtype=np.uint8)
    _mfd_route(z_route, ny, nx, float(dx), float(dy), interior, p,
               receivers, weights, lengths, nb_receivers,
               _MFD_DJ, _MFD_DI, wrap_y, wrap_x, link_mask, out_degree)
    stack = np.empty(nn, dtype=np.int64)
    _mfd_stack(link_mask, out_degree, ny, nx, _MFD_DJ, _MFD_DI, wrap_y, wrap_x,
               stack)
    basin = functools.partial(_d8_basin_labels, z_route, interior, ny, nx,
                              dx, dy, wrap_y, wrap_x)
    return (receivers, weights, lengths, nb_receivers, stack,
            basin if lazy_basin else basin())


# ---------------------------------------------------------------------------
# 11. Sediment accumulation (SedimentTracker.run_step)
# ---------------------------------------------------------------------------
def accumulate_sediment(denudation, cell_area, stack, receivers, nb_receivers,
                        weights, shape):
    """Route this step's denuded rock volume ``max(denudation, 0) * cell_area``
    down the flow graph in one accumulation pass. Body of
    ``SedimentTracker.run_step`` (the per-step ``flux``; the caller owns the
    cross-step running integral ``_cum``)."""
    field = np.maximum(denudation, 0.0).astype(np.float64).flatten() * float(cell_area)
    if receivers.ndim == 2:   # D-inf
        nb_rec = np.asarray(nb_receivers, dtype=np.int64)
        recs   = np.asarray(receivers,   dtype=np.int64)
        wts    = np.asarray(weights,     dtype=np.float64)
        _flow_accumulate_dinf(field, stack, nb_rec, recs, wts)
    else:                     # SFR
        _flow_accumulate_sd(field, stack, receivers)
    return field.reshape(shape)


def edge_sediment(flux, border_status):
    """Sum the routed per-node ``flux`` (the ``(ny, nx)`` field
    :func:`accumulate_sediment` returns) over each domain-edge outlet ring —
    the volume (m^3) leaving the domain across that edge this step.

    Returns a length-4 array in ``border_status`` order
    (``siim.siim2d.siim._BL_SIDES`` = left, right, bottom, top), NaN on any
    side that is not ``'fixed_value'``: only fixed-value rings are
    self-receiving outlets (:func:`siim._core.routing.d8_interior_mask`), so a
    'core' or 'looped' edge has no sediment to deliver, which is not the same
    as delivering none. Each corner node is counted exactly ONCE, by the same
    rule ``siim._core.driver._bl_field`` uses for the water datum: an outlet
    x-side (left/right) owns it, else the y-side.

    Shared by the in-house driver and ``SedimentTracker.run_step`` so the two
    front ends report the identical sums."""
    fixed = [b == 'fixed_value' for b in np.broadcast_to(border_status, 4)]
    f = np.asarray(flux)
    # The y-side rings give up a corner to whichever x-side is an outlet.
    trim = slice(1 if fixed[0] else 0, -1 if fixed[1] else None)
    out = np.full(4, np.nan)
    for s in range(4):
        if fixed[s]:
            ring = f[SIDE_SLICES[s]]
            out[s] = (ring if s < 2 else ring[trim]).sum()
    return out


# ---------------------------------------------------------------------------
# 12. Glacial flexure (GlacialFlexure.run_step; plate solve injected)
# ---------------------------------------------------------------------------
def footprint_ice_column(H, alpha_g, dy, dx, wrap_y=False, wrap_x=False, border=None,
                         weight=None):
    """The flexure ice column (m per cell): each channel's hc-free section
    ``alpha_g*H**2`` spread over the carve's power-diagram footprint (icy cells
    seed discs of radius alpha_g*H/2, the base-level ``border`` ring does not, the
    thickest disc takes a cell), as the cell-width average of the section's
    parabola at the cell's distance from its source. Overlapping widths count
    once, the thicker channel's (a thinner one inside it is part of it), so a
    broad cap loads about 1.5*H a cell where per-cell sections stacked; an
    axis-aligned valley on square cells conserves its section exactly, a
    diagonal one within ~5 % (the width window is radial). ``weight`` (per
    cell, 0..1) scales each SOURCE's section and de-seeds zero-weight cells:
    mode D's s, as the carve blends by the source's s (ED-184).
    """
    ny, nx = H.shape
    nn = ny * nx
    L = (dx * dy) ** 0.5
    Hf = np.ascontiguousarray(H, dtype=np.float64).ravel()
    offsets = np.empty((ny, nx))
    D = np.empty((ny, nx))
    SRC = np.empty((ny, nx), dtype=np.int64)
    rec_marker = np.full(nn, -1, dtype=np.int64)
    if border is not None:                          # base-level cells seed no disc
        rec_marker[np.ravel(border)] = np.arange(nn)[np.ravel(border)]
    seed = (np.ones(nn, dtype=np.int8) if weight is None
            else (np.ravel(weight) > 0.0).astype(np.int8))
    if _carve_offsets(Hf, rec_marker, float(alpha_g), offsets.ravel(), seed) == 0:
        return np.zeros((ny, nx))
    if wrap_y or wrap_x:
        _power_dt_2d_periodic(offsets, dy, dx, D, SRC, wrap_y, wrap_x)
    else:
        _power_dt_2d(offsets, dy, dx, D, SRC)
    Df, S = D.ravel(), SRC.ravel()
    # every cell with a source, not only those inside its disc: a cell whose
    # width straddles the disc edge carries the parabola's tail (zero beyond)
    cell = np.nonzero((S >= 0) & np.isfinite(Df))[0]
    src = S[cell]
    R = 0.5 * float(alpha_g) * Hf[src]
    r = np.sqrt(np.maximum(Df[cell] + R * R, 0.0))      # the cell's distance from its source
    a = np.maximum(r - 0.5 * L, -R)
    b = np.minimum(r + 0.5 * L, R)
    col = np.zeros(nn)
    # 1.5 = a parabola's centreline over its mean depth (geometry, not the hc datum)
    col[cell] = np.where(b > a, 1.5 * Hf[src] * ((b - a) - (b ** 3 - a ** 3) / (3.0 * R * R)) / L, 0.0)
    if weight is not None:
        col[cell] *= np.ravel(weight)[src]
    return col.reshape(ny, nx)


def glacial_flexure_step(elevation, denudation, surface_upward, ice_thickness,
                         alpha_g, cell_area, lithos_density, asthen_density,
                         e_thickness, ibc, shape, length, col_prev, ice_load,
                         flexure_solve, s_conf=None, wrap_y=False, wrap_x=False):
    """Incremental flexural isostasy: rock unloading (``surface_upward -
    denudation``) plus, when ``ice_load``, the per-step glacial ice load
    ``(rho_ice/lithos)*d(col)`` with ``col`` the channel cross-sections spread
    over their footprints (:func:`footprint_ice_column`, hc-free; a slab ``H`` on
    mode D's sheet cells, below; looped axes wrap the footprint). Body of ``GlacialFlexure.run_step``; the biharmonic plate
    solve is INJECTED as ``flexure_solve(elev_post, elev_eq, nx, ny, xl, yl,
    lithos, asthen, Te, ibc, wrap_y=, wrap_x=)`` (the in-house FFT solve of
    :mod:`siim._core.flexure`; injected to keep this module framework-free). Returns ``(rebound, col_new)``;
    the caller keeps ``col_new`` as the cross-step ``_col_prev``. ``s_conf``
    (``(ny, nx)``, mode D's valley <-> sheet reading, this step's, the carve's)
    weights each footprint by its source's s (a sheet source spreads nothing) and
    adds a slab ``(1 - s)*H`` on each cell; ``None`` is the footprint everywhere.
    s is 0 or 1 in practice, so a cell that flips at unchanged H steps its load.
    ``cell_area`` is no longer read (the footprint takes dx, dy from
    ``length``); it stays for the signature.

    The returned ``rebound`` is exactly 0 on every ``'fixed_value'`` border row
    and column (read off ``ibc``, so the in-house driver and the xsimlab adapter
    share it) — those nodes already get no block uplift and no erosion, so
    letting them subside was the one place their value was not actually fixed."""
    ny, nx = shape
    yl, xl = length

    lithos_density = np.broadcast_to(lithos_density, shape).flatten()
    elevation_eq = elevation.flatten()
    diff = (surface_upward - denudation).ravel()
    # Mass-conserving glacial ice load: each channel's cross-section alpha_g*H**2
    # (= Qg/V, hc-free) over its footprint, once (a per-cell alpha_g*H**2/L
    # counted overlapping wide channels several times; ED-184).
    fixed = np.zeros(shape, dtype=bool)           # the 'fixed_value' ring
    _all = slice(None)
    for is_fixed, side in zip(_fixed_flags(ibc), ((0, _all), (-1, _all), (_all, 0), (_all, -1))):
        fixed[side] |= is_fixed
    col = (footprint_ice_column(np.asarray(ice_thickness, dtype=np.float64).reshape(shape),
                                alpha_g, yl / (ny - 1), xl / (nx - 1), wrap_y, wrap_x,
                                border=fixed, weight=s_conf)
           if ice_load else np.zeros(shape))
    if s_conf is not None:
        # a sheet cell is a slab of thickness H, not a channel (audit #28)
        col = col + (1.0 - s_conf) * ice_thickness
    if ice_load:
        # Incremental rock-equivalent ice load: rho_ice*g*d(col) ==
        # rho_lithos*g*((rho_ice/lithos_density)*d(col)); g cancels, lithos_density
        # is per-cell so divide element-wise.
        dcol = (col - col_prev).ravel()
        diff = diff + (RHO_ICE / lithos_density) * dcol

    elevation_pre = elevation_eq + diff
    elevation_post = elevation_pre.copy()
    flexure_solve(
        elevation_post,
        elevation_eq,
        nx,
        ny,
        xl,
        yl,
        lithos_density,
        asthen_density,
        e_thickness,
        ibc,
        wrap_y=wrap_y,      # the plate is periodic along a looped axis (ED-181)
        wrap_x=wrap_x,
    )
    rebound = (elevation_post - elevation_pre).reshape(shape)
    # A 'fixed_value' border is a base-level node: block uplift is already
    # masked off it (uplift_mask) and only mode B moves it (border_bed_uplift),
    # so it gets no rebound either -- otherwise the edge share of the
    # subsidence sinks it into a trench that becomes the base level.
    rebound[fixed] = 0.0
    col_new = col.copy()
    return rebound, col_new


# ---------------------------------------------------------------------------
# 13. Vertical-motion composition (stock fastscape group-sum, reproduced for
#     the in-house driver, which calls it; the xsimlab model keeps using the
#     stock TectonicForcing/TotalErosion/TotalVerticalMotion).
# ---------------------------------------------------------------------------
def sum_erosion(*erosion_terms):
    """Reproduce fastscape ``TotalErosion.height = sum(erosion group)`` =
    ``glacial_spl.erosion + diffusion.erosion``. Sums from 0 like the stock
    builtin ``sum``, so a single term returns a copy (``0 + a == a``)."""
    return sum(erosion_terms)


def compose_vertical_motion(surface_forcing, bedrock_forcing, rebound,
                            erosion_total):
    """Reproduce fastscape's ``TotalVerticalMotion`` group-sum composition::

        surface_up = sum(surface_upward group) - sum(surface_downward group)
                   = (surface_forcing + rebound) - erosion_total
        bedrock_up = sum(bedrock_upward group)  = bedrock_forcing + rebound

    ``surface_forcing`` / ``bedrock_forcing`` are the tectonic forcings
    (``TectonicForcing.{surface,bedrock}_upward`` = block uplift here); ``rebound``
    the flexural rebound in both upward groups (``None`` when flexure is off);
    ``erosion_total`` the ``TotalErosion.height`` (:func:`sum_erosion`). The
    finalize commit ``topo += surface_up`` lives in the driver."""
    if rebound is None:
        surface_up = surface_forcing - erosion_total
        bedrock_up = bedrock_forcing
    else:
        surface_up = (surface_forcing + rebound) - erosion_total
        bedrock_up = bedrock_forcing + rebound
    return surface_up, bedrock_up


# ---------------------------------------------------------------------------
# 14. Valley <-> sheet devices: the border rule and the per-cell datum
#     (mode D, constants.HC_SHEET; the driver's devices consume s)
# ---------------------------------------------------------------------------
def border_inherits(s, receivers, nb_receivers, ice_flux):
    """Hand each base-level border cell (a self-receiver) the confinedness of
    its dominant donor — the kernels' ``dom``: over EVERY receiver link of the
    router (``receivers`` flat for D8, ``(n, k)`` with ``nb_receivers`` for
    D-inf and MFD), the icy donor draining the most ice into it, the lowest
    index on a tie — so the border's column datum follows the same cell its
    zero-gradient thickness is copied from. A border read on its own can
    disagree with that donor: an hc = 1.5 wall at the outlet of an hc = 1
    sheet, which the zero-gradient rule then raises without bound (measured
    under the retired fullness switch: 113 km of ice in 15 steps). Borders
    without icy donors keep their value; interior cells are untouched."""
    shape = np.shape(s)
    s = np.asarray(s, dtype=np.float64).ravel().copy()
    rec = np.asarray(receivers, dtype=np.int64)
    if rec.ndim == 1:                           # one link per cell
        rec, nb = rec.reshape(-1, 1), np.ones(s.size, dtype=np.int64)
    else:                                       # every (donor, receiver) link in use
        nb = np.asarray(nb_receivers, dtype=np.int64)
    _border_inherits_kernel(s, rec, nb, np.asarray(ice_flux, dtype=np.float64).ravel())
    return s.reshape(shape)


@numba.njit(cache=True)
def _border_inherits_kernel(s, rec, nb, Q):
    """:func:`border_inherits` on the flat ``s``, IN PLACE: the kernels' ``dom``
    loop (ascending donors, a strict '>', so the lowest index wins a tie) over
    every live link, kept to the border receivers. Writes only borders and
    reads only interior donors, so the visiting order cannot matter."""
    n = s.shape[0]
    dom = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if Q[i] > 0.0 and rec[i, 0] != i:
            for k in range(nb[i]):
                r = rec[i, k]
                if r != i and rec[r, 0] == r and (dom[r] < 0 or Q[i] > Q[dom[r]]):
                    dom[r] = i
    for r in range(n):
        if dom[r] >= 0:
            s[r] = s[dom[r]]


def hc_cell_field(s, hc, hc_sheet):
    """The per-cell channel-floor datum ``hc_cell = hc*s + hc_sheet*(1 - s)``:
    the run's ``hc`` (a parabolic channel, ``constants.HC_OVER_H``) where
    s = 1 and the sheet's (``constants.HC_SHEET``: the surface is zb + H)
    where s = 0, exact at both ends."""
    return hc * s + hc_sheet * (1.0 - s)
