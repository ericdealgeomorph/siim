"""Framework-free composition-chain step functions for the siim 2D model.

Each ``run_step`` body of siim's own ``@xs.process`` classes
(:mod:`siim.fastscape.processes`) is extracted here as a ``self``-free
module-level function taking state + params explicitly. The xs.process classes
become thin shells that unpack ``self.*``, call the function, and assign the
outputs; the standalone in-house driver (added later in the migration) calls the
same functions. One implementation, two front ends — no divergence.

This module stays numpy/numba/scipy-only (like the rest of :mod:`siim._core`):
it imports NOTHING from xsimlab / fastscape / siim.fastscape. The single fortran
seam that survives S1 — the FFT flexure solve — is injected as a callable
(:func:`glacial_flexure_step`'s ``flexure_solve`` argument), so this module never
imports ``fastscapelib_fortran`` either.

The subtleties that must survive the extraction bit-for-bit (see Map 1 of
``docs/dev/standalone_migration_maps.md``):

* the ``ice_thickness`` one-step lag (router/accumulator see ``H(t-1)`` — an
  ordering artifact reproduced by reading H into the routing surface before the
  kernel overwrites it);
* exactly ONE ``routing_relax`` EMA update per step (:func:`ema_thickness`; the
  EMA carry ``_H_eff`` is cross-step state AND reused within-step by the trunk
  subclass — the shell owns that single update);
* size-1 clock-sliced inputs de-squeezed inside the extracted fns
  (``bl = float(asarray(bl).ravel()[-1])``, ``bbu[0]`` on a ``(1, ny, nx)`` slice);
* the FIREWALL (measured twice): the mode-B kernel reconstructs its OWN
  ``zs = zb + hc*H`` from raw H; the relaxed / fabricated surfaces
  (:func:`ema_thickness` / :func:`_fabricate_trunk_surface`) feed ONLY the router
  graph + the mass-balance surface, never a flux closure, the carve, the flexure
  load, or the outputs.
"""

import numpy as np

from ..constants import GRAVITY, RHO_ICE
from .. import constants as _constants
from .params import GlacialParams
from .routing import (
    _flow_accumulate_sd, _flow_accumulate_sd_2,
    _flow_accumulate_dinf, _flow_accumulate_dinf_2,
    _priority_flood_eps, _dinf_route, _dinf_topo_stack, _dinf_pack,
    _DINF_E1_DJ, _DINF_E1_DI, _DINF_E2_DJ, _DINF_E2_DI,
    _d8_receivers, _d8_stack, _d8_basin, d8_interior_mask,
)
from .skeleton import (
    _glac_fast_solve_modeA_sfr, _glac_fast_solve_modeA_dinf,
    _glac_fast_solve_modeB_sfr, _glac_fast_solve_modeB_dinf,
)
from .solvers import (
    LAW_EFFEXP, LAW_POWER, LAW_COULOMB,
    _modeb_closure,
)
from .carve import (
    _power_dt_2d, _power_dt_2d_periodic, _carve_offsets, _carve_subgrid_width,
)


# ---------------------------------------------------------------------------
# 1. Law record (GlacialLaw.initialize + _glacial_params_and_code)
# ---------------------------------------------------------------------------
def build_glacial_params(sliding_law, Ko, ce, n, nu, m, mu, Ac, alpha_g,
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
    bs = list(border_status)
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
def plateau_surface(x, y, shape, border_status, seed, noise_amplitude,
                    plateau_zo, plateau_dz, plateau_frac, plateau_w):
    """Arctan-smoothed plateau initial topography + tie-breaking noise (zeroed on
    'fixed_value' edges). Body of ``PlateauSurface.initialize``."""
    X, _Y = np.meshgrid(x, y)
    Lx = float(x[-1] - x[0])
    x_esc = (1.0 - plateau_frac) * Lx
    x_or = X - float(x[0])
    ramp = (1.0 + 2.0 / np.pi * np.arctan((x_or - x_esc) / plateau_w)) / 2.0
    slope = plateau_dz * np.clip(x_or - x_esc, 0.0, None) / max(Lx - x_esc, 1e-9)
    topo = ramp * (plateau_zo - slope)

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
    bs = list(border_status)
    if bs[0] == "fixed_value": noise[:,  0] = 0.0
    if bs[1] == "fixed_value": noise[:, -1] = 0.0
    if bs[2] == "fixed_value": noise[0,  :] = 0.0
    if bs[3] == "fixed_value": noise[-1, :] = 0.0
    return topo + noise


def wave_uplift(x, y, shape, mask, dt, t, delta_h, wave_width, wave_velocity,
                x_escarpment, wave_calibration, U_inf):
    """Moving-Gaussian uplift wave, midpoint-sampled (``t + dt/2``) with
    calibration 1.0 = exact ``delta_h`` deposition over the passage. Body of
    ``WaveUplift.run_step``; the border mask is :func:`uplift_mask`."""
    time = t + 0.5 * dt
    wave_center = float(x[0]) + x_escarpment + time * wave_velocity
    X, _Y = np.meshgrid(x, y)
    U0 = (wave_calibration * delta_h * wave_velocity
          / (wave_width * np.sqrt(np.pi)))
    rate = U_inf + U0 * np.exp(-(X - wave_center) ** 2 / wave_width ** 2)
    return rate * mask * dt


# ---------------------------------------------------------------------------
# 3. Block uplift (BlockUplift.initialize mask + GlacialBlockUplift.run_step)
# ---------------------------------------------------------------------------
def uplift_mask(border_status, shape):
    """Binary uplift mask: 0 on 'fixed_value' border rings, 1 elsewhere. Body of
    fastscape ``BlockUplift.initialize`` (the mask half)."""
    mask = np.ones(shape)
    _all = slice(None)
    slices = [(_all, 0), (_all, -1), (0, _all), (-1, _all)]
    for status, border in zip(border_status, slices):
        if status == "fixed_value":
            mask[border] = 0.0
    return mask


def block_uplift(rate, dt, mask, shape):
    """``uplift = rate * dt``, zeroed on fixed borders by ``mask``. Body of
    ``GlacialBlockUplift.run_step``; drops the leading size-1 tstep dim left by
    xsimlab on a ``(nt, y, x)`` slice before broadcasting."""
    rate = np.asarray(rate)
    if rate.ndim == 3:
        rate = rate[0]
    rate = np.broadcast_to(rate, shape) * mask
    return rate * dt


# ---------------------------------------------------------------------------
# 4. Routing surface (GlacialSurfaceToErode: EMA thickness + hc reconstruction)
# ---------------------------------------------------------------------------
def ema_thickness(H_lag, H_eff_prev, r):
    """The (optionally EMA-relaxed) lagged thickness feeding this step's routing
    + mass-balance surface. ``r == 0`` returns the raw lagged H unchanged
    (bit-for-bit); otherwise ``H_eff = r*H_eff_prev + (1-r)*H_lag`` (seeded at
    ``H_lag`` on the first step, ``H_eff_prev is None``).

    The caller owns the single per-step update: it stores the return as the EMA
    carry (cross-step) AND reuses the SAME value within-step (the trunk
    subclass). One update per model step — never call twice."""
    if r == 0.0:
        return H_lag                             # raw lagged H (bit-for-bit)
    H_lag = np.asarray(H_lag, dtype=np.float64)
    if H_eff_prev is None:
        return H_lag
    return r * H_eff_prev + (1.0 - r) * H_lag


def routing_surface(post_uplift_surface, hc_over_H, H_eff):
    """The mode-B routing/mass-balance surface ``zs = post_uplift_bed + hc*H_eff``
    (the reconstructed ice column on the post-uplift bed). ``H_eff`` is the
    relaxed lagged thickness (raw H when ``routing_relax == 0``). FIREWALL: this
    surface reaches ONLY the router graph + the accumulator's mass-balance
    surface — never a physics closure, the carve, the flexure load, or outputs."""
    return post_uplift_surface + float(hc_over_H) * H_eff


# ---------------------------------------------------------------------------
# 5. Fabricated trunk routing surface (relocated verbatim; already pure)
# ---------------------------------------------------------------------------
def _fabricate_trunk_surface(zs_dyn, zb, H_lag, border, alpha_g, dx, dy,
                             k_dip, floor, offsets, D, SRC, wrap_y, wrap_x):
    """Build the fabricated trunk routing surface (design record
    ``docs/dev/trunk_surface_routing.md``). Pure function (no process state) so
    the process and the channel-persistence test share one implementation.

    ``zs_dyn`` (ny, nx) is the dynamic ice surface ``zb + hc*H_lag``; ``zb`` the
    matching bed; ``H_lag`` the lagged thickness; ``border`` (ny, nx bool) the
    base-level edges (excluded as seeds + never fabricated). ``offsets/D/SRC``
    are (ny, nx) scratch buffers. Returns a fresh (ny, nx) elevation: the
    V-dipped trunk surface at footprint cells (``max(zs_geo, zb)``), ``zs_dyn``
    elsewhere.
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
    elev[mem_idx] = np.maximum(zs_geo, zb_flat[mem_idx])
    return elev.reshape(ny, nx)


# ---------------------------------------------------------------------------
# 6. Glacial flow accumulation (GlacialFlowAccumulator.run_step)
# ---------------------------------------------------------------------------
def accumulate_glacial_flow(surface, surface_upward, zELA, beta, runoff,
                            cell_area, width_hack_k, width_hack_p, shape,
                            stack, receivers, nb_receivers, weights, lengths,
                            basin):
    """Accumulate water flux, ice flux, drainage area + the routing-topology
    outputs (basin_ids, receivers_2d, stack_2d). Body of
    ``GlacialFlowAccumulator.run_step``. ``surface`` is the post-uplift routing
    surface (``zs_route``); the ELA-relative mass balance ``b(z)`` is evaluated
    on the PRE-uplift climate surface ``z_clim = surface - surface_upward`` (no
    O(U*dt) bias). Returns ``(ice_flux, water_flux, area, basin_ids,
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
    # Above ELA, snow falls on the cell, not the glacier's footprint.
    accum_area = np.where(z_clim < zELA, wide_area, cell_area)

    # 3. Source field for ice accumulation, capped at the cell's precip flux.
    field_ice = beta * (z_clim - zELA) * accum_area
    field_ice = np.where(field_ice > field, field, field_ice)

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


# ---------------------------------------------------------------------------
# 7. Mode A step (GlacialSPLModeA: kernel + border-H closure + commit)
# ---------------------------------------------------------------------------
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
    is read only on the D-inf branch (may be None under SFR)."""
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
            if k == 1:
                m &= nb == 2
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
                   hc_over_H, shape):
    """Mode A erosion step (ice-surface state). Body of ``GlacialSPLModeA``:
    dispatch to the routing-specific mode-A skeleton (mutating a fresh copy of
    the surface + H), solve the border-H closure, then commit. Returns
    ``(z_eroded, H_new, bedrock_surface, erosion, denudation)`` — ``erosion ==
    denudation`` (mode A erodes the ice surface as its single hc-invariant
    state)."""
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
                     parallel, wrap_y, wrap_x):
    """No-carve mode-B kernel dispatch (SFR or D-inf). Mutates ``zb_flat`` and
    ``H_flat`` in place; returns the kernel's ``surface_out`` (= zb + hc*H).

    FIREWALL: the kernel reconstructs its OWN ``zs = zb + hc*H`` from the raw H
    for every closure and erosion slope — it never sees the relaxed/fabricated
    routing surface. De-squeezes the size-1 clock slices of ``border_bed_uplift``
    and ``bl`` inside, so the shell and the driver share the treatment."""
    ny, nx = int(shape[0]), int(shape[1])
    dx_cell = float(dx)
    dy_cell = float(dy)
    surface_out = np.empty_like(zb_flat)
    # Drop the leading size-1 tstep dim left by xsimlab on a (nt, y, x) slice.
    bbu = np.asarray(border_bed_uplift, dtype=np.float64)
    if bbu.ndim == 3:
        bbu = bbu[0]
    bbu_flat = np.broadcast_to(bbu, (ny, nx)).ravel()
    # Per-step water-line datum (the clock strips the 'tstep' dim, so bl is a
    # scalar each step) + the global flotation gate and its ramp width.
    bl = float(np.asarray(bl).ravel()[-1])
    gate = bool(gate)
    ramp = float(ramp)
    par = bool(parallel)
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
            bl, gate, ramp, par)
    else:
        _glac_fast_solve_modeB_sfr(
            zb_flat, ice_flux.ravel(), water_flux.ravel(),
            H_flat, surface_out, law_code, gp,
            dt, lengths, stack, receivers,
            ny, nx, dx_cell, dy_cell, bbu_flat, bl, gate, ramp,
            wrap_y, wrap_x, par)
    return surface_out


# ---------------------------------------------------------------------------
# 9. Sub-grid width carve (GlacialSPLModeC._carve_bed)
# ---------------------------------------------------------------------------
def carve_bed(zb_flat, H_flat, surface_out, zb_pre, receivers, alpha_g,
              hc_over_H, widening_factor, shape, dx, dy, wrap_y, wrap_x,
              offsets=None, D=None, SRC=None, zb_kern=None):
    """Apply the sub-grid width carve (see :mod:`siim._core.carve`) to the
    post-kernel bed ``zb_flat`` IN PLACE. ``zb_pre`` is the pre-kernel bed (the
    denudation datum + descent-cap origin); ``surface_out`` the kernel's
    reconstructed ice surface (updated for carved cells). Routing-agnostic:
    receivers enter only through the border marker ``rec[i] == i`` (self-receiving
    = base-level border). ``offsets/D/SRC/zb_kern`` are optional reusable scratch
    buffers (an optimization, not semantics — allocated fresh if omitted)."""
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
    _carve_subgrid_width(zb_flat, zb_kern, zb_pre, H_flat,
                         surface_out, rec_marker, D, SRC,
                         offsets, widening_factor, hc_over_H)


# ---------------------------------------------------------------------------
# 10. Flow routing (fill-then-route). Two in-house producers on the SAME
#     eps-filled surface: the D8 single-flow router (route_d8, the framework-free
#     replacement for fastscape's fortran SingleFlowRouter) and the D-inf
#     Tarboton router (route_dinf). Both derive the interior/boundary mask
#     directly from border_status (Map 3 §4) — no fortran call.
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


def route_dinf(elevation, shape, dx, dy, border_status):
    """D-infinity flow directions on the eps-filled surface. Body of
    ``DinfFlowRouter.run_step`` — fully fortran-free (S4): the interior/boundary
    mask comes from ``border_status`` directly (Map 3 §4, provably identical to
    the old fortran ``sfr_rec != i`` mask), and ``basin`` from the in-house
    outlet labeling. Returns ``(receivers, weights, lengths, nb_receivers,
    stack, basin)`` — ``receivers``/``weights``/``lengths`` ``(n, 2)``,
    ``basin`` ``(ny, nx)``."""
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
    d8_rec = np.empty(nn, dtype=np.int64)
    d8_len = np.empty(nn, dtype=np.float64)
    _d8_receivers(z_route, ny, nx, float(dx), float(dy), interior,
                  d8_rec, d8_len, wrap_y, wrap_x)
    d8_stack = np.empty(nn, dtype=np.int64)
    _d8_stack(d8_rec, nn, d8_stack)
    basin = np.empty(nn, dtype=np.int64)
    _d8_basin(d8_rec, d8_stack, nn, basin)
    return receivers, weights, lengths, nb_receivers, stack, basin.reshape((ny, nx))


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


# ---------------------------------------------------------------------------
# 12. Glacial flexure (GlacialFlexure.run_step; fortran solve injected)
# ---------------------------------------------------------------------------
def glacial_flexure_step(elevation, denudation, surface_upward, ice_thickness,
                         alpha_g, cell_area, lithos_density, asthen_density,
                         e_thickness, ibc, shape, length, col_prev, ice_load,
                         flexure_solve):
    """Incremental flexural isostasy: rock unloading (``surface_upward -
    denudation``) plus, when ``ice_load``, the per-step glacial ice load
    ``(rho_ice/lithos)*d(col)`` with ``col = alpha_g*H**2/L`` (hc-free,
    mass-conserving). Body of ``GlacialFlexure.run_step``; the biharmonic plate
    solve is INJECTED as ``flexure_solve(elev_post, elev_eq, nx, ny, xl, yl,
    lithos, asthen, Te, ibc)`` (fortran ``fs.flexure`` at S1, in-house FFT
    later — keeps this module framework-free). Returns ``(rebound, col_new)``;
    the caller keeps ``col_new`` as the cross-step ``_col_prev``."""
    ny, nx = shape
    yl, xl = length

    lithos_density = np.broadcast_to(lithos_density, shape).flatten()
    elevation_eq = elevation.flatten()
    diff = (surface_upward - denudation).ravel()
    # Mass-conserving glacial ice load: the channel cross-section alpha_g*H**2
    # (= Qg/V, hc-free) carried over a cell-sized length L -> areal column
    # alpha_g*H**2 / L, with L = sqrt(cell_area). Width-aware (∝ H**2).
    L = float(cell_area) ** 0.5
    col = float(alpha_g) * ice_thickness ** 2 / L
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
    )
    rebound = (elevation_post - elevation_pre).reshape(shape)
    col_new = col.copy()
    return rebound, col_new


# ---------------------------------------------------------------------------
# 13. Vertical-motion composition (stock fastscape group-sum; written fresh
#     for the future standalone driver — NOT wired into the xsimlab model at S1,
#     which keeps using the stock TectonicForcing/TotalErosion/TotalVerticalMotion).
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
