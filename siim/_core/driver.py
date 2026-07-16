"""The in-house time loop for the standalone siim 2D model.

Owns the merged step chain and the two-cadence snapshot that xsimlab's driver
performed for the ``siim.fastscape`` adapter path, calling the SAME
framework-free step functions (:mod:`siim._core.step`) the ``@xs.process`` shells
call — one implementation, two front ends. See Map 1 §1 (step order) and Map 2
(driver / output I/O) of ``docs/dev/standalone_migration_maps.md``.

Merged step order (a valid topological sort of the composition graph)::

    uplift -> tectonics -> surf2erode -> route -> accumulate -> kernel ->
    diffusion -> flexure -> (erosion sum + vertical-motion compose) -> sediment
    -> [SNAPSHOT] -> finalize (topo += surface_up)

Snapshot-timing invariant (load-bearing, Map 2 §2): intermediate frames are
captured AFTER the step but BEFORE the uplift-committing finalize; the LAST frame
(``out_idx[-1] == nt-1``) is captured AFTER the loop (post the final finalize).
Frame 0's ``topography__elevation`` is thus the initial topo while ``H`` / flux /
area / erosion already reflect one solved step.

Cross-step state OWNED by the driver: the ``ice_thickness`` one-step lag (the
router/accumulator read ``H(t-1)`` — an ordering artifact reproduced by reading
H into the routing surface before the kernel overwrites it), the ``_H_eff`` EMA
carry (``routing_relax``), the flexure ``_col_prev``, the sediment ``_cum``, and
the ``topography`` state (committed only at finalize).

FIREWALL (Map 1 §2): the relaxed / fabricated routing surface (``zs_route``)
reaches ONLY the router graph, the mass-balance surface, and the ice-surface
hillslope diffusion — the mode-B kernel reconstructs its OWN ``zs = zb + hc*H``
from raw H for every closure/erosion slope.

Routing is INJECTED as a callable (``cfg.route``) — the fortran backend at S3,
the in-house D8 producer at S4 — so this module stays framework-free (no
xsimlab / fastscape / fastscapelib import). The flexure plate solve is injected
too (``cfg.flexure_solve``); the in-house hillslope diffuser
(:func:`siim._core.hillslope.diffuse`) is called directly.
"""
import numpy as np
from tqdm import tqdm

from .hillslope import diffuse as _inhouse_diffuse
from .outputs import output_spec, allocate_buffers
from .step import (
    ema_thickness, routing_surface, _fabricate_trunk_surface,
    accumulate_glacial_flow, run_modeA_step, run_modeB_kernel, carve_bed,
    accumulate_sediment, glacial_flexure_step, sum_erosion,
    compose_vertical_motion,
)


def _edge_border_mask(border_status, shape):
    """Non-looped domain edges (base-level borders), as the trunk-surface
    fabrication excludes them (TrunkSurfaceToErode.initialize)."""
    ny, nx = shape
    bs = list(np.broadcast_to(border_status, 4))   # [left, right, top, bottom]
    border = np.zeros((ny, nx), dtype=bool)
    if bs[0] != 'looped': border[:,  0] = True
    if bs[1] != 'looped': border[:, -1] = True
    if bs[2] != 'looped': border[0,  :] = True
    if bs[3] != 'looped': border[-1, :] = True
    return border


def _slice_forcing(series, scalar, k):
    """Per-step forcing value: ``series[k]`` when a clock series is present, the
    static scalar/field otherwise (the driver indexes ``arr[k]`` directly,
    replacing xsimlab's ``(('tstep',), arr)`` groupby slice, Map 2 §4)."""
    return series[k] if series is not None else scalar


def run_loop(cfg):
    """Run the merged in-house time loop; return the packed ``ds_out`` step
    buffers (a ``{name: (nt_out, ny, nx) ndarray}`` dict per the output spec).
    ``cfg`` is the resolved-parameter bundle assembled by
    :meth:`siim.siim2d.siim._run_inhouse` (see that method for the field
    contract)."""
    ny, nx = int(cfg.ny), int(cfg.nx)
    shape = (ny, nx)
    length = (cfg.yl, cfg.xl)
    hc = cfg.hc_over_H

    spec = output_spec(cfg.mode, cfg.flexure, cfg.sediment)
    buffers = allocate_buffers(spec, cfg.nt_out, ny, nx)

    # --- cross-step state owned by the driver ---
    topo = np.array(cfg.initial_surface, dtype=np.float64).reshape(shape).copy()
    H = np.zeros(shape)                       # ice_thickness init (siim2d.py:890)
    H_eff = None                              # routing_relax EMA carry (seed None)
    col_prev = np.zeros(shape) if cfg.flexure else None
    cum = np.zeros(shape) if cfg.sediment else None

    is_mode_b = cfg.mode == 'B'
    # --- persistent per-run scratch (allocation only; no semantic state) ---
    if is_mode_b and cfg.trunk_surface:
        tb_border = _edge_border_mask(cfg.border_status, shape)
        tb_off = np.empty(shape); tb_D = np.empty(shape)
        tb_SRC = np.empty(shape, dtype=np.int64)
    if is_mode_b and cfg.carve:
        cv_off = np.empty(shape); cv_D = np.empty(shape)
        cv_SRC = np.empty(shape, dtype=np.int64); cv_zbk = np.empty(ny * nx)

    t = np.asarray(cfg.t, dtype=np.float64)
    out_idx = np.asarray(cfg.out_idx, dtype=np.int64)
    nt = int(cfg.nt)
    cur = {}                                  # step buffers held for the last frame
    bar = tqdm(total=nt - 1) if cfg.progress_bar else None
    jf = 0

    for k in range(nt - 1):
        dt = float(t[k + 1] - t[k])

        # --- forcing slices (arr[k] direct indexing) ---
        uplift = cfg.uplift_fn(k, dt)              # (ny, nx); block or wave
        surface_forcing = uplift                  # TectonicForcing group sum (1 member)
        zELA_k = _slice_forcing(cfg.zELA_series, cfg.zELA, k)
        runoff_k = _slice_forcing(cfg.runoff_series, cfg.P, k)
        bl_k = _slice_forcing(cfg.bl_series, cfg.bl, k)
        bbu_k = _slice_forcing(cfg.bbu_series, cfg.bbu_static, k)

        # --- surf2erode (post-uplift routing/mass-balance surface) ---
        if is_mode_b:
            post_uplift = topo + surface_forcing
            H_eff = ema_thickness(H, H_eff, cfg.routing_relax)   # ONE EMA update
            zs_route = routing_surface(post_uplift, hc, H_eff)
            if cfg.trunk_surface:
                zb_pu = zs_route - hc * H_eff      # match the shell subtraction exactly
                zs_route = _fabricate_trunk_surface(
                    zs_route, zb_pu, H_eff, tb_border, cfg.alpha_g,
                    cfg.dx, cfg.dy, cfg.trunk_dip_k, cfg.trunk_dip_floor,
                    tb_off, tb_D, tb_SRC, cfg.wrap_y, cfg.wrap_x)
        else:   # mode A: SurfaceAfterTectonics (plain post-uplift surface)
            zs_route = topo + surface_forcing

        # --- flow directions (injected backend) + accumulation ---
        rt = cfg.route(zs_route)
        (ice_flux, water_flux, area, basin_ids,
         receivers_2d, stack_2d) = accumulate_glacial_flow(
            zs_route, surface_forcing, zELA_k, cfg.beta, runoff_k, cfg.cell_area,
            cfg.width_hack_k, cfg.width_hack_p, shape,
            rt.stack, rt.receivers, rt.nb_receivers, rt.weights, rt.lengths,
            rt.basin)

        # --- erosion kernel (writes H(t); FIREWALL: raw H, own zs) ---
        if is_mode_b:
            zb_flat = topo.flatten()               # pre-uplift bed (fresh copy)
            H_flat = H.flatten()                   # raw lagged H(t-1) (fresh copy)
            zb_in = zb_flat.copy()
            surface_out = run_modeB_kernel(
                zb_flat, H_flat, ice_flux, water_flux, cfg.law_code, cfg.gp, dt,
                rt.stack, rt.receivers, rt.nb_receivers, rt.weights, rt.lengths,
                shape, cfg.dx, cfg.dy, bbu_k, bl_k, cfg.flotation_gate,
                cfg.flotation_ramp, cfg.parallel_erode, cfg.wrap_y, cfg.wrap_x)
            if cfg.carve:
                carve_bed(
                    zb_flat, H_flat, surface_out, zb_in, rt.receivers,
                    cfg.alpha_g, hc, cfg.widening_factor, shape, cfg.dx, cfg.dy,
                    cfg.wrap_y, cfg.wrap_x, cv_off, cv_D, cv_SRC, cv_zbk)
            H = H_flat.reshape(shape)              # H(t)
            denudation = (zb_in - zb_flat).reshape(shape)
            glacial_erosion = denudation
            bedrock_surface = None
        else:
            (_z_eroded, H, bedrock_surface,
             glacial_erosion, denudation) = run_modeA_step(
                zs_route, H, ice_flux, water_flux, cfg.law_code, cfg.gp, dt,
                rt.stack, rt.receivers, rt.nb_receivers, rt.weights, rt.lengths,
                hc, shape)
        erosion_rate = glacial_erosion / dt

        # --- hillslope diffusion of the surf2erode surface (in-house ADI) ---
        diffused = _inhouse_diffuse(zs_route, cfg.D, dt, nx, ny, cfg.xl, cfg.yl,
                                    cfg.ibc)
        diff_erosion = zs_route.reshape(shape) - diffused

        # --- flexure (optional) ---
        if cfg.flexure:
            rebound, col_prev = glacial_flexure_step(
                topo, denudation, surface_forcing, H, cfg.alpha_g, cfg.cell_area,
                cfg.lithos_density, cfg.asthen_density, cfg.e_thickness, cfg.ibc,
                shape, length, col_prev, cfg.ice_load, cfg.flexure_solve)
        else:
            rebound = None

        # --- erosion group sum + vertical-motion composition ---
        erosion_total = sum_erosion(glacial_erosion, diff_erosion)
        surface_up, _bedrock_up = compose_vertical_motion(
            surface_forcing, surface_forcing, rebound, erosion_total)

        # --- sediment (optional) ---
        if cfg.sediment:
            flux = accumulate_sediment(
                denudation, cfg.cell_area, rt.stack, rt.receivers,
                rt.nb_receivers, rt.weights, shape)
            cum = cum + flux                       # new array (prior snapshots valid)

        # --- stage this step's output buffers ---
        cur['topography__elevation'] = topo       # PRE-finalize for in-loop frames
        cur['glacial_spl__ice_thickness'] = H
        cur['glacial_flow__ice_flux'] = ice_flux
        cur['glacial_flow__water_flux'] = water_flux
        cur['glacial_flow__area'] = area
        cur['glacial_flow__basin_ids'] = basin_ids
        cur['glacial_flow__receivers_2d'] = receivers_2d
        cur['glacial_flow__stack_2d'] = stack_2d
        cur['glacial_spl__erosion_rate'] = erosion_rate
        cur['glacial_spl__denudation'] = denudation
        cur['uplift__uplift'] = surface_forcing
        if cfg.mode == 'A':
            cur['glacial_spl__bedrock_surface'] = bedrock_surface
        if cfg.sediment:
            cur['sediment__flux'] = flux
            cur['sediment__cumulative'] = cum
        if cfg.flexure:
            cur['flexure__rebound'] = rebound

        # --- snapshot frames at this master step (BEFORE the finalize commit) ---
        while jf < cfg.nt_out and out_idx[jf] == k:
            _snapshot(buffers, jf, cur)
            jf += 1

        # --- finalize: the single state commit (SurfaceTopography.finalize_step) ---
        topo = topo + surface_up
        if bar is not None:
            bar.update(1)

    # --- last frame(s) (out_idx == nt-1): post-finalize topo, last-step buffers ---
    cur['topography__elevation'] = topo
    while jf < cfg.nt_out:
        _snapshot(buffers, jf, cur)
        jf += 1
    if bar is not None:
        bar.close()
    return buffers


def _snapshot(buffers, jf, cur):
    """Copy this frame's step values into the preallocated output buffers (dtype
    cast happens on assignment: int32 buffers, float64 fields)."""
    for name, buf in buffers.items():
        buf[jf] = cur[name]
