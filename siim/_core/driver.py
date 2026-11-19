"""The in-house time loop for the standalone siim 2D model.

Owns the merged step chain and the two-cadence snapshot that xsimlab's driver
performed for the ``siim.fastscape`` adapter path, calling the SAME
framework-free step functions (:mod:`siim._core.step`) the ``@xs.process`` shells
call — one implementation, two front ends. The public model and output
contracts are documented in ``docs/guides/concepts.md`` and
``docs/guides/outputs_and_io.md``.

Merged step order (a valid topological sort of the composition graph)::

    uplift -> tectonics -> inner passes (surf2erode -> route -> accumulate ->
    kernel, ``inner_passes`` times) -> mode D's regime read -> carve ->
    diffusion -> flexure -> (erosion sum + vertical-motion compose) -> sediment
    -> [SNAPSHOT] -> finalize (topo += surface_up)

Snapshot-timing invariant (load-bearing, Map 2 §2): intermediate frames are
captured AFTER the step but BEFORE the uplift-committing finalize; the LAST frame
(``out_idx[-1] == nt-1``) is captured AFTER the loop (post the final finalize).
Frame 0's ``topography__elevation`` is thus the initial topo while ``H`` / flux /
area / erosion already reflect one solved step.

Cross-step state OWNED by the driver: the ``ice_thickness`` one-step lag (the
router/accumulator read ``H(t-1)`` — an ordering artifact reproduced by reading
H into the routing surface before the kernel overwrites it), the adaptive
damping presets' per-cell weight and previous move, mode D's relaxed s and
the icy mask it was read on (semantic state: read from the committed H AND
the bed the kernel just eroded, which the devices themselves shaped), the
confinedness ``s_conf`` the devices take and its smoothed twin for the datum,
the flexure ``_col_prev``, the sediment ``_cum`` (and its per-domain-edge
twin), and the ``topography`` state (committed only at finalize). The loop
returns the physical part of it (:data:`RESTART_GROUPS`) and starts on it
when handed back (``cfg.initial_state``): an exact restart.

Valley <-> sheet devices (mode D, ``regime_switch``; the five devices are
described once, at ``constants.HC_SHEET``, the reading at
``constants.REGIME_ICY_H``). What the driver owns: once per step, after the
passes, the binary class is read (:mod:`siim._core.regime`) on the COMMITTED H
and the bed after the kernel's erosion, before the carve, from the discharge of
a separate routing pass over the slab ``zb + H`` (:func:`_diagnostic_discharge`);
s moves a fixed fraction of the way toward the class, borders inherit their
dominant donor's, and the datum takes s smoothed over the icy cells, its
borders again taking their donor's value. The carve
consumes s at once, so a trough is never cut into a patch that just became a
sheet; the routing surface, the accumulator and the kernel consume it on the
NEXT step, lagged like H itself (s = 1, the valley code, on the first step).
Outside mode D no call changes.

FIREWALL (Map 1 §2), in three clauses. (1) The routing surface
``zs_route = routing_surface(post_uplift, hc, x)``, built on the lagged pass
iterate, reaches the router graph, the mass-balance surface and the ice-surface
hillslope diffusion. (2) The fabricated trunk V (``trunk_surface``) and the
filtered surface (``routing_filter_factor`` > 0; exclusive outside mode D, where
both apply, the filter acting on the fabricated V) are
``zs_router``: they construct the router input ALONE. Nothing routed reaches a
flux closure, the carve, the flexure load or the outputs — the mode-B kernel
reconstructs its OWN ``zs = zb + hc*H`` from raw H for every closure/erosion
slope; under the within-step iteration (``inner_passes`` / ``inner_relax``)
"raw H" is the RAW PASS ITERATE, and the carve, the flexure load and the
outputs take the COMMITTED state. The routing-derived graph and fluxes remain
inputs downstream. Consumer audit of the two surfaces (2026-09-22):
``zs_router`` -> ``cfg.route`` only; ``zs_route`` -> ``accumulate_glacial_flow``,
the mode-A step (mode A: the plain post-uplift surface, no device applies),
``_inhouse_diffuse`` and the ``diff_erosion`` subtraction. The xsimlab adapter
mirrors the two as ``SurfaceToErode.elevation`` and the ``router_surface``
group (``TrunkSurfaceToErode.routing_elevation``). (3) Mode D's diagnostic
slab ``zb + H``, its filtered twin, its graph and its flux
(:func:`_diagnostic_discharge`) feed the channel/sheet class ONLY.

Routing is INJECTED as a callable (``cfg.route``) — the fortran backend at S3,
the in-house D8 producer at S4 — so this module stays framework-free (no
xsimlab / fastscape / fastscapelib import). The flexure plate solve is injected
too (``cfg.flexure_solve``); the in-house hillslope diffuser
(:func:`siim._core.hillslope.diffuse`) is called directly.
"""
import numpy as np
from tqdm import tqdm

from ..constants import (ADAPTIVE_GROWTH, ADAPTIVE_SHRINK, INNER_DAMPING,
                         INNER_DAMPING_FLOORS, INNER_RELAX_BARE_H, REGIME_ICY_H,
                         TOE_SLOPE_CORRECTION_2D, FOOTPRINT_FROM_H,
                         FOOTPRINT_WIDTH_FACTOR)
from .hillslope import diffuse as _inhouse_diffuse
from .outputs import output_spec, allocate_buffers
from .regime import channel_class, swap, smooth_icy
from .step import (
    routing_surface, filter_routing_surface,
    _fabricate_trunk_surface, border_inherits,
    hc_cell_field,
    accumulate_glacial_flow, accumulate_sheet_ice_flow,
    run_modeA_step, run_modeB_kernel, carve_bed,
    accumulate_sediment, edge_sediment, glacial_flexure_step, sum_erosion,
    compose_vertical_motion, transect_flux, toe_phi, SIDE_SLICES,
)


def _edge_border_mask(border_status, shape):
    """Non-looped domain edges (base-level borders), as the trunk-surface
    fabrication excludes them (TrunkSurfaceToErode.initialize)."""
    ny, nx = shape
    bs = list(np.broadcast_to(border_status, 4))   # [left, right, bottom, top]: 2 = row 0
    border = np.zeros((ny, nx), dtype=bool)
    if bs[0] != 'looped': border[:,  0] = True
    if bs[1] != 'looped': border[:, -1] = True
    if bs[2] != 'looped': border[0,  :] = True
    if bs[3] != 'looped': border[-1, :] = True
    return border


def _diagnostic_discharge(cfg, zb, H, surface_forcing, zELA, runoff, shape):
    """Mode D's reading discharge per unit width ``q = Q/(alpha_g*H)`` (0 where
    ``H <= REGIME_ICY_H``): one routing + accumulation pass over the plain slab
    ``zb + H`` with every cell treated as sheet (the filter on the full
    thickness, the footprint the cell), on the step's own forcing. FIREWALL:
    its surface and flux feed the class reading ONLY; the router hands back
    fresh arrays, so the step's own graph is untouched."""
    zs = zb + H
    zr = (filter_routing_surface(zs, H, cfg.dx, cfg.routing_filter_factor,
                                 cfg.wrap_y, cfg.wrap_x)
          if cfg.routing_filter_factor else zs)
    rt = cfg.route(zr)
    Q = accumulate_sheet_ice_flow(
        zs, surface_forcing, zELA, cfg.beta, runoff, cfg.cell_area, shape,
        rt.stack, rt.receivers, rt.nb_receivers, rt.weights)
    icy = H > REGIME_ICY_H
    return np.where(icy & (Q > 0), Q / (cfg.alpha_g * np.where(icy, H, 1.0)), 0.0)


def _slice_forcing(series, scalar, k):
    """Per-step forcing value: ``series[k]`` when a clock series is present, the
    static scalar/field otherwise (the driver indexes ``arr[k]`` directly,
    replacing xsimlab's ``(('tstep',), arr)`` groupby slice, Map 2 §4)."""
    return series[k] if series is not None else scalar


def _bl_field(bl_sides, shape, k):
    """Per-node water datum for step ``k`` (PER-SIDE ``bl``): each
    ``fixed_value`` border node carries its own side's datum; interior entries
    are placeholders (0) that the mode-B kernel overwrites with the datum of
    the outlet its basin drains to. Corner nodes touched by two fixed sides
    take the x-side (left/right) value — the x-sides are written last.
    ``bl_sides`` is the 4-list of ``(scalar, series)`` entries parsed by
    ``siim.siim2d.siim.set_and_check_parameters`` (``None`` off fixed sides)."""
    field = np.zeros(shape)
    for s in (2, 3, 0, 1):                      # bottom, top, then left, right
        if bl_sides[s] is not None:
            scalar, series = bl_sides[s]
            field[SIDE_SLICES[s]] = _slice_forcing(series, scalar, k)
    return field.ravel()


def _adaptive(cfg):
    """The adaptive damping presets are on (constants.INNER_DAMPING_FLOORS)."""
    return getattr(cfg, 'inner_damping', INNER_DAMPING) in INNER_DAMPING_FLOORS


def _regime(cfg):
    """Mode D's valley <-> sheet devices are on (constants.HC_SHEET)."""
    return cfg.mode == 'B' and bool(getattr(cfg, 'regime_switch', False))


#: The cross-step state a restart carries (:meth:`siim.siim2d.siim.restart_state`),
#: grouped by the device that owns it: ``group: (in use for cfg?, {key: (cold
#: fill, dtype, lo, hi)})``. A group is carried whole or not at all
#: (:func:`check_restart_groups`), a run that uses a group its start state
#: lacks starts that group cold, and ``lo``/``hi`` (inclusive, None = open) are
#: the values a carried array may take. The topography's cold start is the
#: run's initial surface (``cfg.initial_surface``), no fill. The sediment
#: running totals are outputs and count from each run's start.
RESTART_GROUPS = {
    'surface': (lambda cfg: True, {'topography': (None, np.float64, None, None)}),
    'ice': (lambda cfg: True, {'ice_thickness': (0.0, np.float64, 0.0, None)}),
    # the plate's previous ice column
    'flexure': (lambda cfg: bool(cfg.flexure),
                {'flexure_column': (0.0, np.float64, 0.0, None)}),
    # the adaptive presets' per-cell w (in [any preset's floor, 1]) and d_prev
    'adaptive': (_adaptive,
                 {'damping_weight': (1.0, np.float64, min(INNER_DAMPING_FLOORS.values()), 1.0),
                  'damping_prev_move': (0.0, np.float64, None, None)}),
    # mode D's relaxed s, the icy mask it was read on, s_conf and its smoothed
    # twin for the datum: valley (1) everywhere and nothing icy on a cold start
    'regime': (_regime,
               {'regime_s': (1.0, np.float64, 0.0, 1.0),
                'regime_icy': (False, np.bool_, None, None),
                'regime_s_conf': (1.0, np.float64, 0.0, 1.0),
                'regime_s_datum': (1.0, np.float64, 0.0, 1.0)}),
}


def check_restart_groups(state):
    """Raise ValueError unless each of :data:`RESTART_GROUPS` is in ``state``
    whole or not at all."""
    for name, (_in_use, fields) in RESTART_GROUPS.items():
        if 0 < sum(k in state for k in fields) < len(fields):
            raise ValueError(f"state carries part of its {name!r} group: it needs "
                             f"all of {list(fields)} or none")


def _start_state(cfg, init, shape):
    """The loop's start arrays for every group ``cfg`` uses: ``init[key]`` where
    the start state carries it (no copy when its dtype already matches: the
    loop only rebinds these names, never writes into them), else cold."""
    start = {}
    for in_use, fields in RESTART_GROUPS.values():
        if not in_use(cfg):
            continue
        for key, (cold, dtype, _lo, _hi) in fields.items():
            if key in init:
                value = init[key]
            elif cold is None:                    # the topography
                value = cfg.initial_surface
            else:
                value = np.full(shape, cold, dtype=dtype)
            start[key] = np.asarray(value, dtype=dtype).reshape(shape)
    return start


def run_loop(cfg):
    """Run the merged in-house time loop; return ``(buffers, end_state)``: the
    packed ``ds_out`` step buffers (a ``{name: ndarray}`` dict, one array per
    output-spec row at that row's dims) and the cross-step state the loop
    ends on (a ``{name: ndarray}`` dict of the keys of every
    :data:`RESTART_GROUPS` group the run uses), from which a restarted loop
    continues this one.
    ``cfg`` is the resolved-parameter bundle assembled by
    :meth:`siim.siim2d.siim._run_inhouse` (see that method for the field
    contract); its optional ``initial_state`` (None, or a dict keyed as
    ``end_state``) is the state the loop starts on, each group it lacks
    starting cold (the topography from ``cfg.initial_surface``)."""
    ny, nx = int(cfg.ny), int(cfg.nx)
    shape = (ny, nx)
    length = (cfg.yl, cfg.xl)
    hc = cfg.hc_over_H
    # Mode D's valley <-> sheet devices (constants.HC_SHEET; validated by the
    # facade in _core.regime.check_regime_grid). Off: s_conf stays None
    # and every device call below is today's, bit-for-bit.
    regime = _regime(cfg)
    bed_bar = cfg.channel_bed_bar if regime else None   # mode D's dial (resolved by the facade)

    track_sediment = cfg.sediment or cfg.sediment_edge   # both need the routed flux

    # --- cross-step state owned by the driver (cold, or carried by a restart) ---
    init = getattr(cfg, 'initial_state', None) or {}
    check_restart_groups(init)                    # (the facade checks it too)
    start = _start_state(cfg, init, shape)
    topo = start['topography']
    # ice_thickness: zero, as the adapter path's input (mode A overwrites every cell)
    H = start['ice_thickness']
    col_prev = start['flexure_column'] if cfg.flexure else None
    cum = np.zeros(shape) if cfg.sediment else None
    # mode D's s (valley on the first step), its smoothed twin (borders
    # inherit) for the datum, the relaxed s and the icy mask it was read on
    s_conf = start['regime_s_conf'] if regime else None
    s_datum = start['regime_s_datum'] if regime else None
    s_rel = start['regime_s'] if regime else None
    icy_rel = start['regime_icy'] if regime else None
    hc_route = hc                                 # the datum the routing surface is built on
    edge_cum = np.zeros(4) if cfg.sediment_edge else None   # NaN off fixed sides

    is_mode_b = cfg.mode == 'B'
    closure_flux = getattr(cfg, 'closure_flux', 'cell')
    closure_transect = is_mode_b and closure_flux != 'cell'
    closure_min_rule = closure_flux == 'transect_min'
    # gamma effective-slope normalisation (constants.MFD_GAMMA_FLOOR): resolved
    # by the facade (_core.router.SLOPE_NORMALISED_ROUTERS -> on for D-inf and
    # MFD, off for single-flow, where gamma == 1 identically).
    slope_normalise = bool(getattr(cfg, 'slope_normalise', False))
    # Within-step damped iteration (constants.INNER_PASSES / INNER_RELAX;
    # validated by the facade in _core.router.check_inner_loop_config).
    inner_passes = int(cfg.inner_passes)
    inner_relax = float(cfg.inner_relax)
    # The adaptive presets (constants.INNER_DAMPING_FLOORS; inner_relax carries
    # the floor): per-cell weight w and previous move, cross-step state.
    adaptive = _adaptive(cfg)
    if adaptive:
        w_cell, d_prev = start['damping_weight'], start['damping_prev_move']
    # Routing-surface spatial filter (constants.ROUTING_FILTER_FACTOR; validated
    # by the facade in _core.router.check_routing_filter_config). 0 = off.
    routing_filter_factor = float(cfg.routing_filter_factor)
    # EXPERIMENTAL 2D toe slope correction (constants.TOE_SLOPE_CORRECTION_2D):
    # the kernels read the accumulator's pre-clamp ice flux and the law's toe
    # exponent; the dict keeps the factor's work arrays. Off: no call below
    # changes.
    toe = None
    if getattr(cfg, 'toe_slope_correction_2d', TOE_SLOPE_CORRECTION_2D):
        toe = (np.empty(ny * nx), toe_phi(cfg.law_code, cfg.gp), {})
    # EXPERIMENTAL ablation footprint (constants.FOOTPRINT_FROM_H and
    # FOOTPRINT_WIDTH_FACTOR): the below-ELA footprint width alpha_g*x in place
    # of the Hack width, and its factor f. Off and f = 1: no call below changes.
    footprint_from_H = getattr(cfg, 'footprint_from_H', FOOTPRINT_FROM_H)
    footprint_f = getattr(cfg, 'footprint_width_factor', FOOTPRINT_WIDTH_FACTOR)
    spec = output_spec(cfg.mode, cfg.flexure, cfg.sediment, cfg.sediment_edge,
                       regime_switch=regime, closure_flux=closure_flux,
                       inner_relax=inner_relax, save_closure_pair=cfg.save_closure_pair)
    buffers = allocate_buffers(spec, cfg.nt_out, ny, nx)
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

    bbu_fn = getattr(cfg, 'bbu_fn', None)   # per-step border uplift (the escarpment wave)
    for k in range(nt - 1):
        dt = float(t[k + 1] - t[k])

        # --- forcing slices (arr[k] direct indexing) ---
        uplift = cfg.uplift_fn(k, dt)              # (ny, nx); block or wave
        surface_forcing = uplift                  # TectonicForcing group sum (1 member)
        zELA_k = _slice_forcing(cfg.zELA_series, cfg.zELA, k)
        runoff_k = _slice_forcing(cfg.runoff_series, cfg.P, k)
        bl_k = (_bl_field(cfg.bl_sides, shape, k) if cfg.bl_sides is not None
                else _slice_forcing(cfg.bl_series, cfg.bl, k))
        bbu_k = bbu_fn(k, dt) if bbu_fn is not None else _slice_forcing(cfg.bbu_series, cfg.bbu_static, k)

        if is_mode_b:
            post_uplift = topo + surface_forcing
            zb_in = post_uplift.flatten()     # pre-erosion bed for the carve + denudation
            if regime:
                # the datum the passes build their columns on: from the s of the
                # previous step's committed state; fixed across the passes
                hc_cell = hc_cell_field(s_datum, hc, cfg.hc_sheet)
                hc_route = hc_cell

        # --- the within-step damped iteration (constants.INNER_PASSES) --------
        # n passes of  x <- x + omega*(F(x) - x)  over ONE surf2erode -> route ->
        # accumulate -> kernel evaluation F, the first n-1 at dt = 0 (rehearsals:
        # the kernel moves no bed and runs no thickness diffusion there) and the
        # last at the real dt. Uplift, carve, erosion rate, hillslope diffusion,
        # flexure, sediment, staging and the topography commit all stay OUTSIDE,
        # once per step. (1, 1.0) is a single fresh evaluation.
        x = H                                     # the thickness iterate
        # the routing-topology outputs (basin_ids, receivers_2d, stack_2d) feed
        # nothing but the frames: built on the last pass of a step a frame saves
        # (the snapshot loop's own condition below), and of the last step,
        # whose state is the one staged after the loop
        topo_out = k == nt - 2 or (jf < cfg.nt_out and out_idx[jf] == k)
        for p in range(inner_passes):
            dt_pass = dt if p == inner_passes - 1 else 0.0

            # --- surf2erode (post-uplift routing/mass-balance surface) ---
            if is_mode_b:
                zs_route = routing_surface(post_uplift, hc_route, x)   # zb + hc_cell*H
            else:   # mode A: SurfaceAfterTectonics (plain post-uplift surface)
                zs_route = topo + surface_forcing

            # --- the router's surface (FIREWALL, module docstring): the fabricated
            # V and/or the filtered surface (mode B/C; exclusive outside mode D, whose s
            # fades each where the other applies) feed the router ALONE
            zs_router = zs_route
            if cfg.trunk_surface:
                zb_pu = zs_route - hc_route * x  # match the shell subtraction exactly
                # the V blended per SOURCE by s in mode D (inside)
                zs_router = _fabricate_trunk_surface(
                    zs_route, zb_pu, x, tb_border, cfg.alpha_g,
                    cfg.dx, cfg.dy, cfg.trunk_dip_k, cfg.trunk_dip_floor,
                    tb_off, tb_D, tb_SRC, cfg.wrap_y, cfg.wrap_x, s_conf=s_conf)
            if routing_filter_factor and (regime or not cfg.trunk_surface):
                # (exclusive of the V outside mode D, as the facade
                # guards; kept here for direct run_loop callers)
                # the filter's thickness fades with s: a valley cell opens no
                # window and weighs nothing in a sheet neighbour's
                H_f = x if not regime else x * (1.0 - s_conf)
                zs_router = filter_routing_surface(zs_router, H_f, cfg.dx,
                                                   routing_filter_factor,
                                                   cfg.wrap_y, cfg.wrap_x)

            # --- flow directions (injected backend) + accumulation ---
            rt = cfg.route(zs_router)
            (ice_flux, water_flux, area, basin_ids,
             receivers_2d, stack_2d) = accumulate_glacial_flow(
                zs_route, surface_forcing, zELA_k, cfg.beta, runoff_k,
                cfg.cell_area, cfg.width_hack_k, cfg.width_hack_p, shape,
                rt.stack, rt.receivers, rt.nb_receivers, rt.weights, rt.lengths,
                rt.basin if topo_out and p == inner_passes - 1 else None,
                s_conf=s_conf, raw_out=None if toe is None else toe[0],
                width=cfg.alpha_g * x if footprint_from_H else None,
                width_factor=footprint_f)

            # --- erosion kernel (solves F(x); FIREWALL: raw iterate, own zs) ---
            if is_mode_b:
                zb_flat = post_uplift.flatten()    # post-uplift bed (kernel contract; fresh copy)
                H_flat = x.flatten()               # the raw pass iterate (fresh copy)
                # EXPERIMENTAL closure flux (constants.CLOSURE_FLUX): the kernel's
                # closures + eroders see the transect flux built from the TRUE routed
                # flux and the RAW pass iterate; routing/outputs/carve/flexure/sediment
                # keep ``ice_flux``. 'cell' = bit-for-bit today's path.
                if closure_transect:
                    Q_close = transect_flux(
                        ice_flux, x, rt.receivers, rt.weights, rt.nb_receivers,
                        shape, cfg.dx, cfg.dy, cfg.alpha_g, cfg.wrap_y,
                        cfg.wrap_x, min_rule=closure_min_rule)
                else:
                    Q_close = ice_flux
                surface_out = run_modeB_kernel(
                    zb_flat, H_flat, Q_close, water_flux, cfg.law_code, cfg.gp,
                    dt_pass, rt.stack, rt.receivers, rt.nb_receivers, rt.weights,
                    rt.lengths, shape, cfg.dx, cfg.dy, bbu_k, bl_k,
                    cfg.flotation_gate, cfg.flotation_ramp, cfg.parallel_erode,
                    cfg.wrap_y, cfg.wrap_x, slope_normalise,
                    hc_cell=hc_cell.ravel() if regime else None, toe=toe)
                # Commit the move. omega == 1 is a DIRECT assignment: evaluating
                # x + 1*(F - x) rounds (1000.0 + 1*(0.1 - 1000.0) != 0.1), which
                # would cost the bit-for-bit default. Never in place — the first
                # pass's iterate aliases the outer H. Below 1, a cell
                # the kernel left bare snaps to 0 (constants.INNER_RELAX_BARE_H).
                F = H_flat.reshape(shape)
                if inner_relax == 1.0 and not adaptive:
                    x = F
                else:
                    d = F - x
                    if adaptive:
                        # w halves where the move reverses, grows elsewhere,
                        # clipped to [floor, 1] (constants.ADAPTIVE_*)
                        w_cell = np.clip(np.where(d * d_prev < 0.0, ADAPTIVE_SHRINK * w_cell,
                                                  ADAPTIVE_GROWTH * w_cell), inner_relax, 1.0)
                        d_prev = d
                    x = x + (w_cell if adaptive else inner_relax) * d
                    x = np.where((F <= 0.0) & (x < INNER_RELAX_BARE_H), 0.0, x)
            else:
                (_z_eroded, x, bedrock_surface,
                 glacial_erosion, denudation) = run_modeA_step(
                    zs_route, x, ice_flux, water_flux, cfg.law_code, cfg.gp,
                    dt_pass, rt.stack, rt.receivers, rt.nb_receivers, rt.weights,
                    rt.lengths, hc, shape, slope_normalise, cfg.dx, cfg.dy,
                    cfg.wrap_y, cfg.wrap_x, toe=toe)
        H = x                                      # H(t) = the committed state

        if is_mode_b:
            if regime:
                # mode D: the class from the COMMITTED H and the bed the
                # kernel just eroded (zb_flat, before the carve), read on the
                # discharge of a separate slab routing pass; s relaxes toward
                # it, borders inherit, the datum takes s smoothed and the
                # border rule again, so a border's datum is its donor's too
                zb_now = zb_flat.reshape(shape)
                q_diag = _diagnostic_discharge(
                    cfg, zb_now, H, surface_forcing, zELA_k, runoff_k, shape)
                chan = channel_class(q_diag, H, zb_now, cfg.dx, cfg.alpha_g,
                                     bed_bar, cfg.wrap_y, cfg.wrap_x)
                icy = H > REGIME_ICY_H
                s_rel = swap(s_rel, icy_rel, chan, icy)
                icy_rel = icy
                s_conf = border_inherits(s_rel, rt.receivers, rt.nb_receivers,
                                         ice_flux)
                s_datum = border_inherits(smooth_icy(s_conf, icy, cfg.wrap_y, cfg.wrap_x),
                                          rt.receivers, rt.nb_receivers, ice_flux)
            if cfg.carve:
                # The carve reads the COMMITTED H (under omega < 1 the kernel's
                # own closure H is not what the state carries). ``surface_out``
                # is write-only from here — the carve updates it for carved
                # cells and the driver drops it — so the kernel's copy is the
                # matching surface. At omega == 1 H.ravel() shares H_flat's
                # buffer (a contiguous view; identity is not relied on).
                carve_bed(
                    zb_flat, H.ravel(), surface_out, zb_in, rt.receivers,
                    cfg.alpha_g, hc, cfg.widening_factor, shape, cfg.dx, cfg.dy,
                    cfg.wrap_y, cfg.wrap_x, cv_off, cv_D, cv_SRC, cv_zbk,
                    s_conf=s_conf)
            denudation = (zb_in - zb_flat).reshape(shape)
            glacial_erosion = denudation
            bedrock_surface = None
        erosion_rate = glacial_erosion / dt

        # --- hillslope diffusion of the surf2erode surface (in-house ADI) ---
        diffused = _inhouse_diffuse(zs_route, cfg.D, dt, nx, ny, cfg.xl, cfg.yl,
                                    cfg.ibc, cfg.wrap_y, cfg.wrap_x)
        diff_erosion = zs_route.reshape(shape) - diffused

        # --- flexure (optional) ---
        if cfg.flexure:
            rebound, col_prev = glacial_flexure_step(
                topo, denudation, surface_forcing, H, cfg.alpha_g, cfg.cell_area,
                cfg.lithos_density, cfg.asthen_density, cfg.e_thickness, cfg.ibc,
                shape, length, col_prev, cfg.ice_load, cfg.flexure_solve,
                s_conf=s_conf, wrap_y=cfg.wrap_y, wrap_x=cfg.wrap_x)
        else:
            rebound = None

        # --- erosion group sum + vertical-motion composition ---
        erosion_total = sum_erosion(glacial_erosion, diff_erosion)
        surface_up, _bedrock_up = compose_vertical_motion(
            surface_forcing, surface_forcing, rebound, erosion_total)

        # --- sediment (optional) ---
        if track_sediment:
            flux = accumulate_sediment(
                denudation, cfg.cell_area, rt.stack, rt.receivers,
                rt.nb_receivers, rt.weights, shape)
            if cfg.sediment:
                cum = cum + flux                   # new array (prior snapshots valid)
            if cfg.sediment_edge:
                edge_flux = edge_sediment(flux, cfg.border_status)
                edge_cum = edge_cum + edge_flux

        # --- stage this step's output buffers ---
        cur['topography__elevation'] = topo       # PRE-finalize for in-loop frames
        cur['glacial_spl__ice_thickness'] = H
        if regime:
            cur['glacial_spl__hc_cell'] = hc_cell  # the datum this step's columns were built on
            cur['glacial_spl__channel'] = chan     # the class read on this step's committed H
        if is_mode_b and cfg.save_closure_pair:
            # the closure's own pair (last pass), what the post-hoc speed
            # inverts; snapshotted only where output_spec names it
            cur['glacial_spl__closure_ice_flux'] = Q_close
            cur['glacial_spl__closure_ice_thickness'] = F
        cur['glacial_flow__ice_flux'] = ice_flux
        cur['glacial_flow__water_flux'] = water_flux
        cur['glacial_flow__area'] = area
        if topo_out:
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
        if cfg.sediment_edge:
            cur['sediment__edge_flux'] = edge_flux
            cur['sediment__edge_cumulative'] = edge_cum
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
    # the state the next step would start on (the loop's own arrays; the
    # facade's restart_state() copies): a restart from it continues this run
    carried = dict(topography=topo, ice_thickness=H, flexure_column=col_prev,
                   regime_s=s_rel, regime_icy=icy_rel, regime_s_conf=s_conf,
                   regime_s_datum=s_datum)
    if adaptive:
        carried.update(damping_weight=w_cell, damping_prev_move=d_prev)
    end_state = {k: carried[k] for in_use, fields in RESTART_GROUPS.values()
                 if in_use(cfg) for k in fields}
    return buffers, end_state


def _snapshot(buffers, jf, cur):
    """Copy this frame's step values into the preallocated output buffers (dtype
    cast happens on assignment: int32 buffers, float64 fields)."""
    for name, buf in buffers.items():
        buf[jf] = cur[name]
