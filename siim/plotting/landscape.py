"""``landscape`` / ``animate_landscape`` — the deluxe atlas-style cartographic
render: supersampled + Gaussian-smoothed terrain, sub-grid across-channel-width
ice (footprint fill — the display dual of the width carve), priority-flooded
lakes, hillshade, contours, and an optional cross-section + hypsometry panel.
Frame + animate only (no viewer, by design). See docs/dev/plotting_plan.md.

``field`` ∈ {``'bedrock'``, ``'bedrock+ice'``, ``'bedrock+lakes'``}.

The numba render backend (footprint ice, smoothing, hillshade, path tracing)
lives in ``_render.py``.
"""

import numpy as np

from ._render import (
    _add_colorbar, output_path,
    _footprint_ice_surface, _smooth_ice_mask, _field_ice_mask,
    _clean_ice_mask, _mean_recent_H, _compute_glacier_field,
    _priority_flood, _shade_rgb_soft,
)


def _resolve_style(style, field, H_threshold, ice_sigma_cells, ice_time_avg,
                   sigma_cells, oversample, hillshade, ice_extent,
                   show_trimline, area_threshold, contour_interval):
    """Resolve every unset (``None``) style-differentiated knob per ``style``;
    an explicit value always wins over the preset.

    ``'smooth'`` (the default) is the cartographic view: bare bed
    (field 'bedrock'), supersampled (oversample 4) + Gaussian-de-staircased
    terrain (sigma_cells = oversample), hillshaded, contoured
    (contour_interval 100 m), ice (when a +ice field is asked for) drawn across
    the sub-grid glacier width (ice_extent 'footprint') with thin ice
    thresholded off (H_threshold 100), no area gate (area_threshold 0), and its
    outline de-staircased (ice_sigma_cells 2 subgrid px).

    ``'raw'`` is the naked model output: bed AND ice by default (field
    'bedrock+ice' — raw is an inspection view, so it shows the ice), one pixel
    per model cell (oversample 1), NO terrain or ice smoothing (sigma_cells 0,
    ice_sigma_cells 0), NO hillshade, NO trimline (show_trimline False), NO
    contours (contour_interval 0), ice shown as the glaciated channel CELLS
    themselves (ice_extent 'cells') with every icy cell kept by thickness
    (H_threshold 0) but small-catchment specks dropped by a
    contributing-area gate (area_threshold 1e6 m², Eric 2026-07-07). Use it to
    inspect exactly what the model committed, unretouched.

    ``ice_time_avg`` resolves to 1 (off) for both — it blends OUTPUT frames,
    which at coarsely-spaced saves double-exposes two glacial epochs; a larger
    ``ice_sigma_cells`` globs nearby ice into blobs and erases 1-cell threads
    (the mask is a 1/2-level set of a blurred indicator, i.e. a closing). Both
    were tried in the smooth preset and REJECTED on use; they stay opt-in for
    densely-sampled animations only. Pinned by test_ice_display.
    """
    if style not in ('raw', 'smooth'):
        raise ValueError(f"style must be 'raw' or 'smooth', got {style!r}")
    raw = style == 'raw'
    if field is None:
        field = 'bedrock+ice' if raw else 'bedrock'
    if H_threshold is None:
        H_threshold = 0.0 if raw else 100.0
    if ice_sigma_cells is None:
        ice_sigma_cells = 0.0 if raw else 2.0
    if ice_time_avg is None:
        ice_time_avg = 1
    if oversample is None:
        oversample = 1 if raw else 4
    if hillshade is None:
        hillshade = False if raw else True
    if ice_extent is None:
        ice_extent = 'cells' if raw else 'footprint'
    if sigma_cells is None:
        sigma_cells = 0.0 if raw else float(oversample)
    if show_trimline is None:
        show_trimline = False if raw else True
    if area_threshold is None:
        # raw drops small-catchment specks (Eric, 2026-07-07); smooth keeps all.
        area_threshold = 1e6 if raw else 0.0
    if contour_interval is None:
        contour_interval = 0.0 if raw else 100.0
    return (field, H_threshold, ice_sigma_cells, ice_time_avg,
            sigma_cells, oversample, hillshade, ice_extent, show_trimline,
            area_threshold, contour_interval)


_ANIM_WORKER = {}

# Everything landscape() reads off self.model, split by transport: the big
# (time, y, x) arrays ship as .npy files loaded plainly per worker, the
# scalars/small arrays ride a tiny pickle. NEVER ship the full run state
# (ds_out): at production scale that is ~1-2 GB *per worker* once unpickled,
# and 8 workers of that is a machine-freezing swap storm (learned the hard
# way). The five arrays alone are ~5-10x smaller, and the parent additionally
# caps the worker count against a RAM budget. (mmap_mode='r' was tried and
# rejected: readonly slices are a distinct numba signature, so every worker
# recompiled the render kernels — +40% wall clock.)
_ANIM_ARRAYS = ('z_out', 'zb_out', 'H_out', 'area_out', 'receivers_out')
_ANIM_META = ('output_times', '_zELA_output', 'Lx', 'Ly', 'hc_over_H',
              'alpha_g', 'grid_nx', 'grid_ny')


def _animate_worker_init(bundle_dir, fig_width, landscape_kwargs):
    """Initializer for the parallel ``animate_landscape`` workers (spawned
    processes): single-thread numba (the pool parallelizes across frames, not
    within kernels), headless matplotlib, and a lightweight model shim built
    from the five bundle arrays + metadata — workers never import the 2D
    stack or load the run state."""
    import os
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    import matplotlib
    matplotlib.use('Agg')
    import pickle
    from types import SimpleNamespace
    from . import siim_plotter
    with open(os.path.join(bundle_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    arrays = {name: np.load(os.path.join(bundle_dir, name + '.npy'))
              for name in _ANIM_ARRAYS}
    _ANIM_WORKER['plot'] = siim_plotter(SimpleNamespace(**meta, **arrays))
    _ANIM_WORKER['fig_width'] = fig_width
    _ANIM_WORKER['kwargs'] = landscape_kwargs


def _animate_render_frame(job):
    """Render one animation frame to PNG (runs in a worker process)."""
    import matplotlib.pyplot as plt
    idx, png_path, dpi = job
    fig, _ = _ANIM_WORKER['plot'].landscape(
        i=idx, fig_width=_ANIM_WORKER['fig_width'], **_ANIM_WORKER['kwargs'])
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    return idx


class LandscapeMixin:
    """Deluxe cartographic renderer: ``landscape`` + ``animate_landscape``."""

    def landscape(self, field=None, i=-1, style='smooth', oversample=None,
                  channel_threshold=1e5, H_threshold=None, area_threshold=None,
                  z_min=0.0, z_max=None, cmap_bed='gist_earth',
                  ice_extent=None, ice_color='#e5e8ed',
                  ice_smoothing='mask', min_ice_cells=0, ice_time_avg=None,
                  ice_cmap=None, H_min=0.0, H_max=None,
                  lake_color='#a8c8e8', lake_min_depth=0.5, lake_min_area=0.0,
                  sigma_cells=None, ice_sigma_cells=None,
                  hillshade=None, azdeg=315, altdeg=45, ve=2.0,
                  contour_interval=None, contour_color='k',
                  contour_lw=0.2, contour_alpha=0.4,
                  show_trimline=None, trimline_color='#36506b', trimline_lw=0.6,
                  cross_section=None, cross_section_color='red', hyp_max=15.0,
                  show_smoothed_paths=False, smoothed_path_color='red',
                  smoothed_path_lw=0.4, fig_width=14, fig=None, ax=None,
                  ax_cs=None, ax_hyp=None, colorbar=True,
                  save=None):
        """Atlas-style render of the TRUE model state.

        ``field`` (``None`` -> smooth ``'bedrock'``, raw ``'bedrock+ice'``):
          - ``'bedrock'`` — the tracked bed ``z - HC_OVER_H*H``.
          - ``'bedrock+ice'`` — the presented surface with ice filled across the
            channel width (``ice_extent='footprint'``, the display dual of the
            width carve; ``'cells'`` shows the raw channel-cell ice). Pass
            ``ice_cmap`` (e.g. ``'Blues'``) to colour ice by column depth instead
            of flat ``ice_color``.
          - ``'bedrock+lakes'`` — bed with priority-flooded lakes.

        ``style`` — ONE knob for the whole rendering mode. It sets the unset
        (``None``) values of ``field`` / ``H_threshold`` / ``ice_sigma_cells`` /
        ``ice_time_avg`` / ``sigma_cells`` / ``oversample`` / ``hillshade`` /
        ``ice_extent`` / ``show_trimline`` / ``area_threshold`` /
        ``contour_interval``; any explicit value overrides the preset:

          - ``'smooth'`` (DEFAULT, stills and movies): the cartographic view —
            supersampled (``oversample=4``) + Gaussian-de-staircased terrain
            (``sigma_cells=oversample``), ``hillshade=True``, contoured
            (``contour_interval=100``), ice drawn across the sub-grid glacier
            width (``ice_extent='footprint'``) with thin ice thresholded off
            (``H_threshold=100``) and no area gate (``area_threshold=0``), its
            outline de-staircased (``ice_sigma_cells=2`` subgrid px). Hides
            nothing categorical (``min_ice_cells`` stays off — it can hide real
            small glacierets and is opt-in only).
          - ``'raw'``: the naked model output — one pixel per model cell
            (``oversample=1``), NO terrain or ice smoothing
            (``sigma_cells=0``, ``ice_sigma_cells=0``), NO hillshade, NO
            trimline (``show_trimline=False``), NO contours
            (``contour_interval=0``), and ice shown as the glaciated channel
            CELLS themselves (``ice_extent='cells'``) — every icy cell kept by
            thickness (``H_threshold=0``) but small-catchment specks dropped by
            a contributing-area gate (``area_threshold=1e6`` m²), rendered
            blocky (``imshow`` nearest-neighbour, so a 1-cell ice channel is not
            blurred away). Use it to inspect exactly what the model committed,
            unretouched.

        ``ice_time_avg`` resolves to ``1`` (off) for both — it averages OUTPUT
        frames, so with coarsely spaced saves it double-exposes two glacial
        epochs; use it only for densely-sampled animations, where
        ``ice_sigma_cells=3, ice_time_avg=2`` is the classic movie recipe of
        ``docs/dev/step_flicker.md``. NB ``ice_sigma_cells`` is in SUBGRID
        pixels, so its native-scale strength shifts with ``oversample`` and
        values >= ~1 native cell glob nearby ice and erase 1-cell threads.

        The parameters group as: what-to-draw (``field``, ``i``, ``style``,
        ``ice_extent``) · ice display (the thresholds/smoothing/cleanup knobs
        below) · terrain (``oversample``, ``sigma_cells``, ``hillshade``/
        ``azdeg``/``altdeg``/``ve``, ``contour_*``, ``z_min``/``z_max``,
        ``cmap_bed``) · overlays (``show_trimline``/``trimline_*``,
        ``cross_section``/``hyp_max``, ``show_smoothed_paths``/
        ``smoothed_path_*``/``channel_threshold``) · colours (``ice_color``/
        ``ice_cmap``/``H_min``/``H_max``, ``lake_*``) · figure plumbing
        (``fig_width``, ``fig``/``ax``/``ax_cs``/``ax_hyp``, ``colorbar``,
        ``save``).

        Contours sit at fixed elevation multiples of ``contour_interval`` (m),
        anchored at zero — so the spacing stays constant across animation frames
        as the elevation range changes. ``contour_interval`` of ``None`` resolves
        per ``style`` (smooth 100, raw off); ``0`` explicitly disables contours.
        ``sigma_cells=0`` disables smoothing; ``ice_sigma_cells=0`` keeps a crisp
        blocky ice outline.
        ``cross_section`` (a y in km) adds a profile + hypsometry panel below;
        the hypsometry x-axis is fixed at 0..``hyp_max`` % area (default 15 —
        a constant, frame-comparable scale; ``None`` auto-scales).
        When ``ax`` is supplied, pass ``ax_cs`` (and optionally ``ax_hyp``) to
        draw the section/hypsometry into your own axes for multi-panel layouts;
        ``colorbar=False`` suppresses the appended colorbar(s) so the caller can
        place shared ones.

        Ice display is cropped two independent ways (a cell shows ice only if it
        passes both): ``H_threshold`` (m, by column thickness; ``None``
        resolves per ``style`` — smooth 100, raw 0) and
        ``area_threshold`` (m², by upstream drainage area — gates which cells
        *seed* the footprint, so it drops small-catchment specks without
        touching cells that lie under a larger glacier's footprint; crops
        differently from ``H_threshold``). ``None`` resolves per ``style``
        (smooth 0 = off, raw 1e6); ``0`` disables. NB it is an ABSOLUTE
        drainage area, so it must exceed one cell area
        (``Lx*Ly/((nx-1)*(ny-1))``) to gate anything — a value below that is a
        silent no-op.
        ``show_trimline`` outlines the glacier margin (the ice-mask 1/2-level).
        Pass ``ice_cmap`` (e.g. ``'Blues'``) to colour ice by thickness, which
        adds its own colorbar.

        Three display-side anti-flicker knobs damp the per-step ice-mask jitter
        in animations (all opt-in — no preset engages them; see
        ``docs/dev/step_flicker.md``):

          - ``ice_smoothing`` (``'mask'`` default, or ``'field'``) — for
            ``ice_extent='cells'`` only (a ``ValueError`` otherwise, the
            footprint mask is not threshold-generated). ``'field'`` builds the
            ice mask as the level set of a smoothed CLIPPED thickness field
            (``gaussian_filter(upsample(min(H, 2*H_threshold)), ice_sigma_cells)
            > H_threshold``) rather than smoothing the binary ``H>H_threshold``
            indicator, so a cell hovering at the threshold nudges the outline by
            a sub-cell amount instead of popping a blob. The field-smoothed mask
            is a level set of a smoothed field, so it includes the thin
            (``H_threshold/2..H_threshold``) apron and draws a modestly dilated
            extent vs the raw ``H>H_threshold`` mask. Needs ``H_threshold > 0``.
          - ``min_ice_cells`` (int, default 0 = off; NATIVE cells) — cleans the
            final subgrid mask (both extents): removes ice components smaller
            than ``min_ice_cells`` native cells and fills enclosed bare holes
            smaller than the same (holes touching the array border stay open).
          - ``ice_time_avg`` (int >= 1; ``None`` resolves to ``1`` (off) for
            both styles) — replaces the ICE layer's thickness with the
            trailing mean of the last ``ice_time_avg`` output frames (clamped
            at the run start); the terrain, hypsometry and cross-section stay
            on frame ``i``. Display-only — the model state is untouched.

        Optionally add ``min_ice_cells=6`` to drop specks, but note it HIDES
        real small glacierets (why it is in no preset). On extent:
        ``ice_extent='footprint'`` (the default) is the width-honest view — ice
        drawn across the model's claimed width ``W = alpha_g*H``; ``'cells'`` is
        the raw state view (only the glaciated channel cells themselves).
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize, to_rgb
        from scipy.ndimage import map_coordinates, gaussian_filter

        m = self.model
        (field, H_threshold, ice_sigma_cells, ice_time_avg, sigma_cells,
         oversample, hillshade, ice_extent, show_trimline, area_threshold,
         contour_interval) = _resolve_style(
            style, field, H_threshold, ice_sigma_cells, ice_time_avg,
            sigma_cells, oversample, hillshade, ice_extent, show_trimline,
            area_threshold, contour_interval)
        if field not in ('bedrock', 'bedrock+ice', 'bedrock+lakes'):
            raise ValueError("field must be 'bedrock', 'bedrock+ice', or "
                             f"'bedrock+lakes', got {field!r}")
        show_ice = 'ice' in field
        show_lake = 'lake' in field
        if ice_smoothing not in ('mask', 'field'):
            raise ValueError(
                f"ice_smoothing must be 'mask' or 'field', got {ice_smoothing!r}")
        if ice_extent not in ('cells', 'footprint'):
            raise ValueError(
                f"ice_extent must be 'cells' or 'footprint', got {ice_extent!r}")
        if ice_smoothing == 'field' and ice_extent == 'footprint':
            raise ValueError(
                "ice_smoothing='field' applies to ice_extent='cells' only "
                "(the footprint mask is not threshold-generated).")
        if int(ice_time_avg) < 1:
            raise ValueError(f"ice_time_avg must be >= 1, got {ice_time_avg!r}")

        z_in = m.z_out[i]
        H_in = m.H_out[i]
        # ICE-layer thickness: optionally a trailing time-average (display-only
        # anti-flicker). Terrain / hypsometry / cross-section stay on frame i via
        # z_in / zb_in / area_in below.
        H_ice = H_in if int(ice_time_avg) <= 1 else _mean_recent_H(
            m.H_out, i, int(ice_time_avg))
        zb_in = m.zb_out[i]
        area_in = m.area_out[i]
        rec_in = m.receivers_out[i]

        ny, nx = z_in.shape
        ny_sub = (ny - 1) * oversample + 1
        nx_sub = (nx - 1) * oversample + 1
        y_idx = np.arange(ny_sub) * (ny - 1) / (ny_sub - 1)
        x_idx = np.arange(nx_sub) * (nx - 1) / (nx_sub - 1)
        Y, X = np.meshgrid(y_idx, x_idx, indexing='ij')
        # The TRUE (kernel-committed) bed, upsampled — bed view, hypsometry,
        # and cross-section bedrock all read this (zb_out, not z - hc*H).
        zb_sub = map_coordinates(zb_in, [Y, X], order=3, mode='nearest')

        if show_ice and ice_extent == 'footprint':
            # Footprint fill (display dual of the width carve): per-cell ice
            # surface from the power-diagram attribution, computed native + up.
            bs = list(getattr(m, 'boundary_status', ['fixed_value'] * 4))
            z_fill_nat, ice_nat, depth_nat = _footprint_ice_surface(
                H_ice, zb_in, rec_in, m.alpha_g,
                m.Ly / (ny - 1), m.Lx / (nx - 1),
                wrap_y=bs[2] == 'looped', wrap_x=bs[0] == 'looped',
                H_threshold=H_threshold, area_in=area_in,
                area_threshold=area_threshold, hc_over_H=m.hc_over_H)
            # order-1 (bilinear) for the ice-filled surface: it carries a
            # trimline step (flat z_s meeting terrain), and bicubic overshoots
            # there into a faint margin halo; the later Gaussian smooth recovers
            # interior smoothness regardless.
            z_sub = map_coordinates(z_fill_nat, [Y, X], order=1, mode='nearest')
            ice_mask = _smooth_ice_mask(ice_nat, Y, X, ice_sigma_cells)
            ice_mask = _clean_ice_mask(ice_mask, min_ice_cells, oversample)
            # depth for ice_cmap colouring: bilinear + mask-confined. order-0
            # staircased against the bicubic terrain; a plain order-1 upsample
            # bleeds toward 0 across the ice/rock edge, so renormalise by the
            # upsampled ice fraction to keep full column depth at the margin.
            ice_frac = map_coordinates(ice_nat.astype(float), [Y, X],
                                       order=1, mode='nearest')
            depth_up = map_coordinates(depth_nat, [Y, X], order=1, mode='nearest')
            depth_up = np.divide(depth_up, ice_frac, out=np.zeros_like(depth_up),
                                 where=ice_frac > 1e-3)
            H_col = np.where(ice_mask, depth_up, 0.0)
        else:
            z_sub = map_coordinates(z_in, [Y, X], order=3, mode='nearest')
            H_sub = map_coordinates(H_ice, [Y, X], order=0, mode='nearest')
            if ice_smoothing == 'field':
                # Smooth the CLIPPED thickness field, then threshold (anti-flicker
                # dual of the binary-mask smooth). Area gate: pre-zero H below the
                # area threshold so the level set still respects it.
                H_field = H_ice
                if area_threshold > 0:
                    H_field = np.where(area_in >= area_threshold, H_ice, 0.0)
                ice_mask = _field_ice_mask(H_field, Y, X, ice_sigma_cells,
                                           H_threshold)
            else:
                cells_ok = H_ice > H_threshold
                if area_threshold > 0:
                    cells_ok = cells_ok & (area_in >= area_threshold)
                ice_mask = _smooth_ice_mask(cells_ok, Y, X, ice_sigma_cells)
            ice_mask = _clean_ice_mask(ice_mask, min_ice_cells, oversample)
            H_col = np.where(ice_mask, m.hc_over_H * H_sub, 0.0)

        paths_xy = None
        if show_smoothed_paths:
            gf_paths = _compute_glacier_field(
                rec_in, H_in, area_in, None, m.grid_nx, m.grid_ny, m.Lx, m.Ly,
                m.alpha_g, channel_threshold, oversample, 0.0, 0.0, 'depth',
                hc_over_H=m.hc_over_H)
            paths_xy = gf_paths.paths_xy

        # Terrain surface: the ice-inflated surface (z_sub = zb + hc*H) ONLY
        # where ice is actually DRAWN; the bare bed (zb_sub) everywhere else. Ice
        # filtered out by H_threshold/area_threshold otherwise still bulges the
        # terrain by hc*H and — since that H grows/melts frame to frame — makes
        # the "bedrock" pulse in animations (and shifts the auto z_max/hillshade).
        z_composite = (np.where(ice_mask, z_sub, zb_sub) if show_ice else zb_sub)
        if sigma_cells > 0:
            z_smooth = gaussian_filter(z_composite, sigma=sigma_cells)
        else:
            z_smooth = z_composite
        if z_max is None:
            z_max = float(z_smooth.max())

        cmap = plt.get_cmap(cmap_bed)
        norm = Normalize(vmin=z_min, vmax=z_max)
        rgb = cmap(norm(z_smooth))[..., :3]

        ice_cmap_obj = None
        ice_norm = None
        if show_ice and ice_cmap is not None:
            if H_max is None:
                # Guard an ice-free frame (all-False mask): .max() on an empty
                # array raises. No ice is drawn, so any scale works — mirror the
                # H_max_eff <= H_min fixup below.
                H_max_eff = float(H_col[ice_mask].max()) if ice_mask.any() else H_min + 1.0
            else:
                H_max_eff = float(H_max)
            if H_max_eff <= H_min:
                H_max_eff = H_min + 1.0
            ice_cmap_obj = plt.get_cmap(ice_cmap)
            ice_norm = Normalize(vmin=H_min, vmax=H_max_eff)

        if show_ice and ice_cmap_obj is not None:
            rgb[ice_mask] = ice_cmap_obj(ice_norm(H_col[ice_mask]))[..., :3]
        elif show_ice:
            rgb[ice_mask] = np.asarray(to_rgb(ice_color))

        z_lake_surface = z_smooth
        lake_mask = None
        lake_filled = None
        if show_lake:
            lake_filled = _priority_flood(z_smooth, wrap_y=bs[2] == 'looped',
                                          wrap_x=bs[0] == 'looped')
            lake_mask = (lake_filled - z_smooth) > lake_min_depth
            if show_ice:
                lake_mask = lake_mask & ~ice_mask
            if lake_min_area > 0 and lake_mask.any():
                from scipy.ndimage import label
                labeled, _ = label(lake_mask)
                counts = np.bincount(labeled.ravel())
                pixel_area = (m.Lx / (nx_sub - 1)) * (m.Ly / (ny_sub - 1))
                keep = counts * pixel_area >= lake_min_area
                keep[0] = False
                lake_mask = keep[labeled]
            rgb[lake_mask] = np.asarray(to_rgb(lake_color))
            z_lake_surface = np.where(lake_mask, lake_filled, z_smooth)

        if hillshade:
            dx_sub = m.Lx / (nx_sub - 1)
            dy_sub = m.Ly / (ny_sub - 1)
            rgb = _shade_rgb_soft(rgb, z_lake_surface, dx_sub, dy_sub,
                                  ve, azdeg, altdeg)

        if ax is None:
            fig_height = fig_width * (m.Ly / m.Lx)
            if cross_section is not None:
                cs_height = fig_width * 0.25 * 2 / 3
                hyp_w_in = fig_width * 0.05
                if fig is None:
                    fig = plt.figure(figsize=(fig_width + hyp_w_in,
                                              fig_height + cs_height))
                gs = fig.add_gridspec(
                    2, 2, height_ratios=[fig_height, cs_height],
                    width_ratios=[fig_width, hyp_w_in], hspace=0.25, wspace=0.04)
                ax = fig.add_subplot(gs[0, 0])
                ax_cs = fig.add_subplot(gs[1, 0])
                ax_hyp = fig.add_subplot(gs[1, 1], sharey=ax_cs)
            else:
                if fig is None:
                    fig = plt.figure(figsize=(fig_width, fig_height))
                ax = fig.add_subplot(1, 1, 1)
        fig = ax.figure
        extent = [0, m.Lx / 1e3, 0, m.Ly / 1e3]
        # 'raw' renders blocky (nearest): bilinear blends a 1-cell ice channel
        # ~50/50 into its neighbours when oversample=1, washing out exactly the
        # thin ice raw is meant to show honestly. 'smooth' keeps bilinear (its
        # oversample=4 grid is dense enough that features survive, and the blur
        # de-staircases the atlas look).
        interp = 'nearest' if style == 'raw' else 'bilinear'
        ax.imshow(rgb, origin='lower', extent=extent, interpolation=interp)

        x_axis = np.linspace(0, m.Lx / 1e3, nx_sub)
        y_axis = np.linspace(0, m.Ly / 1e3, ny_sub)

        contour_obj = None
        if contour_interval:  # None or 0 disables
            lo = np.floor(np.nanmin(z_smooth) / contour_interval)
            hi = np.ceil(np.nanmax(z_smooth) / contour_interval)
            levels = np.arange(lo, hi + 1) * contour_interval
            contour_obj = ax.contour(
                x_axis, y_axis, z_smooth, levels=levels,
                colors=contour_color, linewidths=contour_lw,
                alpha=contour_alpha, zorder=2)

        # Crisp glacier margin: the 1/2-level of the (smoothed) ice mask — the
        # same outline the fill is thresholded at, drawn as a thin line so the
        # ice extent reads at a glance and a one-cell footprint flip is a small
        # wobble rather than a colour snap.
        if show_ice and show_trimline and ice_mask.any():
            ax.contour(x_axis, y_axis, ice_mask.astype(float), levels=[0.5],
                       colors=[trimline_color], linewidths=trimline_lw,
                       alpha=0.9, zorder=2.6)

        if show_smoothed_paths and paths_xy:
            for px_d, py_d in paths_xy:
                ax.plot(px_d / 1e3, py_d / 1e3, color=smoothed_path_color,
                        lw=smoothed_path_lw, zorder=3)

        ax.set_aspect('equal')
        if cross_section is None:
            ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.tick_params(axis='y', labelrotation=90)
        ax.set_title(f"{m.output_times[i] / 1e3:.0f} kyr", loc='right')

        if colorbar:
            sm_bed = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            if show_ice and ice_cmap_obj is not None:
                # Two stacked bars MUST share one divider, else separate
                # make_axes_locatable calls each place a bar immediately right of
                # the axes and they overlap. pad leaves room for the elevation
                # bar's tick labels + title between the two.
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax_bed = divider.append_axes('right', size='4%', pad=0.05)
                ax.figure.colorbar(sm_bed, cax=cax_bed, label='Surface elevation (m)')
                sm_ice = plt.cm.ScalarMappable(cmap=ice_cmap_obj, norm=ice_norm)
                cax_ice = divider.append_axes('right', size='4%', pad=0.7)
                ax.figure.colorbar(sm_ice, cax=cax_ice, label='Ice thickness (m)')
            else:
                _add_colorbar(sm_bed, ax, label='Surface elevation (m)')

        if cross_section is not None:
            y_km = float(cross_section)
            ax.axhline(y_km, color=cross_section_color, lw=1.0, alpha=0.8, zorder=3)
            if ax_cs is not None:
                dy_sub = m.Ly / (ny_sub - 1)
                j_cs = max(0, min(ny_sub - 1, int(round(y_km * 1e3 / dy_sub))))
                # The section reads the PRE-smoothing composite, never z_smooth:
                # the map's Gaussian (sigma_cells) bridges bed slots narrower
                # than its kernel, and painting the bridge-to-bed gap as ice
                # fabricates phantom glaciers in relict carved notches (H = 0).
                # Unsmoothed, the profile collapses onto the bed wherever no
                # ice is drawn, so the ice band below is exactly the drawn ice
                # column and nothing else.
                cs_profile = z_composite[j_cs, :]
                cs_x = np.linspace(0, m.Lx / 1e3, nx_sub)
                n_z = 200
                z_grid = np.linspace(z_min, z_max, n_z).reshape(-1, 1)
                row_colors = cmap(norm(z_grid.flatten()))

                if show_ice:
                    cs_bedrock = np.minimum(zb_sub[j_cs, :], cs_profile)
                    upper_for_bed = cs_bedrock
                else:
                    cs_bedrock = None
                    upper_for_bed = cs_profile

                bed_mask = (z_grid <= upper_for_bed.reshape(1, -1)).astype(float)
                bed_img = np.broadcast_to(row_colors[:, np.newaxis, :],
                                          (n_z, nx_sub, 4)).copy()
                bed_img[..., 3] = bed_mask
                ax_cs.imshow(bed_img, origin='lower',
                             extent=[0, m.Lx / 1e3, z_min, z_max],
                             aspect='auto', interpolation='nearest', zorder=1)

                if show_ice:
                    ice_mask_2d = ((z_grid >= cs_bedrock.reshape(1, -1)) &
                                   (z_grid <= cs_profile.reshape(1, -1))).astype(float)
                    ice_img = np.zeros((n_z, nx_sub, 4))
                    if ice_cmap_obj is not None:
                        # colour each ice column by its thickness on the SAME
                        # norm as the map, so the transect matches the plan view
                        cs_thick = np.maximum(cs_profile - cs_bedrock, 0.0)
                        col_rgb = ice_cmap_obj(ice_norm(cs_thick))[:, :3]
                        ice_img[..., :3] = col_rgb[np.newaxis, :, :]
                    else:
                        ice_img[..., :3] = np.asarray(to_rgb(ice_color))
                    ice_img[..., 3] = ice_mask_2d
                    ax_cs.imshow(ice_img, origin='lower',
                                 extent=[0, m.Lx / 1e3, z_min, z_max],
                                 aspect='auto', interpolation='nearest', zorder=2)

                if show_lake and lake_filled is not None:
                    cs_lake = lake_filled[j_cs, :]
                    cs_in_lake = lake_mask[j_cs, :]
                    lake_mask_2d_cs = ((z_grid >= cs_profile.reshape(1, -1)) &
                                       (z_grid <= cs_lake.reshape(1, -1)) &
                                       cs_in_lake.reshape(1, -1)).astype(float)
                    lake_img = np.zeros((n_z, nx_sub, 4))
                    lake_img[..., :3] = np.asarray(to_rgb(lake_color))
                    lake_img[..., 3] = lake_mask_2d_cs
                    ax_cs.imshow(lake_img, origin='lower',
                                 extent=[0, m.Lx / 1e3, z_min, z_max],
                                 aspect='auto', interpolation='nearest', zorder=2.5)

                if contour_obj is not None:
                    for lev in contour_obj.levels:
                        ax_cs.axhline(lev, color='lightgray', lw=0.3,
                                      alpha=0.6, zorder=3)

                if show_ice:
                    ax_cs.plot(cs_x, cs_bedrock, color='black', lw=1.0, zorder=4)
                    ax_cs.plot(cs_x, cs_profile, color='black', lw=0.5, zorder=4)
                else:
                    ax_cs.plot(cs_x, cs_profile, color='black', lw=1.0, zorder=4)

                ax_cs.axhline(0.0, color='black', lw=0.8, zorder=4)
                ax_cs.axhline(float(m._zELA_output[i]), color='black',
                              linestyle='--', lw=0.6, zorder=4)
                ax_cs.set_xlim(0, m.Lx / 1e3)
                ax_cs.set_ylim(z_min, z_max)
                ax_cs.set_xlabel('x (km)')
                ax_cs.set_ylabel('Elevation (m)')
                ax_cs.spines['top'].set_visible(False)
                ax_cs.spines['right'].set_visible(False)
                fig.canvas.draw()
                pos_main = ax.get_position()
                pos_cs = ax_cs.get_position()
                ax_cs.set_position([pos_main.x0, pos_cs.y0,
                                    pos_main.width, pos_cs.height])

                if ax_hyp is not None:
                    bedrock_surface = zb_sub
                    if contour_obj is not None and len(contour_obj.levels) >= 2:
                        lev = np.asarray(contour_obj.levels, dtype=float)
                        step = float(np.median(np.diff(lev)))
                        while lev[0] - step >= z_min:
                            lev = np.concatenate(([lev[0] - step], lev))
                        while lev[-1] + step <= z_max:
                            lev = np.concatenate((lev, [lev[-1] + step]))
                        edges = lev
                    else:
                        edges = np.linspace(z_min, z_max, 21)
                    if edges[0] > z_min:
                        edges = np.concatenate(([z_min], edges))
                    if edges[-1] < z_max:
                        edges = np.concatenate((edges, [z_max]))
                    hist, _ = np.histogram(bedrock_surface.ravel(), bins=edges)
                    pct = 100.0 * hist / bedrock_surface.size
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    heights = np.diff(edges)
                    ax_hyp.barh(centers, pct, height=heights, align='center',
                                color=cmap(norm(centers)), edgecolor='black',
                                linewidth=0.3)
                    ax_hyp.axhline(0.0, color='black', lw=0.8, zorder=4)
                    ax_hyp.axhline(float(m._zELA_output[i]), color='black',
                                   linestyle='--', lw=0.6, zorder=4)
                    ax_hyp.set_ylim(z_min, z_max)
                    ax_hyp.tick_params(axis='y', labelleft=False)
                    from matplotlib.ticker import FormatStrFormatter
                    ax_hyp.xaxis.set_major_formatter(FormatStrFormatter('%g'))
                    ax_hyp.set_xlabel('% area')
                    if hyp_max is None:
                        ax_hyp.set_xlim(left=0)
                    else:
                        ax_hyp.set_xlim(0, hyp_max)
                    ax_hyp.spines['top'].set_visible(False)
                    ax_hyp.spines['right'].set_visible(False)
                    pos_cs_now = ax_cs.get_position()
                    hyp_x0 = pos_cs_now.x1 + 0.015
                    ax_hyp.set_position([hyp_x0, pos_cs_now.y0,
                                         max(0.97 - hyp_x0, 0.05),
                                         pos_cs_now.height])

        if save is not None:
            fig.savefig(output_path(save, 'images'), dpi=200, bbox_inches='tight')
        return fig, ax

    def animate_landscape(self, path='landscape_animate', run_id=None, *,
                          interval=42, fig_width=14, n_workers=None,
                          **landscape_kwargs):
        """MP4 of ``landscape`` (its kwargs passed through). Returns the path.

        Frames are independent, so they render IN PARALLEL by default
        (audit N35): the arrays ``landscape`` reads are dumped once and
        loaded by ``n_workers`` spawned processes — ONLY those five arrays
        plus a small metadata pickle, never the full run state, and the
        worker count is additionally capped so the workers' combined array
        memory stays within ~25% of system RAM. Each worker renders its
        share of frames to PNG via the unchanged :meth:`landscape`; ffmpeg
        assembles the MP4. ``n_workers=None`` auto-sizes (up to 8, capped by
        cores, frame count and the RAM budget, and only when a timed probe
        frame projects the serial render past ~30 s);
        ``n_workers=1`` is the original in-process serial render (also the
        automatic fallback when ``fig``/``ax`` are passed or the model lacks
        the arrays the workers need). Frame content is identical either way —
        only the wall clock changes.

        Inherits ``landscape``'s ``style='smooth'`` default (the cartographic
        view — supersampled + hillshaded terrain, footprint-width ice, thin
        ice thresholded at ``H_threshold=100``). Pass ``style='raw'`` for the
        naked model output (one pixel per cell, no hillshade or smoothing, ice
        cells shown directly), or override individual knobs (explicit values
        always win over the preset). For
        DENSELY-SAMPLED output (frames close in model time), the classic
        anti-flicker movie recipe is ``ice_sigma_cells=3, ice_time_avg=2``
        (``docs/dev/step_flicker.md``) — deliberately NOT defaulted:
        ``ice_time_avg`` blends output frames (a double exposure of two
        glacial epochs when saves are far apart) and larger ``ice_sigma``
        globs and erases thin ice. ``min_ice_cells=6`` drops specks but
        HIDES real small glacierets — deliberately in no preset. Keep
        ``ice_extent='footprint'`` (the default) for the width-honest view (ice
        across the claimed width ``W = alpha_g*H``); ``'cells'`` shows the raw
        channel-cell state. See :meth:`landscape` for the full knob reference.
        """
        import os
        import matplotlib.pyplot as plt
        import matplotlib.animation as anm
        import tqdm
        m = self.model
        if 'save' in landscape_kwargs:
            raise ValueError(
                "animate_landscape renders every frame; per-frame 'save' "
                "makes no sense here — use 'path' for the movie file.")
        if run_id is not None:
            path = f"{run_id}_landscape"
        path = output_path(path, 'movies')
        nframes = len(m.output_times)

        parallel_ok = (
            'fig' not in landscape_kwargs and 'ax' not in landscape_kwargs
            and all(hasattr(m, a) for a in _ANIM_ARRAYS + _ANIM_META)
            and nframes >= 4)
        if n_workers is None and parallel_ok:
            # Adaptive: time one probe frame; parallel only when the projected
            # serial render dwarfs the ~5 s/worker spawn+load startup.
            import time
            t0 = time.perf_counter()
            fig_probe, _ = self.landscape(i=0, fig_width=fig_width,
                                          **landscape_kwargs)
            plt.close(fig_probe)
            t_frame = time.perf_counter() - t0
            n_workers = (min(8, os.cpu_count() or 1, nframes)
                         if t_frame * nframes > 30.0 else 1)
        elif n_workers is None:
            n_workers = 1
        n_workers = max(1, int(n_workers))
        if n_workers > 1 and not parallel_ok:
            n_workers = 1
        if n_workers > 1:
            # RAM-budget cap: each worker plain-loads the five bundle arrays,
            # so keep n_workers * bundle within ~25% of system RAM
            # (swap-storm guard; the reason we never ship the full ds_out).
            bundle_bytes = sum(np.asarray(getattr(m, name)).nbytes
                               for name in _ANIM_ARRAYS)
            try:
                total_ram = (os.sysconf('SC_PAGE_SIZE')
                             * os.sysconf('SC_PHYS_PAGES'))
                n_workers = max(1, min(n_workers,
                                       int(0.25 * total_ram
                                           / max(bundle_bytes, 1))))
            except (ValueError, OSError):
                pass                          # no sysconf (exotic platform)

        if n_workers > 1:
            return self._animate_parallel(path, interval, fig_width,
                                          n_workers, landscape_kwargs)

        fig, _ = self.landscape(i=0, fig_width=fig_width, **landscape_kwargs)
        pbar = tqdm.tqdm(total=nframes + 1, desc='Rendering frames')
        pbar.update(1)

        def update(idx):
            fig.clear()
            self.landscape(i=idx, fig=fig, fig_width=fig_width, **landscape_kwargs)
            pbar.update(1)
            return fig.axes

        anim = anm.FuncAnimation(fig, update, frames=range(nframes),
                                 interval=interval, blit=False, repeat=False)
        anim.save(f"{path}.mp4", writer='ffmpeg', dpi=150)
        pbar.close()
        plt.close(fig)
        return f"{path}.mp4"

    def _animate_parallel(self, path, interval, fig_width, n_workers,
                          landscape_kwargs, dpi=150):
        """Parallel frame renderer for :meth:`animate_landscape`: dump ONLY
        the arrays/metadata ``landscape`` reads, render frames to PNG across
        spawned workers via a lightweight plotter shim, assemble with ffmpeg
        (the same binary matplotlib's writer uses)."""
        import os
        import pickle
        import shutil
        import subprocess
        import tempfile
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import get_context
        import matplotlib.animation as anm
        import tqdm

        m = self.model
        nframes = len(m.output_times)
        tmpdir = tempfile.mkdtemp(prefix='siim_animate_')
        try:
            for name in _ANIM_ARRAYS:
                np.save(os.path.join(tmpdir, name + '.npy'),
                        np.asarray(getattr(m, name)))
            meta = {name: getattr(m, name) for name in _ANIM_META}
            meta['boundary_status'] = list(
                getattr(m, 'boundary_status', ['fixed_value'] * 4))
            with open(os.path.join(tmpdir, 'meta.pkl'), 'wb') as f:
                pickle.dump(meta, f)
            jobs = [(idx, os.path.join(tmpdir, f'frame_{idx:05d}.png'), dpi)
                    for idx in range(nframes)]
            with ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=get_context('spawn'),
                    initializer=_animate_worker_init,
                    initargs=(tmpdir, fig_width, landscape_kwargs)) as pool:
                futures = [pool.submit(_animate_render_frame, j) for j in jobs]
                for fut in tqdm.tqdm(as_completed(futures), total=nframes,
                                     desc=f'Rendering frames ({n_workers} workers)'):
                    fut.result()          # surface worker exceptions
            fps = 1000.0 / interval
            cmd = [anm.FFMpegWriter.bin_path(), '-y',
                   '-framerate', f'{fps:g}',
                   '-i', os.path.join(tmpdir, 'frame_%05d.png'),
                   # libx264 needs even dimensions; crop at most 1 px
                   '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2',
                   '-pix_fmt', 'yuv420p', f'{path}.mp4']
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg assembly failed:\n{proc.stderr[-2000:]}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return f"{path}.mp4"
