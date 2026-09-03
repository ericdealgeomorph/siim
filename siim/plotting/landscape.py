"""``landscape`` / ``animate_landscape`` — the deluxe atlas-style cartographic
render: supersampled + Gaussian-smoothed terrain, sub-grid across-channel-width
ice (footprint fill — the display dual of the width carve), priority-flooded
lakes, hillshade, contours, and an optional cross-section + hypsometry panel.
Frame + animate only (no viewer, by design). See
``docs/guides/outputs_and_io.md`` for the public plotting contract.

``field`` ∈ {``'bedrock'``, ``'bedrock+ice'``, ``'bedrock+lakes'``}.

The numba render backend (footprint ice, smoothing, hillshade, path tracing)
lives in ``_render.py``.
"""

import numpy as np

from ._render import (
    _add_colorbar, output_path, _ice_ramp,
    _footprint_ice_surface, _smooth_ice_mask, _field_ice_mask,
    _clean_ice_mask, _mean_recent_H, _compute_glacier_field,
    _channel_closure, _trunk_ribbons,
    _priority_flood, _shade_rgb_soft,
)

# The ELA line in the cross-section / hypsometry panels: a forcing datum, not
# another contour, so it gets its own colour.
ELA_COLOR = '#c0392b'


def _veil_alpha(t):
    """Veil opacity at normalised column depth ``t``: a LINEAR ramp topping out
    at ``t = 0.35``.

    The old ``0.18 + 0.82*sqrt(t/0.40)`` front-loading put alpha near 0.5 on a
    30 m column, so a highland of sub-resolution apron ice read as one
    half-opaque sheet with the trunks barely darker streaks through it (measured
    on the reference run at glacial max: trunk-vs-apron RGB distance 0.13,
    against 0.67 for apron-vs-bare). Linear keeps thin ice subordinate and
    spends the ramp where the depth gradient carries the information."""
    return np.clip(0.12 + 0.88 * np.clip(t / 0.35, 0.0, 1.0), 0.0, 1.0)


def _resolve_style(style, field, H_threshold, ice_sigma_cells, ice_time_avg,
                   sigma_cells, oversample, hillshade, ice_extent,
                   show_margin, area_threshold, contour_interval,
                   ice_shading=None, trunk_display=None):
    """Resolve every unset (``None``) style-differentiated knob per ``style``;
    an explicit value always wins over the preset.

    ``'smooth'`` (the default) is the cartographic view: bed AND ice
    (field 'bedrock+ice' — the ice is the model's headline state, so both
    presets draw it), supersampled (oversample 4) + Gaussian-de-staircased
    terrain (sigma_cells = oversample), hillshaded, contoured
    (contour_interval 100 m), ice drawn across the sub-grid glacier width
    (ice_extent 'footprint') with NOTHING hidden by thickness (H_threshold 0)
    and no area gate (area_threshold 0), its outline de-staircased
    (ice_sigma_cells 2 subgrid px) and margin drawn (show_margin True), and
    shaded as a depth-graded translucent veil (ice_shading 'veil') with the
    resolved trunk glaciers drawn over it as true-width ribbons
    (trunk_display 'ribbons').

    ``'raw'`` is the naked model output: bed AND ice (field 'bedrock+ice'),
    one pixel per model cell (oversample 1), NO terrain or ice smoothing
    (sigma_cells 0, ice_sigma_cells 0), NO hillshade, NO margin outline
    (show_margin False), NO contours (contour_interval 0), FLAT ice
    (ice_shading 'flat' — one colour, no depth grading to read as structure),
    NO trunk ribbons (trunk_display 'none'),
    ice shown as the glaciated channel CELLS themselves (ice_extent 'cells')
    with every icy cell kept by thickness (H_threshold 0, as in smooth) but
    small-catchment
    specks dropped by a contributing-area gate (area_threshold 1e6 m², Eric
    2026-07-07). Use it to inspect exactly what the model committed,
    unretouched.

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
        # Both presets show ice: resolving smooth to bare 'bedrock' made every
        # ice knob a silent no-op on the default render (Eric, 2026-09-03).
        field = 'bedrock+ice'
    if ice_shading is None:
        ice_shading = 'flat' if raw else 'veil'
    if H_threshold is None:
        # BOTH styles keep every icy cell (Eric, 2026-09-03): the veil already
        # fades thin ice out, so a thickness gate only DELETES real glacierets.
        H_threshold = 0.0
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
    if show_margin is None:
        show_margin = False if raw else True
    if area_threshold is None:
        # raw drops small-catchment specks (Eric, 2026-07-07); smooth keeps all.
        area_threshold = 1e6 if raw else 0.0
    if contour_interval is None:
        contour_interval = 0.0 if raw else 100.0
    if trunk_display is None:
        # raw draws the cells the model committed and nothing else; the
        # cartographic view fills the resolved trunks as ribbons.
        trunk_display = 'none' if raw else 'ribbons'
    if ice_shading not in ('flat', 'veil'):
        raise ValueError(
            f"ice_shading must be 'flat' or 'veil', got {ice_shading!r}")
    if trunk_display not in ('none', 'ribbons'):
        raise ValueError(
            f"trunk_display must be 'none' or 'ribbons', got {trunk_display!r}")
    return (field, H_threshold, ice_sigma_cells, ice_time_avg,
            sigma_cells, oversample, hillshade, ice_extent, show_margin,
            area_threshold, contour_interval, ice_shading, trunk_display)


def _run_global_H_max(m):
    """The RUN-GLOBAL drawn-column depth ``hc_over_H * max H_out`` — the
    default top of the ice scale.

    The veil and the ice colorbar both read it, so a still and every frame of a
    movie put the same colour on the same depth (a per-frame max would repaint
    each frame's thickest trunk the same dark blue). ``_frozen_scale_kwargs``
    freezes ``H_max`` to exactly this value, so the two paths agree."""
    return float(m.hc_over_H * np.nanmax(np.asarray(m.H_out)))


def _frozen_scale_kwargs(m, landscape_kwargs):
    """Freeze ``landscape``'s AUTO colour scales across a whole animation.

    A still autoscales per frame, which in a movie puts every frame on its own
    scale: the bed colorbar top tracks each frame's highest peak and the
    ``ice_cmap`` norm collapses to ``H_min + 1`` on ice-free frames, so colours
    pulse for reasons that are not in the model (``map`` freezes its clim
    globally for the same reason). Resolve both ONCE over all output frames —
    ``max z_out`` bounds every frame's composited surface and
    ``hc_over_H * max H_out`` every drawn column — unless the caller pinned
    them. Returns a new kwargs dict; both the serial and the parallel frame
    paths then render on the one scale."""
    kwargs = dict(landscape_kwargs)
    if kwargs.get('z_max') is None:
        kwargs['z_max'] = float(np.nanmax(np.asarray(m.z_out)))
    if kwargs.get('H_max') is None:
        kwargs['H_max'] = _run_global_H_max(m)
    return kwargs


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
                  ice_extent=None, ice_color='#e5e8ed', ice_shading=None,
                  trunk_display=None, trunk_width_cells=1.0, trunk_alpha=0.85,
                  ice_smoothing='mask', min_ice_cells=0, ice_time_avg=None,
                  ice_cmap=None, H_min=0.0, H_max=None,
                  lake_color='#a8c8e8', lake_min_depth=0.5, lake_min_area=0.0,
                  sigma_cells=None, ice_sigma_cells=None,
                  hillshade=None, azdeg=315, altdeg=45, ve=2.0,
                  contour_interval=None, contour_color='k',
                  contour_lw=0.2, contour_alpha=0.4,
                  show_margin=None, margin_color='#1b4f72', margin_lw=1.0,
                  show_trimline=None, trimline_color=None, trimline_lw=None,
                  cross_section=None, cross_section_color='red', hyp_max=15.0,
                  show_smoothed_paths=False, smoothed_path_color='red',
                  smoothed_path_lw=0.4, fig_width=14, fig=None, ax=None,
                  ax_cs=None, ax_hyp=None, colorbar=True,
                  save=None):
        """Atlas-style render of the TRUE model state.

        ``field`` (``None`` -> ``'bedrock+ice'`` under BOTH styles):
          - ``'bedrock'`` — the tracked bed ``z - HC_OVER_H*H``.
          - ``'bedrock+ice'`` (the default) — the presented surface with ice
            filled across the channel width (``ice_extent='footprint'``, the
            display dual of the width carve; ``'cells'`` shows the raw
            channel-cell ice), shaded per ``ice_shading``. Pass ``ice_cmap``
            (e.g. ``'Blues'``) to colour ice by column depth with your own
            colormap instead.
          - ``'bedrock+lakes'`` — bed with priority-flooded lakes.

        ``ice_shading`` (``None`` -> smooth ``'veil'``, raw ``'flat'``) — how
        ice with no explicit ``ice_cmap`` is painted:

          - ``'veil'`` — a depth-graded TRANSLUCENT glacier ramp alpha-blended
            over the bed, so thin apron ice lets the terrain read through and
            the trunk core saturates. Depth is normalised on ``H_max`` (default
            the RUN-GLOBAL ``hc_over_H * max H_out``, so frames of a movie are
            directly comparable) and gets its own colorbar.
          - ``'flat'`` — one opaque ``ice_color`` everywhere ice is drawn.

        ``ice_cmap`` overrides both (opaque, your colormap, per-frame ``H_max``
        fallback).

        ``trunk_display`` (``None`` -> smooth ``'ribbons'``, raw ``'none'``) —
        how the TRUNK GLACIERS are drawn. A trunk cell is any icy cell
        DOWNSTREAM of a cell whose claimed width ``W = alpha_g*H`` already
        spans ``trunk_width_cells`` grid cells (default 1.0 — the width the
        grid can just resolve), the class carried along the receivers to the
        terminus so a thinning tongue stays a trunk down to its toe.

          - ``'ribbons'`` — the class is traced and drawn at its true width
            (parabolic column depth on the same ramp, its own flat ice surface
            in the hillshade and the section) at ``trunk_alpha`` opacity, and
            the veil is built from the ice that is LEFT, so the trunks read as
            coherent tongues instead of streaks in a uniform apron.
            ``show_margin`` then outlines the ribbons only.
          - ``'none'`` — the veil alone over every icy cell (the pre-ribbon
            render).

        ``trunk_alpha`` is a FLOOR on the ribbon opacity, so deep ice keeps the
        ramp's own (higher) value; ``ice_shading='flat'`` paints ribbons the
        single ``ice_color``, like the rest of the ice.

        ``style`` — ONE knob for the whole rendering mode. It sets the unset
        (``None``) values of ``field`` / ``H_threshold`` / ``ice_sigma_cells`` /
        ``ice_time_avg`` / ``sigma_cells`` / ``oversample`` / ``hillshade`` /
        ``ice_extent`` / ``show_margin`` / ``area_threshold`` /
        ``contour_interval`` / ``ice_shading``; any explicit value overrides the
        preset:

          - ``'smooth'`` (DEFAULT, stills and movies): the cartographic view —
            bed AND ice (``field='bedrock+ice'``), supersampled
            (``oversample=4``) + Gaussian-de-staircased terrain
            (``sigma_cells=oversample``), ``hillshade=True``, contoured
            (``contour_interval=100``), ice drawn across the sub-grid glacier
            width (``ice_extent='footprint'``) as a depth-graded translucent
            veil (``ice_shading='veil'``) with the resolved trunks filled as
            true-width ribbons over it (``trunk_display='ribbons'``), nothing
            hidden by thickness
            (``H_threshold=0``) and no area gate (``area_threshold=0``), its
            outline de-staircased (``ice_sigma_cells=2`` subgrid px) and its
            margin outlined (``show_margin=True``). Hides
            nothing categorical (``min_ice_cells`` stays off — it can hide real
            small glacierets and is opt-in only).
          - ``'raw'``: the naked model output — one pixel per model cell
            (``oversample=1``), NO terrain or ice smoothing
            (``sigma_cells=0``, ``ice_sigma_cells=0``), NO hillshade, NO
            margin outline (``show_margin=False``), NO contours
            (``contour_interval=0``), FLAT single-colour ice
            (``ice_shading='flat'``), NO trunk ribbons
            (``trunk_display='none'``), and ice shown as the glaciated channel
            CELLS themselves (``ice_extent='cells'``) — every icy cell kept by
            thickness (``H_threshold=0``) but small-catchment specks dropped by
            a contributing-area gate (``area_threshold=1e6`` m²), rendered
            blocky (``imshow`` nearest-neighbour, so a 1-cell ice channel is not
            blurred away). Use it to inspect exactly what the model committed,
            unretouched.

        ``ice_time_avg`` resolves to ``1`` (off) for both — it averages OUTPUT
        frames, so with coarsely spaced saves it double-exposes two glacial
        epochs; use it only for densely-sampled animations, where
        ``ice_sigma_cells=3, ice_time_avg=2`` is the recommended anti-flicker
        movie recipe. NB ``ice_sigma_cells`` is in SUBGRID
        pixels, so its native-scale strength shifts with ``oversample`` and
        values >= ~1 native cell glob nearby ice and erase 1-cell threads.

        The parameters group as: what-to-draw (``field``, ``i``, ``style``,
        ``ice_extent``, ``trunk_display``) · ice display (the thresholds/
        smoothing/cleanup knobs
        below) · terrain (``oversample``, ``sigma_cells``, ``hillshade``/
        ``azdeg``/``altdeg``/``ve``, ``contour_*``, ``z_min``/``z_max``,
        ``cmap_bed``) · overlays (``show_margin``/``margin_*``,
        ``cross_section``/``hyp_max``, ``show_smoothed_paths``/
        ``smoothed_path_*``/``channel_threshold``) · colours (``ice_shading``/
        ``ice_color``/``ice_cmap``/``H_min``/``H_max``,
        ``trunk_alpha``, ``lake_*``) · figure plumbing
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
        resolves to ``0`` under BOTH styles — nothing is hidden by thickness,
        since the depth-graded veil already fades thin ice out and a gate only
        deletes real glacierets; raise it explicitly to crop them) and
        ``area_threshold`` (m², by upstream drainage area — gates which cells
        *seed* the footprint, so it drops small-catchment specks without
        touching cells that lie under a larger glacier's footprint; crops
        differently from ``H_threshold``). ``None`` resolves per ``style``
        (smooth 0 = off, raw 1e6); ``0`` disables. NB it is an ABSOLUTE
        drainage area, so it must exceed one cell area
        (``Lx*Ly/((nx-1)*(ny-1))``) to gate anything — a value below that is a
        silent no-op.
        ``show_margin`` outlines the CURRENT ice margin (the ice-mask
        1/2-level — or the trunk ribbons alone under
        ``trunk_display='ribbons'``) in ``margin_color``/``margin_lw``. (The old
        ``show_trimline``/``trimline_color``/``trimline_lw`` names are
        deprecated aliases — they still work, with a ``DeprecationWarning``;
        the outline is the live margin, not a trimline.) Pass ``ice_cmap``
        (e.g. ``'Blues'``) to colour ice by thickness with your own colormap;
        it, like the default veil, adds its own colorbar.

        Three display-side anti-flicker knobs damp the per-step ice-mask jitter
        in animations (all opt-in — no preset engages them; see
        ``docs/guides/outputs_and_io.md``):

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
            smaller than the same (holes touching a non-looped array border
            stay open). Seam-aware: on a looped axis a glacier straddling the
            seam is ONE component, not two sub-minimum halves.
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
        # Deprecated 'trimline' spelling: the outline is the CURRENT margin.
        for old, new, val in (('show_trimline', 'show_margin', show_trimline),
                              ('trimline_color', 'margin_color', trimline_color),
                              ('trimline_lw', 'margin_lw', trimline_lw)):
            if val is not None:
                import warnings
                warnings.warn(f"{old} is deprecated, use {new} (the outline is "
                              "the current ice margin, not a trimline)",
                              DeprecationWarning, stacklevel=2)
        if show_trimline is not None and show_margin is None:
            show_margin = show_trimline
        if trimline_color is not None:
            margin_color = trimline_color
        if trimline_lw is not None:
            margin_lw = trimline_lw
        (field, H_threshold, ice_sigma_cells, ice_time_avg, sigma_cells,
         oversample, hillshade, ice_extent, show_margin, area_threshold,
         contour_interval, ice_shading, trunk_display) = _resolve_style(
            style, field, H_threshold, ice_sigma_cells, ice_time_avg,
            sigma_cells, oversample, hillshade, ice_extent, show_margin,
            area_threshold, contour_interval, ice_shading, trunk_display)
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
        if ice_smoothing == 'field' and H_threshold <= 0:
            raise ValueError(
                "ice_smoothing='field' needs H_threshold > 0: the clipped "
                f"field min(H, 2*H_threshold) is all zero at {H_threshold!r}, "
                "so NO ice would be drawn (both styles resolve H_threshold "
                "to 0 — pass an explicit H_threshold with it).")
        if int(ice_time_avg) < 1:
            raise ValueError(f"ice_time_avg must be >= 1, got {ice_time_avg!r}")

        z_in = m.z_out[i]
        H_in = m.H_out[i]
        # ICE-layer thickness: optionally a trailing time-average (display-only
        # anti-flicker). It feeds ONLY the ice mask and the depth colouring —
        # terrain / hypsometry / cross-section stay on frame i via z_in / H_in /
        # zb_in / area_in below (in BOTH extents; see the footprint branch).
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

        # Looped axes: the footprint fill, the mask cleanup and the lake flood
        # all need the same seam awareness, so read the flags once.
        bs = list(getattr(m, 'boundary_status', ['fixed_value'] * 4))
        wrap_x = bs[0] == 'looped'
        wrap_y = bs[2] == 'looped'

        # The TRUNK CLASS (trunk_display='ribbons'): the resolved glaciers —
        # every icy cell downstream of one whose claimed width W = alpha_g*H
        # already spans trunk_width_cells grid cells. They are drawn below as
        # traced true-width ribbons, so the cell veil is built from the thin
        # ice that is LEFT (no cell is drawn twice). A frame whose ice never
        # reaches the resolution has an EMPTY class and paints the plain veil.
        ribbons = show_ice and trunk_display == 'ribbons'
        trunk_cells = None
        if ribbons:
            cell = max(m.Lx / (nx - 1), m.Ly / (ny - 1))

            def _trunk_class(H_src):
                return _channel_closure(
                    H_src, rec_in, m.alpha_g * np.asarray(H_src)
                    >= trunk_width_cells * cell)

            def _ribbons(H_src, cells):
                return _trunk_ribbons(
                    rec_in, H_src, area_in,
                    np.asarray(zb_in) + m.hc_over_H * np.asarray(H_src),
                    cells, nx, ny, m.Lx, m.Ly, m.alpha_g, oversample,
                    wrap_y=wrap_y, wrap_x=wrap_x, hc_over_H=m.hc_over_H)

            trunk_cells = _trunk_class(H_ice)
        H_veil = np.where(trunk_cells, 0.0, H_ice) if ribbons else H_ice

        if show_ice and ice_extent == 'footprint':
            # Footprint fill (display dual of the width carve): per-cell ice
            # surface from the power-diagram attribution, computed native + up.
            z_fill_nat, ice_nat, depth_nat = _footprint_ice_surface(
                H_veil, zb_in, rec_in, m.alpha_g,
                m.Ly / (ny - 1), m.Lx / (nx - 1),
                wrap_y=wrap_y, wrap_x=wrap_x,
                H_threshold=H_threshold, area_in=area_in,
                area_threshold=area_threshold, hc_over_H=m.hc_over_H)
            if int(ice_time_avg) > 1:
                # ...but the fill also carries the TERRAIN here, and
                # ice_time_avg is an ice-layer knob: rebuild the composited
                # surface from frame i's H so the trailing mean never leaks
                # into terrain / hillshade / contours / hypsometry / section
                # (the 'cells' extent already reads z_in). Skipped entirely at
                # ice_time_avg = 1, where the single fill above is frame i.
                z_fill_nat = _footprint_ice_surface(
                    np.where(trunk_cells, 0.0, H_in) if ribbons else H_in,
                    zb_in, rec_in, m.alpha_g,
                    m.Ly / (ny - 1), m.Lx / (nx - 1),
                    wrap_y=wrap_y, wrap_x=wrap_x,
                    H_threshold=H_threshold, area_in=area_in,
                    area_threshold=area_threshold, hc_over_H=m.hc_over_H)[0]
            # order-1 (bilinear) for the ice-filled surface: it carries a
            # trimline step (flat z_s meeting terrain), and bicubic overshoots
            # there into a faint margin halo; the later Gaussian smooth recovers
            # interior smoothness regardless.
            z_sub = map_coordinates(z_fill_nat, [Y, X], order=1, mode='nearest')
            ice_mask = _smooth_ice_mask(ice_nat, Y, X, ice_sigma_cells)
            ice_mask = _clean_ice_mask(ice_mask, min_ice_cells, oversample,
                                       wrap_y=wrap_y, wrap_x=wrap_x)
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
            H_sub = map_coordinates(H_veil, [Y, X], order=0, mode='nearest')
            if ice_smoothing == 'field':
                # Smooth the CLIPPED thickness field, then threshold (anti-flicker
                # dual of the binary-mask smooth). Area gate: pre-zero H below the
                # area threshold so the level set still respects it.
                H_field = H_veil
                if area_threshold > 0:
                    H_field = np.where(area_in >= area_threshold, H_veil, 0.0)
                ice_mask = _field_ice_mask(H_field, Y, X, ice_sigma_cells,
                                           H_threshold)
            else:
                cells_ok = H_veil > H_threshold
                if area_threshold > 0:
                    cells_ok = cells_ok & (area_in >= area_threshold)
                ice_mask = _smooth_ice_mask(cells_ok, Y, X, ice_sigma_cells)
            ice_mask = _clean_ice_mask(ice_mask, min_ice_cells, oversample,
                                       wrap_y=wrap_y, wrap_x=wrap_x)
            H_col = np.where(ice_mask, m.hc_over_H * H_sub, 0.0)

        trunk_mask = None
        if ribbons:
            # True-width ribbons over the veil: the sub-cell apron stays a
            # veil, the resolved trunks become coherent tongues with their own
            # (flat) ice surface. The ribbon carries BOTH the drawn column
            # depth and that surface, so the hillshade shades the glacier and
            # the cross-section reads the same composite the map does.
            trunk_mask = np.zeros_like(ice_mask)
            if trunk_cells.any():
                trunk_depth, trunk_zs = _ribbons(H_ice, trunk_cells)
                trunk_mask = trunk_depth > 0.0
                H_col = np.where(trunk_mask, trunk_depth, H_col)
                if int(ice_time_avg) > 1:
                    # ...but the ribbon SURFACE is terrain (it goes into the
                    # composite), and ice_time_avg is an ice-layer knob: take
                    # it from frame i's own ice, exactly as the footprint fill
                    # does above. Trunk pixels with no frame-i ribbon fall back
                    # to the bare bed below.
                    cells_i = _trunk_class(H_in)
                    trunk_zs = (_ribbons(H_in, cells_i)[1] if cells_i.any()
                                else np.zeros_like(trunk_zs))
                # never below the local bed (a ribbon over rising ground)
                z_sub = np.where(trunk_mask, np.maximum(trunk_zs, zb_sub),
                                 z_sub)
                ice_mask = ice_mask | trunk_mask

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
            # nanmax, not max: one NaN otherwise blanks the whole map silently.
            # The cross-section draws the UNSMOOTHED composite against this same
            # limit, so cover it too — else the map's Gaussian shaves the peaks
            # and the section profile runs off the top of its panel.
            z_max = float(max(np.nanmax(z_smooth), np.nanmax(z_composite)))

        cmap = plt.get_cmap(cmap_bed)
        norm = Normalize(vmin=z_min, vmax=z_max)
        rgb = cmap(norm(z_smooth))[..., :3]

        ice_cmap_obj = None
        ice_norm = None
        veil = show_ice and ice_cmap is None and ice_shading == 'veil'
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
        elif veil:
            # RUN-GLOBAL depth scale, not this frame's max: the veil is meant
            # to be frame-comparable in a movie (animate freezes H_max to the
            # very same value, so still and movie agree).
            H_max_eff = _run_global_H_max(m) if H_max is None else float(H_max)
            if H_max_eff <= H_min:
                H_max_eff = H_min + 1.0
            ice_cmap_obj = _ice_ramp()
            ice_norm = Normalize(vmin=H_min, vmax=H_max_eff)

        if veil:
            # Depth-graded TRANSLUCENT ice: alpha-blend the glacier ramp onto
            # the bed instead of stamping one opaque colour over it, so thin
            # apron ice lets the terrain read through while the trunk core
            # saturates (the ramp is _veil_alpha).
            t_veil = np.asarray(np.clip(ice_norm(H_col), 0.0, 1.0))
            a_veil = np.where(ice_mask, _veil_alpha(t_veil), 0.0)
            if trunk_mask is not None:
                # The ribbons must read as GLACIERS, not as more veil: paint
                # them at trunk_alpha, a FLOOR (deep ice is already at least
                # this opaque on the ramp) — the bed stops showing through the
                # trunk, so trunk and apron separate at a glance.
                a_veil = np.where(trunk_mask, np.maximum(a_veil, trunk_alpha),
                                  a_veil)
            a_veil = a_veil[..., np.newaxis]
            rgb = (1.0 - a_veil) * rgb + a_veil * ice_cmap_obj(t_veil)[..., :3]
        elif show_ice and ice_cmap_obj is not None:
            rgb[ice_mask] = ice_cmap_obj(ice_norm(H_col[ice_mask]))[..., :3]
        elif show_ice:
            rgb[ice_mask] = np.asarray(to_rgb(ice_color))

        z_lake_surface = z_smooth
        lake_mask = None
        lake_filled = None
        if show_lake:
            # Flood the TRUE (pre-smoothing) composite — the very surface the
            # cross-section paints down to. Flooding z_smooth instead bridges
            # bed slots narrower than the Gaussian and hands the section a
            # water table it cannot meet (the phantom-ice split, 2026-07-16).
            lake_filled = _priority_flood(z_composite, wrap_y=wrap_y,
                                          wrap_x=wrap_x)
            lake_mask = (lake_filled - z_composite) > lake_min_depth
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

        dx_sub = m.Lx / (nx_sub - 1)
        dy_sub = m.Ly / (ny_sub - 1)
        if hillshade:
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
        # imshow places pixel CENTRES inside `extent`, but the array is
        # NODE-valued: run the raster half a subgrid pixel past each edge so
        # node j lands on x = Lx*j/(nx_sub-1) — the coordinate the contours,
        # trimline and section line use. (Axis limits stay at [0, Lx]/[0, Ly].)
        extent = [-0.5 * dx_sub / 1e3, (m.Lx + 0.5 * dx_sub) / 1e3,
                  -0.5 * dy_sub / 1e3, (m.Ly + 0.5 * dy_sub) / 1e3]
        # 'raw' renders blocky (nearest): bilinear blends a 1-cell ice channel
        # ~50/50 into its neighbours when oversample=1, washing out exactly the
        # thin ice raw is meant to show honestly. 'smooth' keeps bilinear (its
        # oversample=4 grid is dense enough that features survive, and the blur
        # de-staircases the atlas look).
        interp = 'nearest' if style == 'raw' else 'bilinear'
        ax.imshow(rgb, origin='lower', extent=extent, interpolation=interp)
        ax.set_xlim(0, m.Lx / 1e3)
        ax.set_ylim(0, m.Ly / 1e3)

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
        # wobble rather than a colour snap. Under ribbons it outlines the
        # RIBBONS only: outlining the veil too would ring every sub-cell
        # glacieret, and the veil's whole point is to stay subordinate.
        margin_of = ice_mask if trunk_mask is None else trunk_mask
        if show_ice and show_margin and margin_of.any():
            ax.contour(x_axis, y_axis, margin_of.astype(float), levels=[0.5],
                       colors=[margin_color], linewidths=margin_lw,
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
            # Without ice the composite IS the bare bed — name it as map() does.
            bed_label = ('Surface elevation (m)' if show_ice
                         else 'Bedrock elevation (m)')
            if show_ice and ice_cmap_obj is not None:
                # Two stacked bars MUST share one divider, else separate
                # make_axes_locatable calls each place a bar immediately right of
                # the axes and they overlap. pad leaves room for the elevation
                # bar's tick labels + title between the two.
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax_bed = divider.append_axes('right', size='4%', pad=0.05)
                ax.figure.colorbar(sm_bed, cax=cax_bed, label=bed_label)
                sm_ice = plt.cm.ScalarMappable(cmap=ice_cmap_obj, norm=ice_norm)
                # pad is in INCHES and must clear the elevation bar's tick
                # labels + rotated title — a TYPOGRAPHIC width, not a fraction
                # of the page. Measured on this layout: 0.7 in clears down to
                # fig_width=5, while a purely relative 0.05*fig_width collides
                # everywhere below ~14 in. So 0.7 is the FLOOR and the pad only
                # grows (relatively) on a figure wider than that.
                cax_ice = divider.append_axes('right', size='4%',
                                              pad=max(0.7, 0.05 * fig_width))
                # What the map paints is the local COLUMN depth to the flat ice
                # surface (hc_over_H * H at the thalweg, more on carved flanks)
                # — not the width-mean H that map(field='ice') shows.
                ax.figure.colorbar(sm_ice, cax=cax_ice,
                                   label='Ice column depth (m)')
            else:
                _add_colorbar(sm_bed, ax, label=bed_label)

        if cross_section is not None:
            y_km = float(cross_section)
            ax.axhline(y_km, color=cross_section_color, lw=1.0, alpha=0.8, zorder=3)
            if ax_cs is not None:
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
                    # Bilinear bed for the SECTION only: the map's BICUBIC
                    # zb_sub overshoots above the (bilinear) filled ice
                    # surface, which zeroed the ice band on columns the map
                    # paints as ice. Match orders instead — and keep the bed
                    # ON the profile wherever no ice is drawn, so an ice-free
                    # column still paints a zero-thickness band (no phantom
                    # ice from the order-3/order-1 gap).
                    cs_zb = map_coordinates(
                        zb_in, [np.full(nx_sub, y_idx[j_cs]), x_idx],
                        order=1, mode='nearest')
                    cs_bedrock = np.where(ice_mask[j_cs, :],
                                          np.minimum(cs_zb, cs_profile),
                                          cs_profile)
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
                        # colour each ice column by its depth on the SAME norm
                        # as the map (the veil ramp, or an explicit ice_cmap),
                        # so the transect matches the plan view
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
                # The ELA is the one forcing line in the panel — colour and
                # name it so it is not read as another contour.
                zela = float(m._zELA_output[i])
                ax_cs.axhline(zela, color=ELA_COLOR, linestyle='--', lw=0.9,
                              zorder=4)
                ax_cs.annotate('ELA', xy=(0.995, zela),
                               xycoords=('axes fraction', 'data'),
                               ha='right', va='bottom', color=ELA_COLOR,
                               fontsize='small', zorder=4,
                               bbox=dict(fc='white', ec='none', alpha=0.7,
                                         pad=0.5))
                ax_cs.set_xlim(0, m.Lx / 1e3)
                ax_cs.set_ylim(z_min, z_max)
                ax_cs.set_xlabel('x (km)')
                ax_cs.set_ylabel('Elevation (m)')
                ax_cs.spines['top'].set_visible(False)
                ax_cs.spines['right'].set_visible(False)
                fig.canvas.draw()
                # Vertical exaggeration of the section, from the axes' own
                # display box vs its data range — the panel is much wider than
                # tall, so the relief it shows is not to scale.
                bb = ax_cs.get_window_extent()
                ve_cs = ((bb.height / max(z_max - z_min, 1e-9))
                         / (bb.width / m.Lx))
                ax_cs.annotate(f'VE {ve_cs:.0f}$\\times$', xy=(0.005, 0.94),
                               xycoords='axes fraction', ha='left', va='top',
                               fontsize='small', color='#666666', zorder=4)
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
                    ax_hyp.axhline(float(m._zELA_output[i]), color=ELA_COLOR,
                                   linestyle='--', lw=0.9, zorder=4)
                    ax_hyp.set_ylim(z_min, z_max)
                    ax_hyp.tick_params(axis='y', labelleft=False)
                    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
                    ax_hyp.xaxis.set_major_formatter(FormatStrFormatter('%g'))
                    # At narrow fig_width the default tick set put a 0 hard
                    # against the section's last x tick — three ticks, no zero.
                    ax_hyp.xaxis.set_major_locator(MaxNLocator(3, prune='lower'))
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

        The auto colour scales are FROZEN over the run: ``z_max`` and the ice
        norm's ``H_max`` are resolved once across all output frames instead of
        per frame, so a frame's colours mean the same thing throughout the
        movie; pass explicit values to override. (``H_max`` freezes to the
        run-global ``hc_over_H * max H_out``, which is already the still
        render's default for the depth-graded veil.)

        Inherits ``landscape``'s ``style='smooth'`` default (the cartographic
        view — supersampled + hillshaded terrain, footprint-width ice drawn as
        a depth-graded translucent veil, with nothing hidden by thickness —
        ``H_threshold=0``). Pass ``style='raw'`` for the
        naked model output (one pixel per cell, no hillshade or smoothing, flat
        ice cells shown directly), or override individual knobs (explicit values
        always win over the preset). For
        DENSELY-SAMPLED output (frames close in model time), the classic
        anti-flicker movie recipe is ``ice_sigma_cells=3, ice_time_avg=2`` —
        deliberately NOT defaulted:
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
        landscape_kwargs = _frozen_scale_kwargs(m, landscape_kwargs)
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
