"""``landscape`` / ``animate_landscape`` — the deluxe atlas-style cartographic
render: supersampled + Gaussian-smoothed terrain, sub-grid across-channel-width
ice (footprint fill — the display dual of the width carve), priority-flooded
lakes, hillshade, contours, and an optional cross-section + hypsometry panel.
Frame + animate only (no viewer, by design). See
``docs/guides/outputs_and_io.md`` for the public plotting contract.

``field`` ∈ {``'bedrock'``, ``'bedrock+ice'``, ``'bedrock+lakes'``,
``'bedrock+speed'``}.

The numba render backend (footprint ice, smoothing, hillshade, path tracing)
lives in ``_render.py``.
"""

import contextlib
import functools
import sys
from typing import NamedTuple

import numpy as np

from ..constants import check_range
from .._output import output_path
from .._core.outputs import closure_pair, warn_if_closure_pair_missing
from .._core.solvers import ice_speeds
from ._colormaps import bed_cmap
from ._render import (
    _add_colorbar, _ice_ramp, _node_extent,
    _footprint_ice_surface, _smooth_ice_mask, _field_ice_mask,
    _clean_ice_mask, _smoothed_glacier_paths,
    _channel_closure, _trunk_ribbons, _label_looped,
    _priority_flood, _shade_rgb_soft, _shade_stretch,
)
from ._style import COLORS, time_label
from ._animation import MOVIE_DPI, frame_indices, movie_path, save_animation


def _veil_alpha(depth, opaque_depth, sharpness=1.0):
    """Veil opacity of a column ``depth`` metres deep, in absolute depth: the
    S-curve ``t**k / (t**k + (1 - t)**k)`` of ``t = depth/opaque_depth``
    (clipped to [0, 1]), with ``k = sharpness``. It is 0 at no ice, 50 % at
    half ``opaque_depth`` and fully opaque from ``opaque_depth`` down, with no
    floor; ``k = 1`` is the plain line, and a larger ``k`` fades thin ice
    further and saturates thick ice sooner (flat at both ends for k > 1).

    Absolute, not scaled by the colour ramp's ``H_max``: on a sheet run H_max is
    kilometres, so a depth fraction left 300 m trunks translucent; and with no
    floor a few metres of ice no longer tints the terrain (Eric, 2026-11-05:
    the ice "below 20 m made more transparent", "100 % opacity at 100 m",
    then "sharper to drop the opacity at 20 m": k = 2)."""
    t = np.clip(np.asarray(depth, dtype=float) / opaque_depth, 0.0, 1.0)
    tk = t ** sharpness
    return tk / (tk + (1.0 - t) ** sharpness)


# The fields that draw ice (and so shade the ice surface, not the bed).
_ICE_FIELDS = ('bedrock+ice', 'bedrock+speed')

_BED_EXTENDS = ('neither', 'min', 'max', 'both')

# contourpy's 'serial' algorithm traces the same contour levels 1.6-1.7x faster
# than matplotlib's default 'mpl2014'; only the anti-aliasing of a few contour
# pixels moves (about a dozen pixels a frame at most, by at most 5 colour levels).
# Eric, 2026-11-06, for stills and movies alike (ED-168).
_CONTOUR_ALGORITHM = 'serial'

# animate_landscape's quality presets, (oversample cap, ffmpeg args) (Eric,
# 2026-11-06): 'high' (the default) is a movie to keep, the full still-render look;
# 'draft' is a research movie that will soon be replaced: at most oversample 2 (so
# ice_sigma_cells 1) and x264's 'veryfast' preset, about 1.6-1.7x faster than 'high'
# (ED-168). An explicit oversample wins; None args keep matplotlib's defaults.
_MOVIE_QUALITY = {'high': (None, None), 'draft': (2, ['-preset', 'veryfast'])}


def _place_axes(fig):
    """Settle every axes' drawn position (aspect and axes locators, as the
    first step of ``Axes.draw``) without rendering the figure.

    ``landscape`` reads the map's aspect-shrunk box to align the section under
    it; a full ``canvas.draw()`` did the same at the price of a second render
    of every frame (ED-168: about 10 % of a movie frame; pixel-identical)."""
    get_renderer = getattr(fig.canvas, 'get_renderer', None)
    if get_renderer is None or fig.get_layout_engine() is not None:
        fig.canvas.draw()   # a non-Agg canvas, or a layout engine to run: as before
        return
    renderer = get_renderer()
    for axes in fig.axes:
        locator = axes.get_axes_locator()
        axes.apply_aspect(locator(axes, renderer) if locator else None)


def _hypsometry_counts(values, edges):
    """``np.histogram(values, bins=edges)[0]`` without its sort of every value
    (25-61 ms a movie frame, ED-168): each value finds its bin by a search among
    the few edges. The same bins: closed on the left, the last also on the
    right, values outside the edges (and NaN) dropped; edges that do not
    increase are refused, as np.histogram refuses them."""
    if np.any(edges[:-1] > edges[1:]):
        raise ValueError('hypsometry bin edges must increase')
    n = edges.size - 1
    idx = np.searchsorted(edges, values, side='right') - 1
    idx[values == edges[-1]] = n - 1
    return np.bincount(idx[(idx >= 0) & (idx < n)], minlength=n)


def _resolved_style_of(kwargs):
    """The style ``landscape`` resolves for these kwargs (its defaults
    filled in)."""
    kw = {**_landscape_defaults(), **kwargs}
    return _resolve_style(kw['style'], **{f: kw[f] for f in _ResolvedStyle._fields})


def _font_points(key):
    """The active rcParams font size ``key`` (e.g. 'legend.fontsize'), in pt."""
    from matplotlib import rcParams
    from matplotlib.font_manager import FontProperties
    return FontProperties(size=rcParams[key]).get_size_in_points()


def _text_scale():
    """How much larger the active tick and axis-label type is than the 9 pt /
    10 pt that ``landscape``'s inch margins were tuned for (never below 1)."""
    return max(1.0, _font_points('xtick.labelsize') / 9.0,
               _font_points('ytick.labelsize') / 9.0,
               _font_points('axes.labelsize') / 10.0)


def _ice_column(H, hc):
    """The model's ice column depth ``hc * H`` on iced cells (H > 0), flat."""
    H = np.asarray(H, dtype=float)
    return (np.asarray(hc, dtype=float) * H)[H > 0]


def _ice_extend(lo, hi, vmin, vmax):
    """The ice bar's arrows: where the model's column depth range [lo, hi]
    passes the colour scale's [vmin, vmax] (no low end when vmin is 0). One
    rule for a still and for a movie's frozen bar."""
    return _bed_extend(vmin > 0 and lo < vmin, hi > vmax)


def _bed_extend(below, above):
    """The bed colorbar's ``extend``: an arrow at each end where the drawn
    surface saturates, below ``z_min`` and/or above ``z_max``."""
    return _BED_EXTENDS[bool(below) + 2 * bool(above)]


def _refuse_ice_time_avg(method):
    """Name the removed ``ice_time_avg`` knob instead of letting the signature
    raise a bare unexpected-keyword error (0.10.0)."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if 'ice_time_avg' in kwargs:
            raise ValueError(
                "ice_time_avg was removed in 0.10.0: it averaged trailing "
                "OUTPUT frames, which double-exposes two glacial epochs when "
                "saves are far apart, and had been off by default since "
                "2026-07-07. Drop the argument; every render shows frame i "
                "alone.")
        return method(self, *args, **kwargs)
    return wrapper


class _ResolvedStyle(NamedTuple):
    """Named resolved defaults; public landscape calls retain their shape."""
    field: str
    H_threshold: float
    ice_sigma_cells: float
    sigma_cells: float
    oversample: int
    hillshade: bool
    ice_extent: str
    show_margin: bool
    area_threshold: float
    contour_interval: float
    ice_shading: str
    trunk_display: str


def _resolve_style(style, field, H_threshold, ice_sigma_cells,
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
    (ice_sigma_cells = oversample / 2 subgrid px, half a model cell: 2 px at
    the default oversample, which erases a glacier that occupies a single
    cell) and margin drawn (show_margin True), and
    shaded as a depth-graded translucent veil (ice_shading 'veil'), with no
    trunk ribbons (trunk_display 'none').

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

    A larger ``ice_sigma_cells`` globs nearby ice into blobs and erases
    1-cell threads (the mask is a 1/2-level set of a blurred indicator, i.e. a
    closing). It was tried in the smooth preset and REJECTED on use; it stays
    opt-in for densely-sampled animations only. Pinned by test_ice_display.
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
    if oversample is None:
        oversample = 1 if raw else 4
    if ice_sigma_cells is None:
        # Half the oversample (Eric, 2026-11-06): the same half-cell outline
        # smoothing in model cells at any oversample (2 subgrid px at 4).
        ice_sigma_cells = 0.0 if raw else oversample / 2
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
        # Ribbons are OFF in both presets (Eric, 2026-11-05). The footprint
        # already draws the carve's parabolic cross-section wherever a glacier
        # spans more than about two cells, and on sheet runs the ribbons drew
        # rings, because the trunk class reads sheet ice as channel. The layer
        # stays available as an explicit 'ribbons'; once mode D's channel/sheet
        # mapping is fixed (ED-138) it is meant to seed from the model's
        # channel cells (ED-137).
        trunk_display = 'none'
    if ice_shading not in ('flat', 'veil'):
        raise ValueError(
            f"ice_shading must be 'flat' or 'veil', got {ice_shading!r}")
    if trunk_display not in ('none', 'ribbons'):
        raise ValueError(
            f"trunk_display must be 'none' or 'ribbons', got {trunk_display!r}")
    return _ResolvedStyle(field, H_threshold, ice_sigma_cells, sigma_cells,
            oversample, hillshade, ice_extent, show_margin,
            area_threshold, contour_interval, ice_shading, trunk_display)


def _run_global_H_max(m, frames=None):
    """The RUN-GLOBAL drawn-column depth ``hc_over_H * max H_out`` (or
    ``max hc_cell_out * H_out`` on a mode-D run) — the default
    top of the ice scale (over ``frames`` alone when a movie encodes only some
    of them).

    The veil's colour and the ice colorbar both read it (its opacity follows
    ``ice_opaque_depth`` instead), so a still and every frame of a
    movie put the same colour on the same depth (a per-frame max would repaint
    each frame's thickest trunk the same dark blue). ``_frozen_scale_kwargs``
    freezes ``H_max`` to exactly this value, so the two paths agree."""
    H_out = np.asarray(m.H_out)
    hc = getattr(m, 'hc_cell_out', None)
    if frames is not None:
        H_out = H_out[frames]
        hc = None if hc is None else np.asarray(hc)[frames]
    if hc is not None:
        return float(np.nanmax(np.asarray(hc) * H_out))
    return float(m.hc_over_H * np.nanmax(H_out))


def _landscape_defaults():
    """``landscape``'s keyword defaults, read from its signature (their one
    home), for the scales a movie must resolve as the frames will."""
    import inspect
    return {name: p.default for name, p in
            inspect.signature(LandscapeMixin.landscape).parameters.items()
            if p.default is not inspect.Parameter.empty}


def _frozen_scale_kwargs(m, landscape_kwargs, frames=None):
    """Freeze ``landscape``'s AUTO colour scales across a whole animation.

    A still autoscales per frame, which in a movie puts every frame on its own
    scale: the bed colorbar top tracks each frame's highest peak and the
    ``ice_cmap`` norm collapses to ``H_min + 1`` on ice-free frames, so colours
    pulse for reasons that are not in the model (``map`` freezes its clim
    globally for the same reason). Resolve both ONCE over the frames the movie
    will ENCODE (``frames``, all of them by default) — ``max z_out`` bounds
    every frame's composited surface and ``hc_over_H * max H_out`` every drawn
    column — unless the caller pinned them. Scaling a windowed movie to the
    whole run would wash it out. Returns a new kwargs dict; both the serial and
    the parallel frame paths then render on the one scale.

    The bed bar's ``extend_bed`` is decided once too (its arrows would
    otherwise come and go as frames cross ``z_min``/``z_max``), from the native
    bed's lowest and the native surface's highest point over the frames, and
    the ice bar's ``extend_ice`` by the still's rule (``_ice_extend``) on the
    frames' column range; and
    the hillshade stretch (a steep feature appearing elsewhere would otherwise
    re-light the whole map), estimated on the native surface smoothed as the
    render smooths it."""
    kwargs = dict(landscape_kwargs)
    # frame by frame: stacking the frames would copy whole output arrays
    idx = range(len(m.z_out)) if frames is None else frames
    kw = {**_landscape_defaults(), **kwargs}
    style = _resolved_style_of(kwargs)
    surface = m.z_out if style.field in _ICE_FIELDS else m.zb_out
    want_ice = kwargs.get('extend_ice') is None and style.field == 'bedrock+ice'
    hc = getattr(m, 'hc_cell_out', None)
    # one pass over the frames for every reduction; np.nanmin/nanmax over the
    # frames' extremes skips an all-NaN frame (Python's max() is NaN-order
    # dependent)
    z_hi, zb_lo, s_hi, col_lo, col_hi = [], [], [], [], []
    for k in idx:
        z_hi.append(np.nanmax(m.z_out[k]))
        zb_lo.append(np.nanmin(m.zb_out[k]))
        s_hi.append(np.nanmax(surface[k]))
        if want_ice:
            col = _ice_column(m.H_out[k], m.hc_over_H if hc is None else hc[k])
            if col.size:
                col_lo.append(np.nanmin(col))
                col_hi.append(np.nanmax(col))
    if kwargs.get('z_max') is None:
        kwargs['z_max'] = float(np.nanmax(z_hi))
    if kwargs.get('H_max') is None:
        kwargs['H_max'] = _run_global_H_max(m, frames)
    if kwargs.get('extend_bed') is None:
        # the drawn surface lies between the bare bed and the ice surface
        kwargs['extend_bed'] = _bed_extend(np.nanmin(zb_lo) < kw['z_min'],
                                           np.nanmax(s_hi) > kwargs['z_max'])
    if want_ice:
        # the still's rule (_ice_extend) on the union of the frames' columns,
        # against the norm a frame builds (its top bumped as there)
        vmin = float(kw['H_min'])
        vmax = kwargs['H_max'] if kwargs['H_max'] > vmin else vmin + 1.0
        kwargs['extend_ice'] = (_ice_extend(min(col_lo), max(col_hi), vmin, vmax)
                                if col_hi else 'neither')
    if style.hillshade and kwargs.get('shade_stretch') is None:
        from scipy.ndimage import gaussian_filter
        sigma = style.sigma_cells / style.oversample
        ny, nx = np.shape(surface[0])
        # lazy: the native surface, not the rendered composite (footprint,
        # ribbons, lakes, upsample): it missed ~5 % of the span at the dark
        # end, clipping <= 0.05 % of pixels, on the getting-started run;
        # upgrade: stretch each frame's composite via a shared ice-geometry
        # stage (plan-2026-11-05-plotting-revisit, phase C).
        ranges = [_shade_stretch(gaussian_filter(z, sigma) if sigma > 0 else z,
                                 m.Lx / (nx - 1), m.Ly / (ny - 1),
                                 kw['ve'], kw['azdeg'], kw['altdeg'])
                  for z in (np.asarray(surface[k], dtype=float) for k in idx)]
        kwargs['shade_stretch'] = (min(r[0] for r in ranges),
                                   max(r[1] for r in ranges))
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
# The metadata with one entry per saved frame: dumped for the selected frames
# alone, as the arrays are.
_ANIM_FRAME_META = ('output_times', '_zELA_output')
# Shipped when present (mode D's per-cell datum); the worker
# shim reads None otherwise.
_ANIM_OPTIONAL = ('hc_cell_out',)
# ...plus what field='bedrock+speed' alone reads (shipped only for it): the
# closure pair (_core.outputs.closure_pair), whose flux is closure_Qg_out where
# the run stores it and Qg_out otherwise, never both.
_SPEED_META = ('sliding_law', 'Ac', 'lambda_p', 'lambda_c', 'tau_c')
_SPEED_OPTIONAL = ('Qg_out', 'closure_Qg_out', 'closure_H_out')


def _anim_fields(landscape_kwargs):
    """The (arrays, metadata, optional arrays) names the workers need for
    these kwargs; an optional array ships only when the model has it."""
    if landscape_kwargs.get('field') == 'bedrock+speed':
        return (_ANIM_ARRAYS, _ANIM_META + _SPEED_META,
                _ANIM_OPTIONAL + _SPEED_OPTIONAL)
    return _ANIM_ARRAYS, _ANIM_META, _ANIM_OPTIONAL


def _anim_shipped(m, names):
    """The bundle arrays among ``names`` that ``m`` has; ``Qg_out`` stays
    home when ``closure_Qg_out`` replaces it."""
    replaced = getattr(m, 'closure_Qg_out', None) is not None
    return [name for name in names if getattr(m, name, None) is not None
            and not (replaced and name == 'Qg_out')]


@contextlib.contextmanager
def _launching_script_hidden():
    """Keep spawned workers from re-running the launching script.

    A spawned process re-imports ``__main__`` from its path, so a script with
    no ``if __name__ == '__main__':`` guard runs again in every worker, the
    model included, and then breaks the pool when it reaches its own
    ``animate_landscape``. Its path and spec are hidden for the life of the
    pool (workers start as frames are submitted), as they are absent in a
    notebook. The workers then see nothing the script set up: the matplotlib
    settings reach them through ``_animate_worker_init``; a colormap the
    script registered by name does not. ``_animate_parallel`` leaves the
    script visible when a keyword or rcParams object needs a class it defines. Not safe
    against a second spawning pool in another thread."""
    main = vars(sys.modules['__main__'])
    hidden = {key: main[key] for key in ('__file__', '__spec__') if key in main}
    main.pop('__file__', None)
    main['__spec__'] = None         # multiprocessing reads the attribute itself
    try:
        yield
    finally:
        main.pop('__spec__', None)
        main.update(hidden)


def _animate_worker_init(bundle_dir, fig_width, landscape_kwargs, rc=None):
    """Initializer for the parallel ``animate_landscape`` workers (spawned
    processes): single-thread numba (the pool parallelizes across frames, not
    within kernels), headless matplotlib with the parent's settings ``rc``
    (a style set in the script or notebook, as the serial render has it), and
    a lightweight model shim built
    from the five bundle arrays + metadata — workers never import the 2D
    stack or load the run state. The bundle holds the movie's frames alone,
    each once, so a job names its frame by row in the bundle."""
    import os
    import numba
    numba.set_num_threads(1)
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update(rc or {})
    # every frame the full figure, whatever a matplotlibrc says: the stream
    # needs one frame size
    matplotlib.rcParams['savefig.bbox'] = 'standard'
    import pickle
    from types import SimpleNamespace
    with open(os.path.join(bundle_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    names, _meta, optional = _anim_fields(landscape_kwargs)
    arrays = {name: np.load(os.path.join(bundle_dir, name + '.npy'))
              for name in names}
    for name in optional:
        path = os.path.join(bundle_dir, name + '.npy')
        arrays[name] = np.load(path) if os.path.exists(path) else None
    plot = LandscapeMixin()
    plot.model = SimpleNamespace(**meta, **arrays)
    _ANIM_WORKER['plot'] = plot
    _ANIM_WORKER['fig_width'] = fig_width
    _ANIM_WORKER['kwargs'] = landscape_kwargs
    _ANIM_WORKER['cancelled'] = os.path.join(bundle_dir, 'cancelled')


def _animate_render_frame(job):
    """Render one animation frame to a raw RGBA file (runs in a worker
    process); returns its size in pixels, or None for a frame skipped because
    the movie has already failed.

    Raw, not PNG: the parent streams the frames into ffmpeg, and compressing
    each one to a PNG that ffmpeg then decodes was 40 % of a movie (ED-168)."""
    import os
    import matplotlib.pyplot as plt
    idx, row, raw_path, dpi = job
    if os.path.exists(_ANIM_WORKER['cancelled']):
        return None
    fig, _ = _ANIM_WORKER['plot'].landscape(
        i=row, fig_width=_ANIM_WORKER['fig_width'], **_ANIM_WORKER['kwargs'])
    fig.savefig(raw_path, dpi=dpi, format='rgba')
    # The canvas still holds the buffer savefig just rendered.
    height, width = np.asarray(fig.canvas.buffer_rgba()).shape[:2]
    plt.close(fig)
    if os.path.getsize(raw_path) != 4 * width * height:
        raise RuntimeError(
            f'frame {idx}: the raw file does not match its {width} x {height} '
            'px canvas')
    return width, height


class LandscapeMixin:
    """Deluxe cartographic renderer: ``landscape`` + ``animate_landscape``."""

    @_refuse_ice_time_avg
    def landscape(self, field=None, i=-1, style='smooth', oversample=None,
                  channel_threshold=1e5, H_threshold=None, area_threshold=None,
                  z_min=-500.0, z_max=None, cmap_bed=None, extend_bed=None, extend_ice=None,
                  ice_extent=None, ice_color='#e5e8ed', ice_shading=None,
                  trunk_display=None, trunk_width_cells=1.0, trunk_alpha=0.85,
                  ice_opaque_depth=100.0, ice_alpha_sharpness=2.0,
                  ice_smoothing='mask', min_ice_cells=0,
                  ice_cmap=None, H_min=0.0, H_max=None,
                  lake_color=COLORS['water'], lake_min_depth=0.5, lake_min_area=0.0,
                  sigma_cells=None, ice_sigma_cells=None,
                  hillshade=None, azdeg=315, altdeg=45, ve=4.0,
                  shade_stretch=None,
                  contour_interval=None, contour_color='k',
                  contour_lw=0.2, contour_alpha=0.4,
                  show_margin=None, margin_color='#1b4f72', margin_lw=0.3,
                  show_trimline=None, trimline_color=None, trimline_lw=None,
                  cross_section=None, cross_section_color='#716477', hyp_max=15.0,
                  section_ve=4.0,
                  show_smoothed_paths=False, smoothed_path_color='red',
                  smoothed_path_lw=0.4, fig_width=12, fig=None, ax=None,
                  ax_cs=None, ax_hyp=None, colorbar=True,
                  save=None, speed_cmap='magma_r', speed_min=20.0,
                  speed_max=1000.0):
        """Cartographic view of the stored bed, ice or lake display layers.

        Returns ``(fig, ax)``. ``i`` selects the output snapshot (default last).
        Display processing never changes stored model arrays.

        Presets and layers
        ------------------
        ``style='smooth'`` draws bedrock + ice with oversample=4,
        sigma_cells=oversample, hillshade, 100 m contours, footprint ice,
        depth-graded translucent shading and margins.
        ``style='raw'`` uses oversample=1, unsmoothed channel cells, flat ice,
        no contours/margins/hillshade and area_threshold=1e6 m².
        Both use H_threshold=0 and draw no trunk ribbons. An explicit argument always overrides the
        preset; None selects the preset's value.

        ``field`` selects 'bedrock', 'bedrock+ice' (default),
        'bedrock+lakes' or 'bedrock+speed'. Bedrock is the stored true bed
        ``zb_out``. Ice is a display reconstruction; lakes are priority-flooded
        on the true bed and drawn only on ground the same preset draws free
        of ice (no ice is drawn). These are separate recipes, not names of
        stored quantities.

        'bedrock+speed' draws the same ice geometry as 'bedrock+ice' but
        colours it by ice SURFACE speed (m/yr), recomputed from the
        closure's own stored flux and thickness (see
        ``siim.siim2d.siim.ice_speed``; no speed is stored).
        ``speed_cmap`` (default 'magma_r') on a log scale from ``speed_min``
        to ``speed_max`` (default 20-1000 m/yr); ice slower than
        ``speed_min`` is left transparent, faster saturates. Under the
        footprint extent each pixel takes the speed of the source whose ice
        surface it shows; outlet (self-receiving) cells, whose H is copied
        from their donor rather than solved, show none. An explicit
        ``trunk_display='ribbons'`` carries the speed along each ribbon.
        Caveats: every icy cell is a channel
        of width W=alpha_g*H, so thick interior ice that covers its cell as
        a sheet (W > dx) reads slow — the same flux through the cell width
        would move about W/dx times faster; under 'coulomb' a few thin-ice
        cells near the yield stress tau_c can show very large speeds
        (saturated here).
        Under 'eff-exp' there is no sliding split and the speed is the
        cross-section mean.

        Ice extent and filtering
        ------------------------
        ``ice_extent='footprint'`` fills the claimed width W=alpha_g*H;
        'cells' shows the ice-bearing channel cells. ``trunk_display='ribbons'``
        traces ice downstream of cells spanning ``trunk_width_cells`` native
        cells; 'none' (the default) disables this extra layer. Ribbons carry
        their source ice surface into hillshade and the cross-section. Under
        'bedrock+speed' they cost a second path trace (about 7x a frame) and
        ring thick slow trunks in margin outlines; under 'bedrock+lakes' they
        clip proglacial lakes (a ribbon runs a cell past the terminus).

        ``H_threshold`` is a gate on WIDTH-MEAN thickness H (m), and
        ``area_threshold`` gates upstream area (m²). Both govern the source
        cells for footprints and ribbons; they do not clip the width grown
        from a surviving source. Zero disables each gate. The raw area's
        default can be disabled to inspect every ice-bearing cell.

        ``ice_sigma_cells`` smooths the ice mask in SUBGRID pixels (smooth
        default ``oversample / 2``, half a model cell at any oversample; raw 0).
        An explicit value's native-grid strength is divided by oversample.
        ``ice_smoothing='mask'`` smooths a binary indicator; 'field' smooths
        clipped thickness min(H, 2*H_threshold) before thresholding. The field
        option requires extent='cells' and H_threshold>0.
        ``min_ice_cells`` removes small connected components and fills small
        enclosed holes in the final mask, including ribbons. Its unit is
        native cells; it respects looped seams and defaults to 0 (off).
        Component removal is opt-in because it can hide small glaciers.

        Colors and terrain
        ------------------
        ``ice_shading='veil'`` blends the depth ramp over terrain, its
        opacity an S-curve in column depth from 0 to fully opaque at
        ``ice_opaque_depth`` (m, default 100), 50 % at half of it, its
        steepness set by ``ice_alpha_sharpness`` (default 2; 1 is a straight
        line, higher fades thin ice further); 'flat' uses ``ice_color``.
        ``ice_cmap`` overrides either with opaque column-depth coloring.
        ``H_min``/``H_max`` bound the colour ramp's COLUMN depth, not H; the
        veil's opacity ignores them.
        The veil's automatic maximum is run-global hc_over_H*max(H_out)
        (max hc_cell_out*H_out on a mode-D run);
        an explicit ice_cmap otherwise scales its maximum per snapshot.
        ``trunk_alpha`` is the minimum opacity of opt-in ribbons under veil
        shading.
        ``lake_color`` (default ``COLORS['water']``, siim's one standing-water
        colour), ``lake_min_depth`` (m), ``lake_min_area`` (m²) govern lakes.

        ``z_min``/``z_max`` and ``cmap_bed`` set the terrain color scale
        (``z_min`` default -500 m, below which colours saturate). ``z_min`` is
        fixed, not the data minimum, so bed colours compare across frames and
        runs; ``map`` instead spans its field's data range. The default
        bed colormap is Crameri's turku with its darkest 20 % trimmed
        (embedded in ``siim.plotting._colormaps``, with its attribution); a
        light grey ramp under 'bedrock+speed', so the speed colours are the
        only colours on the map. The bed colorbar
        gets an arrow at each end the drawn surface saturates past;
        ``extend_bed`` ('neither', 'min', 'max' or 'both') fixes it
        (animate_landscape decides it once over the movie). The ice colorbar
        does the same where the model's ice column ``hc_over_H*H`` passes
        ``H_min``/``H_max`` (``extend_ice``): with the default run-global
        ``H_max`` it has none.
        ``oversample`` sets subgrid resolution. ``sigma_cells`` is terrain
        Gaussian smoothing in subgrid pixels; 0 disables smoothing.
        ``hillshade``, ``azdeg``/``altdeg`` (degrees) and ``ve`` (the
        hillshade's vertical exaggeration, default 4) set lighting;
        ``azdeg`` is the direction the light comes from, clockwise from
        north (315, the default, is the NW). The shading is stretched to
        each render's own illumination range; ``shade_stretch=(imin, imax)``
        fixes that range (animate_landscape freezes it over the movie).
        ``contour_interval`` is in meters, anchored at zero; 0 disables it.
        ``contour_color``/``contour_lw``/``contour_alpha`` style those contours.

        Overlays and composition
        ------------------------
        ``show_margin``, ``margin_color`` and ``margin_lw`` control the current
        ice outline (ribbons alone when present). The deprecated aliases
        ``show_trimline``, ``trimline_color`` and ``trimline_lw`` still work,
        with a warning, until their removal in 1.0.
        ``show_smoothed_paths`` adds a diagnostic centerline overlay, gated
        by ``channel_threshold`` (upstream area in m²), with
        ``smoothed_path_color``/``smoothed_path_lw`` styling; those three
        act only with ``show_smoothed_paths=True``.
        ``cross_section`` is y in km, or ``'mean'`` to average over the
        rendered rows of the section grid (an interpolated ``oversample`` grid,
        so the two end rows carry less ground than the interior ones) — an
        averaged profile whose ice band is the average ice column per unit x,
        with the ice thickness's 25-75% band across those rows shaded on
        the mean bed and the bed's quartiles dashed, both named in a legend,
        and with no map locator line and no lake layer
        (a reduced water table is not a water table). Either way it adds a
        section and bed hypsometry. ``section_ve`` (default 4) sizes the
        section panel for that vertical exaggeration over ``z_min``..``z_max``,
        its height clamped to 0.8-2.6 in; the panel prints the VE it got.
        ``cross_section_color`` styles its map locator; a y outside the
        domain warns and draws the edge row. The hypsometry shares the
        section's elevation axis (``z_min``..``z_max``): bed outside it is
        off the panel, and its bars are percentages of the whole area.
        ``hyp_max`` fixes the percent-area scale (default 15; a taller bar
        runs off it); None auto-scales it.

        ``fig_width`` is in inches (default 12). ``fig`` and supplied ``ax``/``ax_cs``/
        ``ax_hyp`` must belong to one figure. With a supplied map ax, supply
        ax_cs (and optionally ax_hyp) for section panels. ``colorbar=False``
        permits caller-managed shared bars. ``save`` names the image output.
        See ``docs/guides/outputs_and_io.md`` for examples and output paths.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import (LinearSegmentedColormap, LogNorm,
                                       Normalize, to_rgb)
        from scipy.ndimage import map_coordinates, gaussian_filter

        m = self.model
        supplied_axes = [axis for axis in (ax, ax_cs, ax_hyp) if axis is not None]
        if fig is None and supplied_axes:
            fig = supplied_axes[0].figure
        if any(axis.figure is not fig for axis in supplied_axes):
            raise ValueError('fig and all supplied axes must belong to one figure')
        # Deprecated 'trimline' spelling: the outline is the CURRENT margin.
        for old, new, val in (('show_trimline', 'show_margin', show_trimline),
                              ('trimline_color', 'margin_color', trimline_color),
                              ('trimline_lw', 'margin_lw', trimline_lw)):
            if val is not None:
                import warnings
                warnings.warn(f"{old} is deprecated and will be removed in 1.0; "
                              f"use {new} (the outline is the current ice "
                              "margin, not a trimline)",
                              DeprecationWarning, stacklevel=2)
        if show_trimline is not None and show_margin is None:
            show_margin = show_trimline
        if trimline_color is not None:
            margin_color = trimline_color
        if trimline_lw is not None:
            margin_lw = trimline_lw
        resolved = _resolve_style(
            style,
            field=field,
            H_threshold=H_threshold,
            ice_sigma_cells=ice_sigma_cells,
            sigma_cells=sigma_cells,
            oversample=oversample,
            hillshade=hillshade,
            ice_extent=ice_extent,
            show_margin=show_margin,
            area_threshold=area_threshold,
            contour_interval=contour_interval,
            ice_shading=ice_shading,
            trunk_display=trunk_display,
        )
        field = resolved.field
        H_threshold = resolved.H_threshold
        ice_sigma_cells = resolved.ice_sigma_cells
        sigma_cells = resolved.sigma_cells
        oversample = resolved.oversample
        hillshade = resolved.hillshade
        ice_extent = resolved.ice_extent
        show_margin = resolved.show_margin
        area_threshold = resolved.area_threshold
        contour_interval = resolved.contour_interval
        ice_shading = resolved.ice_shading
        trunk_display = resolved.trunk_display
        if field not in ('bedrock', 'bedrock+ice', 'bedrock+lakes',
                         'bedrock+speed'):
            raise ValueError("field must be 'bedrock', 'bedrock+ice', "
                             "'bedrock+lakes', or 'bedrock+speed', "
                             f"got {field!r}")
        show_speed = field == 'bedrock+speed'
        show_ice = field in _ICE_FIELDS
        show_lake = 'lake' in field
        # Lakes are drawn on ice-free ground only, so the lake layer needs the
        # ice extent the preset would draw, though it draws no ice itself.
        ice_geom = show_ice or show_lake
        if show_speed and ice_cmap is not None:
            raise ValueError("field='bedrock+speed' colours ice by speed; "
                             "use speed_cmap, not ice_cmap")
        if show_speed and not 0 < speed_min < speed_max:
            raise ValueError("need 0 < speed_min < speed_max, got "
                             f"{speed_min!r}, {speed_max!r}")
        if show_speed:
            warn_if_closure_pair_missing(m)
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
        for name, value in (('extend_bed', extend_bed), ('extend_ice', extend_ice)):
            if value is not None and value not in _BED_EXTENDS:
                raise ValueError(f"{name} must be one of {_BED_EXTENDS} or "
                                 f"None, got {value!r}")
        section_ve = float(section_ve)
        check_range('section_ve', section_ve, 0.0)
        # Resolve the section selector BEFORE anything is drawn: a typo used to
        # raise only after the map, its colorbars and the panels existed,
        # leaking the figure.
        if isinstance(cross_section, str):
            if cross_section != 'mean':
                raise ValueError("cross_section must be a y in km or 'mean', "
                                 f"got {cross_section!r}")
        elif cross_section is not None:
            cross_section = float(cross_section)
        if shade_stretch is not None:
            lo, hi = (float(v) for v in shade_stretch)
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError("shade_stretch must be a finite (lo, hi) with "
                                 f"lo < hi, got {shade_stretch!r}")
        ice_opaque_depth = float(ice_opaque_depth)      # a scalar depth, not a field
        check_range('ice_opaque_depth', ice_opaque_depth, 0.0)
        ice_alpha_sharpness = float(ice_alpha_sharpness)
        check_range('ice_alpha_sharpness', ice_alpha_sharpness, 0.0)

        z_in = m.z_out[i]
        H_in = m.H_out[i]
        # Column datum for frame i: the scalar hc_over_H, or the per-cell
        # hc_cell (1 on sheet cells to 1.5 on valley cells, fractional where s blends) on a mode-D run.
        hc_cell = getattr(m, 'hc_cell_out', None)
        hc_in = m.hc_over_H if hc_cell is None else np.asarray(hc_cell[i])
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
        ribbons = ice_geom and trunk_display == 'ribbons'
        trunk_cells = None
        if ribbons:
            # Only ribbon sources are hard-gated here. Field smoothing needs
            # the threshold-adjacent veil thickness until its final level set.
            H_ribbon = np.where((H_in > H_threshold)
                                & (area_in >= area_threshold), H_in, 0.0)
            cell = max(m.Lx / (nx - 1), m.Ly / (ny - 1))
            trunk_cells = _channel_closure(
                H_ribbon, rec_in, m.alpha_g * np.asarray(H_ribbon)
                >= trunk_width_cells * cell)

            def _ribbons(H_src, cells, payload=None):
                # payload: the per-cell scalar carried along each ribbon —
                # the ice surface by default, the speed for 'bedrock+speed'
                if payload is None:
                    payload = np.asarray(zb_in) + hc_in * np.asarray(H_src)
                return _trunk_ribbons(
                    rec_in, H_src, area_in, payload,
                    cells, nx, ny, m.Lx, m.Ly, m.alpha_g, oversample,
                    wrap_y=wrap_y, wrap_x=wrap_x, hc_over_H=hc_in)
        H_veil = np.where(trunk_cells, 0.0, H_in) if ribbons else H_in

        # Per-cell ice surface speed (m/yr) of frame i, recomputed from the
        # closure's own (H, Qg) pair (as siim.ice_speed).
        speed_nat = speed_sub = None
        if show_speed:
            speed_nat = ice_speeds(*closure_pair(m, i), m.sliding_law,
                                   m.alpha_g, m.Ac, m.lambda_p, m.lambda_c,
                                   m.tau_c, receivers=rec_in)[0]

        if not ice_geom:
            z_sub = zb_sub
            ice_mask = np.zeros_like(zb_sub, dtype=bool)
            H_col = np.zeros_like(zb_sub)
        elif ice_extent == 'footprint':
            # Footprint fill (display dual of the width carve): per-cell ice
            # surface from the power-diagram attribution, computed native + up.
            # Speed rides the same attribution: each footprint cell takes the
            # speed of the source whose ice surface it shows.
            fill = _footprint_ice_surface(
                H_veil, zb_in, rec_in, m.alpha_g,
                m.Ly / (ny - 1), m.Lx / (nx - 1),
                wrap_y=wrap_y, wrap_x=wrap_x,
                H_threshold=H_threshold, area_in=area_in,
                area_threshold=area_threshold, hc_over_H=hc_in,
                value=speed_nat)
            z_fill_nat, ice_nat, depth_nat = fill[:3]
            # order-1 (bilinear) for the ice-filled surface: it carries a
            # trimline step (flat z_s meeting terrain), and bicubic overshoots
            # there into a faint margin halo; the later Gaussian smooth recovers
            # interior smoothness regardless.
            z_sub = map_coordinates(z_fill_nat, [Y, X], order=1, mode='nearest')
            ice_mask = _smooth_ice_mask(ice_nat, Y, X, ice_sigma_cells)
            # depth for ice_cmap colouring: bilinear + mask-confined. order-0
            # staircased against the bicubic terrain; a plain order-1 upsample
            # bleeds toward 0 across the ice/rock edge, so renormalise by the
            # upsampled ice fraction to keep full column depth at the margin.
            ice_frac = map_coordinates(ice_nat.astype(float), [Y, X],
                                       order=1, mode='nearest')
            depth_up = map_coordinates(depth_nat, [Y, X], order=1, mode='nearest')
            depth_up = np.divide(depth_up, ice_frac, out=np.zeros_like(depth_up),
                                 where=ice_frac > 1e-3)
            H_col = depth_up
            if show_speed:
                # same mask-confined bilinear upsample as the depth
                speed_up = map_coordinates(fill[3], [Y, X], order=1,
                                           mode='nearest')
                speed_sub = np.divide(speed_up, ice_frac,
                                      out=np.zeros_like(speed_up),
                                      where=ice_frac > 1e-3)
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
            hc_sub = hc_in if hc_cell is None else map_coordinates(
                hc_in, [Y, X], order=0, mode='nearest')
            H_col = hc_sub * H_sub
            if show_speed:
                speed_sub = map_coordinates(speed_nat, [Y, X], order=0,
                                            mode='nearest')

        trunk_mask = None
        if ribbons:
            # True-width ribbons over the veil: the sub-cell apron stays a
            # veil, the resolved trunks become coherent tongues with their own
            # (flat) ice surface. The ribbon carries BOTH the drawn column
            # depth and that surface, so the hillshade shades the glacier and
            # the cross-section reads the same composite the map does.
            trunk_mask = np.zeros_like(ice_mask)
            if trunk_cells.any():
                trunk_depth, trunk_zs = _ribbons(H_ribbon, trunk_cells)
                trunk_mask = trunk_depth > 0.0
                H_col = np.where(trunk_mask, trunk_depth, H_col)
                if show_speed:
                    # each ribbon pixel: the speed of its own path node
                    speed_sub = np.where(trunk_mask, _ribbons(
                        H_ribbon, trunk_cells, speed_nat)[1], speed_sub)
                ice_mask = ice_mask | trunk_mask
                # ...and the ribbon SURFACE is terrain (it goes into the
                # composite), read only where the ribbons have depth (elsewhere
                # the surface is unset; audit #69), never below the local bed
                # (a ribbon over rising ground)
                z_sub = np.where(trunk_mask, np.maximum(trunk_zs, zb_sub), z_sub)

        if ice_geom:
            # Count each connected glacier once, after its veil and ribbons
            # have joined. Keep depth data until cleanup also fills any holes.
            ice_mask = _clean_ice_mask(ice_mask, min_ice_cells, oversample,
                                      wrap_y=wrap_y, wrap_x=wrap_x)
            if ribbons:
                trunk_mask &= ice_mask
            H_col = np.where(ice_mask, H_col, 0.0)

        paths_xy = None
        if show_smoothed_paths:
            paths_xy = _smoothed_glacier_paths(
                rec_in, H_in, area_in, nx, ny, m.Lx, m.Ly,
                m.alpha_g, channel_threshold, oversample)

        # Terrain surface: the ice-inflated surface (z_sub = zb + hc*H) ONLY
        # where ice is actually DRAWN; the bare bed (zb_sub) everywhere else.
        # Ice filtered out by H_threshold/area_threshold otherwise still bulges
        # the terrain by hc*H and — since that H grows/melts frame to frame — makes
        # the "bedrock" pulse in animations (and shifts the auto z_max/hillshade).
        z_composite = (np.where(ice_mask, z_sub, zb_sub) if show_ice
                       else zb_sub)
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

        if cmap_bed is None and show_speed:
            # light grey only, so the speed colours are the only colours on
            # the map (a coloured bed reads as more speed)
            cmap_bed = LinearSegmentedColormap.from_list(
                'bed_grey', ['#8a8a8a', '#f2f2f2'])
        cmap = bed_cmap() if cmap_bed is None else plt.get_cmap(cmap_bed)
        norm = Normalize(vmin=z_min, vmax=z_max)
        rgb = cmap(norm(z_smooth))[..., :3]

        ice_cmap_obj = None
        ice_norm = None
        speed_cmap_obj = speed_norm = None
        # 'bedrock+speed' replaces the ice colouring (veil / flat / ice_cmap)
        # with the speed layer; the ice GEOMETRY above is shared.
        veil = (show_ice and not show_speed and ice_cmap is None
                and ice_shading == 'veil')
        if show_speed:
            speed_cmap_obj = plt.get_cmap(speed_cmap)
            speed_norm = LogNorm(vmin=speed_min, vmax=speed_max)
        elif show_ice and ice_cmap is not None:
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
            a_veil = np.where(ice_mask, _veil_alpha(H_col, ice_opaque_depth,
                                                    ice_alpha_sharpness), 0.0)
            if trunk_mask is not None:
                # The ribbons must read as GLACIERS, not as more veil: paint
                # them at trunk_alpha, a FLOOR (deep enough ice is already at
                # least this opaque on the veil's S-curve)
                # — the bed stops showing through the trunk, so trunk and apron
                # separate at a glance.
                a_veil = np.where(trunk_mask, np.maximum(a_veil, trunk_alpha),
                                  a_veil)
            a_veil = a_veil[..., np.newaxis]
            rgb = (1.0 - a_veil) * rgb + a_veil * ice_cmap_obj(t_veil)[..., :3]
        elif show_speed and speed_sub is not None:
            # Slower than speed_min stays transparent; faster saturates.
            fast = ice_mask & (speed_sub >= speed_min)
            rgb[fast] = speed_cmap_obj(speed_norm(speed_sub[fast]))[..., :3]
        elif show_ice and ice_cmap_obj is not None:
            rgb[ice_mask] = ice_cmap_obj(ice_norm(H_col[ice_mask]))[..., :3]
        elif show_ice:
            rgb[ice_mask] = np.asarray(to_rgb(ice_color))
        # The second colour bar: surface speed, or column depth.
        ice_bar = None
        if show_speed:
            ice_bar = (speed_cmap_obj, speed_norm, 'Ice surface speed (m/yr)')
        elif show_ice and ice_cmap_obj is not None:
            # What the map paints is the local COLUMN depth to the flat ice
            # surface (hc_over_H * H at the thalweg, more on carved flanks)
            # — not the width-mean H that map(field='ice') shows.
            ice_bar = (ice_cmap_obj, ice_norm, 'Ice column depth (m)')
        if ice_bar is not None and extend_ice is None:
            if show_speed:
                extend_ice = 'max'                  # faster than speed_max saturates
            else:
                # an arrow at each end the model's ice column (hc*H) passes the
                # colour scale, as the bed bar does; animate_landscape applies
                # the same rule once over its frames
                col = _ice_column(H_in, hc_in)
                # an ice_cmap top left to the frame is its own thickest drawn
                # column, so no colour saturates: no arrow there (the model's
                # unsmoothed column tops the drawn one by a hair)
                auto_top = ice_cmap is not None and H_max is None
                extend_ice = (_ice_extend(np.nanmin(col),
                                          ice_norm.vmax if auto_top else np.nanmax(col),
                                          ice_norm.vmin, ice_norm.vmax)
                              if col.size else 'neither')

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
            # ...and drawn on ice-free ground only: a flooded bed under the
            # ice is not a lake anyone sees.
            lake_mask = (((lake_filled - z_composite) > lake_min_depth)
                         & ~ice_mask)
            if lake_min_area > 0 and lake_mask.any():
                # seam-aware, so a lake across a looped edge counts whole
                labeled = _label_looped(lake_mask, wrap_y, wrap_x)
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
                                  ve, azdeg, altdeg, stretch=shade_stretch)

        # Every inch below that holds text (tick labels, axis and bar labels)
        # grows with the active font sizes: they were tuned for 9 pt ticks and
        # 10 pt labels, and fixed inches clipped larger rcParams type off the
        # canvas (movie frames are not cropped).
        k = _text_scale()
        bar_width, bar_pad, ice_bar_pad = 0.12, 0.18, 0.7 * k
        hyp_width = 0.8
        if ax is None:
            # Size the map at its geographic aspect, with typographic margins
            # in inches. Relative padding inflated the gap below square maps
            # and made the bars/hypsometry dominate long, narrow domains.
            bar_space = bar_pad + bar_width if colorbar else 0.0
            if colorbar and ice_bar is not None:
                bar_space += ice_bar_pad + bar_width
            left, bottom, top, gap = 0.65 * k, 0.5 * k, 0.3 * k, 0.4 * k
            if cross_section == 'mean':
                # the IQR legend sits in one row in this gap
                gap += 1.0 * _font_points('legend.fontsize') / 72.0
            right_margin = 0.65 * k if colorbar else 0.25 * k
            if cross_section is not None:
                right_margin = max(right_margin,
                                   bar_pad + hyp_width + 0.2 * k - bar_space)
            map_width = fig_width - left - right_margin - bar_space
            map_height = map_width * (m.Ly / m.Lx)
            # the section panel's height gives it section_ve over its
            # z_min..z_max range (its ylim), within readable bounds
            ve_height = section_ve * (z_max - z_min) / m.Lx * map_width
            cs_height = float(np.clip(ve_height if np.isfinite(ve_height) else 0.8,
                                      0.8, 2.6))
            fig_height = map_height + bottom + top
            if cross_section is not None:
                fig_height += cs_height + gap
            if fig is None:
                fig = plt.figure(figsize=(fig_width, fig_height))
            margins = dict(left=left / fig_width,
                           right=1 - right_margin / fig_width,
                           bottom=bottom / fig_height, top=1 - top / fig_height)
            if cross_section is not None:
                gs = fig.add_gridspec(
                    2, 2, height_ratios=[map_height, cs_height],
                    width_ratios=[map_width, 1.0],
                    hspace=2 * gap / (map_height + cs_height), wspace=0,
                    **margins)
                ax = fig.add_subplot(gs[0, :])
                if ax_cs is None:
                    ax_cs = fig.add_subplot(gs[1, 0])
                if ax_hyp is None:
                    ax_hyp = fig.add_subplot(gs[1, 1], sharey=ax_cs)
            else:
                gs = fig.add_gridspec(1, 1, **margins)
                ax = fig.add_subplot(gs[0, 0])
        fig = ax.figure
        # imshow places pixel CENTRES inside `extent`, but the array is
        # NODE-valued: run the raster half a subgrid pixel past each edge so
        # node j lands on x = Lx*j/(nx_sub-1) — the coordinate the contours,
        # trimline and section line use. (Axis limits stay at [0, Lx]/[0, Ly].)
        extent = _node_extent(m.Lx, m.Ly, nx_sub, ny_sub)
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
                alpha=contour_alpha, zorder=2, algorithm=_CONTOUR_ALGORITHM)

        # Crisp glacier margin: the 1/2-level of the (smoothed) ice mask — the
        # same outline the fill is thresholded at, drawn as a thin line so the
        # ice extent reads at a glance and a one-cell footprint flip is a small
        # wobble rather than a colour snap. By default (no ribbons) that is the
        # true ice edge. Under opt-in ribbons it outlines the RIBBONS only, as
        # it did when they were the default: on a sheet that fences thinly
        # iced highs off as rock (the 2026-11-05 ribbon study).
        margin_of = ice_mask if trunk_mask is None else trunk_mask
        if show_ice and show_margin and margin_of.any():
            ax.contour(x_axis, y_axis, margin_of.astype(float), levels=[0.5],
                       colors=[margin_color], linewidths=margin_lw,
                       alpha=0.9, zorder=2.6, algorithm=_CONTOUR_ALGORITHM)

        if show_smoothed_paths and paths_xy:
            for px_d, py_d in paths_xy:
                ax.plot(px_d / 1e3, py_d / 1e3, color=smoothed_path_color,
                        lw=smoothed_path_lw, zorder=3)

        ax.set_aspect('equal')
        if cross_section is None:
            ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_title(time_label(m.output_times[i]), loc='right')

        if colorbar:
            sm_bed = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            # Without ice the composite IS the bare bed — name it as map() does.
            bed_label = ('Surface elevation (m)' if show_ice
                         else 'Bedrock elevation (m)')
            # One divider keeps both bars flush with the map's top/bottom.
            # Fixed inch widths stay slender even on a 20-inch panorama.
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            if extend_bed is None:
                # The bed colours saturate at z_min / z_max by design: an
                # arrow on the bar says where the drawn surface goes past
                # (the composite, as the section draws it; the map's
                # smoothing only pulls its extremes in).
                extend_bed = _bed_extend(np.nanmin(z_composite) < z_min,
                                         np.nanmax(z_composite) > z_max)
            _add_colorbar(sm_bed, ax, bed_label, size=bar_width, pad=bar_pad,
                          divider=divider, extend=extend_bed)
            if ice_bar is not None:
                bar_cmap, bar_norm, bar_label = ice_bar
                sm_ice = plt.cm.ScalarMappable(cmap=bar_cmap, norm=bar_norm)
                # Clear the first bar's ticks and label at paper widths too.
                cb_ice = _add_colorbar(
                    sm_ice, ax, bar_label, size=bar_width, pad=ice_bar_pad,
                    divider=divider, extend=extend_ice)
                if show_speed:
                    # 1-2-5 labels: bare decades leave 20-1000 one label
                    from matplotlib.ticker import FormatStrFormatter, LogLocator
                    cb_ice.ax.yaxis.set_major_locator(
                        LogLocator(subs=(1.0, 2.0, 5.0)))
                    cb_ice.ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

        if cross_section is not None:
            # 'mean' takes every RENDERED row of the section grid (ny_sub rows,
            # so oversample-dependent) instead of reading one, so the painted
            # band is the average ice column per unit x (and vanishes where
            # nothing is iced) and no single y can be marked. The quartile
            # bands below carry the spread the mean hides.
            # Validated to 'mean' or a float above.
            reduce_y = np.mean if cross_section == 'mean' else None
            if reduce_y is None:
                y_km = cross_section
                j_cs = max(0, min(ny_sub - 1, int(round(y_km * 1e3 / dy_sub))))
                if not -1e-9 <= y_km <= m.Ly / 1e3 + 1e-9:   # float slack at the edges
                    # clamped to the edge row, as before — but not silently
                    import warnings
                    warnings.warn(
                        f"cross_section y = {y_km:g} km lies outside the "
                        f"domain's y range [0, {m.Ly / 1e3:g}] km; drawing its "
                        f"edge row {j_cs} (y = {j_cs * dy_sub / 1e3:g} km)",
                        UserWarning, stacklevel=2)
                ax.axhline(y_km, color=cross_section_color, lw=1.0, alpha=0.8,
                           zorder=3)
            else:
                j_cs = slice(None)
            if ax_cs is not None:
                # The section reads the PRE-smoothing composite, never z_smooth:
                # the map's Gaussian (sigma_cells) bridges bed slots narrower
                # than its kernel, and painting the bridge-to-bed gap as ice
                # fabricates phantom glaciers in relict carved notches (H = 0).
                # Unsmoothed, the profile collapses onto the bed wherever no
                # ice is drawn, so the ice band below is exactly the drawn ice
                # column and nothing else.
                cs_rows = z_composite[j_cs]
                cs_profile = cs_rows if reduce_y is None else reduce_y(cs_rows, axis=0)
                cs_x = np.linspace(0, m.Lx / 1e3, nx_sub)
                n_z = 200
                z_grid = np.linspace(z_min, z_max, n_z).reshape(-1, 1)
                row_colors = cmap(norm(z_grid.flatten()))
                # The section images sample the nodes (cs_x) and z_grid, so
                # centre their pixels there, as the map's extent does.
                dz_cs = (z_max - z_min) / (n_z - 1)
                cs_extent = extent[:2] + [z_min - dz_cs / 2, z_max + dz_cs / 2]

                if show_ice:
                    # Bilinear bed for the SECTION only: the map's BICUBIC
                    # zb_sub overshoots above the (bilinear) filled ice
                    # surface, which zeroed the ice band on columns the map
                    # paints as ice. Match orders instead — and keep the bed
                    # ON the profile wherever no ice is drawn, so an ice-free
                    # column still paints a zero-thickness band (no phantom
                    # ice from the order-3/order-1 gap).
                    cs_zb = map_coordinates(zb_in, [Y[j_cs], X[j_cs]],
                                            order=1, mode='nearest')
                    cs_bed = np.where(ice_mask[j_cs],
                                      np.minimum(cs_zb, cs_rows), cs_rows)
                    cs_bedrock = cs_bed if reduce_y is None else reduce_y(cs_bed, axis=0)
                    upper_for_bed = cs_bedrock
                else:
                    cs_bedrock = None
                    upper_for_bed = cs_profile

                bed_mask = (z_grid <= upper_for_bed.reshape(1, -1)).astype(float)
                bed_img = np.broadcast_to(row_colors[:, np.newaxis, :],
                                          (n_z, nx_sub, 4)).copy()
                bed_img[..., 3] = bed_mask
                ax_cs.imshow(bed_img, origin='lower',
                             extent=cs_extent,
                             aspect='auto', interpolation='nearest', zorder=1)

                if show_ice:
                    ice_mask_2d = ((z_grid >= cs_bedrock.reshape(1, -1)) &
                                   (z_grid <= cs_profile.reshape(1, -1))).astype(float)
                    ice_img = np.zeros((n_z, nx_sub, 4))
                    if ice_cmap_obj is not None:
                        # Colour each PIXEL by its depth BELOW THE ICE SURFACE
                        # on the same norm as the map's column-depth bar. The
                        # old per-column H painted every band a single flat
                        # colour, so the section read as vertical bars keyed to
                        # total thickness; grading with depth puts the ramp
                        # where the section actually resolves it (light at the
                        # surface, saturating down towards the bed).
                        cs_depth = np.clip(cs_profile.reshape(1, -1) - z_grid,
                                           0.0, None)
                        ice_img[..., :3] = ice_cmap_obj(ice_norm(cs_depth))[..., :3]
                    else:
                        ice_img[..., :3] = np.asarray(to_rgb(ice_color))
                    ice_img[..., 3] = ice_mask_2d
                    ax_cs.imshow(ice_img, origin='lower',
                                 extent=cs_extent,
                                 aspect='auto', interpolation='nearest', zorder=2)

                if show_lake and lake_filled is not None and reduce_y is None:
                    cs_lake = lake_filled[j_cs, :]
                    cs_in_lake = lake_mask[j_cs, :]
                    lake_mask_2d_cs = ((z_grid >= cs_profile.reshape(1, -1)) &
                                       (z_grid <= cs_lake.reshape(1, -1)) &
                                       cs_in_lake.reshape(1, -1)).astype(float)
                    lake_img = np.zeros((n_z, nx_sub, 4))
                    lake_img[..., :3] = np.asarray(to_rgb(lake_color))
                    lake_img[..., 3] = lake_mask_2d_cs
                    ax_cs.imshow(lake_img, origin='lower',
                                 extent=cs_extent,
                                 aspect='auto', interpolation='nearest', zorder=2.5)

                if reduce_y is not None:
                    # What the mean hides: the SPREAD across the rows it
                    # averaged. The ice THICKNESS gets a translucent 25-75%
                    # band hung on the mean bed (a surface band mostly repeats
                    # the bed's spread — measured, decision
                    # 2026-09-22-section-band-ice-thickness-iqr); the BED gets
                    # two dashed quartile lines instead — a second
                    # filled band under the plateau read as more ice rather
                    # than as topography. Both ride above the section's ice
                    # fill, which is OPAQUE: the bed quartiles reach well into
                    # the ice column (+327 m over the mean bed on the mode-C
                    # test run), so underneath it they would simply vanish.
                    handles = []
                    if show_ice:
                        lo, hi = cs_bedrock + np.percentile(cs_rows - cs_bed,
                                                            [25, 75], axis=0)
                        handles.append(ax_cs.fill_between(
                            cs_x, lo, hi, color=COLORS['ice'], alpha=0.22, lw=0,
                            zorder=2.6, label='ice IQR'))
                    for edge in np.percentile(cs_bed if show_ice else cs_rows,
                                              [25, 75], axis=0):
                        line, = ax_cs.plot(cs_x, edge, color='black',
                                           linestyle='--', lw=0.8, zorder=2.7,
                                           label='bed IQR')
                    handles.append(line)
                    # Legend from this render's own artists (a re-used supplied
                    # ax_cs would otherwise list every earlier render too), in
                    # ONE row in the gap above the section (Eric, 2026-11-05),
                    # clear of the profile it used to sit on.
                    ax_cs.legend(handles=handles, loc='lower left',
                                 bbox_to_anchor=(0.0, 1.0), ncol=len(handles),
                                 borderaxespad=0.15, frameon=False,
                                 handlelength=1.6, columnspacing=1.2)

                if contour_obj is not None:
                    for lev in contour_obj.levels:
                        ax_cs.axhline(lev, color='lightgray', lw=0.3,
                                      alpha=0.3, zorder=3)

                if show_ice:
                    ax_cs.plot(cs_x, cs_bedrock, color=COLORS['bed'], lw=1.0, zorder=4)
                    ax_cs.plot(cs_x, cs_profile, color=COLORS['ice'], lw=0.5, zorder=4)
                else:
                    ax_cs.plot(cs_x, cs_profile, color=COLORS['bed'], lw=1.0, zorder=4)

                ax_cs.axhline(0.0, color='black', lw=0.8, zorder=4)
                # The ELA is the one forcing line in the panel — colour and
                # name it so it is not read as another contour.
                # An ELA outside the panel's elevation range is not drawn:
                # axhline clips, but the annotation does NOT (text defaults to
                # clip_on=False), so its white-boxed label used to float outside
                # the axes and collide with the map above.
                zela = float(m._zELA_output[i])
                if z_min <= zela <= z_max:
                    ax_cs.axhline(zela, color=COLORS['ela'], linestyle='--', lw=0.9,
                                  zorder=4)
                    ax_cs.annotate('ELA', xy=(0.995, zela),
                                   xycoords=('axes fraction', 'data'),
                                   ha='right', va='bottom', color=COLORS['ela'],
                                   zorder=4,
                                   bbox=dict(fc='white', ec='none', alpha=0.7,
                                             pad=0.5))
                ax_cs.set_xlim(0, m.Lx / 1e3)
                ax_cs.set_ylim(z_min, z_max)
                ax_cs.set_xlabel('x (km)')
                ax_cs.set_ylabel('Elevation (m)')
                ax_cs.spines['top'].set_visible(False)
                ax_cs.spines['right'].set_visible(False)
                _place_axes(fig)
                pos_main = ax.get_position()
                pos_cs = ax_cs.get_position()
                ax_cs.set_position([pos_main.x0, pos_cs.y0,
                                    pos_main.width, pos_cs.height])
                # Vertical exaggeration of the section, from the axes' own
                # display box vs its data range — the panel is much wider than
                # tall, so the relief it shows is not to scale.
                bb = ax_cs.get_window_extent()
                ve_cs = ((bb.height / max(z_max - z_min, 1e-9))
                         / (bb.width / m.Lx))
                ax_cs.annotate(f'VE {ve_cs:.0f}$\\times$', xy=(0.005, 0.94),
                               xycoords='axes fraction', ha='left', va='top',
                               color='#666666', zorder=4)
                if ax_hyp is not None:
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
                    hist = _hypsometry_counts(zb_sub.ravel(), edges)
                    pct = 100.0 * hist / zb_sub.size
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    heights = np.diff(edges)
                    ax_hyp.barh(centers, pct, height=heights, align='center',
                                color=cmap(norm(centers)), edgecolor='black',
                                linewidth=0.3)
                    ax_hyp.axhline(0.0, color='black', lw=0.8, zorder=4)
                    if z_min <= zela <= z_max:
                        ax_hyp.axhline(zela, color=COLORS['ela'],
                                       linestyle='--', lw=0.9, zorder=4)
                    ax_hyp.set_ylim(z_min, z_max)
                    ax_hyp.tick_params(axis='y', which='both',
                                       left=False, labelleft=False)
                    ax_hyp.tick_params(axis='x', which='minor', bottom=False)
                    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
                    ax_hyp.xaxis.set_major_formatter(FormatStrFormatter('%g'))
                    # At narrow fig_width the default tick set put a 0 hard
                    # against the section's last x tick — three ticks, no zero.
                    ax_hyp.xaxis.set_major_locator(MaxNLocator(3, prune='lower'))
                    ax_hyp.set_xlabel(r'$\%$ area')
                    if hyp_max is None:
                        ax_hyp.set_xlim(left=0)
                    else:
                        ax_hyp.set_xlim(0, hyp_max)
                    ax_hyp.spines['top'].set_visible(False)
                    ax_hyp.spines['right'].set_visible(False)
                    pos_cs_now = ax_cs.get_position()
                    width_in = fig.get_figwidth()
                    hyp_x0 = pos_cs_now.x1 + bar_pad / width_in
                    ax_hyp.set_position([hyp_x0, pos_cs_now.y0,
                                         hyp_width / width_in,
                                         pos_cs_now.height])

        if save is not None:
            fig.savefig(output_path(save, 'images'), dpi=200, bbox_inches='tight')
        return fig, ax

    @_refuse_ice_time_avg
    def animate_landscape(self, path='landscape_animate', run_id=None, *,
                          interval=42, fps=None, frames=None, fig_width=12,
                          n_workers=None, quality='high', **landscape_kwargs):
        """MP4 of ``landscape`` (its kwargs passed through). Returns the path.

        ``interval`` is milliseconds per frame; an explicit ``fps`` overrides it
        on both render paths. ``frames`` encodes a subset of the saved frames —
        a slice, a range or a sequence of indices, negatives counting from the
        end — in the order given.

        ``quality='high'`` (the default) is a movie to keep: the still render's
        full look. ``quality='draft'`` is for research movies that will soon be
        replaced: at most ``oversample=2`` (so ``ice_sigma_cells=1``; ``'raw'``
        stays at 1) and x264's ``veryfast`` preset (with the default ``h264``
        codec alone), about 1.6-1.7x faster. An explicit ``oversample`` wins.

        Frames are independent, so they render IN PARALLEL by default
        (audit N35): the arrays ``landscape`` reads, for the encoded frames
        alone, are dumped once and
        loaded by ``n_workers`` spawned processes — ONLY those five arrays
        (``field='bedrock+speed'`` adds the closure's flux, ``Qg_out`` or
        ``closure_Qg_out``, and ``closure_H_out`` where the run has it)
        plus a small metadata pickle, never the full run state, and the
        worker count is additionally capped so the workers' combined array
        memory stays within ~25% of system RAM. Each worker renders its
        share of frames as raw RGBA via the unchanged :meth:`landscape`, and
        the frames stream into ffmpeg in movie order while the rest render;
        on this path the MP4 replaces ``path`` only once it is complete (the
        serial render writes ``path`` as it encodes). The workers take the
        calling process's ``rcParams``; a colormap registered by name, or
        anything else a script sets up at import, does not reach them.
        ``n_workers=None`` auto-sizes (up to 8, capped by cores, frame count
        and the RAM budget, and only when a timed probe frame projects the
        serial render past ~10 s); ``n_workers=1`` is the original
        in-process serial render (also the automatic fallback when
        ``fig``/``ax`` are passed or the model lacks the arrays the workers
        need). The two paths size the frame differently (matplotlib's writer
        resizes the figure to even pixels, the parallel path crops a row), so
        their frames can differ by a few colour levels.

        The auto colour scales are FROZEN over the movie: ``z_max`` and the ice
        norm's ``H_max`` are resolved once across the ENCODED frames instead of
        per frame, so a frame's colours mean the same thing throughout the
        movie; pass explicit values to override. (Over the whole run, ``H_max``
        freezes to the run-global ``hc_over_H * max H_out``, which is already
        the still render's default for the depth-graded veil; a ``frames``
        window scales to that window, not to relief it never shows.) The bed
        bar's saturation arrows (``extend_bed``) are decided once too, so the
        bar keeps one shape.

        Inherits ``landscape``'s ``style='smooth'`` default (the cartographic
        view — supersampled + hillshaded terrain, footprint-width ice drawn as
        a depth-graded translucent veil, with nothing hidden by thickness —
        ``H_threshold=0``). Pass ``style='raw'`` for the
        naked model output (one pixel per cell, no hillshade or smoothing, flat
        ice cells shown directly), or override individual knobs (explicit values
        always win over the preset). A larger ``ice_sigma_cells`` globs and
        erases thin ice, and ``min_ice_cells=6`` drops specks but HIDES real
        small glacierets — both deliberately in no preset. Keep
        ``ice_extent='footprint'`` (the default) for the width-honest view (ice
        across the claimed width ``W = alpha_g*H``); ``'cells'`` shows the raw
        channel-cell state. See :meth:`landscape` for the full knob reference.
        """
        import os
        import matplotlib.pyplot as plt
        m = self.model
        if 'save' in landscape_kwargs:
            raise ValueError(
                "animate_landscape renders every frame; per-frame 'save' "
                "makes no sense here — use 'path' for the movie file.")
        path = movie_path(path, run_id, 'landscape')[:-4]
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError('interval must be positive and finite')
        if fps is not None and (not np.isfinite(fps) or fps <= 0):
            raise ValueError('fps must be positive and finite')
        if quality not in _MOVIE_QUALITY:
            raise ValueError(f"quality must be one of {sorted(_MOVIE_QUALITY)}, "
                             f"got {quality!r}")
        oversample_cap, encoder_args = _MOVIE_QUALITY[quality]
        if plt.rcParams['animation.codec'] != 'h264':
            # the preset is x264's, and 'h264' is the one name matplotlib's
            # writer treats as x264 (its yuv420p rule), on both render paths
            encoder_args = None
        if oversample_cap is not None and landscape_kwargs.get('oversample') is None:
            # a cap on the style's own default, so 'raw' keeps one pixel a cell
            landscape_kwargs['oversample'] = min(
                oversample_cap, _resolved_style_of(landscape_kwargs).oversample)
        nframes = len(m.output_times)
        # Resolve the selection ONCE (negatives included); n_out is what the
        # worker gates and the encoder actually render. The colour scales
        # freeze over THOSE frames, so a windowed movie is not washed out by
        # relief it never shows.
        frames = frame_indices(nframes, frames)
        n_out = len(frames)
        landscape_kwargs = _frozen_scale_kwargs(m, landscape_kwargs, frames)
        anim_arrays, anim_meta, anim_optional = _anim_fields(landscape_kwargs)

        parallel_ok = (
            not any(landscape_kwargs.get(key) is not None
                    for key in ('fig', 'ax', 'ax_cs', 'ax_hyp'))
            and all(hasattr(m, a) for a in anim_arrays + anim_meta)
            and n_out >= 4)
        if n_workers is None and parallel_ok:
            # Adaptive: time one probe frame, its savefig included (a frame is
            # about half render, half savefig; timing the render alone sent
            # short movies down the 1.8x slower serial path, ED-168). Parallel
            # once the projected serial render clearly beats the pool's ~2 s
            # startup (spawn, import, bundle load, first-frame warm-up).
            import io
            import time
            t0 = time.perf_counter()
            fig_probe, _ = self.landscape(i=frames[0], fig_width=fig_width,
                                          **landscape_kwargs)
            fig_probe.savefig(io.BytesIO(), dpi=MOVIE_DPI, format='rgba')
            plt.close(fig_probe)
            t_frame = time.perf_counter() - t0
            n_workers = (min(8, os.cpu_count() or 1, n_out)
                         if t_frame * n_out > 10.0 else 1)
        elif n_workers is None:
            n_workers = 1
        n_workers = max(1, int(n_workers))
        if n_workers > 1 and not parallel_ok:
            n_workers = 1
        if n_workers > 1:
            # RAM-budget cap: each worker plain-loads the five bundle arrays,
            # so keep n_workers * bundle within ~25% of system RAM
            # (swap-storm guard; the reason we never ship the full ds_out).
            # The bundle holds the encoded frames, each once.
            bundle_bytes = len(set(frames)) * sum(
                np.asarray(getattr(m, name))[0].nbytes
                for name in _anim_shipped(m, anim_arrays + anim_optional))
            try:
                page_size = os.sysconf('SC_PAGE_SIZE')
                page_count = os.sysconf('SC_PHYS_PAGES')
                if page_size <= 0 or page_count <= 0:
                    n_workers = 1
                else:
                    total_ram = page_size * page_count
                    n_workers = max(1, min(n_workers,
                                           int(0.25 * total_ram
                                               / max(bundle_bytes, 1))))
            except (AttributeError, ValueError, OSError):
                # Without a RAM estimate, avoid parallel copies of the bundle.
                n_workers = 1

        if n_workers > 1:
            return self._animate_parallel(path, interval, fig_width,
                                          n_workers, landscape_kwargs,
                                          frames=frames, fps=fps,
                                          encoder_args=encoder_args)

        supplied_axes = [landscape_kwargs[key] for key in ('ax', 'ax_cs', 'ax_hyp')
                         if landscape_kwargs.get(key) is not None]
        supplied_fig = landscape_kwargs.get('fig')
        if supplied_fig is None and supplied_axes:
            supplied_fig = supplied_axes[0].figure
        if any(ax.figure is not supplied_fig for ax in supplied_axes):
            raise ValueError('fig and all supplied axes must belong to one figure')
        previous_axes = set(supplied_fig.axes) if supplied_fig is not None else set()
        positions = {ax: ax.get_position(original=True).frozen() for ax in supplied_axes}
        if supplied_fig is not None:
            landscape_kwargs['fig'] = supplied_fig
        fig, _ = self.landscape(i=frames[0], fig_width=fig_width, **landscape_kwargs)
        render_kwargs = {**landscape_kwargs, 'fig': fig}

        def update(idx):
            for ax in list(fig.axes):
                if ax not in previous_axes:
                    ax.remove()
            for ax in supplied_axes:
                ax.clear()
                ax.set_axes_locator(None)
                ax.set_position(positions[ax])
            self.landscape(i=idx, fig_width=fig_width, **render_kwargs)
            return fig.axes

        return save_animation(fig, update, nframes, path + '.mp4', fps=fps,
                              frames=frames, interval=interval,
                              close=supplied_fig is None,
                              extra_args=encoder_args)

    def _animate_parallel(self, path, interval, fig_width, n_workers,
                          landscape_kwargs, frames, fps=None, dpi=MOVIE_DPI,
                          encoder_args=None):
        """Parallel frame renderer for :meth:`animate_landscape`: dump ONLY
        the arrays/metadata ``landscape`` reads, of those frames alone, render
        the already-resolved
        ``frames`` to raw RGBA across spawned workers via a lightweight plotter
        shim, and stream them into ffmpeg (the same binary matplotlib's writer
        uses) in movie order while the rest render. ffmpeg writes inside the
        temporary directory and the finished movie is swapped in from beside
        ``path``, so ``path`` changes only once the movie is complete.
        ``encoder_args`` go to ffmpeg before the output name."""
        import os
        import pickle
        import shutil
        import subprocess
        import tempfile
        import uuid
        from collections import deque
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        import matplotlib
        import matplotlib.animation as anm
        import tqdm

        m = self.model
        anim_arrays, anim_meta, anim_optional = _anim_fields(landscape_kwargs)
        tmpdir = tempfile.mkdtemp(prefix='siim_animate_')
        movie = os.path.join(tmpdir, 'movie.mp4')
        # the finished movie, beside its target (a name of this call's own)
        part = f'{path}.{uuid.uuid4().hex[:8]}.part.mp4'
        log_path = os.path.join(tmpdir, 'ffmpeg.log')
        encoder = None

        def ffmpeg_failed():
            try:                            # let ffmpeg finish its diagnostic
                encoder.wait(timeout=10)
            # gate: the log tail is reported either way; the finally kills a hung ffmpeg
            except subprocess.TimeoutExpired:
                pass
            with open(log_path, 'rb') as f:
                tail = f.read()[-2000:].decode(errors='replace')
            return RuntimeError(f"ffmpeg encoding failed:\n{tail}")

        try:
            # The selected frames alone, each once (a repeat or a reversed
            # order reads the same rows): landscape reads row i alone of every
            # per-frame array, and the scales across frames are already frozen
            # (_frozen_scale_kwargs), so the workers need nothing else.
            kept, rows = np.unique(frames, return_inverse=True)
            # a contiguous run (the whole movie, a window) as a slice: a view,
            # where an index array would copy each array before its save
            take = (slice(kept[0], kept[-1] + 1)
                    if kept[-1] - kept[0] + 1 == len(kept) else kept)
            for name in _anim_shipped(m, anim_arrays + anim_optional):
                np.save(os.path.join(tmpdir, name + '.npy'),
                        np.asarray(getattr(m, name))[take])
            meta = {name: getattr(m, name) for name in anim_meta}
            for name in _ANIM_FRAME_META:
                meta[name] = np.asarray(meta[name])[kept]
            meta['boundary_status'] = list(
                getattr(m, 'boundary_status', ['fixed_value'] * 4))
            with open(os.path.join(tmpdir, 'meta.pkl'), 'wb') as f:
                pickle.dump(meta, f)
            fps = 1000.0 / interval if fps is None else fps
            codec = matplotlib.rcParams['animation.codec']
            bitrate = matplotlib.rcParams['animation.bitrate']
            # the user's own args first, as on the serial path
            user_args = [*matplotlib.rcParams['animation.ffmpeg_args'],
                         *(encoder_args or [])]
            # Frames are keyed by POSITION in the selection and fed in that
            # order, so a `frames` subset encodes exactly those frames; each
            # job carries its saved frame and that frame's bundle row.
            jobs = [(idx, int(row), os.path.join(tmpdir, f'frame_{pos:05d}.rgba'), dpi)
                    for pos, (idx, row) in enumerate(zip(frames, rows))]
            size = None
            rc = {key: value for key, value in matplotlib.rcParams.items()
                  if key != 'backend'}
            # The pool starts its workers as frames are submitted, so the
            # script stays hidden for the pool's life. A keyword or rcParams
            # object of a class the script defines (its pickle names __main__)
            # needs the script in the workers: a guarded script then works as
            # it did, an unguarded one must add its guard.
            hide = (_launching_script_hidden
                    if b'__main__' not in pickle.dumps((landscape_kwargs, rc))
                    else contextlib.nullcontext)
            with open(log_path, 'wb') as log, hide(), ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=get_context('spawn'),
                    initializer=_animate_worker_init,
                    initargs=(tmpdir, fig_width, landscape_kwargs, rc)) as pool:
                # In movie order, at most `window` frames in flight: that
                # bounds the raw files on disk, and a failure stops the render
                # within a window rather than after the whole movie.
                window = 2 * n_workers
                pending = deque((job, pool.submit(_animate_render_frame, job))
                                for job in jobs[:window])
                try:
                    for k in tqdm.trange(len(jobs),
                                         desc=f'Rendering frames ({n_workers} workers)'):
                        (idx, _, raw, _), fut = pending.popleft()
                        if k + window < len(jobs):
                            nxt = jobs[k + window]
                            pending.append((nxt, pool.submit(_animate_render_frame, nxt)))
                        width, height = fut.result()        # surface worker exceptions
                        if encoder is None:
                            size = (width, height)
                            encoder = subprocess.Popen(
                                [anm.FFMpegWriter.bin_path(), '-y', '-loglevel', 'error',
                                 '-f', 'rawvideo', '-pix_fmt', 'rgba',
                                 '-s', f'{width}x{height}', '-framerate', f'{fps:g}',
                                 '-i', 'pipe:',
                                 # the codec matplotlib's writer names
                                 '-vcodec', codec,
                                 # libx264 needs even dimensions; crop at most 1 px
                                 '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2',
                                 # as matplotlib's writer: yuv420p for h264
                                 # alone, unless the user names a pixel format
                                 *(['-pix_fmt', 'yuv420p'] if codec == 'h264'
                                   and '-pix_fmt' not in user_args else []),
                                 *(['-b', f'{bitrate:d}k'] if bitrate > 0 else []),
                                 *user_args, movie],
                                stdin=subprocess.PIPE, stderr=log)
                        elif (width, height) != size:
                            raise RuntimeError(
                                f'frame {idx} renders at {width} x {height} px, '
                                f'the movie at {size[0]} x {size[1]}')
                        with open(raw, 'rb') as f:
                            try:
                                encoder.stdin.write(f.read())
                            except BrokenPipeError:
                                raise ffmpeg_failed() from None
                        os.remove(raw)
                except BaseException:
                    # A failed frame or encode stops the movie: the queued
                    # frames skip their render (the pool has already handed
                    # them to its workers' call queue, beyond the reach of
                    # cancel_futures); the frames being rendered finish.
                    open(os.path.join(tmpdir, 'cancelled'), 'wb').close()
                    raise
            encoder.communicate()       # closes stdin (a broken pipe included), waits
            if encoder.returncode != 0:
                raise ffmpeg_failed()
            # Out of the temporary directory first: across filesystems
            # shutil.move copies, and a copy cut short over ``path`` would
            # leave half a movie there. Within one, os.replace is atomic.
            shutil.move(movie, part)
            os.replace(part, f'{path}.mp4')
        finally:
            if encoder is not None and encoder.poll() is None:
                encoder.kill()
                encoder.wait()
            shutil.rmtree(tmpdir, ignore_errors=True)
            if os.path.exists(part):        # a move cut short
                os.remove(part)
        return f"{path}.mp4"
