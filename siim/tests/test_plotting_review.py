"""Regressions found by the independent plotting review."""

from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.ticker import FuncFormatter
import numpy as np
import pytest

from siim.siim1d import siim as Siim1D
from siim.siim2d import siim as Siim2D
from siim.plotting.landscape import LandscapeMixin


@pytest.fixture(scope='module')
def models():
    one = Siim1D(dict(U=1e-3, zELA=600, beta=1e-2, L=3e4, dx=500,
                     xo=1e3, T=2e4, nt=21, left_bc='reflecting',
                     right_bc='base_level', progress_bar=False))
    two = Siim2D(dict(U=1e-3, zELA=300, T=1e5, nt=11, nt_out=4,
                     nx=31, ny=31, Lx=3e4, Ly=3e4, seed=7,
                     initial_max_elevation=800, progress_bar=False,
                     boundary_status=['fixed_value'] * 4))
    one.run()
    two.run()
    return {'1d': one, '2d': two}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.mark.parametrize('dimension', ['1d', '2d'])
def test_static_profile_preserves_existing_artists_and_formatter(models, dimension):
    model = models[dimension]
    fig, ax = plt.subplots()
    measured = ax.plot([0, 30], [350, 350], label='Measured elevation')[0]
    note = ax.annotate('Field measurement', xy=(10, 350))
    formatter = FuncFormatter(lambda value, _: f'{value:g} m')
    ax.yaxis.set_major_formatter(formatter)
    returned, axes = model.plot.profile(ax=ax, analytical=False)
    assert returned is fig and axes[0] is ax
    assert measured in ax.lines and note in ax.texts
    assert ax.yaxis.get_major_formatter() is formatter
    assert 'Measured elevation' in ax.get_legend_handles_labels()[1]


@pytest.mark.parametrize('dimension', ['1d', '2d'])
@pytest.mark.parametrize('mode', ['snapshot', 'viewer', 'movie'])
def test_profile_direction_and_repeated_frame_updates(models, monkeypatch, tmp_path,
                                                       dimension, mode):
    model = models[dimension]
    if dimension == '1d':
        expected_xlim = (model.x[0] / 1e3, model.x[-1] / 1e3)
        thickness = model.H_out.T
    else:
        channel = model.extract_channel(i=-1, basin_rank=0)
        expected_xlim = (channel.distance[0] / 1e3, channel.distance[-1] / 1e3)
        thickness = channel.H
    seen = []

    def check(ax, index):
        assert ax.get_xlim() == expected_xlim
        # A frame update must replace its own artists rather than accumulate.
        assert len(ax.lines) == 1 and len(ax.collections) == 1
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), thickness[index])
        seen.append(index)

    if mode == 'snapshot':
        _, axes = model.plot.profile('ice_thickness', i=1, analytical=False)
        check(axes[0], 1)
    elif mode == 'viewer':
        def slider(frame, *args, **kwargs):
            _, ax = plt.subplots()
            for index in (0, 1, 0):
                frame([ax], index)
                check(ax, index)
        monkeypatch.setattr('siim.plotting._render._profile_slider', slider)
        model.plot.view_profile('ice_thickness', analytical=False)
    else:
        def save(fig, update, nframes, path, **kwargs):
            for index in (0, 1, 0):
                update(index)
                check(fig.axes[0], index)
            return path
        monkeypatch.setattr('siim.plotting._profiles.save_animation', save)
        model.plot.animate_profile('ice_thickness', analytical=False,
                                   path=str(tmp_path / 'profile'))
    assert seen == ([1] if mode == 'snapshot' else [0, 1, 0])


def _render_ice_mask(H, receivers, cell, **options):
    """Render real ice geometry against a grayscale bed with an explicit ice color."""
    ny, nx = H.shape
    model = SimpleNamespace(
        H_out=H[None], zb_out=np.zeros((1, ny, nx)), z_out=(1.5 * H)[None],
        area_out=np.full((1, ny, nx), 1e8), receivers_out=receivers[None],
        Lx=(nx - 1) * cell, Ly=(ny - 1) * cell, grid_nx=nx, grid_ny=ny,
        alpha_g=8., hc_over_H=1.5, output_times=np.array([0.]),
        boundary_status=['fixed_value'] * 4, _zELA_output=np.array([100.]),
    )
    settings = dict(style='raw', ice_extent='cells', area_threshold=0,
                    oversample=1, ice_sigma_cells=0, ice_shading='flat',
                    ice_color='#ff00ff', cmap_bed='Greys', show_margin=False,
                    hillshade=False, contour_interval=0, sigma_cells=0,
                    colorbar=False, z_min=0, z_max=200)
    settings.update(options)
    fig, ax = LandscapeMixin.landscape(SimpleNamespace(model=model), **settings)
    rgb = np.asarray(ax.images[0].get_array())
    mask = np.all(np.isclose(rgb, to_rgb(settings['ice_color'])), axis=-1)
    plt.close(fig)
    return mask


@pytest.mark.parametrize('extent', ['cells', 'footprint'])
@pytest.mark.parametrize('minimum', [7, 9, 10])
def test_component_filter_counts_the_connected_veil_and_ribbon_union(extent, minimum):
    H = np.zeros((15, 15))
    H[7, 2:5] = 5.   # Three narrow veil cells attached to six resolved trunk cells.
    H[7, 5:11] = 20.
    receivers = np.arange(H.size).reshape(H.shape)
    receivers[7, 2:10] += 1
    actual = _render_ice_mask(H, receivers, 100., ice_extent=extent,
                              trunk_display='ribbons', min_ice_cells=minimum)
    expected = H > 0 if minimum <= 9 else np.zeros_like(H, dtype=bool)
    np.testing.assert_array_equal(actual, expected)


def test_field_smoothing_keeps_threshold_fringe_when_ribbons_are_enabled():
    receivers = np.arange(24 * 24).reshape(24, 24)
    settings = dict(H_threshold=50., ice_sigma_cells=2., oversample=2)
    fields, masks = [], []
    for fringe in (47., 53.):
        H = np.zeros((24, 24))
        H[:, :10] = 100.
        H[:, 10:12] = fringe
        # At 1 km cell spacing none of these widths qualifies as a trunk.
        plain = _render_ice_mask(H, receivers, 1000., trunk_display='none',
                                 ice_smoothing='field', **settings)
        ribbon = _render_ice_mask(H, receivers, 1000., trunk_display='ribbons',
                                  ice_smoothing='field', **settings)
        np.testing.assert_array_equal(ribbon, plain)
        fields.append(ribbon)
        masks.append(_render_ice_mask(H, receivers, 1000., trunk_display='ribbons',
                                      ice_smoothing='mask', **settings))
    changed_field = np.logical_xor(*fields).sum()
    changed_mask = np.logical_xor(*masks).sum()
    assert 0 < changed_field < changed_mask


# --- the 2026-11-05 plotting audit: known-input landscape renders -----------

def _known(zb, H=None, cell=1000.0, boundary_status=None):
    """A plotter over known (t, y, x) beds — row = y, north up — with no run:
    exactly what landscape() reads. No ice and self-receivers by default."""
    from siim.plotting import siim_plotter
    zb = np.asarray(zb, dtype=float)
    zb = zb[None] if zb.ndim == 2 else zb
    nt, ny, nx = zb.shape
    H = (np.zeros_like(zb) if H is None
         else np.broadcast_to(np.asarray(H, dtype=float), zb.shape).copy())
    receivers = np.arange(ny * nx, dtype=float).reshape(ny, nx)
    return siim_plotter(SimpleNamespace(
        zb_out=zb, H_out=H, z_out=zb + 1.5 * H,
        area_out=np.full(zb.shape, 1e8),
        receivers_out=np.broadcast_to(receivers, zb.shape).copy(),
        output_times=np.arange(nt) * 1e4, _zELA_output=np.full(nt, 1e5),
        Lx=(nx - 1) * cell, Ly=(ny - 1) * cell, hc_over_H=1.5, alpha_g=8.0,
        grid_nx=nx, grid_ny=ny,
        boundary_status=boundary_status or ['fixed_value'] * 4))


def _hyp_axes(fig):
    return [a for a in fig.axes if a.get_xlabel() == r'$\%$ area'][0]


def _section_axes(fig):
    return [a for a in fig.axes if a.get_ylabel() == 'Elevation (m)'][0]


def _visible_bars(ax_hyp):
    """Widths of the hypsometry bars that lie wholly inside the panel."""
    x1, (y0, y1) = ax_hyp.get_xlim()[1], ax_hyp.get_ylim()
    return np.array([b.get_width() for b in ax_hyp.patches
                     if b.get_y() >= y0 - 1e-6
                     and b.get_y() + b.get_height() <= y1 + 1e-6
                     and b.get_width() <= x1])


def _ramp(lo, hi, n=41):
    return np.tile(np.linspace(lo, hi, n), (n, 1))


def _plateau(n=41):
    x = np.linspace(0, 40e3, n)
    return np.tile(np.where(x < 16e3, 200.0 + 40.0 * x / 1e3,
                            1000.0 + 50.0 * (x - 16e3) / 24e3), (n, 1))


def test_hypsometry_shares_the_section_axis_over_the_whole_area():
    """The hypsometry shares the section's elevation axis, from the -500 m
    default z_min, so bed beyond it is off the panel; its bars are
    percentages of the WHOLE area, so the bars shown sum to the share of the
    bed inside the axis, not to 100 % (Eric, 2026-11-05)."""
    plot = _known(_ramp(-1000.0, 1600.0))
    fig, _ = plot.landscape(field='bedrock', cross_section=20.0, hillshade=False)
    ax_hyp, ax_cs = _hyp_axes(fig), _section_axes(fig)
    assert ax_hyp.get_shared_y_axes().joined(ax_hyp, ax_cs)
    z_lo, z_hi = ax_cs.get_ylim()
    assert z_lo == -500.0
    zb = np.asarray(plot.model.zb_out[-1])
    inside = 100.0 * ((zb >= z_lo) & (zb <= z_hi)).mean()
    assert inside < 90.0, "test setup: the bed must reach past the axis"
    assert _visible_bars(ax_hyp).sum() == pytest.approx(inside, abs=2.5)


def test_ice_opaque_depth_is_validated():
    """A depth that is not finite and positive raises instead of dividing the
    veil's opacity by zero or a negative depth."""
    plot = _known(_ramp(0.0, 1000.0))
    for bad in (0.0, -300.0, np.nan):
        with pytest.raises(ValueError, match='ice_opaque_depth'):
            plot.landscape(field='bedrock', ice_opaque_depth=bad)
    for bad in (0.0, -2.0, np.nan):
        with pytest.raises(ValueError, match='ice_alpha_sharpness'):
            plot.landscape(field='bedrock', ice_alpha_sharpness=bad)


def test_shade_stretch_is_validated_and_the_frozen_z_max_skips_nan_frames():
    """A reversed shade_stretch raises instead of silently falling back to the
    per-render stretch; the movie's frozen z_max skips an all-NaN frame."""
    from siim.plotting.landscape import _frozen_scale_kwargs
    plot = _known(_ramp(0.0, 1000.0))
    with pytest.raises(ValueError, match='shade_stretch'):
        plot.landscape(field='bedrock', shade_stretch=(0.9, 0.2))
    beds = np.stack([np.full((41, 41), np.nan), _ramp(0.0, 1000.0)])
    kwargs = _frozen_scale_kwargs(_known(beds).model,
                                  dict(field='bedrock', hillshade=False), [0, 1])
    assert kwargs['z_max'] == 1000.0


def test_frozen_hillshade_stretch_ignores_relief_elsewhere():
    """The soft hillshade stretches its illumination to each render's own
    range, so a steep feature appearing far away re-lit an unchanged hill
    (-69 % contrast). animate_landscape freezes the stretch over the movie
    (audit C10); a still keeps its own."""
    from siim.plotting.landscape import _frozen_scale_kwargs
    n = 101
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    hill = 300.0 * np.exp(-((xx - 30) ** 2 + (yy - 50) ** 2) / 200.0)
    cone = 3000.0 * np.clip(1 - np.hypot(xx - 85, yy - 50) / 6, 0, None)
    plot = _known(np.stack([hill, hill + cone]), cell=100.0)
    settings = dict(field='bedrock', oversample=1, sigma_cells=0,
                    contour_interval=0, hillshade=True, colorbar=False)
    kwargs = _frozen_scale_kwargs(plot.model, settings, [0, 1])
    hill_px = (slice(35, 66), slice(15, 46))

    def hill_rgb(i, **kw):
        _, ax = plot.landscape(i=i, **kw)
        return np.asarray(ax.get_images()[0].get_array(), dtype=float)[hill_px]

    np.testing.assert_allclose(hill_rgb(0, **kwargs), hill_rgb(1, **kwargs),
                               atol=1e-12)
    still = {**settings, 'z_max': kwargs['z_max']}
    assert not np.allclose(hill_rgb(0, **still), hill_rgb(1, **still)), \
        "test setup: the far cone does not move the per-render stretch"


def _lake_px(ax):
    from siim.plotting._style import COLORS
    # gate: lake_color's default is now siim's one standing-water colour
    lake = np.asarray(to_rgb(COLORS['water']))       # lake_color default
    rgb = np.asarray(ax.get_images()[0].get_array(), dtype=float)
    return np.abs(rgb - lake).max(axis=-1) < 1e-9


def test_lakes_are_drawn_on_ice_free_ground_only():
    """'bedrock+lakes' flooded the bare bed whatever the ice, so on a
    glaciated run nearly every lake pixel lay under ice (audit C12). A basin
    under drawn ice is not a lake; an ice-free one still is."""
    zb = np.full((31, 31), 500.0)
    zb[5:9, 5:9] = 200.0                       # ice-free basin
    zb[20:24, 20:24] = 200.0                   # the same basin, under ice
    H = np.zeros_like(zb)
    H[20:24, 20:24] = 100.0
    _, ax = _known(zb, H).landscape(field='bedrock+lakes', style='raw',
                                    oversample=1)
    wet = _lake_px(ax)
    assert wet[5:9, 5:9].all()
    assert not wet[20:24, 20:24].any()


def test_lake_min_area_counts_a_lake_across_a_looped_seam():
    """lake_min_area labelled the lakes without the seam, so a 30 km² lake
    split by a looped x edge counted as two 15 km² halves and was dropped at
    20 km² (audit C15)."""
    zb = np.full((21, 21), 500.0)
    zb[8:13, :3] = zb[8:13, -3:] = 200.0       # 5 x 6 cells across the x seam
    plot = _known(zb, boundary_status=['looped', 'looped',
                                       'fixed_value', 'fixed_value'])
    _, ax = plot.landscape(field='bedrock+lakes', style='raw', oversample=1,
                           lake_min_area=20e6)
    assert _lake_px(ax).sum() == 30


def test_section_images_sit_on_the_nodes():
    """The section rasters spanned [0, Lx] with one column per node, so their
    pixel centres sat up to half a pixel off the nodes the section lines are
    drawn on: a bed step at node 2 (x = 2 km) drew 0.44 km late (audit C14)."""
    x = np.arange(41) * 1000.0
    step = np.tile(np.where(x >= 2e3, 1000.0, 0.0), (41, 1))
    for ov in (1, 4):
        fig, ax = _known(step).landscape(field='bedrock', style='raw',
                                         oversample=ov, cross_section=15.0,
                                         z_min=-100.0, z_max=1100.0)
        bed = [im for im in _section_axes(fig).get_images()
               if im.get_zorder() == 1][0]
        alpha = np.asarray(bed.get_array())[..., 3]
        x0, x1, z0, z1 = bed.get_extent()
        nz, ncol = alpha.shape
        centres = x0 + (np.arange(ncol) + 0.5) * (x1 - x0) / ncol
        np.testing.assert_allclose(centres, np.linspace(0, 40, ncol), atol=1e-9)
        np.testing.assert_allclose(z0 + (np.arange(nz) + 0.5) * (z1 - z0) / nz,
                                   np.linspace(-100.0, 1100.0, nz), atol=1e-9)
        if ov == 1:
            assert centres[np.argmax(alpha[nz // 2] > 0)] == pytest.approx(2.0)
        plt.close(fig)


def test_cross_section_outside_the_domain_warns():
    """A y outside [0, Ly] still draws the edge row, but no longer silently
    (audit C16)."""
    import warnings
    plot = _known(np.full((11, 11), 500.0))
    with pytest.warns(UserWarning, match=r'y = 50 km .*\[0, 10\] km.*'
                                         r'row 40 \(y = 10 km\)'):
        plot.landscape(field='bedrock', cross_section=50.0, hillshade=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        plot.landscape(field='bedrock', cross_section=5.0, hillshade=False)
    assert not [w for w in caught if 'cross_section' in str(w.message)]


def test_ice_time_avg_is_refused_with_its_reason(tmp_path, monkeypatch):
    """ice_time_avg averaged trailing OUTPUT frames (a double exposure of two
    glacial epochs at coarse saves) and was removed (Eric, 2026-11-05). Any
    value, 1 included, gets an error that says so, not the signature's bare
    unexpected-keyword TypeError; the movie refuses it before rendering."""
    plot = _known(_ramp(0.0, 1000.0))
    monkeypatch.chdir(tmp_path)
    for value in (1, 2, None):
        with pytest.raises(ValueError, match=r'ice_time_avg was removed in '
                                             r'0\.10\.0: it averaged'):
            plot.landscape(field='bedrock', ice_time_avg=value)
        with pytest.raises(ValueError, match='ice_time_avg was removed'):
            plot.animate_landscape(path='refused', n_workers=1,
                                   ice_time_avg=value)
    assert not plt.get_fignums(), "the refusal must come before any figure"


def _bed_bar_extend(fig):
    """The ``extend`` of a landscape figure's bed (elevation) colorbar."""
    bars = [a._colorbar for a in fig.axes
            if getattr(a, '_colorbar', None) is not None
            and a.get_ylabel().endswith('elevation (m)')]
    assert len(bars) == 1, "test setup: one bed colorbar expected"
    return bars[0].extend


def test_bed_colorbar_arrows_where_the_drawn_bed_saturates():
    """The bed colours saturate at z_min / z_max by design; the bar now
    says so with an arrow at each end the drawn surface goes past (Eric,
    2026-11-05), and keeps no arrow for a bed inside the range."""
    deep, inside = _known(_ramp(-1000.0, 1600.0)), _known(_ramp(0.0, 1000.0))
    for style in ('smooth', 'raw'):
        cases = ((deep, {}, 'min'), (inside, {}, 'neither'),
                 (inside, dict(z_max=500.0), 'max'),
                 (deep, dict(z_max=500.0), 'both'),
                 (deep, dict(extend_bed='neither'), 'neither'))   # explicit wins
        for plot, kw, want in cases:
            fig, _ = plot.landscape(field='bedrock', style=style, **kw)
            assert _bed_bar_extend(fig) == want, (style, kw)
            plt.close(fig)
    with pytest.raises(ValueError, match='extend_bed'):
        inside.landscape(field='bedrock', extend_bed='up')


def test_movie_frames_share_one_bed_extend(tmp_path, monkeypatch):
    """A movie decides the bed bar's arrows ONCE over the encoded frames, so
    the bar keeps its shape: a frame whose own bed stays inside the range
    still carries the arrow another frame needs."""
    import siim.plotting.landscape as landscape_module
    from siim.plotting._animation import frame_indices
    from siim.plotting.landscape import _frozen_scale_kwargs
    plot = _known(np.stack([_ramp(0.0, 1000.0), _ramp(-1000.0, 1000.0)]))
    settings = dict(field='bedrock', hillshade=False)
    stills = []
    for i in (0, 1):
        fig, _ = plot.landscape(i=i, **settings)
        stills.append(_bed_bar_extend(fig))
        plt.close(fig)
    assert stills == ['neither', 'min'], \
        "test setup: only the second frame reaches below z_min"
    assert _frozen_scale_kwargs(plot.model, settings)['extend_bed'] == 'min'
    # ...and the encoded frames carry it: render every frame in place
    seen = []

    def render_frames(fig, update, nframes, path, *, frames=None, close=True,
                      **_):
        for idx in frame_indices(nframes, frames):
            update(idx)
            seen.append(_bed_bar_extend(fig))
        if close:
            plt.close(fig)
        return path

    monkeypatch.setattr(landscape_module, 'save_animation', render_frames)
    monkeypatch.chdir(tmp_path)
    plot.animate_landscape(path='extend', n_workers=1, **settings)
    assert seen == ['min', 'min']


# --- the 2026-11-05 plotting audit: analytical references -------------------

def test_1d_analytical_is_oriented_as_profile(models):
    """The 1D analytical() set xlim (0, L) against profile()'s
    (x[0], x[-1]), so the reference was the profile's mirror image
    (audit C5)."""
    model = models['1d']
    _, axes = model.plot.profile(analytical=False)
    ax = model.plot.analytical()
    assert ax is not None, "test setup: no analytical reference"
    assert ax.get_xlim() == axes[0].get_xlim()
    assert ax.get_xlim() == (model.analytical.x[0] / 1e3,
                             model.analytical.x[-1] / 1e3)


def test_2d_analytical_draws_the_requested_basin(models):
    """Every extract_channel refits m.analytical to its own basin, so after
    largest_basins() the 2D analytical() drew the LAST basin's reference
    (audit C17). It reads basin_rank/ref, as profile() does."""
    model = models['2d']
    model.plot.largest_basins(3, plot=False)
    last = model.plot._analytical_overlay(True).surface    # basin 2's
    ax = model.plot.analytical()
    drawn = np.asarray(ax.lines[0].get_ydata())
    model.extract_channel(i=-1, basin_rank=0)
    wanted = model.plot._analytical_overlay(True).surface
    # gate: new test (not in HEAD); its setup check states the precondition
    assert wanted.shape != last.shape or not np.allclose(wanted, last), \
        "test setup: basins 0 and 2 share one reference"
    np.testing.assert_allclose(drawn, wanted)


# --- the 2026-11-05 consistency cleanup --------------------------------------

@pytest.mark.pin
def test_model_imports_pull_in_no_pyplot():
    """siim1d imported pyplot at module level; the plots import it when they
    draw, so importing either model stays headless."""
    import subprocess
    import sys
    code = ("import sys, siim.siim1d, siim.siim2d; "
            "assert 'matplotlib.pyplot' not in sys.modules, 'pyplot imported'")
    subprocess.run([sys.executable, '-c', code], check=True)


def test_analytical_methods_share_one_signature():
    """One analytical() call shape in 1D and 2D; the 1D deprecated keywords
    are keyword-only, after it."""
    import inspect
    from siim.plotting import siim_plotter as Plot2D
    from siim.siim1d import siim_plotter as Plot1D
    shapes = []
    for cls in (Plot1D, Plot2D):
        params = inspect.signature(cls.analytical).parameters.values()
        shapes.append([(p.name, p.default) for p in params
                       if p.kind is p.POSITIONAL_OR_KEYWORD])
    assert shapes[0] == shapes[1] == [
        ('self', inspect.Parameter.empty), ('ax', None), ('bistable', True),
        ('basin_rank', 0), ('ref', -1)]


@pytest.mark.parametrize('case', ['plot_analytical', 'alpha_ice', 'alpha_zb',
                                  'figsize', 'trimline', 'i_ref'])
def test_deprecated_plot_keywords_name_their_removal(models, monkeypatch, case):
    """Each deprecated plotting keyword warns, names its removal in 1.0, and
    keeps working until then."""
    import siim.plotting.profiles as profiles_module
    one, two = models['1d'], models['2d']
    seen = {}
    monkeypatch.setattr(profiles_module, '_slider_view',
                        lambda make_draw, sliders, *, figsize: seen.update(figsize=figsize))
    calls = {
        'plot_analytical': lambda: one.plot.analytical(plot_analytical=False),
        'alpha_ice': lambda: one.plot.analytical(alpha_ice=0.3),
        'alpha_zb': lambda: one.plot.analytical(alpha_zb=0.3),
        'figsize': lambda: two.plot.view_xsection(figsize=(4, 2)),
        'trimline': lambda: two.plot.landscape(style='raw', show_trimline=True),
        'i_ref': lambda: two.plot.largest_basins_history(n_basins=2, i_ref=-1, plot=False),
    }
    with pytest.warns(DeprecationWarning, match='removed in 1.0'):
        result = calls[case]()
    if case == 'plot_analytical':
        assert result is None                     # False still draws nothing
    elif case.startswith('alpha'):
        assert result is not None                 # ignored, the plot still drawn
    elif case == 'figsize':
        assert seen['figsize'] == (4, 2)          # still sizes the viewer


def test_old_positional_1d_analytical_calls_warn_and_still_work(models):
    """The 1D analytical() used to take (ax, alpha_ice, alpha_zb,
    plot_analytical, bistable): such a positional call is read that way, with
    a DeprecationWarning, instead of binding silently to the new order."""
    one = models['1d']
    with pytest.warns(DeprecationWarning, match='positional arguments'):
        assert one.plot.analytical(None, 0.15, 0.2, False) is None    # plot_analytical=False
    with pytest.warns(DeprecationWarning, match='positional arguments'):
        ax = one.plot.analytical(None, 0.15, 0.2, True, False)          # bistable=False
    assert ax is not None
    plt.close('all')


def test_flux_and_basin_maps_are_drawn_in_km(models):
    """flux() and the basin maps were laid out in metres under km tick labels,
    so a km overlay landed in the corner; they share map()'s km raster now."""
    from siim.plotting._render import _node_extent
    model = models['2d']
    extent = _node_extent(model.Lx, model.Ly, model.grid_nx, model.grid_ny)
    ax = model.plot.flux('water')
    np.testing.assert_allclose(ax.get_images()[0].get_extent(), extent)
    assert ax.get_xlabel() == 'x (km)' and ax.get_ylabel() == 'y (km)'
    assert ax.get_xlim() == (0, model.Lx / 1e3) and ax.get_ylim() == (0, model.Ly / 1e3)
    data = model.plot.largest_basins(2)
    ax_map = plt.gcf().axes[0]
    np.testing.assert_allclose(ax_map.get_images()[0].get_extent(), extent)
    outlet = np.asarray(ax_map.lines[0].get_xydata())[-1]
    np.testing.assert_allclose(outlet, [data.channels[0].x_coord[-1] / 1e3,
                                        data.channels[0].y_coord[-1] / 1e3])


def test_one_colour_label_and_width_per_role(models):
    """The ELA, the bed, the distance axis and the figure widths follow one
    style: 6 in for a single panel, 12 in for a stack and for landscape."""
    from matplotlib.colors import to_hex
    from siim.plotting._style import COLORS
    one, two = models['1d'], models['2d']
    fig, axes = two.plot.profile(analytical=False)
    assert fig.get_figwidth() == 6
    assert axes[0].get_xlabel() == 'Distance from divide (km)'
    beds = [c for c in axes[0].collections
            if to_hex(c.get_facecolor()[0]) == COLORS['bed']]
    assert len(beds) == 1                         # one translucent bed fill
    fig, axes = one.plot.profile(['elevation', 'ice_thickness'], analytical=False)
    assert fig.get_figwidth() == 12
    assert axes[-1].get_xlabel() == 'x (km)'      # the 1D grid coordinate
    for model in (one, two):
        ax = model.plot.analytical()
        assert ax.figure.get_figwidth() == 6
        ela = [ln for ln in ax.lines if ln.get_label() == 'Analytical ELA']
        assert len(ela) == 1 and to_hex(ela[0].get_color()) == COLORS['ela']
    fig, _ = two.plot.landscape(style='raw')
    assert fig.get_figwidth() == 12
