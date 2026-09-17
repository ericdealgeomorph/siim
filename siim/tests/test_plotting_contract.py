"""Public plotting regressions from the correctness and structure audit."""

from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from siim.siim1d import siim as Siim1D
from siim.siim2d import siim as Siim2D
from siim.plotting.diagnostics import DiagnosticsMixin
from siim.plotting._animation import save_animation
from siim.plotting._profiles import field_limits


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


def line(ax, label):
    return next(artist for artist in ax.lines if artist.get_label() == label)


@pytest.mark.parametrize('dimension', ['1d', '2d'])
def test_common_field_inputs_and_display_units(models, dimension):
    model = models[dimension]
    fig, axes = model.plot.profile({'Elevation': None, 'Ice_Thickness': (0, 500)}, analytical=False)
    assert axes[0].get_ylabel() == 'Elevation (m)'
    assert axes[1].get_ylabel() == 'Mean ice thickness (m)'
    assert axes[1].get_ylim() == (0, 500)
    assert 't =' in axes[0].get_title(loc='right')
    assert not axes[0].get_legend().get_frame_on()
    fig.canvas.draw()
    with pytest.raises(ValueError, match='at least one'):
        model.plot.profile([])
    with pytest.raises(ValueError, match='single field'):
        model.plot.profile(['elevation', 'ice_flux'], field_min=0)
    with pytest.raises(ValueError, match='positive'):
        model.plot.profile('erosion_rate', field_min=0, analytical=False)


def test_shear_axis_and_cycle_use_kpa(models):
    model = models['1d']
    idx = int(np.unravel_index(np.argmax(model.tau_out), model.tau_out.shape)[1])
    _, axes = model.plot.profile('shear_stress', i=idx, analytical=False)
    peak = float(model.tau_out.max() / 1000)
    assert axes[0].get_ylim()[1] == pytest.approx(max(peak, model.tau_c / 1000) * 1.05)
    assert max(line(axes[0], 'Basal shear stress').get_ydata()) == pytest.approx(peak)
    assert model.plot._cycle_field_ylim('shear_stress', [idx])[1] == pytest.approx(axes[0].get_ylim()[1])


def test_profiles_include_negative_bed_and_transients(models, monkeypatch):
    one = models['1d']
    bed = one.zb_out.copy()
    bed[0, -1] = -2000
    monkeypatch.setattr(one, 'zb_out', bed)
    _, axes = one.plot.profile(analytical=False)
    assert axes[0].get_ylim()[0] < -2000
    two = models['2d']
    channel = two.plot._get_channel(-1, 0)
    flux = np.ones_like(channel.Qg)
    flux[1, 2] = 100
    monkeypatch.setattr(channel, 'Qg', flux)
    _, axes = two.plot.profile('ice_flux', i=1, analytical=False)
    assert axes[0].get_ylim()[1] >= 100
    assert line(axes[0], 'Ice flux').get_ydata()[2] == 100


@pytest.mark.parametrize('dimension', ['1d', '2d'])
def test_erosion_limits_include_small_rates_and_zero_frames(models, monkeypatch, dimension):
    model = models[dimension]
    source = model if dimension == '1d' else model.plot._get_channel(-1, 0)
    attr = 'erosion_rate_out' if dimension == '1d' else 'erosion_rate'
    rates = np.ones_like(getattr(source, attr)) * 1e-6
    rates.flat[0] = 1.0
    monkeypatch.setattr(source, attr, rates)
    _, axes = model.plot.profile('erosion_rate', analytical=False)
    lo, hi = axes[0].get_ylim()
    assert lo < 1e-6 < 1e-3 < 1 < hi
    monkeypatch.setattr(source, attr, np.zeros_like(rates))
    fig, axes = model.plot.profile('erosion_rate', analytical=False)
    fig.canvas.draw()
    assert any('No positive erosion' in text.get_text() for text in axes[0].texts)
    assert axes[0].get_ylim()[0] > 0
    assert field_limits('erosion_rate', [0, np.nan, np.inf]) == (1e-12, 1.0)


def test_cached_channel_restores_its_analytical_reference(models):
    model = models['2d']
    _, axes = model.plot.profile(basin_rank=0)
    before = line(axes[0], 'Analytical ice surface').get_ydata().copy()
    geometry = model.L, model.xo, model.k_h, model.d
    cached = model._profile_channel
    model.extract_channel(basin_rank=1)
    _, axes = model.plot.profile(basin_rank=0)
    np.testing.assert_array_equal(line(axes[0], 'Analytical ice surface').get_ydata(), before)
    assert (model.L, model.xo, model.k_h, model.d) == geometry
    assert model._profile_channel is cached


@pytest.mark.parametrize('shape', ['time', 'space', 'time-space'])
def test_uplift_reference_uses_selected_frame_and_channel(models, monkeypatch, shape):
    model = models['2d']
    space = np.linspace(1e-3, 2e-3, model.grid_nx * model.grid_ny).reshape(model.grid_ny, model.grid_nx)
    time = np.linspace(1, 3, model.nt)
    forcing = {'time': time * 1e-3, 'space': space,
               'time-space': time[:, None, None] * space}[shape]
    monkeypatch.setattr(model, '_U_user', forcing)
    channel = model.plot._get_channel(-1, 0)
    _, axes = model.plot.profile('erosion_rate', i=-1, analytical=False)
    expected = np.full(len(channel.nodes), 3e-3) if shape == 'time' else (
        space.ravel()[channel.nodes] * (3 if shape == 'time-space' else 1))
    np.testing.assert_allclose(line(axes[0], 'Uplift rate').get_ydata(), expected)
    assert axes[0].get_ylim()[1] >= max(expected)


@pytest.mark.parametrize('frames', [1, 2, 4])
def test_short_steady_state_sequences_and_temporal_forcing(frames):
    model = SimpleNamespace(
        boundary_status=['fixed_value'] * 4, grid_nx=3, grid_ny=3, nt=5,
        output_times=np.linspace(0, 100, frames),
        z_out=np.ones((frames, 3, 3)), erosion_rate_out=np.full((frames, 3, 3), 1e-3),
        _zELA_output=np.full(frames, 300),
        _make_uplift_field=lambda: (('tstep',), np.full(5, 1e-3)),
    )
    result = DiagnosticsMixin.steady_state(SimpleNamespace(model=model))
    assert result == (-1 if frames == 1 else 0)


@pytest.mark.parametrize('gate', ['H_threshold', 'area_threshold', 'min_ice_cells'])
def test_display_gates_cover_trunks(models, gate):
    model = models['2d']
    options = dict(colorbar=False, hillshade=False, contour_interval=0,
                   show_margin=False, oversample=2)
    value = {'H_threshold': float(model.H_out.max() + 1),
             'area_threshold': float(model.area_out.max() + 1),
             'min_ice_cells': model.grid_nx * model.grid_ny * 2}[gate]
    _, actual = model.plot.landscape(**options, **{gate: value})
    _, bare = model.plot.landscape(field='bedrock', **options)
    np.testing.assert_array_equal(actual.images[0].get_array(), bare.images[0].get_array())


def test_map_pixel_centers_match_model_nodes(models):
    model = models['2d']
    ax = model.plot.map()
    left, right, bottom, top = ax.images[0].get_extent()
    centers = left + (np.arange(model.grid_nx) + 0.5) * (right - left) / model.grid_nx
    np.testing.assert_allclose(centers, np.linspace(0, model.Lx / 1000, model.grid_nx), atol=1e-12)
    assert ax.get_xlim() == (0, model.Lx / 1000)


@pytest.mark.parametrize('supplied', ['fig', 'ax', 'section'])
def test_landscape_movie_keeps_supplied_axes_and_other_content(models, monkeypatch, tmp_path, supplied):
    model = models['2d']
    fig, axes = plt.subplots(2, 2)
    sentinel = axes[1, 1].plot([0, 1], [2, 3])[0]
    kwargs = {'fig': fig}
    if supplied in ('ax', 'section'):
        kwargs['ax'] = axes[0, 0]
    if supplied == 'section':
        kwargs.update(ax_cs=axes[1, 0], ax_hyp=axes[0, 1], cross_section=15)
    counts = []

    def save(movie, *args, **options):
        movie._draw_was_started = True
        for idx in (0, 1, 3):
            movie._func(idx)
            fig.canvas.draw()
            assert sentinel in axes[1, 1].lines
            assert sum(bool(ax.images) for ax in fig.axes) >= 1
            if supplied != 'fig':
                assert axes[0, 0] in fig.axes and axes[0, 0].images
            counts.append(len(fig.axes))

    monkeypatch.setattr('matplotlib.animation.Animation.save', save)
    target = model.plot.animate_landscape(path=str(tmp_path / 'movie.mp4'),
                                           style='raw', n_workers=1, **kwargs)
    assert target.endswith('movie.mp4') and not target.endswith('.mp4.mp4')
    assert len(set(counts)) == 1
    assert plt.fignum_exists(fig.number)


def test_movie_failure_closes_owned_figures(monkeypatch, tmp_path):
    fig, _ = plt.subplots()
    def fail(movie, *args, **kwargs):
        movie._draw_was_started = True
        raise RuntimeError('encoder failed')
    monkeypatch.setattr('matplotlib.animation.Animation.save', fail)
    with pytest.raises(RuntimeError, match='encoder failed'):
        save_animation(fig, lambda _: (), 2, str(tmp_path / 'failed.mp4'))
    assert not plt.fignum_exists(fig.number)


@pytest.mark.pin
@pytest.mark.parametrize('failure', ['missing', 'unsupported', 'oserror', 'page_size', 'page_count'])
def test_landscape_movie_uses_serial_when_ram_is_unknown(models, monkeypatch, tmp_path, failure):
    import os

    model = models['2d']
    if failure == 'missing':
        monkeypatch.delattr(os, 'sysconf', raising=False)
    else:
        def sysconf(name):
            if failure == 'unsupported':
                raise ValueError('unknown configuration name')
            if failure == 'oserror':
                raise OSError('memory query failed')
            if name == 'SC_PAGE_SIZE':
                return -1 if failure == 'page_size' else 4096
            return -1 if failure == 'page_count' else 1024
        monkeypatch.setattr(os, 'sysconf', sysconf, raising=False)

    def no_parallel(*args, **kwargs):
        pytest.fail('unknown RAM must not launch parallel workers')

    frames = []
    def save(movie, *args, **kwargs):
        movie._draw_was_started = True
        for idx in range(len(model.output_times)):
            movie._func(idx)
            frames.append(idx)

    monkeypatch.setattr(model.plot, '_animate_parallel', no_parallel)
    monkeypatch.setattr('matplotlib.animation.Animation.save', save)
    result = model.plot.animate_landscape(
        path=str(tmp_path / 'serial.mp4'), n_workers=2, style='raw',
        hillshade=False, contour_interval=0)
    assert result.endswith('serial.mp4')
    assert frames == list(range(len(model.output_times)))


def test_landscape_movie_selects_frames_and_rate(models, monkeypatch, tmp_path):
    """``frames`` encodes exactly the requested saved frames (slices and
    negative indices included) and ``fps`` reaches the encoder."""
    model = models['2d']
    nframes = len(model.output_times)
    recorded = {}

    def save(movie, *args, **kwargs):
        movie._draw_was_started = True
        recorded['fps'] = kwargs.get('fps')
        recorded['frames'] = []
        for idx in movie.new_frame_seq():
            movie._func(idx)
            recorded['frames'].append(idx)

    monkeypatch.setattr('matplotlib.animation.Animation.save', save)
    style = dict(n_workers=1, style='raw', hillshade=False, contour_interval=0)
    model.plot.animate_landscape(path=str(tmp_path / 'tail.mp4'),
                                 frames=slice(1, None), **style)
    assert recorded['frames'] == list(range(1, nframes))
    model.plot.animate_landscape(path=str(tmp_path / 'ends.mp4'),
                                 frames=[0, -1], fps=5, **style)
    assert recorded['frames'] == [0, nframes - 1]
    assert recorded['fps'] == 5


@pytest.mark.parametrize('empty', [[], slice(0, 0), slice(10, 20)])
def test_movie_refuses_an_empty_frame_selection(models, monkeypatch, tmp_path, empty):
    """A selection that resolves to nothing wrote a 262-byte unplayable mp4 and
    reported success; every animator must raise before the encoder instead."""
    model = models['2d']

    def save(movie, *args, **kwargs):
        pytest.fail('an empty selection must not reach the encoder')

    monkeypatch.setattr('matplotlib.animation.Animation.save', save)
    with pytest.raises(ValueError, match='No simulation output'):
        model.plot.animate_landscape(path=str(tmp_path / 'empty.mp4'),
                                     n_workers=1, style='raw', frames=empty)
    with pytest.raises(ValueError, match='No simulation output'):
        model.plot.animate_map(path=str(tmp_path / 'empty_map.mp4'), frames=empty)


def test_slider_view_sets_canvas_traits_and_updates(monkeypatch):
    pytest.importorskip('ipympl', reason='interactive viewer requires the optional notebook backend')
    from siim.plotting._render import _slider_view

    displayed, figures, frames = [], [], []
    monkeypatch.setattr('IPython.display.display', displayed.append)
    previous_backend = matplotlib.get_backend()

    def make_draw(fig):
        figures.append(fig)
        return frames.append

    try:
        _slider_view(make_draw, [(3, 'Frame', -1)], figsize=(3, 2))
        canvas = figures[0].canvas
        assert all(getattr(canvas, name) is False for name in (
            'header_visible', 'footer_visible', 'toolbar_visible', 'resizable'))
        assert frames == [2]
        displayed[0].children[0].value = 1
        assert frames == [2, 1]
        assert matplotlib.get_backend() == previous_backend
        assert not plt.fignum_exists(figures[0].number)
    finally:
        for widget in displayed:
            for child in widget.children:
                child.close()
            widget.close()


@pytest.mark.pin
@pytest.mark.parametrize('name', ['header_visible', 'resizable'])
@pytest.mark.parametrize('backend_name', ['Agg', 'module://ipympl.backend_nbagg'])
def test_slider_view_propagates_canvas_errors_and_restores_backend(monkeypatch, name, backend_name):
    backend = pytest.importorskip(
        'ipympl.backend_nbagg', reason='interactive viewer requires the optional notebook backend')
    from siim.plotting._render import _slider_view

    trait = backend.Canvas.class_traits()[name]
    original_set = trait.set
    initial_backend = matplotlib.get_backend()
    plt.switch_backend(backend_name)
    previous_backend = matplotlib.get_backend()

    def fail(canvas, value):
        if value is False:
            raise RuntimeError('canvas setting failed')
        original_set(canvas, value)

    def no_draw(fig):
        pytest.fail('a failed canvas setup must stop before drawing')

    monkeypatch.setattr(trait, 'set', fail)
    try:
        with pytest.raises(RuntimeError, match='canvas setting failed'):
            _slider_view(no_draw, [(2, 'Frame', 0)], figsize=(3, 2))
        assert matplotlib.get_backend() == previous_backend
        assert not plt.get_fignums()
    finally:
        plt.switch_backend(initial_backend)


def test_basin_history_missing_early_channel_is_nan(models, monkeypatch):
    model = models['2d']
    receivers = model.receivers_out.copy()
    receivers[0] = np.arange(model.grid_nx * model.grid_ny).reshape(model.grid_ny, model.grid_nx)
    monkeypatch.setattr(model, 'receivers_out', receivers)
    result = model.plot.largest_basins_history(n_basins=2, n_samples=2, plot=False)
    assert np.isnan(result.L[:, 0]).all()
    assert np.isnan(result.k_h[:, 0]).all()
    assert np.isfinite(result.L[:, -1]).all()
    np.testing.assert_array_equal(result.stats['L']['n'], [1, 1])
    np.testing.assert_allclose(result.stats['L']['mean'], result.L[:, -1])


@pytest.mark.parametrize('field', ['bedrock', 'ice'])
@pytest.mark.parametrize('width', [4, 6])
def test_map_colorbar_labels_fit_inside_export(models, field, width):
    ax = models['2d'].plot.map(field, fig_width=width)
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for panel in fig.axes:
        bounds = panel.get_tightbbox(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0
        assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
