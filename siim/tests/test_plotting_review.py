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
