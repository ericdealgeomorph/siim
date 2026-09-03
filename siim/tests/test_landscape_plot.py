"""Regression tests for the landscape() ice/lake display layer.

Opens with the ice_cmap-on-an-ice-free-frame guard (audit B8); the tests after
the phantom-ice one pin the 2026-09-02 display-layer sweep (unreachable lake
layer, silently ice-free 'field' smoothing, per-frame animation colour scales,
the ice_time_avg terrain contract, the section's own y-limits, raster/overlay
registration, the section bed line, and the bare-bed colorbar label).

With ``ice_cmap`` set, an ice-free frame has an all-False ice mask, and
``H_col[ice_mask].max()`` raised 'zero-size array to reduction operation
maximum which has no identity' -- which also aborted animate_landscape mid-run
on runs that start or cycle ice-free. An ice-free frame must render (no ice
drawn), not raise.
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')            # headless; no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from siim.siim2d import siim as siim2d  # noqa: E402
from siim.plotting._render import _priority_flood  # noqa: E402
from siim.plotting.landscape import _frozen_scale_kwargs  # noqa: E402
import pytest  # noqa: E402

LAKE_RGB = np.asarray(matplotlib.colors.to_rgb('#a8c8e8'))   # lake_color default
ICE_RGB = np.asarray(matplotlib.colors.to_rgb('#e5e8ed'))    # ice_color default


def _map_rgb(ax):
    """The rendered map raster (ny_sub, nx_sub, 3)."""
    return np.asarray(ax.get_images()[0].get_array(), dtype=float)


def _section_axes(fig):
    """The cross-section panel of a landscape(cross_section=...) figure."""
    return [a for a in fig.axes if a.get_ylabel() == 'Elevation (m)'][0]


def _section_lines(ax_cs, n):
    """The section's profile/bedrock lines (the n-column ones, in draw order:
    bedrock then surface when ice is drawn, surface alone otherwise)."""
    return [np.asarray(ln.get_ydata()) for ln in ax_cs.lines
            if len(ln.get_ydata()) == n]


@pytest.fixture(autouse=True)
def _run_under_both_drivers(both_drivers):
    """S3 (Map 4 §1 PARAM): every test in this file runs under BOTH drivers --
    the conftest ``both_drivers`` fixture patches ``constants.DRIVER_DEFAULT``,
    so the existing assertions gate the in-house driver too."""



def test_landscape_ice_cmap_ice_free_frame():
    """A completed run with a huge ELA (never any ice) must render with
    ice_cmap set instead of crashing on the empty-mask reduction."""
    cfg = dict(
        U=1e-3, P=2, beta=1e-3, Ko=1e-6, n=1, ce=1e-4, nu=2, Ac=2e-24,
        lambda_p=5e2, lambda_c=1e2, alpha_g=8, sliding_law='eff-exp',
        zELA=1e5, T=1e5, Lx=3e4, Ly=3e4, nx=31, ny=31, nt=11, nt_out=4,
        D=1e-3, seed=7, boundary_status=['fixed_value'] * 4,
        initial_max_elevation=500, noise_amplitude=10, k=1,
        width_hack_k=1.0, width_hack_p=0.5, flow_routing='single',
        progress_bar=False,
    )
    m = siim2d(cfg)
    m.run()
    # Precondition: the run really is ice-free (so ice_mask is all-False).
    assert float(np.max([h.max() for h in m.H_out])) == 0.0

    # Must not raise on the empty ice mask.
    m.plot.landscape(field='bedrock+ice', ice_cmap='Blues', i=-1, colorbar=True)
    plt.close('all')


def test_cross_section_paints_no_phantom_ice():
    """The smooth-style cross-section must not paint ice in ice-free bed
    notches (the phantom-ice bug, 2026-07-16). The section previously filled
    everything between the UNsmoothed bed and the Gaussian-SMOOTHED display
    terrain as ice, so a relict 1-cell-wide carved slot rendered as a
    permanently ice-filled valley (~60% of notch depth painted, H = 0
    beneath). The section now reads the pre-smoothing composite: an ice-free
    frame must paint exactly zero ice pixels, notches included."""
    ny, nx = 31, 31
    zb0 = np.full((ny, nx), 500.0)
    zb0[5:26, 15] = 200.0            # 1-cell-wide, 300 m-deep relict slot
    cfg = dict(
        U=1e-3, zELA=1e5, T=1e3, nt=2, nt_out=2,
        Lx=3e4, Ly=3e4, nx=nx, ny=ny, seed=7,
        initial_topography=zb0, noise_amplitude=0,
        boundary_status=['fixed_value'] * 4, progress_bar=False,
    )
    m = siim2d(cfg)
    m.run()
    # Preconditions: truly ice-free, and the slot survived the (1-step) run —
    # otherwise the assertion below could pass vacuously.
    assert float(np.max([h.max() for h in m.H_out])) == 0.0
    assert float(m.zb_out[-1][:, 15].min()) < 350.0

    fig, _ = m.plot.landscape(field='bedrock+ice', i=-1, cross_section=15,
                              H_threshold=0)
    # The section's ice layer is the only zorder-2 image (bed image is 1,
    # lakes 2.5, the map is a single composite imshow).
    ice_imgs = [im for a in fig.axes for im in a.get_images()
                if im.get_zorder() == 2]
    assert ice_imgs, "cross-section ice layer not found"
    painted = sum(float(np.asarray(im.get_array())[..., 3].sum())
                  for im in ice_imgs)
    assert painted == 0.0, \
        f"phantom ice painted in an ice-free section (alpha sum {painted})"
    plt.close('all')


def test_animate_parallel_renders_mp4(tmp_path, monkeypatch):
    """The parallel animate path (audit N35): workers reload the pickled run
    and render frames via the unchanged landscape(); ffmpeg assembles.
    n_workers forced to 2 to exercise the machinery regardless of the
    adaptive auto gate (which picks serial for a job this small)."""
    import os
    import matplotlib.animation as anm
    import pytest
    if not anm.FFMpegWriter.isAvailable():
        pytest.skip("ffmpeg not available")
    from siim.siim2d import siim as siim2d
    monkeypatch.chdir(tmp_path)   # model_outputs/ + movie land in tmp
    m = siim2d(dict(U=1e-3, zELA=300, T=1e5, nt=11, nt_out=4,
                    nx=31, ny=31, Lx=3e4, Ly=3e4, seed=7,
                    initial_max_elevation=800, progress_bar=False,
                    boundary_status=['fixed_value'] * 4))
    m.run()
    # cross_section exercises the worker shim's meta attrs (_zELA_output etc.)
    out = m.plot.animate_landscape(path='par_anim', n_workers=2,
                                   field='bedrock+ice', oversample=2,
                                   cross_section=15)
    assert out.endswith('.mp4') and os.path.getsize(out) > 10_000


# --- the 2026-09-02 display-layer sweep -------------------------------------

@pytest.fixture(scope='module')
def _basin_model():
    """Short ice-free run over a bed with ONE closed interior basin and one
    1-cell peak — the two features the lake layer and the section y-limit
    need."""
    ny = nx = 31
    zb0 = np.full((ny, nx), 500.0)
    zb0[13:18, 13:18] = 200.0        # closed basin, 300 m deep
    zb0[15, 8] = 1200.0              # 1-cell peak ON the y = 15 km section row
    cfg = dict(U=1e-3, zELA=1e5, T=1e3, nt=2, nt_out=2,
               Lx=3e4, Ly=3e4, nx=nx, ny=ny, seed=7,
               initial_topography=zb0, noise_amplitude=0,
               boundary_status=['fixed_value'] * 4, progress_bar=False)
    m = siim2d(cfg)
    m.run()
    return m


@pytest.fixture(scope='module')
def _iced_model():
    """Small glacial mode-C run (ice on every output frame). The bed is
    deliberately rough (noise_amplitude=60): the section's bed line is drawn
    against the ice fill, and a bicubic upsample only overshoots it where the
    bed has sharp relief."""
    cfg = dict(
        U=1e-3, P=1, beta=1e-2, Ko=1e-6, n=1, ce=1e-5, nu=2,
        sliding_law='power', lambda_p=300.0, alpha_g=8,
        zELA=400.0, T=3e5, Lx=3e4, Ly=3e4, nx=31, ny=31, nt=11, nt_out=4,
        D=1e-3, seed=7, boundary_status=['fixed_value'] * 4,
        initial_max_elevation=800, noise_amplitude=60, mode='C',
        flow_routing='single', progress_bar=False,
    )
    m = siim2d(cfg)
    m.run()
    return m


def test_lakes_field_renders_and_floods_the_true_surface(_basin_model):
    """field='bedrock+lakes' must render at all (the boundary_status list was
    bound only inside the footprint-ice branch, so every lake render raised
    UnboundLocalError: 'bs'), and the flood must run on the TRUE composite —
    not on the Gaussian-smoothed display terrain the section cannot meet (the
    phantom-ice split, 2026-07-16)."""
    from scipy.ndimage import gaussian_filter
    m = _basin_model
    zb = np.asarray(m.zb_out[-1])
    assert zb[15, 15] < 350.0, "test setup: the basin did not survive the run"
    # oversample=1 -> the subgrid IS the native grid, so the rendered composite
    # is this bed and its lake mask can be recomputed here exactly.
    true_lake = (_priority_flood(zb) - zb) > 0.5
    zs = gaussian_filter(zb, sigma=1.0)                  # sigma_cells = oversample
    smooth_lake = (_priority_flood(zs) - zs) > 0.5
    assert true_lake.any(), "test setup: no lake on the true bed"
    assert smooth_lake.sum() != true_lake.sum(), \
        "test setup: smoothing does not change the flood here"

    for extra in (dict(), dict(cross_section=15.0)):
        fig, ax = m.plot.landscape(field='bedrock+lakes', i=-1, oversample=1,
                                   hillshade=False, **extra)
        painted = np.abs(_map_rgb(ax) - LAKE_RGB).max(axis=-1) < 1e-9
        np.testing.assert_array_equal(painted, true_lake)
        if extra:
            # the section's water band (zorder 2.5) is painted from the same
            # flood, so it must be wet where the bed row is.
            wet = sum(float(np.asarray(im.get_array())[..., 3].sum())
                      for im in _section_axes(fig).get_images()
                      if im.get_zorder() == 2.5)
            assert wet > 0.0, "section paints no water through the basin"
        plt.close(fig)


def test_field_smoothing_requires_a_positive_threshold(_basin_model):
    """ice_smoothing='field' clips H to 2*H_threshold before thresholding, so
    at H_threshold <= 0 it draws NO ice at all. style='raw' resolves
    H_threshold to 0, which made landscape(style='raw', ice_smoothing='field')
    a silently ice-free render — it must raise instead."""
    with pytest.raises(ValueError, match='H_threshold'):
        _basin_model.plot.landscape(style='raw', ice_smoothing='field', i=-1)
    with pytest.raises(ValueError, match='H_threshold'):
        _basin_model.plot.landscape(field='bedrock+ice', ice_extent='cells',
                                    ice_smoothing='field', H_threshold=0.0,
                                    i=-1)
    plt.close('all')


def test_animate_freezes_the_auto_colour_scales(_iced_model, tmp_path,
                                                monkeypatch):
    """landscape() autoscales z_max and the ice norm per frame, so an animation
    put every frame on its own scale (bed top swinging with the topography, the
    ice bar collapsing to 0-1 m on ice-free frames). animate_landscape must
    resolve both ONCE over the run and pass them down."""
    import matplotlib.animation as anm
    m = _iced_model
    mdl = m.plot.model
    kwargs = _frozen_scale_kwargs(mdl, dict(field='bedrock+ice',
                                            ice_cmap='Blues'))
    assert kwargs['z_max'] == float(np.nanmax(np.asarray(mdl.z_out)))
    assert kwargs['H_max'] == float(mdl.hc_over_H
                                    * np.nanmax(np.asarray(mdl.H_out)))
    # explicit values still win
    assert _frozen_scale_kwargs(mdl, dict(z_max=7.0))['z_max'] == 7.0

    def limits(i, **kw):
        fig, _ = m.plot.landscape(i=i, oversample=2, **kw)
        got = {a.get_ylabel(): tuple(a.get_ylim()) for a in fig.axes
               if a.get_ylabel() in ('Surface elevation (m)',
                                     'Ice thickness (m)')}
        plt.close(fig)
        return got

    assert limits(0, **kwargs) == limits(-1, **kwargs)
    auto = dict(field='bedrock+ice', ice_cmap='Blues')
    assert limits(0, **auto) != limits(-1, **auto), \
        "test setup: the per-frame autoscale does not move on this run"

    # ...and animate_landscape actually hands the frozen values to landscape().
    seen = []
    plotter = type(m.plot)
    original = plotter.landscape

    def spy(self, **kw):
        seen.append(kw)
        return original(self, **kw)

    monkeypatch.setattr(plotter, 'landscape', spy)
    monkeypatch.setattr(anm.FuncAnimation, 'save', lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():           # the no-op save() never renders
        warnings.simplefilter('ignore', UserWarning)
        m.plot.animate_landscape(path='frozen', n_workers=1,
                                 field='bedrock+ice', ice_cmap='Blues')
    assert seen and all(kw['z_max'] == kwargs['z_max']
                        and kw['H_max'] == kwargs['H_max'] for kw in seen)
    plt.close('all')


def test_ice_time_avg_keeps_the_terrain_on_frame_i(_iced_model):
    """ice_time_avg is an ICE-layer knob: its trailing mean may only reach the
    mask and the depth colouring. In ice_extent='footprint' the whole displayed
    surface was rebuilt from the averaged thickness, so the terrain (and with
    it the hillshade, contours, hypsometry and section profile) drifted off
    frame i, contradicting the documented contract. Both extents must now leave
    the terrain on frame i — and ice_time_avg=1 must be untouched (it still
    takes the single, unaveraged path)."""
    m = _iced_model
    i = -1
    H0, z0 = np.array(m.H_out[i]), np.array(m.z_out[i])
    try:
        m.H_out[i] = np.zeros_like(H0)          # frame i is ice-free ...
        m.z_out[i] = np.array(m.zb_out[i])      # ... so its terrain IS the bed
        assert (np.asarray(m.H_out[i - 1]) > 0).any(), \
            "test setup: no ice in the averaging window"
        for extent in ('footprint', 'cells'):
            profiles, ice_px = {}, {}
            for k in (1, 3):
                fig, ax = m.plot.landscape(
                    field='bedrock+ice', i=i, ice_extent=extent, oversample=1,
                    ice_time_avg=k, H_threshold=0.0, hillshade=False,
                    contour_interval=0, cross_section=15.0)
                nx_sub = _map_rgb(ax).shape[1]
                profiles[k] = _section_lines(_section_axes(fig), nx_sub)[-1]
                ice_px[k] = int((np.abs(_map_rgb(ax) - ICE_RGB).max(axis=-1)
                                 < 1e-9).sum())
                plt.close(fig)
            assert ice_px[3] > 0, \
                f"test setup: {extent} draws no averaged ice to leak"
            assert ice_px[1] == 0                      # frame i really is bare
            np.testing.assert_allclose(profiles[3], profiles[1], atol=1e-6)
    finally:
        m.H_out[i], m.z_out[i] = H0, z0
    plt.close('all')


def test_cross_section_ylim_contains_its_own_profile(_basin_model):
    """The auto z_max came from the SMOOTHED map field while the section draws
    the unsmoothed composite against the same limit, so a peak sharper than the
    Gaussian ran off the top of the panel."""
    m = _basin_model
    fig, ax = m.plot.landscape(field='bedrock', i=-1, cross_section=15.0)
    ax_cs = _section_axes(fig)
    profile = _section_lines(ax_cs, _map_rgb(ax).shape[1])[-1]
    assert profile.max() - np.median(profile) > 300.0, \
        "test setup: the 1-cell peak did not survive the run"
    assert ax_cs.get_ylim()[1] >= profile.max()
    plt.close(fig)


def test_map_raster_registers_with_the_node_coordinates(_basin_model):
    """imshow places pixel CENTRES, but the rendered array is node-valued: with
    extent [0, Lx] every pixel sat half a subgrid pixel off the contour /
    trimline / section-line coordinates (249 m at oversample=1)."""
    m = _basin_model
    mdl = m.plot.model
    for ov in (1, 4):
        fig, ax = m.plot.landscape(field='bedrock', i=-1, oversample=ov,
                                   hillshade=False)
        im = ax.get_images()[0]
        ny_sub, nx_sub = _map_rgb(ax).shape[:2]
        x0, x1, y0, y1 = im.get_extent()
        # the coordinates the overlays use
        x_axis = np.linspace(0, mdl.Lx / 1e3, nx_sub)
        y_axis = np.linspace(0, mdl.Ly / 1e3, ny_sub)
        np.testing.assert_allclose(
            x0 + (np.arange(nx_sub) + 0.5) * (x1 - x0) / nx_sub, x_axis,
            atol=1e-9)
        np.testing.assert_allclose(
            y0 + (np.arange(ny_sub) + 0.5) * (y1 - y0) / ny_sub, y_axis,
            atol=1e-9)
        # the axes still show exactly the domain
        assert ax.get_xlim() == (0.0, mdl.Lx / 1e3)
        assert ax.get_ylim() == (0.0, mdl.Ly / 1e3)
        plt.close(fig)


def test_section_paints_a_band_under_every_masked_ice_column(_iced_model):
    """The section's bed line was the BICUBIC zb_sub against the BILINEAR
    filled ice surface, so wherever the bicubic overshot above the fill the
    band collapsed to nothing on columns the map paints as ice."""
    m = _iced_model
    fig, ax = m.plot.landscape(field='bedrock+ice', i=-1, oversample=4,
                               H_threshold=0.0, ice_sigma_cells=0,
                               hillshade=False, contour_interval=0,
                               cross_section=1.5)
    rgb = _map_rgb(ax)
    ny_sub, nx_sub = rgb.shape[:2]
    j_cs = int(round(1.5e3 / (m.plot.model.Ly / (ny_sub - 1))))
    icy = np.abs(rgb[j_cs] - ICE_RGB).max(axis=-1) < 1e-9
    assert icy.any(), "test setup: no ice on the section row"
    bedrock, profile = _section_lines(_section_axes(fig), nx_sub)
    assert (profile[icy] - bedrock[icy] > 0).all(), (
        f"{int((profile[icy] - bedrock[icy] <= 0).sum())} of {int(icy.sum())} "
        "masked ice columns paint no section band")
    plt.close(fig)


def test_bare_bed_colorbar_is_labelled_bedrock(_basin_model):
    """Without ice the composite IS the bare bed, so the bar must not claim to
    be the surface (map() calls it 'Bedrock elevation (m)')."""
    m = _basin_model
    for field, want in (('bedrock', 'Bedrock elevation (m)'),
                        ('bedrock+lakes', 'Bedrock elevation (m)'),
                        ('bedrock+ice', 'Surface elevation (m)')):
        fig, _ = m.plot.landscape(field=field, i=-1, hillshade=False)
        assert want in [a.get_ylabel() for a in fig.axes]
        plt.close(fig)
