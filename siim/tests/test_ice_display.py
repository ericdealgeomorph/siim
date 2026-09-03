"""Display-side anti-flicker for landscape(): the ice_smoothing='field' variant,
min_ice_cells speckle/hole cleanup, and ice_time_avg trailing average.

Unit-tests the pure mask helpers in siim.plotting._render on small synthetic
arrays (no figure rendering), plus a figure-level smoke test that landscape()
accepts and renders the new kwargs.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')            # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pytest                    # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from siim.plotting._render import (  # noqa: E402
    _smooth_ice_mask, _field_ice_mask, _clean_ice_mask, _mean_recent_H)


def _subgrid_coords(ny, nx, oversample):
    """The (Y, X) upsample coordinates landscape() builds for a native grid."""
    ny_sub = (ny - 1) * oversample + 1
    nx_sub = (nx - 1) * oversample + 1
    y_idx = np.arange(ny_sub) * (ny - 1) / (ny_sub - 1)
    x_idx = np.arange(nx_sub) * (nx - 1) / (nx_sub - 1)
    return np.meshgrid(y_idx, x_idx, indexing='ij')


# ---------------------------------------------------------------------------
# 1. field mode is less sensitive to threshold-straddling flips than mask mode.
# ---------------------------------------------------------------------------
def test_field_mode_damps_threshold_straddle():
    ny = nx = 24
    thr, sigma, os = 50.0, 2.0, 2
    Y, X = _subgrid_coords(ny, nx, os)

    def frame(fringe):
        H = np.zeros((ny, nx))
        H[:, :10] = 100.0        # solid ice interior (= 2*thr, clip keeps it)
        H[:, 10:12] = fringe     # a 2-col margin fringe straddling the threshold
        return H

    Ha, Hb = frame(47.0), frame(53.0)   # fringe flips just across thr (+/- eps)

    m_a = _smooth_ice_mask(Ha > thr, Y, X, sigma)
    m_b = _smooth_ice_mask(Hb > thr, Y, X, sigma)
    changed_mask = int(np.logical_xor(m_a, m_b).sum())

    f_a = _field_ice_mask(Ha, Y, X, sigma, thr)
    f_b = _field_ice_mask(Hb, Y, X, sigma, thr)
    changed_field = int(np.logical_xor(f_a, f_b).sum())

    assert changed_mask > 0, "test setup: mask mode should register the flip"
    assert changed_field < changed_mask, (
        f"field mode not less flip-sensitive: field {changed_field} "
        f">= mask {changed_mask}")


# ---------------------------------------------------------------------------
# 2. min_ice_cells: drop small components, fill enclosed holes, keep border holes.
# ---------------------------------------------------------------------------
def test_clean_ice_mask_components_and_holes():
    os, minc = 2, 4                         # minpix = minc*os**2 = 16 subgrid px
    m = np.zeros((40, 40), dtype=bool)
    m[5:25, 0:25] = True                    # big blob (kept; touches left border)
    m[30:32, 30:32] = True                  # isolated 4-px speckle (removed)
    m[14:16, 14:16] = False                 # 4-px enclosed interior hole (filled)
    m[9:11, 0:2] = False                    # 4-px bare notch on the border (kept bare)

    out = _clean_ice_mask(m, minc, os)

    assert out[6, 6], "big blob removed"
    assert not out[30:32, 30:32].any(), "small speckle not removed"
    assert out[14:16, 14:16].all(), "enclosed interior hole not filled"
    assert not out[9:11, 0:2].any(), "border-touching bare notch wrongly filled"
    # min_ice_cells <= 0 is a no-op (identity).
    assert np.array_equal(_clean_ice_mask(m, 0, os), m)


# ---------------------------------------------------------------------------
# 3. ice_time_avg (trailing mean) indexing.
# ---------------------------------------------------------------------------
def test_mean_recent_H_indexing():
    rng = np.random.RandomState(0)
    H = rng.rand(6, 4, 5)
    # k=1 is identity.
    np.testing.assert_array_equal(_mean_recent_H(H, 3, 1), H[3])
    # window clamps at the start (i=0, k=3 -> just frame 0).
    np.testing.assert_array_equal(_mean_recent_H(H, 0, 3), H[0])
    # general trailing window.
    np.testing.assert_allclose(_mean_recent_H(H, 4, 3), H[2:5].mean(axis=0))
    # i=-1 normalises to the last frame.
    np.testing.assert_allclose(_mean_recent_H(H, -1, 2),
                               _mean_recent_H(H, 5, 2))
    np.testing.assert_allclose(_mean_recent_H(H, -1, 2), H[4:6].mean(axis=0))


# ---------------------------------------------------------------------------
# 4. Defaults-identity: the default kwargs reproduce the current mask pipeline.
# ---------------------------------------------------------------------------
def test_defaults_identity():
    rng = np.random.RandomState(1)
    ny = nx = 20
    os_ = 4
    Y, X = _subgrid_coords(ny, nx, os_)
    mask_nat = rng.rand(ny, nx) > 0.6
    sm = _smooth_ice_mask(mask_nat, Y, X, 2.0)
    # min_ice_cells default (0) leaves the smoothed mask untouched, and
    # ice_time_avg default (1) is the frame-i identity -> the whole default path
    # equals today's _smooth_ice_mask output.
    np.testing.assert_array_equal(_clean_ice_mask(sm, 0, os_), sm)
    H = rng.rand(5, ny, nx)
    np.testing.assert_array_equal(_mean_recent_H(H, -1, 1), H[-1])


# ---------------------------------------------------------------------------
# 5. Figure-level smoke: landscape() accepts + renders the new kwargs.
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def _iced_model():
    from siim.siim2d import siim as siim2d
    cfg = dict(
        U=1e-3, P=1, beta=1e-2, Ko=1e-6, n=1, ce=1e-5, nu=2,
        sliding_law='power', lambda_p=300.0, alpha_g=8,
        zELA=400.0, T=3e5, Lx=3e4, Ly=3e4, nx=31, ny=31, nt=11, nt_out=4,
        D=1e-3, seed=7, boundary_status=['fixed_value'] * 4,
        initial_max_elevation=800, noise_amplitude=10, mode='C',
        flow_routing='single', progress_bar=False,
    )
    m = siim2d(cfg)
    m.run()
    return m


def test_landscape_accepts_new_kwargs(_iced_model):
    m = _iced_model
    # cells + field smoothing + cleanup + time-average.
    m.plot.landscape(field='bedrock+ice', ice_extent='cells',
                     ice_smoothing='field', H_threshold=20.0,
                     min_ice_cells=2, ice_time_avg=2, i=-1, colorbar=False)
    # footprint + cleanup + time-average (field smoothing not allowed here).
    m.plot.landscape(field='bedrock+ice', ice_extent='footprint',
                     min_ice_cells=2, ice_time_avg=2, i=-1, colorbar=False)
    plt.close('all')


def test_landscape_field_footprint_raises(_iced_model):
    with pytest.raises(ValueError, match="ice_smoothing='field'"):
        _iced_model.plot.landscape(field='bedrock+ice', ice_extent='footprint',
                                   ice_smoothing='field', i=-1)
    with pytest.raises(ValueError, match="ice_time_avg"):
        _iced_model.plot.landscape(field='bedrock+ice', ice_time_avg=0, i=-1)
    plt.close('all')


def test_style_preset_resolution():
    """The style preset (Eric, 2026-07-07): 'smooth' (DEFAULT) is the
    cartographic view; 'raw' is the naked model output (one pixel per cell, no
    hillshade or smoothing, ice cells shown directly, contours off, and
    small-catchment ice dropped by an area gate). BOTH styles resolve field to
    'bedrock+ice' (Eric, 2026-09-03: resolving smooth to bare 'bedrock' made
    every ice knob a silent no-op on the default render), and ice_shading
    splits them — smooth veils ice by depth, raw paints it flat. Every
    style-differentiated knob resolves per style when left None; an explicit
    value always wins. H_threshold resolves to 0 under BOTH styles (Eric,
    2026-09-03: the veil fades thin ice out, so a thickness gate only deletes
    real glacierets) — area_threshold still splits them; trunk_display splits
    them too (smooth fills the resolved trunks as ribbons, raw draws none).
    Args/return order: (field, H_threshold,
    ice_sigma_cells, ice_time_avg, sigma_cells, oversample, hillshade,
    ice_extent, show_margin, area_threshold, contour_interval, ice_shading,
    trunk_display)."""
    from siim.plotting.landscape import _resolve_style
    # smooth: bed AND ice, hillshaded, supersampled, footprint width, NOTHING
    # hidden by thickness, no area gate, contours at 100, margin on, veiled ice.
    assert _resolve_style(
        'smooth', None, None, None, None, None, None, None, None, None,
        None, None) == (
        'bedrock+ice', 0.0, 2.0, 1, 4.0, 4, True, 'footprint', True, 0.0,
        100.0, 'veil', 'ribbons')
    # raw: bed AND ice, everything stripped — no smoothing, no hillshade, no
    # margin outline, no contours, one pixel per cell, cells drawn directly,
    # flat ice, and small-catchment specks dropped by the 1e6 m^2 area gate.
    assert _resolve_style(
        'raw', None, None, None, None, None, None, None, None, None,
        None, None) == (
        'bedrock+ice', 0.0, 0.0, 1, 0.0, 1, False, 'cells', False, 1e6, 0.0,
        'flat', 'none')
    # explicit values override the preset, including explicit field / zeros / False
    assert _resolve_style(
        'smooth', 'bedrock+lakes', 20.0, 0, 1, 0.0, 2, False, 'cells', False,
        5e5, 50.0, 'flat', 'none') == (
        'bedrock+lakes', 20.0, 0, 1, 0.0, 2, False, 'cells', False, 5e5, 50.0,
        'flat', 'none')
    # raw with explicit field/hillshade/oversample/margin kept; area_threshold,
    # contour_interval, ice_shading and trunk_display left None still resolve
    # to the raw preset (1e6, off, flat, no ribbons)
    assert _resolve_style(
        'raw', 'bedrock', None, None, None, None, 4, True, None, True,
        None, None) == (
        'bedrock', 0.0, 0.0, 1, 0.0, 4, True, 'cells', True, 1e6, 0.0, 'flat',
        'none')
    assert _resolve_style(
        'smooth', None, None, None, None, None, 2, None, None, None,
        None, None) == (
        'bedrock+ice', 0.0, 2.0, 1, 2.0, 2, True, 'footprint', True, 0.0,
        100.0, 'veil', 'ribbons')
    # ...and H_threshold stays fully functional as an explicit knob.
    assert _resolve_style(
        'smooth', None, 100.0, None, None, None, None, None, None, None,
        None, None)[1] == 100.0
    with pytest.raises(ValueError, match="style"):
        _resolve_style(
            'shiny', None, None, None, None, None, None, None, None, None,
            None, None)
    with pytest.raises(ValueError, match="ice_shading"):
        _resolve_style(
            'smooth', None, None, None, None, None, None, None, None, None,
            None, None, 'frosted')
    with pytest.raises(ValueError, match="trunk_display"):
        _resolve_style(
            'smooth', None, None, None, None, None, None, None, None, None,
            None, None, None, 'streaks')


def test_landscape_style_smoke(_iced_model):
    """Both styles render end-to-end (smooth is the signature default), and a
    bare style='raw' (no explicit field) resolves field to 'bedrock+ice' so it
    draws ice out of the box (Eric, 2026-07-07)."""
    m = _iced_model
    m.plot.landscape(field='bedrock+ice', i=-1, colorbar=False)
    m.plot.landscape(field='bedrock+ice', style='raw', i=-1, colorbar=False)
    # bare raw: field defaults to 'bedrock+ice', so ice draws. Compare against
    # an explicit 'bedrock' raw (no ice): the rendered images must differ.
    assert (m.H_out[-1] > 0).any()                     # fixture has ice
    fig_i, ax_i = m.plot.landscape(style='raw', i=-1, colorbar=False)
    fig_b, ax_b = m.plot.landscape(style='raw', field='bedrock', i=-1, colorbar=False)
    img_i = ax_i.get_images()[0].get_array()
    img_b = ax_b.get_images()[0].get_array()
    assert not np.allclose(img_i, img_b)               # ice overlay changed pixels
    plt.close('all')


def test_subthreshold_ice_does_not_pulse_terrain(_iced_model):
    """Ice hidden by H_threshold must NOT bulge the terrain (Eric, 2026-07-07):
    the rendered 'bedrock' is the bare bed where ice isn't drawn, so two frames
    that differ ONLY in sub-threshold ice render identically — no animation pulse
    of the terrain (or of the auto z_max / hillshade)."""
    m = _iced_model
    i = -1
    zb = m.zb_out[i].copy()
    H0, z0 = m.H_out[i].copy(), m.z_out[i].copy()
    try:
        def render(hpatch):                            # 0 < hpatch < H_threshold
            H = np.zeros_like(zb); H[10:20, 10:20] = hpatch
            m.H_out[i] = H
            m.z_out[i] = zb + m.hc_over_H * H
            fig, ax = m.plot.landscape(field='bedrock+ice', i=i,
                                       H_threshold=100.0, colorbar=False)
            img = np.asarray(ax.get_images()[0].get_array()).copy()
            plt.close(fig)
            return img
        a = render(30.0)
        b = render(95.0)                               # both < 100 -> never drawn
        assert np.array_equal(a, b)                     # terrain identical: no pulse
    finally:
        m.H_out[i], m.z_out[i] = H0, z0


# ---------------------------------------------------------------------------
# 6. The 2026-09-03 ice-rendering pass: ice on by default, the depth-graded
#    veil, honest thickness labels, and the margin rename.
# ---------------------------------------------------------------------------
ICE_COLOR_RGB = np.asarray(matplotlib.colors.to_rgb('#e5e8ed'))  # flat default


def test_default_landscape_draws_ice(_iced_model):
    """A bare landscape() must SHOW the ice (Eric, 2026-09-03). 'smooth' used
    to resolve field to bare 'bedrock', so the signature default render hid the
    model's headline state and every ice kwarg was a silent no-op."""
    m = _iced_model
    assert (np.asarray(m.H_out[-1]) > 0).any()         # fixture has ice
    fig_d, ax_d = m.plot.landscape(colorbar=False)     # NO arguments but the axes
    fig_b, ax_b = m.plot.landscape(field='bedrock', colorbar=False)
    img_d = np.asarray(ax_d.get_images()[0].get_array(), dtype=float)
    img_b = np.asarray(ax_b.get_images()[0].get_array(), dtype=float)
    assert not np.allclose(img_d, img_b), "the default render paints no ice"
    plt.close('all')


def test_veil_is_depth_graded_and_translucent(_iced_model):
    """style='smooth' shades ice as a depth-graded translucent veil: the ramp
    colour at the column's depth, alpha-blended over the terrain so thin apron
    ice lets the bed read through. Deep ice (t >= 0.40) is fully opaque, so it
    lands exactly on the ramp; thin ice must be a strict blend of both."""
    from siim.plotting._render import _ice_ramp
    m = _iced_model
    i = -1
    zb = np.asarray(m.zb_out[i])
    H0, z0 = m.H_out[i].copy(), m.z_out[i].copy()
    try:
        H = np.zeros_like(zb)
        H[5:10, 5:25] = 100.0                          # thin: t = 0.1
        H[15:20, 5:25] = 1000.0                        # thick: t = 1
        m.H_out[i] = H
        m.z_out[i] = zb + m.hc_over_H * H
        # cells extent at oversample=1 with no mask smoothing: H_col is exactly
        # hc_over_H * H, so t is known in closed form.
        fig, ax = m.plot.landscape(
            field='bedrock+ice', i=i, ice_extent='cells', oversample=1,
            ice_sigma_cells=0, sigma_cells=0, hillshade=False,
            contour_interval=0, H_threshold=1.0,
            H_max=m.hc_over_H * 1000.0, colorbar=False)
        rgb = np.asarray(ax.get_images()[0].get_array(), dtype=float)
        plt.close(fig)
        ramp = _ice_ramp()
        thin, thick = rgb[7, 15], rgb[17, 15]
        np.testing.assert_allclose(thick, ramp(1.0)[:3], atol=1e-9)
        assert not np.allclose(thin, ramp(0.1)[:3], atol=1e-3)   # translucent
        assert not np.allclose(thin, ICE_COLOR_RGB, atol=1e-3)   # not the flat fill
        assert not np.allclose(thin, thick, atol=1e-3)           # depth-graded
        # ...and 'flat' still stamps the one opaque ice_color (raw's shading).
        fig, ax = m.plot.landscape(
            field='bedrock+ice', i=i, ice_extent='cells', oversample=1,
            ice_sigma_cells=0, sigma_cells=0, hillshade=False,
            contour_interval=0, H_threshold=1.0, ice_shading='flat',
            colorbar=False)
        flat = np.asarray(ax.get_images()[0].get_array(), dtype=float)
        plt.close(fig)
        np.testing.assert_allclose(flat[7, 15], ICE_COLOR_RGB, atol=1e-9)
        np.testing.assert_allclose(flat[17, 15], ICE_COLOR_RGB, atol=1e-9)
    finally:
        m.H_out[i], m.z_out[i] = H0, z0
    plt.close('all')


def test_veil_depth_scale_is_run_global(_iced_model):
    """The veil normalises on the RUN-GLOBAL column depth, not this frame's
    max, so a movie's frames are directly comparable (and the still agrees with
    what animate_landscape freezes)."""
    from siim.plotting.landscape import _frozen_scale_kwargs, _run_global_H_max
    m = _iced_model
    mdl = m.plot.model
    run_max = _run_global_H_max(mdl)
    assert _frozen_scale_kwargs(mdl, {})['H_max'] == run_max

    def render(**kw):
        fig, ax = m.plot.landscape(field='bedrock+ice', i=0, oversample=1,
                                   hillshade=False, contour_interval=0,
                                   H_threshold=0.0, colorbar=False, **kw)
        img = np.asarray(ax.get_images()[0].get_array(), dtype=float).copy()
        plt.close(fig)
        return img

    frame_max = mdl.hc_over_H * float(np.asarray(mdl.H_out[0]).max())
    assert frame_max < run_max, "test setup: frame 0 already carries the run max"
    np.testing.assert_array_equal(render(), render(H_max=run_max))
    assert not np.allclose(render(), render(H_max=frame_max))


def test_ice_labels_say_what_they_show(_iced_model):
    """The landscape bar shows the local COLUMN depth (hc_over_H * H at the
    thalweg, more on carved flanks); map(field='ice') and the profile panels
    show the width-mean H the physics consumes. The labels must not both say
    'Ice thickness (m)' (Eric, 2026-09-03)."""
    from siim.plotting.maps import FIELD_REGISTRY
    m = _iced_model
    fig, _ = m.plot.landscape(field='bedrock+ice', i=-1)     # veil -> ice bar
    labels = [a.get_ylabel() for a in fig.axes]
    assert 'Ice column depth (m)' in labels
    assert 'Surface elevation (m)' in labels
    plt.close(fig)
    assert FIELD_REGISTRY['ice'][2] == 'Mean ice thickness H̄ (m)'
    ax = m.plot.map(field='ice', i=-1)
    assert FIELD_REGISTRY['ice'][2] in [a.get_ylabel() for a in ax.figure.axes]
    plt.close('all')


def test_map_ice_masks_the_bare_ground(_iced_model):
    """map(field='ice') painted H = 0 the colormap's palest blue, so ice-free
    ground read as a thin skin of ice. Zero ice is masked to a neutral bare
    tone; the figure follows the domain aspect and carries the time stamp."""
    m = _iced_model
    mdl = m.plot.model
    ax = m.plot.map(field='ice', i=-1)
    im = ax.get_images()[0]
    arr = im.get_array()
    assert np.ma.is_masked(arr)
    np.testing.assert_array_equal(np.asarray(np.ma.getmaskarray(arr)),
                                  np.asarray(mdl.H_out[-1]) <= 0)
    np.testing.assert_allclose(im.get_cmap().get_bad()[:3],
                               matplotlib.colors.to_rgb('#eeeae4'), atol=1e-9)
    w, h = ax.figure.get_size_inches()
    assert h / w == pytest.approx(mdl.Ly / mdl.Lx)
    assert ax.get_title(loc='right')                  # output time stamped
    # the bedrock field is untouched (nothing to mask, no bad colour needed)
    ax_b = m.plot.map(field='bedrock', i=-1)
    assert not np.ma.is_masked(ax_b.get_images()[0].get_array())
    plt.close('all')


def test_show_margin_replaces_show_trimline(_iced_model):
    """The outline is the CURRENT ice margin, not a trimline. The old names
    still work as deprecated aliases and must map through unchanged."""
    m = _iced_model
    kw = dict(field='bedrock+ice', i=-1, colorbar=False)
    with pytest.warns(DeprecationWarning, match='show_trimline'):
        fig_a, ax_a = m.plot.landscape(show_trimline=False, **kw)
    fig_b, ax_b = m.plot.landscape(show_margin=False, **kw)
    assert len(ax_a.collections) == len(ax_b.collections)
    fig_on, ax_on = m.plot.landscape(show_margin=True, **kw)
    assert len(ax_on.collections) > len(ax_b.collections)   # margin really drawn
    plt.close('all')
    with pytest.warns(DeprecationWarning, match='trimline_color'):
        m.plot.landscape(trimline_color='red', **kw)
    with pytest.warns(DeprecationWarning, match='trimline_lw'):
        m.plot.landscape(trimline_lw=2.0, **kw)
    plt.close('all')


# ---------------------------------------------------------------------------
# 7. The trunk-ribbon ice look (2026-09-03): the downstream-closure trunk
#    class, its true-width ribbons, and the linear veil ramp underneath.
# ---------------------------------------------------------------------------
STEM_ROW = 4
STEM_NX, STEM_NY = 13, 9
STEM_LX, STEM_LY = 6000.0, 4000.0        # dx = dy = 500 m
STEM_ALPHA_G = 6.0


def _stem_network(H_head=200.0, H_tail=20.0):
    """One stem along row 4 flowing +x: cells 1..5 thick enough to seed the
    trunk class (W = 6*200 = 1200 m >= dx), 6..11 a thinning tongue below the
    cut, cell 12 ice-free (the terminus). Returns (H, rec, area)."""
    H = np.zeros((STEM_NY, STEM_NX))
    H[STEM_ROW, 1:6] = H_head
    H[STEM_ROW, 6:12] = H_tail
    rec = np.arange(STEM_NY * STEM_NX).reshape(STEM_NY, STEM_NX)
    rec[STEM_ROW, 1:12] = np.arange(2, 13) + STEM_ROW * STEM_NX
    return H, rec, np.ones_like(H)


def test_channel_closure_carries_the_class_to_the_terminus():
    """The trunk class is the DOWNSTREAM closure of the width cut, not the cut:
    a plain 'W >= dx' cut stops where the tongue thins (measured on the
    reference run: 0 of 64 termini reached), so the closure carries it down to
    the last icy cell — and never past it, onto bare ground."""
    from siim.plotting._render import _channel_closure
    H, rec, _ = _stem_network()
    dx = STEM_LX / (STEM_NX - 1)
    seed = STEM_ALPHA_G * H >= dx
    assert seed.sum() == 5 and not seed[STEM_ROW, 6:].any()   # cut stops early
    cls = _channel_closure(H, rec, seed)
    want = np.zeros_like(cls)
    want[STEM_ROW, 1:12] = True                              # the whole tongue
    np.testing.assert_array_equal(cls, want)
    # ...and walking receivers from any class cell never leaves the class.
    flat, r = cls.ravel(), rec.ravel()
    for node in np.nonzero(flat)[0]:
        while r[node] != node and H.ravel()[r[node]] > 0:
            node = r[node]
            assert flat[node], 'closure leaks: a downstream icy cell is out'


def test_trunk_ribbons_draw_true_width_ice_only():
    """The ribbon is the claimed width W = alpha_g*H, centreline depth
    hc_over_H*H — and it is drawn ONLY over ice: no ribbon pixel may sit on a
    native cell whose whole 3x3 neighbourhood is bare."""
    from scipy.ndimage import maximum_filter
    from siim.plotting._render import _channel_closure, _trunk_ribbons
    H, rec, area = _stem_network()
    dx = STEM_LX / (STEM_NX - 1)
    cls = _channel_closure(H, rec, STEM_ALPHA_G * H >= dx)
    oversample = 4
    depth, surface = _trunk_ribbons(
        rec, H, area, 100.0 + 1.5 * H, cls, STEM_NX, STEM_NY,
        STEM_LX, STEM_LY, STEM_ALPHA_G, oversample, hc_over_H=1.5)
    ribbon = depth > 0
    assert ribbon.any()
    dy_sub = STEM_LY / ((STEM_NY - 1) * oversample)
    j_mid = STEM_ROW * oversample
    # width at a head column: the drawn band spans W = 6*200 = 1200 m
    col = ribbon[:, 3 * oversample]
    assert col.sum() * dy_sub == pytest.approx(1200.0, abs=2 * dy_sub)
    # centreline depth is the channel-floor column, not the mean thickness
    assert depth[j_mid, 3 * oversample] == pytest.approx(1.5 * 200.0, rel=1e-6)
    # ...and the flat source ice surface rides along (zs = 100 + 1.5*H here)
    assert surface[j_mid, 3 * oversample] == pytest.approx(100.0 + 1.5 * 200.0)
    # honesty: every ribbon pixel has ice in its native 3x3 neighbourhood
    icy_near = maximum_filter(H, size=3, mode='constant') > 0
    Y, X = _subgrid_coords(STEM_NY, STEM_NX, oversample)
    near_sub = icy_near[np.rint(Y).astype(int), np.rint(X).astype(int)]
    assert not (ribbon & ~near_sub).any()


def test_trunk_ribbons_cross_a_looped_seam():
    """A trunk that wraps a looped axis must be drawn on BOTH sides of the
    seam. The path tracer CLOSES its path at a wrap (one segment from column
    nx-1 to column 0 would otherwise rasterize as a stripe across the domain),
    so the wrapped step is drawn explicitly, once per side — without it the
    cell that only wraps draws nothing at all."""
    from siim.plotting._render import _channel_closure, _trunk_ribbons
    H = np.zeros((STEM_NY, STEM_NX))
    H[STEM_ROW, [STEM_NX - 1, 0, 1, 2]] = 200.0        # wraps at the seam
    rec = np.arange(STEM_NY * STEM_NX).reshape(STEM_NY, STEM_NX)
    rec[STEM_ROW, STEM_NX - 1] = STEM_ROW * STEM_NX    # nx-1 -> column 0
    rec[STEM_ROW, 0:3] = np.arange(1, 4) + STEM_ROW * STEM_NX
    dx = STEM_LX / (STEM_NX - 1)
    cls = _channel_closure(H, rec, STEM_ALPHA_G * H >= dx)
    assert cls[STEM_ROW, STEM_NX - 1] and cls[STEM_ROW, 2]
    oversample = 4
    args = (rec, H, np.ones_like(H), 1.5 * H, cls, STEM_NX, STEM_NY,
            STEM_LX, STEM_LY, STEM_ALPHA_G, oversample)
    j_mid = STEM_ROW * oversample
    seam = _trunk_ribbons(*args, wrap_x=True, hc_over_H=1.5)[0]
    cut = _trunk_ribbons(*args, wrap_x=False, hc_over_H=1.5)[0]
    assert seam[j_mid, 0] > 0 and seam[j_mid, -1] > 0     # drawn on both sides
    assert cut[j_mid, 0] > 0 and cut[j_mid, -1] == 0      # the far side is lost


def test_veil_ramp_is_linear_and_keeps_thin_ice_subordinate():
    """The veil opacity is a LINEAR ramp topping out at t = 0.35. The old
    sqrt(t/0.40) front-loading made a 30 m column half-opaque, so an apron of
    sub-resolution ice read as one sheet with the trunks barely darker."""
    from siim.plotting.landscape import _veil_alpha
    t = np.array([0.0, 0.05, 0.175, 0.35, 0.7, 1.0])
    a = _veil_alpha(t)
    np.testing.assert_allclose(
        a, [0.12, 0.12 + 0.88 * 0.05 / 0.35, 0.56, 1.0, 1.0, 1.0], atol=1e-12)
    old_sqrt = 0.18 + 0.82 * np.sqrt(0.05 / 0.40)
    assert a[1] < old_sqrt - 0.2            # thin ice much more see-through


def test_trunk_display_none_is_the_plain_veil(_iced_model, monkeypatch):
    """``trunk_display='none'`` is the pre-ribbon render, unchanged: the trunk
    machinery is never invoked (nor under ``style='raw'``), the trunk kwargs
    are no-ops, and a seed width no cell can reach collapses onto the same
    image pixel for pixel."""
    import siim.plotting.landscape as landscape_mod
    m = _iced_model

    def raster(**kw):
        fig, ax = m.plot.landscape(i=-1, colorbar=False, z_min=0.0,
                                   z_max=1500.0, **kw)
        img = np.asarray(ax.get_images()[0].get_array(), dtype=float).copy()
        plt.close(fig)
        return img

    base = raster(trunk_display='none')
    np.testing.assert_array_equal(base, raster(
        trunk_display='none', trunk_alpha=0.2, trunk_width_cells=0.1))
    np.testing.assert_array_equal(base, raster(trunk_width_cells=1e6))
    assert not np.allclose(base, raster())          # ribbons ARE the default

    def boom(*args, **kwargs):
        raise AssertionError('the ribbon machinery ran on a no-trunk path')

    monkeypatch.setattr(landscape_mod, '_trunk_ribbons', boom)
    monkeypatch.setattr(landscape_mod, '_channel_closure', boom)
    raster(trunk_display='none')
    m.plot.landscape(i=-1, style='raw', colorbar=False)
    plt.close('all')


def test_ribbons_separate_the_trunks_from_the_apron(_iced_model):
    """The point of the look: the resolved trunks stop reading as slightly
    darker streaks in a uniform veil. Measured as the mean RGB distance
    between trunk and apron pixels (the veil-only render scored 0.13 against
    0.67 for apron-vs-bare on the reference run). Under ribbons the margin
    outlines the RIBBONS only, so a frame with no resolved trunk draws no
    outline at all rather than ringing every glacieret."""
    from siim.plotting._render import (_channel_closure, _trunk_ribbons,
                                       _footprint_ice_surface, _smooth_ice_mask)
    m = _iced_model
    i = -1
    H = np.asarray(m.H_out[i], dtype=float)
    zb = np.asarray(m.zb_out[i], dtype=float)
    rec, area = m.receivers_out[i], np.asarray(m.area_out[i], dtype=float)
    ny, nx = H.shape
    dx, dy = m.Lx / (nx - 1), m.Ly / (ny - 1)
    oversample = 4
    cls = _channel_closure(H, rec, m.alpha_g * H >= dx)
    assert cls.any() and cls.sum() < (H > 0).sum()   # some, not all, ice
    trunk = _trunk_ribbons(rec, H, area, zb + m.hc_over_H * H, cls, nx, ny,
                           m.Lx, m.Ly, m.alpha_g, oversample,
                           hc_over_H=m.hc_over_H)[0] > 0
    _, ice_nat, _ = _footprint_ice_surface(H, zb, rec, m.alpha_g, dy, dx,
                                           False, False,
                                           hc_over_H=m.hc_over_H)
    Y, X = _subgrid_coords(ny, nx, oversample)
    apron = _smooth_ice_mask(ice_nat, Y, X, 2.0) & ~trunk
    assert trunk.any() and apron.any()

    def contrast(**kw):
        fig, ax = m.plot.landscape(i=i, colorbar=False, z_min=0.0,
                                   z_max=1500.0, **kw)
        img = np.asarray(ax.get_images()[0].get_array(), dtype=float)
        d = float(np.linalg.norm(img[trunk].mean(0) - img[apron].mean(0)))
        plt.close(fig)
        return d

    assert contrast() > 2.0 * contrast(trunk_display='none')

    def n_outlines(**kw):
        fig, ax = m.plot.landscape(i=i, colorbar=False, contour_interval=0,
                                   z_min=0.0, z_max=1500.0, **kw)
        n = len(ax.collections)
        plt.close(fig)
        return n

    assert n_outlines(trunk_display='none') == 1     # the veil is outlined
    assert n_outlines() == 1                         # ...the ribbons are
    assert n_outlines(trunk_width_cells=1e6) == 0    # ...the veil is NOT
