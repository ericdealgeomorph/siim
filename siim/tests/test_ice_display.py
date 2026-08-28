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
    hillshade or smoothing, ice cells shown directly, field defaults to
    'bedrock+ice', contours off, and small-catchment ice dropped by an area
    gate). Every style-differentiated knob resolves per style when left None; an
    explicit value always wins. Args/return order: (field, H_threshold,
    ice_sigma_cells, ice_time_avg, sigma_cells, oversample, hillshade,
    ice_extent, show_trimline, area_threshold, contour_interval)."""
    from siim.plotting.landscape import _resolve_style
    # smooth: bare bed, hillshaded, supersampled, footprint width, thin ice
    # thresholded, no area gate, contours at 100, trimline on.
    assert _resolve_style(
        'smooth', None, None, None, None, None, None, None, None, None,
        None, None) == (
        'bedrock', 100.0, 2.0, 1, 4.0, 4, True, 'footprint', True, 0.0, 100.0)
    # raw: bed AND ice, everything stripped — no smoothing, no hillshade, no
    # trimline, no contours, one pixel per cell, cells drawn directly, and
    # small-catchment specks dropped by the 1e6 m^2 area gate.
    assert _resolve_style(
        'raw', None, None, None, None, None, None, None, None, None,
        None, None) == (
        'bedrock+ice', 0.0, 0.0, 1, 0.0, 1, False, 'cells', False, 1e6, 0.0)
    # explicit values override the preset, including explicit field / zeros / False
    assert _resolve_style(
        'smooth', 'bedrock+lakes', 20.0, 0, 1, 0.0, 2, False, 'cells', False,
        5e5, 50.0) == (
        'bedrock+lakes', 20.0, 0, 1, 0.0, 2, False, 'cells', False, 5e5, 50.0)
    # raw with explicit field/hillshade/oversample/trimline kept; area_threshold
    # and contour_interval left None still resolve to the raw preset (1e6, off)
    assert _resolve_style(
        'raw', 'bedrock', None, None, None, None, 4, True, None, True,
        None, None) == (
        'bedrock', 0.0, 0.0, 1, 0.0, 4, True, 'cells', True, 1e6, 0.0)
    assert _resolve_style(
        'smooth', None, None, None, None, None, 2, None, None, None,
        None, None) == (
        'bedrock', 100.0, 2.0, 1, 2.0, 2, True, 'footprint', True, 0.0, 100.0)
    with pytest.raises(ValueError, match="style"):
        _resolve_style(
            'shiny', None, None, None, None, None, None, None, None, None,
            None, None)


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
