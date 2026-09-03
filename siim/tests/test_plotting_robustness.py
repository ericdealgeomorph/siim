"""Plotting/display robustness (audit theme 7).

* **m29** — the numba soft hillshade matches matplotlib's LightSource on flat /
  near-flat fields (clip, not darken + amplify noise).
* **m31** — the 1D ice_thickness auto y-limit scales with the data (500 m floor)
  like the 2D plotter, instead of a hard 500 m cap.
* **m32** — the display lake-fill is seam-aware on looped axes.
* **m33** — _chain_length is bounded, so a cyclic receiver array can't hang the
  path tracer (defensive).
"""
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from siim.plotting._render import (_shade_rgb_soft, _priority_flood,        # noqa: E402
                                   _trace_paths_arrays)
from siim.siim1d import siim as siim1d                                       # noqa: E402


# --- m29: flat / near-flat hillshade matches matplotlib ---------------------

def test_m29_flat_hillshade_matches_matplotlib():
    from matplotlib.colors import LightSource
    ny, nx = 6, 6
    rgb = np.full((ny, nx, 3), 0.6)
    az, alt, dx = 315.0, 45.0, 100.0
    ls = LightSource(azdeg=az, altdeg=alt)

    for z in (np.zeros((ny, nx)),                       # flat
              1e-8 * np.random.default_rng(0).random((ny, nx))):  # near-flat noise
        got = _shade_rgb_soft(rgb.copy(), z, dx, dx, 1.0, az, alt)
        want = ls.shade_rgb(rgb.copy(), z, vert_exag=1.0, dx=dx, dy=dx,
                            blend_mode='soft')[..., :3]
        np.testing.assert_allclose(got, want, atol=1e-6)
        assert got.mean() > 0.6                          # brightened, not darkened


# --- m31: 1D ice_thickness ylim scales with data ----------------------------

def _glac1d(**ov):
    p = dict(U=1e-3, P=2.0, beta=1e-2, Ko=1e-6, n=1, ce=1e-4, nu=2,
             Ac=2e-24, lambda_p=5e2, lambda_c=1e2, alpha_g=8,
             sliding_law='eff-exp', zELA=600.0,
             L=3e4, xo=1e3, k_h=5, d=2, sigma=0.5, k=1,
             T=2e4, nt=21, dx=500.0,
             left_bc='reflecting', right_bc='base_level', progress_bar=False)
    p.update(ov)
    return p


def test_m31_ice_thickness_ylim_grows_with_data():
    m = siim1d(_glac1d()); m.run()
    # thick ice (dome regime): the limit grows past the 500 m floor
    m.H_out = np.full_like(m.H_out, 800.0)
    hi = m.plot._compute_field_ylims(['ice_thickness'])['ice_thickness'][1]
    assert hi == max(500.0, 800.0 * 1.05)
    # thin ice: floored at 500 m
    m.H_out = np.full_like(m.H_out, 100.0)
    hi = m.plot._compute_field_ylims(['ice_thickness'])['ice_thickness'][1]
    assert hi == 500.0


# --- m32: display lake-fill wraps looped seams ------------------------------

def test_m32_priority_flood_seam_aware():
    ny, nx = 5, 6
    z = np.full((ny, nx), 10.0)
    z[2, 0] = 1.0
    z[2, -1] = 1.0                            # a depression straddling the x-seam

    f_no = _priority_flood(z, wrap_y=False, wrap_x=False)
    assert f_no[2, 0] == 1.0                  # seam column is a boundary outlet -> drains

    f_wrap = _priority_flood(z, wrap_y=False, wrap_x=True)
    assert f_wrap[2, 0] > 1.0                 # stays one interior lake across the seam
    assert f_wrap[2, -1] > 1.0

    # the default call is unchanged (bit-for-bit with the old fixed-boundary flood)
    np.testing.assert_array_equal(_priority_flood(z), f_no)


# --- m33: cyclic receivers don't hang the path tracer -----------------------

def test_m33_chain_length_bounded_on_cyclic_receivers():
    ny, nx = 2, 4
    nn = ny * nx
    rec = np.arange(nn).astype(float)         # self-receiving
    rec[1] = 5.0; rec[5] = 6.0; rec[6] = 5.0  # feeder 1->5 into a 5<->6 2-cycle
    H = np.zeros(nn); H[[1, 5, 6]] = 10.0
    area = np.zeros(nn); area[[1, 5, 6]] = 1e9
    out = _trace_paths_arrays(rec.reshape(ny, nx), H.reshape(ny, nx),
                              area.reshape(ny, nx), nx, ny, 300.0, 300.0, 1e3)
    assert out is not None and len(out) == 4   # returns (would hang pre-fix)


# --- min_ice_cells cleanup is seam-aware on looped axes ---------------------

def test_clean_ice_mask_seam_aware():
    """_clean_ice_mask labelled with non-periodic connectivity, so a glacier
    straddling a looped seam was two sub-minimum components and min_ice_cells
    deleted it — while the identical glacier mid-domain survived. Holes across
    the seam were likewise never fillable (the seam counted as 'array border').
    The wrap flags default False, so every non-looped call is unchanged."""
    from siim.plotting._render import _clean_ice_mask
    os_, minc = 4, 6                       # minpix = minc*os_**2 = 96 px
    mask = np.zeros((48, 48), dtype=bool)
    mask[20:28, :8] = True                 # 64 px left of the x seam
    mask[20:28, -8:] = True                # 64 px right of it -> 128 together
    mid = np.zeros((48, 48), dtype=bool)
    mid[20:28, 20:36] = True               # the same 128 px, mid-domain

    assert _clean_ice_mask(mid, minc, os_).sum() == 128        # kept anywhere
    assert _clean_ice_mask(mask, minc, os_).sum() == 0         # seam: erased
    assert _clean_ice_mask(mask, minc, os_, wrap_x=True).sum() == 128
    assert _clean_ice_mask(mask, minc, os_, wrap_y=True).sum() == 0  # wrong axis
    assert _clean_ice_mask(mask.T, minc, os_, wrap_y=True).sum() == 128

    # a bare hole straddling the seam is enclosed on a looped axis -> fillable
    hole = np.ones((48, 48), dtype=bool)
    hole[20:24, :3] = False
    hole[20:24, -3:] = False
    assert (~_clean_ice_mask(hole, minc, os_)).sum() == 24     # border: kept open
    assert (~_clean_ice_mask(hole, minc, os_, wrap_x=True)).sum() == 0


# --- the 1D ice panel names the WIDTH-MEAN thickness ------------------------

def test_ice_thickness_panel_names_the_mean(tmp_path, monkeypatch):
    """H_out is the width-MEAN thickness the physics consumes, not the channel
    -floor column depth 1.5*H that landscape() paints; the 1D panel used the
    same 'Ice thickness (m)' label as the map, so the two read as one quantity
    (Eric, 2026-09-03)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    monkeypatch.chdir(tmp_path)
    m = siim1d(_glac1d()); m.run()
    fig, axes = m.plot.profile(fields='ice_thickness')
    ax = axes[0] if np.ndim(axes) else axes
    assert ax.get_ylabel() == 'Mean ice thickness (m)'
    plt.close(fig)
