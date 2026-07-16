"""Regression test for landscape(ice_cmap=...) on an ice-free frame (audit B8).

With ``ice_cmap`` set, an ice-free frame has an all-False ice mask, and
``H_col[ice_mask].max()`` raised 'zero-size array to reduction operation
maximum which has no identity' -- which also aborted animate_landscape mid-run
on runs that start or cycle ice-free. An ice-free frame must render (no ice
drawn), not raise.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')            # headless; no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from siim.siim2d import siim as siim2d  # noqa: E402
import pytest  # noqa: E402


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
