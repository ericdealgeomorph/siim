"""1D/2D parity + H-diffusion edge cases (audit theme 6).

Kernel-level regressions for the adjudicated fixes:

* **m13** — the SFR eff-exp eroders gate on ``Qg > 0`` (like the other six
  eroders): with ``ce = 0`` a glaciated cell gets NO erosion (not fluvial).
* **m15** — the 2D mode-B skeletons re-zero diffusion-smeared H at ice-free
  cells (1D parity; no phantom apron past the terminus).
* **m16** — ``_diffuse_H_2d`` is seam-aware on looped axes (wraps the stencil),
  and bit-for-bit with the old fixed-boundary kernel when not looped.
* **m18** — the mode-B skeletons scrub negative/NaN H BEFORE diffusion, so one
  bad cell can't spread through the 5-point stencil.
"""
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from siim._core.diffusion import _diffuse_H_2d                       # noqa: E402
from siim._core.eroders import _nonlinear_erode_2d, _linear_erode_2d  # noqa: E402
from siim._core.params import GlacialParams                          # noqa: E402
from siim._core.solvers import LAW_EFFEXP                            # noqa: E402
from siim._core.skeleton import _glac_fast_solve_modeB_sfr           # noqa: E402


# --- m13: SFR eff-exp gives no erosion under ice at ce = 0 -------------------

def test_m13_effexp_no_erosion_under_ice_at_ce0():
    # Two-node chain: node 1 -> node 0 (outlet). Node 1 is glaciated (Qg>0) but
    # ce=0 (Kg=0). Old code fell to fluvial and incised it; the fix gates on
    # Qg>0 so the glacial branch runs with G=0 -> no erosion (z stays at zo).
    for erode in (_nonlinear_erode_2d, _linear_erode_2d):
        zo = np.array([0.0, 10.0])
        z = zo.copy()
        Qf = np.array([1.0, 1.0])
        Qg = np.array([0.0, 5.0])          # node 1 under ice
        Kf = np.array([1.0, 1.0])          # fluvial available
        Kg = np.array([0.0, 0.0])          # ce = 0 -> no glacial erosion
        rec = np.array([0, 0])
        stack = np.array([0, 1])
        if erode is _nonlinear_erode_2d:
            erode(z, zo, Qf, Qg, Kf, Kg, 0.5, 0.5, 1.0, 1.0, stack, rec)
        else:
            erode(z, zo, Qf, Qg, Kf, Kg, 0.5, 0.5, stack, rec)
        assert z[1] == zo[1]               # glaciated + ce=0 -> untouched
        assert z[0] == zo[0]


# --- m16: looped-axis H diffusion wraps; non-looped is unchanged -------------

def test_m16_diffuse_wraps_seam_only_when_looped():
    ny, nx = 5, 8
    H0 = np.zeros(ny * nx)
    H0[2 * nx + 0] = 10.0                   # a bump on the x = 0 seam column
    D, dt, dx = 1e3, 10.0, 100.0

    Hn = H0.copy()
    _diffuse_H_2d(Hn, ny, nx, dx, dx, D, dt, False, False)   # no wrap
    Hn = Hn.reshape(ny, nx)
    assert Hn[2, 0] == 10.0                 # column 0 is a held boundary
    assert Hn[2, -1] == 0.0                 # nothing crosses to the far column

    Hw = H0.copy()
    _diffuse_H_2d(Hw, ny, nx, dx, dx, D, dt, False, True)    # wrap_x
    Hw = Hw.reshape(ny, nx)
    assert Hw[2, 0] < 10.0                  # seam column now diffuses
    assert Hw[2, -1] > 0.0                  # ice crosses the seam

    # fully periodic (both axes) conserves mass — no held-boundary sinks
    Hp = H0.copy()
    _diffuse_H_2d(Hp, ny, nx, dx, dx, D, dt, True, True)
    assert np.isclose(Hp.sum(), 10.0)


def test_m16_diffuse_bitforbit_default():
    # The default (non-looped) path must reproduce the old fixed-boundary
    # kernel: interior 5-point update, outer ring held. Check against an
    # explicit reference computation on one sub-step.
    ny, nx = 4, 5
    rng = np.random.default_rng(1)
    H0 = rng.random(ny * nx)
    D, dt, dx = 10.0, 1.0, 100.0            # ax = ay = 1e-3 -> single sub-step
    H = H0.copy()
    _diffuse_H_2d(H, ny, nx, dx, dx, D, dt)
    ref = H0.copy().reshape(ny, nx)
    a = D * dt / dx ** 2
    src = H0.reshape(ny, nx)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            ref[j, i] = src[j, i] + a * (src[j - 1, i] + src[j + 1, i]
                                         + src[j, i - 1] + src[j, i + 1]
                                         - 4.0 * src[j, i])
    np.testing.assert_allclose(H.reshape(ny, nx), ref, rtol=0, atol=0)


# --- m18: a NaN spreads through diffusion; the pre-scrub contains it ---------

def test_m18_nan_spreads_but_prescrub_contains():
    ny, nx = 9, 9
    H = np.zeros(ny * nx)
    H[4 * nx + 4] = np.nan
    bad = H.copy()
    _diffuse_H_2d(bad, ny, nx, 100.0, 100.0, 500.0, 18.0)
    assert np.isnan(bad).sum() > 1                       # the hazard: NaN spreads
    good = H.copy()
    good[~np.isfinite(good)] = 0.0                       # skeleton's pre-diffusion scrub
    _diffuse_H_2d(good, ny, nx, 100.0, 100.0, 500.0, 18.0)
    assert np.all(np.isfinite(good))
    assert np.count_nonzero(good) == 0                   # nothing to diffuse


# --- m15: mode-B skeleton re-zeros diffusion-smeared H at ice-free cells -----

def _gp_effexp(cg, lambda_p, hc, D_H, mu):
    # eff-exp GlacialParams (only cg/lambda_p/mu/hc/D_H are consumed here).
    return GlacialParams(0.0, 0.0, 0.0, 1.0, 2.0, 0.5, mu, cg, 8.0, lambda_p,
                         0.0, 0.0, 0.0, 0.0, hc, D_H)


def test_m15_diffused_H_rezeroed_at_ice_free_cells():
    # Tilted plane draining -x to the column-0 border; ice only on a mid block.
    ny, nx, dx = 5, 12, 100.0
    nn = ny * nx
    _, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    zb = (5.0 + 0.02 * dx * ii.astype(float)).ravel()     # slopes down toward -x
    rec = np.arange(nn, dtype=np.int64)
    lengths = np.zeros(nn)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            rec[j * nx + i] = j * nx + (i - 1)
            lengths[j * nx + i] = dx
    stack = np.argsort(ii.ravel(), kind='stable').astype(np.int64)

    ice = np.zeros(nn)
    icy = (ii >= 3) & (ii <= 5) & (np.arange(ny)[:, None] >= 1) \
        & (np.arange(ny)[:, None] <= ny - 2)
    ice[icy.ravel()] = 1e7                                 # ice on a mid block only
    ice_free = ice <= 0.0

    p = _gp_effexp(cg=2.3e-4, lambda_p=300.0, hc=1.5, D_H=5e5, mu=4.0 / 15.0)
    H = np.zeros(nn)
    surf = np.empty(nn)
    _glac_fast_solve_modeB_sfr(zb.copy(), ice, np.zeros(nn), H, surf,
                               LAW_EFFEXP, p, 100.0, lengths, stack, rec,
                               ny, nx, dx, dx, np.zeros(nn))
    assert np.all(np.isfinite(H))
    # despite a huge D_H, no ice thickness survives at ice-free cells (m15)
    assert np.all(H[ice_free] == 0.0)
    assert np.any(H[~ice_free] > 0.0)                      # ice kept where it exists
