"""Front-door forcing/validation contract (audit theme 4).

Pins the adjudicated fixes:

* **m14** — the standalone fastscape ``GlacialLaw`` rejects an unknown
  ``sliding_law`` with a loud ValueError (it used to fall through to eff-exp).
* **m19** — 1D ``U`` shape contract: a flat ``nx*nt`` array is the row-major
  ``(nx, nt)`` grid; a transposed 2D array is rejected at construction.
* **m21** — 1D ``U`` / ``zELA`` accept plain Python lists (asarray-coerced like
  ``P``), not a raw ``AttributeError``.
* **m22** — 1D consumes per-step forcing at STEP-START (xsimlab-native): step
  ``tj`` uses ``series[tj-1]``, so ``series[0]`` is consumed and ``series[nt-1]``
  is unused — aligned with siim2d.
* **m51** — 2D accepts a length-``nt`` ``U`` (spatially-uniform, time-varying),
  threaded as a per-step scalar; it equals the ``(nt, ny, nx)`` broadcast.
* **m54** — ``constants.XO`` single-sources the fixed ``xo`` default (siim1d +
  the analytical steady_state).
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from siim import constants as _C                       # noqa: E402
from siim.siim1d import siim as siim1d                 # noqa: E402
from siim.siim2d import siim as siim2d                 # noqa: E402


def _s1d(**ov):
    """Tiny 1D run; nx (=21) != nt (=41) so U orientations are distinguishable."""
    p = dict(U=1e-3, P=1.0, beta=1e-2, Ko=1e-6, n=1, ce=1e-4, nu=2,
             sliding_law='eff-exp', zELA=600.0,
             L=2e4, xo=1e3, k_h=5, d=2, sigma=0.5, k=1,
             T=2e4, nt=41, dx=1000.0,
             left_bc='reflecting', right_bc='base_level',
             progress_bar=False)
    p.update(ov)
    return p


def _s2d(**ov):
    """Tiny 2D run, pure fluvial (zELA huge) so uplift alone drives the bed."""
    p = dict(U=1e-3, zELA=1e5, beta=1e-2, P=1, alpha_g=12, Ko=2e-6, n=1,
             ce=1e-4, nu=2, sliding_law='power', lambda_p=500, k=0.9,
             T=2e4, nt=21, nt_out=11, Lx=1e4, Ly=1e4, nx=21, ny=21, seed=3,
             boundary_status=['fixed_value'] * 4, initial_max_elevation=600,
             mode='B', flow_routing='single', progress_bar=False)
    p.update(ov)
    return p


# --- m14: unknown sliding_law is rejected (standalone GlacialLaw) ------------

def test_m14_unknown_sliding_law_raises():
    """siim2d falls through to derive_power for an unknown law at construction;
    the GlacialLaw kernel now raises on run() instead of silently running
    eff-exp (the 1D front end already raised)."""
    with pytest.raises(ValueError, match='sliding_law'):
        m = siim2d(_s2d(sliding_law='weertman'))
        m.run()


# --- m19: 1D U shape contract ------------------------------------------------

def test_m19_flat_nxnt_is_row_major_grid():
    nt, dx, L = 41, 1000.0, 2e4
    nx = int(L / dx) + 1
    U2d = 1e-3 + 1e-4 * np.arange(nx)[:, None] + 1e-5 * np.arange(nt)[None, :]
    grid = siim1d(_s1d(U=U2d))
    flat = siim1d(_s1d(U=U2d.ravel()))            # flat nx*nt -> (nx, nt) row-major
    assert np.array_equal(grid.U_matrix, U2d)
    assert np.array_equal(flat.U_matrix, U2d)     # reshaped, not stored 1-D
    grid.run(); flat.run()                        # no downstream IndexError
    assert np.array_equal(grid.z_out, flat.z_out)


def test_m19_transposed_U_rejected():
    nt, dx, L = 41, 1000.0, 2e4
    nx = int(L / dx) + 1
    U_bad = np.full((nt, nx), 1e-3)               # (nt, nx) — the wrong orientation
    with pytest.raises(ValueError, match='shape'):
        siim1d(_s1d(U=U_bad))


# --- m21: list forcing coerced like an array --------------------------------

def test_m21_list_forcing_equals_array():
    nt = 41
    arr = siim1d(_s1d(U=np.full(nt, 2e-3), zELA=np.full(nt, 600.0)))
    lst = siim1d(_s1d(U=[2e-3] * nt, zELA=[600.0] * nt))   # used to AttributeError
    arr.run(); lst.run()
    assert np.array_equal(arr.z_out, lst.z_out)
    assert np.array_equal(arr.zb_out, lst.zb_out)


# --- m22: 1D step-start forcing (series[0] consumed, series[nt-1] unused) -----

def test_m22_1d_step_start_consumption():
    nt = 41
    const = siim1d(_s1d(U=np.full(nt, 1e-3))); const.run()

    U_last = np.full(nt, 1e-3); U_last[-1] = 5e-3     # spike at the unused index
    last = siim1d(_s1d(U=U_last)); last.run()
    assert np.array_equal(const.zb_out, last.zb_out)  # series[nt-1] never consumed

    U_first = np.full(nt, 1e-3); U_first[0] = 5e-3    # spike at the first index
    first = siim1d(_s1d(U=U_first)); first.run()
    assert not np.array_equal(const.zb_out, first.zb_out)   # series[0] consumed


# --- m51 + m22: 2D length-nt U, step-start ------------------------------------

def test_m51_2d_lengthnt_U_equals_broadcast():
    nt, ny, nx = 21, 21, 21
    U_series = np.linspace(1e-3, 2e-3, nt)            # time-varying, spatially uniform
    U3d = np.broadcast_to(U_series[:, None, None], (nt, ny, nx)).copy()
    m1d = siim2d(_s2d(U=U_series)); m1d.run()
    m3d = siim2d(_s2d(U=U3d)); m3d.run()
    assert np.array_equal(m1d.zb_out, m3d.zb_out)     # (nt,) == (nt, ny, nx) broadcast

    # and a scalar == a constant (nt,) series (the per-step-scalar route)
    scalar = siim2d(_s2d(U=1e-3)); scalar.run()
    const = siim2d(_s2d(U=np.full(nt, 1e-3))); const.run()
    assert np.array_equal(scalar.zb_out, const.zb_out)


def test_m22_2d_step_start_consumption():
    nt = 21
    const = siim2d(_s2d(U=np.full(nt, 1e-3))); const.run()

    U_last = np.full(nt, 1e-3); U_last[-1] = 5e-3
    last = siim2d(_s2d(U=U_last)); last.run()
    assert np.array_equal(const.zb_out, last.zb_out)  # series[nt-1] unused (step-start)

    U_first = np.full(nt, 1e-3); U_first[0] = 5e-3
    first = siim2d(_s2d(U=U_first)); first.run()
    assert not np.array_equal(const.zb_out, first.zb_out)   # series[0] consumed


# --- m24: scalar boundary_status='fixed_value' handled ------------------------

def test_m24_scalar_boundary_status_initial_topo():
    scalar = siim2d(_s2d(boundary_status='fixed_value'))
    listed = siim2d(_s2d(boundary_status=['fixed_value'] * 4))
    ts = scalar._make_initial_topo()
    tl = listed._make_initial_topo()
    assert np.array_equal(ts, tl)                     # scalar behaves like the 4-list
    assert not np.allclose(ts, ts.flat[0])            # NOT the flat locked-high artifact
    assert ts[0, 0] < 1.0                             # fixed corner at the datum, not 1000
    assert ts.max() > 100.0                           # interior still high


# --- m54: single-sourced xo default ------------------------------------------

def test_m54_xo_single_sourced():
    from siim.analytical.steady_state import analytical_steady_state_solution as SS
    assert _C.XO == 300.0
    p = _s1d(); p.pop('xo')                            # use the default, not the fixture's
    assert siim1d(p).xo == _C.XO
    assert SS({'zELA': 1000.0}).xo == _C.XO
