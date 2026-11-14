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

from siim import constants as _C
from siim.siim1d import siim as siim1d
from siim.siim2d import siim as siim2d


@pytest.fixture(autouse=True)
def _run_under_both_drivers(both_drivers):
    """S3 (Map 4 §1 PARAM): every test in this file runs under BOTH drivers --
    the conftest ``both_drivers`` fixture patches ``constants.DRIVER_DEFAULT``,
    so the existing assertions gate the in-house driver too."""



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


def test_scalar_boundary_status_runs_as_the_four_list():
    """A scalar boundary_status is the 4-list everywhere, not only in the
    initial topography: the uplift mask and the driver's initial surface used
    to iterate the string's characters (audit #32)."""
    runs = []
    for bs in ('fixed_value', ['fixed_value'] * 4):
        m = siim2d(_s2d(boundary_status=bs))
        m.run()
        runs.append(m)
    assert runs[0].boundary_status == ['fixed_value'] * 4
    assert np.array_equal(runs[0].zb_out, runs[1].zb_out)


# --- m54: single-sourced xo default ------------------------------------------

def test_m54_xo_single_sourced():
    from siim.analytical.steady_state import analytical_steady_state_solution as SS
    assert _C.XO == 300.0
    p = _s1d(); p.pop('xo')                            # use the default, not the fixture's
    assert siim1d(p).xo == _C.XO
    assert SS({'zELA': 1000.0}).xo == _C.XO


@pytest.mark.parametrize('bs, match', [
    (['fixed'] * 4, 'sides must be one of'),                       # typo for fixed_value
    (['fixed_value', None, 'looped', 'looped'], 'sides must be one of'),
    (['looped', 'fixed_value', 'looped', 'looped'], "'looped' pairs"),
    (['fixed_value', 'fixed_value', 'looped', 'fixed_value'], "'looped' pairs"),
    ('core', "at least one 'fixed_value'"),
    (['looped'] * 4, "at least one 'fixed_value'"),
    (['fixed_value'] * 3, 'one value or four'),
])
def test_boundary_status_is_validated(bs, match):
    """A typo, an unpaired 'looped' side or a domain with no outlet used to run
    silently (nothing drained, or a half-periodic seam; audit #1, #15, #16)."""
    with pytest.raises(ValueError, match=match):
        siim2d(_s2d(boundary_status=bs))


def test_boundary_status_accepts_the_supported_sets():
    """Each supported form becomes the [left, right, bottom, top] list of str,
    in order (the wrap flags and the per-side bl read it by position)."""
    for bs, want in (('fixed_value', ['fixed_value'] * 4),
                     (['fixed_value', 'core', 'looped', 'looped'],
                      ['fixed_value', 'core', 'looped', 'looped']),
                     (('looped', 'looped', 'fixed_value', 'core'),
                      ['looped', 'looped', 'fixed_value', 'core'])):
        assert siim2d(_s2d(boundary_status=bs)).boundary_status == want


def test_a_mutated_input_array_does_not_change_the_run():
    """The model keeps its own copy of every input: U, P, bl and
    initial_topography aliased the caller's array, so mutating it after
    construction changed the run while m.U and the analytical stayed (audit #35)."""
    U = np.full((21, 21), 1e-3)
    ref = siim2d(_s2d(U=U.copy()))
    m = siim2d(_s2d(U=U))
    U[:] = 5e-3
    ref.run(); m.run()
    np.testing.assert_array_equal(m.zb_out, ref.zb_out)


def test_a_zt_series_pairs_with_the_p_series():
    """zELA(t) = zT(t) - P(t)/beta when both are series; the zT series used the
    time-mean P and gave a constant zELA (audit #34)."""
    nt = 21
    P = np.linspace(3.0, 1.0, nt)
    m = siim2d(_s2d(zELA=None, zT=np.full(nt, 1800.0), P=P))
    np.testing.assert_allclose(m._zELA_series, 1800.0 - P / 1e-2)
    assert m.zELA == pytest.approx(1500.0)            # the series' most glaciated step


def test_forcing_builders_follow_the_model_clock():
    """The builders built linspace(0, T, nt) while the models run linspace(to,
    T, nt), so a forcing for a run with to != 0 was stretched (audit #27); with
    to they land on the model's steps, and a cycle starts with the run."""
    from siim.forcing import ela_sawtooth, interp_forcing, uplift_step
    to, T, nt = 1e6, 3e6, 201
    model_t = np.linspace(to, T, nt)
    t, ela = ela_sawtooth(T, nt, period=1e5, to=to)
    np.testing.assert_array_equal(t, model_t)
    np.testing.assert_array_equal(ela, ela_sawtooth(T - to, nt, period=1e5)[1])
    t, U = uplift_step(T, nt, 1e-3, 2e-3, step_frac=0.5, to=to)
    assert U[t < 2e6].max() == 1e-3 and U[t >= 2e6].min() == 2e-3
    t, P = interp_forcing(T, nt, [to, T], [2.0, 1.0], to=to)
    np.testing.assert_allclose(P[[0, -1]], [2.0, 1.0])


def test_the_escarpment_wave_starts_at_x_escarpment_whatever_to():
    """The wave sat at x_escarpment + t*v on the absolute clock, so a run
    starting at to != 0 began with the wave already moved; it is measured from
    the run's start now, and the run is the to = 0 run shifted (audit #27)."""
    from siim.escarpment import siim_escarpment
    cfg = _s2d(uplift_type='wave', delta_h=800, wave_velocity=0.1,
               wave_width=1e4, x_escarpment=1e4, mode='A')
    a = siim_escarpment(dict(cfg, to=0.0, T=2e4)); a.run()
    b = siim_escarpment(dict(cfg, to=1e6, T=1e6 + 2e4)); b.run()
    assert a.U == b.U
    np.testing.assert_allclose(b.z_out, a.z_out, rtol=0, atol=1e-6)


def test_the_border_bed_follows_the_escarpment_wave():
    """Under the wave with no border_bed_uplift, a border bed rises at the wave's
    local rate each step; it rose at the run-average U meant for the analytical
    reference, about 8x the wave's peak at the escarpment defaults (audit #25).
    An explicit border_bed_uplift still wins."""
    from siim._core.step import wave_rate
    from siim.escarpment import siim_escarpment
    cfg = _s2d(uplift_type='wave', delta_h=800, wave_velocity=0.1, wave_width=2e3,
               x_escarpment=0.0, mode='C')
    m = siim_escarpment(cfg)
    # gate: the wave's border default is None (the adapter's border_bed_uplift None
    # takes the published rate, ED-178 #44), no longer a static 0 beside a flag
    assert m._make_border_bed_uplift() is None
    fn = m._driver_bbu_fn()
    x, y = np.linspace(0, m.Lx, m.grid_nx), np.linspace(0, m.Ly, m.grid_ny)
    for k in (0, 10):
        np.testing.assert_array_equal(fn(k, m.dt), wave_rate(
            x, y, m.dt, m.t[k] - m.t[0], 800, 2e3, 0.1, 0.0, 1.0, 0.0))
    assert fn(0, m.dt)[:, 0].max() > 10 * fn(0, m.dt)[:, -1].max()   # the wave starts at the left edge
    explicit = siim_escarpment(dict(cfg, border_bed_uplift=1e-4))
    assert explicit._driver_bbu_fn() is None and explicit._make_border_bed_uplift() == 1e-4
