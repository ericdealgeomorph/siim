"""Time-varying precipitation P(t): a per-step P series, paralleling zELA / U.

Pins that a length-nt ``P`` array (Thea's use case — falling precipitation
co-varying with a falling ELA) drives both of P's roles per step — the
ice-accumulation cap (``B_cap``) and the fluvial water flux ``Qf`` — while a
*constant* series reproduces the scalar run bit-for-bit (the scalar path stays
untouched). The series reduces to its time-MEAN for the scalar analytical
reference (cf. the U convention; zELA uses its MIN). Covers 1D + 2D and the
``interp_forcing`` table-to-series builder.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from siim.siim1d import siim as siim1d            # noqa: E402
from siim.siim2d import siim as siim2d            # noqa: E402
from siim.forcing import interp_forcing           # noqa: E402


_NT_1D = 301
_NT_2D = 61


def _p1d(**ov):
    """Small glaciated 1D profile (eff-exp); cap_ice_accumulation=True so the
    P-cap on ice accumulation actually binds (exercises P's ice-cap role)."""
    p = dict(U=1e-3, P=2.0, beta=1e-2, Ko=1e-6, n=1, ce=1e-4, nu=2,
             Ac=2e-24, lambda_p=5e2, lambda_c=1e2, alpha_g=8,
             sliding_law='eff-exp', zELA=600.0,
             L=5e4, xo=1e3, k_h=5, d=2, sigma=0.5, k=1,
             T=3e5, nt=_NT_1D, dx=100.0,
             left_bc='reflecting', right_bc='base_level',
             cap_ice_accumulation=True, progress_bar=False)
    p.update(ov)
    return p


def _p2d(**ov):
    """Strongly glaciated small 2D run (mirrors test_denudation._glac), so both
    P roles — the ice cap and the fluvial Qf — are exercised."""
    p = dict(U=1e-3, zELA=150, beta=1e-2, P=2, alpha_g=12, Ko=2e-6, n=1,
             ce=1e-4, nu=2, sliding_law='power', lambda_p=500, k=0.9,
             T=6e4, nt=_NT_2D, nt_out=13, Lx=2e4, Ly=2e4, nx=41, ny=41, seed=7,
             boundary_status=['fixed_value'] * 4, initial_max_elevation=800,
             mode='B', flow_routing='single', progress_bar=False)
    p.update(ov)
    return p


def test_1d_precip_series():
    """1D: a constant series ≡ scalar bit-for-bit; a falling series is stored
    per-step in P_run, reduces to its time-mean for the reference, and bites."""
    scalar = siim1d(_p1d()); scalar.run()
    const = siim1d(_p1d(P=np.full(_NT_1D, 2.0))); const.run()
    assert np.array_equal(scalar.z_out, const.z_out)     # plumbing leaves scalar path intact
    assert np.array_equal(scalar.Qf_out, const.Qf_out)

    P = np.linspace(2.5, 0.6, _NT_1D)                    # falling precip (Thea's direction)
    ramp = siim1d(_p1d(P=P)); ramp.run()
    assert np.array_equal(ramp.P_run, P)                 # exact per-step series
    assert ramp.P == pytest.approx(P.mean())             # scalar reference = time-mean
    assert np.all(np.isfinite(ramp.z_out))
    assert not np.array_equal(ramp.z_out, scalar.z_out)  # time variation reaches the output


def test_2d_precip_series():
    """2D: same contract — constant series ≡ scalar (z, zb, Qf bit-for-bit), a
    falling series is stored in _P_series, reduces to its mean, and bites."""
    scalar = siim2d(_p2d()); scalar.run()
    const = siim2d(_p2d(P=np.full(_NT_2D, 2.0))); const.run()
    assert np.array_equal(scalar.z_out, const.z_out)
    assert np.array_equal(scalar.zb_out, const.zb_out)
    assert np.array_equal(scalar.Qf_out, const.Qf_out)

    P = np.linspace(3.0, 0.5, _NT_2D)
    ramp = siim2d(_p2d(P=P)); ramp.run()
    assert np.array_equal(ramp._P_series, P)
    assert ramp.P == pytest.approx(P.mean())
    assert np.all(np.isfinite(ramp.z_out))
    assert not np.array_equal(ramp.z_out, scalar.z_out)


def test_1d_zT_precip_series_zELA_emerges():
    """1D audit F2: with zT given (zELA=None) and P a series, the per-step ELA
    emerges as zELA_run = zT − P(t)/β (paper semantics), while the scalar
    self.zELA keeps the time-MEAN for the analytical reference. A constant-P
    series reproduces the scalar-P run bit-for-bit."""
    beta, zT, nt = 1e-2, 800.0, _NT_1D
    P = np.linspace(2.5, 0.6, nt)
    m = siim1d(_p1d(zELA=None, zT=zT, P=P))
    np.testing.assert_array_equal(m.zELA_run, zT - P / beta)      # time-varying, exact
    assert m.zELA == pytest.approx(zT - P.mean() / beta)          # scalar ref = time-mean
    # a constant-P series with zT is identical to the scalar-P run with zT
    sc = siim1d(_p1d(zELA=None, zT=zT, P=2.0)); sc.run()
    cn = siim1d(_p1d(zELA=None, zT=zT, P=np.full(nt, 2.0))); cn.run()
    assert np.array_equal(sc.z_out, cn.z_out)
    np.testing.assert_array_equal(sc.zELA_run, np.full(nt, zT - 2.0 / beta))


def test_2d_zT_precip_series_zELA_emerges():
    """2D audit F2: zT + P-series wires a time-varying zELA onto the clock
    (_zELA_series = zT − P(t)/β); the scalar self.zELA keeps the time-mean. A
    constant-P series reproduces the scalar-P run bit-for-bit (scalar P leaves
    the ELA constant, _zELA_series None)."""
    beta, zT, nt = 1e-2, 400.0, _NT_2D
    P = np.linspace(3.0, 0.5, nt)
    m = siim2d(_p2d(zELA=None, zT=zT, P=P))
    np.testing.assert_array_equal(m._zELA_series, zT - P / beta)
    assert m.zELA == pytest.approx(zT - P.mean() / beta)
    sc = siim2d(_p2d(zELA=None, zT=zT, P=2.0)); sc.run()
    cn = siim2d(_p2d(zELA=None, zT=zT, P=np.full(nt, 2.0))); cn.run()
    assert np.array_equal(sc.z_out, cn.z_out)
    assert sc._zELA_series is None                    # scalar P -> constant ELA


def test_wrong_length_series_raises():
    """A P series whose length != nt is rejected at construction (1D and 2D)."""
    with pytest.raises(ValueError):
        siim1d(_p1d(P=np.full(_NT_1D - 5, 2.0)))
    with pytest.raises(ValueError):
        siim2d(_p2d(P=np.full(_NT_2D - 3, 2.0)))


def test_interp_forcing():
    """The table-to-series builder reproduces node values and holds endpoints
    flat outside the node range."""
    T, nt = 3e6, 600
    t, s = interp_forcing(T, nt, times=[0, 1.5e6, 3e6], values=[2.0, 1.2, 0.65])
    assert len(t) == nt and len(s) == nt
    assert s[0] == pytest.approx(2.0)                                  # first node
    assert s[-1] == pytest.approx(0.65)                               # last node
    assert float(np.interp(1.5e6, t, s)) == pytest.approx(1.2, rel=1e-3)  # interior node

    # flat extrapolation below times[0] and above times[-1]
    _, s2 = interp_forcing(4e6, 5, times=[1e6, 2e6], values=[10.0, 20.0])
    assert s2[0] == pytest.approx(10.0)
    assert s2[-1] == pytest.approx(20.0)
