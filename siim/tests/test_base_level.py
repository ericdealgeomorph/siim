"""Base-level BC: BL(t) water-line datum plumbing (docs/dev/boundary_conditions.md).

Gates for the ``bl`` (water-line datum) forcing:

* **BL=0 bit-for-bit** — with ``bl`` unset / 0 / a constant-zero series, outputs
  are identical across 3 laws x {1D, 2D SFR, 2D D-inf} (the plumbing is a no-op
  at the datum; the constant-series path matches the scalar path, as for P/zELA).
* **BL(t) knickpoint** — a step DROP in ``bl`` (mode A, pure fluvial, n=1)
  launches a knickpoint at the outlet that migrates upstream at the analytical
  stream-power celerity ``n*Ko*Qf^m*S^(n-1)`` (Whipple & Tucker 1999; n=1 ->
  slope-independent ``Ko*Qf^m``).
* **BL rise floods the erosion view** — raising ``bl`` above the surface
  suppresses all incision below the new datum (beds only uplift), in BOTH 1D and
  2D SFR, while the interior output still reports the true (unfloored) state.
* **kernel parity** — ``bl`` threads identically through the SFR and D-inf
  mode-B kernels (both agree bit-close at a nonzero ``bl``).
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

LAWS = ['eff-exp', 'power', 'coulomb']
_NT_1D = 301
_NT_2D = 61


def _p1d(**ov):
    """Small glaciated 1D profile (ice reaches the base-level border), mirroring
    test_precip_timeseries so the bl plumbing rides a live mode-B run."""
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
    """Strongly glaciated small 2D run (ice reaches the fixed-value borders)."""
    p = dict(U=1e-3, zELA=150, beta=1e-2, P=2, alpha_g=12, Ko=2e-6, n=1,
             ce=1e-4, nu=2, sliding_law='power', lambda_p=500, k=0.9,
             T=6e4, nt=_NT_2D, nt_out=13, Lx=2e4, Ly=2e4, nx=41, ny=41, seed=7,
             boundary_status=['fixed_value'] * 4, initial_max_elevation=800,
             mode='B', flow_routing='single', progress_bar=False)
    p.update(ov)
    return p


# --- BL=0 bit-for-bit ------------------------------------------------------

@pytest.mark.parametrize('law', LAWS)
def test_bl_zero_bit_for_bit_1d(law):
    """1D mode B: bl unset == explicit bl=0 == a constant-zero series, for all
    three laws (the plumbing is a no-op at the datum; scalar==series path)."""
    default = siim1d(_p1d(sliding_law=law)); default.run()
    explicit = siim1d(_p1d(sliding_law=law, bl=0.0)); explicit.run()
    series = siim1d(_p1d(sliding_law=law, bl=np.zeros(_NT_1D))); series.run()
    for other in (explicit, series):
        assert np.array_equal(default.z_out, other.z_out)
        assert np.array_equal(default.zb_out, other.zb_out)
        assert np.array_equal(default.H_out, other.H_out)


@pytest.mark.parametrize('routing', ['single', 'dinf'])
@pytest.mark.parametrize('law', LAWS)
def test_bl_zero_bit_for_bit_2d(routing, law):
    """2D SFR + D-inf, all three laws: bl unset == explicit 0 == zero series."""
    kw = dict(sliding_law=law, flow_routing=routing)
    default = siim2d(_p2d(**kw)); default.run()
    explicit = siim2d(_p2d(bl=0.0, **kw)); explicit.run()
    series = siim2d(_p2d(bl=np.zeros(_NT_2D), **kw)); series.run()
    for other in (explicit, series):
        assert np.array_equal(default.zb_out, other.zb_out)
        assert np.array_equal(default.H_out, other.H_out)
        assert np.array_equal(default.z_out, other.z_out)


# --- BL(t) knickpoint: migration at the stream-power celerity ---------------

def test_bl_knickpoint_celerity_mode_a_1d():
    """A step DROP in bl launches a knickpoint that migrates upstream at the
    detachment-limited celerity C = n*Ko*Qf^m*S^(n-1) (n=1 -> Ko*Qf^m). The
    half-amplitude front (the wave's characteristic point) is tracked over a
    mid-domain window and its travel time is compared to the celerity integral
    Int ds/C — pure fluvial (zELA huge -> no ice), mode A (bl anchors the
    outlet as the Dirichlet datum)."""
    nt, drop_step, dbl = 4001, 1000, -40.0
    bl = np.zeros(nt); bl[drop_step:] = dbl
    p = dict(U=1e-3, P=1.0, beta=1e-2, Ko=4e-6, n=1, m=0.5, nu=2, ce=1e-5,
             sliding_law='eff-exp', zELA=1e5, L=5e4, xo=1e3, k_h=5, d=2,
             sigma=0.5, k=1, T=1.2e7, nt=nt, nt_out=801, dx=100.0,
             left_bc='reflecting', right_bc='base_level', mode='A',
             progress_bar=False, bl=bl)
    m = siim1d(p); m.run()
    x, t, z, Qf = m.x, m.output_times, m.z_out, m.Qf_out   # x = distance from outlet

    t_drop = m.t[drop_step]
    pre = np.searchsorted(t, t_drop) - 1
    assert z[-1, pre] == pytest.approx(0.0)               # outlet at old datum
    assert z[-1, -1] == pytest.approx(dbl)                # outlet dropped to bl

    amp = np.abs(z[:, -1] - z[:, pre])                    # eventual change / node
    half = amp / 2.0

    def front_x(k):
        reached = (np.abs(z[:, k] - z[:, pre]) >= half) & (half > 0.5)
        return x[reached].max() if reached.any() else None

    # steady, purely-geometric Qf -> fixed celerity C(x) = Ko*Qf^m
    C = m.Ko * Qf[:, -1] ** m.m
    mid = [(t[k], front_x(k)) for k in range(pre + 1, len(t))]
    mid = [(tk, fx) for tk, fx in mid if fx is not None and 7e3 < fx < 4e4]
    assert len(mid) >= 5, "wave under-resolved in the mid-domain window"

    (t1, s1), (t2, s2) = mid[0], mid[-1]
    assert s2 > s1 + 1e4                                  # migrated upstream
    band = (x >= min(s1, s2)) & (x <= max(s1, s2)) & (C > 0)
    pred_dt = float(np.sum(m.dx / C[band]))              # Int ds/C over the window
    meas_dt = t2 - t1
    assert 0.8 < meas_dt / pred_dt < 1.25, (meas_dt, pred_dt)


# --- BL rise floods the erosion view: no erosion below the new datum ---------

def test_bl_rise_no_erosion_below_datum_1d():
    """1D pure fluvial: raising bl above the domain submerges the whole erosion
    view -> zero incision (beds only uplift); the bl=0 twin incises. True-state
    outputs everywhere: the outlet reports its true (unfloored) bed too
    (recovering toward bl at U, not floored to it), exactly like the interior."""
    def build(bl):
        return siim1d(dict(
            U=1e-3, P=1.0, beta=1e-2, Ko=1e-5, n=1, m=0.5, nu=2, ce=1e-5,
            sliding_law='eff-exp', zELA=1e5, L=5e4, xo=1e3, k_h=5, d=2,
            sigma=0.5, k=1, T=1e6, nt=501, dx=100.0, left_bc='reflecting',
            right_bc='base_level', mode='B', progress_bar=False, bl=bl))
    B = 2000.0
    base = build(0.0); base.run()
    flood = build(B); flood.run()
    zb0 = base.zb_out[:, 0]
    assert (base.zb_out[:, -1] - zb0).min() < -1.0                    # bl=0 incises
    assert ((flood.zb_out[:, -1] - zb0) >= -1e-6).all()              # no incision <= datum
    # true-state output: the outlet reports its true bed (below the raised bl),
    # unfloored — it recovers toward bl at U but is NOT pinned to it.
    assert flood.z_out[-1, -1] == flood.zb_out[-1, -1]               # border unfloored
    assert flood.z_out[-1, -1] < B                                   # true bed below the datum
    mid = flood.nx // 2                                               # interior stays true-state
    assert flood.z_out[mid, -1] < B - 100.0


def test_bl_rise_no_erosion_below_datum_2d_sfr():
    """2D SFR pure fluvial: same flood behavior as 1D (the 1D/2D-SFR parity of
    the datum floor) — raising bl above the domain suppresses all incision."""
    def build(bl):
        return siim2d(dict(
            U=1e-3, zELA=1e5, beta=1e-2, P=1, alpha_g=12, Ko=2e-6, n=1,
            ce=1e-4, nu=2, sliding_law='power', lambda_p=500, k=0.9, T=5e4,
            nt=51, nt_out=6, Lx=2e4, Ly=2e4, nx=31, ny=31, seed=3,
            boundary_status=['fixed_value'] * 4, initial_max_elevation=600,
            mode='B', flow_routing='single', progress_bar=False, bl=bl))
    base = build(0.0); base.run()
    flood = build(2000.0); flood.run()
    zb0 = base.zb_out[0]
    assert (base.zb_out[-1] - zb0).min() < -1.0                       # bl=0 incises
    assert ((flood.zb_out[-1] - zb0) >= -1e-6).all()                 # no incision <= datum


# --- kernel parity: bl threads identically through SFR and D-inf -------------

@pytest.mark.parametrize('law', LAWS)
def test_bl_kernel_parity_sfr_dinf_nonzero_bl(law):
    """Extend the cardinal-plane SFR/D-inf mode-B parity to a nonzero bl: with
    the datum threaded, both kernels still agree bit-close (rtol 1e-10)."""
    from siim._core.params import GlacialParams
    from siim._core.solvers import LAW_EFFEXP, LAW_POWER, LAW_COULOMB
    from siim._core.skeleton import (_glac_fast_solve_modeB_sfr,
                                     _glac_fast_solve_modeB_dinf)
    from .test_dinf_routing import _route_full, _edges_interior

    ny, nx, dx = 7, 20, 100.0
    nn = ny * nx
    _, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    zb0 = 5.0 + 0.01 * dx * ii.astype(float)          # drains -x to the col-0 border
    interior = _edges_interior(ny, nx)
    ice = np.full(nn, 1e7)                            # match the existing SFR/D-inf
    water = np.zeros(nn)                              # parity config (coulomb-safe)
    bbu = np.zeros(nn)
    cg, lam_p, lam_c, tau_c, rho_g_g = 2.3e-4, 300.0, 1e-3, 1.2e5, 9016.0
    Ko, Co, ce = 1e-6, 1e-9, 1e-5
    n, nu, m, mu, hc, alpha_g, clamp, dt = 1.0, 1.0, 0.5, 4.0/15.0, 1.5, 8.0, 1e-12, 500.0
    if law == 'power':
        code = LAW_POWER
        p = GlacialParams(Ko, 0.0, ce, n, nu, m, 0.0, cg, alpha_g, lam_p,
                          0.0, 0.0, 0.0, 0.0, hc, 0.0)
    elif law == 'coulomb':
        code = LAW_COULOMB
        p = GlacialParams(Ko, 0.0, ce, n, nu, m, 0.0, cg, alpha_g, 0.0,
                          lam_c, tau_c, clamp, rho_g_g, hc, 0.0)
    else:
        code = LAW_EFFEXP
        p = GlacialParams(Ko, Co, 0.0, n, nu, m, mu, cg, alpha_g, lam_p,
                          0.0, 0.0, 0.0, 0.0, hc, 0.0)

    rec_sfr = np.arange(nn, dtype=np.int64)
    lengths_sfr = np.zeros(nn)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            rec_sfr[j * nx + i] = j * nx + (i - 1)
            lengths_sfr[j * nx + i] = dx
    stack_sfr = np.argsort(ii.ravel(), kind='stable').astype(np.int64)
    _, receivers, weights, lengths, nb_rec, stack = _route_full(zb0, dx, dx, interior)

    def run(bl):
        zb_s = zb0.ravel().copy(); H_s = np.zeros(nn); surf_s = np.empty(nn)
        _glac_fast_solve_modeB_sfr(
            zb_s, ice, water, H_s, surf_s, code, p, dt,
            lengths_sfr, stack_sfr, rec_sfr, ny, nx, dx, dx, bbu, bl, True)
        zb_d = zb0.ravel().copy(); H_d = np.zeros(nn); surf_d = np.empty(nn)
        _glac_fast_solve_modeB_dinf(
            zb_d, ice, water, H_d, surf_d, code, p, dt,
            stack, nb_rec, receivers, weights, lengths, ny, nx, dx, dx, bbu,
            False, False, bl, True)
        return zb_s, H_s, surf_s, zb_d, H_d, surf_d

    # bl threads identically through the SFR and D-inf mode-B kernels: both agree
    # bit-close for the whole field, at bl = 0 and a nonzero bl. (bl's behavioral
    # bite — the flotation gate + the ice-free-border erosion-view floor — is
    # covered by the outflow-BC battery; on this all-icy grounded plane the
    # arrival-slope budget is bl-independent.)
    for bl in (0.0, -3.0):
        zb_s, H_s, surf_s, zb_d, H_d, surf_d = run(bl)
        np.testing.assert_allclose(zb_d, zb_s, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(H_d, H_s, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(surf_d, surf_s, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("bl_val", [0.0, 20.0])
def test_water_display_floors_at_bl_1d(bl_val):
    """The 1D elevation-panel water layer is a DISPLAY reconstruction
    (_water_display_1d): an ICE-FREE base-level outlet floors the waterline at
    the datum bl (its Dirichlet BC), NOT at the outlet's own (sub-datum) bed
    sill, and lake_fill propagates that level up the submerged basin. Regression
    for the plotter having pegged the waterline to the downstream zb sill."""
    m = siim1d(_p1d(bl=bl_val)); m.run()
    nx, i = m.nx, -1
    assert m.didx_r < m.nx and m.didx_l < 0        # node nx-1 is the base-level outlet
    assert m.bl_run[m.output_steps[i]] == bl_val

    # Craft a submerged, ice-free outlet basin: bed below bl, dipping to a
    # sub-datum sill at the outlet.
    zb = m.zb_out[:, i].copy(); H = m.H_out[:, i].copy()
    zb[nx-3:] = [bl_val - 40.0, bl_val - 60.0, bl_val - 80.0]
    H[nx-3:] = 0.0
    m.zb_out[:, i] = zb; m.H_out[:, i] = H
    m.z_out[:, i] = zb + m.hc_over_H * H

    zs_true, z_fill, wet = m.plot._water_display_1d(i)
    assert z_fill[nx-1] == pytest.approx(bl_val)   # outlet floored at bl, not the -80 sill
    assert (z_fill[nx-3:] >= bl_val - 1e-9).all()  # whole basin floored >= bl
    assert wet[nx-3:].all()                        # basin reads as wet (below bl)

    # An ICY (outflow) outlet keeps its true surface — NO floor, no spurious sea.
    H2 = m.H_out[:, i].copy(); H2[nx-1] = 500.0
    m.H_out[:, i] = H2; m.z_out[:, i] = m.zb_out[:, i] + m.hc_over_H * H2
    _, z_fill2, wet2 = m.plot._water_display_1d(i)
    assert z_fill2[nx-1] == pytest.approx(m.z_out[nx-1, i])   # true surface, not bl
    assert not wet2[nx-1]
