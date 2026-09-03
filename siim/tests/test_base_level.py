"""Base-level BC: BL(t) water-line datum plumbing.

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
import warnings

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from siim.siim1d import siim as siim1d            # noqa: E402
from siim.siim2d import siim as siim2d            # noqa: E402


@pytest.fixture(autouse=True)
def _run_under_both_drivers(both_drivers):
    """S3 (Map 4 §1 PARAM): every test in this file runs under BOTH drivers --
    the conftest ``both_drivers`` fixture patches ``constants.DRIVER_DEFAULT``,
    so the existing assertions gate the in-house driver too."""


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


# --- PER-SIDE bl: one water datum per fixed_value outlet --------------------

def _p2d_sides(**ov):
    """2D fixture with only the x-sides as base-level outlets (y looped), so a
    per-side ``bl`` dict has exactly two legal keys."""
    return _p2d(boundary_status=['fixed_value', 'fixed_value',
                                 'looped', 'looped'], **ov)


@pytest.mark.parametrize('routing', ['single', 'dinf'])
def test_bl_per_side_uniform_matches_scalar_2d(routing, both_drivers):
    """A per-side dict carrying the SAME datum on every fixed side is the
    scalar path bit-for-bit (the per-node datum is uniform, so every
    substituted value is equal), SFR and D-inf."""
    if both_drivers == 'xsimlab':
        pytest.skip("per-side bl is in-house-driver only (scalar adapter input)")
    kw = dict(flow_routing=routing)
    scalar = siim2d(_p2d_sides(bl=100.0, **kw)); scalar.run()
    sides = siim2d(_p2d_sides(bl={'left': 100.0, 'right': 100.0}, **kw))
    sides.run()
    assert np.array_equal(scalar.zb_out, sides.zb_out)
    assert np.array_equal(scalar.H_out, sides.H_out)
    assert np.array_equal(scalar.z_out, sides.z_out)


def test_bl_per_side_rejects_non_outlet_and_unknown_side():
    """Only 'fixed_value' sides are base-level outlets: a datum on a looped
    (or 'core') side is a contradiction, and an unknown key is a typo — both
    raise instead of silently doing nothing. A mis-shaped series reports its
    shape, and a 0-d array is just a scalar."""
    with pytest.raises(ValueError, match="boundary_status is 'looped'"):
        siim2d(_p2d_sides(bl={'bottom': 100.0}))
    with pytest.raises(ValueError, match='unknown bl side'):
        siim2d(_p2d_sides(bl={'north': 100.0}))
    with pytest.raises(ValueError, match=r'got shape \(2, 1\)'):
        siim2d(_p2d_sides(bl={'left': np.zeros((2, 1))}))
    with pytest.raises(ValueError, match=f'length-nt={_NT_2D}'):
        siim2d(_p2d_sides(bl={'left': np.zeros(_NT_2D - 1)}))
    zerod = siim2d(_p2d_sides(bl={'left': np.float64(50.0)}))
    assert zerod._bl_sides[0] == (50.0, None)


def test_bl_per_side_warns_that_the_analytical_is_not_offset_corrected():
    """With outlets at different data there is no single offset to add to the
    analytical reference, so the warning says so instead of promising an ~bl
    shift (self.bl stays the fixed-side mean, a label). It fires when the
    analytical overlay is drawn — NOT at construction, where it only nagged
    every batch run that never looks at the overlay."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        m = siim2d(_p2d_sides(bl={'left': 0.0, 'right': 300.0}))
    assert m.bl == 150.0
    with pytest.warns(UserWarning, match='CANNOT be offset-corrected'):
        m.plot._analytical_overlay(bistable=False)


def test_bl_nonzero_scalar_warns_only_when_the_overlay_is_drawn():
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        m = siim2d(_p2d_sides(bl=300.0))
    with pytest.warns(UserWarning, match='offset by ~bl'):
        m.plot._analytical_overlay(bistable=False)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        siim2d(_p2d_sides(bl=0.0)).plot._analytical_overlay(bistable=False)


def test_bl_per_side_is_2d_only():
    """The 1D profile has no domain sides — a dict is rejected with a directed
    message rather than a raw TypeError out of np.asarray."""
    with pytest.raises(ValueError, match='per-side bl is 2D only'):
        siim1d(_p1d(bl={'left': 100.0}))


def test_bl_per_side_ice_free_border_recovery_2d(both_drivers):
    """Each ice-free border recovers toward ITS OWN datum: with the right
    border rock at 900 m and bl={'right': 1000}, the right border rises at U
    and clamps at 1000 while the left border sits at its own datum 0. The
    scalar-0 twin leaves the right border at 900 (nothing to recover to)."""
    if both_drivers == 'xsimlab':
        pytest.skip("per-side bl is in-house-driver only (scalar adapter input)")
    ny, nx = 11, 21
    topo = np.linspace(0.0, 900.0, nx)[None, :] * np.ones((ny, 1))

    def build(bl):
        return siim2d(dict(
            U=1e-3, zELA=1e5, beta=1e-2, P=1, alpha_g=12, Ko=2e-6, n=1,
            ce=1e-4, nu=2, sliding_law='power', lambda_p=500, k=0.9, T=2e5,
            nt=51, nt_out=6, Lx=2e4, Ly=1e4, nx=nx, ny=ny, seed=3,
            noise_amplitude=0, initial_topography=topo, mode='B',
            boundary_status=['fixed_value', 'fixed_value', 'looped', 'looped'],
            flow_routing='single', progress_bar=False, bl=bl))

    sides = build({'right': 1000.0}); sides.run()
    control = build(0.0); control.run()
    assert sides.H_out.max() == 0.0                      # ice-free (fluvial)
    np.testing.assert_array_equal(sides.zb_out[-1][:, -1], 1000.0)  # clamped at bl
    np.testing.assert_array_equal(sides.zb_out[-1][:, 0], 0.0)      # own datum
    np.testing.assert_array_equal(control.zb_out[-1][:, -1], 900.0)  # no recovery


# Synthetic one-row SFR graph: two basins draining to opposite border outlets
# (nodes 0 and 8), the divide at node 4 (left basin 0-4, right basin 5-8).
_KERN_REC = np.array([0, 0, 1, 2, 3, 6, 7, 8, 8], dtype=np.int64)
_KERN_STACK = np.array([0, 1, 2, 3, 4, 8, 7, 6, 5], dtype=np.int64)   # outlets first
_KERN_HC = 1.5


def _kern_run(zb0, bl, dt, nstep):
    """Iterate the SFR mode-B kernel on the synthetic two-basin row. D_H = 0, so
    nothing couples the basins across the divide and the two halves are exactly
    independent."""
    from siim._core.params import GlacialParams
    from siim._core.solvers import LAW_EFFEXP
    from siim._core.skeleton import _glac_fast_solve_modeB_sfr

    n = len(zb0)
    lengths = np.full(n, 100.0); lengths[0] = 0.0; lengths[-1] = 0.0
    ice = np.full(n, 1e8)
    p = GlacialParams(1e-6, 5e-7, 0.0, 1.0, 1.0, 0.5, 4.0 / 15.0, 2.3e-4, 10.0,
                      300.0, 0.0, 0.0, 0.0, 0.0, _KERN_HC, 0.0)
    zb = zb0.copy(); H = np.zeros(n); surf = np.empty(n)
    for _ in range(nstep):
        _glac_fast_solve_modeB_sfr(zb, ice, np.zeros(n), H, surf, LAW_EFFEXP, p,
                                   dt, lengths, _KERN_STACK, _KERN_REC, 1, n,
                                   100.0, 100.0, np.zeros(n), bl, True, 0.1)
    return zb, H, surf


def test_bl_per_side_interior_datum_follows_basin_outlet():
    """Every interior node takes the datum of the OUTLET ITS BASIN DRAINS TO:
    on a two-basin row with bl = 0 at the left outlet and 800 at the right, the
    left basin is bit-for-bit the uniform-0 run and the right basin bit-for-bit
    the uniform-800 run (the two uniform runs differ on both halves, so neither
    equality is vacuous)."""
    zb0 = np.array([0.0, 60.0, 140.0, 220.0, 300.0, 300.0, 220.0, 140.0, 60.0])
    bl_sides = np.zeros(len(zb0)); bl_sides[-1] = 800.0   # interior = placeholder
    zb_lo, H_lo, _ = _kern_run(zb0, 0.0, 500.0, 1)
    zb_hi, H_hi, _ = _kern_run(zb0, 800.0, 500.0, 1)
    zb_ps, H_ps, _ = _kern_run(zb0, bl_sides, 500.0, 1)
    left, right = slice(0, 5), slice(5, 9)
    np.testing.assert_array_equal(zb_ps[left], zb_lo[left])
    np.testing.assert_array_equal(H_ps[left], H_lo[left])
    np.testing.assert_array_equal(zb_ps[right], zb_hi[right])
    np.testing.assert_array_equal(H_ps[right], H_hi[right])
    assert not np.array_equal(zb_lo[left], zb_hi[left])    # non-vacuous
    assert not np.array_equal(zb_lo[right], zb_hi[right])


def test_bl_per_side_icy_border_digs_to_its_own_draft():
    """An icy outflow border digs to ITS OWN flotation draft: with the right
    outlet's datum at 1000 the right border stops with its surface at/above
    1000 (zb ~ 1000 - hc*H), while the uniform-0 twin keeps digging ~1 km
    deeper toward -hc*H."""
    BR = 1000.0
    zb0 = np.array([0.0, 60.0, 140.0, 220.0, 300.0,
                    1300.0, 1240.0, 1180.0, 1100.0])
    bl_sides = np.zeros(len(zb0)); bl_sides[-1] = BR
    zb_ps, H_ps, surf_ps = _kern_run(zb0, bl_sides, 2e4, 1000)
    zb_lo, H_lo, surf_lo = _kern_run(zb0, 0.0, 2e4, 1000)
    assert zb_ps[-1] < zb0[-1] - 100.0                  # the border dug
    assert surf_ps[-1] >= BR                            # bounded at its draft
    assert surf_lo[-1] < BR - 500.0                     # uniform 0: far deeper


def test_bl_per_side_round_trips_through_save_load(tmp_path, both_drivers):
    """A per-side run reloads as itself: the dict rides in ``_user_params`` (like
    every other forcing), so ``load`` re-parses it into the same side table and
    the stored outputs come back unchanged. A series inside the dict round-trips
    too (compared entry-wise — an array in the tuple makes ``==`` ambiguous)."""
    if both_drivers == 'xsimlab':
        pytest.skip("per-side bl is in-house-driver only (scalar adapter input)")
    ramp = np.linspace(0.0, 300.0, _NT_2D)
    model = siim2d(_p2d_sides(bl={'left': 0.0, 'right': ramp}))
    model.run()
    loaded = siim2d.load(model.save(tmp_path / "per_side_bl.pkl"))
    for got, want in zip(loaded._bl_sides, model._bl_sides):
        assert (got is None) == (want is None)
        if want is None:
            continue
        assert got[0] == want[0]
        assert (got[1] is None and want[1] is None) or np.array_equal(got[1],
                                                                     want[1])
    assert np.array_equal(loaded.zb_out, model.zb_out)
    assert np.array_equal(loaded.H_out, model.H_out)
    assert model._bl_sides[1][1] is not None          # the series really rode along


def test_bl_per_side_corner_takes_x_side_value(both_drivers):
    """All four sides fixed, four distinct data: every border ring carries its
    own datum and a CORNER — touched by two fixed sides — takes the x-side
    (left/right) value. Checked on the driver's field builder and end-to-end
    through an ice-free run whose borders recover to (and clamp at) the datum
    they were handed."""
    if both_drivers == 'xsimlab':
        pytest.skip("per-side bl is in-house-driver only (scalar adapter input)")
    from siim._core.driver import _bl_field

    field = _bl_field([(10.0, None), (20.0, None), (30.0, None), (40.0, None)],
                      (5, 6), 0).reshape(5, 6)
    np.testing.assert_array_equal(field[:, 0], 10.0)      # left, corners included
    np.testing.assert_array_equal(field[:, -1], 20.0)     # right, corners included
    np.testing.assert_array_equal(field[0, 1:-1], 30.0)   # bottom, between corners
    np.testing.assert_array_equal(field[-1, 1:-1], 40.0)  # top, between corners
    np.testing.assert_array_equal(field[1:-1, 1:-1], 0.0)  # interior placeholder

    n = 21
    model = siim2d(dict(
        U=1e-3, zELA=1e5, beta=1e-2, P=1, alpha_g=12, Ko=2e-6, n=1, ce=1e-4,
        nu=2, sliding_law='power', lambda_p=500, k=0.9, T=6e5, nt=61, nt_out=4,
        Lx=1e4, Ly=1e4, nx=n, ny=n, seed=3, noise_amplitude=0,
        initial_topography=np.zeros((n, n)), mode='B', flow_routing='single',
        boundary_status=['fixed_value'] * 4, progress_bar=False,
        bl={'left': 100.0, 'right': 200.0, 'bottom': 300.0, 'top': 400.0}))
    model.run()
    zb = model.zb_out[-1]
    assert model.H_out.max() == 0.0                       # ice-free (fluvial)
    np.testing.assert_array_equal(zb[:, 0], 100.0)        # left ring + its corners
    np.testing.assert_array_equal(zb[:, -1], 200.0)       # right ring + its corners
    np.testing.assert_array_equal(zb[0, 1:-1], 300.0)     # bottom between corners
    np.testing.assert_array_equal(zb[-1, 1:-1], 400.0)    # top between corners


def _pyramid_sfr(ny=13, nx=13, slope=60.0):
    """Synthetic SFR pyramid: every interior cell steps toward its nearest
    border, so the four quadrants drain to the four different fixed sides."""
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    d = np.minimum(np.minimum(ii, nx - 1 - ii), np.minimum(jj, ny - 1 - jj))
    side = np.where(ii == d, 0, np.where(nx - 1 - ii == d, 1,
                                         np.where(jj == d, 2, 3)))
    rec = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx).copy()
    for j in range(ny):
        for i in range(nx):
            if d[j, i] == 0:
                continue                                   # border: self-receiving
            s = side[j, i]
            rec[j, i] = (j * nx + i - 1 if s == 0 else j * nx + i + 1 if s == 1
                         else (j - 1) * nx + i if s == 2 else (j + 1) * nx + i)
    stack = np.argsort(d.ravel(), kind='stable').astype(np.int64)  # outlets first
    lengths = np.where(d.ravel() == 0, 0.0, 100.0)
    return (slope * d).astype(float).ravel(), rec.ravel(), stack, lengths


def test_bl_per_side_interior_datum_covers_all_four_sides():
    """With all four sides fixed and four distinct data, each interior node is
    bit-for-bit the uniform run at ITS OWN side's datum — pinned on a pyramid
    whose quadrants drain to all four borders. The data are placed so that each
    side has interior nodes where its own arm differs from all three others (no
    quadrant passes by accident)."""
    from siim._core.driver import _bl_field
    from siim._core.params import GlacialParams
    from siim._core.solvers import LAW_EFFEXP
    from siim._core.skeleton import _glac_fast_solve_modeB_sfr

    ny = nx = 13
    zb0, rec, stack, lengths = _pyramid_sfr(ny, nx)
    nn = ny * nx
    ice = np.full(nn, 1e8)
    p = GlacialParams(1e-6, 5e-7, 0.0, 1.0, 1.0, 0.5, 4.0 / 15.0, 2.3e-4, 10.0,
                      300.0, 0.0, 0.0, 0.0, 0.0, _KERN_HC, 0.0)

    def run(bl):
        zb = zb0.copy(); H = np.zeros(nn); surf = np.empty(nn)
        _glac_fast_solve_modeB_sfr(zb, ice, np.zeros(nn), H, surf, LAW_EFFEXP,
                                   p, 500.0, lengths, stack, rec, ny, nx,
                                   100.0, 100.0, np.zeros(nn), bl, True, 0.1)
        return zb

    data = [0.0, 150.0, 290.0, 415.0]        # tuned to the pyramid's surfaces
    bl_flat = _bl_field([(d, None) for d in data], (ny, nx), 0)
    zb_ps = run(bl_flat)
    arms = [run(d) for d in data]
    # expected side per node: the datum its basin outlet carries (rec walk)
    expected = np.empty(nn, dtype=np.int64)
    for i in stack:
        expected[i] = (data.index(bl_flat[i]) if rec[i] == i
                       else expected[rec[i]])
    for q in range(4):
        sel = expected == q
        assert sel.any()                                   # all four sides used
        np.testing.assert_array_equal(zb_ps[sel], arms[q][sel])
        unique = [i for i in np.where(sel)[0]
                  if all(arms[q][i] != arms[o][i] for o in range(4) if o != q)]
        assert unique, f"side {q} never distinguishable — test is vacuous"


def test_bl_per_side_dinf_walk_follows_dominant_receiver():
    """The D-inf interior walk follows the DOMINANT (largest-weight) receiver,
    not ``receivers[:, 0]`` — that slot is _dinf_pack's cardinal facet, which
    carries the smaller share at a good fraction of two-receiver cells. The
    walk is replayed in Python on the routed surface and the kernel's per-node
    output is compared to the uniform run at the replayed datum; the config is
    checked to contain cells where a slot-0 walk would land on the other
    datum (so it would fail this test)."""
    from siim._core.params import GlacialParams
    from siim._core.solvers import LAW_EFFEXP
    from siim._core.skeleton import _glac_fast_solve_modeB_dinf
    from .test_dinf_routing import _route_full

    ny, nx, dx, BR = 21, 31, 100.0, 5000.0
    nn = ny * nx
    rng = np.random.default_rng(11)
    _jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    z = (100.0 - 100.0 * np.abs(ii - (nx - 1) / 2) / ((nx - 1) / 2)
         + 120.0 * rng.random((ny, nx)))         # rough tent draining both ways
    z[:, 0] = 0.0; z[:, -1] = 0.0
    interior = np.ones(nn, dtype=np.int8)
    interior[::nx] = 0; interior[nx - 1::nx] = 0    # x-sides are the outlets
    _zr, receivers, weights, lengths, nb_rec, stack = _route_full(
        z, dx, dx, interior, wrap_y=True)

    def replay(dominant):
        """Python twin of the kernel's step-0a walk."""
        bl = np.zeros(nn); bl[nx - 1::nx] = BR      # left 0, right BR
        for idx in range(nn - 1, -1, -1):           # stack is donor-first
            i = stack[idx]
            r = receivers[i, 0]
            if r == i:
                continue
            if dominant:
                w = weights[i, 0]
                for k in range(1, nb_rec[i]):
                    if weights[i, k] > w:
                        w, r = weights[i, k], receivers[i, k]
            bl[i] = bl[r]
        return bl

    expected = replay(True)
    slot0 = replay(False)
    ice = np.full(nn, 1e8)
    p = GlacialParams(1e-6, 5e-7, 0.0, 1.0, 1.0, 0.5, 4.0 / 15.0, 2.3e-4, 10.0,
                      300.0, 0.0, 0.0, 0.0, 0.0, _KERN_HC, 0.0)

    def run(bl):
        zb = z.ravel().copy(); H = np.zeros(nn); surf = np.empty(nn)
        _glac_fast_solve_modeB_dinf(zb, ice, np.zeros(nn), H, surf, LAW_EFFEXP,
                                    p, 500.0, stack, nb_rec, receivers, weights,
                                    lengths, ny, nx, dx, dx, np.zeros(nn),
                                    True, False, bl, True, 0.1)
        return zb

    bl_flat = np.zeros(nn); bl_flat[nx - 1::nx] = BR
    zb_ps, zb_lo, zb_hi = run(bl_flat), run(0.0), run(BR)
    lo, hi = expected == 0.0, expected == BR
    assert lo.any() and hi.any()
    np.testing.assert_array_equal(zb_ps[lo], zb_lo[lo])
    np.testing.assert_array_equal(zb_ps[hi], zb_hi[hi])
    assert (zb_lo[lo] != zb_hi[lo]).any()                 # non-vacuous
    assert (zb_lo[hi] != zb_hi[hi]).any()
    # the discriminating power: slot-0 would put these cells on the other arm
    mis = expected != slot0
    assert mis.any() and (zb_lo[mis] != zb_hi[mis]).all()
