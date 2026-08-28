"""Routing-surface EMA relaxation (the ``routing_relax`` flag).

Gates for the anti-flicker routing design: the mode-B/C
surf2erode routing + mass-balance surface is built from an EMA-relaxed lagged
thickness ``H_eff = r*H_eff_prev + (1-r)*H_lag`` instead of the raw lagged H,
damping the discrete-D8 per-step planview ice flicker (a cosmetic period-2
slosh). The relaxed H reaches ONLY the router graph + the accumulation surface;
the kernel/carve/outputs stay on raw state (the runaway firewall). r=0 (default)
is bit-for-bit with the pre-feature path (the full suite is the broader guard).
"""
import numpy as np
import pytest

from siim.siim2d import siim as siim2d


@pytest.fixture(autouse=True)
def _run_under_both_drivers(both_drivers):
    """S3 (Map 4 §1 PARAM): every test in this file runs under BOTH drivers --
    the conftest ``both_drivers`` fixture patches ``constants.DRIVER_DEFAULT``,
    so the existing assertions gate the in-house driver too."""



# ---------------------------------------------------------------------------
# Flicker-reduction regime: a small fluvial-warmed valley landscape glaciated
# to ~50% coverage, where near-ELA D8 receiver flips produce the period-2 ice
# flicker (lag-2 mask Jaccard > lag-1). Ko is raised so the warmup relief lands
# near the ELA on a fast grid.
# ---------------------------------------------------------------------------
_NX, _NY, _DX = 71, 41, 500.0
_FBASE = dict(
    U=1e-3, P=1, beta=.5e-2, n=1, ce=1e-5, nu=2, Ko=5e-6,
    sliding_law='coulomb', tau_c=1.2e5, lambda_c=1000,
    Lx=(_NX - 1) * _DX, Ly=(_NY - 1) * _DX, nx=_NX, ny=_NY,
    boundary_status=['fixed_value', 'fixed_value', 'looped', 'looped'],
    seed=109, initial_max_elevation=1500, mode='C', widening_rate=5,
    flow_routing='single', progress_bar=False)


@pytest.fixture(scope='module')
def _warm_topo():
    """Fluvial warmup (huge ELA -> no ice) to carve valley relief."""
    m = siim2d({**_FBASE, 'zELA': 1e6, 'T': 15e6, 'nt': 61, 'nt_out': 2})
    m.run()
    return m.z_out[-1]


def _jac(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 1.0


def _mask_jaccards(H, s0):
    """Median consecutive-step (lag 1) and lag-2 Jaccard of the H>50 m ice mask,
    over frames [s0:]."""
    m = H > 50.0
    j1 = [_jac(m[i], m[i + 1]) for i in range(s0, m.shape[0] - 1)]
    j2 = [_jac(m[i], m[i + 2]) for i in range(s0, m.shape[0] - 2)]
    return float(np.median(j1)), float(np.median(j2))


def _flick_run(_warm_topo, r, nt=161):
    p = dict(_FBASE, zELA=750.0, initial_topography=_warm_topo,
             T=(nt - 1) * 1000.0, nt=nt, nt_out=nt, routing_relax=r)
    m = siim2d(p)
    m.run()
    return np.asarray(m.H_out)


def test_flicker_reduction_headline(_warm_topo):
    """r=0.6 damps the planview flicker: the consecutive-step ice-mask Jaccard
    rises by a clear margin AND the r=0 period-2 signature (lag-2 J > lag-1 J)
    is gone (lag-1 >= lag-2 within epsilon)."""
    nt = 161
    s0 = nt // 3                                # discard the glacial-onset spin-up
    j1_0, j2_0 = _mask_jaccards(_flick_run(_warm_topo, 0.0, nt), s0)
    j1_6, j2_6 = _mask_jaccards(_flick_run(_warm_topo, 0.6, nt), s0)

    # Baseline actually flickers with the period-2 fingerprint (lag-2 > lag-1).
    assert j2_0 > j1_0 + 0.01, (
        f"r=0 baseline lacks the period-2 flicker: lag1={j1_0:.4f} lag2={j2_0:.4f}")
    # (1) Relaxation raises the consecutive-step Jaccard by a clear margin.
    assert j1_6 > j1_0 + 0.02, (
        f"routing_relax did not damp flicker: lag1 r=0 {j1_0:.4f} -> r=0.6 {j1_6:.4f}")
    # (2) The period-2 signature is absent at r=0.6 (lag-1 >= lag-2 within eps).
    assert j1_6 >= j2_6 - 0.02, (
        f"period-2 persists at r=0.6: lag1={j1_6:.4f} lag2={j2_6:.4f}")


def test_off_bit_for_bit():
    """routing_relax=0 is bit-for-bit with the plain mode-B + carve path (r=0
    uses the raw lagged H; the feature is a pure add, and the C alias resolves
    to B + carve). trunk_surface pinned off so the routing surface is the only
    thing compared. (The mode-C standard defaulting routing_relax=0.5 is pinned
    in test_mode_names.test_mode_C_standard_defaults_resolve.)"""
    cfg = dict(_FBASE, zELA=750.0, initial_max_elevation=900,
               T=6e5, nt=13, nt_out=4, trunk_surface=False, routing_relax=0.0)
    a = siim2d({**cfg, 'mode': 'C'}); a.run()
    b = siim2d({**cfg, 'mode': 'B', 'carve_width': True}); b.run()
    np.testing.assert_array_equal(a.H_out, b.H_out)
    np.testing.assert_array_equal(a.zb_out, b.zb_out)


# ---------------------------------------------------------------------------
# No-runaway on the harsh axis (dt=300 kyr, eta=5), both surf2erode providers
# (plain GlacialSurfaceToErode and the TrunkSurfaceToErode subclass): relaxation
# smooths the routing surface, it must not blow up the bed or H vs the r=0 twin.
# ---------------------------------------------------------------------------
_HNX, _HDX = 81, 250.0
_HBASE = dict(U=1e-3, P=1, beta=.5e-2, Ko=1e-6, n=1, ce=1e-5, nu=2,
              tau_c=1.2e5, lambda_c=1000, lambda_p=300.0, alpha_g=10,
              Lx=(_HNX - 1) * _HDX, Ly=(_HNX - 1) * _HDX, nx=_HNX, ny=_HNX,
              boundary_status=['fixed_value'] * 4, seed=109,
              flow_routing='single', progress_bar=False)


@pytest.fixture(scope='module')
def _warm_harsh():
    p = dict(_HBASE, zELA=1e6, sliding_law='power', mode='B', carve_width=False,
             initial_max_elevation=1000, T=30e6, nt=101, nt_out=2)
    m = siim2d(p); m.run()
    return m.z_out[-1]


@pytest.mark.parametrize("trunk_surface", [False, True])
def test_routing_relax_no_runaway(trunk_surface, _warm_harsh):
    z0 = _warm_harsh

    def run(r):
        p = dict(_HBASE, sliding_law='coulomb', initial_topography=z0,
                 zELA=float(np.quantile(z0, 0.50)), mode='C', widening_rate=5,
                 trunk_surface=trunk_surface, routing_relax=r,
                 T=9e6, nt=31, nt_out=31)
        m = siim2d(p); m.run()
        return m

    on, off = run(0.6), run(0.0)
    assert on.zb_out.min() >= off.zb_out.min() - 1.0, \
        f"routing_relax bed {on.zb_out.min():.0f} << r=0 {off.zb_out.min():.0f}"
    assert on.H_out.max() <= 5.0 * off.H_out.max()


# ---------------------------------------------------------------------------
# Composition + guards.
# ---------------------------------------------------------------------------
def test_compose_trunk_surface_conserves():
    """trunk_surface=True + routing_relax=0.6 runs and conserves mass (ice +
    water flux finite and non-negative everywhere)."""
    cfg = dict(_HBASE, sliding_law='power', lambda_p=300.0, zELA=500.0, mode='C',
               trunk_surface=True, routing_relax=0.6, initial_max_elevation=900,
               T=1.5e6, nt=16, nt_out=3)
    m = siim2d(cfg); m.run()
    ice = m.ds_out['glacial_flow__ice_flux'].values
    water = m.ds_out['glacial_flow__water_flux'].values
    assert np.isfinite(ice).all() and np.isfinite(water).all()
    assert (ice >= -1e-6).all() and (water >= -1e-6).all()


def test_routing_relax_guards():
    cfg = dict(_HBASE, sliding_law='power', lambda_p=300.0, zELA=800.0,
               initial_max_elevation=800, T=2e5, nt=3, nt_out=2)
    # mode A has no glacial surf2erode provider to relax.
    with pytest.raises(ValueError, match="routing_relax > 0 requires mode"):
        siim2d({**cfg, 'mode': 'A', 'routing_relax': 0.5})
    # r must be in [0, 1).
    with pytest.raises(ValueError, match=r"routing_relax must be in \[0, 1\)"):
        siim2d({**cfg, 'mode': 'C', 'routing_relax': 1.0})
    with pytest.raises(ValueError, match=r"routing_relax must be in \[0, 1\)"):
        siim2d({**cfg, 'mode': 'C', 'routing_relax': -0.1})
