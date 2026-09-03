"""
Regression / smoke tests for siim2d (and 1D <-> 2D parity).

Run from the repo root:
    pytest siim/tests -q

These are intentionally small/fast — sized to catch big regressions
(numerical blowup, missing variable plumbing, broken accumulator, BC
drift) rather than fine numerical drift. The whole siim/tests suite runs in ~15 s.
"""

import os
import sys
import numpy as np
import pytest

# Add the repo root (parent of the `siim` package) to sys.path so the test can
# `from siim.siim2d import siim` regardless of where pytest is invoked.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from siim.siim2d import siim as siim2d  # noqa: E402
from siim.siim1d import siim as siim1d  # noqa: E402
from siim.escarpment import siim_escarpment  # noqa: E402


@pytest.fixture(autouse=True)
def _run_under_both_drivers(both_drivers):
    """S3 (Map 4 §1 PARAM): every test in this file runs under BOTH drivers --
    the conftest ``both_drivers`` fixture patches ``constants.DRIVER_DEFAULT``,
    so the existing assertions gate the in-house driver too."""



# Shared physics — mirrors Eric's canonical good-params at smaller cost.
SHARED = dict(
    U=1e-3, P=2, beta=1e-3,
    Ko=1e-6, n=1,
    ce=1e-4, nu=2,
    Ac=2e-24, lambda_p=5e2,
    lambda_c=1e2,             # generous lambda_c -> coulomb runs at ~1.2x eff-exp
    alpha_g=8,
    sliding_law='eff-exp',
)


def _small_2d(**overrides):
    """Small-grid 2D params for fast tests (~5 s/run for eff-exp/power)."""
    return {**SHARED,
            'zELA': 1000,
            'T': 5e5,
            'Lx': 5e4, 'Ly': 5e4, 'nx': 31, 'ny': 31,
            'nt': 251, 'nt_out': 26,
            'D': 1e-3, 'seed': 111,
            'boundary_status': ['fixed_value'] * 4,
            'initial_max_elevation': 500, 'noise_amplitude': 10,
            'k': 1,
            'width_hack_k': 1.0, 'width_hack_p': 0.5,
            'flow_routing': 'single',
            'progress_bar': False,
            **overrides}


# ---------------------------------------------------------------------------
# 1. Smoke: every (sliding_law, flow_routing) combination runs without NaN.
#    Quinn MFR is currently disabled at the user-facing API, so we only smoke
#    the SFR and D-inf routings.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sliding_law", ["eff-exp", "power", "coulomb"])
@pytest.mark.parametrize("flow_routing", ["single", "dinf"])
def test_smoke_sfr_dinf_x_sliding_laws(sliding_law, flow_routing):
    m = siim2d(_small_2d(sliding_law=sliding_law, flow_routing=flow_routing))
    m.run()
    z = m.z_out[-1]
    H = m.H_out[-1]
    assert np.all(np.isfinite(z)), "non-finite values in final elevation"
    assert np.all(np.isfinite(H)), "non-finite values in final ice thickness"
    assert z.max() > 0, "final topography is degenerate (max z <= 0)"
    assert H.min() >= 0, "negative ice thickness"


def test_unknown_flow_routing_rejected():
    """flow_routing must be 'single' or 'dinf'; anything else raises."""
    with pytest.raises(ValueError, match="flow_routing must be"):
        siim2d(_small_2d(flow_routing='multiple'))


# ---------------------------------------------------------------------------
# 2. Base-level invariant: z stays exactly at IC value on fixed_value edges.
# ---------------------------------------------------------------------------

def test_base_level_invariant():
    m = siim2d(_small_2d())
    m.run()
    bs = m.boundary_status  # [left, right, bottom, top]
    for t in range(len(m.output_times)):
        z = m.z_out[t]
        if bs[0] == 'fixed_value':
            assert np.allclose(z[:,  0], 0, atol=1e-9), f"t={t}: left edge drifted"
        if bs[1] == 'fixed_value':
            assert np.allclose(z[:, -1], 0, atol=1e-9), f"t={t}: right edge drifted"
        if bs[2] == 'fixed_value':
            assert np.allclose(z[ 0, :], 0, atol=1e-9), f"t={t}: bottom edge drifted"
        if bs[3] == 'fixed_value':
            assert np.allclose(z[-1, :], 0, atol=1e-9), f"t={t}: top edge drifted"


# ---------------------------------------------------------------------------
# 3. SS snapshot: gross statistics at Eric's good-params (scaled down).
#    Bands are wide enough to allow ~10% numerical drift but tight enough to
#    catch a structural change (factor-of-2+ shift in mean z or max H).
# ---------------------------------------------------------------------------

def test_ss_snapshot():
    p = _small_2d(nx=51, ny=51, Lx=5e4, Ly=5e4, T=1e6, nt=501, nt_out=26)
    m = siim2d(p)
    m.run()
    z = m.z_out[-1]
    H = m.H_out[-1]
    mean_z = float(z.mean())
    max_z = float(z.max())
    max_H = float(H.max())
    # Wide sanity bands — meant to flag structural breakage, not catch drift.
    assert 100 < mean_z < 3000, f"mean_z={mean_z:.1f} outside [100, 3000]"
    assert 200 < max_z < 6000, f"max_z={max_z:.1f} outside [200, 6000]"
    assert 0 <= max_H < 2000, f"max_H={max_H:.1f} outside [0, 2000]"


# ---------------------------------------------------------------------------
# 4. 1D <-> 2D parity: channel z RMS should match within ~25%.
#    Two regimes:
#      - very high ELA  -> fluvial-only (no ice)
#      - ELA = 1000     -> mixed glacial-fluvial
#    Per Eric: 10-20% RMS is typical; 25% is the "really wrong" cutoff.
# ---------------------------------------------------------------------------

def _params_1d_from_2d_channel(p2d, ch):
    """Build 1D params from shared physics + the 2D channel's Hack fit.

    `ch.xo` is the cell-area-equivalent x for the head cell (computed inside
    extract_channel as (ar_ref[0]/k_h_fit)^(1/d_fit)). Using it instead of
    p2d['xo'] is what makes the 1D head BC's catchment flux match the 2D
    head cell's literal cell_area-driven flux.
    """
    keep = ('U', 'P', 'beta', 'Ko', 'n', 'ce', 'nu', 'Ac',
            'lambda_p', 'lambda_c', 'alpha_g', 'sliding_law', 'zELA', 'T', 'mode')
    p1d = {k: p2d[k] for k in keep if k in p2d}
    p1d.update({
        'L': float(ch.L),
        'xo': float(ch.xo),
        'k_h': float(ch.k_h),
        'd': float(ch.d),
        'sigma': 0.5,
        'k': 1,
        'nt': 2001,
        'dx': max(20.0, float(ch.L) / 1000.0),
        'left_bc': 'reflecting',
        'right_bc': 'base_level',
        'cap_ice_accumulation': False,
        'progress_bar': False,
    })
    return p1d


def _channel_rms_pct(z_2d_channel, m1d, ch):
    """RMS percent difference between the 2D channel z (head->outlet) and the
    1D z, evaluated on the channel's distance axis.

    1D conventions: `m1.x` is descending [L, ..., 0] (divide->outlet); `z_out`
    is shape (nx, nt) so the final-step profile is `z_out[:, -1]` with the
    same divide->outlet axis. To express in head->outlet form we use
    `m1.L - m1.x` (ascending [0, ..., L]); the corresponding z is unchanged
    because index 0 already holds the divide (== head) value.
    """
    x_head_to_outlet = m1d.L - m1d.x       # [0, dx, ..., L]
    z_head_to_outlet = m1d.z_out[:, -1]    # divide==head at index 0
    z1_at_ch = np.interp(ch.distance, x_head_to_outlet, z_head_to_outlet)
    rms = np.sqrt(np.mean((z_2d_channel - z1_at_ch) ** 2))
    denom = max(1e-9, float(np.mean(np.abs(z_2d_channel))))
    return 100.0 * rms / denom


def test_1d_2d_parity_fluvial():
    """High ELA -> no ice in the domain. 1D and 2D channel z should agree."""
    p2d = _small_2d(zELA=1e6, T=1e6, nt=501, nt_out=26, initial_max_elevation=1000)
    m2 = siim2d(p2d)
    m2.run()
    ch = m2.extract_channel(basin_rank=0)
    m1 = siim1d(_params_1d_from_2d_channel(p2d, ch))
    m1.run()
    rms = _channel_rms_pct(ch.z[-1], m1, ch)
    assert rms < 25.0, f"fluvial 1D-2D channel RMS = {rms:.1f}% (expected <25%)"


def test_1d_2d_parity_glacial():
    """ELA = 1000 -> glaciers form in the upper basin."""
    # Pin both sides to mode A: snapshot 1D/2D parity is the clean steady-state
    # comparison (mode B carries bed memory and can cycle — compare attractors,
    # not snapshots — and 2D sub-grid carving has no 1D analogue). mode='A' also
    # coerces 2D's now-default carving off, so the beds compare like-for-like.
    p2d = _small_2d(zELA=1000, T=1e6, nt=501, nt_out=26, initial_max_elevation=500,
                    mode='A')
    m2 = siim2d(p2d)
    m2.run()
    ch = m2.extract_channel(basin_rank=0)
    m1 = siim1d(_params_1d_from_2d_channel(p2d, ch))
    m1.run()
    rms = _channel_rms_pct(ch.z[-1], m1, ch)
    assert rms < 25.0, f"glacial 1D-2D channel RMS = {rms:.1f}% (expected <25%)"


def test_rms_vs_analytical_perfect_model_is_zero():
    """rms_vs_analytical uses the channel-floor datum on BOTH sides (audit B4):
    feeding the analytical SS back as the model output must give surface_rms and
    bed_rms == 0. The old mean-bed reconstruction (z - H vs the analytical
    z - HC*H) left a residual bed_rms = rms(0.5*H) under ice."""
    m = siim1d({'zELA': 1000.0, 'U': 1e-3, 'L': 3e4, 'nx': 151, 'T': 5e4,
                'nt': 101, 'sliding_law': 'eff-exp', 'progress_bar': False})
    a = m.analytical
    assert a.surface is not None, "config must have an analytical SS"
    m.run()
    a_H_grid = a.surface - a.bed                          # = HC_OVER_H * H
    a_surf = np.interp(m.x[::-1], a.x[::-1], a.surface[::-1])[::-1]
    a_H    = np.interp(m.x[::-1], a.x[::-1], a_H_grid[::-1])[::-1]
    assert (a_H > 0).any(), "config must have ice for the datum test to bite"
    # feed the analytical solution back as the last model snapshot
    m.z_out[:, -1] = a_surf
    m.H_out[:, -1] = a_H / m.hc_over_H
    surf_rms, bed_rms = m.rms_vs_analytical(-1)
    assert surf_rms < 1e-9, surf_rms
    assert bed_rms < 1e-9, bed_rms


@pytest.mark.adapter
def test_mu_override_recomputes_Co_across_front_ends():
    """An explicit mu override recomputes the eff-exp erosion prefactor Co with
    the effective mu on all three front ends (audit B5): 1D, analytical, and
    2D/GlacialLaw must agree. Pre-fix, 1D/analytical froze Co at the derived
    mu (=4*nu/15) while 2D used the user mu -> ~5% prefactor divergence."""
    pytest.importorskip('fastscape')
    from siim.analytical.steady_state import analytical_steady_state_solution
    from siim.fastscape.processes import GlacialLaw
    from siim import constants as C

    shared = dict(sliding_law='eff-exp', nu=2, mu=0.5, ce=1e-4,
                  Ac=2.5e-24, alpha_g=10.0, lambda_p=300.0)
    m1 = siim1d({**shared, 'progress_bar': False})
    a = analytical_steady_state_solution({**shared})
    gl = object.__new__(GlacialLaw)
    (gl.sliding_law, gl.nu, gl.mu, gl.ce, gl.Ac, gl.alpha_g, gl.lambda_p) = (
        'eff-exp', 2, 0.5, 1e-4, 2.5e-24, 10.0, 300.0)
    gl.lambda_c = None; gl.tau_c = C.TAU_C; gl.coulomb_clamp = C.COULOMB_CLAMP
    gl.H_diffusivity = None; gl.hc_over_H = C.HC_OVER_H
    gl.Ko = 1e-6; gl.n = 1; gl.m = None
    gl.initialize()

    Co_expected = C.Co_power(1e-4, C.cg_prefactor(10.0, 2.5e-24), 300.0, 10.0, 0.5)
    np.testing.assert_allclose(m1.Co, Co_expected, rtol=1e-12)
    np.testing.assert_allclose(a.Co, Co_expected, rtol=1e-12)
    np.testing.assert_allclose(gl.params[1].Co, Co_expected, rtol=1e-12)
    # and the stale (derived-mu) Co the fix retired is measurably different
    Co_stale = C.Co_power(1e-4, C.cg_prefactor(10.0, 2.5e-24), 300.0, 10.0,
                          4.0 * 2 / 15.0)
    assert abs(Co_stale - Co_expected) / Co_expected > 0.01


# ---------------------------------------------------------------------------
# 5. Escarpment subclass (siim_escarpment): wave uplift + plateau init ride on
#    top of siim2d through the _default_params / _process_overrides /
#    _forcing_input_vars seams. Default (block + sloped) mode must reproduce
#    the base model exactly.
# ---------------------------------------------------------------------------

def _small_esc(**overrides):
    """Smaller/faster grid for the escarpment smoke tests."""
    return _small_2d(nx=21, ny=21, T=3e5, nt=151, nt_out=11, **overrides)


def test_escarpment_defaults_match_base():
    """In default (block uplift + sloped init) mode, siim_escarpment must be
    byte-identical to base siim2d — i.e. the subclass adds nothing unless its
    switches are flipped."""
    p = _small_esc()
    me = siim_escarpment(p); me.run()
    mb = siim2d(p);          mb.run()
    assert np.array_equal(np.asarray(me.z_out), np.asarray(mb.z_out)), \
        "escarpment default mode diverged from base siim2d"


def test_escarpment_wave_plateau_smoke():
    """Plateau initial topography + moving-wave uplift runs without NaN."""
    m = siim_escarpment(_small_esc(
        init_type='plateau', plateau_zo=1500,
        uplift_type='wave', delta_h=800, wave_velocity=0.1,
        wave_width=1e4, x_escarpment=1e4,
    ))
    m.run()
    assert np.all(np.isfinite(m.z_out[-1])), "non-finite elevation"
    assert np.all(np.isfinite(m.H_out[-1])), "non-finite ice thickness"
    assert m.z_out[-1].max() > 0, "degenerate topography"


def test_plateau_edges_sit_on_their_datums():
    """The plateau ramp is rescaled so the fixed x-borders start EXACTLY on
    0 / zo - dz. The raw arctan only reaches 0/1 asymptotically (default
    frac/w on 50 km left the low edge at 0.25*zo — a sill above the border's
    water datum that never lowers), and `plateau_topography` hands back the
    matching right-side datum for plain siim2d runs."""
    from siim._core.step import plateau_profile
    from siim.escarpment import plateau_topography
    x = np.linspace(0, 50e3, 21)
    z = plateau_profile(x, 1500.0, 1.0, 0.8, 10e3)
    assert z[0] == 0.0 and z[-1] == pytest.approx(1499.0, abs=1e-9)
    assert np.all(np.diff(z[:8]) > 0), "ramp must rise monotonically across the escarpment"

    topo, bl = plateau_topography(21, 5, 50e3, zo=1500, frac=0.8, w=10e3, dz=1)
    assert topo.shape == (5, 21) and bl == {'right': 1499.0}
    assert np.array_equal(topo[0], z)

    # In a plain (ice-free) siim2d run both fixed borders stay on their datums.
    p = _small_esc(zELA=15000); p['nx'], p['ny'] = 21, 21
    p['initial_topography'], p['bl'] = plateau_topography(21, 21, p['Lx'], zo=1500)
    m = siim2d(p); m.run()
    assert np.all(m.z_out[-1][:, 0] == 0.0)
    assert np.all(m.z_out[-1][:, -1] == pytest.approx(1499.0))


@pytest.mark.parametrize("bad,match", [
    ({'uplift_type': 'wave'},   "requires delta_h"),
    ({'init_type': 'plateau'},  "requires plateau_zo"),
    ({'uplift_type': 'nope'},   "must be 'block' or 'wave'"),
    ({'init_type': 'nope'},     "must be 'sloped' or 'plateau'"),
])
def test_escarpment_param_validation(bad, match):
    with pytest.raises(ValueError, match=match):
        siim_escarpment(_small_esc(**bad))


# ---------------------------------------------------------------------------
# 6. Flexural isostasy (base-class feature): off by default, and when enabled
#    it actually feeds back into the steady landscape.
# ---------------------------------------------------------------------------

@pytest.mark.adapter
def test_flexure_off_by_default():
    # _process_overrides() builds the xsimlab slot dict (adapter assembly); the
    # standalone physics of flexure=False/True is covered by
    # test_denudation.test_flexure_acts_on_topography + test_standalone_no_fastscape.
    m = siim2d(_small_2d())
    assert m.flexure is False
    assert 'flexure' not in m._process_overrides()


@pytest.mark.adapter
def test_flexure_on_runs_and_acts():
    p = _small_2d(U=2e-3)
    off = siim2d(p);                                      off.run()
    on = siim2d({**p, 'flexure': True, 'e_thickness': 20e3}); on.run()
    assert 'flexure' in on._process_overrides()
    assert np.all(np.isfinite(on.z_out[-1])), "non-finite elevation with flexure on"
    # Flexure must measurably change the steady landscape vs. the no-flexure run.
    assert np.abs(on.z_out[-1] - off.z_out[-1]).max() > 1.0, \
        "flexure=True produced no change — not coupling into the topography"
