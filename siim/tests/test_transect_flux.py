"""Tests for the EXPERIMENTAL transect closure flux (``closure_flux``,
branch closure-transect-flux; ``siim._core.step.transect_flux``).

Synthetic routing fields built with the in-house D8 router on planar surfaces,
so the crossing geometry is known exactly:

* a single trunk (cardinal or diagonal) with flux-free flanks sees its own Q;
* a uniform cardinal sheet sees Q*alpha_g*H/dx exactly (the one-cell linear
  ramp at the transect ends sums to exactly 2R on integer-spaced links);
* a uniform diagonal sheet sees Q*alpha_g*H/(dx/sqrt2) to a few percent;
* the min rule keeps thin flanks from borrowing a trunk's discharge;
* looped seams are invariant; the default ``'cell'`` is bit-for-bit.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from siim._core.step import route_d8, transect_flux
from siim.siim2d import siim as siim2d

ALPHA_G = 5.0
DX = 100.0
NY, NX = 41, 41
FIXED = ['fixed_value'] * 4


def _routing(z, boundary_status=FIXED, dx=DX, dy=DX):
    rec, w, lengths, nb, stack, basin = route_d8(z, z.shape, dx, dy, boundary_status)
    return rec, w, nb


def _x_slope():
    """Surface falling in +x (flow east, cardinal links)."""
    x = np.arange(NX) * DX
    return np.tile(-0.01 * x, (NY, 1))


def _diag_slope():
    """Surface falling along +x+y (flow to the SE diagonal neighbour)."""
    x = np.arange(NX) * DX
    y = np.arange(NY) * DX
    return -0.01 * (x[None, :] + y[:, None])


def _run(Q, H, z, boundary_status=FIXED, min_rule=False):
    rec, w, nb = _routing(z, boundary_status)
    return transect_flux(Q, H, rec, w, nb, z.shape, DX, DX, ALPHA_G,
                         boundary_status[2] == 'looped',
                         boundary_status[0] == 'looped', min_rule=min_rule)


def test_cardinal_trunk_sees_own_flux_exactly():
    z = _x_slope()
    Q = np.zeros((NY, NX)); H = np.zeros((NY, NX))
    j = NY // 2
    Q[j, 1:-1] = 1e6
    H[j, 1:-1] = 200.0                    # R = 500 m = 5 cells
    Qt = _run(Q, H, z)
    assert np.array_equal(Qt[j, 1:-1], Q[j, 1:-1])
    assert np.all(Qt[Q == 0] == 0.0)


def test_diagonal_trunk_sees_own_flux_exactly():
    z = _diag_slope()
    Q = np.zeros((NY, NX)); H = np.zeros((NY, NX))
    idx = np.arange(1, NY - 1)
    Q[idx, idx] = 1e6
    H[idx, idx] = 200.0
    Qt = _run(Q, H, z)
    assert np.array_equal(Qt[idx, idx], Q[idx, idx])
    assert np.all(Qt[Q == 0] == 0.0)


@pytest.mark.parametrize("H0", [100.0, 130.0, 170.0, 250.0])   # R/dx = 2.5, 3.25, 4.25, 6.25
def test_uniform_cardinal_sheet_is_exact_for_any_R(H0):
    z = _x_slope()
    Q = np.full((NY, NX), 1e5)
    H = np.full((NY, NX), H0)
    Qt = _run(Q, H, z)
    expect = 1e5 * ALPHA_G * H0 / DX        # q*2R with q = Q/dx
    # interior rows far enough from the fixed y-borders that the line is fully inside
    margin = int(np.ceil(ALPHA_G * H0 / 2 / DX)) + 3
    core = Qt[margin:-margin, 1:-1]
    assert np.allclose(core, expect, rtol=1e-12, atol=0.0)


def test_uniform_diagonal_sheet_matches_per_width_flux():
    z = _diag_slope()
    Q = np.full((NY, NX), 1e5)
    H = np.full((NY, NX), 200.0)           # R = 5 cells
    Qt = _run(Q, H, z)
    # diagonal flow: the cell's collecting width is dx/sqrt2, so q = Q*sqrt2/dx
    expect = 1e5 * np.sqrt(2.0) * ALPHA_G * 200.0 / DX
    m = 12
    core = Qt[m:-m, m:-m]
    assert np.allclose(core, expect, rtol=0.03)


def test_thin_ice_floors_at_own_flux_and_zero_stays_zero():
    z = _x_slope()
    Q = np.full((NY, NX), 1e5)
    H = np.full((NY, NX), 10.0)            # R = 25 m < dx/2 -> cell alone
    Qt = _run(Q, H, z)
    assert np.array_equal(Qt[1:-1, 1:-1], Q[1:-1, 1:-1])
    Q[5, 5] = 0.0
    Qt = _run(Q, H, z)
    assert Qt[5, 5] == 0.0
    assert np.all(Qt >= Q)


def test_min_rule_keeps_flanks_from_borrowing_the_trunk():
    z = _x_slope()
    Q = np.full((NY, NX), 1e3)             # thin flank flux everywhere
    H = np.full((NY, NX), 200.0)           # flank lines DO reach the trunk
    j = NY // 2
    Q[j, :] = 1e6
    plain = _run(Q, H, z, min_rule=False)
    capped = _run(Q, H, z, min_rule=True)
    flank = j + 2
    # plain: the flank sees the trunk's discharge; min rule: at most its own per neighbour
    assert plain[flank, 10] > 0.5e6
    assert capped[flank, 10] <= 1e3 * (2 * int(ALPHA_G * 200 / 2 / DX) + 2)
    # the trunk itself is unchanged to within its flanks' small contributions
    assert 1e6 <= capped[j, 10] <= plain[j, 10] <= 1e6 + 1e3 * 12


def test_looped_seam_invariance():
    """A uniform cardinal sheet under y-looped boundaries: rows at the seam see
    the same transect flux as interior rows."""
    z = _x_slope()
    Q = np.full((NY, NX), 1e5)
    H = np.full((NY, NX), 200.0)
    bs = ['fixed_value', 'fixed_value', 'looped', 'looped']
    Qt = _run(Q, H, z, boundary_status=bs)
    expect = 1e5 * ALPHA_G * 200.0 / DX
    assert np.allclose(Qt[:, 1:-1], expect, rtol=1e-12)


# ---------------------------------------------------------------------------
# model-level: default bit-for-bit, transect runs, validation
# ---------------------------------------------------------------------------
_NT, _NY, _NX = 11, 21, 21


def _cfg(**overrides):
    return {
        'P': 2, 'beta': 1e-3,
        'Ko': 1e-6, 'n': 1, 'ce': 1e-4, 'nu': 2,
        'Ac': 2e-24, 'lambda_p': 5e2, 'lambda_c': 1e2, 'alpha_g': 8,
        'sliding_law': 'eff-exp',
        'zELA': 1000, 'T': 1e5,
        'Lx': 3e4, 'Ly': 3e4, 'nx': _NX, 'ny': _NY,
        'nt': _NT, 'nt_out': 6,
        'D': 1e-3, 'seed': 7,
        'boundary_status': ['fixed_value'] * 4,
        'initial_max_elevation': 500, 'noise_amplitude': 10,
        'k': 1, 'width_hack_k': 1.0, 'width_hack_p': 0.5,
        'flow_routing': 'single', 'progress_bar': False, 'mode': 'B',
        **overrides,
    }


def test_default_closure_flux_resolves_per_mode_and_is_bit_for_bit():
    """Mode B/C default = constants.CLOSURE_FLUX ('transect_min'); mode A = 'cell'.
    The default must be bit-for-bit with the same value passed explicitly, and an
    explicit 'cell' must differ from the transect default (i.e. the switch bites)."""
    from siim import constants as C
    m0 = siim2d(_cfg()); m0.run()
    assert m0.closure_flux == C.CLOSURE_FLUX == 'transect_min'
    m1 = siim2d(_cfg(closure_flux='transect_min')); m1.run()
    assert np.array_equal(m0.z_out, m1.z_out)
    assert np.array_equal(m0.H_out, m1.H_out)
    ma = siim2d(_cfg(mode='A'))
    assert ma.closure_flux == 'cell'
    # The switch must bite where ice is wider than a cell: a small flat-bed
    # accumulation sheet (zELA far below the surface -> b = P everywhere).
    sheet = dict(zELA=-1e4, U=0.0, D=0.0, Lx=15e3, Ly=15e3, nx=31, ny=31,
                 T=2e4, nt=21, nt_out=3, initial_max_elevation=20, noise_amplitude=1)
    s0 = siim2d(_cfg(**sheet)); s0.run()
    s1 = siim2d(_cfg(**sheet, closure_flux='cell')); s1.run()
    assert s0.H_out[-1].max() > 0.0
    assert np.all(np.isfinite(s0.H_out))
    assert not np.array_equal(s0.H_out, s1.H_out)
    # (No "thicker" assertion: a short undamped sheet run is dominated by the
    # flow-concentration limit cycle, which the transect weakens — the thicker,
    # Vialov-matching dome is a property of the DAMPED fixed point, measured in
    # docs/dev/ice_sheet_dome_probes/dome_transect.py, not of 20 raw steps.)


@pytest.mark.parametrize("variant", ['transect', 'transect_min'])
@pytest.mark.parametrize("routing", ['single', 'dinf'])
def test_transect_runs_and_is_finite(variant, routing):
    m = siim2d(_cfg(closure_flux=variant, flow_routing=routing)); m.run()
    assert np.all(np.isfinite(m.z_out[-1]))
    assert np.all(np.isfinite(m.H_out[-1]))
    assert m.closure_flux == variant


def test_transect_validation():
    with pytest.raises(ValueError):
        siim2d(_cfg(closure_flux='bogus'))
    with pytest.raises(ValueError):
        siim2d(_cfg(closure_flux='transect', mode='A'))
