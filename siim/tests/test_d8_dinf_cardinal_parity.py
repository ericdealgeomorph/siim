"""S4 gate — in-house D8 receiver == in-house D-inf primary receiver on a
cardinal-tilted plane.

The SFR/D-inf parity invariant (previously fortran-SFR vs in-house-D-inf,
``test_dinf_routing`` §cardinal), now BOTH in-house: on terrain draining along a
single cardinal direction, D-inf's steepest facet has flow angle 0, so it
degenerates to one receiver with weight 1 — the cardinal neighbour — which must
be exactly the D8 steepest-descent receiver. Framework-free (numpy/numba only)."""
import numpy as np
import pytest

from siim._core.step import route_d8, route_dinf

NY, NX = 9, 17
DX = DY = 100.0
BS = ['fixed_value', 'fixed_value', 'fixed_value', 'fixed_value']


def _interior():
    m = np.ones(NY * NX, dtype=bool)
    m[0::NX] = m[NX - 1::NX] = False
    m[:NX] = m[NX * (NY - 1):] = False
    return m


def _dinf_primary(surface):
    receivers, weights, _l, _nr, _stk, _b = route_dinf(surface, (NY, NX), DX, DY, BS)
    k = np.argmax(weights, axis=1)
    return receivers[np.arange(NY * NX), k]


@pytest.mark.parametrize("axis", ["x", "y"])
def test_d8_equals_dinf_primary_cardinal(axis):
    jj, ii = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij')
    if axis == "x":
        surface = (5.0 + 0.01 * DX * ii).astype(float)     # drains -x to col 0
    else:
        surface = (5.0 + 0.01 * DY * jj).astype(float)     # drains -y to row 0
    rec_d8, _w, _l, _nr, _stk, _b = route_d8(surface, (NY, NX), DX, DY, BS)
    rec_dinf = _dinf_primary(surface)
    interior = _interior()
    assert np.array_equal(rec_d8[interior], rec_dinf[interior]), (
        f"{axis}-tilt: D8 receiver != D-inf primary on "
        f"{int(np.sum(rec_d8[interior] != rec_dinf[interior]))} interior cell(s)")
    # sanity: the receiver is the correct cardinal neighbour (strictly downhill)
    z = surface.ravel()
    assert np.all(z[rec_d8[interior]] < z[np.arange(NY * NX)[interior]])
