"""The hillslope ADI on looped axes (audit #2, #26; ED-174): a looped axis has
no edge, so its first and last rows (columns) are neighbours across the seam
and the implicit systems are cyclic. Before, a looped edge was a no-flux wall."""
import numpy as np
import pytest

from siim._core.hillslope import _solve_cyclic_tridiagonal, diffuse
from siim.siim2d import _ibc_from_border_status

_LR_FIXED = _ibc_from_border_status(['fixed_value', 'fixed_value', 'looped', 'looped'])
_TB_FIXED = _ibc_from_border_status(['looped', 'looped', 'fixed_value', 'fixed_value'])
_ALL_FREE = _ibc_from_border_status(['looped'] * 4)


def test_cyclic_tridiagonal_matches_a_dense_solve():
    rng = np.random.default_rng(3)
    n = 9
    lower, upper = -rng.uniform(.1, .5, n), -rng.uniform(.1, .5, n)
    diag = 1.0 - lower - upper + rng.uniform(0., .2, n)
    rhs = rng.normal(size=n)
    A = np.diag(diag) + np.diag(lower[1:], -1) + np.diag(upper[:-1], 1)
    A[0, -1], A[-1, 0] = lower[0], upper[-1]
    got = np.empty(n)
    _solve_cyclic_tridiagonal(lower, diag, upper, rhs, got, n)
    np.testing.assert_allclose(got, np.linalg.solve(A, rhs), rtol=1e-13, atol=1e-13)


@pytest.mark.pin
@pytest.mark.parametrize('wrap', [(True, False), (False, True), (True, True)],
                         ids=['y', 'x', 'xy'])
def test_diffusion_commutes_with_a_seam_shift(wrap):
    """Rolling the surface k cells along a looped axis rolls the diffused
    surface k cells: the seam is invisible (it was a no-flux wall)."""
    rng = np.random.default_rng(5)
    ny, nx = 13, 16
    z = rng.uniform(0., 1000., (ny, nx))
    kd = rng.uniform(.5, 2., (ny, nx))
    ibc = _ALL_FREE if wrap == (True, True) else (_LR_FIXED if wrap[0] else _TB_FIXED)

    def run(z_, kd_):
        return diffuse(z_, kd_, 2e3, nx, ny, 15e3, 12e3, ibc, *wrap)

    base = run(z, kd)
    for axis, looped in enumerate(wrap):
        if looped:
            for k in (1, 5):
                np.testing.assert_allclose(run(np.roll(z, k, axis), np.roll(kd, k, axis)),
                                           np.roll(base, k, axis), rtol=0, atol=1e-9)


def test_periodic_axis_matches_the_dense_adi_and_conserves():
    """A surface that varies along the looped y axis only: the x pass leaves it
    alone (free x edges), so each column is the periodic 1-D ADI pair,
    z' = (I - h L)^-1 (I + h L) z with h = dt/2, exactly; and the column sums
    are conserved."""
    ny, nx, D, dt, dy = 11, 6, 1.5, 4e3, 1e3
    zy = np.sin(2 * np.pi * np.arange(ny) / ny) * 300 + np.arange(ny) % 3 * 40.
    z = np.repeat(zy[:, None], nx, axis=1)
    got = diffuse(z, D, dt, nx, ny, (nx - 1) * 1e3, (ny - 1) * dy, _ALL_FREE, True, False)
    a = D * 0.5 * dt / dy ** 2
    L = -2 * np.eye(ny) + np.roll(np.eye(ny), 1, 1) + np.roll(np.eye(ny), -1, 1)
    want = np.linalg.solve(np.eye(ny) - a * L, (np.eye(ny) + a * L) @ zy)
    np.testing.assert_array_equal(got[:, [0, -1]], z[:, [0, -1]])   # frozen x edges (fortran quirk)
    for c in range(1, nx - 1):
        np.testing.assert_allclose(got[:, c], want, rtol=0, atol=1e-9)
        assert abs(got[:, c].sum() - zy.sum()) < 1e-8
