"""The n = nu = 1 eroders, 1D and 2D: the closed form keeps the receiver-above
guard that every nonlinear solver applies (audit #7)."""
import numpy as np
import pytest

from siim._core.eroders import (_linear_erode_2d, _linear_erode_2d_dinf,
                                _nonlinear_erode_2d_dinf)
from siim.siim1d import _linear_erode


def test_sfr_linear_eroder_leaves_a_cell_below_its_receiver():
    """A pit at 10 m draining to a receiver at 20 m (mode A erodes the true
    surface on the filled graph) was raised to 15 m; the n = 1 solver leaves it."""
    z = np.array([20.0, 10.0])
    zo = z.copy()
    one = np.ones(2)
    _linear_erode_2d(z, zo, one, np.zeros(2), one, one, 0.5, 0.5,
                     np.array([0, 1]), np.array([0, 0]))
    assert z[1] == 10.0


def test_1d_linear_eroder_leaves_a_cell_below_its_receiver():
    """The 1D twin of the SFR case: the grounded branch erodes the raw surface,
    so a cell can sit below its receiver; it was raised from 10 m to 15 m."""
    z = np.array([20.0, 10.0, 30.0])
    _linear_erode(z, z.copy(), np.ones(3), np.zeros(3), 1.0, 1.0, 0.5, 0.5, 3, 1, 3)
    assert z[1] == 10.0


@pytest.mark.pin
def test_dinf_linear_eroder_matches_the_nonlinear_root():
    """On random cells with 1-8 receivers, some above the cell, the closed form
    returns the root of z - zo + sum_k G_k max(0, z - z_rk) = 0 that the n = 1
    Newton solver finds (the closed form summed every receiver, so one above
    the cell under-eroded it or deposited)."""
    rng = np.random.default_rng(0)
    dt, Ko, Co, m, mu = 1e3, 1e-3, 0.0, 0.5, 0.5
    n_above = 0
    for _ in range(300):
        k = int(rng.integers(1, 9))
        z = np.concatenate([[100.0], 100.0 + rng.uniform(-30.0, 30.0, k)])
        n_above += int((z[1:] > z[0]).any())
        receivers = np.zeros((k + 1, 8), dtype=np.int64)
        weights = np.zeros((k + 1, 8))
        lengths = np.zeros((k + 1, 8))
        receivers[0, :k] = np.arange(1, k + 1)
        weights[0, :k] = rng.dirichlet(np.ones(k))
        lengths[0, :k] = rng.uniform(50.0, 150.0, k)
        receivers[1:, 0] = np.arange(1, k + 1)       # receivers are their own outlets
        nb = np.ones(k + 1, dtype=np.int64)
        nb[0] = k
        stack = np.arange(k + 1, dtype=np.int64)
        Qf = np.full(k + 1, rng.uniform(1e2, 1e5))
        Qg = np.zeros(k + 1)
        z_lin, z_new = z.copy(), z.copy()
        _linear_erode_2d_dinf(z_lin, z.copy(), Qf, Qg, dt, Ko, Co, m, mu,
                              stack, nb, receivers, weights, lengths)
        _nonlinear_erode_2d_dinf(z_new, z.copy(), Qf, Qg, dt, Ko, Co, m, mu, 1.0, 1.0,
                                 stack, nb, receivers, weights, lengths,
                                 epsilon=1e-12, max_iter=200)
        assert z_lin[0] <= z[0]                     # never deposits
        np.testing.assert_allclose(z_lin[0], z_new[0], rtol=0.0, atol=1e-9)
    assert n_above > 100, "test setup: too few cells with a receiver above"
