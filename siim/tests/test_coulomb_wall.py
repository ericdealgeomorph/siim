"""Regression tests for the coulomb H-closure at an adverse bed step (audit B7).

For a < 0 (overdeepening / adverse step) with low ice flux, the pole-far
initial guess fell below the domain wall H_min = -a, got clipped to the wall,
and the first Newton step from the wall was astronomical; the halving line
search then died and the solver returned the wall (an exactly flat ice surface
with zero glacial erosion for the step). The fix mirrors the eff-exp / power
siblings' wall-aware initial guess. These tests pin the solver against a brentq
oracle on the exact closure residual.
"""

import os
import sys

import numpy as np
import pytest
from scipy.optimize import brentq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from siim import constants as C           # noqa: E402
from siim._core.solvers import _modeb_closure_coulomb  # noqa: E402

# The finder's regime: coulomb defaults with alpha_g=10 (the default until
# 2026-07-02) pinned explicitly so the brentq-oracle numbers stay valid.
CG = C.cg_prefactor(alpha_g=10.0)
RHO_G_G = C.RHO_ICE_G
TAU_C = C.TAU_C
LAMBDA_C = C.LAMBDA_C
CLAMP = C.COULOMB_CLAMP
HC = 1.5


def _solver_resid(G, a, K_c, lam_c, beta):
    """Residual of the closure the solver actually solves (in G = hc*H space):
    G^5 (G+a)^3 (G + lam_c/(1 - phi^3)) = K_c, phi = beta G (G+a)."""
    phi = beta * G * (G + a)
    return G**5 * (G + a)**3 * (G + lam_c / (1.0 - phi**3)) - K_c


def _oracle_H(a, qi, L):
    """brentq root of the closure (independent of the Newton solver), in mean
    depth H = G/hc. Mirrors the rescaling _modeb_closure_coulomb applies."""
    hc6 = HC**6
    lam_cc = HC * LAMBDA_C
    K_c = qi * L**3 / CG
    beta = RHO_G_G / (TAU_C * L)
    Ks, bs = hc6 * K_c, beta / HC
    target = 1.0 - CLAMP
    H_safe = 0.5 * (-a + np.sqrt(a * a + 4.0 * target / bs))
    lo, hi = -a + 1e-9, H_safe * (1.0 - 1e-9)
    G = brentq(_solver_resid, lo, hi, args=(a, Ks, lam_cc, bs),
               xtol=1e-11, maxiter=300)
    return G / HC, (Ks, lam_cc, bs)


def test_coulomb_wall_clipped_matches_oracle():
    """The finder's confirmed case: a=-150, qi=1e4, L=50, hc=1.5. Without the
    fix the solver returns the wall H = -a/hc = 100.0; the true root is
    ~100.5210."""
    a, qi, L = -150.0, 1e4, 50.0
    H_true, _ = _oracle_H(a, qi, L)
    H_ret = _modeb_closure_coulomb(False, a, qi, L, HC, CG,
                                   LAMBDA_C, TAU_C, RHO_G_G, CLAMP)
    assert H_true == pytest.approx(100.5210, abs=1e-3)     # sanity vs the finding
    assert H_ret == pytest.approx(H_true, rel=1e-8)
    assert H_ret > -a / HC + 0.1, "solver stuck at the flat-surface wall"


@pytest.mark.parametrize("a", [-50.0, -80.0, -120.0, -150.0, -200.0])
@pytest.mark.parametrize("L", [25.0, 50.0, 100.0, 200.0])
@pytest.mark.parametrize("qi", [1e1, 1e2, 1e3, 1e4])
def test_coulomb_wall_clipped_sweep_residuals(a, L, qi):
    """A sweep of low-flux adverse-step (wall-clipping-prone) configs: the
    returned H must satisfy the closure residual, not sit at the wall."""
    H_true, (Ks, lam_cc, bs) = _oracle_H(a, qi, L)
    H_ret = _modeb_closure_coulomb(False, a, qi, L, HC, CG,
                                   LAMBDA_C, TAU_C, RHO_G_G, CLAMP)
    assert H_ret == pytest.approx(H_true, rel=1e-7)
    # |residual| at the returned root, in G space, relative to K_c.
    resid = _solver_resid(H_ret * HC, a, Ks, lam_cc, bs)
    assert abs(resid) <= 1e-6 * Ks
