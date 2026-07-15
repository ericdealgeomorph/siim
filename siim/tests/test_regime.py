"""Tests for siim.analytical.regime (nondimensional regime map).

The regime map must agree, point by point, with the dimensional profile
classes — same closure, two independent implementations — and its
saddle-node boundary must be a genuine fold of the closure (checked by
finite differences and by root counting, not by re-using the trace's own
derivative formula).
"""
import numpy as np
import pytest

from siim.analytical import GeneralProfile, MarginalCoulombProfile, RegimeMap

# Exponent sets exercised throughout: the marginal-Coulomb case of the
# theory paper's figures 1-3, and the general (sub-marginal) case of
# figures 4-5.
MARGINAL = dict(d=2.0, sigma=0.45, phi=0.5, theta=0.5, k=0.6, xo=1e-3)
GENERAL = dict(d=1.8, sigma=0.45, phi=0.3, theta=0.45, k=1.0, xo=1e-3)

L_DIM = 5.0e4   # m, arbitrary dimensional anchor
KS_DIM = 300.0  # fluvial steepness anchor


def _profile_from_nondim(rm, kappa, Y, L=L_DIM, ks=KS_DIM):
    """Map a nondim (kappa, Y) point onto a dimensional GeneralProfile."""
    zfo = ks * L ** (1.0 - rm.dtheta) * rm.F_xo
    cs = kappa * ks * L ** (1.0 - rm.dtheta - rm.r)
    return GeneralProfile(ks=ks, cs=cs, zELA=Y * zfo, L=L, xo=rm.xo * L,
                          d=rm.d, sigma=rm.sigma, phi=rm.phi,
                          theta=rm.theta, k=rm.k)


def _stable_glaciated(profile):
    """Largest-Lt stable glaciated solution, or None."""
    sols = [s for s in profile.solutions
            if s.regime != 'fluvial' and s.stable]
    return max(sols, key=lambda s: s.Lt) if sols else None


# ---------------------------------------------------------------------------
# Shape functions
# ---------------------------------------------------------------------------

def test_F_closed_forms():
    rm = RegimeMap(**GENERAL)             # d*theta = 0.81
    u = np.array([1e-6, 1e-3, 0.1, 0.5, 0.99])
    e = 1.0 - rm.dtheta
    np.testing.assert_allclose(rm.F(u), (1.0 - u ** e) / e, rtol=1e-13)

    rm1 = RegimeMap(**MARGINAL)           # d*theta = 1 exactly
    np.testing.assert_allclose(rm1.F(u), np.log(1.0 / u), rtol=1e-13)

    # the expm1 form reaches the log limit continuously
    rm_eps = RegimeMap(**{**MARGINAL, 'theta': 0.5 + 1e-13})
    np.testing.assert_allclose(rm_eps.F(u), np.log(1.0 / u), rtol=1e-9)


def test_G_marginal_is_arcosh():
    # At d*phi = 1, G(u) = (2/k) arcosh(u^(-k/2)) — the b = 0 kernel branch.
    rm = RegimeMap(**MARGINAL)
    u = np.array([1e-6, 1e-3, 0.1, 0.5, 0.999])
    expected = (2.0 / rm.k) * np.arccosh(u ** (-rm.k / 2.0))
    np.testing.assert_allclose(rm.G(u), expected, rtol=1e-10)


def test_closure_derivative_matches_finite_difference():
    # The analytic dY/dLt (which the saddle-node trace is built from) must
    # match a centered difference of closure_Y on both branches.
    for params in (MARGINAL, GENERAL):
        rm = RegimeMap(**params)
        for kappa in (0.3, 0.9, 2.0):
            for Lt in (0.01, 0.2, 0.8, 1.5, 5.0):
                h = Lt * 1e-6
                fd = (rm.closure_Y(Lt + h, kappa)
                      - rm.closure_Y(Lt - h, kappa)) / (2 * h)
                an = rm._dclosure_dLt(Lt, kappa)
                assert np.isclose(an, fd, rtol=1e-4), (params, kappa, Lt)


# ---------------------------------------------------------------------------
# Agreement with the dimensional profile classes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('params', [MARGINAL, GENERAL],
                         ids=['marginal', 'general'])
def test_pointwise_agreement_with_general_profile(params):
    rm = RegimeMap(**params)
    kappas = [0.2, 0.6, 0.9 * rm.kappa_c, 1.2 * rm.kappa_c]
    Ys = [0.25, 0.6, 0.9, 1.05]
    n_glaciated = 0
    n_fluvial = 0
    for kappa in kappas:
        for Y in Ys:
            p = _profile_from_nondim(rm, kappa, Y)
            assert np.isclose(p.kappa, kappa, rtol=1e-12)
            ref = _stable_glaciated(p)
            Lt_nd = float(rm.Lt(kappa, Y))
            if ref is None:
                assert np.isnan(Lt_nd), (kappa, Y, Lt_nd)
                assert float(rm.zo(kappa, Y)) == 1.0
                n_fluvial += 1
                continue
            if ref.Lt / L_DIM > 0.8 * rm.Lt_max:
                continue                      # beyond the map's trace window
            assert np.isclose(Lt_nd, ref.Lt / L_DIM, rtol=1e-8), (kappa, Y)
            zo_nd = float(rm.zo(kappa, Y))
            assert np.isclose(zo_nd, ref.zo / p.zfo, rtol=1e-8), (kappa, Y)
            zt_nd = float(rm.zt(kappa, Y))
            assert np.isclose(zt_nd, ref.zt / p.zfo,
                              rtol=1e-7, atol=1e-9), (kappa, Y)
            fluv, mixed, glac = rm.masks(kappa, Y)
            assert bool(glac) == (ref.regime == 'glacial'), (kappa, Y)
            assert bool(mixed) == (ref.regime == 'mixed'), (kappa, Y)
            n_glaciated += 1
    # the grid must have exercised both glaciated and fluvial outcomes
    assert n_glaciated >= 6
    assert n_fluvial >= 2


def test_marginal_oracle_against_mcp():
    # In the marginal case kappa = cs/ks and zfo = ks log(L/xo): the regime
    # map must match MarginalCoulombProfile's exact arcosh solution.
    rm = RegimeMap(**MARGINAL)
    ks = 500.0
    for kappa, Y in [(0.1, 0.5), (0.5, 0.8), (0.5, 0.3), (1.2, 0.4)]:
        zfo = ks * np.log(1.0 / rm.xo)
        p = MarginalCoulombProfile(ks=ks, cs=kappa * ks, zELA=Y * zfo,
                                   L=L_DIM, sigma=rm.sigma, k=rm.k,
                                   xo=rm.xo * L_DIM)
        sols = [s for s in p.solutions if s.regime != 'fluvial' and s.stable]
        Lt_nd = float(rm.Lt(kappa, Y))
        if not sols:
            assert np.isnan(Lt_nd)
            continue
        ref = max(sols, key=lambda s: s.Lt)
        assert np.isclose(Lt_nd, ref.Lt / L_DIM, rtol=1e-7), (kappa, Y)
        assert np.isclose(float(rm.zo(kappa, Y)), ref.zo / p.zfo,
                          rtol=1e-7), (kappa, Y)


# ---------------------------------------------------------------------------
# Saddle-node boundary
# ---------------------------------------------------------------------------

def test_saddle_node_marginal_closed_form():
    # Marginal Coulomb: the fold satisfies kappa/kappa_c = sqrt(1 - 1/v^2),
    # v = (Lt/xo)^(k/2) — the closed-form arcosh saddle condition.
    rm = RegimeMap(**MARGINAL)
    b = rm.saddle_node(n_samples=200)
    v = (b.Lt[1:] / rm.xo) ** (rm.k / 2.0)
    expected = rm.kappa_c * np.sqrt(np.clip(1.0 - 1.0 / v ** 2, 0.0, None))
    np.testing.assert_allclose(b.kappa[1:], expected, rtol=1e-9)
    # exact corner prepended
    assert b.kappa[0] == 0.0 and b.Y[0] == 1.0 and b.zo[0] == 1.0


@pytest.mark.parametrize('params', [MARGINAL, GENERAL],
                         ids=['marginal', 'general'])
def test_saddle_node_is_stationary_and_on_closure(params):
    # Non-circular fold check: at trace points, (1) the (kappa, Y) pair sits
    # ON the closure at the trace's Lt, and (2) the FINITE-DIFFERENCE
    # dY/dLt of the closure vanishes there (relative to its scale h away).
    # Sampled away from the Lt -> xo corner, where curvature makes any
    # finite difference meaningless (the analytic cancellation there is
    # exercised separately by the closed-form marginal test).
    rm = RegimeMap(**params)
    b = rm.saddle_node(n_samples=300)
    usable = np.flatnonzero(np.isfinite(b.kappa) & (b.kappa > 0)
                            & (b.Lt > 10 * rm.xo) & (b.Lt < 0.9))
    assert usable.size >= 10
    for i in usable[np.linspace(0, usable.size - 1, 7).astype(int)]:
        Lt, kappa, Y = b.Lt[i], b.kappa[i], b.Y[i]
        assert np.isclose(float(rm.closure_Y(Lt, kappa)), Y, rtol=1e-10)
        h = Lt * 1e-4
        fd0 = (float(rm.closure_Y(Lt + h, kappa))
               - float(rm.closure_Y(Lt - h, kappa))) / (2 * h)
        fd_away = (float(rm.closure_Y(Lt + 60 * h, kappa))
                   - float(rm.closure_Y(Lt + 40 * h, kappa))) / (20 * h)
        scale = max(abs(fd_away), 1e-12 * abs(Y) / Lt)
        assert abs(fd0) < 0.05 * scale, (params, i, fd0, fd_away)


def _n_mixed_roots(rm, kappa, Y, n=40000):
    """Sign changes of the closure on a dense mixed-interval grid."""
    Lts = np.linspace(rm.xo * (1 + 1e-9), 1.0, n)
    A, B = rm._closure_pieces(Lts)
    f = (A + kappa * B) / rm.F_xo - Y
    s = np.sign(f)
    return int(np.sum(s[:-1] * s[1:] < 0))


def test_saddle_node_changes_root_count():
    # Crossing the fold from below (colder) to above (warmer) removes the
    # warm/cold root pair. The fold pair exists only for Y strictly between
    # the closure's endpoint values (Y(xo) = 1 and the L2 endpoint
    # alpha*kappa/kappa_c) and the fold itself, so probe at the fold point
    # where that strip is WIDEST (the strip pinches shut at both the corner
    # and the tip of the fold curve).
    widths = {}
    for name, params in [('marginal', MARGINAL), ('general', GENERAL)]:
        rm = RegimeMap(**params)
        b = rm.saddle_node(n_samples=400)
        ok = np.isfinite(b.kappa) & (b.kappa > 0)
        floor = np.maximum(1.0, rm.alpha * b.kappa / rm.kappa_c)
        width = np.where(ok, b.Y - floor, -np.inf)
        i = int(np.argmax(width))
        kappa, Y_fold = b.kappa[i], b.Y[i]
        widths[name] = width[i]
        if width[i] <= 1e-7:
            continue          # no resolvable bistable strip at these exponents
        below = _n_mixed_roots(rm, kappa, floor[i] + 0.5 * width[i])
        above = _n_mixed_roots(rm, kappa, Y_fold + 0.5 * width[i])
        assert below - above == 2, (name, kappa, Y_fold, below, above)
    # the marginal set must exhibit a real bistable strip
    assert widths['marginal'] > 1e-3


# ---------------------------------------------------------------------------
# Vectorized solves and conventions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('params', [MARGINAL, GENERAL],
                         ids=['marginal', 'general'])
def test_grid_matches_scalar(params):
    rm = RegimeMap(**params)
    kap = np.geomspace(0.1, 3.0, 8)
    Y = np.linspace(0.2, 1.1, 7)
    K, YY = np.meshgrid(kap, Y)
    Lt_grid = rm.Lt(K, YY)
    zo_grid = rm.zo(K, YY)
    for idx in [(0, 0), (3, 2), (6, 5), (2, 7), (5, 4)]:
        ks, yx = K[idx], YY[idx]
        Lt_s = float(rm.Lt(ks, yx))
        if np.isnan(Lt_s):
            assert np.isnan(Lt_grid[idx])
            assert zo_grid[idx] == 1.0
        else:
            assert np.isclose(Lt_grid[idx], Lt_s, rtol=1e-8), idx
            assert np.isclose(zo_grid[idx], float(rm.zo(ks, yx)),
                              rtol=1e-8), idx


def test_warm_branch_below_cold():
    # Inside a bistable window the warm root must sit at smaller Lt than the
    # cold root; outside, the two coincide.
    rm = RegimeMap(**MARGINAL)
    b = rm.saddle_node()
    ok = np.isfinite(b.kappa) & (b.kappa > 0.3 * rm.kappa_c) \
        & (b.kappa < 0.7 * rm.kappa_c)
    i = int(np.flatnonzero(ok)[len(np.flatnonzero(ok)) // 2])
    kappa, Y_fold = b.kappa[i], b.Y[i]
    Y_in = Y_fold * 0.995                      # just inside the fold
    cold = float(rm.Lt(kappa, Y_in, branch='cold'))
    warm = float(rm.Lt(kappa, Y_in, branch='warm'))
    assert warm < cold

    Y_mono = 0.3                                # deep in the mixed wedge
    c2 = float(rm.Lt(kappa, Y_mono, branch='cold'))
    w2 = float(rm.Lt(kappa, Y_mono, branch='warm'))
    assert np.isclose(c2, w2, rtol=1e-9)


def test_scalar_cold_solve_stays_on_descending_fold():
    # audit B1: in the pinched near-corner bistable strip the closed-form
    # guess can land in the warm fold basin; the trace-free scalar cold solve
    # used to converge onto the warm (ascending, unstable-saddle) root because
    # the coarse-scan fallback only fired on Newton FAILURE. It must return the
    # largest-Lt (descending-side) cold root instead.
    rm = RegimeMap(d=1.8, sigma=0.4, phi=0.3, theta=0.45, k=0.9, xo=2e-3)
    cold = float(rm.Lt(1.66795, 1.0003, branch='cold'))
    assert np.isclose(cold, 0.0021798792, rtol=1e-6)       # true cold root
    assert not np.isclose(cold, 0.0020028029, rtol=1e-4)   # not the warm root
    assert rm._dclosure_dLt(cold, 1.66795) < 0.0           # descending side
    assert rm._Lt_grid is None                             # stayed trace-free
    warm = float(rm.Lt(1.66795, 1.0003, branch='warm'))    # (builds the trace)
    assert np.isclose(warm, 0.0020028029, rtol=1e-6)
    assert rm._dclosure_dLt(warm, 1.66795) > 0.0           # ascending side
    assert cold > warm

    # across the bistable strip the scalar cold root must equal the grid root
    # (the correct, trace-interpolated reference) and never fall below the warm
    # root — pre-fix ~half these points leaked the warm root or NaN.
    kap = np.linspace(1.05, 1.67, 12)
    Y = np.linspace(1.0 + 1e-6, 1.001, 12)
    K, YY = np.meshgrid(kap, Y)
    Lt_grid = rm.Lt(K, YY)
    warm_grid = rm.Lt(K, YY, branch='warm')
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            s = float(rm.Lt(K[i, j], YY[i, j]))            # scalar path
            g = Lt_grid[i, j]
            assert np.isnan(s) == np.isnan(g), (K[i, j], YY[i, j])
            if not np.isnan(s):
                assert np.isclose(s, g, rtol=1e-6), (K[i, j], YY[i, j])
                assert s >= warm_grid[i, j] - 1e-12, (K[i, j], YY[i, j])


def test_scalar_finds_warm_corner_sliver_m8():
    # audit m8: the `Y > 1 and kappa >= kappa_c -> NaN` early return wrongly
    # reported fluvial-only inside the warm corner sliver, where a stable mixed
    # root exists just above Y = 1. The scalar path must now find it (agreeing
    # with the array path / GeneralProfile) and return NaN only where genuinely
    # fluvial.
    rm = RegimeMap(d=1.8, sigma=0.45, phi=0.3, theta=0.45, k=1.0, xo=1e-3)
    kappa, Y = rm.kappa_c, 1.000349          # kappa_c corner, just above Y = 1
    Lt_s = float(rm.Lt(kappa, Y))
    Lt_g = float(rm.Lt(np.array([kappa]), np.array([Y]))[0])
    assert np.isfinite(Lt_s)                 # sliver root exists (was NaN)
    assert np.isclose(Lt_s, Lt_g, rtol=1e-6)
    # genuinely fluvial (large Y, kappa >> kappa_c): both paths still return NaN
    assert np.isnan(rm.Lt(2.5, 1.5))
    assert np.isnan(rm.Lt(np.array([2.5]), np.array([1.5]))[0])


def test_grid_polish_stays_on_glacial_branch_across_kink():
    # audit B2: the vectorized grid polish used a per-iteration half-cell cap
    # that let an iterate escape its bracket, cross the sqrt-singular Lt = 1
    # kink, and drift down the mixed branch — misclassifying near-full-
    # glaciation cells (returned residual ~6e-3, classed mixed). The
    # safeguarded (bracketed) polish must converge to the glacial root just
    # above 1 and classify the cell glacial.
    rm = RegimeMap(d=2, sigma=0.5, phi=0.5, theta=0.5, k=1, xo=1e-3)
    for kappa, Y, expect in [(2.00866, 1.20001, 1.000100065),
                             (1.92109, 1.14714, 1.0001211)]:
        K = np.array([kappa])
        YY = np.array([Y])
        lt = float(rm.Lt(K, YY)[0])                 # array path -> _solve_grid
        assert np.isclose(lt, expect, rtol=1e-6), (kappa, Y, lt)
        assert lt > 1.0                             # glacial side of the kink
        assert abs(float(rm.closure_Y(lt, kappa)) - Y) < 1e-9   # a true root
        fluvial, mixed, glacial = rm.masks(K, YY)
        assert glacial[0] and not mixed[0] and not fluvial[0], (kappa, Y)


def test_closure_extrema_consistent_with_saddle():
    # Y_max from closure_extrema at a fold kappa must equal the fold Y.
    rm = RegimeMap(**MARGINAL)
    b = rm.saddle_node(n_samples=600)
    ok = np.isfinite(b.kappa) & (b.kappa > 0.4 * rm.kappa_c) \
        & (b.kappa < 0.6 * rm.kappa_c)
    i = int(np.flatnonzero(ok)[len(np.flatnonzero(ok)) // 2])
    Y_max, _ = rm.closure_extrema(b.kappa[i])
    assert np.isclose(Y_max, b.Y[i], rtol=1e-5)


def test_validation_errors():
    with pytest.raises(ValueError, match='phi'):
        RegimeMap(phi=1.0)
    with pytest.raises(ValueError, match='xo'):
        RegimeMap(xo=2.0)
    with pytest.raises(ValueError, match='k must be positive'):
        RegimeMap(k=0.0)
    with pytest.raises(ValueError, match='Lt_max'):
        RegimeMap(Lt_max=0.5)
    rm = RegimeMap()
    with pytest.raises(ValueError, match='branch'):
        rm.Lt(0.5, 0.5, branch='hot')
