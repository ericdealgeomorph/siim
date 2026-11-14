"""Tests for siim.analytical (kernel + profile classes) and siim.constants.

These run without fastscape/xsimlab — the analytical layer must stay
importable on numpy/scipy alone (guarded by test_light_import).
"""
import subprocess
import sys
import warnings

import numpy as np
import pytest

from siim import constants
from siim.analytical import (GeneralProfile, MarginalCoulombProfile,
                             Solution, incomplete_beta)


# ---------------------------------------------------------------------------
# Packaging / layering invariants
# ---------------------------------------------------------------------------

def test_light_import():
    """siim.analytical (and siim itself) import without the model stack."""
    code = (
        "import sys; import siim, siim.analytical, siim.constants; "
        "heavy = [m for m in ('matplotlib', 'fastscape', 'xsimlab', 'numba',"
        " 'pandas', 'tqdm') if m in sys.modules]; "
        "assert not heavy, f'heavy imports leaked: {heavy}'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_lazy_top_level_exports():
    import siim
    assert siim.GeneralProfile is GeneralProfile
    assert siim.MarginalCoulombProfile is MarginalCoulombProfile


def test_derive_single_source():
    """derive_* / incomplete_beta resolve to single copies — derive_* in
    siim.constants, incomplete_beta in siim.analytical.core (re-exported by
    siim.analytical)."""
    from siim.analytical import core as _core
    assert incomplete_beta is _core.incomplete_beta

    # Value regression: consolidation must not have changed the algebra.
    c = constants.derive_coulomb(1e-4, 10, 1e5, 920.0, 9.8, nu=1)
    assert (c.ell, c.nu, c.mu, c.phi) == (0.5, 1, 0.5, 0.5)
    np.testing.assert_allclose(
        c.Co, 1e-4 * ((920.0 * 9.8) ** 2 / (10 * 1e5 ** 2)) ** 0.5)
    p = constants.derive_power(1e-4, constants.cg_prefactor(), 300.0, 10, nu=1)
    assert (p.ell, p.nu, p.mu) == (0.6, 1, 4.0 / 15.0)


def test_physical_constants_unchanged():
    """Consolidated constants keep the package's historical values."""
    from siim.analytical import analytical_steady_state_solution
    a = analytical_steady_state_solution({})
    assert a.rho_g == 920.0
    assert a.g == 9.8
    assert a.kt == 365.25 * 24 * 3600.0
    assert constants.RHO_ICE_G == 920.0 * 9.8


def test_ac_convention_preserves_cg():
    """Model-paper convention: cg = alpha_g*kt*(2*Ac/5)*(rho*g)^3 with Ac the
    physical Glen coefficient (default 2.5e-24). The 2/5 with the halved
    default must reproduce the package's historical cg (which absorbed the 2
    into Ac = 5e-24) bit-for-bit, here and in the physical front end."""
    historical = constants.ALPHA_G * constants.KT * 1e-24 * constants.RHO_ICE_G ** 3
    assert constants.cg_prefactor() == historical
    from siim.analytical import analytical_steady_state_solution
    a = analytical_steady_state_solution({})
    assert a.cg == historical
    # lambda_p is a plain param: defaults to the single-source constant, and a
    # user value trumps. cb (and the old sqrt(cb/(2*Ac)) derivation) were removed,
    # so passing cb is now rejected as an unexpected parameter.
    assert a.lambda_p == constants.LAMBDA_P
    b = analytical_steady_state_solution({'lambda_p': 250.0})
    assert b.lambda_p == 250.0
    with pytest.raises(ValueError):
        analytical_steady_state_solution({'cb': 1e-19})


# ---------------------------------------------------------------------------
# Aligned constructor conventions
# ---------------------------------------------------------------------------

def test_aligned_signatures_and_defaults():
    kw = dict(ks=100.0, cs=150.0, zELA=500.0, L=1e5)
    m_kw = MarginalCoulombProfile(**kw)
    m_pos = MarginalCoulombProfile(100.0, 150.0, 500.0, 1e5)
    m_dct = MarginalCoulombProfile(kw)
    g_pos = GeneralProfile(100.0, 150.0, 500.0, 1e5)
    assert m_kw.Lt == m_pos.Lt == m_dct.Lt

    # Same positional order as GeneralProfile, same shared defaults.
    for obj in (m_kw, g_pos):
        assert (obj.ks, obj.cs, obj.zELA, obj.L) == (100.0, 150.0, 500.0, 1e5)
        assert obj.k == 1.0
        assert obj.sigma == 0.5
        assert obj.kh == constants.KH
        assert obj.alpha_g == constants.ALPHA_G

    # MCP picks up the package constants set.
    assert m_kw.beta == constants.BETA
    assert m_kw.lam_c == constants.LAMBDA_C
    assert m_kw.Ac == constants.AC
    assert m_kw.tau_c == constants.TAU_C
    # Fixed exponents of the marginal-Coulomb case.
    assert (m_kw.d, m_kw.phi, m_kw.theta, m_kw.gamma) == (2.0, 0.5, 0.5, 0.5)

    with pytest.raises(TypeError, match="missing required"):
        MarginalCoulombProfile(ks=100.0, cs=150.0)
    with pytest.raises(TypeError, match="unknown parameter"):
        MarginalCoulombProfile(dict(ks=1, cs=1, zELA=1, L=1e5, bogus=3))


# ---------------------------------------------------------------------------
# MCP as cross-check oracle of GeneralProfile: with the exact b = 0
# logarithmic limit in the kernel (no nudge), the general machinery must
# reproduce the closed-form arcosh solution to roundoff.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zELA, regime", [
    (300.0, "glacial"),
    (500.0, "glacial"),
    (650.0, "mixed"),
    (800.0, "fluvial"),
])
def test_marginal_agreement(zELA, regime):
    common = dict(ks=100.0, cs=150.0, L=1e5, sigma=0.5, k=1.0)
    m = MarginalCoulombProfile(zELA=zELA, **common)
    g = GeneralProfile(zELA=zELA, d=2.0, phi=0.5, theta=0.5, **common)
    assert m.regime == regime
    assert g.regime == regime
    if regime == "fluvial":
        assert np.isnan(m.Lt) and np.isnan(g.Lt)
        np.testing.assert_allclose(g.zo, m.zo, rtol=1e-12)
    else:
        np.testing.assert_allclose(g.Lt, m.Lt, rtol=1e-9)
        np.testing.assert_allclose(g.zo, m.zo, rtol=1e-9)


# ---------------------------------------------------------------------------
# Shared presentation machinery
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [GeneralProfile, MarginalCoulombProfile])
def test_profile_thickness_repr(cls):
    obj = cls(ks=100.0, cs=150.0, zELA=650.0, L=1e5)
    assert obj.regime == "mixed"
    x, z = obj.profile()
    assert x.shape == z.shape and x.size > 0
    assert np.all(np.isfinite(z))
    assert np.all(np.diff(x) > 0)
    H = obj.thickness(x)
    assert H.shape == x.shape
    assert np.all(H >= 0)
    assert H.max() > 0  # there is ice in the mixed regime
    # ice only within the glacier footprint
    assert np.all(H[x > obj.Lt] == 0)
    r = repr(obj)
    assert cls.__name__ in r and "mixed" in r

    sols = obj.solutions
    assert all(isinstance(s, Solution) for s in sols)
    assert sum(s.stable for s in sols) >= 1


def test_solution_dataclass_shared():
    """One Solution type across both classes (cross-class comparisons work)."""
    m = MarginalCoulombProfile(ks=100.0, cs=150.0, zELA=800.0, L=1e5)
    g = GeneralProfile(ks=100.0, cs=150.0, zELA=800.0, L=1e5)
    assert type(m.primary) is type(g.primary) is Solution
    # Field-wise identical fluvial state (Lt/zt are NaN, so == can't be used).
    assert (m.primary.regime, m.primary.stable) == (g.primary.regime, g.primary.stable)
    np.testing.assert_allclose(m.primary.zo, g.primary.zo, rtol=1e-12)
    assert np.isnan(m.primary.Lt) and np.isnan(g.primary.Lt)


# ---------------------------------------------------------------------------
def test_m27_mutate_then_solve_matches_fresh():
    """audit m27: the eager-solving profile classes expose solve(), so mutating
    a raw parameter attribute then calling solve() re-derives everything and
    matches a fresh construction with that parameter."""
    from siim.analytical import (GeneralProfile, MarginalCoulombProfile,
                                 SteadyStateProfile)
    g = GeneralProfile(ks=1000, cs=2000, zELA=1500, L=1e5)
    g.zELA = 1200.0
    assert g.solve() is g                          # returns self, in place
    fresh = GeneralProfile(ks=1000, cs=2000, zELA=1200, L=1e5)
    assert g.regime == fresh.regime
    np.testing.assert_allclose(g.Lt, fresh.Lt, rtol=1e-12)

    m = MarginalCoulombProfile(ks=1000, cs=2000, zELA=1500, L=1e5)
    m.zELA = 1200.0
    m.solve()
    fm = MarginalCoulombProfile(ks=1000, cs=2000, zELA=1200, L=1e5)
    assert m.regime == fm.regime
    np.testing.assert_allclose(m.Lt, fm.Lt, rtol=1e-12)

    s = SteadyStateProfile({'zELA': 1500.0})
    s.zELA = 1000.0
    s.solve()
    fs = SteadyStateProfile({'zELA': 1000.0})
    np.testing.assert_allclose(s.surface, fs.surface, rtol=1e-12)


# Kernel
# ---------------------------------------------------------------------------

def test_incomplete_beta_positive_b_matches_scipy():
    import scipy.special as sp
    x = np.array([0.1, 0.5, 0.9])
    got = incomplete_beta(x, 0.5, 2.5)
    want = sp.beta(0.5, 2.5) * sp.betainc(0.5, 2.5, x)
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_incomplete_beta_negative_b_quadrature():
    from scipy.integrate import quad
    for a, b in [(0.5, -0.7), (1.2, -2.4), (0.5, -1.0), (0.5, -2.0)]:
        for x in (0.2, 0.6, 0.9):
            want, _ = quad(lambda t, a=a, b=b: t ** (a - 1) * (1 - t) ** (b - 1),
                           0, x, limit=400)
            np.testing.assert_allclose(incomplete_beta(x, a, b), want,
                                       rtol=1e-8)


def test_incomplete_beta_b_zero_exact():
    """The marginal case b = 0: (1/k) B(1-u^k; 1/2, 0) equals the closed
    arcosh form (2/k) arccosh(u^(-k/2)) — this identity is what makes
    MarginalCoulombProfile the oracle for GeneralProfile."""
    from siim.analytical import incomplete_beta_compl
    for k in (0.6, 1.0, 3.0):
        for u in (1e-9, 1e-4, 0.1, 0.5, 0.9, 0.999999):
            got = incomplete_beta_compl(u ** k, 0.5, 0.0) / k
            want = (2.0 / k) * np.arccosh(u ** (-k / 2.0))
            np.testing.assert_allclose(got, want, rtol=1e-8)


def test_incomplete_beta_b_zero_a_gt_one_m4():
    """audit m4: for a > 1, b = 0, scipy's hyp2f1 is defective at the
    c-a-b = 0 logarithmic point on 1e-6 < eps <~ 1e-3 (wrong sign, O(1) error).
    The exact logarithmic series must now agree with mpmath (dps=40) across the
    audit's failing grid, including the negative-integer-b lift. mpmath is a
    TEST-only dependency (siim.analytical stays numpy/scipy-only)."""
    mp = pytest.importorskip('mpmath')
    from siim.analytical import incomplete_beta_compl
    mp.mp.dps = 40

    def ref(eps, a, b):
        x = mp.mpf(1) - mp.mpf(eps)
        return float(mp.betainc(mp.mpf(a), mp.mpf(b), 0, x))   # B(x; a, b)

    a_grid = [1.01, 1.055, 1.155, 1.255, 1.355, 1.455, 1.555, 1.655,
              1.755, 1.855, 1.955, 1.2, 1.5]
    eps_grid = [1e-6, 3e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.3, 0.5]
    worst = 0.0
    for a in a_grid:
        for eps in eps_grid:
            got = float(incomplete_beta_compl(np.array([eps]), a, 0.0)[0])
            want = ref(eps, a, 0.0)
            worst = max(worst, abs(got - want) / max(abs(want), 1e-300))
    assert worst < 1e-9, worst          # was O(1) (sign-flipped) pre-fix

    # the negative-integer-b lift reaches the same b = 0 branch
    for a in (1.2, 1.955):
        for eps in (1e-6, 1e-4, 1e-3, 1e-2, 0.3):
            got = float(incomplete_beta_compl(np.array([eps]), a, -1.0)[0])
            want = ref(eps, a, -1.0)
            assert abs(got - want) / abs(want) < 1e-9, (a, eps)


def test_incomplete_beta_compl_divergent_tail():
    """The eps-form is exact deep in the tail where the old clip froze G."""
    from siim.analytical import incomplete_beta_compl
    import scipy.special as sp
    # b = 0: -log(eps) - psi(a) - gamma asymptotic
    v = incomplete_beta_compl(1e-30, 0.5, 0.0)
    want = -np.log(1e-30) - sp.digamma(0.5) - np.euler_gamma
    np.testing.assert_allclose(v, want, rtol=1e-12)
    # b < 0: leading boundary term eps^b / |b|
    v = incomplete_beta_compl(1e-200, 0.5, -1.0)
    np.testing.assert_allclose(v, 1e200, rtol=1e-6)
    # eps = 1 -> B(0; .) = 0 on all branches
    for b in (2.5, 0.0, -0.7, -2.0):
        assert incomplete_beta_compl(1.0, 0.5, b) == 0.0


def test_incomplete_beta_is_inf_at_one_for_nonpositive_b():
    """B(1; a, b <= 0) diverges for every a > 0, as the docstring says. At a >= 1
    and b <= -1 the recurrence returned inf - inf = NaN there (audit #63)."""
    for a, b in [(1.5, -1.0), (2.5, -2.0), (1.0, -1.0), (3.0, -2.5), (0.5, -1.0), (1.5, 0.0)]:
        assert incomplete_beta(1.0, a, b) == np.inf
    got = incomplete_beta(np.array([0.9, 1.0]), 1.0, -1.0)   # B(0.9; 1, -1) = 1/0.1 - 1
    assert got[0] == pytest.approx(9.0) and got[1] == np.inf


def test_incomplete_beta_compl_is_inf_at_zero_for_nonpositive_b():
    """The complement form diverges the same way at eps = 0 (x = 1); it
    returned the same inf - inf = NaN at a >= 1, b <= -1."""
    from siim.analytical import incomplete_beta_compl
    for a, b in [(1.5, -1.0), (3.0, -2.5), (0.5, -1.0), (1.5, 0.0)]:
        assert incomplete_beta_compl(0.0, a, b) == np.inf
    got = incomplete_beta_compl(np.array([0.1, 0.0]), 1.0, -1.0)   # B(0.9; 1, -1) = 9
    assert got[0] == pytest.approx(9.0) and got[1] == np.inf


def test_incomplete_beta_guards():
    import time
    with pytest.raises(ValueError, match="a > 0"):
        incomplete_beta(0.5, -0.5, 1.0)
    with pytest.raises(ValueError, match="x in \\[0, 1\\]"):
        incomplete_beta(1.5, 0.5, 1.0)
    with pytest.raises(ValueError, match="too negative"):
        incomplete_beta(0.5, 0.5, -1e7)  # previously an infinite loop
    # b = -2500.3 was 7.9 s and NaN-corrupted; the true value exceeds float
    # range here, so inf (not NaN) — and fast.
    t0 = time.time()
    v = incomplete_beta(0.4, 0.5, -2500.3)
    assert time.time() - t0 < 1.0
    assert np.isinf(v) and v > 0


# ---------------------------------------------------------------------------
# Closure root-solver
# ---------------------------------------------------------------------------

def test_find_closure_roots_basic():
    from siim.analytical.core import find_closure_roots
    roots = find_closure_roots(lambda x: np.sin(np.log(x)), 1.1, 1e5)
    want = np.exp(np.pi * np.arange(1, 4))
    np.testing.assert_allclose(roots, want, rtol=1e-9)


def test_find_closure_roots_fold_pair():
    """Sub-cell saddle-node pairs are caught by the refinement pass."""
    from siim.analytical.core import find_closure_roots
    roots = find_closure_roots(lambda x: (np.log(x) - 3.0) ** 2 - 1e-8,
                               1.0, 1e4, refine=256)
    assert len(roots) == 2
    np.testing.assert_allclose(roots, [np.exp(3.0 - 1e-4), np.exp(3.0 + 1e-4)],
                               rtol=1e-6)


def test_find_closure_roots_exact_zero_node():
    """A grid node where f is exactly zero counts as one root, not two."""
    from siim.analytical.core import find_closure_roots
    roots = find_closure_roots(lambda x: np.asarray(x) - 10.0, 1.0, 1e3,
                               kink=10.0)  # kink inserts x=10 exactly
    assert len(roots) == 1
    np.testing.assert_allclose(roots, [10.0], rtol=1e-12)


# ---------------------------------------------------------------------------
# Review-repro regressions (numbers from the analytical_code review)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zELA", [2071.0, 2072.0, 2073.0, 2074.0, 2075.0])
def test_seam_consistency(zELA):
    """Exact arcosh on both closure branches: MCP and GP agree through the
    mixed/glacial seam (the old asymptotic-zgo glacial closure made them
    disagree on regime for zELA in (2071.1, 2074.1) and spawned a phantom
    stable glacial root at ~L)."""
    m = MarginalCoulombProfile(ks=400, cs=600, zELA=zELA, L=1e5, k=0.6)
    g = GeneralProfile(ks=400, cs=600, zELA=zELA, L=1e5, k=0.6)
    assert m.regime == g.regime
    np.testing.assert_allclose(g.Lt, m.Lt, rtol=1e-9)
    assert [s.stable for s in m.solutions] == [s.stable for s in g.solutions]
    assert sum(s.stable for s in m.solutions) == 1


def test_warm_saddle_demoted_and_window_scanned():
    """The warm saddle is unstable, and the cold root in the formerly
    unscanned (0.9999 L, L) window is found and stable."""
    m = MarginalCoulombProfile(ks=390, cs=500, zELA=2763.6957747145, L=1e5,
                               k=3.0, xo=100)
    glaciated = [s for s in m.solutions if np.isfinite(s.Lt)]
    assert len(glaciated) == 2
    saddle, cold = glaciated
    assert saddle.Lt < 200 and not saddle.stable      # was stable & primary
    assert cold.Lt > 0.999 * 1e5 and cold.stable      # was missed entirely
    assert any(s.regime == 'fluvial' and s.stable for s in m.solutions)


def test_g_no_saturation():
    """G(xo; Lt) keeps growing ~Lt^(k/6) where the old clip froze it."""
    g = GeneralProfile(ks=80, cs=1000, zELA=800, L=4e4, phi=0.75, k=3.0)
    vals = [g._G(g.xo, Lt) for Lt in (4e5, 4e6, 4e7, 4e8)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    # growth rate per decade approaches 10^(k/6) (subleading terms decay)
    np.testing.assert_allclose(np.diff(np.log10(vals)), 3.0 / 6.0, rtol=5e-3)
    with pytest.raises(ValueError, match="beyond the terminus"):
        g._G(8e5, 4e5)


def test_far_glacial_root_found():
    """Glacial roots far beyond the old 1e3 L scan cap are found, and GP
    agrees with the closed form out there."""
    m = MarginalCoulombProfile(ks=400, cs=380, zELA=-1500.0, L=1e5)
    g = GeneralProfile(ks=400, cs=380, zELA=-1500.0, L=1e5)
    assert m.regime == g.regime == 'glacial'
    assert m.Lt > 1e3 * 1e5
    np.testing.assert_allclose(g.Lt, m.Lt, rtol=1e-9)


def test_sigma_edge_cases_construct():
    """sigma = 0 (kappa_c = 1) and tiny sigma no longer crash/overflow."""
    m0 = MarginalCoulombProfile(ks=100, cs=600, zELA=1000, L=1e5, sigma=0.0)
    m1 = MarginalCoulombProfile(ks=100, cs=600, zELA=1000, L=1e5, sigma=1e-4)
    assert m0.regime is not None and m1.regime is not None


def test_stability_flags_match_slopes():
    """Stability is the closure slope sign, one-sided at the Lt = L kink
    (the straddling FD used to invert it and report 3 'stable' states)."""
    g = GeneralProfile(ks=100, cs=1000, zELA=4143.142194616984, L=1e5,
                       xo=100.33555007658357, k=1.0)
    for s in g.solutions:
        if not np.isfinite(s.Lt):
            continue
        h = s.Lt * 1e-4
        if s.Lt <= g.L:
            x1 = min(s.Lt, g.L * 0.9999)
            slope = (g._zELA_of_Lt(x1) - g._zELA_of_Lt(x1 - h)) / h
        else:
            slope = (g._zELA_of_Lt(s.Lt + h) - g._zELA_of_Lt(s.Lt)) / h
        assert (slope < 0) == s.stable
    assert sum(s.stable for s in g.solutions) == 2


def test_aar_truncated_at_base_level():
    """Fully glacial state: ELA crossing beyond the orogen means the whole
    on-orogen glacier accumulates (aar = 1); no off-orogen areas, no
    lam -> 1 cancellation."""
    g = GeneralProfile(ks=80, cs=200, zELA=300, L=1e5)
    assert g.regime == 'glacial'
    r = g.aar('both')
    assert r['powerlaw'].aar == 1.0           # crossing off-orogen
    assert 0.0 < r['gsurface'].aar <= 1.0
    assert np.isfinite(r['gsurface'].Aa_over_Ac)

    # lam -> 1: betaincc keeps the ablation integral from cancelling to 0
    g2 = GeneralProfile(ks=100, cs=150, zELA=650, L=1e5, lam=1 - 1e-12)
    r2 = g2.aar('powerlaw')
    assert np.isfinite(r2.aar)

    # sigma = 0 -> lam = 0: crossing collapses to the divide, aar = 0. At
    # sigma = 0 this set is glacial only for zELA ~1040-1240 m; at 650 the
    # closure has no root (regime None) and the branch never ran (audit #64)
    g3 = GeneralProfile(ks=100, cs=150, zELA=1150, L=1e5, sigma=0.0)
    assert g3.regime == 'glacial'
    r3 = g3.aar('powerlaw')
    # gate: was behind `if g3.regime in (...)`, which never held (audit #64)
    assert r3.aar == 0.0 and np.isinf(r3.Aa_over_Ac)


def test_eta_bar_consistency_check():
    """eta_bar (glacierized fraction of the below-ELA swath) is reported and
    warns above 1, where the width and Hack closures are inconsistent
    (model paper, AAR appendix)."""
    g = GeneralProfile(ks=100.0, cs=150.0, zELA=650.0, L=1e5)
    r = g.aar('gsurface')
    assert np.isfinite(r.eta_bar)
    if r.eta_bar <= 1.0:
        # consistent set: no warning expected
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            g.aar('gsurface')
    # crank the thickness prefactor until the footprint overflows the valley
    big = GeneralProfile(ks=100.0, cs=150.0, zELA=650.0, L=1e5, kH=50.0)
    with pytest.warns(UserWarning, match='eta_bar'):
        rb = big.aar('gsurface')
    assert rb.eta_bar > 1.0


# ---------------------------------------------------------------------------
# Two-sided base-level mirror (SteadyStateProfile)
# ---------------------------------------------------------------------------

def test_two_sided_base_level_mirror_even_nx():
    """audit B3: two-sided base-level profiles are solved on the upstream half
    and mirrored. With no grid node on the divide (even nx) the old
    xp = x[nx//2:]/Ld started at (nx-2)/(nx-1) < 1, so the mirror violated
    z = 0 at the outlets (off by ~half a cell, tens of metres inland). Built
    from distance-to-divide, xp is exact for any nx; odd nx is unchanged."""
    from siim.analytical import SteadyStateProfile

    L = 5.0e4
    cfg = dict(L=L, left_bc='base_level', right_bc='base_level')

    def z_true(p):
        # independent reference on the FULL grid: analytical profile of the
        # distance-to-divide |x - L/2| / Ld (matches at odd nx pre-fix).
        xp = np.abs(p.x - L / 2.0) / p.Ld
        z = p.analytical_z_fluvial(xp) if p.glacier_flag == 1 else p.analytical_z(xp)
        return z

    # pure fluvial (huge zELA), even nx: base level honored, profile exact
    for nx in (1000, 626):
        p = SteadyStateProfile({**cfg, 'nx': nx, 'zELA': 1e9})
        assert len(p.surface) == nx
        assert abs(p.surface[0]) < 1e-9 and abs(p.surface[-1]) < 1e-9, nx
        assert np.max(np.abs(p.surface - z_true(p))) < 1e-6, nx

    # glaciated, even nx: outlets still exactly at base level
    pg = SteadyStateProfile({**cfg, 'nx': 1000, 'zELA': 1500})
    assert abs(pg.surface[0]) < 1e-9 and abs(pg.surface[-1]) < 1e-9
    assert np.max(np.abs(pg.surface - z_true(pg))) < 1e-6

    # odd nx unchanged: xp is bit-for-bit the historical x[nx//2:]/Ld, and the
    # reassembled profile matches the independent reference to machine zero.
    po = SteadyStateProfile({**cfg, 'nx': 1001, 'zELA': 1e9})
    np.testing.assert_array_equal(po.xp, po.x[po.nx // 2:] / po.Ld)
    assert abs(po.surface[0]) < 1e-9
    assert np.max(np.abs(po.surface - z_true(po))) < 1e-9


def test_validation_errors():
    base = dict(ks=80, cs=120, zELA=1500, L=4e4)
    with pytest.raises(ValueError, match="phi"):
        GeneralProfile(**base, phi=1.25)      # G diverges at the head
    with pytest.raises(ValueError, match="cs"):
        GeneralProfile(**{**base, 'cs': -5})  # fabricated 'stable glacial'
    with pytest.raises(ValueError, match="ks"):
        GeneralProfile(**{**base, 'ks': 0})
    with pytest.raises(ValueError, match="xo"):
        GeneralProfile(**base, xo=0)
    with pytest.raises(ValueError, match="zELA"):
        GeneralProfile(**{**base, 'zELA': np.inf})
    with pytest.raises(ValueError, match="beta"):
        MarginalCoulombProfile(**base, beta=0)
    with pytest.raises(ValueError, match="lam"):
        GeneralProfile(**base, lam=1.0)


def test_negative_integer_b_profile_constructs():
    """d = 4 (b = -1, a non-positive integer) used to ZeroDivisionError."""
    g = GeneralProfile(ks=80, cs=120, zELA=1500, L=4e4, d=4.0)
    assert g.regime is not None


def test_zela_zero_queryable():
    """zELA = 0 (ELA at base level) is physical: construction, repr and the
    dimensionless groups must not raise (Nf/Ng -> inf; kappa is
    zELA-independent and stays finite)."""
    g = GeneralProfile(ks=100.0, cs=150.0, zELA=0.0, L=1e5)
    assert g.regime in ('mixed', 'glacial')
    assert np.isinf(g.Nf) and np.isinf(g.Ng)
    assert np.isfinite(g.kappa)
    assert 'GeneralProfile' in repr(g)
    m = MarginalCoulombProfile(ks=100.0, cs=150.0, zELA=0.0, L=1e5)
    assert 'MarginalCoulombProfile' in repr(m)
    np.testing.assert_allclose(g.Lt, m.Lt, rtol=1e-9)


def test_zela_zero_has_a_finite_thickness():
    """At zELA = 0 the thickness is finite (NH = H/zELA is infinite): the
    profiles and the steady-state front end build it instead of raising
    ZeroDivisionError (audit #57)."""
    from siim.analytical.steady_state import SteadyStateProfile
    x = np.linspace(0.0, 1e5, 201)
    for cls in (GeneralProfile, MarginalCoulombProfile):
        p = cls(ks=100.0, cs=150.0, zELA=0.0, L=1e5)
        H = p.thickness(x)
        assert np.isfinite(H).all() and H.max() > 0.0
        assert np.isinf(p.NH())
    ss = SteadyStateProfile({'zELA': 0.0})
    assert np.isfinite(ss.analytical_ice_thickness(np.linspace(0.0, 1.0, 11))).all()


@pytest.mark.pin
@pytest.mark.parametrize('a', [0.5, 0.25, 0.9])
@pytest.mark.parametrize('eps', [1e-6, 1e-3, 0.1, 0.4])
def test_incomplete_beta_is_continuous_through_b_zero(a, eps):
    """b = (1 - d*phi)/k one rounding step off 0 (a phi from np.arange) gives
    the b = 0 value to rounding, not an O(1) error: the reflection and the
    recurrence each subtracted two ~1/b terms (audit #56)."""
    from siim.analytical.core import incomplete_beta_compl
    b0 = incomplete_beta_compl(eps, a, 0.0)
    for b in (4.4e-16, -4.4e-16, 1.1e-16, -1e-13):
        assert abs(incomplete_beta_compl(eps, a, b) - b0) <= 1e-12 * abs(b0)


def test_incomplete_beta_small_b_matches_mpmath():
    """Against 50-digit quadrature on both sides of the small-b switch."""
    mp = pytest.importorskip('mpmath')
    from siim.analytical.core import _b_small, incomplete_beta_compl
    with mp.workdps(50):
        for a in (0.5, 0.05, 1.7):
            s = _b_small(a)
            # near 0, both sides of the switch, and the recurrence's hand-off
            # into the small-b form (b = -1 -+ delta)
            for b in (1e-10, -1e-10, 0.98 * s, -0.98 * s, 1.02 * s, -1.02 * s,
                      -1.0 + 1e-14, -1.0 - 1e-14):
                for eps in (1e-6, 0.1, 0.3, 0.4):
                    want = float(mp.betainc(a, b, 0, 1 - mp.mpf(eps)))
                    got = incomplete_beta_compl(eps, a, b)
                    assert abs(got - want) <= 1e-11 * abs(want), (a, b, eps, got, want)


def test_profile_is_continuous_in_phi_through_d_phi_one():
    """GeneralProfile at phi one ULP either side of 1/d returns the marginal
    profile (Lt was 99715 m against 32548 m before the fix)."""
    ref = GeneralProfile(ks=100.0, cs=150.0, zELA=650.0, L=1e5, phi=0.5)
    for phi in (np.arange(0.3, 0.71, 0.01)[20], 0.49999999999999994):
        assert phi != 0.5
        g = GeneralProfile(ks=100.0, cs=150.0, zELA=650.0, L=1e5, phi=phi)
        np.testing.assert_allclose(g.Lt, ref.Lt, rtol=1e-9)


def test_front_end_inherits_kernel():
    """analytical_steady_state_solution: no p-nudge, exact marginal Go."""
    import math
    from siim.analytical import analytical_steady_state_solution
    a = analytical_steady_state_solution({'d': 2.0, 'sliding_law': 'coulomb',
                                          'zELA': 1000})
    assert a.p == 1.0                          # was nudged to 1.01
    assert a.glacier_flag in (2, 3, 4, 5)
    Go_arcosh = (2.0 / a.k) * math.acosh((a.Lt / a.xo) ** (a.k / 2.0))
    np.testing.assert_allclose(a.Go, Go_arcosh, rtol=1e-8)


def test_mcp_nh_structure():
    """NH sliding term scales with the shape factor Go = Bo/k (the old code
    had a phi factor in place of the 1/k — ~phi/k off when sliding
    dominates)."""
    import math
    from siim.constants import KT, RHO_ICE_G
    m = MarginalCoulombProfile(ks=100.0, cs=150.0, zELA=650.0, L=1e5)
    Lt = m.Lt
    # deformation-free limit (lam_c = 0): NH = (Lt/zELA) * lam_tau / cs
    m0 = MarginalCoulombProfile(ks=100.0, cs=150.0, zELA=650.0, L=1e5,
                                lam_c=0.0)
    lam_tau = m0.tau_c / RHO_ICE_G
    np.testing.assert_allclose(m0.NH(Lt), (Lt / m0.zELA) * lam_tau / m0.cs,
                               rtol=1e-12)
    # sliding-dominated limit: NH^3 -> kh beta lam (Bo/k) / (lam_c cg lam_tau^2).
    # rtol 1e-2 still distinguishes Bo/k from the old phi*Bo structure
    # (a factor (phi/k)^(1/3) = 0.79 at these defaults).
    big = MarginalCoulombProfile(ks=100.0, cs=150.0, zELA=650.0, L=1e5,
                                 lam_c=1e6, alpha_g=10.0)  # asymptotic regime
                                 # validated at alpha_g=10 (default moved to 5)
    Bo = 2.0 * math.acosh((Lt / big.xo) ** (big.k / 2.0))
    cg = KT * big.alpha_g * (2.0 * big.Ac / 5.0) * RHO_ICE_G ** 3
    want = ((big.kh * big.beta * big.lam * (Bo / big.k))
            / (big.lam_c * cg * (big.tau_c / RHO_ICE_G) ** 2)) ** (1.0 / 3.0)
    np.testing.assert_allclose(big.NH(Lt), (Lt / big.zELA) * want, rtol=1e-2)


# ---------------------------------------------------------------------------
# Physical front end (SteadyStateProfile) on the shared solver — phase 3
# ---------------------------------------------------------------------------

def test_steady_state_equivalence_regression():
    """Pre-swap reference values (old fixed-point + single-bracket solver,
    captured at commit dac5908; full 64-config sweep agreed 64/64 at 1e-6).
    Pins three representative configs so the delegated solver stays put."""
    from siim.analytical import SteadyStateProfile
    # Pinned values were captured under alpha_g=10 (the default until 2026-07-02)
    # and xo=500 (the default until m54 single-sourced XO=300); pass both
    # explicitly so the oracle numbers stay valid.
    cases = [
        ({'sliding_law': 'power', 'zELA': 1500.0, 'alpha_g': 10.0, 'xo': 500},
         2, 25586.281484010742, 2022.3987072224695, 8.609065195772311),
        ({'sliding_law': 'coulomb', 'zELA': 800.0, 'alpha_g': 10.0, 'xo': 500},
         2, 41999.957361997054, 1263.2649182178848, 103.918934013564),
        ({'sliding_law': 'power', 'd': 2.0, 'sigma': 0.5, 'k': 3.0,
          'zELA': 1100.0, 'alpha_g': 10.0, 'xo': 500},
         2, 10844.665226405807, 1238.8337403853398, 10.05324001987857),
    ]
    for params, flag, Lt, zo, cs in cases:
        a = SteadyStateProfile(params)
        assert a.glacier_flag == flag
        np.testing.assert_allclose(a.Lt, Lt, rtol=1e-6)
        np.testing.assert_allclose(a.zo, zo, rtol=1e-6)
        np.testing.assert_allclose(a.cs, cs, rtol=1e-6)


def test_steady_state_u_zero():
    """U = 0 is a flat fluvial steady state (the old solver divided by
    Nf = 0 and crashed)."""
    from siim.analytical import SteadyStateProfile
    a = SteadyStateProfile({'U': 0.0})
    assert a.glacier_flag == 1
    assert np.all(a.surface == 0.0)
    assert np.isnan(a.kappa)  # 0/0 steepness ratio reported as NaN


def test_steady_state_scalar_evaluators():
    """analytical_z / analytical_zb accept scalars (old: TypeError on 0-d
    indexing) and still match the array path."""
    from siim.analytical import SteadyStateProfile
    a = SteadyStateProfile({'zELA': 800.0})
    z = a.analytical_z(0.5)
    zb = a.analytical_zb(0.5)
    assert isinstance(z, float) and isinstance(zb, float)
    np.testing.assert_allclose([z, zb],
                               [a.analytical_z(np.array([0.5]))[0],
                                a.analytical_zb(np.array([0.5]))[0]],
                               rtol=1e-13)


@pytest.mark.parametrize('bcs', [
    {}, {'left_bc': 'reflecting', 'right_bc': 'base_level'}])
def test_steady_state_plot_axis_is_distance_from_divide(bcs):
    """``plot`` draws the divide (z = zo) at x = 0, base level at x = L and
    the ice column over [0, Lt] whichever end holds base level (old: a fixed
    ``L - x`` flip, right only with base level on the right)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from siim.analytical import SteadyStateProfile
    a = SteadyStateProfile(bcs)
    assert a.glacier_flag == 2
    ax = a.plot(show_bedrock=False, label='topo')  # ice fill = sole collection
    topo = next(ln for ln in ax.lines if ln.get_label() == 'topo')
    x_km, z = topo.get_xdata(), topo.get_ydata()
    np.testing.assert_allclose(z[np.argmin(x_km)], a.zo)
    assert z[np.argmax(x_km)] == 0.0
    ice_km = ax.collections[0].get_paths()[0].vertices[:, 0]
    assert ice_km.min() == 0.0
    assert a.Lt - a.dx <= ice_km.max() * 1e3 <= a.Lt
    bed_km = next(ln for ln in ax.lines if ln is not topo).get_xdata()
    assert ice_km.min() <= bed_km.min() and bed_km.max() <= ice_km.max()
    plt.close(ax.figure)


def test_steady_state_shared_vocabulary():
    """SteadyStateProfile exposes Solution-typed steady states, and the public
    import paths resolve to the one object."""
    import siim
    from siim.analytical import SteadyStateProfile
    from siim.analytical.steady_state import analytical_steady_state_solution

    assert SteadyStateProfile is analytical_steady_state_solution
    assert siim.SteadyStateProfile is SteadyStateProfile
    assert siim.analytical_steady_state_solution is SteadyStateProfile

    a = SteadyStateProfile({'zELA': 800.0})
    assert all(type(s) is Solution for s in a.solutions)
    assert any(s.stable and np.isfinite(s.Lt) for s in a.solutions)
    # primary = largest stable root, matching GP/MCP semantics
    best = max((s for s in a.solutions if s.stable and np.isfinite(s.Lt)),
               key=lambda s: s.Lt)
    np.testing.assert_allclose(a.Lt, best.Lt, rtol=1e-12)


# ---------------------------------------------------------------------------
# Paper-workflow sweep helper
# ---------------------------------------------------------------------------

def test_sweep_regime_grid():
    from siim.analytical import sweep
    cs = np.linspace(80, 300, 4)
    zELA = np.linspace(200, 1200, 5)[:, None]
    out = sweep(ks=120.0, cs=cs, zELA=zELA, L=1e5)
    assert out['regime'].shape == (5, 4)
    assert set(np.unique(out['regime'])) <= {-1, 0, 1, 2}
    assert out['regime_codes'] == {'fluvial': 0, 'mixed': 1, 'glacial': 2}
    # spot-check one grid point against direct construction
    p = GeneralProfile(120.0, cs[2], float(zELA[1, 0]), 1e5)
    assert out['regime'][1, 2] == {None: -1, 'fluvial': 0,
                                   'mixed': 1, 'glacial': 2}[p.regime]
    np.testing.assert_allclose(out['Lt'][1, 2], p.Lt, equal_nan=True)
    # warming (higher zELA) never increases the regime code at fixed cs
    assert np.all(np.diff(out['regime'].astype(int), axis=0) <= 0)


@pytest.mark.pin
@pytest.mark.parametrize("law", ["power", "coulomb"])
def test_ce_ell_collapse_at_fixed_sliding_speed(law):
    """The steady state depends on (ce, ell, U) only through the sliding speed
    u_b* = (U/ce)^(1/ell) [m/yr]: every pair on ce = U / u_b*^ell gives the
    identical profile (phi = mu/nu is ell-free for both laws). Measured
    2026-09-17 to 1e-14; a derive_power/derive_coulomb refactor that breaks
    it would silently falsify the parameter guide and the model paper."""
    from siim.analytical import SteadyStateProfile
    U = constants.U
    ref = SteadyStateProfile({"sliding_law": law})
    ub_star = (U / constants.CE) ** (1.0 / ref.ell)
    for ell in (0.8, 1.0, 1.5, 2.0, 3.0):
        alt = SteadyStateProfile({"sliding_law": law, "ell": ell,
                                  "ce": U / ub_star ** ell})
        assert alt.glacier_flag == ref.glacier_flag == 2
        assert alt.zo == pytest.approx(ref.zo, rel=1e-10)
        assert alt.Lt == pytest.approx(ref.Lt, rel=1e-10)
        np.testing.assert_allclose(alt.surface, ref.surface, rtol=1e-10)


def test_solve_rederives_after_any_input_changes():
    """solve() re-ran only the root solve, so a mutated input mixed with stale
    derived values (K, Co, lam, the grid); a GeneralProfile's default xo stayed
    at the old L/1000 (audit #59). Each case now matches a fresh build."""
    from siim.analytical import GeneralProfile, MarginalCoulombProfile, SteadyStateProfile
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for change in ({'Ko': 4e-6}, {'k': 3.0}, {'nu': 1.5}, {'L': 8e4}):
            s = SteadyStateProfile({'zELA': 1500.0})
            for name, value in change.items():
                setattr(s, name, value)
            s.solve()
            fresh = SteadyStateProfile({'zELA': 1500.0, **change})
            assert s.glacier_flag == fresh.glacier_flag, change
            np.testing.assert_allclose(s.x, fresh.x)
            if fresh.surface is not None:
                np.testing.assert_allclose(s.surface, fresh.surface, rtol=1e-12)
        s = SteadyStateProfile({'zELA': 1500.0})
        zT = s.zT - 300.0
        s.zT = zT
        s.solve()
        assert s.zELA == SteadyStateProfile({'zT': zT}).zELA == zT - s.P / s.beta
    for cls in (GeneralProfile, MarginalCoulombProfile):
        g = cls(ks=100, cs=150, zELA=650, L=1e5)
        g.L = 2e5
        g.solve()
        fresh = cls(ks=100, cs=150, zELA=650, L=2e5)
        assert g.xo == fresh.xo == 200.0
        assert g.regime == fresh.regime
        np.testing.assert_allclose(g.Lt, fresh.Lt, rtol=1e-12)
    g = GeneralProfile(ks=100, cs=150, zELA=650, L=1e5, xo=300.0)
    g.L = 2e5
    g.solve()
    assert g.xo == 300.0                     # an xo given is kept
    g.L = 200.0
    with pytest.raises(ValueError, match="xo"):
        g.solve()                            # a kept xo past the new L is refused
    s = SteadyStateProfile({'dx': 50.0})
    s.nx = 201
    s.solve()
    assert s.x.size == 201 == SteadyStateProfile({'nx': 201}).x.size
    s.zELA, s.zT = 1200.0, 2500.0
    with pytest.raises(ValueError, match="not both"):
        s.solve()
    s = SteadyStateProfile({'zELA': 1500.0})
    s.U = np.full(5, 1e-3)
    with pytest.raises(TypeError, match="U must be a scalar"):
        s.solve()


@pytest.mark.parametrize("params, msg", [
    ({'right_bc': 'fixed_value'}, "right_bc must be one of"),
    ({'sliding_law': 'columb'}, "sliding_law must be one of"),
    ({'sigma': 0.0}, "sigma must be finite and > 0"),
])
def test_steady_state_profile_refuses_bad_inputs(params, msg):
    """A typo'd boundary condition crashed later on a missing attribute, an
    unknown sliding law ran the power law, sigma = 0 divided by zero (audit #58)."""
    from siim.analytical import SteadyStateProfile
    with pytest.raises(ValueError, match=msg):
        SteadyStateProfile(params)


@pytest.mark.parametrize('bcs', [
    {}, {'left_bc': 'reflecting', 'right_bc': 'base_level'}])
def test_steady_state_plot_profile_matches_plot(bcs):
    """plot_profile drew x as stored, so with base level on the right its
    glacier sat at 26-50 km while plot() draws it from the divide at 0; and
    its fixed 2*zELA top clipped the profile (1260 m against 1200 m at
    zELA = 600; audit C6)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from siim.analytical import SteadyStateProfile
    a = SteadyStateProfile({'zELA': 600.0, **bcs})
    assert a.surface.max() > 2 * a.zELA, "test setup: the old top clips"
    ax = a.plot(show_bedrock=False, label='topo')
    topo = next(ln for ln in ax.lines if ln.get_label() == 'topo')
    _, (panel,) = a.plot_profile('elevation')
    # gate: plot_profile's elevation panel is now plot() itself, its line 'Surface'
    surface = next(ln for ln in panel.lines if ln.get_label() == 'Surface')
    # the same surface over the same x, whatever the order it is drawn in
    drawn = []
    for line in (topo, surface):
        x, z = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
        drawn.append((x[np.argsort(x)], z[np.argsort(x)]))
    np.testing.assert_allclose(drawn[1][0], drawn[0][0])
    np.testing.assert_allclose(drawn[1][1], drawn[0][1])
    assert panel.get_ylim()[1] >= a.surface.max()
    plt.close('all')


def test_steady_state_plot_two_sided_bed_line_under_the_ice():
    """With base level at both ends, plot() took the Lt range from the left
    end, not the divide, so the sub-ice bed line lay at 0.3-9.45 km under an
    ice column at 15.6-34.4 km, on an axis called distance from the divide
    (audit #61, C7)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from siim.analytical import SteadyStateProfile
    a = SteadyStateProfile({'left_bc': 'base_level', 'right_bc': 'base_level',
                            'zELA': 1500.0, 'nx': 1001})
    ax = a.plot(show_bedrock=False, label='topo')
    ice_km = ax.collections[0].get_paths()[0].vertices[:, 0]
    bed_km = next(ln for ln in ax.lines if ln.get_label() != 'topo').get_xdata()
    assert ice_km.min() < a.L / 2e3 < ice_km.max()
    assert ice_km.min() <= bed_km.min() and bed_km.max() <= ice_km.max()
    assert bed_km.max() - bed_km.min() > 0.9 * (ice_km.max() - ice_km.min())
    assert ax.get_xlabel() == 'x (km)'
    plt.close(ax.figure)


def test_analytical_plots_share_the_model_plotters_colours():
    """The analytical ELA was red (plot) and goldenrod (plot_profile), its ice
    a paler blue and its bed a neutral grey: they read the model plotters'
    siim._palette.COLORS now (2026-11-05), still without importing
    matplotlib at package import (test_light_import)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    from siim._palette import COLORS
    from siim.analytical import SteadyStateProfile
    from siim.plotting._style import COLORS as PLOT_COLORS
    assert PLOT_COLORS is COLORS                      # one table, re-exported
    a = SteadyStateProfile({'zELA': 600.0})
    for ax in (a.plot(show_bedrock=False),
               GeneralProfile(ks=100.0, cs=150.0, zELA=300.0, L=1e5).plot(show_bedrock=False)):
        ela = [ln for ln in ax.lines if len(set(ln.get_ydata())) == 1]
        assert [to_hex(ln.get_color()) for ln in ela] == [COLORS['ela']]
        assert to_hex(ax.collections[0].get_facecolor()[0]) == COLORS['ice_fill']
        assert ax.figure.get_figwidth() == 6
    fig, axes = a.plot_profile(['elevation', 'ice_thickness', 'water_flux'])
    assert fig.get_figwidth() == 12
    assert axes[1].get_ylabel() == 'Ice column depth (m)'     # surface - bed = hc*H
    assert to_hex(axes[1].lines[0].get_color()) == COLORS['ice']
    assert to_hex(axes[2].lines[0].get_color()) == COLORS['water_flux']
    assert axes[-1].get_xlabel() == 'Distance from divide (km)'
    plt.close('all')
