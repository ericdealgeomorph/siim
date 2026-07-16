"""Boundary/flux regression tests for the numerical solvers.

Covers the package-review fixes: the two-sided head-catchment flux factor,
the base-level outlet ghost-receiver convention (mode A diagnostics and the
mode-B walks, 1D and 2D), and the glacial erosion / flow-accumulator processes
(GlacialSPLModeA / GlacialFlowAccumulator) reading the post-tectonics surface.

Surfaces are zs = zb + HC*H with HC = constants.HC_OVER_H (the tracked bed
is the channel floor, H the width-mean depth); cap/floor/freeboard
expectations below are written in terms of HC so they pin whichever
convention ships.
"""
import numpy as np
import pytest

from siim.constants import HC_OVER_H

HC = float(HC_OVER_H)


# ---------------------------------------------------------------------------
# Two-sided head-catchment flux (siim1d._solve_ice_flux)
# ---------------------------------------------------------------------------

def test_two_sided_divide_flux_closure():
    """At each divide node the head-catchment closure gives Qg = B * A(xo)
    = k_h * B * xo^d exactly (the analytical's Beff*A form). The old code
    added the bare integrand — a factor xo/(d*sigma) too small (~100x)."""
    from siim.siim1d import _solve_ice_flux
    nx, L = 101, 5e4
    x = L - np.linspace(0, L, nx)              # decreasing, siim1d convention
    s = np.linspace(0, L, nx)
    z = 2000.0 - 1500.0 * ((s - L / 2) / (L / 2)) ** 2   # symmetric dome
    beta, zELA, d, sigma, k_h, P, xo = 1e-2, 1000.0, 1.8, 0.45, 5.0, 1.0, 500.0
    mid = nx // 2
    didx_l, didx_r = mid, mid + 1
    B, Qg, Qf = _solve_ice_flux(x, z, zELA, beta, d, sigma, k_h, P,
                                xo, xo, didx_l, didx_r, nx, np.inf)
    for i in (didx_l, didx_r):
        B_div = beta * (z[i] - zELA)
        np.testing.assert_allclose(Qg[i], k_h * B_div * xo ** d, rtol=1e-9)

    # and the two-sided divide flux matches a one-sided run's divide flux
    # (same closure; the one-sided branch adds an S-extrapolation that
    # vanishes at a locally flat divide)
    nh = mid + 1
    B1, Qg1, _ = _solve_ice_flux(x[:nh] - x[mid], z[:nh], zELA, beta, d,
                                 sigma, k_h, P, xo, xo, nh - 1, nh, nh, np.inf)
    np.testing.assert_allclose(Qg[didx_l], Qg1[nh - 1], rtol=2e-2)


def test_one_sided_head_cap_honors_B_cap():
    """The one-sided (reflecting-divide) head integral honors the P cap B_cap
    (audit F1): bit-for-bit the uncapped closed form when the cap never binds
    (incl. B_cap=inf, i.e. cap_ice_accumulation=False); the exact all-capped
    divide flux k_h*B_cap*xo^d when it binds across the whole head; and the
    verified piecewise value for a partial bind (checked against an independent
    quadrature)."""
    from siim.siim1d import _solve_ice_flux
    nx, dx = 60, 500.0
    x = (nx - 1) * dx - np.arange(nx) * dx      # decreasing; divide at index nx-1
    z = 800.0 + 20.0 * np.arange(nx)            # rises toward the divide
    zELA, beta, d, sigma, k_h, P = 700.0, 1e-2, 1.8, 0.45, 5.0, 1.0
    xo = 5000.0
    didx_l, didx_r = nx - 1, nx                 # one-sided, left block only
    args = (x, z, zELA, beta, d, sigma, k_h, P, xo, xo, didx_l, didx_r, nx)

    # (c) B_cap = inf: unchanged; (b) present but never binding: bit-for-bit == inf
    _, Qg_inf, _ = _solve_ice_flux(*args, np.inf)
    _, Qg_big, _ = _solve_ice_flux(*args, 1e12)
    np.testing.assert_array_equal(Qg_big, Qg_inf)

    # (a) cap binds across the whole head -> Qg(divide) = k_h*B_cap*xo^d exactly
    B_cap = 1.0                                  # << b_min over head = beta*(z[-1]-zELA)
    _, Qg_cap, _ = _solve_ice_flux(*args, B_cap)
    np.testing.assert_allclose(Qg_cap[didx_l], k_h * B_cap * xo ** d, rtol=1e-12)
    assert Qg_cap[didx_l] < Qg_inf[didx_l]       # capping lowers the divide flux

    # partial bind: divide flux matches an independent quadrature of the capped
    # head integrand (u = x'^(d*sigma) substitution removes the x'=0 singularity)
    z_first = z[didx_l]
    S = (z[didx_l] - z[didx_l - 1]) / dx
    dsig = d * sigma
    B_partial = beta * (z_first - zELA) + 0.5 * beta * S * xo   # midway across head
    _, Qg_p, _ = _solve_ice_flux(*args, B_partial)
    from scipy.integrate import trapezoid   # np.trapz removed in numpy 2.0
    u = np.linspace(0.0, xo ** dsig, 1_000_001)
    xp = u ** (1.0 / dsig)
    b = np.minimum(beta * (z_first + S * (xo - xp) - zELA), B_partial)
    I_head = trapezoid(b, u) / dsig
    Qg_ref = (k_h * sigma * d / xo ** (d * (sigma - 1.0))) * I_head
    np.testing.assert_allclose(Qg_p[didx_l], Qg_ref, rtol=1e-6)
    assert Qg_cap[didx_l] < Qg_p[didx_l] < Qg_inf[didx_l]


# ---------------------------------------------------------------------------
# Base-level outlet OUTFLOW BC (mode-B walks, 1D): zero-gradient thickness +
# arrival-slope bed erosion + flotation gate (docs/dev/boundary_conditions.md).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("law", ["eff-exp", "power", "coulomb"])
def test_mode_b_outlet_outflow_1d(law):
    """A through-flowing base-level outlet is an OUTFLOW border: it gets its
    interior neighbour's thickness (zero-gradient), so H[0] == H[1] exactly and
    there is no one-cell ice cliff. Qg = 0 keeps the terminus condition H = 0."""
    from ._kernel_adapters import (_diag_walk_eff_exp, _diag_walk_power,
                                   _diag_walk_coulomb)
    nx, dx = 50, 100.0
    zb = np.linspace(0.0, 980.0, nx)           # outlet at i=0, divide at nx-1
    Qg = np.linspace(1e6, 1e4, nx)             # ice everywhere, exits at i=0
    H = np.zeros(nx)
    cg, lam_p, lam_c, tau_c, rho_g_g, clamp = 2.3e-4, 300.0, 1e-3, 1e5, 9016.0, 1e-12
    args = dict(didx_l=nx - 1, didx_r=nx, nx=nx, hc=HC)  # one-sided, left block only
    if law == "eff-exp":
        _diag_walk_eff_exp(zb, Qg, H, cg, lam_p, dx, **args)
    elif law == "power":
        _diag_walk_power(zb, Qg, H, cg, lam_p, dx, **args)
    else:
        _diag_walk_coulomb(zb, Qg, H, cg, lam_c, tau_c, rho_g_g, dx, clamp,
                           **args)
    assert H[0] > 0.0, "through-flowing outlet must keep finite ice"
    assert H[0] == H[1]                         # zero-gradient thickness, no cliff
    # terminus condition unchanged when no ice flows through
    Qg0 = Qg.copy()
    Qg0[0] = 0.0
    H2 = np.zeros(nx)
    if law == "eff-exp":
        _diag_walk_eff_exp(zb, Qg0, H2, cg, lam_p, dx, **args)
    elif law == "power":
        _diag_walk_power(zb, Qg0, H2, cg, lam_p, dx, **args)
    else:
        _diag_walk_coulomb(zb, Qg0, H2, cg, lam_c, tau_c, rho_g_g, dx, clamp,
                           **args)
    assert H2[0] == 0.0


@pytest.mark.parametrize("law", ["eff-exp", "power", "coulomb"])
def test_mode_b_outlet_bounded_first_arrival_1d(law):
    """First-arrival regression: a mega-flux reaching a bare, near-flat outlet
    gets a BOUNDED border H — the zero-gradient thickness H[0] = H[1] (the
    interior neighbour's from-a closure value), not a reflected-slope pillar
    (which returned H ~ 10 km here and the monotone walk jacked the whole
    profile). The walk stays bounded."""
    from ._kernel_adapters import (_diag_walk_eff_exp, _diag_walk_power,
                                   _diag_walk_coulomb)
    nx, dx = 50, 100.0
    zb = np.linspace(0.0, 10.0, nx)            # near-flat bare foreland
    Qg = np.full(nx, 1e10)                     # flux just reached the outlet
    H = np.zeros(nx)                           # lagged H = 0 everywhere
    cg, lam_p, lam_c, tau_c, rho_g_g, clamp = 2.3e-4, 300.0, 1e-3, 1e5, 9016.0, 1e-12
    args = dict(didx_l=nx - 1, didx_r=nx, nx=nx, hc=HC)
    if law == "eff-exp":
        _diag_walk_eff_exp(zb, Qg, H, cg, lam_p, dx, **args)
    elif law == "power":
        _diag_walk_power(zb, Qg, H, cg, lam_p, dx, **args)
    else:
        _diag_walk_coulomb(zb, Qg, H, cg, lam_c, tau_c, rho_g_g, dx, clamp,
                           **args)
    assert H[0] == H[1]                          # zero-gradient thickness
    assert H[0] > 0.0
    assert H.max() < 1000.0, law                 # bounded, no pillar


def test_mode_b_outlet_bounded_first_arrival_2d():
    """2D twin of the first-arrival regression: the border column gets the
    bounded zero-gradient thickness (its dominant donor's value), not a
    reflected-slope pillar."""
    from ._kernel_adapters import glac_fast_solve_modeB
    n = 6
    zb = np.linspace(0.0, 5.0, n)              # near-flat lagged foreland
    H = np.zeros(n)
    surf = np.empty(n)
    ice = np.full(n, 1e10)
    rec = np.array([0, 0, 1, 2, 3, 4], dtype=np.int64)
    stack = np.arange(n, dtype=np.int64)
    lengths = np.full(n, 100.0)
    lengths[0] = 0.0
    glac_fast_solve_modeB(zb, ice, np.zeros(n), H, surf,
                          1e-6, 1e-9, 1.0, 1.0, 0.5, 4.0 / 15.0,
                          2.3e-4, 10.0, 300.0,
                          1000.0, lengths, stack, rec,
                          0.0, 1, n, 100.0, 100.0, np.zeros(n), HC)
    assert H[0] == H[1]                          # zero-gradient from the donor
    assert H[0] > 0.0
    assert H.max() < 1000.0


def test_mode_b_border_implicit_1d():
    """1D border bed (OUTFLOW BC): an icy outlet bed evolves by the IMPLICIT
    arrival-slope budget — _erode_border_bed_1d commits exactly the closed-form
    backward-Euler step of dzb/dt = U - f*E(S_arr) with S_arr the interior
    flow slope (zs[2] - zs[1])/dx floored at S_FLOOR_BC, computed independently
    here. A huge dt lands the bed bounded near the flotation draft (monotone
    step — no overshoot). The ice-free branch (U-recovery toward bl) is
    separate and unchanged."""
    from siim.siim1d import siim as siim1
    from siim._core.skeleton import _implicit_border_step
    from siim._core.eroders import _modeb_border_erosion
    from siim._core.solvers import LAW_EFFEXP
    from siim import constants as _constants
    m = siim1({'L': 2e3, 'nx': 21, 'T': 10.0, 'nt': 2, 'mode': 'B',
               'sliding_law': 'eff-exp', 'left_bc': 'base_level',
               'right_bc': 'reflecting', 'progress_bar': False,
               'border_bed_uplift': 0.0})
    m.initialize_simulation()
    m.tj = 1
    m.locate_divide()
    # rising interior surface so the arrival slope zs[2]-zs[1] > 0
    m.zb[:] = 0.0
    m.H[:] = 0.0
    m.H[0], m.H[1], m.H[2] = 5.0, 200.0, 260.0
    m.Qg[:] = 0.0
    m.Qg[0] = m.Qg[1] = 1e8
    m.dt = 1e7                                   # huge dt: monotone, no overshoot
    zs = m.zb + m.hc_over_H * m.H
    S_arr = max(_constants.S_FLOOR_BC, (zs[2] - zs[1]) / m.dx)
    E = _modeb_border_erosion(LAW_EFFEXP, m.Qg[0], S_arr, m.H[0], m.Co, m.mu,
                              m.nu, m.ce, m.cg, m.alpha_g, m.lambda_p)
    want = _implicit_border_step(m.zb[0], 0.0, E, m.dt, m.H[0], m.hc_over_H,
                                 0.0, m.flotation_ramp)
    m._erode_border_bed_1d()
    assert m.zb[0] == want                       # bit-for-bit the closed form
    assert m.zb[0] >= -m.hc_over_H * m.H[0]      # bounded at/above the draft
    assert m.zb[0] < 0.0                         # eroding below the datum
    # ice-free branch unchanged: recovery at U toward bl, clamped
    m.Qg[0] = 0.0
    m.zb[0] = -50.0
    m.border_bed_uplift = 1e-3
    m.dt = 1000.0
    m._erode_border_bed_1d()
    np.testing.assert_allclose(m.zb[0], -49.0)


def test_mode_b_outlet_outflow_2d():
    """2D SFR mode-B walk (OUTFLOW BC): a through-flowing border gets zero-
    gradient thickness (its dominant donor's H), its bed erodes by the implicit
    arrival-slope budget (below the datum here, bounded by the flotation
    draft), and the committed border surface is the true zb + hc*H. Terminus
    condition (no through-flux -> H = 0) unchanged."""
    from ._kernel_adapters import glac_fast_solve_modeB
    n = 6                                       # 1 x 6 strip, outlet at 0
    zb = np.linspace(0.0, 500.0, n)
    H = np.zeros(n)
    surf = np.empty(n)
    ice = np.full(n, 5e5)
    water = np.zeros(n)
    rec = np.array([0, 0, 1, 2, 3, 4], dtype=np.int64)   # node 0 self-receives
    stack = np.arange(n, dtype=np.int64)
    lengths = np.full(n, 100.0)
    lengths[0] = 0.0                            # fastscape: 0 at base level
    glac_fast_solve_modeB(zb, ice, water, H, surf,
                          1e-6, 1e-9, 1.0, 1.0, 0.5, 4.0 / 15.0,
                          2.3e-4, 10.0, 300.0,
                          1000.0, lengths, stack, rec,
                          0.0, 1, n, 100.0, 100.0, np.zeros(n), HC)
    assert H[0] == H[1]                         # zero-gradient from the donor
    assert -HC * H[0] <= zb[0] < 0.0            # eroding, draft-bounded
    assert surf[0] == zb[0] + HC * H[0]         # true state
    # terminus: no through-flux -> H = 0; the ice-free border bed below bl
    # recovers at border_bed_uplift (= 0 here, so unchanged).
    ice0 = ice.copy(); ice0[0] = 0.0
    H2 = np.zeros(n); surf2 = np.empty(n)
    glac_fast_solve_modeB(zb, ice0, water, H2, surf2,
                          1e-6, 1e-9, 1.0, 1.0, 0.5, 4.0 / 15.0,
                          2.3e-4, 10.0, 300.0,
                          1000.0, lengths, stack, rec,
                          0.0, 1, n, 100.0, 100.0, np.zeros(n), HC)
    assert H2[0] == 0.0
    assert surf2[0] == zb[0]                    # true state (bare bed shows)


def test_mode_b_glaciated_to_edge_1d():
    """Integration: 1D mode B with ice through the outlet under the OUTFLOW BC.
    The border gets its interior neighbour's thickness (zero-gradient, no
    cliff: H[0] == H[1]), its bed carves below the datum by the implicit
    arrival-slope budget (bounded at the flotation draft), and the border
    surface is the true zb + hc*H. border_bed_uplift is the U in the border
    budget, so net uplift resists carving; an ice-free outlet sits at the
    datum."""
    from siim.siim1d import siim as siim1
    base = {'zELA': -200.0, 'L': 2e4, 'nx': 201, 'T': 1e5, 'nt': 101,
            'mode': 'B', 'sliding_law': 'eff-exp', 'nu': 1, 'ce': 1e-4,
            'left_bc': 'base_level', 'right_bc': 'reflecting',
            'progress_bar': False}
    m = siim1({**base, 'border_bed_uplift': 0.0})
    m.run()
    assert m.Qg[0] > 0.0
    assert m.H[0] == m.H[1]                     # zero-gradient thickness, no cliff
    assert m.zb[0] < 0.0                        # bed carved below the datum
    assert m.zb[0] >= -m.hc_over_H * m.H[0] - 1.0   # bounded at the draft
    # surface is the true state.
    np.testing.assert_allclose(m.z[0], m.zb[0] + m.hc_over_H * m.H[0], rtol=0, atol=0)

    mu_ = siim1({**base, 'border_bed_uplift': 1e-3})
    mu_.run()
    assert mu_.zb[0] >= m.zb[0]                 # net uplift resists carving

    f = siim1({**base, 'zELA': 8000.0})         # no ice anywhere
    f.run()
    assert f.z[0] == 0.0                        # the datum, exactly (ice-free)


def test_mode_b_border_budget_recovery_waterline_2d():
    """2D mode-B border-bed semantics (OUTFLOW BC): under ice the bed erodes by
    the implicit arrival-slope budget — bounded near the flotation draft even
    at a huge dt (the monotone closed-form step cannot overshoot); an ice-free
    bed below the datum recovers at border_bed_uplift (clamped at the datum)
    with bed memory kept; and the output surface reports the TRUE state
    everywhere (border included — no water-line floor, true-state output
    convention), so the mass balance sees the deep trough and ice flux dies
    crossing it."""
    from ._kernel_adapters import glac_fast_solve_modeB
    n = 6
    rec = np.array([0, 0, 1, 2, 3, 4], dtype=np.int64)
    stack = np.arange(n, dtype=np.int64)
    lengths = np.full(n, 100.0)
    lengths[0] = 0.0

    def run(zb, ice, bbu, dt, Co=1e-9):
        H = np.zeros(n)
        surf = np.empty(n)
        bbu_arr = np.full(n, float(bbu)) if np.isscalar(bbu) else bbu
        glac_fast_solve_modeB(zb, ice, np.zeros(n), H, surf,
                              1e-6, Co, 1.0, 1.0, 0.5, 4.0 / 15.0,
                              2.3e-4, 10.0, 300.0,
                              dt, lengths, stack, rec,
                              0.0, 1, n, 100.0, 100.0, bbu_arr, HC)
        return zb, H, surf

    # under ice at a HUGE dt: the implicit budget lands bounded at/above the
    # flotation draft (the gate-era explicit budget dug km/step here — the
    # deep-hole failure; the monotone closed-form step cannot overshoot)
    zb, H, _ = run(np.linspace(0.0, 2000.0, n), np.full(n, 5e7),
                   bbu=0.0, dt=5e4, Co=1e-3)
    assert zb[0] < 0.0                           # eroding below the datum
    assert zb[0] >= -HC * H[0] - 1e-9            # never below the draft

    # ice-free relict hole rises at border_bed_uplift, clamped at the datum
    zb = np.linspace(0.0, 500.0, n); zb[0] = -50.0
    ice = np.full(n, 5e5); ice[0] = 0.0
    zb, _, _ = run(zb, ice, bbu=1e-3, dt=1000.0)
    np.testing.assert_allclose(zb[0], -49.0)

    # the rate is LOCAL: the border cell's own value is used even when the
    # rest of the field (hence any mean) is wildly different
    zb = np.linspace(0.0, 500.0, n); zb[0] = -50.0
    bbu_field = np.full(n, 7.0); bbu_field[0] = 1e-3
    zb, _, _ = run(zb, ice, bbu=bbu_field, dt=1000.0)
    np.testing.assert_allclose(zb[0], -49.0)
    zb = np.linspace(0.0, 500.0, n); zb[0] = -0.5
    zb, _, _ = run(zb, ice, bbu=1e-3, dt=1000.0)
    assert zb[0] == 0.0

    # ice-free relict fjord: true-state output convention — EVERY cell (border
    # included) reports its true drowned bed, no water-line floor; bed memory
    # intact.
    fjord = np.array([-300.0, -250.0, -200.0, -100.0, 100.0, 500.0])
    zb, _, surf = run(fjord.copy(), np.zeros(n), bbu=0.0, dt=1000.0)
    np.testing.assert_array_equal(surf, fjord)   # border unfloored, whole fjord shows
    np.testing.assert_array_equal(zb, fjord)


@pytest.mark.parametrize("law", ["eff-exp", "power", "coulomb"])
def test_mode_b_walk_hc_residuals(law):
    """The hc threading uses the column substitution G = hc*H with rescaled
    Newton constants instead of touching the solver bodies. Pin the math:
    at hc = 1.5 every interior H returned by the walk must satisfy the
    DIRECT hc-residual of its law, with the receiver surface zb_r + hc*H_r:

      eff-exp:  H * (a + hc*H)^(2/3)            = (Qg/(cg lam^1.5))^(2/9) dx^(2/3)
      power:    H^4 (H^2+lam^2) (a + hc*H)^3    = Qg dx^3 / cg
      coulomb:  H^5 (a + hc*H)^3 (H + lam_c/(1-phi^3)) = Qg dx^3 / cg,
                phi = beta*H*(a + hc*H), beta = rho_g*g/(tau_c*dx)

    (independent of which HC_OVER_H ships — the substitution must be exact
    for any hc; hc = 1.5 exercises the rescaling, and a control run at
    hc = 1.0 must differ, proving the parameter is live)."""
    from ._kernel_adapters import (_diag_walk_eff_exp, _diag_walk_power,
                                   _diag_walk_coulomb)
    hc = 1.5
    nx, dx = 40, 100.0
    zb = np.linspace(0.0, 800.0, nx)
    Qg = np.linspace(8e5, 1e4, nx)
    cg, lam_p, lam_c, tau_c, rho_g_g, clamp = 2.3e-4, 300.0, 1e-3, 1e5, 9016.0, 1e-12

    def walk(hc_val):
        H = np.zeros(nx)
        args = dict(didx_l=nx - 1, didx_r=nx, nx=nx, hc=hc_val)
        if law == "eff-exp":
            _diag_walk_eff_exp(zb, Qg, H, cg, lam_p, dx, **args)
        elif law == "power":
            _diag_walk_power(zb, Qg, H, cg, lam_p, dx, **args)
        else:
            _diag_walk_coulomb(zb, Qg, H, cg, lam_c, tau_c, rho_g_g, dx,
                               clamp, **args)
        return H

    H = walk(hc)
    assert (H[1:] > 0.0).all()
    # Node 1's receiver is the zero-gradient OUTFLOW border (H[0] = H[1]),
    # solved with a one-pass lag by the 2-pass walk, so its residual against the
    # final H[0] carries that lag; the interior hc-substitution math is fully
    # exercised by nodes 2..nx-1 (receiver is an ordinary interior node).
    for i in range(2, nx):
        a = zb[i] - (zb[i - 1] + hc * H[i - 1])
        col = a + hc * H[i]
        assert col > 0.0
        if law == "eff-exp":
            lhs = H[i] * col ** (2.0 / 3.0)
            rhs = (Qg[i] / (cg * lam_p ** 1.5)) ** (2.0 / 9.0) * dx ** (2.0 / 3.0)
        elif law == "power":
            lhs = H[i] ** 4 * (H[i] ** 2 + lam_p ** 2) * col ** 3
            rhs = Qg[i] * dx ** 3 / cg
        else:
            beta = rho_g_g / (tau_c * dx)
            phi = beta * H[i] * col
            assert phi < 1.0
            lhs = H[i] ** 5 * col ** 3 * (H[i] + lam_c / (1.0 - phi ** 3))
            rhs = Qg[i] * dx ** 3 / cg
        np.testing.assert_allclose(lhs, rhs, rtol=1e-8, err_msg=f"node {i}")
    # the parameter is live: hc = 1 gives a different profile
    H1 = walk(1.0)
    assert np.max(np.abs(H - H1)) > 1e-3


@pytest.mark.parametrize("law", ["eff-exp", "power", "coulomb"])
def test_diag_walk_routes_through_shared_closure(law):
    """The 1D mode-B walk solves interior H through the SAME _modeb_closure
    the 2D kernels use (the law_code collapse unified the 1D/2D closure path
    and the power/coulomb K_p grouping). Pin an interior node bit-for-bit
    against a direct _modeb_closure call so the 1D grouping can't drift back."""
    from ._kernel_adapters import (_diag_walk_eff_exp, _diag_walk_power,
                                   _diag_walk_coulomb)
    from siim._core.solvers import (_modeb_closure, LAW_EFFEXP, LAW_POWER,
                                    LAW_COULOMB)
    nx, dx, hc = 6, 100.0, 1.5
    zb = np.array([0.0, 60.0, 140.0, 240.0, 360.0, 500.0])
    Qg = np.full(nx, 3e5)                       # ice everywhere, exits at i=0
    cg, lam_p, lam_c, tau_c, rho_g_g, clamp = 2.3e-4, 300.0, 1e-3, 1e5, 9016.0, 1e-12
    args = dict(didx_l=nx - 1, didx_r=nx, nx=nx, hc=hc)   # left block only
    H = np.zeros(nx)
    if law == "eff-exp":
        _diag_walk_eff_exp(zb, Qg, H, cg, lam_p, dx, **args)
        code, lp, lc, tc, rg, cl = LAW_EFFEXP, lam_p, 0.0, 0.0, 0.0, 0.0
    elif law == "power":
        _diag_walk_power(zb, Qg, H, cg, lam_p, dx, **args)
        code, lp, lc, tc, rg, cl = LAW_POWER, lam_p, 0.0, 0.0, 0.0, 0.0
    else:
        _diag_walk_coulomb(zb, Qg, H, cg, lam_c, tau_c, rho_g_g, dx, clamp, **args)
        code, lp, lc, tc, rg, cl = LAW_COULOMB, 0.0, lam_c, tau_c, rho_g_g, clamp
    i = 2                                       # interior node, receiver i-1
    a = zb[i] - (zb[i - 1] + hc * H[i - 1])
    want = _modeb_closure(code, False, a, Qg[i], dx, hc, cg, lp, lc, tc, rg, cl)
    assert H[i] == want                         # bit-for-bit: same closure


def test_modeC_carve_bed_is_topography():
    """Mode B + carve (the citizen GlacialSPLModeC): the carving path is a
    fastscape citizen too — topography IS the tracked bed, so there is NO
    separate bedrock_surface output, zb_out == topography__elevation, and the
    ice surface z_out is the reconstruction zb_out + hc*H_out. The bed is a
    clean tracked state (no snapshot-pairing flicker — the retired surface-
    replace class stored a bedrock_surface precisely to dodge that; the citizen
    doesn't need it). Regression for the 'landscape jumping' videos."""
    from siim.siim2d import siim as siim2d
    p = dict(U=1e-3, P=2, beta=1e-3, Ko=1e-6, n=1, ce=1e-4, nu=2,
             Ac=2e-24, lambda_p=5e2, alpha_g=8, zELA=300, T=2e4,
             Lx=3e4, Ly=3e4, nx=21, ny=21, nt=11, nt_out=6,
             D=1e-3, seed=3, boundary_status=['fixed_value'] * 4,
             initial_max_elevation=500, noise_amplitude=10, k=1,
             width_hack_k=1.0, width_hack_p=0.5, flow_routing='single',
             sliding_law='eff-exp', mode='B', carve_width=True,
             widening_rate=0.0, progress_bar=False)
    m = siim2d(p)
    m.run()
    assert m._citizen_mode_b                      # carve -> citizen ModeC path
    assert 'glacial_spl__bedrock_surface' not in m.ds_out
    np.testing.assert_array_equal(
        m.zb_out, m.ds_out['topography__elevation'].values)
    np.testing.assert_allclose(
        m.z_out, m.zb_out + m.hc_over_H * m.H_out, rtol=0, atol=0)


def test_citizen_modeB_bed_is_topography():
    """Citizen mode B (Fork B; no carve): topography IS the tracked bed, so
    there is NO separate bedrock_surface output, zb_out == topography__elevation
    and the ice surface z_out is the reconstruction zb_out + hc*H_out."""
    from siim.siim2d import siim as siim2d
    from siim import constants as _constants
    p = dict(U=1e-3, P=2, beta=1e-3, Ko=1e-6, n=1, ce=1e-4, nu=2,
             Ac=2e-24, lambda_p=5e2, alpha_g=8, zELA=300, T=2e4,
             Lx=3e4, Ly=3e4, nx=21, ny=21, nt=11, nt_out=6,
             D=1e-3, seed=3, boundary_status=['fixed_value'] * 4,
             initial_max_elevation=500, noise_amplitude=10, k=1,
             width_hack_k=1.0, width_hack_p=0.5, flow_routing='single',
             sliding_law='eff-exp', mode='B', carve_width=False,
             progress_bar=False)
    m = siim2d(p)
    m.run()
    assert m._citizen_mode_b
    assert 'glacial_spl__bedrock_surface' not in m.ds_out
    np.testing.assert_array_equal(
        m.zb_out, m.ds_out['topography__elevation'].values)
    np.testing.assert_allclose(
        m.z_out, m.zb_out + m.hc_over_H * m.H_out, rtol=0, atol=0)
    assert m.hc_over_H == float(_constants.HC_OVER_H)


# ---------------------------------------------------------------------------
# Mode A: surface-anchored base-level BC (zs pinned; outlet H from the
# per-law closure at the upwind surface slope)
# ---------------------------------------------------------------------------

def test_mode_a_outlet_surface_anchored_1d():
    """Mode A's state is the ice surface, pinned at base level. An outlet
    with through-flowing ice gets H from the same H(Q, S) closure as the
    interior, with S the upwind surface slope into the node."""
    from siim.siim1d import siim as siim1
    m = siim1({'zELA': -200.0, 'L': 2e4, 'nx': 201, 'T': 2e4, 'nt': 21,
               'mode': 'A',  # this test asserts the mode-A surface-anchored outlet H
               'sliding_law': 'eff-exp', 'left_bc': 'base_level',
               'right_bc': 'reflecting', 'progress_bar': False})
    m.run()
    assert m.Qg[0] > 0.0, "config must drive ice through the outlet"
    S0 = max(0.0, (m.z[1] - m.z[0]) / m.dx)
    assert S0 > 0.0
    want = (m.Qg[0] / (m.cg * m.lambda_p ** 1.5 * S0 ** 3)) ** (2.0 / 9.0)
    # H was computed from the pre-erosion surface of the final step; the
    # recomputation here uses the post-erosion surface — one erosion
    # increment apart, so compare at per-step accuracy.
    assert m.H[0] > 0.0
    np.testing.assert_allclose(m.H[0], want, rtol=5e-3)
    assert np.isfinite(m.tau[0]) and m.tau[0] > 0.0


@pytest.mark.adapter
def test_mode_a_border_surface_anchored_2d():
    """2D parity with the 1D outlet treatment: a self-receiving border node
    with ice flux gets H(Q, S_upwind) from the per-law closure."""
    pytest.importorskip('fastscape')
    from siim.fastscape.processes import GlacialSPLModeA
    from siim._core.params import GlacialParams
    from siim._core.solvers import LAW_EFFEXP
    p = object.__new__(GlacialSPLModeA)
    # The law constants + code now come from the GlacialLaw record the erosion
    # process reads via foreign (self._law_code, self._gp); set them directly here.
    p._law_code = LAW_EFFEXP
    p._gp = GlacialParams(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.3e-4, 0.0, 300.0,
        1e-3, 1e5, 1e-12, 9016.0, 1.5, 0.0)
    n = 6
    p.receivers = np.array([0, 0, 1, 2, 3, 4], dtype=np.int64)
    p.lengths = np.array([0.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    p.ice_flux = np.full(n, 5e5)
    z = np.linspace(0.0, 500.0, n)
    H = np.zeros(n)
    p._solve_border_H_modeA(z, H)
    S = (z[1] - z[0]) / 100.0
    want = (5e5 / (p._gp.cg * p._gp.lambda_p ** 1.5 * S ** 3)) ** (2.0 / 9.0)
    np.testing.assert_allclose(H[0], want, rtol=1e-12)
    assert np.all(H[1:] == 0.0)        # interior untouched by the border pass


# ---------------------------------------------------------------------------
# D-inf router and H diffusion on anisotropic grids
# ---------------------------------------------------------------------------

def _route(z, dx, dy):
    from siim._core.routing import (
        _dinf_route, _DINF_E1_DJ, _DINF_E1_DI, _DINF_E2_DJ, _DINF_E2_DI)
    ny, nx = z.shape
    nn = ny * nx
    interior = np.ones(nn, dtype=np.int8)
    interior[:nx] = 0
    interior[-nx:] = 0
    interior[::nx] = 0
    interior[nx - 1::nx] = 0
    r1 = np.zeros(nn, np.int64); r2 = np.zeros(nn, np.int64)
    w1 = np.zeros(nn); w2 = np.zeros(nn)
    l1 = np.zeros(nn); l2 = np.zeros(nn); s = np.zeros(nn)
    _dinf_route(z.ravel().astype(float), ny, nx, dx, dy, interior,
                r1, r2, w1, w2, l1, l2, s,
                _DINF_E1_DJ, _DINF_E1_DI, _DINF_E2_DJ, _DINF_E2_DI,
                False, False)
    return r1, r2, w1, w2, l1, l2, s


def test_dinf_anisotropic_plane():
    """dx=100, dy=10 (review repro): slopes are normalised by the right
    spacing (was 10x small for N/S gradients), N/S receiver lengths are dy,
    and an S-dominant plane routes overwhelmingly S (was 41%E / 59%SE)."""
    ny = nx = 9
    dx, dy = 100.0, 10.0
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    c = 4 * nx + 4

    z = -0.005 * (jj * dy)                       # pure +y descent
    r1, r2, w1, w2, l1, l2, s = _route(z, dx, dy)
    np.testing.assert_allclose(s[c], 0.005, rtol=1e-12)
    assert r1[c] - c == nx and l1[c] == dy and w1[c] == 1.0

    z = -0.001 * (ii * dx) - 0.005 * (jj * dy)   # S 5x steeper than E
    r1, r2, w1, w2, l1, l2, s = _route(z, dx, dy)
    np.testing.assert_allclose(s[c], np.hypot(0.001, 0.005), rtol=1e-12)
    frac_S = (w1[c] if r1[c] - c == nx else 0.0) + (w2[c] if r2[c] - c == nx else 0.0)
    assert frac_S > 0.8, f"S-fraction {frac_S} (anisotropy ignored again?)"


def test_dinf_square_grid_semantics_unchanged():
    """dx == dy reduces exactly to the original square-grid formulas."""
    ny = nx = 9
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    c = 4 * nx + 4
    z = -0.01 * (ii * 100.0)
    r1, r2, w1, w2, l1, l2, s = _route(z, 100.0, 100.0)
    assert r1[c] - c == 1 and w1[c] == 1.0
    np.testing.assert_allclose(s[c], 0.01, rtol=1e-12)
    z = -0.01 * (ii * 100.0) - 0.01 * (jj * 100.0)   # exact 45 degrees
    r1, r2, w1, w2, l1, l2, s = _route(z, 100.0, 100.0)
    diag_w = w2[c] if (r2[c] - c) == nx + 1 else w1[c]
    np.testing.assert_allclose(diag_w, 1.0, rtol=1e-9)
    np.testing.assert_allclose(s[c], 0.01 * np.sqrt(2.0), rtol=1e-12)


def test_h_diffusion_anisotropic():
    """Square grids reproduce the old kernel exactly; on dx >> dy the spread
    per cell is much larger along the fine axis (equal physical spread)."""
    from siim._core.diffusion import _diffuse_H_2d
    H = np.zeros((21, 21)); H[10, 10] = 100.0
    Hf = H.ravel().copy()
    _diffuse_H_2d(Hf, 21, 21, 100.0, 100.0, 100.0, 5.0)   # ax=ay=0.05, 1 sub-step
    ref = H.ravel().copy(); work = ref.copy(); a = 0.05
    for j in range(1, 20):
        for i in range(1, 20):
            k = j * 21 + i
            ref[k] = work[k] + a * (work[k - 21] + work[k + 21]
                                    + work[k - 1] + work[k + 1] - 4.0 * work[k])
    np.testing.assert_allclose(Hf, ref, rtol=0, atol=1e-12)

    Hf2 = H.ravel().copy()
    _diffuse_H_2d(Hf2, 21, 21, 100.0, 10.0, 100.0, 5.0)
    G = Hf2.reshape(21, 21)
    assert abs(G.sum() - 100.0) < 0.5          # boundary ring absorbs the tail
    assert G[8, 10] > 10 * G[10, 8]            # fine axis (y) spreads per-cell


# ---------------------------------------------------------------------------
# WaveUplift: exact passage integral, midpoint sampling
# ---------------------------------------------------------------------------

@pytest.mark.adapter
def test_wave_uplift_integrates_to_delta_h():
    """The Gaussian wave deposits exactly delta_h at every interior point the
    full wave passes (calibration 1.0; old default 1.2 overshot by 20%), and
    the wave center is sampled at the step midpoint (no one-step position
    bias)."""
    pytest.importorskip('fastscape')
    from siim.fastscape import WaveUplift
    w = object.__new__(WaveUplift)
    nx_, ny_ = 41, 3
    w.x = np.linspace(0.0, 400e3, nx_)
    w.y = np.linspace(0.0, 2e3, ny_)
    w.shape = (ny_, nx_)
    w.delta_h = 1500.0
    w.wave_width = 50e3
    w.wave_velocity = 15e-3
    w.x_escarpment = -200e3            # start the wave well left of the domain
    w.wave_calibration = 1.0
    w.U_inf = 0.0
    w._mask = np.ones(w.shape)         # skip border masking for the integral

    dt = 1000.0
    total = np.zeros(w.shape)
    n_steps = int((900e3 / w.wave_velocity) / dt)      # 60 Myr: wave crosses all x
    for k in range(n_steps):
        w.run_step(dt, k * dt)
        total += w.uplift
    # every interior column fully passed by the wave received delta_h
    np.testing.assert_allclose(total[1, 5:-5], w.delta_h, rtol=1e-3)

    # midpoint sampling: peak of the step-k rate sits at c(t + dt/2)
    w.run_step(dt, 0.0)
    X = w.x
    k_peak = int(np.argmax(w.uplift[1]))
    c_expected = w.x[0] + w.x_escarpment + 0.5 * dt * w.wave_velocity
    assert abs(X[k_peak] - max(c_expected, X[0])) <= (X[1] - X[0])


# ---------------------------------------------------------------------------
# Bracketed implicit-erosion Newtons (sub-linear exponents 2-cycled to
# silent zero erosion; power-law default t = 0.9 hit it)
# ---------------------------------------------------------------------------

def test_erosion_newton_bracketed_sublinear():
    """The solvers converge to the actual root for any G*dt; the old plain
    Newton 2-cycled for exponents < 1 and returned zo unchanged."""
    from siim._core.solvers import (
        _solver_fluvial, _solver_glacial, _solver_glacial_power)
    import siim.siim1d as s1
    zo, zr = 1000.0, 0.0
    prev = zo
    for Gi in np.logspace(-4, 6, 41):
        for solver in (_solver_glacial_power, s1._solver_glacial_power):
            z = solver(zo, zr, Gi, 0.9, 1e-8, 200)
            resid = z - zo + Gi * max(z - zr, 0.0) ** 0.9
            assert abs(resid) < 1e-4 * max(1.0, Gi), (Gi, z, resid)
            assert zr <= z <= zo
        assert z <= prev + 1e-9          # erosion monotonic in G*dt
        prev = z
    for Gi in np.logspace(-3, 6, 19):
        z = _solver_fluvial(zo, zr, Gi, 0.8, 1e-8, 200)
        assert abs(z - zo + Gi * max(z - zr, 0.0) ** 0.8) < 1e-4 * max(1.0, Gi)
        z = _solver_glacial(zo, zr, Gi, 0.9, 1e-8, 200)
        assert abs(z - zo + Gi * max(z - zr, 0.0) ** 0.9) < 1e-4 * max(1.0, Gi)
        z = s1._solver_fluvial(zo, zr, Gi, 0.8, 1e-8, 200)
        assert abs(z - zo + Gi * max(z - zr, 0.0) ** 0.8) < 1e-4 * max(1.0, Gi)


def test_erosion_monotonic_in_dt():
    """Review repro: a 6-node power-law chain eroded ~100 m/node at dt=1e4
    but EXACTLY 0 at dt=1e5 (the larger step pushed the Newton into its
    2-cycle). Total erosion must be positive and monotonic in dt."""
    from siim._core.eroders import _power_erode_2d
    n_nodes, L = 6, 100.0
    rec = np.array([0, 0, 1, 2, 3, 4], dtype=np.int64)
    stack = np.arange(n_nodes, dtype=np.int64)      # receivers before donors
    t = 0.9                                         # = 3*ell/2, ell = 0.6
    base = 1e-4 * (2.3e-4 * 300.0 ** 2 / 10.0 ** 2) ** 0.3

    def total_erosion(dt):
        z = np.linspace(0.0, 500.0, n_nodes).copy()
        zo = z.copy()
        Qf = np.full(n_nodes, 1e6)
        Qg = np.full(n_nodes, 1e7)
        H = np.full(n_nodes, 100.0)
        Kf = np.full(n_nodes, dt / L ** 1.0 * 1e-6)
        Kg = np.full(n_nodes, dt / L ** t * base)
        _power_erode_2d(z, zo, Qf, Qg, H, Kf, Kg, 0.5, 1.0, t, 300.0,
                        stack, rec)
        return float((zo - z)[1:].sum())

    eroded = [total_erosion(dt) for dt in (1e3, 1e4, 1e5, 1e6)]
    assert all(e > 0.0 for e in eroded), eroded
    assert all(b > a for a, b in zip(eroded, eroded[1:])), eroded


# ---------------------------------------------------------------------------
# Glacial erosion / flow-accumulator processes read the post-tectonics surface
# ---------------------------------------------------------------------------

@pytest.mark.adapter
def test_processes_read_post_tectonic_surface():
    import attr
    pytest.importorskip('fastscape')
    from fastscape.processes import SurfaceToErode
    from siim.fastscape.processes import (
        GlacialSPLModeA, GlacialSPLModeB, GlacialSPLModeC, GlacialFlowAccumulator)
    # The erosion processes inherit `surface` from GlacialSPLBase; assert it
    # resolves on each concrete process that fills a slot.
    for cls in (GlacialSPLModeA, GlacialSPLModeB, GlacialSPLModeC,
                GlacialFlowAccumulator):
        fld = attr.fields_dict(cls)['surface']
        assert fld.metadata.get('other_process_cls') is SurfaceToErode, (
            f"{cls.__name__}.surface must read SurfaceToErode (post-tectonics)")


def test_trace_paths_nan_receivers_self_receive():
    """NaN receivers (zarr round-trip artifacts at boundary nodes) map to
    SELF inside _trace_paths_arrays, not to node 0 — nan_to_num's 0 is
    the corner cell, which would grow a bogus path segment from nodes
    near the corner. The traced network must contain only channel nodes."""
    from siim.plotting._render import _trace_paths_arrays
    ny, nx = 9, 21
    Lx, Ly = 2000.0, 800.0
    # straight channel along row 4, flowing toward x=0
    rec = np.arange(ny * nx).reshape(ny, nx).astype(float)
    for c in range(1, nx):
        rec[4, c] = 4 * nx + (c - 1)
    rec[0, 0] = np.nan                      # zarr-style NaN at a boundary node
    H = np.zeros((ny, nx)); H[4, :] = 80.0
    area = np.zeros((ny, nx)); area[4, :] = np.linspace(1e5, 2e6, nx)[::-1]
    paths, xc, yc, H_flat = _trace_paths_arrays(
        rec, H, area, nx, ny, Lx, Ly, channel_threshold=1e5)
    nodes = {n for p in paths for n in p}
    assert nodes, "channel must be traced"
    channel_nodes = {4 * nx + c for c in range(nx)}
    assert nodes <= channel_nodes, (
        "traced paths leaked off the channel (NaN receiver pulled in the "
        f"corner?): extra nodes {sorted(nodes - channel_nodes)[:5]}")
