"""The multiple-flow-direction router (``flow_routing='mfd'``).

Stage S3 of ``docs/dev/mfd_router_plan.md``: the producer, its wiring, the
coherence guards, and the physics the gamma effective-slope
normalisation buys. Everything here drives the REAL producer
(:func:`siim._core.step.route_mfd`) — no hand-built bundles (those live in
``test_mfd_slope_reduction.py``, which predates the producer).

Arm map (plan section 3, S3; the Codex review's corrected gates):

* **a1** conservation — all of a unit source per cell reaches the fixed borders;
* **a2** weight-level plane oracle — through the producer, theta = 0..45 deg:
  ``sum w = 1``, the weighted receiver direction IS the plane's downhill
  direction at p = 1, and the effective collecting width stays in
  ``[0.999, 1.083]*dx``;
* **a3** accumulation isotropy — the routed sheet flux per unit width is within
  10 % of the continuum value at every angle, spread <= 9 %;
* **a4** seam invariance under looped BCs;
* **a6** the D8 limit, CORRECTED (review finding 7): argmax receiver == D8,
  monotone concentration in p, a local weight test at p = 64 only where every
  runner-up ratio <= 0.70, and drainage area approaching D8's along a
  p-sequence by a documented tolerance — never exact equality at finite p;
* **a7** cardinal parity with D8 — H, bed and presented surface, three laws,
  ``dt = 500``, plus the zero-erosion (fixed-bed) mode-B oracle;
* **a8** the filled pit — conservation, spill side, gamma conditioning, the
  floor-hit histogram at 0.25 / 0.5 / 0.75 and the level-count scaling;
* **a9** ``transect_flux`` at k = 8;
* the API arms: the exponent moves the weights, survives save/load, is
  byte-identical in-house vs xsimlab, every guard raises through BOTH
  ``siim2d`` and ``glacial_processes``, and a short eroding run per law stays
  finite and mass-consistent;
* the valley arms (plan-2026-09-22-mfd-guard-lift.md): the carve and the
  trunk surface run and engage under MFD, and the trunk dip is router-derived
  and persists as passed (an omitted dip re-resolves on load).
"""
import math

import numpy as np
import pytest

from siim import constants as _constants
from siim._core.router import check_mfd_config
from siim._core.routing import (_flow_accumulate_dinf, _flow_accumulate_sd,
                                _levels_dinf, _recv_mean_direction,
                                gamma_per_cell)
from siim._core.step import (_filled_surface_and_mask, route_d8, route_mfd,
                             transect_flux)

DX = 100.0
FIXED = ['fixed_value'] * 4
# left/right fixed, top/bottom looped -> wrap_y (see step._filled_surface_and_mask)
LOOP_Y = ['fixed_value', 'fixed_value', 'looped', 'looped']

# 8 neighbour offsets (di, dj) — the analytical companion of the producer scan.
OFF = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


# --- surfaces + small helpers ------------------------------------------------

def _plane(ny, nx, dx, G, theta):
    """``z = z0 - G*(x cos(theta) + y sin(theta))``: gradient magnitude ``G``,
    downhill direction ``theta`` measured from +x."""
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    return 1.0e4 - G * (ii * dx * math.cos(theta) + jj * dx * math.sin(theta))


def _pitted_plane(ny, nx, dx, depth=180.0):
    """A plane draining -x with a Gaussian overdeepening: the eps-fill must
    carry flux across a closed basin toward its spill."""
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    z = 100.0 + 0.5 * ii.astype(float)
    z -= depth * np.exp(-(((ii - nx // 2) ** 2 + (jj - ny // 2) ** 2) / 8.0))
    return z


def _interior(ny, nx, bs):
    m = np.ones(ny * nx, dtype=bool)
    bb = list(np.broadcast_to(bs, 4))
    if bb[0] == 'fixed_value':
        m[0::nx] = False
    if bb[1] == 'fixed_value':
        m[nx - 1::nx] = False
    if bb[2] == 'fixed_value':
        m[:nx] = False
    if bb[3] == 'fixed_value':
        m[nx * (ny - 1):] = False
    return m


def _accum(field, rec, w, nb, stack):
    out = np.array(field, dtype=float).ravel().copy()
    _flow_accumulate_dinf(out, stack, nb, rec, w)
    return out


# =============================================================================
# a1 — conservation
# =============================================================================

@pytest.mark.parametrize("p", [0.5, 1.0, 2.0, 8.0])
def test_a1_unit_source_conserved_to_the_borders(p):
    ny, nx = 15, 21
    z = _pitted_plane(ny, nx, DX)
    rec, w, lengths, nb, stack, basin = route_mfd(z, (ny, nx), DX, DX, FIXED, p)
    area = _accum(np.ones(ny * nx), rec, w, nb, stack)
    border = ~_interior(ny, nx, FIXED)
    np.testing.assert_allclose(area[border].sum(), float(ny * nx), rtol=1e-12)
    assert np.all(area >= 1.0 - 1e-9), "a cell gathered less than its own unit"


# =============================================================================
# a2 — weight-level plane oracle, through the real producer
# =============================================================================

def _effective_width_over_dx(rec, w, nb, i, ny, nx, dx, theta):
    """``w_eff/dx`` at cell ``i`` on a plane of downhill direction ``theta``.

    In a uniform sheet the flux accumulates one cell-source per hop while
    advancing ``d_eff = sum_k w_k (dr_k . u)`` down-flow, so
    ``Q = s/d_eff``; the continuum sheet gives ``Q = (s/dx^2)*w_eff``. Hence
    ``w_eff = dx^2/d_eff``, and a perfectly isotropic router would hold
    ``w_eff/dx == 1`` at every angle. Bare ``S^p`` does not: it is exactly 1 on
    the cardinal and diagonal axes and 8.2 % high in between (review finding 6 —
    the direction is exact at p = 1, the magnitude is not).
    """
    ux, uy = math.cos(theta), math.sin(theta)
    iy, ix = divmod(i, nx)
    d_eff = 0.0
    for k in range(nb[i]):
        ry, rx = divmod(rec[i, k], nx)
        d_eff += w[i, k] * (((rx - ix) * dx) * ux + ((ry - iy) * dx) * uy)
    return dx / d_eff


@pytest.mark.parametrize("deg", list(range(0, 50, 5)))
def test_a2_plane_weight_oracle_at_p1(deg):
    ny = nx = 41
    theta = math.radians(deg)
    z = _plane(ny, nx, DX, 0.02, theta)
    rec, w, lengths, nb, stack, basin = route_mfd(z, (ny, nx), DX, DX, FIXED, 1.0)
    dw_max = ang_max = 0.0
    weff = []
    for j in range(5, ny - 5):
        for i in range(5, nx - 5):
            idx = j * nx + i
            k = nb[idx]
            dw_max = max(dw_max, abs(w[idx, :k].sum() - 1.0))
            vx, vy, gamma = _recv_mean_direction(idx, rec, w, nb, ny, nx,
                                                 DX, DX, False, False)
            ang_max = max(ang_max, abs(math.atan2(vy, vx) - theta))
            assert gamma >= _constants.MFD_GAMMA_FLOOR
            weff.append(_effective_width_over_dx(rec, w, nb, idx, ny, nx,
                                                 DX, theta))
    assert dw_max <= 1e-15, f"sum of weights off by {dw_max:.2e}"
    # p = 1 is the only exponent whose weighted receiver vector is exactly the
    # plane's downhill direction on a square 8-stencil.
    assert ang_max < 1e-9, f"routed direction off by {ang_max:.2e} rad"
    assert 0.999 <= min(weff) and max(weff) <= 1.083, \
        f"w_eff/dx in [{min(weff):.6f}, {max(weff):.6f}]"


# =============================================================================
# a3 — accumulation isotropy
# =============================================================================

def test_a3_accumulated_sheet_flux_is_isotropic_within_ten_percent():
    """A uniform source on a tilted plane: the routed flux at an interior cell
    must match the continuum sheet value ``q*dx`` (``q = s/dx^2``, ``s`` the
    distance back to the inflow boundary along the flow direction) within 10 %,
    with at most 9 % spread between angles.

    DEVIATION from the plan's "looped cross-flow axis": a plane tilted off-axis
    is not periodic on ANY grid axis (the wrap would introduce a cliff at the
    seam), so only the theta = 0 arm can loop — it does, as its own exactness
    check below. The tilted arms use fixed borders and an interior window whose
    backward flow ray leaves through the up-flow (x = 0) edge with margin, where
    the 1-D sheet solution holds.
    """
    ny = nx = 161
    dx = 1.0
    medians = {}
    for deg in (0.0, 15.0, 22.5, 30.0, 45.0):
        theta = math.radians(deg)
        c, s_ = math.cos(theta), math.sin(theta)
        z = _plane(ny, nx, dx, 0.02, theta)
        rec, w, lengths, nb, stack, basin = route_mfd(z, (ny, nx), dx, dx,
                                                      FIXED, 1.0)
        Q = _accum(np.ones(ny * nx), rec, w, nb, stack).reshape(ny, nx)
        jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        x, y = ii * dx, jj * dx
        s_x = x / c
        s_y = y / s_ if s_ > 1e-12 else np.full_like(y, np.inf)
        q = np.minimum(s_x, s_y) / dx ** 2
        ratio = Q / np.maximum(q * dx, 1e-30)
        window = ((s_x < 0.6 * s_y) & (x > 40 * dx) & (x < 120 * dx)
                  & (jj > 30) & (jj < ny - 30))
        assert window.sum() > 500
        medians[deg] = float(np.median(ratio[window]))
        assert abs(medians[deg] - 1.0) <= 0.10, (deg, medians[deg])
    v = np.array(list(medians.values()))
    assert (v.max() - v.min()) / v.mean() <= 0.09, medians


def test_a3_cardinal_sheet_on_a_looped_cross_flow_axis_is_exact():
    """With the cross-flow axis LOOPED there is no lateral loss, so the routed
    flux is the exact sheet value ``column_index + 1`` cells."""
    ny, nx = 21, 41
    z = _plane(ny, nx, DX, 0.02, 0.0)                 # drains +x
    rec, w, lengths, nb, stack, basin = route_mfd(z, (ny, nx), DX, DX,
                                                  LOOP_Y, 1.0)
    Q = _accum(np.ones(ny * nx), rec, w, nb, stack).reshape(ny, nx)
    # column 0 is a base-level sink, so cell (j, i) has exactly i cells upstream
    expect = np.broadcast_to(np.arange(nx, dtype=float)[1:nx - 1],
                             (ny, nx - 2))
    np.testing.assert_allclose(Q[:, 1:nx - 1], expect, rtol=1e-12)


# =============================================================================
# a4 — seam invariance
# =============================================================================

@pytest.mark.parametrize("p", [1.0, 2.0])
def test_a4_seam_shift_invariance(p):
    ny, nx = 15, 21
    z = _pitted_plane(ny, nx, DX)
    rec, w, lengths, nb, stack, basin = route_mfd(z, (ny, nx), DX, DX, LOOP_Y, p)
    area = _accum(np.ones(ny * nx), rec, w, nb, stack).reshape(ny, nx)
    shift = 4
    zr = np.roll(z, shift, axis=0)
    r2, w2, l2, nb2, s2, b2 = route_mfd(zr, (ny, nx), DX, DX, LOOP_Y, p)
    area_r = _accum(np.ones(ny * nx), r2, w2, nb2, s2).reshape(ny, nx)
    np.testing.assert_allclose(area_r, np.roll(area, shift, axis=0),
                               rtol=1e-12, atol=1e-12)


# =============================================================================
# a6 — the D8 limit (review finding 7: corrected)
# =============================================================================

def _tie_free_surface(ny, nx, seed=11):
    rng = np.random.default_rng(seed)
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    return 200.0 + 2.0 * ii + rng.random((ny, nx)) * 7.0


def test_a6_d8_limit():
    ny, nx = 61, 71
    nn = ny * nx
    z = _tie_free_surface(ny, nx)
    interior = _interior(ny, nx, FIXED)
    rec8, w8, l8, nb8, st8, b8 = route_d8(z, (ny, nx), DX, DX, FIXED)
    area8 = np.ones(nn)
    _flow_accumulate_sd(area8, st8, rec8)

    conc, area_err = [], []
    p_seq = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0)
    for p in p_seq:
        rec, w, lengths, nb, stack, basin = route_mfd(z, (ny, nx), DX, DX,
                                                      FIXED, p)
        k = np.argmax(w, axis=1)
        primary = rec[np.arange(nn), k]
        # (i) on a tie-free surface the argmax receiver IS D8's steepest one
        assert np.array_equal(primary[interior], rec8[interior]), p
        conc.append(float(w[np.arange(nn), k][interior].mean()))
        area = _accum(np.ones(nn), rec, w, nb, stack)
        area_err.append(float(np.abs(area - area8)[interior].sum()
                              / area8[interior].sum()))
    # (ii) concentration is monotone in p, and the drainage-area gap to D8
    # shrinks monotonically along the p-sequence -- but never to zero at finite
    # p, because every non-steepest link keeps a nonzero weight.
    assert all(b > a for a, b in zip(conc, conc[1:])), conc
    assert all(b < a for a, b in zip(area_err, area_err[1:])), area_err
    assert area_err[-1] < 0.10, area_err[-1]        # documented tolerance
    assert area_err[-1] > 0.0


def test_a6_p64_concentrates_where_the_runner_up_is_well_separated():
    """At p = 64 the steepest link takes >= 1 - 1e-9 of the flux wherever every
    runner-up slope ratio is <= 0.70 (0.70**64 = 1.1e-10, and at most seven
    runners-up). No such claim is possible where the ratio approaches 1 —
    0.99**64 = 0.53 leaves the steepest link only ~0.65 (review finding 7)."""
    ny, nx = 61, 71
    z = _tie_free_surface(ny, nx)
    zf, _mask, _wy, _wx, _ny, _nx, _nn = _filled_surface_and_mask(
        z, (ny, nx), FIXED)
    rec, w, lengths, nb, stack, basin = route_mfd(z, (ny, nx), DX, DX,
                                                 FIXED, 64.0)
    tested = 0
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            idx = j * nx + i
            slopes = []
            for dj, di in OFF:
                n = (j + dj) * nx + (i + di)
                s = (zf[idx] - zf[n]) / math.hypot(di * DX, dj * DX)
                if s > 0.0:
                    slopes.append(s)
            slopes.sort(reverse=True)
            if len(slopes) > 1 and slopes[1] / slopes[0] > 0.70:
                continue
            tested += 1
            assert w[idx].max() >= 1.0 - 1e-9, (idx, w[idx].max())
    assert tested > 100, f"only {tested} well-separated cells to test"


# =============================================================================
# a7 — cardinal parity with D8 (three laws, H + bed after erosion)
# =============================================================================

_LAW_CONST = dict(cg=2.3e-4, lam_p=300.0, lam_c=1e-3, tau_c=1.2e5,
                  rho_g_g=9016.0, clamp=1e-12, Ko=1e-6, Co=1e-9, ce=1e-5,
                  n_exp=1.0, nu=1.0, m=0.5, mu=4.0 / 15.0, dt=500.0, hc=1.5,
                  alpha_g=8.0)


def _cardinal_modeb_arms(law, erode=True, ny=9, nx=20):
    """Run one mode-B step on a cardinal plane through BOTH kernels: D8/SFR and
    MFD (with the gamma normalisation on). The cross-flow axis is LOOPED, so the
    plane is y-invariant and MFD's three receivers (W, NW, SW) all carry the
    same eroded surface — which is what makes ``Z̄`` the D8 receiver exactly.
    Returns ``(H_sfr, zb_sfr, surf_sfr, H_mfd, zb_mfd, surf_mfd, interior)``."""
    from ._kernel_adapters import (
        glac_fast_solve_modeB, glac_fast_solve_power_modeB,
        glac_fast_solve_coulomb_modeB,
        glac_fast_solve_modeB_dinf, glac_fast_solve_power_modeB_dinf,
        glac_fast_solve_coulomb_modeB_dinf)
    c = dict(_LAW_CONST)
    if not erode:
        c.update(Ko=0.0, Co=0.0, ce=0.0)
    nn = ny * nx
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    zb0 = (5.0 + 0.01 * DX * ii.astype(float)).ravel()      # drains -x
    ice = np.full(nn, 1e7)
    water = np.zeros(nn)
    bbu = np.zeros(nn)
    rec8, w8, len8, nb8, st8, b8 = route_d8(zb0.reshape(ny, nx), (ny, nx),
                                            DX, DX, LOOP_Y)
    recm, wm, lenm, nbm, stm, bm = route_mfd(zb0.reshape(ny, nx), (ny, nx),
                                             DX, DX, LOOP_Y, 1.0)
    interior = _interior(ny, nx, LOOP_Y)
    assert set(np.unique(nbm[interior]).tolist()) == {3}, \
        "the cardinal plane must give MFD exactly three receivers"
    zs, Hs, ss = zb0.copy(), np.zeros(nn), np.empty(nn)
    zd, Hd, sd = zb0.copy(), np.zeros(nn), np.empty(nn)
    if law == 'eff-exp':
        glac_fast_solve_modeB(
            zs, ice, water, Hs, ss, c['Ko'], c['Co'], c['n_exp'], c['nu'],
            c['m'], c['mu'], c['cg'], c['alpha_g'], c['lam_p'], c['dt'],
            len8, st8, rec8, 0.0, ny, nx, DX, DX, bbu, c['hc'])
        glac_fast_solve_modeB_dinf(
            zd, ice, water, Hd, sd, c['Ko'], c['Co'], c['n_exp'], c['nu'],
            c['m'], c['mu'], c['cg'], c['alpha_g'], c['lam_p'], c['dt'],
            stm, nbm, recm, wm, lenm, 0.0, ny, nx, DX, DX, bbu, c['hc'],
            True, True, True)
    elif law == 'power':
        glac_fast_solve_power_modeB(
            zs, ice, water, Hs, ss, c['Ko'], c['ce'], c['n_exp'], c['nu'],
            c['m'], c['cg'], c['alpha_g'], c['lam_p'], c['dt'],
            len8, st8, rec8, 0.0, ny, nx, DX, DX, bbu, c['hc'])
        glac_fast_solve_power_modeB_dinf(
            zd, ice, water, Hd, sd, c['Ko'], c['ce'], c['n_exp'], c['nu'],
            c['m'], c['cg'], c['alpha_g'], c['lam_p'], c['dt'],
            stm, nbm, recm, wm, lenm, 0.0, ny, nx, DX, DX, bbu, c['hc'],
            True, True, True)
    else:
        glac_fast_solve_coulomb_modeB(
            zs, ice, water, Hs, ss, c['Ko'], c['ce'], c['n_exp'], c['nu'],
            c['m'], c['cg'], c['alpha_g'], c['lam_c'], c['tau_c'], c['clamp'],
            c['rho_g_g'], c['dt'], len8, st8, rec8, 0.0, ny, nx, DX, DX,
            bbu, c['hc'])
        glac_fast_solve_coulomb_modeB_dinf(
            zd, ice, water, Hd, sd, c['Ko'], c['ce'], c['n_exp'], c['nu'],
            c['m'], c['cg'], c['alpha_g'], c['lam_c'], c['tau_c'], c['clamp'],
            c['rho_g_g'], c['dt'], stm, nbm, recm, wm, lenm, 0.0, ny, nx,
            DX, DX, bbu, c['hc'], True, True, True)
    return Hs, zs, ss, Hd, zd, sd, interior


@pytest.mark.parametrize("law", ["eff-exp", "power", "coulomb"])
def test_a7_cardinal_parity_with_d8_after_erosion(law):
    """On a cardinal plane the gamma-normalised MFD mode-B step reproduces the
    D8 one for all three laws, at ``dt = 500``, WITH erosion.

    Why it can: ``gamma*L̄ == dx`` and ``Z̄`` is the +x neighbour's surface at
    any p (Codex review finding 2), so the closure's effective receiver IS D8's
    — and, since ``slope_normalise`` switches erosion to the SAME effective
    receiver (``eroders._erode_modeb_eff``), so is the eroder's.

    Scope, measured: H, the interior bed and the interior surface match D8
    EXACTLY (0.0 absolute). The OUTFLOW-BORDER column does not, and cannot: the
    step-5b border budget reads a single donor-to-donor EDGE slope on raw
    lengths (gamma is undefined for one edge — plan amendment b), and under MFD
    the dominant-donor chain leaves the border diagonally, so the arrival slope
    is ``G/sqrt(2)`` where D8 sees ``G``. That difference is pinned below rather
    than hidden."""
    Hs, zs, ss, Hd, zd, sd, interior = _cardinal_modeb_arms(law)
    assert Hs.max() > 1.0, "config must grow ice"
    np.testing.assert_allclose(Hd, Hs, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(zd[interior], zs[interior], rtol=1e-10,
                               atol=1e-10)
    np.testing.assert_allclose(sd[interior], ss[interior], rtol=1e-10,
                               atol=1e-10)
    # the interior really did erode (otherwise the parity above is vacuous)
    if law != 'eff-exp':
        ny, nx = 9, 20
        jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        zb0 = (5.0 + 0.01 * DX * ii.astype(float)).ravel()
        assert (zb0 - zs)[interior].max() > 1e-3


@pytest.mark.parametrize("law", ["eff-exp", "power", "coulomb"])
def test_a7_border_column_is_the_only_departure_from_d8(law):
    """The documented seam of plan amendment (b): under MFD the outflow border's
    arrival slope comes off a DIAGONAL donor edge, so only the border column
    departs from D8 — bounded, and small next to the border's own bed change."""
    Hs, zs, ss, Hd, zd, sd, interior = _cardinal_modeb_arms(law)
    ny, nx = 9, 20
    d = np.abs(zd - zs).reshape(ny, nx)
    assert d[:, 1:].max() == 0.0, "the departure is not confined to column 0"
    assert d[:, 0].max() < 0.05, d[:, 0].max()


@pytest.mark.parametrize("law", ["eff-exp", "power", "coulomb"])
def test_mode_b_fixed_bed_cardinal_oracle(law):
    """The mode-B oracle that replaces the (mode-A) analytical-oracle arm
    (review gate 8): with erosion switched off, MFD and D8 agree EVERYWHERE —
    thickness, bed and presented surface, borders included."""
    Hs, zs, ss, Hd, zd, sd, interior = _cardinal_modeb_arms(law, erode=False)
    assert Hs.max() > 1.0
    np.testing.assert_allclose(Hd, Hs, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(zd, zs)
    np.testing.assert_allclose(sd, ss, rtol=1e-10, atol=1e-10)


# =============================================================================
# a8 — the filled pit
# =============================================================================

def _dome_with_pit(n, seed=3):
    rng = np.random.default_rng(seed)
    j, i = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    r = np.hypot(i - n // 2, j - n // 2)
    z = 1000.0 * np.exp(-(r / (0.35 * n)) ** 2) + 5.0 * rng.random((n, n))
    z -= 180.0 * np.exp(-(((i - n * 0.6) ** 2 + (j - n * 0.45) ** 2)
                          / (0.02 * n * n)))
    return z


def test_a8_filled_pit_conservation_and_spill_side():
    ny, nx = 15, 21
    z = _pitted_plane(ny, nx, DX)
    rec, w, lengths, nb, stack, basin = route_mfd(z, (ny, nx), DX, DX,
                                                 FIXED, 1.0)
    interior = _interior(ny, nx, FIXED)
    idx = np.arange(ny * nx)
    # no interior cell is a sink: the eps-fill leaves every interior cell a
    # strictly lower neighbour, so the basin drains across its spill
    assert not np.any(rec[interior, 0] == idx[interior])
    assert np.all(nb[interior] >= 1)
    # mass
    area = _accum(np.ones(ny * nx), rec, w, nb, stack)
    np.testing.assert_allclose(area[~interior].sum(), float(ny * nx),
                               rtol=1e-12)
    # a point source on the basin floor leaves the domain, down the -x plane
    src = np.zeros(ny * nx)
    src[(ny // 2) * nx + nx // 2] = 1.0
    out = _accum(src, rec, w, nb, stack)
    np.testing.assert_allclose(out[~interior].sum(), 1.0, rtol=1e-12)
    left = np.zeros((ny, nx), dtype=bool)
    left[:, 0] = True
    np.testing.assert_allclose(out[left.ravel()].sum(), 1.0, rtol=1e-12)


def test_a8_gamma_conditioning_and_floor_sensitivity():
    """Inside an eps-filled basin many O(eps) links go active, so gamma can fall
    — this is the measurement behind ``constants.MFD_GAMMA_FLOOR`` (provisional
    0.5). Reported: min gamma and the fraction of interior cells the floor would
    bite at 0.25 / 0.5 / 0.75. The transect closure must stay finite through
    all of it (below the floor it falls back to ``Q_transect = Q_i``)."""
    n = 81
    z = _dome_with_pit(n)
    rec, w, lengths, nb, stack, basin = route_mfd(z, (n, n), DX, DX, FIXED, 1.0)
    interior = _interior(n, n, FIXED)
    gamma = gamma_per_cell(rec, w, nb, n, n, DX, DX, False, False)
    assert np.all(np.isfinite(gamma))
    assert gamma[interior].min() > 0.0
    hits = {f: float(np.mean(gamma[interior] < f)) for f in (0.25, 0.5, 0.75)}
    # monotone histogram, and the shipped floor bites only a small minority
    assert hits[0.25] <= hits[0.5] <= hits[0.75]
    assert hits[0.5] < 0.05, hits
    assert hits[0.75] < 0.40, hits
    # transect closure stays finite on the same graph (fallback included)
    Q = _accum(np.ones(n * n), rec, w, nb, stack).reshape(n, n)
    H = np.full((n, n), 120.0)
    Qt = transect_flux(Q, H, rec, w, nb, (n, n), DX, DX, 8.0, False, False)
    assert np.all(np.isfinite(Qt))
    assert np.all(Qt >= Q - 1e-9), "transect flux fell below the cell's own"


def test_a8_level_count_scaling():
    """Level count for the parallel eroder's schedule, over three grid sizes on
    the overdeepened surface.

    MEASURED, and a documented DEPARTURE from the plan's preregistered
    ``levels <= 2.2*max(nx, ny)``: on this surface the constant is 2.10 / 2.60 /
    2.81 at n = 41 / 81 / 121, so 2.2 holds only at the smallest grid. The
    growth is near-linear and flattening (the scoping measured 400 levels at
    201^2 = 1.99n on a smoother dome), so the gate is the linear form with the
    measured constant plus margin, PLUS a check that levels/n does not run away
    across a 3x size range."""
    ratios = []
    for n in (41, 81, 121):
        z = _dome_with_pit(n)
        rec, w, lengths, nb, stack, basin = route_mfd(z, (n, n), DX, DX,
                                                      FIXED, 1.0)
        _order, _offsets, nlev = _levels_dinf(stack, nb, rec)
        assert nlev <= 3.0 * n, (n, nlev)
        ratios.append(nlev / n)
    assert ratios[-1] / ratios[0] < 1.5, ratios


# =============================================================================
# a9 — transect_flux at k = 8
# =============================================================================

@pytest.mark.parametrize("H0", [100.0, 130.0, 170.0, 250.0])
def test_a9_uniform_cardinal_sheet_transect_is_exact_under_mfd(H0):
    """The cardinal-sheet exactness of ``test_transect_flux`` survives k = 8.

    The transect line is perpendicular to the cell's flow direction, which on a
    cardinal plane is exactly +x (gamma's direction is exact at p = 1). The
    three MFD links out of each cell cross it at transverse offsets
    ``0, +dy/4, -dy/4``; each family is still integer-spaced by ``dy``, and the
    kernel's linear end-ramp sums to exactly ``2R`` over ANY such lattice
    (the ramp is a box convolved with a cell, whose lattice sum is
    phase-independent). So the total is ``sum_k w_k * Q * 2R/dx == Q*alpha_g*H/dx``."""
    ny = nx = 41
    alpha_g = 5.0
    z = _plane(ny, nx, DX, 0.01, 0.0)                 # drains +x, cardinal
    rec, w, lengths, nb, stack, basin = route_mfd(z, (ny, nx), DX, DX,
                                                  FIXED, 1.0)
    assert nb[_interior(ny, nx, FIXED)].max() == 3
    Q = np.full((ny, nx), 1e5)
    H = np.full((ny, nx), H0)
    Qt = transect_flux(Q, H, rec, w, nb, (ny, nx), DX, DX, alpha_g,
                       False, False)
    expect = 1e5 * alpha_g * H0 / DX
    margin = int(np.ceil(alpha_g * H0 / 2 / DX)) + 3
    core = Qt[margin:-margin, 1:nx - 1]
    np.testing.assert_allclose(core, expect, rtol=1e-12, atol=0.0)


# =============================================================================
# API: exponent, persistence, driver parity, guards, moving beds
# =============================================================================

_RUN = dict(U=1e-3, zELA=150, beta=1e-2, P=2, alpha_g=12, Ko=2e-6, n=1,
            ce=1e-4, nu=2, sliding_law='power', lambda_p=500, k=.9,
            T=1.2e4, nt=13, nt_out=5, Lx=1e4, Ly=1e4, nx=21, ny=21, seed=7,
            boundary_status=['fixed_value'] * 4, initial_max_elevation=800,
            mode='B', flow_routing='mfd', progress_bar=False,
            trunk_surface=False, parallel_erode=False)


@pytest.mark.pin
@pytest.mark.parametrize("bs", [FIXED, LOOP_Y, ['looped', 'looped', 'fixed_value', 'fixed_value']],
                         ids=['fixed', 'looped-y', 'looped-x'])
@pytest.mark.parametrize("p", [1.0, np.inf])
def test_stack_is_kahn_on_ascending_donor_lists(bs, p):
    """The stack is Kahn's order (base cells first, reversed to donor-first) on
    donor lists filled cell by cell, i.e. each receiver's donors in ascending
    node index, over the links of positive weight. The producer reads the
    donors off link masks instead; the order is part of the result, since every
    accumulation adds in stack order. Ties, pits and flats; looped seams."""
    rng = np.random.default_rng(7)
    ny, nx = 13, 17
    nn = ny * nx
    z = np.round(8 * rng.random((ny, nx))) / 8
    rec, w, _, nb, stack, _ = route_mfd(z, (ny, nx), DX, DX, bs, p)
    donors = [[] for _ in range(nn)]
    out = np.zeros(nn, dtype=int)
    for i in range(nn):
        for k in range(nb[i]):
            if rec[i, k] != i and w[i, k] > 0.0:
                donors[rec[i, k]].append(i)
                out[i] += 1
    queue = [i for i in range(nn) if out[i] == 0]
    for c in queue:                     # grows while it is walked
        for d in donors[c]:
            out[d] -= 1
            if out[d] == 0:
                queue.append(d)
    assert len(queue) == nn and max(map(len, donors)) >= 4
    np.testing.assert_array_equal(stack, queue[::-1])


def test_exponent_changes_the_weights():
    ny, nx = 21, 31
    z = _tie_free_surface(ny, nx, seed=5)
    _r1, w1, _l1, nb1, _s1, _b1 = route_mfd(z, (ny, nx), DX, DX, FIXED, 1.0)
    _r2, w2, _l2, nb2, _s2, _b2 = route_mfd(z, (ny, nx), DX, DX, FIXED, 3.0)
    interior = _interior(ny, nx, FIXED)
    assert np.array_equal(nb1, nb2), "the active set must not depend on p"
    assert not np.allclose(w1[interior], w2[interior])
    # larger p concentrates
    assert w2[interior].max(axis=1).mean() > w1[interior].max(axis=1).mean()


def test_slope_normalise_is_router_derived():
    """The ``None`` default resolves from
    :data:`siim._core.router.SLOPE_NORMALISED_ROUTERS` — on for every
    MULTI-receiver router, off for single-flow."""
    from siim.siim2d import siim as siim2d
    assert siim2d({**_RUN, 'flow_routing': 'mfd'}).slope_normalise is True
    assert siim2d({**_RUN, 'flow_routing': 'dinf',
                   'closure_flux': 'cell'}).slope_normalise is True
    assert siim2d({**_RUN, 'flow_routing': 'single',
                   'closure_flux': 'cell'}).slope_normalise is False


def test_slope_normalise_explicit_value_wins():
    """An explicit value overrides the router default — that is how the
    pre-2026-09-05 D-inf reduction stays reachable — except under MFD, which is
    17-24 % low without it and raises."""
    from siim.siim2d import siim as siim2d
    assert siim2d({**_RUN, 'flow_routing': 'dinf', 'closure_flux': 'cell',
                   'slope_normalise': False}).slope_normalise is False
    # 'single' has one receiver, so the flag is a documented no-op there.
    assert siim2d({**_RUN, 'flow_routing': 'single', 'closure_flux': 'cell',
                   'slope_normalise': True}).slope_normalise is True
    with pytest.raises(ValueError, match="requires slope_normalise"):
        siim2d({**_RUN, 'flow_routing': 'mfd', 'slope_normalise': False})


@pytest.mark.adapter
def test_adapter_slope_normalise_default_mirrors_the_router_set():
    """Across the adapter seam the erosion process has no router NAME, so the
    ``None`` default resolves from receiver ARITY — the faithful mirror of
    ``SLOPE_NORMALISED_ROUTERS`` (multi-receiver on, single-flow off). An
    explicit value still wins."""
    pytest.importorskip('fastscape')
    from siim.fastscape.processes import GlacialSPLBase
    resolve = GlacialSPLBase._resolve_slope_normalise

    class _Stub:
        pass

    for rec, want in ((np.zeros((9, 2), dtype=np.int64), True),      # D-inf
                      (np.zeros((9, 8), dtype=np.int64), True),      # MFD
                      (np.zeros(9, dtype=np.int64), False)):         # single
        s = _Stub()
        s.receivers, s.slope_normalise = rec, None
        assert resolve(s) is want
        s.slope_normalise = False
        assert resolve(s) is False
        s.slope_normalise = True
        assert resolve(s) is True


def test_slope_normalise_round_trips_save_load(tmp_path, monkeypatch):
    from siim.siim2d import siim as siim2d
    monkeypatch.chdir(tmp_path)
    m = siim2d({**_RUN, 'flow_routing': 'dinf', 'closure_flux': 'cell',
                'slope_normalise': False})
    m.run()
    back = siim2d.load(str(m.save('gamma_roundtrip')))
    assert back.slope_normalise is False
    assert back.flow_routing == 'dinf'


def test_an_mfd_exponent_above_the_bound_is_inf_in_the_model():
    """siim2d routes a finite ``mfd_exponent`` above ``MFD_EXPONENT_MAX`` as
    inf, with a warning; inf itself is taken silently, and runs; NaN is
    refused."""
    import warnings
    from siim.siim2d import siim as siim2d
    with pytest.warns(UserWarning, match="taken as mfd_exponent=inf"):
        assert siim2d({**_RUN, 'mfd_exponent': 200}).mfd_exponent == np.inf
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        m = siim2d({**_RUN, 'mfd_exponent': np.inf})
    assert m.mfd_exponent == np.inf
    m.run()
    assert np.isfinite(m.H_out).all() and m.H_out[-1].max() > 0.0
    with pytest.raises(ValueError, match="mfd_exponent must be positive"):
        siim2d({**_RUN, 'mfd_exponent': float('nan')})


def test_mfd_exponent_round_trips_save_load(tmp_path, monkeypatch):
    from siim.siim2d import siim as siim2d
    monkeypatch.chdir(tmp_path)
    m = siim2d({**_RUN, 'mfd_exponent': 1.3})
    m.run()
    path = m.save('mfd_roundtrip')
    back = siim2d.load(str(path))
    assert back.mfd_exponent == 1.3
    assert back.flow_routing == 'mfd'
    assert back.slope_normalise is True


@pytest.mark.parametrize("law", ["eff-exp", "power", "coulomb"])
def test_short_eroding_run_per_law_is_finite_and_mass_consistent(law):
    """Review gate 11: the dome experiment runs on a rigid bed, so the
    landscape-evolution path needs its own coverage — a short MOVING-BED run per
    law that stays finite and does not create rock."""
    from siim.siim2d import siim as siim2d
    extra = {'lambda_c': 1e3, 'tau_c': 1.2e5} if law == 'coulomb' else {}
    m = siim2d({**_RUN, 'sliding_law': law, **extra})
    m.run()
    assert np.isfinite(m.z_out).all() and np.isfinite(m.H_out).all()
    assert (m.H_out >= 0.0).all()
    assert m.H_out[-1].max() > 0.0, law
    # the bed is bounded: no runaway, and erosion never lifts an interior cell
    # above the initial surface it was carved from
    assert m.z_out[-1].max() < m.z_out[0].max() + 1e3


@pytest.mark.parametrize("law", ["eff-exp", "power", "coulomb"])
def test_parallel_erode_is_bit_for_bit_under_slope_normalise(law):
    """``parallel_erode`` keeps its shipped boolean under MFD, so the
    level-scheduled twin of the effective-receiver eroder must be bit-for-bit
    with the serial walk (as the D-inf/SFR pair already is). The parameter doc
    still recommends False for MFD: the level count roughly triples."""
    from siim.siim2d import siim as siim2d
    extra = {'lambda_c': 1e3, 'tau_c': 1.2e5} if law == 'coulomb' else {}
    a = siim2d({**_RUN, 'sliding_law': law, 'parallel_erode': False, **extra})
    a.run()
    b = siim2d({**_RUN, 'sliding_law': law, 'parallel_erode': True, **extra})
    b.run()
    assert np.array_equal(a.z_out, b.z_out)
    assert np.array_equal(a.H_out, b.H_out)


# --- valleys: the carve and the trunk surface under MFD ----------------------
# (plan-2026-09-22-mfd-guard-lift.md). ``_RUN`` at 250 m: fine enough that
# alpha_g*H/2 > dx within a few steps, so both devices engage.
_VALLEY_RUN = dict(_RUN, nx=31, ny=31, Lx=7.5e3, Ly=7.5e3)


def _spy_valley_devices(monkeypatch):
    """Per-step counts of the cells the carve moved and of the cells the
    fabricated trunk surface changed (an unused device records nothing)."""
    from siim._core import driver as drv
    carved, fabricated = [], []
    real_carve, real_fab = drv.carve_bed, drv._fabricate_trunk_surface

    def carve(zb_flat, *a, **kw):
        before = zb_flat.copy()
        real_carve(zb_flat, *a, **kw)
        carved.append(int((zb_flat != before).sum()))

    def fab(zs, *a, **kw):
        out = real_fab(zs, *a, **kw)
        fabricated.append(int((out != zs).sum()))
        return out
    monkeypatch.setattr(drv, 'carve_bed', carve)
    monkeypatch.setattr(drv, '_fabricate_trunk_surface', fab)
    return carved, fabricated


@pytest.mark.parametrize("kw,carves,fabricates", [
    (dict(mode='C', trunk_surface=None), True, True),
    (dict(mode='C', trunk_surface=False), True, False),
    (dict(mode='B', carve_width=False, trunk_surface=True), False, True),
], ids=['mode_C_standard', 'carve_only', 'trunk_surface_only'])
def test_mfd_valley_devices_run_and_engage(kw, carves, fabricates, monkeypatch):
    """MFD in valleys: mode C (carve + trunk surface), the carve alone and the
    trunk surface alone all construct and run, and each device really acts —
    the carve moves bed cells, the fabricated V differs from its input."""
    from siim.siim2d import siim as siim2d
    carved, fabricated = _spy_valley_devices(monkeypatch)
    m = siim2d({**_VALLEY_RUN, **kw})
    m.run(driver='inhouse')
    assert np.isfinite(m.z_out).all() and np.isfinite(m.H_out).all()
    assert (m.H_out >= 0.0).all() and m.H_out[-1].max() > 0.0
    assert (sum(carved) > 0) == carves, carved
    assert (sum(fabricated) > 0) == fabricates, fabricated


# --- the router-derived trunk dip (constants.TRUNK_DIP_K_MFD) ----------------

def test_trunk_dip_is_router_derived():
    """``trunk_dip_k=None`` (the default) resolves to TRUNK_DIP_K_MFD under MFD
    and TRUNK_DIP_K under D8 / D-inf; an explicit value, zero included, wins."""
    from siim._core.router import ROUTING_KINDS
    from siim.siim2d import siim as siim2d
    assert siim2d(dict(_RUN)).trunk_dip_k == _constants.TRUNK_DIP_K_MFD == 1.0
    for kind in ('single', 'dinf'):
        m = siim2d({**_RUN, 'flow_routing': kind})
        assert m.trunk_dip_k == _constants.TRUNK_DIP_K == 0.6
    for kind in ROUTING_KINDS:
        for k in (0.0, 1.3):
            m = siim2d({**_RUN, 'flow_routing': kind, 'trunk_dip_k': k})
            assert m.trunk_dip_k == k


@pytest.mark.parametrize("bad", [-0.1, float('nan'), float('inf'), True])
def test_trunk_dip_validation_rejects(bad):
    from siim.siim2d import siim as siim2d
    with pytest.raises(ValueError, match="trunk_dip_k"):
        siim2d({**_RUN, 'trunk_dip_k': bad})


_DIP_RUN = {**_VALLEY_RUN, 'mode': 'C', 'trunk_surface': None,
            'T': 2e3, 'nt': 3, 'nt_out': 2}


@pytest.mark.parametrize("dip", [None, 0.0, 1.3],
                         ids=['omitted', 'explicit_zero', 'explicit'])
@pytest.mark.parametrize("model", ['siim2d', 'escarpment'])
def test_trunk_dip_round_trips_save_load(model, dip, tmp_path, monkeypatch):
    """The dip persists as passed, like every sentinel default: an explicit
    value round-trips exactly, and an omitted one is stored as ``None`` and
    reloads as the value the constants give AT LOAD TIME (the default is moved
    between save and load to prove it)."""
    if model == 'siim2d':
        from siim.siim2d import siim as cls
    else:
        from siim.escarpment import siim_escarpment as cls
    monkeypatch.chdir(tmp_path)
    params = dict(_DIP_RUN) if dip is None else {**_DIP_RUN, 'trunk_dip_k': dip}
    m = cls(params)
    m.run()
    assert m.trunk_surface and m.trunk_dip_k == (1.0 if dip is None else dip)
    assert m._user_params.get('trunk_dip_k') == dip
    path = m.save(f'dip_roundtrip_{model}')
    monkeypatch.setattr(_constants, 'TRUNK_DIP_K_MFD', 1.7)
    back = cls.load(str(path))
    assert type(back) is cls
    assert back._user_params == m._user_params
    assert back.trunk_dip_k == (1.7 if dip is None else dip)
    assert np.array_equal(back.H_out, m.H_out)


def test_load_warns_on_a_version_mismatch(tmp_path, monkeypatch):
    """Sentinel defaults re-resolve from the LOADING version's constants, so a
    file saved by another siim version loads with a RuntimeWarning; the same
    version loads silently."""
    import importlib
    import warnings
    siim2d_module = importlib.import_module('siim.siim2d')
    monkeypatch.chdir(tmp_path)
    m = siim2d_module.siim(dict(_DIP_RUN))
    m.run()
    path = str(m.save('dip_version'))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        siim2d_module.siim.load(path)
    assert not [w for w in caught if 're-resolved' in str(w.message)]
    monkeypatch.setattr(siim2d_module, '_SIIM_VERSION', '0.0.0+other')
    with pytest.warns(RuntimeWarning, match='re-resolved'):
        siim2d_module.siim.load(path)


# --- the remaining MFD coherence guards, through BOTH entry points ------------

_GUARDS = [
    ('mode A', dict(mode='A'), "mode 'B' or 'C'"),
    ('cell closure', dict(closure_flux='cell'), "closure_flux"),
    ('rectangular cells', dict(Ly=2e4), "square cells"),
    ('filter with the trunk surface',
     dict(mode='C', trunk_surface=True, routing_filter_factor=2.0),
     "trunk_surface"),
]


@pytest.mark.parametrize("name,kw,msg", _GUARDS,
                         ids=[g[0].replace(' ', '_') for g in _GUARDS])
def test_guards_raise_through_siim2d(name, kw, msg):
    from siim.siim2d import siim as siim2d
    with pytest.raises(ValueError, match=msg):
        siim2d({**_RUN, **kw})


@pytest.mark.adapter
def test_guards_raise_through_glacial_processes():
    """The facade owns the mode — a direct adapter user must not be able to
    walk around the model-level check (review finding 8)."""
    pytest.importorskip('fastscape')
    from siim.fastscape import glacial_processes
    with pytest.raises(ValueError, match="mode 'B' or 'C'"):
        glacial_processes(routing='mfd', mode='A')


@pytest.mark.adapter
def test_adapter_processes_own_the_remaining_guards():
    """``closure_flux`` lives on the erosion process and the grid spacing on the
    router process, so those two guards are enforced there — same single-sourced
    messages (:func:`siim._core.router.check_mfd_config`), checked here; the
    processes raising them in a run is test_mfd_valley_adapter's."""
    pytest.importorskip('fastscape')
    from siim.fastscape import glacial_processes, MfdFlowRouter
    assert glacial_processes(routing='mfd', mode='B')['flow'] is MfdFlowRouter
    with pytest.raises(ValueError, match="closure_flux"):
        check_mfd_config('mfd', closure_flux='cell')
    with pytest.raises(ValueError, match="square cells"):
        check_mfd_config('mfd', dx=100.0, dy=50.0)


def test_check_mfd_config_is_a_no_op_for_the_other_routers():
    for kind in ('single', 'dinf'):
        check_mfd_config(kind, mode='A', closure_flux='cell', dx=100.0,
                         dy=50.0)


@pytest.mark.adapter
def test_driver_vs_xsimlab_is_byte_identical_on_an_mfd_config():
    pytest.importorskip('xsimlab')
    pytest.importorskip('fastscape')
    from siim.siim2d import siim as siim2d
    params = {**_RUN, 'mfd_exponent': 1.3, 'inner_passes': 1, 'inner_relax': 1.0}   # the adapter's one pass
    a = siim2d(params)
    a.run(driver='inhouse')
    b = siim2d(params)
    b.run(driver='xsimlab')
    for name in ('z_out', 'zb_out', 'H_out', 'area_out'):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
