"""The gamma effective-slope normalisation, through the REAL reduction.

A multi-receiver router reduces a cell's surface gradient to one number::

    S_raw = sum_k w_k * (zs_i - zs_rk) / L_k = g . v ,   v = sum_k w_k * e_k

which is the gradient PROJECTED on the weighted receiver vector, not its
magnitude.  Dividing by ``gamma = |v|`` returns ``|g| * cos(delta)``, delta the
angle between g and v -- exact when v is parallel to g.  siim applies that as
per-cell scaled link lengths ``L_eff = gamma * L``
(:func:`siim._core.routing.gamma_scaled_lengths`), so the closure reduction
picks up ``L_bar_eff = gamma * L_bar`` (with ``Z_bar`` unchanged) and the
eroders pick up the same effective slope.

There is no MFD producer yet (stage S3), so these tests BUILD the receiver
bundles by hand, in the producer contract's layout -- active receivers in the
contiguous prefix, zeroed tail slots, ``receivers[i, 0] == i`` only at borders
-- and drive them through the real
:func:`siim._core.skeleton._dinf_modeB_recv` and the real gamma helpers.

Accuracy, measured here and stated up front (mfd_router_plan.md section 2.4,
Codex review finding 1).  ``S_raw / gamma == |g| cos(delta)`` is an IDENTITY on
any grid, so it is gated at 1e-12 everywhere.  Whether that recovers ``|g|``
itself depends on v being parallel to g, which holds EXACTLY only for bare
``S^1`` weights on a square 8-neighbour stencil:

    square grid   p = 0.5   0.25 %      p = 1     exact
                  p = 1.1   0.0017 %    p = 2     0.0094 %    p = 4  0.0014 %
    dx != dy      p = 1     4-8 %   <- why v1 of the MFD router requires dx == dy
"""
import math

import numpy as np
import pytest

from siim import constants as _constants
from siim._core.router import SLOPE_NORMALISED_ROUTERS
from siim._core.routing import (_recv_mean_direction, gamma_per_cell,
                                gamma_scaled_lengths)
from siim._core.skeleton import _dinf_modeB_recv
from siim._core.step import route_d8, route_dinf

# 8 neighbour offsets in CELLS, (ox, oy); order is irrelevant to every quantity
# tested here (the reduction and gamma are both sums over the active set).
OFF = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
NSLOT = 8

# Documented approximation bound on |S_corrected / G - 1| over theta in [0, 45],
# square grid.  These are measurements, not targets -- see the module docstring.
SQUARE_BOUND = {0.5: 2.5e-3, 1.0: 1e-12, 1.1: 2.0e-5, 2.0: 1.0e-4, 4.0: 2.0e-5}


# --- hand-built MFD bundles, in the producer contract's layout ----------------

def plane_surface(ny, nx, dx, dy, G, theta):
    """z = z0 - G*(x cos(theta) + y sin(theta)): a plane of gradient magnitude
    G whose downhill direction is theta (measured from +x)."""
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    return (1.0e4 - G * (ii * dx * math.cos(theta) + jj * dy * math.sin(theta))
            ).astype(float)


def mfd_bundle(z, dx, dy, p):
    """Hand-build an ``(nn, 8)`` MFD receiver bundle on ``z``.

    Weights ``(S_k / S_max)**p`` over the STRICTLY downhill 8-neighbours,
    normalised to 1.  Contract layout: active receivers fill the contiguous
    prefix ``0 .. nb-1``, tail slots are zeroed in all three arrays, and a
    border cell (any cell on the grid edge) carries the slot-0 self sentinel.
    Deliberately NOT the layout of docs/dev/ice_sheet_dome_probes/_mfdprobe_plane
    et al., whose directional slots leave inactive self-receivers inside the
    active range (Codex review finding 4).
    """
    ny, nx = z.shape
    nn = ny * nx
    zf = z.ravel()
    receivers = np.zeros((nn, NSLOT), dtype=np.int64)
    weights = np.zeros((nn, NSLOT))
    lengths = np.zeros((nn, NSLOT))
    nb = np.zeros(nn, dtype=np.int64)
    for iy in range(ny):
        for ix in range(nx):
            i = iy * nx + ix
            if iy in (0, ny - 1) or ix in (0, nx - 1):
                receivers[i, 0] = i          # border sentinel
                weights[i, 0] = 1.0
                lengths[i, 0] = 0.0
                nb[i] = 1
                continue
            cand = []
            for ox, oy in OFF:
                j = (iy + oy) * nx + (ix + ox)
                L = math.hypot(ox * dx, oy * dy)
                S = (zf[i] - zf[j]) / L
                if S > 0.0:
                    cand.append((j, S, L))
            assert cand, f"interior cell {i} has no downhill neighbour"
            s_max = max(c[1] for c in cand)
            raw = [(c[1] / s_max) ** p for c in cand]
            tot = sum(raw)
            for k, (c, w) in enumerate(zip(cand, raw)):
                receivers[i, k] = c[0]
                weights[i, k] = w / tot
                lengths[i, k] = c[2]
            nb[i] = len(cand)
    return receivers, weights, lengths, nb


def reduction(z, receivers, weights, lengths, nb, i):
    """The REAL closure reduction at cell i: ``(S, L_bar, Z_bar)``.

    Driven through :func:`siim._core.skeleton._dinf_modeB_recv` with zero ice
    and ``hc_over_H = 1``, so the reduced surface IS z.
    """
    zf = np.ascontiguousarray(z.ravel())
    H = np.zeros_like(zf)
    a, L_bar, ok = _dinf_modeB_recv(zf, H, i, nb, receivers, weights,
                                    lengths, np.ones_like(zf))   # per-cell datum 1
    assert ok
    return a / L_bar, L_bar, zf[i] - a


def analytic_cos_delta(weights, lengths, receivers, nb, i, nx, dx, dy, theta):
    """``cos(delta)``, delta the angle between the gradient (pointing downhill
    at ``theta``) and ``v = sum_k w_k e_k`` -- built from the geometry alone,
    independent of the kernel under test."""
    iy, ix = divmod(i, nx)
    vx = vy = 0.0
    for k in range(nb[i]):
        ry, rx = divmod(receivers[i, k], nx)
        ddx, ddy = (rx - ix) * dx, (ry - iy) * dy
        L = math.hypot(ddx, ddy)
        vx += weights[i, k] * ddx / L
        vy += weights[i, k] * ddy / L
    return math.cos(math.atan2(vy, vx) - theta)


# --- (i) p = 1, square grid: direction and corrected slope are exact ----------

THETAS = [math.radians(d) for d in range(0, 46, 5)]


@pytest.mark.parametrize('theta_deg', list(range(0, 46, 5)))
def test_p1_square_direction_and_corrected_slope_are_exact(theta_deg):
    """Bare ``S^1`` on a square 8-stencil is the one scheme whose weighted
    receiver vector is EXACTLY parallel to the gradient, so the routed
    direction reproduces the plane's tilt and the gamma-corrected reduction
    reproduces |grad z| itself."""
    ny = nx = 9
    dx = dy = 100.0
    G, theta = 0.01, math.radians(theta_deg)
    z = plane_surface(ny, nx, dx, dy, G, theta)
    rec, wts, lens, nb = mfd_bundle(z, dx, dy, p=1.0)
    i = (ny // 2) * nx + nx // 2

    vx, vy, gamma = _recv_mean_direction(i, rec, wts, nb, ny, nx, dx, dy,
                                         False, False)
    assert abs(math.atan2(vy, vx) - theta) < 1e-9, "routed direction off-plane"

    # Raw: a projected gradient, 17-24 % low at p = 1.
    s_raw, _, _ = reduction(z, rec, wts, lens, nb, i)
    assert 0.76 * G < s_raw < 0.83 * G
    assert s_raw == pytest.approx(G * gamma, rel=1e-12)

    # Corrected, through the real scaled-length path.
    lens_eff = gamma_scaled_lengths(rec, wts, nb, lens, ny, nx, dx, dy,
                                    False, False)
    s_fix, _, _ = reduction(z, rec, wts, lens_eff, nb, i)
    assert s_fix == pytest.approx(G, rel=1e-12)


# --- (ii) other p: the identity is exact; recovery of |g| is approximate ------

@pytest.mark.parametrize('p', [0.5, 1.1, 2.0, 4.0])
def test_other_exponents_match_the_cos_delta_oracle(p):
    """For p != 1 the corrected reduction is ``|g| cos(delta)``, NOT ``|g|``.
    The identity is gated at 1e-12; the residual ``|cos(delta) - 1|`` is only
    checked against its documented, measured bound."""
    ny = nx = 9
    dx = dy = 100.0
    G = 0.01
    worst = 0.0
    for theta in THETAS:
        z = plane_surface(ny, nx, dx, dy, G, theta)
        rec, wts, lens, nb = mfd_bundle(z, dx, dy, p=p)
        i = (ny // 2) * nx + nx // 2
        lens_eff = gamma_scaled_lengths(rec, wts, nb, lens, ny, nx, dx, dy,
                                        False, False)
        s_fix, _, _ = reduction(z, rec, wts, lens_eff, nb, i)
        oracle = G * analytic_cos_delta(wts, lens, rec, nb, i, nx, dx, dy, theta)
        assert s_fix == pytest.approx(oracle, rel=1e-12), (
            f"p={p} theta={math.degrees(theta):.0f}: implementation != "
            "|g|*cos(delta)")
        worst = max(worst, abs(s_fix / G - 1.0))
    assert worst <= SQUARE_BOUND[p], (
        f"p={p}: recovery of |g| drifted to {100 * worst:.4f} % "
        f"(documented bound {100 * SQUARE_BOUND[p]:.4f} %)")


# --- (iii) rectangular cells: the identity holds, the recovery does not -------

@pytest.mark.parametrize('aspect', [2.0, 4.0])
def test_rectangular_cells_lose_gradient_recovery(aspect):
    """On dx != dy the weighted receiver vector is no longer parallel to the
    gradient, so gamma-correction recovers ``|g| cos(delta)`` (still exact) but
    misses ``|g|`` by 4-8 % at p = 1.  Stage S3 REJECTS dx != dy under MFD for
    exactly this reason (mfd_router_plan.md section 2.4); the case is pinned
    here so the rejection has a measured number behind it."""
    ny = nx = 9
    dx, dy = 100.0, 100.0 / aspect
    G = 0.01
    worst = 0.0
    for theta in THETAS:
        z = plane_surface(ny, nx, dx, dy, G, theta)
        rec, wts, lens, nb = mfd_bundle(z, dx, dy, p=1.0)
        i = (ny // 2) * nx + nx // 2
        lens_eff = gamma_scaled_lengths(rec, wts, nb, lens, ny, nx, dx, dy,
                                        False, False)
        s_fix, _, _ = reduction(z, rec, wts, lens_eff, nb, i)
        # The helper still computes exactly what the plan says it computes.
        oracle = G * analytic_cos_delta(wts, lens, rec, nb, i, nx, dx, dy, theta)
        assert s_fix == pytest.approx(oracle, rel=1e-12)
        worst = max(worst, abs(s_fix / G - 1.0))
    # The measured anisotropy error: 4.2 % at 2:1, 8.5 % at 4:1 (Codex review).
    assert 0.03 < worst < 0.10, f"aspect {aspect}: max error {100 * worst:.2f} %"


# --- (iv) the D-inf control, through the live producer ------------------------

def test_dinf_mid_facet_slope_is_normalised_by_default():
    """D-inf's mid-facet projection bias, and the flip that removed it
    (2026-09-05, docs/dev/dinf_slope_normalisation.md).

    On a plane tilted pi/8 -- mid-facet, D-inf's worst angle, both receivers
    22.5 deg off the gradient -- the UNNORMALISED reduction returns
    0.9239*|grad z|, a 7.6 % slope deficit that inflates H by ~5 % there.  The
    gamma-normalised reduction, which is now the DEFAULT for D-inf, returns
    1.0000*|grad z|.  ``slope_normalise=False`` is the control arm that keeps
    the old number reachable."""
    assert 'dinf' in SLOPE_NORMALISED_ROUTERS, \
        "D-inf is normalised by default since 2026-09-05"
    ny = nx = 21
    dx = dy = 100.0
    G, theta = 0.01, math.pi / 8
    z = plane_surface(ny, nx, dx, dy, G, theta)
    rec, wts, lens, nb, _stack, _basin = route_dinf(
        z, (ny, nx), dx, dy, ['fixed_value'] * 4)
    i = (ny // 2) * nx + nx // 2
    assert nb[i] == 2, "pi/8 is mid-facet: D-inf must split between two receivers"

    # The CONTROL arm (slope_normalise=False): the pre-flip reduction.
    s_raw, _, _ = reduction(z, rec, wts, lens, nb, i)
    assert s_raw / G == pytest.approx(0.9239, abs=5e-5)

    # The DEFAULT arm: exact on a plane.
    lens_eff = gamma_scaled_lengths(rec, wts, nb, lens, ny, nx, dx, dy,
                                    False, False)
    s_fix, _, _ = reduction(z, rec, wts, lens_eff, nb, i)
    assert s_fix / G == pytest.approx(1.0, abs=1e-9)

    # The floor can never bite under D-inf: two receivers 45 deg apart give
    # gamma >= cos(22.5 deg) = 0.9239, well above MFD_GAMMA_FLOOR.
    g = gamma_per_cell(rec, wts, nb, ny, nx, dx, dy, False, False)
    interior = np.ones((ny, nx), dtype=bool)
    interior[0] = interior[-1] = False
    interior[:, 0] = interior[:, -1] = False
    assert g.reshape(ny, nx)[interior].min() > _constants.MFD_GAMMA_FLOOR


# --- (v) the cardinal-plane identity: the foundation of the S3 parity gate ----

@pytest.mark.parametrize('p', [1.0, 2.0, 4.0])
def test_cardinal_plane_effective_receiver_is_d8s(p):
    """On a cardinal +x plane the MFD receivers are E, NE and SE, whose surfaces
    are IDENTICAL, so ``Z_bar`` is the +x neighbour's surface and
    ``gamma * L_bar == dx`` EXACTLY, at any p.  The gamma-corrected effective
    receiver therefore IS D8's receiver -- which is what lets stage S3 extend
    the existing 1e-10 cardinal-plane D8 parity oracle to MFD."""
    ny = nx = 9
    dx = dy = 100.0
    G = 0.01
    z = plane_surface(ny, nx, dx, dy, G, 0.0)
    rec, wts, lens, nb = mfd_bundle(z, dx, dy, p=p)
    i = (ny // 2) * nx + nx // 2
    assert nb[i] == 3, "a cardinal +x plane must route to exactly E, NE, SE"

    lens_eff = gamma_scaled_lengths(rec, wts, nb, lens, ny, nx, dx, dy,
                                    False, False)
    s_fix, L_bar_eff, z_bar = reduction(z, rec, wts, lens_eff, nb, i)

    assert L_bar_eff == pytest.approx(dx, rel=1e-14)
    assert z_bar == pytest.approx(z.ravel()[i + 1], rel=1e-14)
    assert s_fix == pytest.approx(G, rel=1e-14)


# --- the substitution's reach, measured (an S3 input) -------------------------

@pytest.mark.parametrize('law_exponent', [1.0, 4.0 / 3.0, 2.0])
def test_scaled_lengths_normalise_erosion_exactly_only_at_exponent_one(
        law_exponent):
    """Scaling ``lengths`` is EXACTLY the gamma normalisation wherever a kernel
    reads them at exponent 1 -- the closure reduction (``sum w/L``), mode A's
    weighted-mean slope, the linear eroder, and Coulomb's own slope
    reconstruction ``S_eff = sum_k w_k dz_k/L_k``.

    The nonlinear eroder residuals are NOT of that form: they sum
    ``sum_k w_k (dz_k/L_k)**p``, so gamma-scaled lengths give
    ``gamma**-p * sum_k w_k S_k**p`` where the normalised effective slope would
    be ``(gamma**-1 * sum_k w_k S_k)**p``.  Jensen separates the two at p != 1.
    This test MEASURES that gap on the cardinal plane, because stage S3's a7
    gate (full three-law D8 parity at rtol 1e-10 on bed after erosion) can only
    hold for exponent-1 laws unless the eroders are restructured.
    """
    ny = nx = 9
    dx = dy = 100.0
    G = 0.01
    z = plane_surface(ny, nx, dx, dy, G, 0.0)
    rec, wts, lens, nb = mfd_bundle(z, dx, dy, p=1.0)
    i = (ny // 2) * nx + nx // 2
    lens_eff = gamma_scaled_lengths(rec, wts, nb, lens, ny, nx, dx, dy,
                                    False, False)

    zf = z.ravel()
    p = law_exponent
    # What the eroders actually integrate, with gamma-scaled lengths.
    got = sum(wts[i, k] * ((zf[i] - zf[rec[i, k]]) / lens_eff[i, k]) ** p
              for k in range(nb[i]))
    # What a genuinely normalised effective slope would give (D8's answer here,
    # since the cardinal effective receiver IS D8's).
    want = G ** p

    if p == 1.0:
        assert got == pytest.approx(want, rel=1e-14)
    else:
        # Non-trivial, bounded, and biased HIGH (Jensen, since S_k/S_max <= 1
        # and the map is convex in the exponent's tail).
        assert got != pytest.approx(want, rel=1e-6)
        assert 1.0 < got / want < 1.15


# --- the flag is live, not dead code -----------------------------------------

def _modeB_arm(slope_normalise=None):
    """One mode-B D-inf kernel step on a mid-facet plane, with and without the
    normalisation; ``None`` omits the argument (the kernel default). Uses D-inf
    receivers because no MFD producer exists yet (stage S3); the point is the
    PLUMBING, not the router."""
    from siim._core.params import GlacialParams
    from siim._core.solvers import LAW_POWER
    from siim._core.step import run_modeB_kernel

    ny = nx = 21
    dx = dy = 100.0
    z = plane_surface(ny, nx, dx, dy, 0.05, math.pi / 8)
    z -= z.min()
    rec, wts, lens, nb, stack, _basin = route_dinf(
        z, (ny, nx), dx, dy, ['fixed_value'] * 4)
    gp = GlacialParams(Ko=2e-6, ce=1e-4, n=1.0, nu=2.0, m=0.5, cg=2.3e-4,
                       alpha_g=12.0, lambda_p=500.0, hc_over_H=1.5, D_H=0.0)
    zb = np.ascontiguousarray(z.ravel())
    H = np.full(ny * nx, 50.0)
    ice = np.full((ny, nx), 1.0e8)
    water = np.full((ny, nx), 1.0e6)
    flag = {} if slope_normalise is None else dict(slope_normalise=slope_normalise)
    run_modeB_kernel(
        zb, H, ice, water, LAW_POWER, gp, 500.0, stack, rec, nb, wts, lens,
        (ny, nx), dx, dy, np.zeros((ny, nx)), 0.0, True,
        _constants.FLOTATION_RAMP, False, False, False, **flag)
    return zb.copy(), H.copy()


def _modeB_fixed_point(theta, router, slope_normalise, ny=61, nx=61, dx=100.0,
                       G=0.05, nit=40):
    """Interior mode-B thickness on a RIGID plane of gradient ``G`` tilted
    ``theta``, iterated to the closure's fixed point (the bed is restored every
    step, and the law carries no erosion, so only H relaxes).  Uniform imposed
    ice flux, so at the fixed point every interior cell's H is set by the
    closure at that cell's effective slope alone."""
    from siim._core.params import GlacialParams
    from siim._core.solvers import LAW_POWER
    from siim._core.step import run_modeB_kernel

    z = plane_surface(ny, nx, dx, dx, G, theta)
    z -= z.min()
    route = route_dinf if router == 'dinf' else route_d8
    rec, wts, lens, nb, stack, _b = route(z, (ny, nx), dx, dx,
                                          ['fixed_value'] * 4)
    gp = GlacialParams(Ko=0.0, ce=0.0, n=1.0, nu=2.0, m=0.5, cg=2.3e-4,
                       alpha_g=12.0, lambda_p=1e-3, hc_over_H=1.5, D_H=0.0)
    H = np.full(ny * nx, 50.0)
    ice = np.full((ny, nx), 1.0e8)
    water = np.zeros((ny, nx))
    for _ in range(nit):
        zb = np.ascontiguousarray(z.ravel())          # rigid bed
        run_modeB_kernel(
            zb, H, ice, water, LAW_POWER, gp, 500.0, stack, rec, nb, wts, lens,
            (ny, nx), dx, dx, np.zeros((ny, nx)), 0.0, True,
            _constants.FLOTATION_RAMP, False, False, False,
            slope_normalise=slope_normalise)
    interior = np.zeros((ny, nx), dtype=bool)
    interior[12:-12, 12:-12] = True
    return float(np.median(H[interior.ravel()]))


def test_mid_facet_modeB_thickness_moves_toward_the_cardinal_oracle():
    """The bias, end to end in the mode-B kernel.

    ORACLE: the same rigid plane tilted along a grid axis, routed by D8 — where
    the effective slope IS |grad zs| exactly, for any router.  Tilt the same
    plane to pi/8 and route it with D-inf: the UNNORMALISED closure reads
    0.9239*|grad zs| and, since H ~ (Q/(cg S^3))^(1/5), returns a thickness
    (1/0.9239)^(3/5) = 1.049 times the oracle's.  Under the DEFAULT gamma
    normalisation that error collapses to well under 1 %.

    Measured 2026-09-05: raw +4.4 %, normalised +0.7 % (the residual is the
    second-order asymmetry of the two-receiver fan, not the projection)."""
    oracle = _modeB_fixed_point(0.0, 'd8', False)
    raw = _modeB_fixed_point(math.pi / 8, 'dinf', False)
    fixed = _modeB_fixed_point(math.pi / 8, 'dinf', True)

    predicted = (1.0 / math.cos(math.pi / 8)) ** 0.6      # 1.0487
    assert raw / oracle == pytest.approx(predicted, rel=0.02), (
        f"raw mid-facet H is {100 * (raw / oracle - 1):.2f} % high; the "
        f"projection bias predicts {100 * (predicted - 1):.2f} %")
    assert abs(fixed / oracle - 1.0) < 0.015
    assert abs(fixed / oracle - 1.0) < 0.35 * abs(raw / oracle - 1.0), (
        "the normalisation must move mid-facet thickness TOWARD the oracle "
        f"(raw {raw / oracle:.4f}, normalised {fixed / oracle:.4f})")


@pytest.mark.parametrize('theta_deg', [0, 90])
def test_cardinal_plane_modeB_is_bit_for_bit_under_the_flip(theta_deg):
    """gamma == 1 identically on a cardinal plane (D-inf collapses to a single
    receiver with weight 1), so turning the normalisation on must not move a
    single bit -- which is why the cardinal-plane parity gates survive the
    default flip."""
    off = _modeB_fixed_point(math.radians(theta_deg), 'dinf', False, nit=6)
    on = _modeB_fixed_point(math.radians(theta_deg), 'dinf', True, nit=6)
    assert on == off
    assert on == _modeB_fixed_point(math.radians(theta_deg), 'd8', False, nit=6)


def test_slope_normalise_flag_reaches_the_kernel():
    """``slope_normalise=True`` must change the thickness closure AND the bed;
    ``False`` must be bit-for-bit with the default. Without this the flag could
    be plumbed to nowhere and every other test here would still pass."""
    zb_off, H_off = _modeB_arm(False)
    zb_on, H_on = _modeB_arm(True)

    zb_def, H_def = _modeB_arm()
    # gate: was False vs False (a determinism check); compare with the omitted-argument default (audit #21)
    assert np.array_equal(zb_off, zb_def) and np.array_equal(H_off, H_def), \
        "slope_normalise=False is not bit-for-bit with the kernel default"
    # gamma = 0.9239 at mid-facet => S rises 8.2 % => H falls ~5 % (q ~ H^5 S^3).
    icy = H_off > 0.0
    assert icy.sum() > 100
    ratio = H_on[icy] / H_off[icy]
    assert 0.93 < np.median(ratio) < 0.99, f"median H ratio {np.median(ratio)}"
    assert not np.array_equal(zb_on, zb_off), "the bed must move too"
