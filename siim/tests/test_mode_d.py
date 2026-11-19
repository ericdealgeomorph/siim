"""Mode D: mode C with the valley <-> sheet reading choosing, per cell and per
step, whether the valley devices or the sheet treatment act
(``_memory/decisions/2026-11-13-channel-sheet-discharge-reading.md``).

1. the reading against a plain numpy oracle, exact; regions join across
   looped edges; a thick outlet reads channel; the bed dial is monotone;
2. the swap and the datum's smoothing;
3. the diagnostic discharge is the slab pass, apart from the step's own graph;
4. the valley limit (s forced to 1) is mode C with mode D's knobs, exactly;
5. the sheet limit (s forced to 0) is mode B with the filter, exactly at
   hc = 1, and its datum is the slab datum at hc = 1.5;
6. the driver wiring equals a harness built from the oracle;
8. the facade: defaults, explicit values, contradictions, the class stored.
"""
import warnings

import numpy as np
import pytest

from siim import constants as _constants
from siim._core import driver as drv
from siim._core.threads import threading_layer
from siim._core.regime import channel_class, swap, smooth_icy, check_regime_grid, _median_inplace
from siim.siim2d import siim as siim2d
from siim.siim1d import siim as siim1d
from siim.tests.test_sheet_knobs import _assert_same_dataset, _run, _spy_firewall, _spy_kernel, _PHYSICS

HC = float(_constants.HC_OVER_H)

# A small tent cut by four parabolic valleys along x, looped in y, fed so that
# channel and sheet cells coexist: measured about 300 channel cells of 1300
# icy per step at 41 x 41 (250 m), with blended s on every step.
_X, _Y = np.meshgrid(np.linspace(0.0, 1e4, 41), np.linspace(0.0, 1e4, 41))
_BED = (800.0 * (1.0 - np.abs(_X - 5e3) / 5e3)
        - 250.0 * np.clip(1.0 - (np.abs(_Y % 2500.0 - 1250.0) / 500.0) ** 2, 0.0, None))
_D = dict(_PHYSICS, T=1e4, nt=11, nt_out=11, Lx=1e4, Ly=1e4, nx=41, ny=41,
          boundary_status=['fixed_value', 'fixed_value', 'looped', 'looped'], mode='D',
          zELA=150.0, P=30.0, beta=3e-2, alpha_g=5.0, initial_topography=_BED, noise_amplitude=0.0)

# Mode C / mode B spelled out with mode D's knobs (parallel_erode explicit in
# every arm: a default-vs-explicit comparison proves parameter equivalence only).
_D_ROWS = ('glacial_spl__hc_cell', 'glacial_spl__channel')   # the rows only mode D stores
_D_KNOBS = dict(inner_damping=_constants.MODE_BCD_INNER_DAMPING, inner_passes=_constants.INNER_PASSES,
                flow_routing='mfd', width_hack_k=_constants.WIDTH_HACK_K, parallel_erode=False,
                trunk_surface=_constants.MODE_D_TRUNK_SURFACE)


# ---------------------------------------------------------------------------
# The oracle: the channel/sheet reading of constants.REGIME_* written plainly
# (per-cell loops, a breadth-first flood fill on the torus), independent of
# the kernel's arithmetic and of its scipy labelling.
# ---------------------------------------------------------------------------
_AXES = ((0, 1), (1, 0), (1, 1), (1, -1))


def _wrap(j, n, looped):
    """``j`` on an axis of ``n`` cells: wrapped if looped, None past an edge."""
    if looped:
        return j % n
    return j if 0 <= j < n else None


def _oracle_evidence(q, H, zb, dx, ag, kB, wy, wx):
    C = _constants
    ny, nx = H.shape
    icy = H > C.REGIME_ICY_H
    Hz = np.where(icy, H, 0.0)
    weak = np.zeros(H.shape, bool)
    strong = np.zeros(H.shape, bool)
    for iy, ix in zip(*np.nonzero(icy)):
        r0 = 0.5 * ag * H[iy, ix]
        n = int(r0 // dx) + 1
        disc = [(jy, jx) for a in range(-n, n + 1) for b in range(-n, n + 1)
                if (a * a + b * b) * dx * dx <= r0 * r0
                for jy, jx in [(_wrap(iy + a, ny, wy), _wrap(ix + b, nx, wx))]
                if jy is not None and jx is not None]
        qb = max(q[c] for c in disc)
        zmin = min(zb[c] for c in disc)
        lo = r0 + max(C.REGIME_RING_GAP_HW * r0, C.REGIME_RING_GAP_CELLS * dx)
        hi = lo + max(C.REGIME_RING_SPAN_HW * r0, C.REGIME_RING_SPAN_CELLS * dx)
        ring = []
        for ay, ax in _AXES:
            step = dx * np.hypot(ay, ax)
            radii = [r for r in range(1, int(hi / step) + 2) if lo <= r * step < hi]
            radii = radii or [next(r for r in range(1, 10 ** 6) if r * step >= lo)]
            for sgn in (1, -1):
                for r in radii:
                    jy, jx = _wrap(iy + sgn * r * ay, ny, wy), _wrap(ix + sgn * r * ax, nx, wx)
                    if jy is not None and jx is not None:
                        ring.append((jy, jx))
        qs = np.array([q[c] for c in ring])
        bar = True
        if kB > 0:
            mzb = np.median([zb[c] for c in ring]) if ring else zb[iy, ix]
            mH = np.median([Hz[c] for c in ring]) if ring else H[iy, ix]
            bar = 1.0 + (mzb - zmin) / max(mH, C.REGIME_BAR_MIN_H) > kB
        weak[iy, ix] = bar and (not ring or 2 * np.sum(qs >= qb / C.REGIME_WEAK) < len(ring))
        strong[iy, ix] = bar and (not ring or 2 * np.sum(qs >= q[iy, ix] / C.REGIME_STRONG) < len(ring))
    return weak, strong


def _oracle_class(q, H, zb, dx, ag, kB, wy, wx):
    weak, strong = _oracle_evidence(q, H, zb, dx, ag, kB, wy, wx)
    region = weak | strong
    ny, nx = H.shape
    chan = np.zeros(H.shape, bool)
    seen = np.zeros(H.shape, bool)
    for start in zip(*np.nonzero(region)):
        if seen[start]:
            continue
        seen[start] = True
        cells, queue = [], [start]
        while queue:
            cy, cx = queue.pop()
            cells.append((cy, cx))
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    jy, jx = _wrap(cy + a, ny, wy), _wrap(cx + b, nx, wx)
                    if jy is not None and jx is not None and region[jy, jx] and not seen[jy, jx]:
                        seen[jy, jx] = True
                        queue.append((jy, jx))
        core = [H[c] for c in cells if strong[c]]
        if (len(core) >= _constants.REGIME_CORE_CELLS
                and len(core) * dx * dx >= (ag * np.median(core)) ** 2):
            for c in cells:
                chan[c] = True
    return chan & (H > _constants.REGIME_ICY_H)


def _random_scene(rng, dx):
    """A wavy bed, patchy thickness (bare cells, cells exactly at the icy
    threshold) and a lognormal discharge with a few fast streaks."""
    ny, nx = rng.integers(24, 34, 2)
    zb = rng.normal(0, 40, (ny, nx)).cumsum(0) + rng.normal(0, 150, (ny, nx))
    H = np.where(rng.random((ny, nx)) < 0.8, rng.gamma(2, 60 * dx / 250, (ny, nx)), 0.0)
    H[rng.random((ny, nx)) < 0.05] = _constants.REGIME_ICY_H
    q = rng.lognormal(3, 1, (ny, nx))
    q[rng.integers(0, ny, 3)] *= 40.0
    q[:, rng.integers(0, nx, 2)] *= 40.0
    return np.where(H > _constants.REGIME_ICY_H, q, 0.0), H, zb


# ---------------------------------------------------------------------------
# 1-3. the reading, the swap, the dials
# ---------------------------------------------------------------------------
@pytest.mark.pin
@pytest.mark.filterwarnings("ignore:mode 'D'. the thickest ice:UserWarning")   # small looped scenes wrap
@pytest.mark.parametrize('wrap', [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize('alpha_g', [2.0, 5.0])
def test_class_matches_oracle_random(wrap, alpha_g):
    """The kernel and the labelling against the plain oracle, exact, with the
    bed bar on and off, at three resolutions."""
    rng = np.random.default_rng(int(alpha_g) + 10 * wrap[0] + 20 * wrap[1])
    seen = set()
    for dx in (250.0, 500.0, 1000.0):
        q, H, zb = _random_scene(rng, dx)
        for kB in (_constants.REGIME_BED_BAR, 0.0):
            got = channel_class(q, H, zb, dx, alpha_g, kB, *wrap)
            want = _oracle_class(q, H, zb, dx, alpha_g, kB, *wrap)
            np.testing.assert_array_equal(got, want)
            icy = H > _constants.REGIME_ICY_H
            seen.update({'channel'} if got.any() else set())
            seen.update({'sheet'} if (icy & ~got).any() else set())
    assert seen == {'channel', 'sheet'}, f"vacuous: only {seen}"


@pytest.mark.pin
def test_regions_join_across_looped_edges():
    """A strong core split by the looped y seam (3 + 3 cells, under the core
    floor on either side) reads channel as one region only when y is looped;
    the diagonal touch at the corner counts."""
    assert _constants.REGIME_CORE_CELLS == 5
    ny, nx, dx = 40, 40, 1000.0
    H = np.full((ny, nx), 50.0)
    q = np.full((ny, nx), 1.0)
    for y in (ny - 3, ny - 2, ny - 1, 0, 1, 2):
        q[y, 19] = 1e3
    zb = np.zeros((ny, nx))
    looped = channel_class(q, H, zb, dx, 5.0, 0.0, True, False)
    assert looped[[ny - 1, 0], 19].all() and looped.sum() >= 6
    assert not channel_class(q, H, zb, dx, 5.0, 0.0, False, False).any()
    corner = np.full((ny, nx), 1.0)                     # one diagonal step across both seams
    for y, x in ((ny - 2, nx - 2), (ny - 1, nx - 1), (0, 0), (1, 1), (2, 2)):
        corner[y, x] = 1e3
    assert channel_class(corner, H, zb, dx, 5.0, 0.0, True, True)[0, 0]
    assert not channel_class(corner, H, zb, dx, 5.0, 0.0, True, False).any()


@pytest.mark.pin
def test_thick_outlet_reads_channel():
    """The old failure: an 1800 m outlet in a trough (thicker than the rim
    window of the retired reading allowed, 1667 m at alpha_g 6) carrying 60x
    the discharge of the sheet around it reads channel, the sheet sheet."""
    ny, nx, dx = 80, 80, 1000.0
    zb = np.full((ny, nx), 1000.0)
    zb[36:45, 10:70] = -200.0                           # a 9 km trough, 60 km long
    H = 1600.0 - zb                                     # a level surface: 600 m sheet, 1800 m outlet
    q = np.where(zb < 0, 1800.0, 30.0)
    chan = channel_class(q, H, zb, dx, 5.0, _constants.REGIME_BED_BAR, False, False)
    assert chan[36:45, 15:65].all()
    assert not chan[:25].any() and not chan[56:].any()


@pytest.mark.pin
def test_bed_dial_is_monotone():
    """Four troughs 250 / 500 / 800 / 1200 m deep under a level 300 m sheet,
    each carrying fast ice: the bed bar kB admits the troughs deeper than
    (kB - 1) x 300 m (the ring's median thickness), so raising the dial drops
    them in order; 0 is no bar."""
    ny, nx, dx = 90, 80, 1000.0
    zb = np.full((ny, nx), 1000.0)
    rows = {250.0: 10, 500.0: 30, 800.0: 50, 1200.0: 70}
    for depth, y in rows.items():
        zb[y - 2:y + 3, 10:70] = 1000.0 - depth
    H = 1300.0 - zb
    q = np.where(zb < 1000.0, 3000.0, 20.0)
    got = {}
    for kB in (0.0, 1.5, 2.0, 3.0, 4.5, 5.5):
        chan = channel_class(q, H, zb, dx, 2.0, kB, False, False)
        got[kB] = {d for d, y in rows.items() if chan[y, 40]}
    assert got == {0.0: set(rows), 1.5: set(rows), 2.0: {500.0, 800.0, 1200.0},
                   3.0: {800.0, 1200.0}, 4.5: {1200.0}, 5.5: set()}, got


def test_ring_median_is_numpys():
    """The kernel's in-place select returns ``np.median`` exactly (the mean of
    the two middle values for an even count) on distinct values, ties, sorted
    input and the all-equal ring of bare ground; any NaN gives NaN, wherever
    it sits."""
    rng = np.random.default_rng(5)
    for trial in range(400):
        n = int(rng.integers(1, 140))
        a = (rng.random(n), np.round(3 * rng.random(n)), np.zeros(n), np.sort(rng.random(n)),
             np.where(rng.random(n) < 0.6, 0.0, 1e3 * rng.random(n)))[trial % 5]
        assert _median_inplace(a.copy(), n) == np.median(a), (trial, n)
    assert np.isnan(_median_inplace(np.full(7, np.nan), 7))
    for n in (2, 7, 30):                      # one NaN anywhere, as np.median
        for pos in range(n):
            a = rng.random(n)
            a[pos] = np.nan
            assert np.isnan(_median_inplace(a, n)), (n, pos)


@pytest.mark.pin
def test_swap():
    """The devices' s: the class c wherever the cell was not icy on the last
    reading (so on the first, from the driver's s = 1 and no ice);
    1/REGIME_SWAP_STEPS of the way to c otherwise; 1 off the ice."""
    assert _constants.REGIME_SWAP_STEPS == 5
    icy = np.array([True, True, True, False, True])
    chan = np.array([True, False, False, False, False])
    np.testing.assert_array_equal(swap(np.ones(5), np.zeros(5, bool), chan, icy), [1.0, 0.0, 0.0, 1.0, 0.0])
    s_prev = np.array([0.0, 1.0, 0.5, 0.3, 1.0])
    icy_prev = np.array([True, True, True, True, False])
    np.testing.assert_array_equal(swap(s_prev, icy_prev, chan, icy), [0.2, 0.8, 0.4, 1.0, 0.0])
    s = np.ones(1)
    for _ in range(5):
        s = swap(s, np.ones(1, bool), np.zeros(1, bool), np.ones(1, bool))
    np.testing.assert_allclose(s, 0.8 ** 5)


@pytest.mark.pin
def test_datum_smoothing():
    """The datum's s: a Gaussian over the icy cells, normalised by the icy
    weight (a uniform s comes back exactly; bare cells neither weigh nor
    change), wrapped on looped axes."""
    icy = np.ones((9, 9), bool)
    icy[:, :2] = False
    s = np.where(icy, 0.25, 1.0)
    np.testing.assert_allclose(smooth_icy(s, icy, False, False), s, rtol=0, atol=1e-15)
    step = np.where(np.arange(9) < 5, 1.0, 0.0)[None, :] * np.ones((9, 1))
    out = smooth_icy(step, np.ones((9, 9), bool), False, True)
    assert 0.0 < out[4, 0] < 1.0 and 0.0 < out[4, 8] < 1.0      # the wrapped seam blends
    assert smooth_icy(step, np.ones((9, 9), bool), False, False)[4, 0] == 1.0


# ---------------------------------------------------------------------------
# 3. the diagnostic discharge
# ---------------------------------------------------------------------------
@pytest.mark.pin
def test_diagnostic_discharge_is_the_slab_pass():
    """q is read on the plain slab zb + H (the filter on the full thickness,
    the footprint the cell), whatever the run's datum, and the pass leaves the
    step's own routing arrays alone (FIREWALL clause 3)."""
    from siim._core.router import InhouseRouter
    from siim._core.step import filter_routing_surface, accumulate_glacial_flow
    rng = np.random.default_rng(3)
    m = siim2d(dict(_D, hc_over_H=1.2))
    cfg = m._build_driver_config()
    shape = (cfg.ny, cfg.nx)
    route = InhouseRouter(shape, m.boundary_status, 'mfd', cfg.dx, cfg.dy, m.mfd_exponent)
    cfg.route = route
    zb = rng.normal(0, 30, shape).cumsum(1) + 400.0
    H = np.where(rng.random(shape) < 0.7, rng.gamma(2, 80, shape), 0.0)
    own = route(zb + 1.5 * H)
    kept = own.receivers.copy(), own.weights.copy(), own.stack.copy()
    uplift = np.zeros(shape)
    q = drv._diagnostic_discharge(cfg, zb, H, uplift, cfg.zELA, cfg.P, shape)
    for a, b in zip(kept, (own.receivers, own.weights, own.stack)):
        np.testing.assert_array_equal(a, b)
    zs = zb + H
    assert cfg.routing_filter_factor > 0, "vacuous: no filter"
    rt = route(filter_routing_surface(zs, H, cfg.dx, cfg.routing_filter_factor, cfg.wrap_y, cfg.wrap_x))
    Q = accumulate_glacial_flow(zs, uplift, cfg.zELA, cfg.beta, cfg.P, cfg.cell_area, cfg.width_hack_k,
                                cfg.width_hack_p, shape, rt.stack, rt.receivers, rt.nb_receivers, rt.weights,
                                rt.lengths, None, s_conf=np.zeros(shape))[0]
    icy = H > _constants.REGIME_ICY_H
    want = np.where(icy & (Q > 0), Q / (cfg.alpha_g * np.where(icy, H, 1.0)), 0.0)
    np.testing.assert_array_equal(q, want)
    assert (q > 0).sum() > 100, "vacuous: no discharge"
    cfg.hc_over_H = 1.5                                  # the datum does not enter
    np.testing.assert_array_equal(drv._diagnostic_discharge(cfg, zb, H, uplift, cfg.zELA, cfg.P, shape), q)


# ---------------------------------------------------------------------------
# 4-5. the two limits
# ---------------------------------------------------------------------------
def _force_s(monkeypatch, value):
    """Make the devices' s constant; return the ``(class, icy)`` pairs the
    real reading produced (the vacuity check)."""
    real = []

    def forced(s_prev, icy_prev, chan, icy):
        real.append((chan.copy(), icy.copy()))
        return np.full(np.shape(icy), float(value))
    monkeypatch.setattr(drv, 'swap', forced)
    return real


@pytest.mark.pin
@pytest.mark.parametrize('trunk', [_constants.MODE_D_TRUNK_SURFACE, True], ids=['default', 'trunk-on'])
def test_valley_limit_is_mode_c(monkeypatch, trunk):
    """s forced to 1 everywhere: mode D is mode C with D's knobs spelled out,
    bit-for-bit, at mode D's trunk default and with the V on. At s = 1 the
    filter's thickness H*(1 - s) is zero, so the mode-C arm runs unfiltered,
    and the V is blended per source by 1.0."""
    knobs = dict(_D_KNOBS, trunk_surface=trunk)
    real = _force_s(monkeypatch, 1.0)
    on = _run(_D, trunk_surface=trunk)
    # gate: the vacuity check reads the 2026-11-13 class (the dead-band sheet mask retired with the rim metric)
    assert any((icy & ~chan).any() for chan, icy in real), "vacuous: the reading read no sheet"
    assert (on.hc_cell_out == HC).all()
    off = _run(_D, mode='C', routing_filter_factor=0.0, **knobs)
    _assert_same_dataset(on, off, extra=_D_ROWS)
    if trunk:
        no_v = _run(_D, mode='C', routing_filter_factor=0.0, **dict(knobs, trunk_surface=False))
        assert not np.array_equal(on.H_out, no_v.H_out), "the V was inert: vacuous"


@pytest.mark.pin
def test_sheet_limit_exact_arm(monkeypatch):
    """s forced to 0: hc = 1 and the Hack width 0 in both arms (so the first
    step, which runs the valley devices from s = 1 on H = 0, is the same), mode
    D equals mode B with the filter and no V or carve, bit-for-bit: the carve,
    the V and the footprint are inert at s = 0 and the filter sees all of H."""
    monkeypatch.setattr(_constants, 'HC_OVER_H', 1.0)
    real = _force_s(monkeypatch, 0.0)
    knobs = dict(_D_KNOBS, width_hack_k=0.0)
    on = _run(_D, **knobs)
    # gate: the vacuity check reads the 2026-11-13 class (the dead-band sheet mask retired with the rim metric)
    assert any(chan.any() for chan, _ in real), "vacuous: all sheet anyway"
    off = _run(_D, mode='B', routing_filter_factor=2.0, **knobs)   # knobs carry trunk_surface=False
    _assert_same_dataset(on, off, extra=_D_ROWS)
    assert on.H_out[-1].max() > 100.0


def test_sheet_limit_datum_is_the_slab(monkeypatch):
    """At the shipped hc = 1.5, s forced to 0 builds every column after the
    first step on the slab datum, exactly; the first step's s = 1 is the
    channel datum."""
    _force_s(monkeypatch, 0.0)
    on = _run(_D)
    assert (on.hc_cell_out[0] == HC).all()
    assert (on.hc_cell_out[1:] == 1.0).all()


# ---------------------------------------------------------------------------
# 6. the driver wiring
# ---------------------------------------------------------------------------
@pytest.mark.pin
def test_wiring_equals_the_harness(monkeypatch):
    """Mode D equals a harness built from the oracle: the class read from the
    COMMITTED H and the bed the kernel just eroded (its live buffer, whatever
    bed the driver hands the reading), s relaxed toward it by the swap's own
    formula with the harness's own memory. Every output row equal."""
    d = _run(_D)
    zb_live, mem, mixed = {}, {}, []
    read, closure = [], []
    def kernel(dt, zb0, zb1, H0, H1):
        zb_live['zb'] = zb1
        closure.append(H1.copy())
    _spy_kernel(monkeypatch, kernel)

    def harness_class(q, H, zb, dx, alpha_g, kB, wy, wx):
        read.append((np.array(H), kB))
        return _oracle_class(q, H, zb_live['zb'].reshape(H.shape), dx, alpha_g, kB, wy, wx)

    def harness_swap(s_prev, icy_prev, chan, icy):
        c = np.where(icy & ~chan, 0.0, 1.0)
        if 's' in mem:
            c = np.where(mem['icy'] & icy, mem['s'] + 0.2 * (c - mem['s']), c)
        mem.update(s=c, icy=icy)
        mixed.append(((c > 0) & (c < 1)).any())
        return c
    monkeypatch.setattr(drv, 'channel_class', harness_class)
    monkeypatch.setattr(drv, 'swap', harness_swap)
    harness = _run(_D)
    _assert_same_dataset(d, harness)
    # the reading sees the COMMITTED H (the staged frame), not the kernel's own F,
    # and the run's dial
    assert len(read) == len(closure) == d.H_out.shape[0] - 1
    for k, (H, kB) in enumerate(read):
        np.testing.assert_array_equal(H, d.H_out[k])
        assert kB == d.channel_bed_bar
    assert any((F.reshape(H.shape) != H).any() for (H, _), F in zip(read, closure)), \
        "vacuous: the damping never moved the committed H off F"
    # gate: s is relaxed and the datum smoothed since 2026-11-13, so the vacuity checks ask for
    # both classes and a blended s instead of hc taking both binary values and the dead band holding
    icy = d.H_out > _constants.REGIME_ICY_H
    assert d.channel_out[1:].any() and (icy & ~d.channel_out).any(), "vacuous: one class"
    assert any(mixed), "vacuous: the swap never blended"


@pytest.mark.pin
def test_devices_take_the_relaxed_s(monkeypatch):
    """Recipe step 8: the carve and the flexure load take this step's s (after
    the swap and the border rule), the footprint and the trunk V the previous
    step's, through the FIREWALL spies (the reading's own pass is not a
    device); the smoothed s goes to the datum alone (pinned by
    test_valley_sheet_devices.py::test_datum_reaches_the_kernel_and_the_router)."""
    rec = _spy_firewall(monkeypatch)
    s, smooth = [], []
    real_bi, real_sm = drv.border_inherits, drv.smooth_icy

    def bi(x, *a):                     # the devices' s; the datum's call (on the smoothed s) is not one
        out = real_bi(x, *a)
        if not (smooth and x is smooth[-1]):
            s.append(out)
        return out
    monkeypatch.setattr(drv, 'border_inherits', bi)
    monkeypatch.setattr(drv, 'smooth_icy', lambda *a: smooth.append(real_sm(*a)) or smooth[-1])
    d = _run(_D, trunk_surface=True, flexure=True)
    steps = d.H_out.shape[0] - 1
    assert len(s) == len(rec['carve_s']) == len(rec['flexure_s']) == len(rec['fab_s']) == steps
    for k in range(steps):
        lagged = np.ones_like(s[k]) if k == 0 else s[k - 1]
        np.testing.assert_array_equal(rec['carve_s'][k], s[k])
        np.testing.assert_array_equal(rec['flexure_s'][k], s[k])
        np.testing.assert_array_equal(rec['acc_args'][k][1]['s_conf'], lagged)
        np.testing.assert_array_equal(rec['fab_s'][k], lagged)
    assert any((a != b).any() for a, b in zip(s, smooth)), "vacuous: the smoothing changed nothing"
    assert any(((x > 0) & (x < 1)).any() for x in s), "vacuous: s never blended"


@pytest.mark.pin
def test_border_datum_is_its_donors(monkeypatch):
    """A base-level border cell's DATUM is its dominant donor's, as its s and its
    zero-gradient H are (conventions, 2026-10-17): the border rule runs again on
    the smoothed s, so the datum each step hands the next step's hc_cell_field
    is a fixed point of border_inherits on the routing it was read on. Smoothed
    after the rule alone, a border's datum drifted off its donor's."""
    real_bi, real_sm, real_hc = drv.border_inherits, drv.smooth_icy, drv.hc_cell_field
    routing, smoothed, datums = [None], [], []

    def bi(x, *a):                                  # the routing of the devices' call, not the datum's
        if not (smoothed and x is smoothed[-1][0]):
            routing[0] = tuple(None if v is None else np.array(v) for v in a)
        return real_bi(x, *a)

    def sm(*a):
        smoothed.append((real_sm(*a), routing[0]))
        return smoothed[-1][0]

    def hc(x, *a):
        if routing[0] is not None:                  # step 1's datum is the initial s
            datums.append((np.array(x), routing[0]))
        return real_hc(x, *a)
    monkeypatch.setattr(drv, 'border_inherits', bi)
    monkeypatch.setattr(drv, 'smooth_icy', sm)
    monkeypatch.setattr(drv, 'hc_cell_field', hc)
    _run(_D)
    # gate: new test (2026-11-14); a literal count, since _D's values type as a union
    assert len(datums) == 9                         # nt = 11: ten steps, the first on s = 1
    for x, a in datums:
        np.testing.assert_array_equal(real_bi(x, *a), x)
    assert any((real_bi(x, *a) != x).any() for x, a in smoothed), \
        "vacuous: the smoothing never moved a border off its donor"


@pytest.mark.pin
def test_relax_snaps_abandoned_ice_to_bare():
    """A cell the kernel leaves bare snaps to 0 below INNER_RELAX_BARE_H."""
    nt = 21
    zELA = np.where(np.arange(nt) < 5, _D['zELA'], 1e4)      # glaciate, then melt out
    d = _run(_D, T=2e4, nt=nt, nt_out=nt, zELA=zELA, save_closure_pair=True)
    assert d.closure_H_out is not None    # gate: new test; guards the film check against a missing row
    film = ((d.closure_H_out == 0.0) & (d.H_out > 0.0)
            & (d.H_out < _constants.INNER_RELAX_BARE_H))
    assert not film.any()
    snapped = (d.H_out[1:] == 0.0) & (d.H_out[:-1] > 0.0)
    assert snapped.any(), "vacuous: no abandoned cell reached the floor"


@pytest.mark.pin
def test_sparse_frames_are_the_dense_frames():
    """The routing-topology rows (the basin labels, a whole D8 pass under mode
    D's MFD router; the primary receivers; the stack) are built only on the
    steps a frame saves: a run saving 3 frames equals the same frames of a run
    saving all 11, the last one (staged after the loop) included."""
    # mode D runs one pass a step; mode C's two also gate the rows on the last
    for knobs in ({}, dict(_D_KNOBS, mode='C', inner_passes=2)):
        dense, sparse = _run(_D, **knobs), _run(_D, nt_out=3, **knobs)   # frames 0..10 / 0, 5, 10
        basins = dense.ds_out['glacial_flow__basin_ids'].values
        assert (basins[0] != basins[5]).any(), "vacuous: the network never changes"
        for k in dense.ds_out.data_vars:
            np.testing.assert_array_equal(sparse.ds_out[k].values,
                                          dense.ds_out[k].values[[0, 5, 10]], err_msg=k)


# ---------------------------------------------------------------------------
# 8. the facade
# ---------------------------------------------------------------------------
def test_mode_d_defaults_resolve():
    m = siim2d(_D)
    assert (m.mode, m.carve_width, m.regime_switch) == ('B', True, True)   # gate: fullness_switch retired 2026-11-03; mode D is the only s source
    C = _constants
    # gate: routing_relax (MODE_D_ROUTING_RELAX 0) was removed 2026-11-04; the
    # damping is the adaptive preset since (Eric, 2026-11-05; adaptive_capped from 2026-11-04)
    assert m.inner_damping == C.MODE_BCD_INNER_DAMPING
    # gate: mode D's Hack k follows WIDTH_HACK_K since 2026-11-19 (one coefficient for every mode)
    assert (m.inner_relax, m.width_hack_k, m.routing_filter_factor, m.flow_routing,
            m.trunk_surface, m.parallel_erode) == (
        C.INNER_DAMPING_FLOORS[C.MODE_BCD_INNER_DAMPING], C.WIDTH_HACK_K, C.MODE_D_ROUTING_FILTER_FACTOR,
        C.MODE_D_FLOW_ROUTING, C.MODE_D_TRUNK_SURFACE,   # gate: D's own trunk default again (Eric, 2026-11-03: V off)
        # gate: under MFD, D's routing, the parallel_erode default follows the threading layer (ED-168; pinned in test_threading.py)
        threading_layer() in C.MFD_PARALLEL_ERODE_LAYERS)
    assert siim2d(dict(_D, mode='d')).regime_switch
    assert not siim2d(dict(_D, mode='C')).regime_switch


def test_mode_d_explicit_values_win():
    kw = dict(inner_relax=1.0, width_hack_k=0.5, trunk_surface=True,
              routing_filter_factor=0.0, parallel_erode=True, channel_bed_bar=2.0)
    assert kw['channel_bed_bar'] != _constants.REGIME_BED_BAR      # an explicit value, not the default
    m = siim2d(dict(_D, **kw))
    assert {k: getattr(m, k) for k in kw} == kw
    assert m.regime_switch      # gate: the V is now an explicit value (default off since 2026-11-03), checked in kw
    # gate: flow_routing left kw 2026-11-13: mode D accepts MFD alone (test_mode_d_contradictions_raise)
    assert siim2d(_D).channel_bed_bar == _constants.REGIME_BED_BAR
    assert siim2d(dict(_D, mode='C')).channel_bed_bar is None


@pytest.mark.parametrize('kw,match', [
    (dict(carve_width=False), 'carve_width=False'),
    (dict(fullness_switch=True), 'fullness_switch'),
    (dict(fullness_switch=False), 'fullness_switch'),
    (dict(Ly=2e4), 'square cells'),
    (dict(widening_rate=None), 'finite widening_rate'),
    (dict(Ly=3.5e3, ny=15, initial_topography=None), 'looped y axis'),
    (dict(Lx=3.5e3, nx=15, initial_topography=None,
          boundary_status=['looped', 'looped', 'fixed_value', 'fixed_value']), 'looped x axis'),
    (dict(flow_routing='single'), "flow_routing='mfd'"),
    (dict(flow_routing='dinf'), "flow_routing='mfd'"),
    (dict(flow_routing='single', mfd_exponent=60.0), "flow_routing='mfd'"),   # the routing error first
    (dict(mfd_exponent=np.inf), 'finite mfd_exponent'),
    (dict(mfd_exponent=60.0), 'finite mfd_exponent'),
    (dict(mfd_exponent='inf'), 'finite mfd_exponent'),
    (dict(channel_bed_bar=10 ** 400), 'channel_bed_bar must be'),
    (dict(channel_bed_bar='2'), 'channel_bed_bar must be'),
    (dict(channel_bed_bar=[2.0]), 'channel_bed_bar must be'),
    (dict(channel_bed_bar=True), 'channel_bed_bar must be'),
    (dict(channel_bed_bar=-1.0), 'channel_bed_bar must be'),
    (dict(channel_bed_bar=np.inf), 'channel_bed_bar must be'),
    (dict(mode='C', channel_bed_bar=2.0), "requires mode='D'"),
], ids=['carve-off', 'switch-on', 'switch-off', 'rectangular', 'instant-widening',
        'short-looped-y', 'short-looped-x', 'single-flow', 'dinf', 'single-flow-high-exponent', 'infinite-exponent', 'exponent-above-max', 'string-exponent', 'huge-int-bar', 'string-bar',
        'list-bar', 'bool-bar', 'negative-bar', 'infinite-bar', 'bar-outside-d'])
@pytest.mark.filterwarnings('error')          # a refusal comes alone, with no warning before it
def test_mode_d_contradictions_raise(kw, match):
    with pytest.raises(ValueError, match=match):
        siim2d(dict(_D, **kw))


def test_looped_axis_guard_bound():
    """A looped axis must be longer than twice the ring's reach in cells (its
    gap and span floors): at exactly twice, the two ends of an axis pair
    sample the same cell. The guard counts cells."""
    reach = 2 * (_constants.REGIME_RING_GAP_CELLS + _constants.REGIME_RING_SPAN_CELLS)
    # gate: the guard counts cells since 2026-11-13 (the 20 km static reach retired with the rim metric)
    assert reach == 16
    check_regime_grid(17, 10, 500.0, wrap_y=True, wrap_x=False)
    with pytest.raises(ValueError, match='looped y axis'):
        check_regime_grid(16, 10, 500.0, wrap_y=True, wrap_x=False)
    check_regime_grid(10, 10, 500.0, wrap_y=False, wrap_x=False)      # nothing looped
    siim2d(dict(_D, boundary_status=['fixed_value', 'fixed_value', 'looped', 'looped']))   # 41 cells


def test_mode_d_warns_on_coarse_grids():
    with pytest.warns(UserWarning, match='sub-grid'):
        siim2d(dict(_D, Lx=1.2e5, Ly=1.2e5))                           # 3 km cells
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        siim2d(dict(_D, Lx=1e5, Ly=1e5))                               # 2.5 km: silent
    # gate: the warning's text changed with the 2026-11-13 reading (no km-scale bands any more)
    assert not [w for w in caught if 'sub-grid' in str(w.message)]


def test_the_class_is_stored_and_saved(tmp_path):
    """Mode D stores its class per frame (``channel_out``, read on the frame's
    committed H, channel only on icy cells); a save carries it, a run saved
    before the row existed loads with ``channel_out`` None; no other mode
    stores it."""
    d = _run(_D, T=3e3, nt=4, nt_out=4)
    assert d.channel_out.dtype == bool and d.channel_out.shape == d.H_out.shape
    assert not (d.channel_out & (d.H_out <= _constants.REGIME_ICY_H)).any()
    path = d.save(tmp_path / 'd.pkl')
    np.testing.assert_array_equal(siim2d.load(path).channel_out, d.channel_out)
    d.ds_out = d.ds_out.drop_vars('glacial_spl__channel')
    old = siim2d.load(d.save(tmp_path / 'old.pkl'))
    assert old.channel_out is None and old.hc_cell_out is not None
    assert _run(_D, mode='C', T=1e3, nt=2, nt_out=2).channel_out is None


def test_siim1d_refuses_mode_d():
    with pytest.raises(ValueError, match='2D-only'):
        siim1d({'mode': 'D', 'progress_bar': False})


@pytest.mark.adapter
def test_adapter_refuses_mode_d():
    pytest.importorskip('xsimlab')
    pytest.importorskip('fastscape')
    from siim.fastscape import glacial_processes
    with pytest.raises(ValueError, match="mode 'D'.*STANDALONE-only"):
        glacial_processes(mode='D')
    m = siim2d(dict(_D, T=1e3, nt=2, nt_out=2))      # the configured-model entry point
    with pytest.raises(ValueError, match="mode 'D'.*STANDALONE-only"):
        m.run(driver='xsimlab')


@pytest.mark.pin
def test_sheet_ice_loads_the_plate_as_a_slab():
    """Mode D's s blends the flexure ice column from the channel footprint
    column (s = 1, or no s) to a slab of thickness H (s = 0); audit #28, ED-184."""
    from siim._core.step import glacial_flexure_step
    shape = (5, 5)
    seen = {}

    def capture(post, eq, *_, **__):
        seen['load'] = post - eq          # the rock-equivalent load handed to the plate

    rock = _constants.RHO_ICE / 2800.0
    # gate: the s = 1 column is the footprint load (ED-184): a uniform 1 km sheet on
    # 1 km cells is 1.5*H*(1 - L^2/(12 R^2)) = 1480 m a cell, not the overlapping
    # per-cell section's 5 km
    fp = 1500.0 * (1.0 - 1e6 / (12.0 * 2500.0 ** 2))
    for s, ice_m in ((None, fp), (1.0, fp), (0.0, 1000.0), (0.5, 0.5 * (fp + 1000.0))):
        glacial_flexure_step(np.zeros(shape), np.zeros(shape), np.zeros(shape),
                             np.full(shape, 1000.0), 5.0, 1e6, 2800.0, 3200.0, 2e4, 0,
                             shape, (4e3, 4e3), np.zeros(shape), True, capture,
                             s_conf=None if s is None else np.full(shape, s))
        np.testing.assert_allclose(seen['load'], rock * ice_m)


def test_only_mode_d_hands_its_s_to_the_flexure_load(monkeypatch):
    """The driver passes mode D's s to the flexure step and None in every
    other mode, which keeps them bit-for-bit (audit #28)."""
    seen = []
    real = drv.glacial_flexure_step

    def spy(*a, s_conf=None, **kw):
        seen.append(s_conf is not None)
        return real(*a, s_conf=s_conf, **kw)
    monkeypatch.setattr(drv, 'glacial_flexure_step', spy)
    for mode, want in (('D', True), ('C', False)):
        seen.clear()
        siim2d(dict(_D, mode=mode, flexure=True, T=2e3, nt=3, nt_out=2)).run()
        assert seen and all(v == want for v in seen), mode
