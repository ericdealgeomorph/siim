"""Sub-grid glacier-width carving: power-DT kernel + carve semantics.

The footprint is the union of discs of radius alpha_g*H/2 around glaciated
cells (MAT inversion); membership and attribution come from the minimum
power distance d^2 - R^2 via the Felzenszwalb-Huttenlocher generalized
distance transform. Footprint cells — bare AND icy, including terrain
above the ice (no surface gate) — descend toward the parabola hung from
the source's ice surface (rim at zs = zb + HC_OVER_H*H, floor at the
source bed), at a rate-capped, never-additive pace measured from the
pre-step bed. Expectations are parameterized by HC_OVER_H so the test
pins the carve geometry under whichever convention ships.
"""
import numpy as np
import pytest

from siim.constants import HC_OVER_H
from siim._core.carve import (
    PDT_NO_SOURCE, _power_dt_2d, _power_dt_2d_periodic, _carve_offsets,
    _carve_subgrid_width)

HC = float(HC_OVER_H)


def _brute_power(offsets, dy, dx):
    ny, nx = offsets.shape
    ys, xs = np.nonzero(offsets < PDT_NO_SOURCE)
    D = np.full((ny, nx), PDT_NO_SOURCE)
    SRC = np.full((ny, nx), -1, dtype=np.int64)
    for y in range(ny):
        for x in range(nx):
            d2 = ((y - ys) * dy) ** 2 + ((x - xs) * dx) ** 2 + offsets[ys, xs]
            k = np.argmin(d2)
            D[y, x] = d2[k]
            SRC[y, x] = ys[k] * nx + xs[k]
    return D, SRC


def test_power_dt_matches_brute_force():
    """Exact power distances and argmin attribution, anisotropic spacing.

    atol guards the comparison where true power distances cross zero (the
    footprint boundary): FH and the brute force sum the same terms in a
    different order, so ~1-ulp-of-1e7 absolute noise is legitimate there.
    """
    rng = np.random.default_rng(3)
    ny, nx, dy, dx = 23, 31, 130.0, 100.0
    offsets = np.full((ny, nx), PDT_NO_SOURCE)
    src = rng.random((ny, nx)) < 0.06
    offsets[src] = -(rng.uniform(50.0, 800.0, src.sum())) ** 2
    D = np.empty((ny, nx)); SRC = np.empty((ny, nx), np.int64)
    _power_dt_2d(offsets, dy, dx, D, SRC)
    Db, SRCb = _brute_power(offsets, dy, dx)
    np.testing.assert_allclose(D, Db, rtol=1e-12, atol=1e-6)
    # argmin may tie; check the achieved power values, not the indices
    ys, xs = np.divmod(SRC.ravel(), nx)
    d_check = (((np.arange(ny)[:, None] - ys.reshape(ny, nx)) * dy) ** 2
               + ((np.arange(nx)[None, :] - xs.reshape(ny, nx)) * dx) ** 2
               + offsets[ys.reshape(ny, nx), xs.reshape(ny, nx)])
    np.testing.assert_allclose(d_check, Db, rtol=1e-12, atol=1e-6)


def _brute_power_periodic(offsets, dy, dx, wrap_y, wrap_x):
    """Brute-force periodic power distances: each seed contributes through
    its nearest image along the wrapped axes."""
    ny, nx = offsets.shape
    ys, xs = np.nonzero(offsets < PDT_NO_SOURCE)
    D = np.full((ny, nx), PDT_NO_SOURCE)
    SRC = np.full((ny, nx), -1, dtype=np.int64)
    for y in range(ny):
        for x in range(nx):
            ddy = np.abs(y - ys)
            if wrap_y:
                ddy = np.minimum(ddy, ny - ddy)
            ddx = np.abs(x - xs)
            if wrap_x:
                ddx = np.minimum(ddx, nx - ddx)
            d2 = (ddy * dy) ** 2 + (ddx * dx) ** 2 + offsets[ys, xs]
            k = np.argmin(d2)
            D[y, x] = d2[k]
            SRC[y, x] = ys[k] * nx + xs[k]
    return D, SRC


@pytest.mark.parametrize("wrap_y,wrap_x", [(False, True), (True, False),
                                           (True, True)])
def test_power_dt_periodic_matches_brute_force(wrap_y, wrap_x):
    """The wrap-padded transform is exact wherever it matters: footprint
    membership (D < 0 vs >= 0) agrees with the brute-force periodic
    diagram everywhere, and power values + achieved attribution agree on
    the footprint. (Outside every footprint the padded transform may
    lawfully report a larger D — those cells are skipped by the carve.)
    Mixed disc sizes exercise both padding regimes: small discs (pad set
    by R_max) and a disc wider than the half-domain (pad capped at the
    half circumference, where the result is exact everywhere)."""
    rng = np.random.default_rng(5)
    ny, nx, dy, dx = 17, 23, 130.0, 100.0
    offsets = np.full((ny, nx), PDT_NO_SOURCE)
    src = rng.random((ny, nx)) < 0.08
    offsets[src] = -(rng.uniform(80.0, 500.0, src.sum())) ** 2
    # one giant disc to bind the half-circumference cap
    offsets[3, 7] = -(1.4e3) ** 2
    D = np.empty((ny, nx)); SRC = np.empty((ny, nx), np.int64)
    _power_dt_2d_periodic(offsets, dy, dx, D, SRC, wrap_y, wrap_x)
    Db, SRCb = _brute_power_periodic(offsets, dy, dx, wrap_y, wrap_x)
    inside = Db < 0.0
    np.testing.assert_array_equal(D < 0.0, inside)
    np.testing.assert_allclose(D[inside], Db[inside], rtol=1e-12, atol=1e-6)
    # attribution: compare achieved power (argmin may tie), via each
    # cell's wrapped distance to its claimed source
    yy, xx = np.divmod(SRC[inside], nx)
    cy, cx = np.nonzero(inside)
    ddy = np.abs(cy - yy)
    if wrap_y:
        ddy = np.minimum(ddy, ny - ddy)
    ddx = np.abs(cx - xx)
    if wrap_x:
        ddx = np.minimum(ddx, nx - ddx)
    d_check = (ddy * dy) ** 2 + (ddx * dx) ** 2 + offsets[yy, xx]
    np.testing.assert_allclose(d_check, Db[inside], rtol=1e-12, atol=1e-6)


def test_carve_seam_invariance_looped_x():
    """x-looped: the whole carve pipeline commutes with a circular shift
    in x — the seam is invisible. A channel ring spanning the loop, with
    random per-column thickness/bed/walls, is carved twice: once as-is
    and once circularly shifted; the shifted output must equal the
    shifted-input output exactly."""
    rng = np.random.default_rng(11)
    ny, nx, dx = 11, 30, 100.0
    jc = 5                                       # channel row
    Hvals = rng.uniform(150.0, 250.0, nx)        # per-column channel H
    bedvals = rng.uniform(0.0, 10.0, nx)         # per-column channel bed
    wallvals = rng.uniform(100.0, 150.0, (ny, nx))

    # receivers: top/bottom rows self-receive (fixed_value); the channel
    # ring flows +x circularly; walls step toward the channel. The rec
    # FIELD commutes with np.roll along x by construction.
    rec = np.empty((ny, nx), dtype=np.int64)
    for j in range(ny):
        for i in range(nx):
            if j in (0, ny - 1):
                rec[j, i] = j * nx + i
            elif j == jc:
                rec[j, i] = jc * nx + (i + 1) % nx
            else:
                rec[j, i] = (j + (1 if j < jc else -1)) * nx + i
    rec = rec.ravel()

    def carve(shift):
        H = np.zeros((ny, nx))
        H[jc, :] = np.roll(Hvals, shift)
        zb = np.roll(wallvals, shift, axis=1).copy()
        zb[jc, :] = np.roll(bedvals, shift)
        zb_flat = zb.ravel()
        zb_pre = zb_flat.copy()
        zb_pre[jc * nx:(jc + 1) * nx] += 0.5     # channel eroded 0.5 m
        surf = zb_flat + HC * H.ravel()
        offsets = np.empty((ny, nx))
        n_seed = _carve_offsets(H.ravel(), rec, 4.0, offsets.ravel(),
                                np.ones(H.size, np.int8))
        assert n_seed == nx
        D = np.empty((ny, nx)); SRC = np.empty((ny, nx), np.int64)
        _power_dt_2d_periodic(offsets, dx, dx, D, SRC, False, True)
        _carve_subgrid_width(zb_flat, zb_flat.copy(), zb_pre, H.ravel(),
                             surf, rec, D, SRC, offsets, np.inf, HC)
        return zb_flat.reshape(ny, nx), surf.reshape(ny, nx)

    zb0, surf0 = carve(0)
    k = 13
    zbk, surfk = carve(k)
    np.testing.assert_allclose(np.roll(zb0, k, axis=1), zbk,
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.roll(surf0, k, axis=1), surfk,
                               rtol=0, atol=1e-9)
    # sanity: the carve actually fired (walls pulled down toward parabola)
    assert (zb0 < wallvals - 1.0).any()


def test_footprint_ice_surface_display():
    """The footprint ice display (rendering dual of the carve): footprint
    cells take their source's ice surface — carved troughs fill to the
    trimline, terrain above the ice surface stays rock (nunatak), the
    spine's column is exactly HC*H, and cells beyond R get no ice."""
    from siim.plotting._render import _footprint_ice_surface
    ny, nx, dx = 9, 12, 100.0
    jc = 4
    H = np.zeros((ny, nx)); H[jc, :] = 200.0      # R = 400 m = 4 cells
    zb = np.full((ny, nx), 50.0)                   # carved flank beds
    zb[jc, :] = 0.0
    zb[2, 6] = 1000.0                              # nunatak inside footprint
    z = zb.copy()
    z[jc, :] = zb[jc, :] + HC * 200.0              # spine presents zs = 300
    rec = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)
    rec[:, 1:] = rec[:, 1:] - 1                    # interior flows -x; col 0 border
    z_fill, ice, depth = _footprint_ice_surface(
        H, zb, rec.ravel(), alpha_g=4.0, dy=dx, dx=dx,
        wrap_y=False, wrap_x=False)
    zs = HC * 200.0                                # trimline = 300
    # trough flanks fill to the trimline; local column = zs - bed
    assert ice[3, 5] and ice[5, 5]
    np.testing.assert_allclose(z_fill[3, 5], zs)
    np.testing.assert_allclose(depth[3, 5], zs - 50.0)
    # spine: ice, column exactly HC*H, surface unchanged
    assert ice[jc, 5]
    np.testing.assert_allclose(z_fill[jc, 5], zs)
    np.testing.assert_allclose(depth[jc, 5], HC * 200.0)
    # nunatak: terrain above the ice surface stays rock
    assert not ice[2, 6]
    np.testing.assert_allclose(z_fill[2, 6], 1000.0)
    assert depth[2, 6] == 0.0
    # beyond the disc radius (d = 400 >= R): no ice
    assert not ice[0, 5] and not ice[8, 5]
    np.testing.assert_allclose(z_fill[0, 5], 50.0)


def test_smooth_ice_mask_outline():
    """The display-mask smoother (_smooth_ice_mask): sigma=0 reproduces
    the crisp order-0 upsample exactly (pre-smoothing behaviour); sigma>0
    rounds staircase corners while keeping straight edges pinned to the
    native cell-midpoint contour (the 1/2 level set), so the rendered
    glacier neither grows nor shrinks systematically."""
    from scipy.ndimage import map_coordinates
    from siim.plotting._render import _smooth_ice_mask
    ny, nx, o = 20, 24, 8
    ny_sub, nx_sub = (ny - 1) * o + 1, (nx - 1) * o + 1
    y_idx = np.arange(ny_sub) * (ny - 1) / (ny_sub - 1)
    x_idx = np.arange(nx_sub) * (nx - 1) / (nx_sub - 1)
    Y, X = np.meshgrid(y_idx, x_idx, indexing='ij')

    rect = np.zeros((ny, nx), dtype=bool)
    rect[5:15, 6:18] = True

    # sigma=0: bit-for-bit the order-0 nearest upsample
    crisp = _smooth_ice_mask(rect, Y, X, 0)
    ref = map_coordinates(rect.astype(np.uint8), [Y, X],
                          order=0, mode='nearest') > 0
    assert (crisp == ref).all()

    smooth = _smooth_ice_mask(rect, Y, X, float(o))
    # deep interior/exterior untouched
    assert smooth[10 * o, 12 * o] and not smooth[1 * o, 1 * o]
    # straight edges pinned: mid-span the boundary moves by <= 1 subpixel
    col = 12 * o
    assert abs(int(np.argmax(smooth[:, col])) -
               int(np.argmax(crisp[:, col]))) <= 1
    row = 10 * o
    assert abs(int(np.argmax(smooth[row, :])) -
               int(np.argmax(crisp[row, :]))) <= 1
    # convex corners shaved (the corner of the indicator smooths to ~1/4)...
    assert crisp[int(4.5 * o) + 1, int(5.5 * o) + 1]
    assert not smooth[int(4.5 * o) + 1, int(5.5 * o) + 1]
    # ...but the 1/2-level re-threshold keeps the area change small
    a_c, a_s = int(crisp.sum()), int(smooth.sum())
    assert a_s < a_c
    assert (a_c - a_s) / a_c < 0.05


def test_carve_semantics_straight_channel():
    """Strip with one icy channel row: instant mode lands footprint cells
    exactly on the surface-hung parabola (rim at zs = zb + HC*H, floor at
    the source bed) where they stood above it — including terrain above
    the ice surface (ridge-eating capture; no surface gate) and thin ridge
    ice; the channel row keeps its own bed (self-attribution targets the
    own bed — structural no-op); bed memory, floating (E=0) sources,
    border targets, the rate cap, and the never-additive arbitration all
    hold."""
    ny, nx, dx = 9, 12, 100.0
    nn = ny * nx
    H = np.zeros((ny, nx)); H[4, :] = 200.0           # channel along row 4
    H[3, 6] = 5.0                                      # thin ridge ice in the footprint
    alpha_g = 4.0                                      # trunk R = 400 m = 4 cells
    zb = np.full((ny, nx), 120.0)                      # walls below zs = 200
    zb[4, :] = 0.0                                     # channel bed at 0
    zb[6, 5] = -500.0                                  # relict overdeepening
    zb[2, 2] = 1e4                                     # tower above the ice surface
    # Receivers: column 0 is the base-level border (self-receiving);
    # every other cell drains to x- (interior cells never self-receive
    # under SFR — fastscape's basin correction points pits at spillways).
    rec = np.arange(nn, dtype=np.int64).reshape(ny, nx)
    rec[:, 1:] = rec[:, 1:] - 1
    rec = rec.ravel()
    # Pre-step bed: the kernel eroded the channel row by 0.5 m this step
    # (E_dt = zb_pre - zb_kern at sources); one wall cell also eroded by
    # its own kernel (0.2 m) to pin the never-additive arbitration; the
    # x=9 channel cell did not erode (floating source).
    zb_pre = zb.copy()
    zb_pre[4, :] += 0.5
    zb_pre[4, 9] = zb[4, 9]                            # floating: no erosion
    zb[5, 4] -= 0.2                                    # own kernel erosion
    # (zb_pre[5, 4] stays 120: the pre-step bed)

    # Seeds via the production builder: icy AND not self-receiving
    # (border cells are left out so interior sources inherit their
    # attribution). 12 = trunk row x>=1 (11) + the thin ridge cell.
    offsets = np.empty((ny, nx))
    n_seed = _carve_offsets(H.ravel(), rec, alpha_g, offsets.ravel(),
                            np.ones(H.size, np.int8))
    assert n_seed == 12
    D = np.empty((ny, nx)); SRC = np.empty((ny, nx), np.int64)
    _power_dt_2d(offsets, dx, dx, D, SRC)

    zb_flat = zb.ravel().copy()
    surf = zb_flat + HC * H.ravel()
    _carve_subgrid_width(zb_flat, zb_flat.copy(), zb_pre.ravel(),
                         H.ravel(), surf, rec, D, SRC,
                         offsets, np.inf, HC)
    z2 = zb_flat.reshape(ny, nx)

    R = 0.5 * alpha_g * 200.0
    # Parabola hung from the ice surface: rim at zs = HC*200, floor at the
    # source bed 0; at d=100: zb_s + HC*H*(d/R)^2.
    want = 0.0 + HC * 200.0 * (100.0 / R) ** 2                  # HC * 12.5
    np.testing.assert_allclose(z2[3, 4], want, atol=1e-9)
    np.testing.assert_allclose(z2[5, 5], want, atol=1e-9)
    # the channel row keeps its own bed (self-attribution no-op)
    assert (z2[4, 1:] == 0.0).all()
    # thin ridge ice inside the trunk footprint ("in a glacier and didn't
    # know it") IS carved toward the trunk parabola, surface = bed + HC*H
    np.testing.assert_allclose(z2[3, 6], want, atol=1e-9)
    np.testing.assert_allclose(surf.reshape(ny, nx)[3, 6], want + HC * 5.0,
                               atol=1e-9)
    # bed memory: relict hole below its target stays
    assert z2[6, 5] == -500.0
    # NO surface gate: the tower above the ice surface is consumed onto
    # the parabola (d = 200 m: zb_s + HC*H*(d/R)^2 = HC*50) — ridge-eating
    # capture is the point
    np.testing.assert_allclose(z2[2, 2], HC * 200.0 * 0.25, atol=1e-9)
    # rows beyond the footprint (d=400 >= R) untouched
    assert (z2[0, :] == 120.0).all() and (z2[8, :] == 120.0).all()
    # cells owned by the floating source: untouched (E=0 carves nothing)
    assert z2[3, 9] == 120.0 and z2[5, 9] == 120.0
    # border (self-receiving) TARGETS are never carved, even inside the
    # footprint: the border bed belongs to the border budget
    assert z2[3, 0] == 120.0 and z2[5, 0] == 120.0 and z2[4, 0] == 0.0
    assert surf.reshape(ny, nx)[3, 0] == 120.0
    # the carve never deposits
    assert (z2 <= zb).all()

    # rate-capped mode: descent limited to factor * E_dt from the
    # PRE-STEP bed — the cell's own kernel erosion and the carve
    # arbitrate, they never add
    zb_flat2 = zb.ravel().copy()
    surf2 = zb_flat2 + HC * H.ravel()
    _carve_subgrid_width(zb_flat2, zb_flat2.copy(), zb_pre.ravel(),
                         H.ravel(), surf2, rec, D, SRC,
                         offsets, 2.0, HC)
    z3 = zb_flat2.reshape(ny, nx)
    np.testing.assert_allclose(z3[3, 4], 120.0 - 2.0 * 0.5)
    # (5,4) already eroded 0.2 by its own kernel: total descent from the
    # pre-step bed is max(own, cap) = 1.0, not 0.2 + 1.0
    np.testing.assert_allclose(z3[5, 4], 120.0 - 2.0 * 0.5)


def test_carve_width_integration():
    """End-to-end mode-B run with carving on: finite, bare cells beside
    glacial trunks get pulled down (trough wider than the channel), and
    the border invariants survive (ice-free border cells present the
    water line)."""
    from siim import siim2d
    params = {
        'U': 1e-3, 'zELA': 200, 'beta': 1e-2, 'P': 1, 'alpha_g': 8,
        'Ko': 2e-6, 'n': 1, 'ce': 1e-4, 'nu': 2,
        'sliding_law': 'power', 'lambda_p': 500, 'k': .9,
        'T': 1e5, 'nt': 101, 'nt_out': 25,
        'Lx': 2e4, 'Ly': 2e4, 'nx': 41, 'ny': 41, 'seed': 7,
        'boundary_status': ['fixed_value'] * 4,
        'initial_max_elevation': 800,
        'mode': 'B', 'progress_bar': False,
        'carve_width': True, 'widening_rate': 0.0,
        # Isolate the carve from the mode-C standard: the carve-on / carve-off
        # twins below differ ONLY in carve_width.
        'trunk_surface': False, 'routing_relax': 0.0,
    }
    m = siim2d.siim(params)
    m.run()
    assert np.isfinite(m.z_out).all() and np.isfinite(m.zb_out).all()
    # Border presentation under the mode-C citizen: z_out at a border cell is the
    # TRACKED BED (ice-free) or bed + hc*H (icy). The carve never touches border
    # cells (it excludes self-receiving cells as targets — see
    # test_carve_semantics_straight_channel), so any below-water-line border bed
    # is a relict glacial overdeepening set by the kernel's flotation budget (bed
    # memory), NOT the carve punching through — the same behaviour the no-carve
    # citizen shows (z_out = zb + hc*H reports the true bed, not the water-line
    # presented surface the retired surface-replace class stored). What must hold
    # is that the outlet stays finite and physically bounded — no runaway
    # km-scale pillar/pit (the no-flow-reversal-cap pathology).
    border = np.zeros((41, 41), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    z_border = np.asarray(m.z_out[-1])[border]
    floor = -3.0 * float(m.hc_over_H) * float(m.H_out[-1].max())  # flotation-scale bound
    assert np.isfinite(z_border).all()
    assert z_border.min() > floor, (z_border.min(), floor)
    # baseline without carving: same run, carve off — the carve digs the
    # trough deeper/wider somewhere (dynamics diverge, so no global
    # one-sided claim holds end-to-end; never-deposits is pinned per-step
    # in test_carve_semantics_straight_channel)
    p0 = dict(params); p0['carve_width'] = False
    m0 = siim2d.siim(p0)
    m0.run()
    dzb = m.zb_out[-1] - m0.zb_out[-1]
    assert dzb.min() < -1.0
    # mode A + carve_width is refused
    with pytest.raises(ValueError):
        siim2d.siim({**params, 'mode': 'A'})
    # looped boundaries are supported (wrap-padded footprint transform):
    # the run completes finite with carving active
    ml = siim2d.siim({**params,
                      'boundary_status': ['looped', 'looped',
                                          'fixed_value', 'fixed_value']})
    ml.run()
    assert np.isfinite(ml.z_out).all() and np.isfinite(ml.zb_out).all()
    # widening_rate (eta) >= 0: negatives / NaN / unknown strings raise at
    # construction; eta = 0 (no net widening) and the instant sentinels are OK
    for bad in (-1.0, np.nan, 'fast'):
        with pytest.raises(ValueError):
            siim2d.siim({**params, 'widening_rate': bad})
    for ok in (0.0, None, 'inf'):
        siim2d.siim({**params, 'widening_rate': ok})   # constructs without raising


@pytest.mark.parametrize("value,expected", [
    (0.0, 1.0), (1.0, 2.0), (3.0, 4.0),                 # factor = 1 + eta
    (None, np.inf), (np.inf, np.inf), ('inf', np.inf),  # instant sentinels
    ('infinity', np.inf), ('Infinity', np.inf), ('  INF  ', np.inf),
])
def test_widening_factor_from_rate(value, expected):
    """User widening_rate (eta) -> internal E_widening/E_c factor (1 + eta)."""
    from siim.constants import widening_factor_from_rate
    assert widening_factor_from_rate(value) == expected


@pytest.mark.parametrize("bad", [-1e-9, -1.0, float('nan'), 'fast', 'two', ''])
def test_widening_factor_rejects(bad):
    """Negative eta, NaN, and unknown strings are rejected (true off is the
    separate switch carve_width=False)."""
    from siim.constants import widening_factor_from_rate
    with pytest.raises(ValueError):
        widening_factor_from_rate(bad)
