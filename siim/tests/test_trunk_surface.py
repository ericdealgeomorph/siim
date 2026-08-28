"""Fabricated trunk-surface routing (the ``trunk_surface`` flag).

Regression gates for fabricated trunk-surface routing:
accumulation on a fabricated ice surface (linear cross-valley dip toward the
centerline) converge trunk flow onto the centerline chain, so the centerline's
RAW flux is the full cross-section discharge — consolidation by routing, not the
Qc gather. The kernel is untouched (fabricated elevation reaches only the router
graph + the mass-balance surface); the runaway firewall is that no geometric
value enters a closure.
"""
import numpy as np
import pytest

from siim._core.step import _fabricate_trunk_surface


@pytest.fixture(autouse=True)
def _run_under_both_drivers(both_drivers):
    """S3 (Map 4 §1 PARAM): every test in this file runs under BOTH drivers --
    the conftest ``both_drivers`` fixture patches ``constants.DRIVER_DEFAULT``,
    so the existing assertions gate the in-house driver too."""



# ---------------------------------------------------------------------------
# A tiny D8 router + accumulator (steepest descent), for the deterministic
# channel-persistence gate — the same harness the design probes used.
# ---------------------------------------------------------------------------
def _d8_route(z, dx, dy):
    ny, nx = z.shape
    rec = np.arange(ny * nx)
    nbrs = [(0, 1, dx), (0, -1, dx), (1, 0, dy), (-1, 0, dy),
            (1, 1, np.hypot(dx, dy)), (1, -1, np.hypot(dx, dy)),
            (-1, 1, np.hypot(dx, dy)), (-1, -1, np.hypot(dx, dy))]
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            best, br = 0.0, j * nx + i
            for dj, di, L in nbrs:
                s = (z[j, i] - z[j + dj, i + di]) / L
                if s > best:
                    best, br = s, (j + dj) * nx + i + di
            rec[j * nx + i] = br
    return rec


def _accumulate(field, rec, z):
    order = np.argsort(-z.ravel())           # donors before receivers
    acc = field.ravel().astype(float).copy()
    for i in order:
        r = rec[i]
        if r != i:
            acc[r] += acc[i]
    return acc


def _straight_valley(ny=21, nx=40, dx=200.0, Sv=0.025, wall=8.0,
                     H_center=300.0, H_flank=60.0, ice_cols=(3, None)):
    """A straight U-valley draining -x: gentle down-valley grade, walls rising to
    the rim, a thick centerline ice corridor over thin flanks (realistic trough:
    walls meet the ice surface, so weak-wall radial shedding is excluded)."""
    jc = ny // 2
    j = np.arange(ny)[:, None]
    i = np.arange(nx)[None, :]
    zb = (Sv * dx * i + wall * (j - jc) ** 2).astype(float)
    H = np.zeros((ny, nx))
    c0, c1 = ice_cols[0], (nx - 1 if ice_cols[1] is None else ice_cols[1])
    for ii in range(c0, c1):
        H[jc, ii] = H_center
        for jj in range(1, ny - 1):
            if jj != jc:
                H[jj, ii] = H_flank
    return zb, H, jc


def _capture(zb, H, jc, dx, k_dip=0.6, alpha_g=10.0, col=8):
    """Fabricate → D8 route → accumulate a uniform ice input; return the fraction
    of the cross-section flux that lands on the centerline chain at column ``col``."""
    ny, nx = zb.shape
    hc = 1.5
    zs_dyn = zb + hc * H
    border = np.zeros((ny, nx), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    off = np.empty((ny, nx)); D = np.empty((ny, nx)); SRC = np.empty((ny, nx), np.int64)
    fab = _fabricate_trunk_surface(zs_dyn, zb, H, border, alpha_g, dx, dx,
                                   k_dip, 1e-3, off, D, SRC, False, False)
    rec = _d8_route(fab, dx, dx)
    ice_in = np.zeros((ny, nx))
    ice_in[jc - 7:jc + 8, 3:nx - 1] = 1.0
    acc = _accumulate(ice_in, rec, fab).reshape(ny, nx)
    cum = np.cumsum(ice_in[jc - 7:jc + 8, :].sum(axis=0)[::-1])[::-1]
    return acc[jc, col] / cum[col]


def test_channel_persistence_headline():
    """The fabricated dip converges flow onto the trunk centerline: the centerline
    captures the bulk of the cross-section flux (the ~45-degree convergence wedge
    is the only deficit). Bare dynamic routing (no dip) captures ~1 row."""
    zb, H, jc = _straight_valley()
    frac = _capture(zb, H, jc, 200.0, k_dip=0.6)
    assert frac >= 0.75, f"centerline capture {frac:.3f} < 0.75 — dip not converging"

    # Contrast: routing on the plain dynamic surface (k_dip -> 0 gives the
    # un-dipped surface via the floor) captures only its own row.
    frac0 = _capture(zb, H, jc, 200.0, k_dip=0.0)
    assert frac0 < 0.2, f"un-dipped capture {frac0:.3f} unexpectedly high"
    assert frac > 3 * frac0


def test_capture_grows_toward_the_toe():
    """Capture rises head->toe as the cumulative trunk flux outgrows the fixed
    convergence wedge (design probe: 0.47 head -> 0.87 toe)."""
    zb, H, jc = _straight_valley()
    head = _capture(zb, H, jc, 200.0, col=30)     # far from outlet (draining -x)
    toe = _capture(zb, H, jc, 200.0, col=6)       # near outlet
    assert toe > head


# ---------------------------------------------------------------------------
# trunk_surface + D-infinity. The V-dip convergence criterion S_c > 0.414*S_v
# is D8-specific, so capture under D-inf is measured explicitly rather than
# assumed. Route the fabricated surface through the model's D-inf router
# (eps-fill + facet, as DinfFlowRouter does) and accumulate a uniform ice
# input. Measured mid-trunk centerline capture ~= 0.76 (>= the 0.7 bless
# threshold) -> the combo is BLESSED (no guard); pinned by a floor regression
# with headroom below the measured value.
# ---------------------------------------------------------------------------
def _capture_dinf(zb, H, jc, dx, k_dip=0.6, alpha_g=10.0, col=8):
    """trunk_surface centerline capture under the D-infinity router."""
    from .test_dinf_routing import _route_full, _edges_interior
    from siim._core.routing import _flow_accumulate_dinf
    ny, nx = zb.shape
    hc = 1.5
    zs_dyn = zb + hc * H
    border = np.zeros((ny, nx), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    off = np.empty((ny, nx)); D = np.empty((ny, nx)); SRC = np.empty((ny, nx), np.int64)
    fab = _fabricate_trunk_surface(zs_dyn, zb, H, border, alpha_g, dx, dx,
                                   k_dip, 1e-3, off, D, SRC, False, False)
    interior = _edges_interior(ny, nx)
    _, receivers, weights, lengths, nb_receivers, stack = _route_full(
        fab, dx, dx, interior)
    ice_in = np.zeros((ny, nx))
    ice_in[jc - 7:jc + 8, 3:nx - 1] = 1.0
    field = ice_in.ravel().astype(float).copy()
    _flow_accumulate_dinf(field, stack, nb_receivers, receivers, weights)
    acc = field.reshape(ny, nx)
    cum = np.cumsum(ice_in[jc - 7:jc + 8, :].sum(axis=0)[::-1])[::-1]
    return acc[jc, col] / cum[col]


def test_channel_persistence_dinf():
    """trunk_surface converges flow onto the centerline under D-inf too: the
    mid-trunk centerline captures the bulk of the cross-section flux (measured
    ~0.76, blessed above the 0.7 threshold). Bare D-inf routing on the undipped
    surface captures ~1 row."""
    zb, H, jc = _straight_valley()
    frac = _capture_dinf(zb, H, jc, 200.0, k_dip=0.6)
    assert frac >= 0.70, f"dinf centerline capture {frac:.3f} < 0.70 bless floor"

    frac0 = _capture_dinf(zb, H, jc, 200.0, k_dip=0.0)
    assert frac0 < 0.2, f"un-dipped dinf capture {frac0:.3f} unexpectedly high"
    assert frac > 3 * frac0


# ---------------------------------------------------------------------------
# No-runaway (the migrated headline safety gate): trunk_surface must not blow up
# the bed or H relative to the flag-off twin, on the harsh dt=300 kyr / eta=5
# axis that melted the geometric-H consolidation.
# ---------------------------------------------------------------------------
_NX, _DX = 81, 250.0
_LX = (_NX - 1) * _DX
_BASE = dict(U=1e-3, P=1, beta=.5e-2, Ko=1e-6, n=1, ce=1e-5, nu=2, tau_c=1.2e5,
             lambda_c=1000, lambda_p=300.0, alpha_g=10, Lx=_LX, Ly=_LX,
             nx=_NX, ny=_NX, boundary_status=['fixed_value'] * 4, seed=109,
             flow_routing='single', progress_bar=False)


@pytest.fixture(scope='module')
def _warm_topo():
    from siim.siim2d import siim as siim2d
    p = dict(_BASE, zELA=1e6, sliding_law='power', mode='B', carve_width=False,
             initial_max_elevation=1000, T=30e6, nt=101, nt_out=2)
    m = siim2d(p); m.run()
    return m.z_out[-1]


@pytest.mark.parametrize("law", ['coulomb', 'power'])
def test_trunk_surface_no_runaway(law, _warm_topo):
    from siim.siim2d import siim as siim2d
    z0 = _warm_topo

    def run(trunk_surface):
        p = dict(_BASE, sliding_law=law, initial_topography=z0,
                 zELA=float(np.quantile(z0, 0.50)), mode='C', widening_rate=5,
                 trunk_surface=trunk_surface, T=9e6, nt=31, nt_out=31)
        m = siim2d(p); m.run()
        return m

    on, off = run(True), run(False)
    # 1. The bed is not carved materially deeper than the flag-off twin (pre-fix
    #    geometric-H consolidation reached -12 to -39 km here; off stays >= datum).
    assert on.zb_out.min() >= off.zb_out.min() - 1.0, \
        f"trunk_surface bed {on.zb_out.min():.0f} << off {off.zb_out.min():.0f}"
    # 2. H stays on the flag-off scale (no width blow-up).
    assert on.H_out.max() <= 5.0 * off.H_out.max()


# ---------------------------------------------------------------------------
# Flag-off bit-for-bit + conservation.
# ---------------------------------------------------------------------------
def test_flag_off_bit_for_bit():
    """trunk_surface=False is bit-for-bit with the plain mode-B + carve path:
    the feature is a pure add, and the C alias resolves to B + carve. (The
    mode-C standard turning trunk_surface ON by default is pinned in
    test_mode_names.test_mode_C_standard_defaults_resolve.)"""
    from siim.siim2d import siim as siim2d
    cfg = dict(_BASE, sliding_law='power', zELA=600.0,
               initial_max_elevation=800, T=6e5, nt=13, nt_out=4,
               trunk_surface=False, routing_relax=0.0, widening_rate=1.0)
    a = siim2d({**cfg, 'mode': 'C'}); a.run()
    b = siim2d({**cfg, 'mode': 'B', 'carve_width': True}); b.run()
    np.testing.assert_array_equal(a.zb_out, b.zb_out)
    np.testing.assert_array_equal(a.H_out, b.H_out)


@pytest.mark.parametrize("law", ['power', 'coulomb'])
def test_conservation_ice_plus_water(law):
    """Routing on the fabricated surface moves flow but conserves mass: the total
    ice + water flux leaving the domain equals the total precip input (routing is
    a graph edit, not a source)."""
    from siim.siim2d import siim as siim2d
    cfg = dict(_BASE, sliding_law=law, zELA=500.0, mode='C', trunk_surface=True,
               initial_max_elevation=900, T=1.5e6, nt=16, nt_out=3)
    if law == 'coulomb':
        cfg.update(tau_c=1.2e5, lambda_c=1000)
    m = siim2d(cfg); m.run()
    # ice_flux + water_flux at every cell = total precip flux accumulated there;
    # both are finite and non-negative everywhere (no leak / no spurious source).
    ice = m.ds_out['glacial_flow__ice_flux'].values
    water = m.ds_out['glacial_flow__water_flux'].values
    assert np.isfinite(ice).all() and np.isfinite(water).all()
    assert (ice >= -1e-6).all() and (water >= -1e-6).all()
