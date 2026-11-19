"""The 1D toe slope correction (``toe_slope_correction``, siim1d modes A and B; on by default).

siim1d builds each cell's rise from the slope at its upstream node. Near a terminus the steady surface is singular,
S = a Q^-phi with Q ~ s (s the distance from the terminus), so that step pins the terminus to its cell. The correction
scales the upwind slope of every icy node whose receiver carries less flux by c(q), q = Q_receiver/Q_node, and gives
the last icy node the secant from the sub-grid terminus times 1 - phi, which together make the step exact there.
"""
import numpy as np
import pytest

from siim import constants
from siim.siim1d import _toe_flux_factor, _toe_receiver, siim as siim1d


def _cell_factor(u, phi):
    """c(u), the same factor written by the cell's distance u = s/dx from the terminus when Q ~ s (the form first
    built, since replaced in the model by c(q) of the flux ratio; kept here as the reference c(q) is checked against):
    exact for z - z_t ~ s^(1 - phi) on the cell from u - 1 to u."""
    return (1.0 - phi) * u ** -phi / (u ** (1.0 - phi) - (u - 1.0) ** (1.0 - phi))


def _toe_state(law, zeta, outlet, mode='A', nx=121, j_from_outlet=30, S_f=0.07):
    """A model holding a synthetic terminus a fraction ``zeta`` of a cell beyond its last icy node: the flux Q = g s
    and the balance put the flux integral's zero there, the surface is the exact toe S = a Q^-phi above it and a
    fluvial ramp of slope ``S_f`` below. Returns (model, j, s): j the last icy node, s the signed distance of every
    node from the terminus (positive on the ice)."""
    right = outlet == 'right'
    m = siim1d(dict(mode=mode, sliding_law=law, nx=nx, L=(nx - 1) * 100.0, nt=2, T=1.0, progress_bar=False,
                    **(dict(left_bc='reflecting', right_bc='base_level') if right else {})))
    m.initialize_simulation()
    up = -1 if right else 1                               # the upstream direction
    j = nx - 1 - j_from_outlet if right else j_from_outlet
    k = np.arange(nx)
    s = (zeta + up * (k - j)) * m.dx                       # s > 0 on the ice
    xi = (m.x[m.didx_r] - m.x + m.xo_r) if right else (m.x - m.x[m.didx_l] + m.xo_l)
    Qg = np.where(s > 0, 50.0 * s, 0.0)
    i1 = Qg[j] * xi[j] ** (m.d * (m.sigma - 1)) / (m.k_h * m.sigma * m.d)
    r = j - up
    b = -2 * i1 / (zeta * (xi[r] - xi[j]) * (xi[r] ** (m.d * m.sigma - 1) + xi[j] ** (m.d * m.sigma - 1)))
    m.Qg, m.B = Qg, np.full(nx, b)
    phi, a = m.phi_toe, 0.9
    m.z = 1000.0 + np.where(s > 0, a * np.abs(s) ** (1 - phi) / (1 - phi), S_f * s)
    m._toe = m._toe_cells()                     # as calculate_ice_flux sets it
    return m, j, s


def _ice(m, j, outlet):
    """The icy nodes from the last one, j, to the divide."""
    return np.arange(j, m.didx_l + 1) if outlet == 'left' else np.arange(m.didx_r, j + 1)


def test_correction_is_on_by_default():
    """The correction is the default (constants.TOE_SLOPE_CORRECTION, ruled 2026-11-18)."""
    assert constants.TOE_SLOPE_CORRECTION is True
    assert siim1d(dict(nt=2, T=1.0, progress_bar=False)).toe_slope_correction is True


@pytest.mark.pin
@pytest.mark.parametrize('outlet', ['left', 'right'])
@pytest.mark.parametrize('zeta', [0.1, 0.5, 0.9])
@pytest.mark.parametrize('law', ['power', 'eff-exp', 'coulomb'])
def test_toe_slope_is_exact_on_the_singular_toe(law, zeta, outlet):
    """On z - z_t = a s^(1-phi)/(1-phi) with Q ~ s the corrected slope of every icy node, the last one included, is
    the exact a s^-phi to rounding (the uncorrected upwind slope is 25-50 % off at the toe); no bare node is
    touched, and only the last icy node extends its receiver to the terminus."""
    m, j, s = _toe_state(law, zeta, outlet)
    cs, wx = m._toe_cells()
    slope = np.zeros(m.nx)
    m._toe_slope(m.z, slope, (cs, wx))
    ice = _ice(m, j, outlet)
    assert ice.size > 80
    np.testing.assert_allclose(slope[ice], 0.9 * s[ice] ** -m.phi_toe, rtol=1e-11, atol=0)
    assert cs[j] == pytest.approx((1 - m.phi_toe) / zeta, rel=1e-12)
    rest = np.setdiff1d(np.arange(m.nx), ice)
    assert np.all(cs[rest] == 1.0) and np.all(wx[rest] == 0.0)
    assert np.flatnonzero(wx).tolist() == [j]


def test_last_node_zeta_is_the_continued_flux_ratio():
    """The last icy node's zeta is 1/(1 - q) with q the trapezoid-continued (negative) flux ratio across its cell;
    with ice to the outlet there is no terminus, and where the flux grows downstream nothing is corrected."""
    for zeta in (0.2, 0.7):
        m, j, s = _toe_state('power', zeta, 'left')
        cs, wx = m._toe_cells()
        xi = m.x - m.x[m.didx_l] + m.xo_l
        i1 = m.Qg[j] * xi[j] ** (m.d * (m.sigma - 1)) / (m.k_h * m.sigma * m.d)
        i0 = i1 + 0.5 * m.B[j] * (xi[j - 1] ** (m.d * m.sigma - 1) + xi[j] ** (m.d * m.sigma - 1)) * (xi[j - 1] - xi[j])
        q = i0 / i1
        assert 1.0 - wx[j] == pytest.approx(1.0 / (1.0 - q), rel=1e-13) and 1.0 - wx[j] == pytest.approx(zeta)
    m.Qg = 1e5 + 50.0 * np.arange(m.nx) * m.dx          # ice to the outlet, flux falling toward it
    cs, wx = m._toe_cells()
    assert not wx.any() and np.all(cs[1:] < 1.0)
    m.Qg = m.Qg[::-1].copy()                              # flux growing downstream everywhere
    cs, wx = m._toe_cells()
    assert np.all(cs == 1.0) and not wx.any()


def test_toe_phi_is_the_thin_ice_limit():
    """phi_t is each law's thin-ice limit: 1/3 (power), 1/2 (Coulomb), mu/nu (effective exponent)."""
    phi = {law: siim1d(dict(sliding_law=law, nt=2, T=1.0, progress_bar=False)) for law in ('power', 'coulomb',
                                                                                           'eff-exp')}
    assert phi['power'].phi_toe == 1 / 3 and phi['coulomb'].phi_toe == 0.5
    assert phi['eff-exp'].phi_toe == phi['eff-exp'].mu / phi['eff-exp'].nu


@pytest.mark.parametrize('phi', [4 / 15, 1 / 3, 1 / 2])
def test_toe_factor_tends_to_one_away_from_the_toe(phi):
    """c(q) rises monotonically from 1 - phi as q -> 0 toward 1 as 1 - phi (1 - q)/2 + O((1 - q)^2)."""
    q = 1.0 - 1.0 / np.geomspace(1.0 + 1e-12, 1e4, 400)
    c = _toe_flux_factor(q, phi)
    assert _toe_flux_factor(np.array([1e-300]), phi)[0] == pytest.approx(1 - phi, rel=1e-15)
    assert np.all(np.diff(c) > 0) and np.all(c < 1)
    far = q >= 0.5
    assert np.all(np.abs(c[far] - (1 - phi * (1 - q[far]) / 2)) <= phi * (1 - q[far]) ** 2)


@pytest.mark.pin
@pytest.mark.parametrize('phi', [4 / 15, 1 / 3, 1 / 2])
def test_flux_factor_is_the_cell_factor_where_the_flux_is_linear_from_the_toe(phi):
    """c(q) at q = (u - 1)/u, the flux ratio of a cell u cells from the terminus when Q ~ s, is c(u) to rounding."""
    u = np.geomspace(1.0 + 1e-6, 1e4, 300)
    np.testing.assert_allclose(_toe_flux_factor((u - 1) / u, phi), _cell_factor(u, phi), rtol=1e-9, atol=0)
    np.testing.assert_allclose(_toe_flux_factor(1 - np.geomspace(1e-14, 1e-6, 9), phi), 1.0, rtol=1e-6)


@pytest.mark.parametrize('law', ['power', 'eff-exp', 'coulomb'])
def test_mode_b_walk_solves_the_closure_at_the_corrected_slope(law):
    """Mode B's joint walk takes the same toe correction as the explicit slope: mode A's closure, read off the walk's
    own surface zb + hc H with the corrected slope, returns the walk's H at every icy node."""
    m, j, s = _toe_state(law, 0.4, 'left', mode='B')
    m.zb = m.z - 50.0 * (s > 0)                            # a bed under a 33 m column (hc 1.5)
    m.calculate_ice_thickness()
    H_b = m.H.copy()
    m.z = m.zb + m.hc_over_H * H_b
    m._mode_B_active = False
    m.calculate_ice_thickness()
    ice = _ice(m, j, 'left')
    assert np.all(H_b[ice] > 0)
    np.testing.assert_allclose(m.H[ice], H_b[ice], rtol=1e-8)


@pytest.mark.parametrize('law', ['power', 'eff-exp', 'coulomb'])
def test_mode_b_walk_takes_the_same_fallback(law):
    """Below the toe a bare reach steeper than the walk's corrected toe slope (0.3): the walk solves the last icy node
    at the uncorrected step, the branch the explicit slope and the kernels pick on the walk's own surface, so mode A's
    closure read off that surface still returns the walk's H."""
    m, j, s = _toe_state(law, 0.4, 'left', mode='B', S_f=0.3)
    m.zb = m.z - 50.0 * (s > 0)
    m.calculate_ice_thickness()
    H_b = m.H.copy()
    m.z = m.zb + m.hc_over_H * H_b
    cs, wx = m._toe
    assert wx[j] > 0.0 and _toe_receiver(m.z, m.z[j], j, 1, cs, wx) == (m.z[j - 1], 1.0)
    m._mode_B_active = False
    m.calculate_ice_thickness()
    ice = _ice(m, j, 'left')
    np.testing.assert_allclose(m.H[ice], H_b[ice], rtol=1e-8)


class _Neutral(siim1d):
    """The correction on, with every factor neutral (cs = 1, wx = 0): the corrected code path with nothing to do."""
    built = False

    def _toe_cells(self):
        toe = super()._toe_cells()
        if toe is None:
            return None
        self.built = True
        return np.ones(self.nx), np.zeros(self.nx)


@pytest.mark.pin
@pytest.mark.parametrize('mode', ['A', 'B'])
@pytest.mark.parametrize('law', ['power', 'eff-exp', 'coulomb'])
def test_off_is_the_uncorrected_model(law, mode):
    """toe_slope_correction=False takes the untouched uncorrected path (no factors are built, the kernels get none),
    and the corrected path with neutral factors reproduces it bit for bit, so the correction changes a run only
    through its factors. A glacier with a terminus inside the domain, so on and off differ."""
    nx = 101
    cfg = dict(mode=mode, sliding_law=law, nx=nx, L=10e3, T=2e4, nt=21, nt_out=21, zELA=3500.0, progress_bar=False,
               initial_topography=3000.0 + 2000.0 * np.linspace(0, 1, nx) ** 0.5)
    off = siim1d(dict(cfg, toe_slope_correction=False))
    neutral, on = _Neutral(dict(cfg)), siim1d(dict(cfg))
    for m in (off, neutral, on):
        m.run()
    assert off._toe_cells() is None and neutral.built
    for f in ('z_out', 'zb_out', 'H_out', 'erosion_rate_out', 'ub_out'):
        assert np.array_equal(getattr(off, f), getattr(neutral, f)), f
    assert not np.array_equal(off.z_out, on.z_out)


@pytest.mark.parametrize('outlet', ['left', 'right'])
@pytest.mark.parametrize('law', ['power', 'coulomb'])
def test_terminus_beside_the_outlet_is_corrected(law, outlet):
    """A terminus one cell from a base-level outlet: the outlet-adjacent node is the last icy node and takes
    (1 - phi)/zeta on the secant to a toe at the outlet's level (wx = 0: the bare outlet has no downstream rise to
    extend), exact where the bare reach below the terminus is flat; the erosion kernel takes it too and stays finite."""
    m, j, s = _toe_state(law, 0.4, outlet, j_from_outlet=1, S_f=0.0)
    assert j == (1 if outlet == 'left' else m.nx - 2)
    cs, wx = m._toe_cells()
    assert wx[j] == 0.0 and cs[j] == pytest.approx((1 - m.phi_toe) / 0.4, rel=1e-12)
    slope = np.zeros(m.nx)
    m._toe_slope(m.z, slope, (cs, wx))
    ice = _ice(m, j, outlet)
    np.testing.assert_allclose(slope[ice], 0.9 * s[ice] ** -m.phi_toe, rtol=1e-11, atol=0)
    m.calculate_ice_thickness()
    np.copyto(m.z_o, m.z)
    m.calculate_erosion()
    assert np.isfinite(m.z).all() and np.isfinite(m.H).all() and m.H[j] > 0
    assert np.all(m.z <= m.z_o)                               # erosion only, no node raised


def test_receiver_extension_stays_in_the_array():
    """_toe_receiver extends the receiver only where the receiver has a downstream node: next to either end it
    reads nothing past the array (numba does not bounds-check; z[-1] would wrap to the far end)."""
    cs, wx = np.full(5, 0.7), np.full(5, 0.5)
    z = np.array([10.0, 20.0, 30.0, 40.0, 1e6])
    assert _toe_receiver(z, 25.0, 1, 1, cs, wx) == (10.0, 0.7)            # left: receiver is the outlet, node 0
    z = np.array([1e6, 40.0, 30.0, 20.0, 10.0])
    assert _toe_receiver(z, 25.0, 3, -1, cs, wx) == (10.0, 0.7)           # right: receiver is node nx - 1
    # gate: the node at 35 falls back (corrected slope 0.7 * 10 <= the bare rise 10), so 45 checks the extension
    assert _toe_receiver(z, 45.0, 2, -1, cs, wx) == (25.0, 0.7)           # interior: 20 + 0.5 * (20 - 10)
    assert _toe_receiver(z, 35.0, 2, -1, cs, wx) == (20.0, 1.0)           # 0.7 * (35 - 25) <= 10: uncorrected


@pytest.mark.pin
@pytest.mark.parametrize('outlet', ['left', 'right'])
def test_last_node_term_needs_a_glacier_the_grid_resolves(outlet):
    """The last icy node takes (1 - phi)/zeta only where its block (the glacier draining to that terminus) holds at
    least constants.TOE_MIN_CELLS icy nodes; one node fewer and it keeps the uncorrected step (cs = 1, wx = 0) while
    the block's interior keeps c(q)."""
    n, up = constants.TOE_MIN_CELLS, -1 if outlet == 'right' else 1
    for cells in (n - 1, n):
        m, j, s = _toe_state('power', 0.4, outlet)
        block = j + up * np.arange(cells)
        m.Qg[np.setdiff1d(_ice(m, j, outlet), block)] = 0.0       # the glacier is the block alone
        cs, wx = m._toe_cells()
        inner = block[1:]                                        # icy receivers of smaller flux: c(q)
        assert np.all(cs[inner] < 1.0) and not wx[inner].any()
        if cells < n:
            assert cs[j] == 1.0 and wx[j] == 0.0
        else:
            assert cs[j] == pytest.approx((1 - m.phi_toe) / 0.4, rel=1e-12) and wx[j] == pytest.approx(0.6)


@pytest.mark.pin
@pytest.mark.parametrize('outlet', ['left', 'right'])
def test_guard_counts_a_block_that_reaches_the_divide(outlet):
    """A glacier running from its terminus up to the divide (the end of its side) with exactly TOE_MIN_CELLS icy
    nodes takes the last-node term; one node fewer, and the count runs off the side, so the node is guarded."""
    n = constants.TOE_MIN_CELLS
    for cells in (n - 1, n):
        m, j, s = _toe_state('coulomb', 0.4, outlet, j_from_outlet=120 - (cells - 1))
        assert _ice(m, j, outlet).size == cells
        cs, wx = m._toe_cells()
        if cells < n:
            assert cs[j] == 1.0 and wx[j] == 0.0
        else:
            assert cs[j] == pytest.approx((1 - m.phi_toe) / 0.4, rel=1e-12)


@pytest.mark.pin
def test_short_glaciers_at_a_coarse_divide_keep_their_ice(monkeypatch):
    """Power-law mode B on 250 m cells (the paper suite's fixed 1D strip: 50 km, the 1000 m (x/L)^2 ramp, zELA 1500 m,
    Ko and ce x10): its limit cycle passes through glaciers of 1-4 cells at the divide, where the last-node term
    strips the divide of its ice (the 4-8 Myr numbers are at constants.TOE_MIN_CELLS). Checked over 1.5-3 Myr, the
    guard on and then off (TOE_MIN_CELLS = 1), so the guard is what keeps the ice."""
    # gate: new in this change; reviewed to a shorter window with a guard-off control (same bounds)
    nx, T = 201, 3e6

    def divide():
        m = siim1d(dict(U=1e-3, zELA=1500.0, P=1.0, beta=1e-2, Ko=1e-5, n=1, ce=1e-4, nu=2, alpha_g=5.0,
                        sliding_law='power', lambda_p=300.0, k_h=250.0, d=1.0, sigma=1.0, xo=250.0, L=50e3, nx=nx,
                        T=T, nt=3001, nt_out=301, mode='B', cap_ice_accumulation=True, progress_bar=False,
                        initial_topography=1000.0 * np.linspace(0.0, 1.0, nx) ** 2 + 1.5e-3))
        m.run()
        back = np.asarray(m.output_times) >= 1.5e6
        return m.H_out[-1, back], m.z_out[-1, back]

    H_div, zo = divide()
    assert np.all(H_div > 0.0)
    assert zo.min() > 1490.0
    monkeypatch.setattr(constants, 'TOE_MIN_CELLS', 1)
    H_div, zo = divide()
    assert np.mean(H_div <= 0.0) > 0.1 and zo.min() < 1490.0


def test_last_node_zeta_is_floored():
    """A tiny flux at the last icy node puts the terminus right at it (zeta -> 0); zeta is held at
    constants.TOE_ZETA_MIN, so the factor stays (1 - phi)/TOE_ZETA_MIN and the step finite."""
    m, j, s = _toe_state('coulomb', 0.5, 'left')
    m.Qg[j] = 1e-200
    cs, wx = m._toe_cells()
    assert cs[j] == pytest.approx((1 - m.phi_toe) / constants.TOE_ZETA_MIN, rel=1e-12)
    assert wx[j] == pytest.approx(1 - constants.TOE_ZETA_MIN, rel=1e-12)


def test_non_positive_corrected_slope_takes_the_uncorrected_step():
    """Where the last node sits at or below the toe its corrected secant would put (slope <= 0, which would delete
    its ice), the explicit slope and the kernels take the uncorrected step: cs = 1 against the plain receiver."""
    m, j, s = _toe_state('power', 0.5, 'left')
    cs, wx = m._toe_cells()
    z = m.z.copy()
    z[j] = z[j - 1] + 0.4 * wx[j] * (z[j - 1] - z[j - 2])      # below the extended toe, above the receiver
    slope = np.zeros(m.nx)
    m._toe_slope(z, slope, (cs, wx))
    assert slope[j] == pytest.approx((z[j] - z[j - 1]) / m.dx, rel=1e-12) and slope[j] > 0
    assert _toe_receiver(z, z[j], j, 1, cs, wx) == (z[j - 1], 1.0)


@pytest.mark.pin
def test_corrected_slope_no_steeper_than_the_bare_one_takes_the_uncorrected_step():
    """A last icy node whose surface barely clears the extended toe (one just turned icy, its surface still the bare
    one) would read a corrected slope near zero, which mode A's closure turned into a one-step H spike (18.7 km on
    200 m cells: cs 8.73, wx 0.916, the surface 0.5 mm above the toe). Where the corrected slope is no steeper than the
    bare slope the toe was extended along, the node takes the uncorrected step, so its H is the uncorrected model's;
    a surface clearing the toe by more keeps the correction."""
    cs, wx = np.array([1.0, 1.0, 8.73]), np.array([0.0, 0.0, 0.916])
    z = np.array([2739.8218 - 6.2802, 2739.8218, 0.0])
    zt = z[1] + wx[2] * (z[1] - z[0])                     # the extended toe
    for above in (5e-4, 0.999 * 6.2802 / 8.73):        # the spike's 0.5 mm; a corrected slope just under the bare
        assert _toe_receiver(z, zt + above, 2, 1, cs, wx) == (z[1], 1.0)
    assert _toe_receiver(z, zt + 1.0, 2, 1, cs, wx) == (zt, 8.73)
    m, j, s = _toe_state('power', 0.5, 'left')
    cs, wx = m._toe
    assert cs[j] > 1.0 and wx[j] > 0.0
    m.z[j] = m.z[j - 1] + wx[j] * (m.z[j - 1] - m.z[j - 2]) + 5e-4
    H = []
    for toe in ((cs, wx), None):                            # corrected, then the uncorrected model
        m._toe = toe
        m.calculate_ice_thickness()
        H.append(m.H[j])
    assert H[0] == H[1] and 0.0 < H[0] < 1e3


@pytest.mark.parametrize('law', ['power', 'eff-exp', 'coulomb'])
def test_toe_two_cells_from_a_drowned_border_reads_no_border_bed(law):
    """Last icy node two cells from the outlet, the border bed drowned 400 m below bl: the toe is taken at the bare
    receiver's level (wx = 0), so the mode-B walk, which sees the raw border bed, and the kernels, whose working
    surface floors it at bl, take the same receiver; the walk's H does not depend on the border bed and the step
    stays finite."""
    m, j, s = _toe_state(law, 0.4, 'left', mode='B', j_from_outlet=2)
    cs, wx = m._toe
    assert j == 2 and wx[j] == 0.0 and cs[j] == pytest.approx((1 - m.phi_toe) / 0.4, rel=1e-12)
    m.zb = m.z - 50.0 * (s > 0)
    H = []
    for zb0 in (m.zb[0], -400.0):                           # the bare ramp's border, then a drowned one
        m.zb[0], m.H[:] = zb0, 0.0
        m.calculate_ice_thickness()
        H.append(m.H.copy())
    assert np.array_equal(H[0], H[1]) and np.isfinite(H[1]).all() and H[1][j] > 0
    z = m.zb + m.hc_over_H * m.H
    assert _toe_receiver(z, z[j], j, 1, cs, wx) == (m.zb[j - 1], cs[j])  # the walk's receiver, the bare node
    m.z = z
    np.copyto(m.z_o, m.z)
    m.calculate_erosion()
    assert np.isfinite(m.z).all() and np.all(m.z <= m.z_o)
