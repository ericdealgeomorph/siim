"""The 2D toe slope correction (``toe_slope_correction_2d``), EXPERIMENTAL and off by default.

Near a terminus the steady surface is singular, S ~ Q^-phi with Q ~ s (s the distance from the terminus), and the
upwind step (z_i - z_r)/L pins the terminus to its cell. The correction multiplies each routing edge's glacial slope
by c(q) = (1 - phi)(1 - q)/(1 - q^(1 - phi)), q = Q_r/Q*, Q* = min(Q_i, I_r), where 0 < q < 1, and gives an edge to a
bare receiver the slope from the sub-grid terminus, where at least constants.TOE_MIN_CELLS icy cells route through the
node (siim._core.step.toe_slope_factor). Both are exact for z - z_t ~ s^(1 - phi).
"""
import numpy as np
import pytest

from siim import constants
from siim._core.step import (accumulate_glacial_flow, build_glacial_params, run_modeA_step, toe_phi,
                             toe_slope_factor, _H_from_QS_modeA, _toe_inflow_cells_kernel, _toe_slope_factor_kernel)
from siim.siim2d import siim as siim2d

LAWS = ('power', 'eff-exp', 'coulomb')


def _law(law):
    return build_glacial_params(sliding_law=law, Ko=1e-6, ce=1e-5, n=1.0, nu=2.0, m=None, mu=None, Ac=constants.AC,
                                alpha_g=5.0, lambda_p=300.0, lambda_c=None, tau_c=constants.TAU_C,
                                coulomb_clamp=constants.COULOMB_CLAMP, hc_over_H=1.5, H_diffusivity=None)


def _cf(q, phi):
    return (1 - phi) * (1 - q) / (1 - q ** (1 - phi))


def _chain(phi, zeta, layout, n=60, j=20, dx=250.0, a=0.9, S_f=0.07):
    """A flow line of n nodes draining to node 0 (a self-receiving border), the terminus a fraction ``zeta`` of an
    edge beyond its last icy node j (so n - j icy nodes): Q = q0 s on the ice (s the distance from the terminus),
    continued linearly to the bare receiver j - 1 (its pre-clamp flux), the exact toe z - z_t = a s^(1-phi)/(1-phi)
    above it and a fluvial ramp of slope S_f below. ``layout`` 'sfr' gives (n,) receivers and the receiver-first
    stack, 'kary' the (n, 2) multi-receiver pack with one active slot and the donor-first stack. Returns the router
    arrays, z, Q_raw, s and the exact slope a s^-phi."""
    k = np.arange(n)
    s = (k - j + zeta) * dx
    Q_raw = 50.0 * s                                         # m^3/yr; negative below the terminus
    z = 1000.0 + np.where(s > 0, a * np.abs(s) ** (1 - phi) / (1 - phi), S_f * s)
    rec = np.maximum(k - 1, 0)
    lens = np.full(n, dx)
    if layout == 'sfr':
        rt = dict(receivers=rec, nb_receivers=None, weights=None, lengths=lens, stack=k)
    else:
        rt = dict(receivers=np.stack([rec, rec], axis=1), nb_receivers=np.ones(n, np.int64),
                  weights=np.stack([np.ones(n), np.zeros(n)], axis=1), lengths=np.stack([lens, lens], 1),
                  stack=k[::-1])
    return rt, z, Q_raw, s, a * np.abs(s) ** -phi


def _factor(z, Q_raw, rt, phi):
    return toe_slope_factor(z, Q_raw, rt['receivers'], rt['nb_receivers'], rt['weights'], rt['lengths'], phi,
                            rt['stack'])


@pytest.mark.pin
@pytest.mark.parametrize('layout', ['sfr', 'kary'])
@pytest.mark.parametrize('zeta', [0.1, 0.5, 0.9])
@pytest.mark.parametrize('law', LAWS)
def test_toe_slope_is_exact_on_the_singular_toe(law, zeta, layout):
    """On z - z_t = a s^(1-phi)/(1-phi) with Q ~ s, the corrected slope c*(z_i - z_r)/L of every icy node is the exact
    a s^-phi to rounding, the last icy node included (40 icy cells); bare nodes keep c = 1. The bare ramp (0.05) is
    gentler than the toe's surface at the last node (0.06 for Coulomb at zeta 0.9), the premise of extending it
    under the toe: a toe gentler than the ramp takes the uncorrected step."""
    code, gp = _law(law)
    phi = toe_phi(code, gp)
    rt, z, Q_raw, s, exact = _chain(phi, zeta, layout, S_f=0.05)
    c = _factor(z, Q_raw, rt, phi)
    c = c if layout == 'sfr' else c[:, 0]
    icy = s > 0
    slope = c[icy] * (z[icy] - z[np.flatnonzero(icy) - 1]) / 250.0
    np.testing.assert_allclose(slope, exact[icy], rtol=1e-11, atol=0)
    assert np.all(c[~icy] == 1.0)


@pytest.mark.pin
@pytest.mark.parametrize('law', LAWS)
def test_mode_A_closure_reads_the_corrected_slope(law):
    """Through run_modeA_step (SFR, dt = 0): the thickness of every icy node is the point closure H(Q, S) at the
    exact toe slope, and with the correction off it is the closure at the upwind slope."""
    code, gp = _law(law)
    phi = toe_phi(code, gp)
    rt, z, Q_raw, s, exact = _chain(phi, 0.4, 'sfr')
    n = z.size
    Q = np.maximum(Q_raw, 0.0)[None, :]
    args = (z[None, :], np.zeros((1, n)), Q, np.zeros((1, n)), code, gp, 0.0, rt['stack'], rt['receivers'], None,
            None, rt['lengths'], 1.5, (1, n))
    icy = np.flatnonzero(s > 0)
    H_on = run_modeA_step(*args, toe=(Q_raw, phi, None))[1].ravel()
    H_off = run_modeA_step(*args)[1].ravel()
    up = (z[icy] - z[icy - 1]) / 250.0
    for i, S_on, S_off in zip(icy, exact[icy], up):
        assert H_on[i] == pytest.approx(_H_from_QS_modeA(code, gp, Q[0, i], S_on), rel=1e-9)
        assert H_off[i] == pytest.approx(_H_from_QS_modeA(code, gp, Q[0, i], S_off), rel=1e-12)


def test_flux_ratio_factor_limits():
    """c(q) is siim1d's c(u) at q = (u - 1)/u, tends to 1 at q -> 1 and to 1 - phi at q -> 0+, and is 1 at q >= 1
    (accumulation zone, confluences, heads)."""
    phi = 1.0 / 3.0
    u = np.array([1.2, 2.0, 5.0, 40.0])
    c_u = (1 - phi) * u ** -phi / (u ** (1 - phi) - (u - 1) ** (1 - phi))
    q = np.r_[(u - 1) / u, 1 - 1e-12, 1e-300, 1.0, 3.0]
    n = q.size
    rec = np.r_[np.arange(n) + n, np.arange(n) + n]                 # node i drains to node i + n, a border
    Q = np.r_[np.ones(n), q]
    c = toe_slope_factor(np.zeros(2 * n), Q, rec, None, None, np.ones(2 * n), phi, np.r_[np.arange(n) + n,
                                                                                         np.arange(n)])[:n]
    np.testing.assert_allclose(c[:4], c_u, rtol=1e-12)
    assert c[4] == pytest.approx(1.0, abs=1e-11)
    assert c[5] == pytest.approx(1 - phi, rel=1e-12)
    assert c[6] == 1.0 and c[7] == 1.0


@pytest.mark.parametrize('layout', ['sfr', 'kary'])
def test_last_node_term_needs_min_cells(layout):
    """The last-node term applies where at least constants.TOE_MIN_CELLS icy cells route through the node (the node
    included) and is the uncorrected step (c = 1) one cell short; the interior c(q) is kept either way."""
    phi = 0.5
    m = constants.TOE_MIN_CELLS
    out = {}
    for n_icy in (m, m - 1):
        j = 20
        rt, z, Q_raw, s, _ = _chain(phi, 0.3, layout, n=j + n_icy, j=j)
        assert (s > 0).sum() == n_icy
        out[n_icy] = (_factor(z, Q_raw, rt, phi).reshape(z.size, -1)[:, 0], j)
    (at, j), (short, _) = out[m], out[m - 1]
    assert at[j] != 1.0 and short[j] == 1.0
    np.testing.assert_array_equal(short[j + 1:], at[j + 1:m + j - 1])    # the interior nodes, same q
    assert np.all(short[j + 1:] < 1.0)


def test_qstar_is_the_plain_form_on_single_flow():
    """On a single-flow tree (confluences included) Q* = min(Q_i, I_r) is Q_i exactly (I_r sums Q_i with other
    non-negative terms), so the factor is the plain q = Q_r/Q_i form bit for bit: the same kernel with I_r = inf."""
    rng = np.random.default_rng(7)
    n = 300
    rec = np.r_[0, rng.integers(0, np.arange(1, n))]                # node i > 0 drains to a node below it
    stack = np.arange(n)                                            # receiver-first
    Q = rng.uniform(-0.5, 1.0, n) * 10.0 ** rng.uniform(-2, 3, n)
    z = rng.uniform(0.0, 50.0, n) + np.arange(n)
    phi = 1.0 / 3.0
    lens = rng.uniform(50.0, 150.0, n)
    c = toe_slope_factor(z, Q, rec, None, None, lens, phi, stack)
    R, W, L = rec.reshape(-1, 1), np.ones((n, 1)), lens.reshape(-1, 1)
    nb = np.ones(n, np.int64)
    inflow, ncell = np.empty(n), np.empty(n)
    _toe_inflow_cells_kernel(Q, R, nb, W, stack[::-1].copy(), inflow, ncell)
    icy = (Q > 0) & (rec != np.arange(n))
    assert np.all(inflow[rec[icy]] >= Q[icy])
    plain = np.ones((n, 1))
    _toe_slope_factor_kernel(z, Q, R, nb, W, L, phi, constants.TOE_ZETA_MIN, float(constants.TOE_MIN_CELLS),
                             np.full(n, np.inf), ncell, plain)
    assert np.array_equal(c, plain.ravel())
    assert ((c != 1.0) & (Q[rec] > 0)).sum() > 20                  # interior edges exercised
    assert ((c != 1.0) & (Q[rec] <= 0)).sum() > 5                  # and last-node edges


def _split(Q2_source=-500.0, w2=0.1):
    """Node 0 (fed by an icy line 6 -> 5 -> 4 -> 3 -> 0, so 5 icy cells route through it) splits its 100 m^3/yr
    1 - w2 to icy node 1 (a border, its own source -10) and w2 to node 2 (a border, its own source Q2_source)."""
    rec = np.array([[1, 2], [1, 1], [2, 2], [0, 0], [3, 3], [4, 4], [5, 5]])
    w = np.array([[1 - w2, w2]] + [[1.0, 0.0]] * 6)
    Q = np.array([100.0, (1 - w2) * 100.0 - 10.0, w2 * 100.0 + Q2_source, 80.0, 60.0, 40.0, 20.0])
    z = np.array([10.0, 0.0, 0.0, 11.0, 12.0, 13.0, 14.0])
    return z, Q, dict(receivers=rec, nb_receivers=np.array([2, 1, 1, 1, 1, 1, 1]), weights=w,
                      lengths=np.ones((7, 2)), stack=np.array([6, 5, 4, 3, 0, 1, 2]))


def test_qstar_reads_the_multi_receiver_split():
    """Where a multi-receiver node sends a receiver part of its flux, q and zeta read what the receiver gets:
    Q* = I_r = 90 on the icy edge and 10 on the bare one."""
    phi = 0.5
    z, Q, rt = _split()
    c = _factor(z, Q, rt, phi)
    assert c[0, 0] == pytest.approx(_cf(Q[1] / 90.0, phi), rel=1e-12)
    zeta = 10.0 / (10.0 - Q[2])                         # node 2 has no downhill slope: R = 1
    assert c[0, 1] == pytest.approx((1 - phi) / zeta, rel=1e-12)
    assert np.all(c[3:] == 1.0)                         # the feeding line: q >= 1


def test_edge_with_no_inflow_is_skipped():
    """An edge whose receiver gets nothing (weight 0, so Q* = I_r = 0) keeps c = 1."""
    z, Q, rt = _split(Q2_source=-5.0, w2=0.0)
    c = _factor(z, Q, rt, 0.5)
    assert c[0, 1] == 1.0 and c[0, 0] != 1.0


def test_last_node_at_or_below_the_extended_toe_takes_the_uncorrected_step():
    """An icy node at or below the bare reach extended up to its terminus (R <= 0) takes the uncorrected step,
    c = 1, never c = 0, as siim1d; the rest of the line is unchanged."""
    phi = 0.5
    rt, z, Q_raw, s, _ = _chain(phi, 0.3, 'sfr')
    j = np.flatnonzero(s > 0)[0]
    ref = _factor(z, Q_raw, rt, phi)
    for frac in (0.5, 0.7):                             # R < 0, R = 0: z_j at or below z_r + (1 - zeta) L S_f
        z2 = z.copy()
        z2[j] = z2[j - 1] + frac * 0.07 * 250.0
        c = _factor(z2, Q_raw, rt, phi)
        assert c[j] == 1.0
        np.testing.assert_array_equal(np.delete(c, j), np.delete(ref, j))


@pytest.mark.pin
@pytest.mark.parametrize('layout', ['sfr', 'kary'])
def test_corrected_slope_no_steeper_than_the_bare_one_takes_the_uncorrected_step(layout):
    """A node barely clearing the bare reach extended up to its terminus (0 < R << 1) would read a corrected slope
    near zero, which spiked H for one step (13.6 km on the 1 km A-d8 Hack run, c 3.9e-5 against S_r 0.19). Where
    the corrected slope is no steeper than the receiver's bare slope S_r, the edge takes the uncorrected step, c = 1,
    so mode A's H there is the closure at the upwind slope; clearing it by more keeps the correction."""
    phi, zeta, L, S_f = 0.5, 0.3, 250.0, 0.07
    rt, z, Q_raw, s, _ = _chain(phi, zeta, layout)
    j = np.flatnonzero(s > 0)[0]
    ref = _factor(z, Q_raw, rt, phi).reshape(z.size, -1)[:, 0]
    z_toe = z[j - 1] + (1 - zeta) * L * S_f
    lim = zeta * L * S_f / (1 - phi)                    # the clearance at which the corrected slope equals S_f
    for above, kept in ((1e-4, False), (0.999 * lim, False), (2 * lim, True)):
        z2 = z.copy()
        z2[j] = z_toe + above
        c = _factor(z2, Q_raw, rt, phi).reshape(z.size, -1)[:, 0]
        assert (c[j] != 1.0) == kept
        if kept:
            assert c[j] * (z2[j] - z2[j - 1]) / L > S_f
        np.testing.assert_array_equal(np.delete(c, j), np.delete(ref, j))
    if layout == 'sfr':
        code, gp = _law('coulomb')
        z2 = z.copy()
        z2[j] = z_toe + 1e-4
        n = z.size
        Q = np.maximum(Q_raw, 0.0)[None, :]
        args = (z2[None, :], np.zeros((1, n)), Q, np.zeros((1, n)), code, gp, 0.0, rt['stack'], rt['receivers'],
                None, None, rt['lengths'], 1.5, (1, n))
        H_on = run_modeA_step(*args, toe=(Q_raw, phi, None))[1].ravel()
        H_off = run_modeA_step(*args)[1].ravel()
        assert H_on[j] == H_off[j] > 0.0


@pytest.mark.parametrize('layout', ['sfr', 'kary'])
def test_exact_toe_gentler_than_the_ramp_below_takes_the_uncorrected_step(layout):
    """The exact toe at zeta 0.9 above a ramp steeper than its last node's slope (0.06 against 0.07): the last icy
    node takes the uncorrected step, the case the exactness test's gentler ramp (0.05) keeps out."""
    phi, S_f = 0.5, 0.07
    rt, z, Q_raw, s, exact = _chain(phi, 0.9, layout, S_f=S_f)
    j = np.flatnonzero(s > 0)[0]
    assert exact[j] < S_f
    c = _factor(z, Q_raw, rt, phi).reshape(z.size, -1)[:, 0]
    assert c[j] == 1.0


@pytest.mark.parametrize('layout', ['sfr', 'kary'])
def test_no_extension_beside_a_drowned_border(layout):
    """A terminus two cells from the outlet, its bare receiver draining straight to a drowned border (the bed far
    below), takes the toe at the receiver's level, c = (1 - phi)/zeta, as siim1d beside an outlet; one cell further
    up, the receiver's downhill slope extends it (c differs)."""
    phi, zeta = 0.5, 0.3
    for j, extended in ((2, False), (3, True)):
        rt, z, Q_raw, s, _ = _chain(phi, zeta, layout, j=j)
        z = z.copy()
        z[0] -= 200.0                                   # the drowned border
        c = _factor(z, Q_raw, rt, phi).reshape(z.size, -1)[:, 0]
        ze = 1.0 / (1.0 - Q_raw[j - 1] / Q_raw[j])
        assert (c[j] != pytest.approx((1 - phi) / ze, rel=1e-12)) == extended


def test_ice_free_step_returns_none_and_scratch_is_reused():
    """No icy node: the factor is None (the off path). A scratch dict keeps its work arrays between calls and gives
    the same factor as a fresh call."""
    phi = 0.5
    rt, z, Q_raw, s, _ = _chain(phi, 0.3, 'sfr')
    args = (rt['receivers'], rt['nb_receivers'], rt['weights'], rt['lengths'], phi, rt['stack'])
    assert toe_slope_factor(z, -np.abs(Q_raw), *args) is None
    scratch = {}
    first = toe_slope_factor(z, Q_raw, *args, scratch).copy()
    held = scratch['c']
    assert np.array_equal(toe_slope_factor(z, Q_raw, *args, scratch), first) and scratch['c'] is held
    assert np.array_equal(toe_slope_factor(z, Q_raw, *args), first)


def test_zeta_floor():
    """zeta = 1/(1 - Q_r/Q*) is floored at constants.TOE_ZETA_MIN, so a vanishing residual flux gives a finite
    factor, (1 - phi) R/zeta_min with R built on the floored zeta; above the floor zeta is the plain form."""
    phi = 0.5
    zmin = constants.TOE_ZETA_MIN
    for zeta in (1e-9, 1e-3):
        rt, z, Q_raw, s, _ = _chain(phi, zeta, 'sfr')
        j = np.flatnonzero(s > 0)[0]
        c = _factor(z, Q_raw, rt, phi)
        ze = max(1.0 / (1.0 - Q_raw[j - 1] / Q_raw[j]), zmin)
        rise = z[j] - z[j - 1]
        R = 1.0 - (1.0 - ze) * (z[j - 1] - z[j - 2]) / rise
        assert np.isfinite(c[j]) and c[j] == pytest.approx((1 - phi) / ze * R, rel=1e-9)
        assert (ze == zmin) == (zeta < zmin)


def test_accumulator_raw_out_is_the_preclamp_flux():
    """``raw_out`` is the ice flux before the clamp (its positive part is the returned flux) and changes nothing."""
    rng = np.random.default_rng(3)
    ny, nx = 6, 9
    shape, n = (ny, nx), ny * nx
    surface = 2000.0 + 500.0 * rng.random(shape)
    rec = np.r_[np.arange(nx), np.arange(nx, n) - nx]               # each row drains to the row below
    args = (surface, 0.0, 2200.0, 1e-2, 1.0, 1e4, 0.5, 0.5, shape, np.arange(n), rec, None, None,
            np.full(n, 100.0), None)
    ref = accumulate_glacial_flow(*args)
    raw = np.empty(n)
    out = accumulate_glacial_flow(*args, raw_out=raw)
    for a, b in zip(ref[:3], out[:3]):
        assert np.array_equal(a, b)
    assert np.array_equal(np.maximum(raw, 0.0).reshape(shape), ref[0])
    assert (raw < 0).any()


@pytest.mark.pin
def test_option_off_is_the_default_model():
    """toe_slope_correction_2d=False passed explicitly is the default model, bit for bit, and True changes it (mode C
    on MFD, mode A on D8, mode B on D-inf, glaciated)."""
    base = dict(nx=21, ny=17, Lx=20e3, Ly=16e3, T=40e3, nt=41, nt_out=3, seed=5, zELA=900.0,
                initial_max_elevation=2000.0, progress_bar=False)
    assert constants.TOE_SLOPE_CORRECTION_2D is False
    for extra in (dict(mode='C', flow_routing='mfd'), dict(mode='A', flow_routing='single'),
                  dict(mode='B', flow_routing='dinf')):
        runs = []
        for opts in ({}, dict(toe_slope_correction_2d=False), dict(toe_slope_correction_2d=True)):
            m = siim2d(base | extra | opts)
            m.run()
            runs.append(m.ds_out)
        assert float(runs[0]['glacial_spl__ice_thickness'].max()) > 10.0
        for k in runs[0].data_vars:
            if 'basin' not in k:
                assert np.array_equal(runs[0][k].values, runs[1][k].values), k
        H = 'glacial_spl__ice_thickness'
        assert not np.array_equal(runs[0][H].values, runs[2][H].values), extra


def test_option_validated_and_refused_on_the_xsimlab_driver():
    cfg = dict(nx=11, ny=11, nt=3, nt_out=2, T=2e3, progress_bar=False)
    with pytest.raises(ValueError, match='True or False'):
        siim2d(cfg | dict(toe_slope_correction_2d='yes'))
    # an effective exponent with mu/nu >= 1 has no finite toe factor (phi_t = 1 divides 0/0)
    for mu in (2.0, 3.0):
        with pytest.raises(ValueError, match='phi_t < 1'):
            siim2d(cfg | dict(toe_slope_correction_2d=True, sliding_law='eff-exp', nu=2.0, mu=mu))
    m = siim2d(cfg | dict(toe_slope_correction_2d=True))
    with pytest.raises(NotImplementedError, match='in-house-driver'):
        m.run(driver='xsimlab')
