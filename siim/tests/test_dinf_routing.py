"""D-inf depression handling: eps-filled routing surface.

Directions come from the priority-flood-eps surface (every interior cell
strictly drains, closed basins route across their spills — SFR's
depression semantics), while the physics keeps consuming the true
surface. These tests drive the routing primitives directly on synthetic
terrain: fill correctness, mass conservation across an interior pit
(the failure mode that motivated the redesign — pit flux used to be
dropped), spill-directed exit, and looped-seam wrap invariance.
"""
import numpy as np
import pytest

from siim._core.routing import (
    _priority_flood_eps, _dinf_route, _dinf_topo_stack, _dinf_pack,
    _flow_accumulate_dinf,
    _DINF_E1_DJ, _DINF_E1_DI, _DINF_E2_DJ, _DINF_E2_DI)

EPS = 1e-6


def _edges_interior(ny, nx):
    interior = np.ones(ny * nx, dtype=np.int8)
    interior[:nx] = 0
    interior[-nx:] = 0
    interior[::nx] = 0
    interior[nx - 1::nx] = 0
    return interior


def _rows_interior(ny, nx):
    """Only top/bottom rows are outlets (x-looped configuration)."""
    interior = np.ones(ny * nx, dtype=np.int8)
    interior[:nx] = 0
    interior[-nx:] = 0
    return interior


def _route_full(z, dx, dy, interior, wrap_y=False, wrap_x=False):
    """fill -> facets -> pack -> stack, as DinfFlowRouter.run_step does."""
    ny, nx = z.shape
    nn = ny * nx
    z_flat = z.ravel().astype(np.float64)
    z_route = np.empty(nn)
    _priority_flood_eps(z_flat, ny, nx, interior, EPS, wrap_y, wrap_x,
                        z_route)
    r1 = np.zeros(nn, np.int64); r2 = np.zeros(nn, np.int64)
    w1 = np.zeros(nn); w2 = np.zeros(nn)
    l1 = np.zeros(nn); l2 = np.zeros(nn); s = np.zeros(nn)
    _dinf_route(z_route, ny, nx, dx, dy, interior,
                r1, r2, w1, w2, l1, l2, s,
                _DINF_E1_DJ, _DINF_E1_DI, _DINF_E2_DJ, _DINF_E2_DI,
                wrap_y, wrap_x)
    receivers = np.zeros((nn, 2), np.int64)
    weights = np.zeros((nn, 2))
    lengths = np.zeros((nn, 2))
    nb_receivers = np.zeros(nn, np.int64)
    _dinf_pack(r1, r2, w1, w2, l1, l2, nn,
               receivers, weights, lengths, nb_receivers)
    stack_rf = np.zeros(nn, np.int64)
    _dinf_topo_stack(r1, r2, w1, w2, nn, stack_rf)
    stack = stack_rf[::-1].copy()
    return z_route, receivers, weights, lengths, nb_receivers, stack


def _pitted_plane(ny=15, nx=21, dx=100.0):
    """Plane draining +x toward the left edge with a closed basin dug in."""
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    z = 10.0 + 0.01 * (ii * dx)
    z[6:10, 8:13] -= 25.0          # closed basin, floor ~ -3 to -3+
    return z


def test_priority_flood_eps_fill():
    """Fill leaves unponded terrain untouched, raises basin cells to just
    above the spill with strictly positive drainage everywhere: every
    interior cell has a strictly lower 8-neighbour on the filled surface."""
    ny, nx, dx = 15, 21, 100.0
    z = _pitted_plane(ny, nx, dx)
    interior = _edges_interior(ny, nx)
    z_flat = z.ravel()
    z_fill = np.empty(ny * nx)
    _priority_flood_eps(z_flat, ny, nx, interior, EPS, False, False, z_fill)
    assert (z_fill >= z_flat - 1e-12).all()
    pit = np.zeros((ny, nx), dtype=bool)
    pit[6:10, 8:13] = True
    # untouched outside the basin
    np.testing.assert_allclose(z_fill.reshape(ny, nx)[~pit], z[~pit])
    # basin filled to ~ its spill (the lowest rim cell, at the -x rim)
    rim_min = z[6:10, 7].min()
    zf_pit = z_fill.reshape(ny, nx)[pit]
    assert (zf_pit >= rim_min).all()
    assert zf_pit.max() < rim_min + 100 * EPS      # eps-scale, not metres
    # strict drainage: every interior cell has a strictly lower 8-neighbour
    zf = z_fill.reshape(ny, nx)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            nbrs = [zf[j + dj, i + di]
                    for dj in (-1, 0, 1) for di in (-1, 0, 1)
                    if (dj, di) != (0, 0)]
            assert min(nbrs) < zf[j, i], (j, i)


def test_dinf_conservation_across_pit():
    """Unit input per cell: ALL of it must reach the boundary — flux
    entering the closed basin crosses the spill instead of vanishing
    (the old router dropped it: interior pits self-received)."""
    ny, nx, dx = 15, 21, 100.0
    z = _pitted_plane(ny, nx, dx)
    interior = _edges_interior(ny, nx)
    _, receivers, weights, lengths, nb_receivers, stack = _route_full(
        z, dx, dx, interior)
    # no interior cell may be receiver-less anymore
    assert (nb_receivers[interior == 1] >= 1).all()
    field = np.ones(ny * nx)
    _flow_accumulate_dinf(field, stack, nb_receivers, receivers, weights)
    boundary = interior == 0
    total_in = float(ny * nx)
    total_out = float(field[boundary].sum())
    np.testing.assert_allclose(total_out, total_in, rtol=1e-12)


def test_dinf_pit_source_exits_at_spill_side():
    """A point source at the basin floor exits the domain (full mass) and
    leaves via the downhill (-x) side the spill points to."""
    ny, nx, dx = 15, 21, 100.0
    z = _pitted_plane(ny, nx, dx)
    interior = _edges_interior(ny, nx)
    _, receivers, weights, lengths, nb_receivers, stack = _route_full(
        z, dx, dx, interior)
    field = np.zeros(ny * nx)
    field[8 * nx + 10] = 1.0                      # basin floor
    _flow_accumulate_dinf(field, stack, nb_receivers, receivers, weights)
    boundary = interior == 0
    np.testing.assert_allclose(field[boundary].sum(), 1.0, rtol=1e-12)
    # the plane drains -x; everything must leave through the left edge
    left = np.zeros((ny, nx), dtype=bool)
    left[:, 0] = True
    np.testing.assert_allclose(field[left.ravel()].sum(), 1.0, rtol=1e-12)


@pytest.mark.parametrize("law", ["eff-exp", "power", "coulomb"])
def test_mode_b_dinf_sfr_parity_cardinal_plane(law):
    """On terrain draining along a single cardinal direction, D-inf
    degenerates to one receiver with weight 1, so the D-inf mode-B kernel
    must reproduce the SFR mode-B kernel: same joint-walk H, same bed
    after erosion + border budget, same presented surface. Pins the
    weighted-effective-receiver reduction (S = (zs - Z̄)/L̄), the
    priority-flood erosion view against the SFR stack-walk fill, and the
    border closure/cap/floor parity."""
    from ._kernel_adapters import (
        glac_fast_solve_modeB, glac_fast_solve_power_modeB,
        glac_fast_solve_coulomb_modeB,
        glac_fast_solve_modeB_dinf, glac_fast_solve_power_modeB_dinf,
        glac_fast_solve_coulomb_modeB_dinf)
    ny, nx, dx = 7, 20, 100.0
    nn = ny * nx
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    zb0 = 5.0 + 0.01 * dx * ii.astype(float)      # drains -x to col 0
    interior = _edges_interior(ny, nx)

    ice = np.full(nn, 1e7)
    water = np.zeros(nn)
    bbu = np.zeros(nn)
    cg, lam_p, lam_c, tau_c, rho_g_g, clamp = (2.3e-4, 300.0, 1e-3, 1.2e5,
                                               9016.0, 1e-12)
    Ko, Co, ce = 1e-6, 1e-9, 1e-5
    n_exp, nu, m, mu = 1.0, 1.0, 0.5, 4.0 / 15.0
    dt, hc, alpha_g = 500.0, 1.5, 8.0

    # --- SFR graph: interiors flow -x, borders self; stack col-ascending ---
    rec_sfr = np.arange(nn, dtype=np.int64)
    lengths_sfr = np.zeros(nn)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            rec_sfr[j * nx + i] = j * nx + (i - 1)
            lengths_sfr[j * nx + i] = dx
    stack_sfr = np.argsort(ii.ravel(), kind='stable').astype(np.int64)

    zb_s = zb0.ravel().copy()
    H_s = np.zeros(nn)
    surf_s = np.empty(nn)
    if law == 'eff-exp':
        glac_fast_solve_modeB(zb_s, ice, water, H_s, surf_s,
                              Ko, Co, n_exp, nu, m, mu, cg, alpha_g, lam_p,
                              dt, lengths_sfr, stack_sfr, rec_sfr,
                              0.0, ny, nx, dx, dx, bbu, hc)
    elif law == 'power':
        glac_fast_solve_power_modeB(zb_s, ice, water, H_s, surf_s,
                                    Ko, ce, n_exp, nu, m, cg, alpha_g, lam_p,
                                    dt, lengths_sfr, stack_sfr, rec_sfr,
                                    0.0, ny, nx, dx, dx, bbu, hc)
    else:
        glac_fast_solve_coulomb_modeB(zb_s, ice, water, H_s, surf_s,
                                      Ko, ce, n_exp, nu, m, cg, alpha_g,
                                      lam_c, tau_c, clamp, rho_g_g,
                                      dt, lengths_sfr, stack_sfr, rec_sfr,
                                      0.0, ny, nx, dx, dx, bbu, hc)

    # --- D-inf graph from the real router primitives on the same plane ---
    _, receivers, weights, lengths, nb_receivers, stack = _route_full(
        zb0, dx, dx, interior)
    # cardinal descent: every interior cell routes to one west receiver
    inner = (interior == 1)
    assert (nb_receivers[inner] == 1).all()
    assert (weights[inner, 0] == 1.0).all()

    zb_d = zb0.ravel().copy()
    H_d = np.zeros(nn)
    surf_d = np.empty(nn)
    if law == 'eff-exp':
        glac_fast_solve_modeB_dinf(zb_d, ice, water, H_d, surf_d,
                                   Ko, Co, n_exp, nu, m, mu,
                                   cg, alpha_g, lam_p,
                                   dt, stack, nb_receivers, receivers,
                                   weights, lengths,
                                   0.0, ny, nx, dx, dx, bbu, hc,
                                   False, False)
    elif law == 'power':
        glac_fast_solve_power_modeB_dinf(zb_d, ice, water, H_d, surf_d,
                                         Ko, ce, n_exp, nu, m,
                                         cg, alpha_g, lam_p,
                                         dt, stack, nb_receivers, receivers,
                                         weights, lengths,
                                         0.0, ny, nx, dx, dx, bbu, hc,
                                         False, False)
    else:
        glac_fast_solve_coulomb_modeB_dinf(zb_d, ice, water, H_d, surf_d,
                                           Ko, ce, n_exp, nu, m,
                                           cg, alpha_g, lam_c, tau_c, clamp,
                                           rho_g_g,
                                           dt, stack, nb_receivers, receivers,
                                           weights, lengths,
                                           0.0, ny, nx, dx, dx, bbu, hc,
                                           False, False)

    assert H_d.max() > 1.0, "config must grow ice"
    np.testing.assert_allclose(H_d, H_s, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(zb_d, zb_s, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(surf_d, surf_s, rtol=1e-10, atol=1e-10)


def test_mode_b_dinf_integration_with_carve():
    """End-to-end: flow_routing='dinf' + mode='B' runs finite for all three
    sliding laws, and the width carve attaches (bed pulled below the
    no-carve twin somewhere)."""
    from siim.siim2d import siim as siim2d
    base = {
        'U': 1e-3, 'zELA': 150, 'beta': 1e-2, 'P': 2, 'alpha_g': 12,
        'Ko': 2e-6, 'n': 1, 'ce': 1e-4, 'nu': 2,
        'sliding_law': 'power', 'lambda_p': 500, 'k': .9,
        'T': 6e4, 'nt': 61, 'nt_out': 13,
        'Lx': 2e4, 'Ly': 2e4, 'nx': 41, 'ny': 41, 'seed': 7,
        'boundary_status': ['fixed_value'] * 4,
        'initial_max_elevation': 800,
        'mode': 'B', 'flow_routing': 'dinf', 'progress_bar': False,
        # Isolate the carve: pin the mode-C standard flags off so the carve-on /
        # carve-off twins differ ONLY in carve_width (widening_rate is set on the
        # carve-on side below).
        'trunk_surface': False, 'routing_relax': 0.0,
    }
    for law, extra in (('eff-exp', {}), ('coulomb',
                                         {'lambda_c': 1e3, 'tau_c': 1.2e5})):
        m = siim2d({**base, 'sliding_law': law, **extra})
        m.run()
        assert np.isfinite(m.z_out).all() and np.isfinite(m.H_out).all()
        assert m.H_out[-1].max() > 0, law

    m1 = siim2d({**base, 'carve_width': True, 'widening_rate': 0.0})
    m1.run()
    assert np.isfinite(m1.z_out).all() and np.isfinite(m1.zb_out).all()
    m0 = siim2d({**base, 'carve_width': False})  # no-carve twin (carving is now default-on)
    m0.run()
    dzb = m1.zb_out[-1] - m0.zb_out[-1]
    assert dzb.min() < -1.0, "carve must deepen the bed under D-inf too"


def test_dinf_looped_x_shift_invariance():
    """x-looped: fill + routing + accumulation commute with a circular
    shift in x — the seam is invisible. Terrain drains +y to the top/
    bottom outlet rows with an x-asymmetric ridge and a basin, so flow
    genuinely crosses the seam in one layout."""
    rng = np.random.default_rng(3)
    ny, nx, dx = 17, 24, 100.0
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    z = (5.0 + 0.012 * dx * np.minimum(jj, ny - 1 - jj)
         + 3.0 * np.sin(2 * np.pi * ii / nx)
         + 0.3 * rng.random((ny, nx)))
    z[7:10, 2:6] -= 12.0                          # basin near the seam side
    interior = _rows_interior(ny, nx)

    def accumulate(zz):
        _, receivers, weights, lengths, nb_receivers, stack = _route_full(
            zz, dx, dx, interior, wrap_y=False, wrap_x=True)
        field = np.ones(ny * nx)
        _flow_accumulate_dinf(field, stack, nb_receivers, receivers, weights)
        return field.reshape(ny, nx)

    f0 = accumulate(z)
    k = 11
    fk = accumulate(np.roll(z, k, axis=1))
    np.testing.assert_allclose(np.roll(f0, k, axis=1), fk,
                               rtol=1e-9, atol=1e-9)
    # and conservation holds with the seam crossing
    boundary2d = np.zeros((ny, nx), dtype=bool)
    boundary2d[0, :] = boundary2d[-1, :] = True
    np.testing.assert_allclose(f0[boundary2d].sum(), float(ny * nx),
                               rtol=1e-12)


# --- P0 regression: D-inf fluvial erosion Newton must stay bracketed ----------
# The coulomb D-inf eroder's fluvial fallback used to be an unbracketed Newton
# (a bare `zik -= F/dF` loop). For sub-linear fluvial n<1 it overshoots below the
# receiver, 2-cycles, and silently returns zo (zero erosion) — the failure the
# bracket exists to prevent (CLAUDE.md: "Erosion Newtons are bracketed").
# All three *_erode_2d_dinf eroders now share the bracketed _solver_nonlinear_dinf.

def test_solver_nonlinear_dinf_brackets_sublinear():
    """The shared D-inf solver keeps a bracket, so it converges for n<1 where a
    plain Newton overshoots below the receiver, 2-cycles, and ends at zo."""
    from siim._core.solvers import _solver_nonlinear_dinf
    zo_i, zr = 20.0, 10.0
    receivers_i = np.array([1], dtype=np.int64)
    weights_i = np.array([1.0])
    lengths_i = np.array([1.0])
    z_flat = np.array([0.0, zr])
    A, p, dt = 100.0, 0.5, 1.0

    z_new = _solver_nonlinear_dinf(zo_i, 1, receivers_i, weights_i, lengths_i,
                                   z_flat, A, p, dt)
    # eroded down toward the receiver, no overshoot below it
    assert zr - 1e-9 <= z_new < zo_i - 9.0

    # same residual, plain (unbracketed) Newton: overshoots, 2-cycles, ends at zo
    zik = zo_i
    C = A * dt / lengths_i[0] ** p
    for _ in range(50):
        dz = zik - zr
        F = zik - zo_i + (C * dz ** p if dz > 0 else 0.0)
        dF = 1.0 + (p * C * dz ** (p - 1) if dz > 0 else 0.0)
        zik -= F / dF
    assert abs(zik - zo_i) < 1e-6, "plain Newton should be stuck at zo (the bug)"


def test_coulomb_dinf_fluvial_matches_effexp_sublinear():
    """On pure fluvial input (Qg=0) with sub-linear n<1, the coulomb D-inf eroder
    must reproduce the eff-exp D-inf eroder bit-for-bit (they share the bracketed
    solver) AND actually erode — pre-fix the coulomb fallback 2-cycled to ~zo."""
    from siim._core.eroders import (
        _nonlinear_erode_2d_dinf, _coulomb_erode_2d_dinf)
    ny, nx, dx = 7, 12, 100.0
    nn = ny * nx
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    z0 = 5.0 + 0.05 * dx * ii.astype(float)       # drains -x, ~5 m drop per cell
    interior = _edges_interior(ny, nx)
    _, receivers, weights, lengths, nb_receivers, stack = _route_full(
        z0, dx, dx, interior)

    zo = z0.ravel().copy()
    Qf = np.full(nn, 1e7)                          # large flux -> large step
    Qg = np.zeros(nn)                              # pure fluvial: glacial inert
    n = 0.7                                        # sub-linear: 2-cycle regime
    Ko, Co, m, mu, nu = 1e-5, 1e-9, 1.0, 4.0 / 15.0, 1.0
    ce, ell, t = 1e-5, 1.0, 1.5
    cg, rho_g_g, tau_c, lam_c, clamp, alpha_g = (2.3e-4, 9016.0, 1.2e5, 1e-3,
                                                 1e-12, 8.0)
    dt = 1000.0

    z_ee = zo.copy()
    _nonlinear_erode_2d_dinf(z_ee, zo, Qf, Qg, dt, Ko, Co, m, mu, n, nu,
                             stack, nb_receivers, receivers, weights, lengths)
    z_co = zo.copy()
    _coulomb_erode_2d_dinf(z_co, zo, Qf, Qg, dt, Ko, ce, m, n, ell, t,
                           cg, rho_g_g, tau_c, lam_c, clamp, alpha_g,
                           stack, nb_receivers, receivers, weights, lengths)

    inner = interior == 1
    assert (zo[inner] - z_co[inner]).max() > 1.0, "coulomb fluvial must erode"
    np.testing.assert_array_equal(z_co, z_ee)
