"""Outflow base-level ice boundary-condition regression tests.

The domain edge is an arbitrary cut through a continuing glacier: a through-flowing
border gets zero-gradient thickness (H_border = H_dominant_donor) and its bed keeps
ERODING by the IMPLICIT BORDER BUDGET — dzb/dt = U - f*E on the interior ARRIVAL
slope, f the flotation ramp, integrated by the closed-form backward-Euler step,
so the bed approaches the flotation-draft
equilibrium zb* = bl - hc*H + delta*U/E monotonically at any dt. The cheap suite
gates (the probe's 7-battery):

* **sill regression** — the headline guard: on the PI's failing config the mouth
  OPENS (border + donor descend below bl) and the trough keeps carving (the
  zerograd slave locked the pair at bl and killed erosion domain-wide).
* **dt flatness** — border envelope flat-to-shrinking across dt = 100 -> 50 000
  (the gate-era explicit budget went km/step).
* **draft equilibrium** — the deepest border excursion sits within the standoff
  of the flotation draft bl - hc*H.
* **unbounded control** — flotation_gate=False (f == 1) digs past a threshold:
  the ramp is the load-bearing bound, the implicit integrator only the numerics.
* **dome guard** — long-T power Hmax stays ~O(1000) (no drawdown re-arm).
* **cardinal-plane border parity** — zero-gradient H + implicit budget identical
  at the 1D / 2D-SFR / 2D-D-inf border sites (dominant-donor rule).
* **2D no-single-cell-pit** — border digging is a bounded trench spread over
  many face cells (the gate era dug one -9 km cell).
* **no chatter / no dam / true-state output / interior ramp gamma=0-is-binary**
  — carried forward.
"""
import numpy as np
import pytest

from siim.siim1d import siim as siim1d
from siim._core.params import GlacialParams
from siim._core.solvers import LAW_EFFEXP, LAW_POWER, LAW_COULOMB
from siim._core.skeleton import (_glac_fast_solve_modeB_sfr,
                                 _glac_fast_solve_modeB_dinf, _diag_walk,
                                 _implicit_border_step)

LAWS = ['eff-exp', 'power', 'coulomb']
HC = 1.5


def _pi_cfg(law='power', k=0.8, dx=100.0, T=2e5, dt=100.0, **kw):
    """Compact version of the PI's sill config (dx=100 instead of 50)."""
    p = dict(U=1e-3, zELA=200.0, beta=1e-2, P=1, Ko=1e-6, n=1, ce=5e-5, nu=2,
             L=3e4, dx=dx, xo=50, T=T, nt=int(T / dt) + 1, sliding_law=law,
             lambda_p=500.0, k=k, left_bc='base_level', right_bc='reflecting',
             mode='B', cap_ice_accumulation=False, floating_termini=True,
             progress_bar=False, nt_out=51)
    if law == 'coulomb':
        p['tau_c'] = 1.2e5
    p.update(kw)
    return p


# ---------------------------------------------------------------------------
# (i) Sill regression — the headline guard. The zerograd slave locked border
# AND donor at bl under thick ice (the donor's kernel erosion starves on the
# ~0 zero-gradient surface drop, min(zb_donor, bl) copies the stall), ponding
# 3+ km of ice behind a permanent mouth sill. The implicit budget digs the
# border, handing the donor a real receiver drop every step: the mouth opens.
# ---------------------------------------------------------------------------

def test_sill_opens_1d():
    """Compact PI config (power, zELA=200, cap=False, vigorous): border AND
    donor beds descend well below bl within the run (the mouth opens — the
    zerograd slave pinned both at exactly 0 for the whole run), the trough
    keeps carving at a rate comparable to the interior (rate beats uplift; the
    sill run LOST ground at +45 m/100 kyr), and H stays bounded (the sill
    ponded 3.6 km)."""
    m = siim1d(_pi_cfg())
    m.run()
    zb = m.zb_out
    assert zb[0].min() < -500.0, f"border did not open ({zb[0].min():.0f})"
    assert zb[1].min() < -500.0, f"donor did not open ({zb[1].min():.0f})"
    # trough still actively carving over the second half (uplift is +100 m/100kyr;
    # the sill run bled at +45; measured here: -75)
    trough = zb.min(axis=0)
    half = trough.size // 2
    rate = ((trough[-1] - trough[half])
            / (m.output_times[-1] - m.output_times[half]) * 1e5)
    assert rate < 0.0, f"trough stalled (rate {rate:+.1f} m/100 kyr)"
    assert m.H_out.max() < 2000.0             # no ponded dome (sill: 3607)


# ---------------------------------------------------------------------------
# (ii) dt flatness: the closed-form implicit step cannot overshoot, so the
# border envelope is flat-to-shrinking across two-and-a-half decades of dt
# (the gate-era explicit budget went -1774 m at dt=100 to -246 km at dt=50 kyr
# on this regime). (iii) Draft equilibrium: the deepest ICY border excursion
# sits within the standoff delta*U/E of the flotation draft bl - hc*H.
# ---------------------------------------------------------------------------

def test_border_dt_flatness_and_draft_1d():
    """Fixed T, dt swept 100 -> 50 000 yr (coulomb vigorous, cap=False): the
    border envelope stays draft-bounded and FLAT-TO-SHRINKING in dt (measured
    -449 / -148 / -72 m; probe: -320 / -121 / -51 on its dx=50 twin), and at
    the deepest icy excursion the bed sits AT or just above the flotation
    draft bl - hc*H (standoff < 100 m, never below draft - 1 m)."""
    env = {}
    for dtv in (100, 5000, 50000):
        m = siim1d(_pi_cfg(law='coulomb', k=0.65, zELA=300.0, T=1e6, dt=dtv))
        m.run()
        b, H = m.zb_out[0], m.H_out[0]
        env[dtv] = float(b.min())
        assert b.max() <= 1e-12                  # never above bl
        # draft check at the deepest ICY frame
        icy = H > 0.0
        if icy.any():
            k = int(np.argmin(np.where(icy, b, np.inf)))
            draft = -HC * H[k]
            assert b[k] >= draft - 1.0, (dtv, b[k], draft)
            assert b[k] - draft < 100.0, (dtv, b[k], draft)
    # flat-to-shrinking: coarser dt never digs deeper than 2x the finest
    assert env[5000] >= 2.0 * env[100], env      # (both negative)
    assert env[50000] >= 2.0 * env[100], env
    assert env[100] > -1000.0, env               # bounded everywhere


# ---------------------------------------------------------------------------
# (iv) Unbounded control: f == 1 (gate off) digs past a threshold — the ramp
# is the load-bearing bound; the implicit integrator is only the numerics.
# ---------------------------------------------------------------------------

def test_border_unbounded_without_gate_1d():
    """flotation_gate=False removes the ramp from the border budget (f == 1):
    the border digs far past the draft within a short run (measured -4661 m in
    200 kyr where the gated twin bottoms near -900) — documents that the bound
    is the flotation ramp, for diagnostics only."""
    m = siim1d(_pi_cfg(flotation_gate=False))
    m.run()
    assert m.zb_out[0].min() < -2000.0


# ---------------------------------------------------------------------------
# (v) Dome guard: long-T power Hmax stays ~O(1000) — the implicit budget does
# not re-arm the drawdown amplification (the extrap bed domed to 13 km; the
# donor-descent variant to ~5 km; the sill ponded 3.5 km).
# ---------------------------------------------------------------------------

def test_no_dome_long_run_1d():
    """Long-T power run (2.5 Myr, dt=500) on the PI config: Hmax stays ~1050
    (probe: 1044 at 5 Myr) — no drawdown re-arm, no ponding dome."""
    m = siim1d(_pi_cfg(T=2.5e6, dt=500.0))
    m.run()
    assert m.H_out.max() < 2000.0, f"dome? Hmax {m.H_out.max():.0f}"
    assert m.zb_out[0].min() > -2000.0           # border stays draft-bounded


# ---------------------------------------------------------------------------
# No dam (the starvation guard): the zero-gradient border passes ice out — icy
# duty stays ~1 on a vigorous through-flow config with bounded thickness (the
# probe's benign-starvation result: duty 1.00, Hmax normal, no runaway dome).
# ---------------------------------------------------------------------------

def test_no_dam_through_flow_1d():
    """Vigorous power through-flow (zELA=300): the slaved border does not starve
    or dam the glacier — the border stays icy essentially full-time (duty ~ 1,
    ice keeps exiting via the zero-gradient thickness) while H stays bounded and
    normal (probe: Hmax 941 on its config; a dam/dome would blow this up — the
    extrap-bed variant reached 13 km) and the beds stay bounded."""
    m = siim1d(dict(mode='B', sliding_law='power', zELA=300.0, U=1e-3,
                    left_bc='base_level', right_bc='reflecting',
                    progress_bar=False, T=1e6, nt=2000, nt_out=101,
                    L=3e4, nx=151))
    m.run()
    duty = float((m.H_out[0, :] > 0).mean())
    assert duty > 0.9, f"border starved (duty {duty:.2f})"
    assert m.H_out.max() < 2500.0, f"dome? Hmax {m.H_out.max():.0f}"
    assert m.zb_out.min() > -2e3


# ---------------------------------------------------------------------------
# No chatter: the border does not flip icy<->ice-free every step.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('law', ['coulomb', 'power'])
def test_no_chatter_1d(law):
    """A cycling border (zELA=500) either stays icy (through-flow), stays
    ice-free, or cycles slowly — it does not flip-flop every frame. Count the
    icy<->ice-free flips across the output frames and require it small."""
    m = siim1d(dict(mode='B', sliding_law=law, zELA=500.0, U=1e-3,
                    left_bc='base_level', right_bc='reflecting',
                    progress_bar=False, T=1e6, nt=1000, nt_out=101,
                    L=3e4, nx=151))
    m.run()
    icy = m.H_out[0, :] > 0
    flips = int(np.sum(icy[1:] != icy[:-1]))
    assert flips < 15, f"{law}: {flips} border flips over 101 frames (chatter?)"


# ---------------------------------------------------------------------------
# True-state output convention: z_out reports the border's actual zb + hc*H.
# ---------------------------------------------------------------------------

def test_true_state_output_convention_1d():
    """The public z_out reports the border's actual surface zb + hc*H, never a
    water-line presentation floor. A relict ice-free border whose bed sits below
    bl shows THROUGH below bl in the output (== zb_out there, H=0) — exactly like
    an interior trough."""
    m = siim1d(dict(U=1e-3, P=1.0, beta=1e-2, Ko=1e-5, n=1, m=0.5, nu=2,
                    ce=1e-5, sliding_law='eff-exp', zELA=1e5, L=5e4, nx=201,
                    T=1e6, nt=501, nt_out=6, mode='B',
                    left_bc='reflecting', right_bc='base_level',
                    progress_bar=False, bl=2000.0))
    m.run()
    out = -1                                                        # right outlet (last x)
    recon = m.zb_out[out, -1] + m.hc_over_H * m.H_out[out, -1]      # last time frame
    np.testing.assert_array_equal(m.z_out[out, -1], recon)         # true state, no floor
    assert m.z_out[out, -1] < 2000.0                               # relict bed shows below bl


# ---------------------------------------------------------------------------
# Cardinal-plane border parity: 1D walk vs 2D SFR vs 2D D-inf (dominant donor).
# ---------------------------------------------------------------------------

def _params(law):
    cg, lam_p, lam_c, tau_c, rho_g_g = 2.3e-4, 300.0, 1e-3, 1.2e5, 9016.0
    Ko, Co, ce = 1e-6, 1e-9, 1e-5
    n, nu, m, mu, hc, alpha_g, clamp = 1.0, 1.0, 0.5, 4.0 / 15.0, 1.5, 8.0, 1e-12
    if law == 'power':
        return LAW_POWER, GlacialParams(Ko, 0.0, ce, n, nu, m, 0.0, cg, alpha_g,
                                        lam_p, 0.0, 0.0, 0.0, 0.0, hc, 0.0)
    if law == 'coulomb':
        return LAW_COULOMB, GlacialParams(Ko, 0.0, ce, n, nu, m, 0.0, cg, alpha_g,
                                          0.0, lam_c, tau_c, clamp, rho_g_g, hc, 0.0)
    return LAW_EFFEXP, GlacialParams(Ko, Co, 0.0, n, nu, m, mu, cg, alpha_g,
                                     lam_p, 0.0, 0.0, 0.0, 0.0, hc, 0.0)


@pytest.mark.parametrize('law', LAWS)
def test_outflow_border_parity_sfr_dinf_1d(law):
    """The zero-gradient thickness + implicit border budget + flotation ramp
    are identical at the 1D-walk, 2D-SFR and 2D-D-inf border sites. On a
    cardinal plane draining -x to the col-0 border (each row an independent 1D
    chain), the border H and the committed beds agree bit-close (rtol 1e-10)
    across all three — the dominant-donor rule reproduces the 1D 2-pass result,
    and the border bed equals the closed-form implicit step computed
    independently in the test. Run at the default ramp (gamma = 0.1) with two
    waterlines: bl = 0 (everything deeply grounded, f = 1) and bl = 100
    (near-waterline cells: afloat / in-band f<1 / grounded, law-dependent —
    the ramp branch itself must agree SFR vs D-inf)."""
    from .test_dinf_routing import _route_full, _edges_interior
    ny, nx, dx = 7, 20, 100.0
    nn = ny * nx
    _, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    zb0 = 5.0 + 0.01 * dx * ii.astype(float)          # drains -x to col 0
    ice = np.full(nn, 1e7)
    water = np.zeros(nn)
    bbu = np.zeros(nn)
    dt = 500.0
    code, p = _params(law)

    rec_sfr = np.arange(nn, dtype=np.int64)
    lengths_sfr = np.zeros(nn)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            rec_sfr[j * nx + i] = j * nx + (i - 1)
            lengths_sfr[j * nx + i] = dx
    stack_sfr = np.argsort(ii.ravel(), kind='stable').astype(np.int64)
    interior = _edges_interior(ny, nx)
    _, receivers, weights, lengths, nb_rec, stack = _route_full(zb0, dx, dx, interior)

    # 1D walk on the mid row (same bed / flux / border; H is bl-independent)
    row = ny // 2
    zb_1 = zb0[row, :].copy(); H_1 = np.zeros(nx)
    Qg_1 = np.full(nx, 1e7)
    _diag_walk(zb_1, Qg_1, H_1, code, p, dx, nx - 1, nx, nx)

    b = row * nx                                     # mid-row col-0 border (has a donor)
    for bl in (0.0, 100.0):
        # 2D SFR
        zb_s = zb0.ravel().copy(); H_s = np.zeros(nn); surf_s = np.empty(nn)
        _glac_fast_solve_modeB_sfr(zb_s, ice, water, H_s, surf_s, code, p,
                                   dt, lengths_sfr, stack_sfr, rec_sfr,
                                   ny, nx, dx, dx, bbu, bl, True)
        # 2D D-inf
        zb_d = zb0.ravel().copy(); H_d = np.zeros(nn); surf_d = np.empty(nn)
        _glac_fast_solve_modeB_dinf(zb_d, ice, water, H_d, surf_d, code, p,
                                    dt, stack, nb_rec, receivers, weights,
                                    lengths, ny, nx, dx, dx, bbu, False, False,
                                    bl, True)
        # border H parity across the 1D-walk, SFR and D-inf sites.
        np.testing.assert_allclose(H_d[b], H_s[b], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(H_s[b], H_1[0], rtol=1e-10, atol=1e-10)
        assert H_s[b] > 0.0                          # icy border (zero-gradient fired)
        # committed beds (carrying the f-scaled interior erosion + the implicit
        # border budgets) agree everywhere.
        np.testing.assert_allclose(zb_d, zb_s, rtol=1e-10, atol=1e-10)
        assert zb_s[b] < zb0.ravel()[b] + 1e-12      # border eroded (budget live)


# ---------------------------------------------------------------------------
# The implicit budget at the kernel level (SFR; the D-inf twin is pinned
# bed-bit-close by the parity test): the committed border bed is EXACTLY the
# closed-form implicit step computed independently in the test, at small and
# huge dt alike (the step is monotone — no overshoot at any dt).
# ---------------------------------------------------------------------------

def test_border_implicit_budget_kernel_2d():
    """One SFR kernel call on a 1 x n strip: the committed border bed equals
    the closed-form implicit step on the arrival slope (dominant donor's own
    upstream surface slope, floored at S_FLOOR_BC), with E the per-law border
    rate at the post-step-5 beds — replicating the kernel's step-5b arithmetic
    independently. At dt = 50 kyr the bed lands bounded near the flotation
    draft (the gate-era explicit budget dug km/step here)."""
    from ._kernel_adapters import glac_fast_solve_modeB
    from siim._core.eroders import _modeb_border_erosion
    from siim.constants import HC_OVER_H, S_FLOOR_BC, FLOTATION_RAMP
    hc = float(HC_OVER_H)
    n = 6
    rec = np.array([0, 0, 1, 2, 3, 4], dtype=np.int64)
    stack = np.arange(n, dtype=np.int64)
    lengths = np.full(n, 100.0)
    lengths[0] = 0.0
    Co = 1e-6                                    # eff-exp border rate is Co-driven

    def run(zb0, dt):
        zb = zb0.copy(); H = np.zeros(n); surf = np.empty(n)
        glac_fast_solve_modeB(zb, np.full(n, 5e5), np.zeros(n), H, surf,
                              1e-6, Co, 1.0, 1.0, 0.5, 4.0 / 15.0,
                              2.3e-4, 10.0, 300.0,
                              dt, lengths, stack, rec,
                              0.0, 1, n, 100.0, 100.0, np.zeros(n), hc)
        return zb, H, surf

    for dt in (500.0, 5e4):
        zb0 = np.linspace(0.0, 500.0, n)
        zb, H, surf = run(zb0, dt)
        assert H[0] == H[1]                      # zero-gradient thickness
        # replicate step 5b: arrival slope from the dominant-donor chain
        # (dom[0] = 1, dom[1] = 2 on the strip), post-step interior beds
        S_arr = max(S_FLOOR_BC,
                    (zb[2] + hc * H[2] - zb[1] - hc * H[1]) / 100.0)
        E = _modeb_border_erosion(LAW_EFFEXP, 5e5, S_arr, H[0],
                                  Co, 4.0 / 15.0, 1.0, 0.0, 2.3e-4, 10.0, 300.0)
        want = _implicit_border_step(zb0[0], 0.0, E, dt, H[0], hc, 0.0,
                                     float(FLOTATION_RAMP))
        assert zb[0] == want                     # bit-for-bit the closed form
        assert zb[0] < zb0[0]                    # eroding (the sill cure)
        assert zb[0] >= -hc * H[0] - 1e-9        # never below the draft
        assert surf[0] == zb[0] + hc * H[0]      # true state


# ---------------------------------------------------------------------------
# Flotation ramp (the effective-pressure softening of the gate): gamma = 0 IS
# the hard binary gate bit-for-bit,
# the ramp partially scales in-band cells, and it damps the binary gate's
# single-step overshoot at the toe.
# ---------------------------------------------------------------------------

def test_flotation_ramp_gamma0_is_binary():
    """gamma = 0 is the hard binary gate, bit-for-bit, and the ramp scales
    exactly as specified. One SFR kernel call on a fjord strip whose gate-time
    surface spans afloat (zs < bl), in-band (0 < zs - bl < gamma*hc*H) and
    deep-grounded cells:

    * gate=True, ramp=0 == gate=False BITWISE at every grounded cell
      (zs >= bl) and un-eroded at every afloat cell — the binary semantics;
    * ramp=0.1: afloat cells still un-eroded (f = 0 exactly for zs <= bl, the
      backstop), deep-grounded cells BITWISE == gate-off (f = 1), and the
      in-band cell eroded PARTIALLY (strictly between no-erosion and full).

    Plus scalar pins of _flot_factor itself (threshold, linearity, clip,
    thin-ice fallback)."""
    from siim._core.params import GlacialParams
    from siim._core.skeleton import _flot_factor
    HC = 1.5
    n = 8
    bl = -130.0     # waterline chosen so the strip spans all three classes
    zb0 = np.array([-1200.0, -1000.0, -700.0, -420.0, -320.0, -200.0,
                    150.0, 600.0])
    rec = np.array([0, 0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    stack = np.arange(n, dtype=np.int64)
    lengths = np.full(n, 100.0)
    lengths[0] = 0.0
    ice = np.full(n, 1e8)
    p = GlacialParams(1e-6, 5e-7, 0.0, 1.0, 1.0, 0.5, 4.0 / 15.0, 2.3e-4, 10.0,
                      300.0, 0.0, 0.0, 0.0, 0.0, HC, 0.0)

    def run(gate, ramp):
        zb = zb0.copy(); H = np.zeros(n); surf = np.empty(n)
        _glac_fast_solve_modeB_sfr(zb, ice, np.zeros(n), H, surf, LAW_EFFEXP, p,
                                   500.0, lengths, stack, rec, 1, n, 100.0,
                                   100.0, np.zeros(n), bl, gate, ramp)
        return zb, H

    zb_off, H = run(False, 0.0)
    zb_g0, H0 = run(True, 0.0)
    zb_g1, H1 = run(True, 0.1)
    np.testing.assert_array_equal(H, H0)         # the walk never reads the gate
    np.testing.assert_array_equal(H, H1)
    zs_pre = zb0 + HC * H                        # gate-time (pre-erosion) surface
    band = 0.1 * HC * H
    afloat = zs_pre <= bl
    inband = (zs_pre > bl) & (zs_pre - bl < band)
    grounded = zs_pre >= bl
    deep = zs_pre - bl >= band
    intr = np.arange(n) != 0                     # border bed has its own budget
    # the config must exercise all three classes (guards the hand-tuned numbers)
    assert afloat[intr].any() and inband[intr].any() and deep[intr].any()
    assert (zb0 - zb_off)[intr].min() > 0.0      # gate-off erodes everywhere
    # gamma=0 == binary: grounded cells bitwise == gate-off, afloat un-eroded
    m = grounded & intr
    np.testing.assert_array_equal(zb_g0[m], zb_off[m])
    np.testing.assert_array_equal(zb_g0[afloat & intr], zb0[afloat & intr])
    # ramp: afloat still exactly un-eroded (backstop), deep-grounded bitwise
    # == gate-off, in-band strictly partial
    np.testing.assert_array_equal(zb_g1[afloat & intr], zb0[afloat & intr])
    np.testing.assert_array_equal(zb_g1[deep & intr], zb_off[deep & intr])
    k = inband & intr
    assert (zb_g1[k] < zb0[k]).all()             # eroded...
    assert (zb_g1[k] > zb_off[k]).all()          # ...but less than gate-off

    # _flot_factor scalar semantics
    assert _flot_factor(1.0, 100.0, 0.0, HC, 0.0) == 1.0     # gamma=0: grounded
    assert _flot_factor(0.0, 100.0, 0.0, HC, 0.0) == 1.0     # gamma=0: zs=bl grounded
    assert _flot_factor(-1e-9, 100.0, 0.0, HC, 0.0) == 0.0   # gamma=0: afloat
    assert _flot_factor(0.0, 100.0, 0.0, HC, 0.1) == 0.0     # ramp: f=0 at zs=bl
    assert _flot_factor(-5.0, 100.0, 0.0, HC, 0.1) == 0.0    # ramp: f=0 below bl
    d = 0.1 * HC * 100.0                                     # band width 15 m
    np.testing.assert_allclose(
        _flot_factor(0.5 * d, 100.0, 0.0, HC, 0.1), 0.5)     # linear mid-band
    assert _flot_factor(2.0 * d, 100.0, 0.0, HC, 0.1) == 1.0  # clipped above
    assert _flot_factor(1.0, 0.0, 0.0, HC, 0.1) == 1.0       # thin ice: binary
    assert _flot_factor(-1.0, 0.0, 0.0, HC, 0.1) == 0.0


# ---------------------------------------------------------------------------
# (vii) 2D localization: the implicit budget digs a bounded, DISTRIBUTED
# fjord-mouth trench across many border-face cells — no single-cell pit (the
# gate-era explicit budget dug one -9136 m cell at this config/dt).
# ---------------------------------------------------------------------------

def test_border_no_single_cell_pit_2d():
    """Probe d2_impl config (coulomb zELA=300, mode C, 121x61, looped-y) at
    dt = 50 kyr: the border-face deepest-carve envelope is bounded near the
    draft (face_min ~ -340; gate era: -9136), the digging is spread over many
    face cells (n_deep >= 15 of 122; probe: 34-36), and no face cell sits more
    than 500 m below both its along-face neighbours (no single-cell pit)."""
    from siim.siim2d import siim as siim2d
    m = siim2d(dict(U=1e-3, zELA=300.0, P=1, beta=1e-2, Ko=1e-6, n=1, ce=1e-5,
                    nu=2, T=2e6, nt=41, nt_out=41, Lx=120e3, Ly=60e3, nx=121,
                    ny=61,
                    boundary_status=['fixed_value', 'fixed_value', 'looped',
                                     'looped'],
                    initial_max_elevation=3000.0, mode='C', widening_rate=3.0,
                    flow_routing='single', sliding_law='coulomb', tau_c=1.2e5,
                    lambda_c=1000.0, progress_bar=False))
    m.run()
    zb_env = m.zb_out.min(0)                     # deepest-carve envelope (ny, nx)
    for face in (zb_env[:, 0], zb_env[:, -1]):
        assert face.min() > -1500.0, f"unbounded face pit ({face.min():.0f})"
        # no single-cell pit: nowhere more than 500 m below BOTH neighbours
        # (looped y: wrap the neighbour comparison)
        deeper_nb = np.minimum(np.roll(face, 1), np.roll(face, -1))
        assert (face - deeper_nb).min() > -500.0
    both = np.concatenate([zb_env[:, 0], zb_env[:, -1]])
    n_deep = int((both < -100.0).sum())
    assert n_deep >= 15, f"digging not distributed (n_deep {n_deep}/122)"
