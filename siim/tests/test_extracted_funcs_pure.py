"""S1 extraction gate: the framework-free composition-chain functions in
:mod:`siim._core.step`.

Two kinds of check:

* **pure-fn unit tests** (framework-free, always run) — each extracted function
  against an independent computation, including the item-13 ``sum_erosion`` /
  ``compose_vertical_motion`` (written fresh for the standalone driver, so they
  MUST reproduce the stock fastscape group-sum composition given the same
  inputs);
* **shell-vs-pure equivalence** (needs the xsimlab adapter — ``importorskip``) —
  the still-xsimlab process shell and the pure function on identical inputs
  produce identical arrays.

The full-chain bit-for-bit proof that the shelled processes reproduce the
pre-refactor numbers lives in ``test_reference_snapshots.py`` (the S0 battery);
this file adds direct per-function coverage.
"""
import numpy as np
import pytest

from siim import constants as _constants
from siim._core import step
from siim._core.solvers import LAW_EFFEXP, LAW_POWER, LAW_COULOMB


# ---------------------------------------------------------------------------
# 13. Vertical-motion composition (stock fastscape group-sum, reproduced fresh)
# ---------------------------------------------------------------------------
def test_sum_erosion_reproduces_group_sum():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 4))
    b = rng.standard_normal((5, 4))
    # stock TotalErosion.height = sum(erosion group) == builtin sum from 0
    np.testing.assert_array_equal(step.sum_erosion(a, b), sum([a, b]))
    # a single term returns a copy (0 + a), not the same object
    one = step.sum_erosion(a)
    np.testing.assert_array_equal(one, a)
    assert one is not a


def test_compose_vertical_motion_with_flexure():
    rng = np.random.default_rng(1)
    uplift = rng.standard_normal((6, 3))       # both tectonic forcings = block uplift
    rebound = rng.standard_normal((6, 3))
    ero = rng.standard_normal((6, 3))
    surf_up, bed_up = step.compose_vertical_motion(uplift, uplift, rebound, ero)
    # stock TotalVerticalMotion: surface = sum(up group) - sum(down group);
    # bedrock = sum(up group). up group = {tect, rebound}; down = {height}.
    np.testing.assert_array_equal(surf_up, sum([uplift, rebound]) - sum([ero]))
    np.testing.assert_array_equal(bed_up, sum([uplift, rebound]))


def test_compose_vertical_motion_no_flexure():
    rng = np.random.default_rng(2)
    uplift = rng.standard_normal((4, 4))
    ero = rng.standard_normal((4, 4))
    surf_up, bed_up = step.compose_vertical_motion(uplift, uplift, None, ero)
    # No flexure: up group = {tect} only.
    np.testing.assert_array_equal(surf_up, uplift - ero)
    np.testing.assert_array_equal(bed_up, uplift)


# ---------------------------------------------------------------------------
# 2/3. Initial topography + block uplift
# ---------------------------------------------------------------------------
def test_uplift_mask_zeros_fixed_borders():
    shape = (5, 6)
    bs = ['fixed_value', 'core', 'looped', 'fixed_value']  # left, right, top, bottom
    mask = step.uplift_mask(bs, shape)
    expect = np.ones(shape)
    expect[:, 0] = 0.0     # left fixed
    expect[-1, :] = 0.0    # bottom fixed
    np.testing.assert_array_equal(mask, expect)


def test_block_uplift_desqueezes_and_masks():
    shape = (4, 4)
    mask = step.uplift_mask(['fixed_value'] * 4, shape)
    dt = 100.0
    # scalar rate
    up = step.block_uplift(2.0, dt, mask, shape)
    assert up.shape == shape
    np.testing.assert_array_equal(up, 2.0 * dt * mask)
    # (1, ny, nx) rate — the xsimlab size-1 tstep slice; leading dim dropped
    rate3 = np.full((1,) + shape, 3.0)
    up3 = step.block_uplift(rate3, dt, mask, shape)
    np.testing.assert_array_equal(up3, 3.0 * dt * mask)


def test_initial_topography_seeded_and_edges():
    shape = (5, 5)
    z0 = np.full(shape, 100.0)
    bs = ['fixed_value', 'core', 'core', 'fixed_value']  # left, right, top, bottom
    a = step.initial_topography(z0, shape, bs, seed=7, noise_amplitude=10.0)
    b = step.initial_topography(z0, shape, bs, seed=7, noise_amplitude=10.0)
    np.testing.assert_array_equal(a, b)                 # seeded reproducibility
    assert np.all(a[:, 0] == 100.0)                     # left fixed: no noise
    assert np.all(a[-1, :] == 100.0)                    # bottom fixed
    assert np.any(a[1:-1, 1:] != 100.0)                 # interior perturbed


# ---------------------------------------------------------------------------
# 4. Routing surface: EMA thickness + hc reconstruction
# ---------------------------------------------------------------------------
def test_ema_thickness_raw_passthrough_at_r_zero():
    H = np.arange(6.0)
    out = step.ema_thickness(H, None, 0.0)
    assert out is H                                     # bit-for-bit: same object


def test_ema_thickness_seed_and_update():
    H = np.array([2.0, 4.0, 6.0])
    seed = step.ema_thickness(H, None, 0.5)             # first step seeds at H
    np.testing.assert_array_equal(seed, H)
    prev = np.array([1.0, 1.0, 1.0])
    out = step.ema_thickness(H, prev, 0.5)
    np.testing.assert_array_equal(out, 0.5 * prev + 0.5 * H)


def test_routing_surface():
    bed = np.array([[0.0, 1.0], [2.0, 3.0]])
    H = np.array([[10.0, 20.0], [0.0, 5.0]])
    np.testing.assert_array_equal(step.routing_surface(bed, 1.5, H), bed + 1.5 * H)


# ---------------------------------------------------------------------------
# 1. Law record
# ---------------------------------------------------------------------------
def _law_kwargs(**over):
    base = dict(sliding_law='power', Ko=1e-6, ce=1e-4, n=1.0, nu=2.0, m=None,
                mu=None, Ac=2e-24, alpha_g=8.0, lambda_p=500.0, lambda_c=None,
                tau_c=1e5, coulomb_clamp=_constants.COULOMB_CLAMP,
                hc_over_H=1.5, H_diffusivity=None)
    base.update(over)
    return base


def test_build_glacial_params_power():
    code, gp = step.build_glacial_params(**_law_kwargs(sliding_law='power'))
    assert code == LAW_POWER
    assert gp.Ko == 1e-6 and gp.n == 1.0
    assert gp.m == 0.5                                  # default n/2
    assert gp.alpha_g == 8.0
    assert gp.hc_over_H == 1.5 and gp.D_H == 0.0


def test_build_glacial_params_coulomb():
    code, gp = step.build_glacial_params(**_law_kwargs(sliding_law='coulomb'))
    assert code == LAW_COULOMB
    assert gp.tau_c == 1e5
    assert gp.lambda_c == _constants.LAMBDA_C           # None -> constants default
    assert gp.rho_g_g == _constants.RHO_ICE * _constants.GRAVITY


def test_build_glacial_params_effexp():
    code, gp = step.build_glacial_params(**_law_kwargs(sliding_law='eff-exp'))
    assert code == LAW_EFFEXP
    assert np.isfinite(gp.Co) and gp.Co > 0.0
    assert np.isfinite(gp.mu)


def test_build_glacial_params_explicit_overrides():
    code, gp = step.build_glacial_params(
        **_law_kwargs(m=0.7, mu=0.3, H_diffusivity=1e-2))
    assert gp.m == 0.7 and gp.D_H == 1e-2


def test_build_glacial_params_validation():
    with pytest.raises(ValueError):
        step.build_glacial_params(**_law_kwargs(sliding_law='bogus'))
    with pytest.raises(ValueError):
        step.build_glacial_params(**_law_kwargs(hc_over_H=0.0))
    with pytest.raises(ValueError):
        step.build_glacial_params(**_law_kwargs(hc_over_H=-1.0))


# ---------------------------------------------------------------------------
# 9. Carve: the optional-scratch-buffer path must equal the passed-buffer path
# ---------------------------------------------------------------------------
def test_carve_bed_optional_buffers_match():
    ny, nx = 9, 9
    nn = ny * nx
    hc = _constants.HC_OVER_H
    # A single tall interior ice source: disc R = alpha_g*H/2 = 400 m spans the
    # neighbours. The "kernel" incised it deeply (bed 200 -> 140), so the carve
    # hangs a parabola from the source surface and lowers the footprint.
    H_flat = np.zeros(nn)
    c = 4 * nx + 4                                      # centre cell (flat index)
    H_flat[c] = 100.0
    bed_pre = np.full(nn, 200.0)                        # flat pre-kernel bed
    bed_post0 = bed_pre.copy()
    bed_post0[c] = 140.0                                # kernel incised the source
    # SFR receivers: boundary ring self-receiving (border marker), interior -> up.
    rec = np.arange(nn, dtype=np.int64)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            rec[j * nx + i] = (j - 1) * nx + i

    def _run(pass_buffers):
        zb = bed_post0.copy()
        surf = zb + hc * H_flat
        buf = {}
        if pass_buffers:
            buf = dict(offsets=np.empty((ny, nx)), D=np.empty((ny, nx)),
                       SRC=np.empty((ny, nx), dtype=np.int64),
                       zb_kern=np.empty(nn))
        step.carve_bed(zb, H_flat.copy(), surf, bed_pre.copy(), rec,
                       alpha_g=8.0, hc_over_H=hc, widening_factor=float('inf'),
                       shape=(ny, nx), dx=100.0, dy=100.0,
                       wrap_y=False, wrap_x=False, **buf)
        return zb

    zb_passed = _run(True)
    zb_alloc = _run(False)
    np.testing.assert_array_equal(zb_passed, zb_alloc)
    assert np.any(zb_passed < bed_post0)                # the widening carve bit


# ---------------------------------------------------------------------------
# Shell-vs-pure equivalence (needs the xsimlab adapter)
# ---------------------------------------------------------------------------
@pytest.mark.adapter
def test_modeA_border_shell_matches_pure():
    pytest.importorskip('fastscape')
    pytest.importorskip('xsimlab')
    from siim.fastscape.processes import GlacialSPLModeA
    from siim._core.params import GlacialParams

    gp = GlacialParams(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.3e-4, 0.0, 300.0,
        1e-3, 1e5, 1e-12, 9016.0, 1.5, 0.0)
    n = 6
    rec = np.array([0, 0, 1, 2, 3, 4], dtype=np.int64)
    lengths = np.array([0.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    ice = np.full(n, 5e5)
    z = np.linspace(0.0, 500.0, n)

    # Pure function.
    H_pure = np.zeros(n)
    step._solve_border_H_modeA(z.copy(), H_pure, ice, rec, None, lengths,
                               LAW_EFFEXP, gp)

    # Still-xsimlab shell method (a bare instance; the delegator wiring +
    # getattr(nb_receivers) under SFR is what this guards).
    p = object.__new__(GlacialSPLModeA)
    p._law_code = LAW_EFFEXP
    p._gp = gp
    p.receivers = rec
    p.lengths = lengths
    p.ice_flux = ice
    H_shell = np.zeros(n)
    p._solve_border_H_modeA(z.copy(), H_shell)

    np.testing.assert_array_equal(H_shell, H_pure)
    assert H_pure[0] > 0.0                              # border cell was solved
