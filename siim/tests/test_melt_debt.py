"""Melt-debt (audit m2) regression: the 1D ice-flux integral must NOT carry an
ablation deficit past a terminus.

PI adjudication (2026-07-06): "melt debt physically can't be carried." The 1D
walk (`_solve_ice_flux`) is fixed to a running clip (Skorokhod reflection at 0)
of the cumulative balance integral, equivalent to the paper's b=0-below-terminus
branch on a chain and to the 2D accumulator's per-hop `max(field_ice, 0)`. The 2D
accumulator (`_flow_accumulate_sd_2`) was already correct and is untouched.

Evidence for the divergence being fixed: docs/dev/toe_1d2d_parity.md §C.2 — on a
non-monotone flowline with a proglacial forebulge rising back above the ELA, the
old 1D suppressed re-nucleation (one icy run) while 2D re-nucleated a second
glacier (two runs). After the fix the 1D walk re-nucleates too.
"""
import numpy as np

from siim.siim1d import _solve_ice_flux, _reflect_nonneg, _cumtrapz
from siim._core.routing import _flow_accumulate_sd_2


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _icy_runs(mask):
    """Contiguous (start, end) index runs where mask is True."""
    runs, s = [], None
    for k in range(len(mask)):
        if mask[k] and s is None:
            s = k
        if not mask[k] and s is not None:
            runs.append((s, k - 1)); s = None
    if s is not None:
        runs.append((s, len(mask) - 1))
    return runs


def _ref_old_new(x, z, zELA, beta, d, sigma, k_h, xo, nx):
    """Independent reference for the one-sided (reflecting-divide at nx-1,
    outlet at 0), UNCAPPED (B_cap=inf) flank. Returns (Qg_old, Qg_new):

      Qg_old = max(prefactor * raw_integral, 0)     # the pre-m2 pointwise clip
      Qg_new = prefactor * reflect(raw_integral)    # the m2 running clip

    raw_integral reproduces `_solve_ice_flux`'s cumulative Hack-weighted integral
    plus the uncapped one-sided head term, so Qg_new is a bit-for-bit oracle for
    the model and Qg_old is the debt-carrying behavior it replaces."""
    dsigma = d * sigma
    xfd = x - x[nx - 1] + xo
    B = beta * (z - zELA)
    cum = _cumtrapz(((xfd ** (dsigma - 1)) * B)[::-1], xfd[::-1])[::-1]
    z_first = z[nx - 1]
    S = max((z[nx - 1] - z[nx - 2]) / (x[nx - 2] - x[nx - 1]), 0.0)
    I_head = (beta * (z_first + S * xo - zELA) * xo ** dsigma / dsigma
              - beta * S * xo ** (dsigma + 1.0) / (dsigma + 1.0))
    raw = cum + I_head
    pref = k_h * sigma * d / xfd ** (d * (sigma - 1.0))
    Qg_old = np.maximum(pref * raw, 0.0)
    refl = raw.copy()
    _reflect_nonneg(refl, nx - 1, -1, -1)   # flow order: divide (nx-1) -> outlet 0
    return Qg_old, pref * refl


def _forebulge_profile(nx, trough_extra=0.0):
    """One-sided flowline (flow high-index -> low-index): accumulation dome
    (>ELA) near the divide, deep ablation trough that exhausts the flux, then a
    proglacial forebulge that rises back above the ELA, then a final drop.
    `trough_extra` deepens the dead ablation trough between the two glaciers
    (more carried debt) without touching the forebulge."""
    i = np.arange(nx).astype(float)
    z = np.where(i >= 95, 1700.0 - (120.0 - i) * 4.0,
        np.where(i >= 40, 1600.0 - (95.0 - i) * 20.0,
        np.where(i >= 15, 500.0 + (40.0 - i) * 70.0,
                 500.0 + 25.0 * 70.0 - (15.0 - i) * 70.0)))
    band = (i >= 45) & (i <= 70)     # dead trough, well below ELA, between glaciers
    z[band] -= trough_extra
    return z


# shared physical params (one-sided, uncapped head)
_PARS = dict(zELA=1500.0, beta=1e-2, d=1.8, sigma=0.45, k_h=5.0, xo=500.0)
_NX, _DX = 121, 250.0
_X = (_NX - 1) * _DX - np.arange(_NX) * _DX     # decreasing with index (siim1d convention)


# ---------------------------------------------------------------------------
# 1. the reflection helper reproduces the paper / 2D semantics exactly
# ---------------------------------------------------------------------------
def test_reflect_is_pointwise_clip_on_monotone():
    """On a running integral that rises then falls monotonically through zero
    (a single terminating tongue), the reflection is bit-for-bit the old
    pointwise max(., 0) clip — the invariance the fix must preserve."""
    raw = np.array([0.5, 2.0, 3.0, 1.0, -0.5, -2.0, -5.0])
    a = raw.copy()
    _reflect_nonneg(a, 0, len(a), 1)
    np.testing.assert_array_equal(a, np.maximum(raw, 0.0))


def test_reflect_matches_2d_per_hop_clip_on_chain():
    """The 1D running clip reproduces the 2D accumulator's per-hop
    max(field_ice, 0) on a linear flow chain: both are the Lindley recursion
    C_i = max(C_{i-1} + src_i, 0). This is the audit-m2 alignment of the two
    front ends (docs/dev/toe_1d2d_parity.md §C.2)."""
    rng = np.random.default_rng(0)
    src = rng.normal(size=40) * 3.0            # mixed accumulation / ablation
    # 2D chain 0->1->...->n-1 (outlet self-receiving); donor-first is stack[::-1]
    n = src.size
    receivers = np.arange(1, n + 1, dtype=np.int64); receivers[-1] = n - 1
    stack = np.arange(n - 1, -1, -1, dtype=np.int64)   # receiver-before-donor
    field = np.zeros(n); field_ice = src.copy()
    _flow_accumulate_sd_2(field, field_ice, stack, receivers)
    Qg_2d = np.maximum(field_ice, 0.0)         # the processes.py output clamp
    # 1D running clip of the cumulative source
    Qg_1d = np.cumsum(src)
    _reflect_nonneg(Qg_1d, 0, n, 1)
    np.testing.assert_allclose(Qg_1d, Qg_2d, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. forebulge re-nucleation in the real _solve_ice_flux
# ---------------------------------------------------------------------------
def test_forebulge_renucleates_second_glacier():
    """A proglacial forebulge above the ELA now grows a second glacier (two icy
    runs) where the debt-carrying code grew only one — matching the 2D chain."""
    z = _forebulge_profile(_NX)
    _, Qg, _ = _solve_ice_flux(_X, z, _PARS['zELA'], _PARS['beta'], _PARS['d'],
                               _PARS['sigma'], _PARS['k_h'], 1.0,
                               _PARS['xo'], 0.0, _NX - 1, _NX, _NX, np.inf)
    Qg_old, Qg_new = _ref_old_new(_X, z, **_PARS, nx=_NX)

    # model matches the reflection oracle bit-for-bit
    np.testing.assert_array_equal(Qg, Qg_new)

    runs_new = _icy_runs(Qg > 0)
    runs_old = _icy_runs(Qg_old > 0)
    assert len(runs_old) == 1, f"pre-m2 should suppress re-nucleation, got {runs_old}"
    assert len(runs_new) == 2, f"m2 fix should re-nucleate a 2nd glacier, got {runs_new}"

    # the second (downstream, low-index) run sits on the above-ELA forebulge
    downstream = runs_new[0]                    # lowest indices = farthest downstream
    fore = np.where(z > _PARS['zELA'])[0]
    assert downstream[0] <= fore.min() + 2 and downstream[1] >= 2


# ---------------------------------------------------------------------------
# 3. monotone bit-for-bit invariance (mathematical-equivalence oracle)
# ---------------------------------------------------------------------------
def test_monotone_tongue_bit_for_bit():
    """On a single monotone-terminating tongue the fix is bit-for-bit the old
    pointwise-clip flux (the reflection never binds above the terminus, and both
    give 0 below it)."""
    i = np.arange(_NX).astype(float)
    z = np.where(i >= 60, 1400.0 + (i - 60.0) * 8.0,     # rises to the divide
                 1400.0 - (60.0 - i) * 30.0)             # monotone drop downstream
    _, Qg, _ = _solve_ice_flux(_X, z, _PARS['zELA'], _PARS['beta'], _PARS['d'],
                               _PARS['sigma'], _PARS['k_h'], 1.0,
                               _PARS['xo'], 0.0, _NX - 1, _NX, _NX, np.inf)
    Qg_old, Qg_new = _ref_old_new(_X, z, **_PARS, nx=_NX)
    assert len(_icy_runs(Qg > 0)) == 1
    np.testing.assert_array_equal(Qg, Qg_old)   # == the pre-m2 pointwise clip
    np.testing.assert_array_equal(Qg_old, Qg_new)


# ---------------------------------------------------------------------------
# 4. debt-free restart: the dead trough's depth cannot reduce the 2nd glacier
# ---------------------------------------------------------------------------
def test_debt_free_restart_independent_of_trough_depth():
    """The re-nucleated glacier's flux is set by the forebulge's own
    accumulation, not by how much ablation debt the dead trough upstream
    accrued. Deepening the trough (more debt) leaves the second glacier's Qg
    unchanged (running clip resets to zero), whereas the debt-carrying code
    would shrink or extinguish it."""
    sec = slice(0, 26)                           # the downstream (second) glacier
    z_shallow = _forebulge_profile(_NX, trough_extra=0.0)
    z_deep = _forebulge_profile(_NX, trough_extra=500.0)

    def qg(z):
        _, Q, _ = _solve_ice_flux(_X, z, _PARS['zELA'], _PARS['beta'], _PARS['d'],
                                  _PARS['sigma'], _PARS['k_h'], 1.0,
                                  _PARS['xo'], 0.0, _NX - 1, _NX, _NX, np.inf)
        return Q

    Q_shallow, Q_deep = qg(z_shallow), qg(z_deep)
    # second glacier is genuinely present, and its flux is identical (to floating
    # point) whether the upstream debt is shallow or deep — the restart is
    # debt-free: it depends only on the forebulge's own accumulation.
    assert (Q_shallow[sec] > 0).sum() >= 10
    np.testing.assert_allclose(Q_deep[sec], Q_shallow[sec], rtol=1e-9)

    # contrast: the pre-m2 debt-carrying rule extinguishes the second glacier
    # entirely (the carried deficit is never repaid) — for both trough depths.
    # The nonzero, depth-invariant restart above is exactly what the fix buys.
    old_shallow, _ = _ref_old_new(_X, z_shallow, **_PARS, nx=_NX)
    old_deep, _ = _ref_old_new(_X, z_deep, **_PARS, nx=_NX)
    assert (old_shallow[sec] > 0).sum() == 0
    assert (old_deep[sec] > 0).sum() == 0
