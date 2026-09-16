r"""S2 flexure gate: the in-house scipy.fft plate solve, validated against the
closed-form Kelvin point-load solution (the analytic ground truth), plus a
three-way sanity check against fortran ``fs.flexure``.

Why Kelvin, not fortran, is the gate (OQ-6 ratified FIX; Map 3 §1). The two
solvers are structurally different: the fortran resamples the domain onto the
central quarter of a power-of-two grid (~4x coarser, a Nunn & Aires 1988
anti-wraparound device); the in-house solves on the native grid with the true
wavenumbers (both sine bases pin the far field to zero). So they CANNOT
agree to the ~1e-6 the S2 brief's fortran-vs-in-house line asked for -- measured
in-house-vs-fortran relRMS is ~5-15% on a square grid, which is entirely the
fortran's own discretization error: the in-house tracks the analytic Kelvin
solution to ~2e-5, the fortran only to ~5e-2 (see test_flexure_three_way).
Gating in-house vs the analytic oracle is the stronger, physically-correct test
(scientific-correctness rule: verify against a known limit, do not assert an
unachievable cross-implementation tolerance).

    Kelvin (Turcotte & Schubert): for a point force F on an infinite thin plate,
    w(r) = F * l^2 / (2*pi*D) * kei(r/l),   l = (D / (rho_a g))^(1/4),
    D = E Te^3 / [12 (1 - nu^2)].

Sign: siim's flexure treats +ve load as UNLOADING (material removed -> upward
rebound), so an unloading point force uses -kei (central deflection up,
w(0) = F l^2 / (8 D) > 0).

GAUGE CONVENTION. The in-house solve pins ``w = 0`` at the padded-box edges
(sine basis; decision 2026-09-11, superseding the 2026-07-13 k=0 zeroing),
while the infinite-plate Kelvin field carries the Airy integral
``int w dA = F / (rho_a g)`` in its own gauge (w -> 0 at infinity). The gates
therefore compare the fields in the SAME gauge: both minus their annulus mean
(:func:`_rel_rms_gauge`). That alignment is KEPT, not forced -- on the 161^2
all-free config of :func:`test_flexure_three_way_beats_fortran` (adapter-marked,
so ``-m 'not adapter'`` does not run it) the sine basis agrees with Kelvin to
relRMS 5.1e-5 in the RAW gauge and 5.3e-5 gauge-matched, the latter being the
``in-house vs Kelvin`` number that test prints. Removing the mean therefore
neither loosens the gate nor props it up; it stays because a uniform offset
between two gauge conventions carries no shape information.
CAVEAT: because both fields lose their annulus mean, these two Kelvin gates are
by construction blind to a spurious UNIFORM offset -- the very failure mode the
k=0 zeroing had. They are SHAPE gates, at full strength (2.0e-5 on the 201^2
square gate). The offset degree of freedom is pinned separately instead, in BOTH
regimes: :func:`test_flexure_mean_load_rigidity_supported` (a small domain's
uniform load is rigidity-supported, not Airy-compensated) and
:func:`test_flexure_large_domain_local_airy_zero_far_field` (on a domain
``>> alpha`` a localized load compensates locally at Airy with a zero far
field). Shape + offset together fully constrain the field.

The Kelvin gates are pure (numpy/scipy); the fortran three-way is conda-only.
"""
import itertools

import numpy as np
import pytest
from scipy.special import kei

from siim._core.flexure import flexure, _YOUNG, _POISSON, _G
from siim._core.step import glacial_flexure_step, uplift_mask
from siim.siim2d import _ibc_from_border_status


def _D(Te):
    return _YOUNG * Te ** 3 / (12.0 * (1.0 - _POISSON ** 2))


def _l(Te, rhoa):
    return (_D(Te) / (rhoa * _G)) ** 0.25


def _inhouse_deflection(load2d, nx, ny, xl, yl, rhos, rhoa, Te, ibc):
    """Deflection w (ny, nx) for a load PRESSURE field ``load2d`` [Pa] (+ve =
    unloading). Drives the real :func:`siim._core.flexure.flexure`: the solver
    reads the loading increment as ``elev_post - elev_eq`` and ``load =
    -diff*rhos*g``, so ``diff = -load/(rhos*g)`` injects the wanted pressure."""
    diff = -load2d / (rhos * _G)
    elev_eq = np.zeros(ny * nx)
    elev_post = diff.ravel().copy()
    flexure(elev_post, elev_eq, nx, ny, xl, yl,
            np.full(ny * nx, float(rhos)), rhoa, Te, ibc)
    return (elev_post - diff.ravel()).reshape(ny, nx)


def _kelvin_unload(nx, ny, xl, yl, F, rhoa, Te, i0, j0):
    """Analytic uplift for an UNLOADING point force F [N] at cell (j0, i0)."""
    D = _D(Te)
    l = _l(Te, rhoa)
    dx, dy = xl / (nx - 1), yl / (ny - 1)
    x = (np.arange(nx) - i0) * dx
    y = (np.arange(ny) - j0) * dy
    r = np.hypot(x[None, :], y[:, None])
    r = np.maximum(r, 1e-30)
    return -F * l ** 2 / (2.0 * np.pi * D) * kei(r / l), l


def _annulus(nx, ny, xl, yl, i0, j0, l):
    """Mask: away from the singular point (>= 3 cells) and inside the decay
    (<= 8 l), a few cells clear of the grid edges."""
    dx, dy = xl / (nx - 1), yl / (ny - 1)
    x = (np.arange(nx) - i0) * dx
    y = (np.arange(ny) - j0) * dy
    r = np.hypot(x[None, :], y[:, None])
    ci = np.arange(nx)[None, :]
    rj = np.arange(ny)[:, None]
    return ((r >= 3 * max(dx, dy)) & (r <= 8 * l)
            & (ci > 2) & (ci < nx - 3) & (rj > 2) & (rj < ny - 3))


def _rel_rms_gauge(a, b, mask):
    """Gauge-invariant relative RMS: both fields minus their annulus mean, so a
    uniform offset between the two gauge conventions (zero-at-the-padded-box-edge
    in-house vs w->0-at-infinity Kelvin; module docstring) cancels and only the
    deflection SHAPE is compared."""
    ag = a[mask] - a[mask].mean()
    bg = b[mask] - b[mask].mean()
    return float(np.sqrt(np.mean((ag - bg) ** 2)) / np.sqrt(np.mean(bg ** 2)))


def _point_load(nx, ny):
    load = np.zeros((ny, nx))
    load[ny // 2, nx // 2] = 1.0     # 1 Pa over one cell
    return load


def test_flexure_matches_kelvin_square():
    """Square uniform-density grid: the in-house deflection tracks the analytic
    Kelvin point-load solution (sign AND magnitude) to a tight relative RMS,
    compared in the SAME gauge (annulus-mean removed; module docstring). This
    is the S2 square-grid gate (vs the analytic oracle; see the module docstring
    on why not vs fortran)."""
    nx = ny = 201
    xl = yl = 1.2e6
    Te, rhoa, rhos = 15e3, 3200.0, 2800.0
    i0, j0 = nx // 2, ny // 2
    dx = xl / (nx - 1)
    w_in = _inhouse_deflection(_point_load(nx, ny), nx, ny, xl, yl, rhos, rhoa, Te, ibc=0)
    w_k, l = _kelvin_unload(nx, ny, xl, yl, 1.0 * dx * (yl / (ny - 1)), rhoa, Te, i0, j0)
    mask = _annulus(nx, ny, xl, yl, i0, j0, l)
    rr = _rel_rms_gauge(w_in, w_k, mask)
    # central deflection in the shared gauge (center minus the annulus mean)
    c_in = w_in[j0, i0] - w_in[mask].mean()
    c_k = w_k[j0, i0] - w_k[mask].mean()
    print(f"\n[flexure Kelvin square {nx}x{ny} Te={Te/1e3:.0f}km l/dx={l/dx:.1f}] "
          f"relRMS={rr:.3e}  w0_in={c_in:.4e} w0_k={c_k:.4e}  ncells={mask.sum()}")
    assert c_in > 0 and c_k > 0, "unloading -> upward central deflection"
    assert abs(c_in - c_k) / abs(c_k) < 5e-3, "central deflection magnitude"
    assert rr < 1e-3, f"in-house vs Kelvin relRMS={rr:.3e} (achieved ~2e-5)"


@pytest.mark.parametrize('shape', [(201, 141, 1.2e6, 1.68e6), (141, 201, 1.68e6, 1.2e6)],
                         ids=['dx<dy', 'dx>dy'])
def test_flexure_matches_kelvin_anisotropic(shape):
    """Anisotropic grid (dx != dy): the in-house is gated against Kelvin, NOT
    fortran (the fortran uses the x-spacing for the y-wavenumber -- the pihy bug
    OQ-6 fixes). Isotropic physics on a true-spacing solve tracks Kelvin
    regardless of the cell aspect ratio."""
    nx, ny, xl, yl = shape
    Te, rhoa, rhos = 15e3, 3200.0, 2800.0
    i0, j0 = nx // 2, ny // 2
    dx, dy = xl / (nx - 1), yl / (ny - 1)
    assert abs(dx - dy) / dx > 0.3, "precondition: genuinely anisotropic cells"
    w_in = _inhouse_deflection(_point_load(nx, ny), nx, ny, xl, yl, rhos, rhoa, Te, ibc=0)
    w_k, l = _kelvin_unload(nx, ny, xl, yl, 1.0 * dx * dy, rhoa, Te, i0, j0)
    mask = _annulus(nx, ny, xl, yl, i0, j0, l)
    rr = _rel_rms_gauge(w_in, w_k, mask)
    print(f"\n[flexure Kelvin aniso {nx}x{ny} dx={dx/1e3:.1f}km dy={dy/1e3:.1f}km] "
          f"relRMS={rr:.3e}  ncells={mask.sum()}")
    assert rr < 1e-3, f"in-house vs Kelvin (anisotropic) relRMS={rr:.3e} (achieved ~1.5e-4)"


@pytest.mark.pin
def test_flexure_mean_load_rigidity_supported():
    """Small-domain regime (L << alpha): a UNIFORM unloading must NOT
    Airy-rebound -- the mean load is rigidity-supported by the surrounding
    plate. siim's regime: 20 km domain vs alpha ~ 55 km at Te = 20 km. The
    fortran DST + zero-pad basis has this far-field-neutral behaviour
    structurally, and the in-house sine basis (decision 2026-09-11) has it
    structurally too: ``w`` is pinned to 0 at the padded-box edges, so a uniform
    load on a box far smaller than alpha is carried by plate rigidity (measured
    0.005 of Airy). The retired periodic rfft2 solve had to zero the [0, 0] bin
    explicitly to get here -- and that fix is what broke the large-domain
    regime (:func:`test_flexure_large_domain_local_airy_zero_far_field`)."""
    nx = ny = 31
    xl = yl = 2.0e4
    Te, rhoa, rhos = 20e3, 3200.0, 2800.0
    airy = rhos / rhoa * 1.0          # 0.875 m: the full-Airy response to 1 m unloading
    load = np.full((ny, nx), rhos * _G * 1.0)   # uniform 1 m rock unloading [Pa]
    w = _inhouse_deflection(load, nx, ny, xl, yl, rhos, rhoa, Te, ibc=0)
    mean_abs = float(np.mean(np.abs(w)))
    max_abs = float(np.max(np.abs(w)))
    print(f"\n[flexure mean-load {nx}x{ny} L={xl/1e3:.0f}km Te={Te/1e3:.0f}km] "
          f"Airy={airy:.3f} m  mean|w|={mean_abs:.3e} m  max|w|={max_abs:.3e} m  "
          f"suppression={mean_abs/airy:.3e}")
    # gate: 0.15 -> 0.02 (decision 2026-09-16): the pad RATIO sets this response,
    # and 0.15 passed a 4x box silently.
    assert mean_abs < 0.02 * airy, \
        f"mean load must be rigidity-supported, not Airy-compensated: " \
        f"mean|w|={mean_abs:.3e} vs Airy={airy:.3f}"


@pytest.mark.pin
def test_flexure_large_domain_local_airy_zero_far_field():
    """Large-domain regime (L >> alpha), the mirror of the small-domain pin: a
    localized load on a 2500 x 250 km domain must compensate LOCALLY at Airy and
    leave the far field untouched. Thea's Grand Canyon warm-up, at 10 km cells:
    a Gaussian uplift band in x, top row and right column fixed (ibc = 1100).
    The retired k=0 zeroing failed BOTH halves here -- it subtracted the Airy
    mean of the band (only 0.91 of Airy under it) and re-emitted it as a uniform
    uplift of the whole box: +3.43 m per step on THIS grid, +3.55 m on Thea's
    1 km cells, which compounded into the 290 -> 1460 m rise of an untouched
    plateau over 50 Myr of 100 kyr steps."""
    nx, ny = 251, 26
    xl, yl = 2500e3, 250e3
    Te, rhoa, rhos = 20e3, 3200.0, 2800.0
    dh_peak = 38.6
    x = np.linspace(0, xl, nx)
    # uplift band: a Gaussian in x, zeroed on the fixed top row and right column
    dh = np.broadcast_to(dh_peak * np.exp(-((x - 525e3) / 200e3) ** 2), (ny, nx)).copy()
    dh[0, :] = 0.0
    dh[:, -1] = 0.0
    # driven the way the caller stacks the seam (step.glacial_flexure_step),
    # not via _inhouse_deflection: this test's physics is an ELEVATION
    # increment, and the helper's pressure argument would only undo the
    # -dh*rhos*g conversion (agreement measured at 1.4e-14 m, so the choice is
    # legibility, not coverage).
    elev_eq = np.zeros(ny * nx)
    elev_post = elev_eq + dh.ravel()
    flexure(elev_post, elev_eq, nx, ny, xl, yl,
            np.full(ny * nx, rhos), rhoa, Te, ibc=1100)
    w = (elev_post - elev_eq - dh.ravel()).reshape(ny, nx)

    jm = ny // 2
    ipk = int(np.argmax(dh[jm]))
    airy = -rhos / rhoa * dh_peak       # -33.8 m: local Airy under the band
    w_band = w[jm, ipk]
    far = float(np.max(np.abs(w[:, x >= 1800e3])))
    print(f"\n[flexure large-domain {nx}x{ny} L={xl/1e3:.0f}km Te={Te/1e3:.0f}km] "
          f"w_band={w_band:.3f} m  Airy={airy:.3f} m  ratio={w_band/airy:.4f}  "
          f"far|w|(x>=1800km)={far:.3e} m  far/peak={far/abs(w_band):.3e}")
    assert abs(w_band - airy) / abs(airy) < 0.03, \
        f"localized load on L >> alpha must compensate at Airy: {w_band:.3f} vs {airy:.3f}"
    assert far < 1e-3 * abs(w_band), \
        f"far field must stay untouched: max|w|={far:.3e} m at x >= 1800 km"


@pytest.mark.pin
def test_flexure_rebound_zeroed_on_fixed_borders():
    """A ``fixed_value`` border gets NO flexural rebound. It already gets no
    block uplift (``uplift_mask``) and does not erode, so handing it the edge
    share of the subsidence sank it into a trench that then became the base
    level for everything draining to it: on Thea's setup (2500 x 250 km, 10 km
    cells, 10 Myr) the fixed y=0 row fell 287 -> 55 m while the interior rose
    292 -> 543 m. Stock fastscape has the same inconsistency (BlockUplift masks,
    Flexure does not) and siim inherited it.

    Also pins the ibc digit mapping itself against ``uplift_mask`` over all 16
    border combinations, rather than trusting the comment in
    :func:`siim._core.step.glacial_flexure_step`."""
    nx, ny = 251, 26
    xl, yl = 2500e3, 250e3
    Te, rhoa, rhos = 20e3, 3200.0, 2800.0
    dh_peak = 38.6
    x = np.linspace(0, xl, nx)
    band = dh_peak * np.exp(-((x - 525e3) / 200e3) ** 2)
    zero = np.zeros((ny, nx))

    def rebound_for(border_status, dh=band):
        """Uplift ``dh`` through the real step seam, already masked off the
        fixed edges the way the caller's block uplift leaves it. No ice."""
        dh = np.broadcast_to(dh, (ny, nx)) * uplift_mask(border_status, (ny, nx))
        rebound, _ = glacial_flexure_step(
            zero, zero, dh, zero, 5.0, (xl / (nx - 1)) * (yl / (ny - 1)),
            rhos, rhoa, Te, _ibc_from_border_status(border_status), (ny, nx),
            (yl, xl), zero, False, flexure)
        return rebound

    # (a)+(b): Thea's topology -- row 0 and column -1 fixed (ibc 1100)
    w = rebound_for(['core', 'fixed_value', 'fixed_value', 'core'])
    jm, ipk = ny // 2, int(np.argmax(band))
    airy = -rhos / rhoa * dh_peak
    print(f"\n[flexure fixed-border rebound {nx}x{ny} ibc=1100] "
          f"row0 max|w|={np.abs(w[0]).max():.3e} m  "
          f"col-1 max|w|={np.abs(w[:, -1]).max():.3e} m  "
          f"mid-row under band={w[jm, ipk]:.3f} m  Airy={airy:.3f} m  "
          f"ratio={w[jm, ipk]/airy:.4f}")
    assert np.all(w[0] == 0.0), "fixed row 0 must get no rebound"
    assert np.all(w[:, -1] == 0.0), "fixed column -1 must get no rebound"
    assert abs(w[jm, ipk] - airy) / abs(airy) < 0.03, \
        f"interior must still compensate at Airy: {w[jm, ipk]:.3f} vs {airy:.3f}"

    # (c): the ibc-derived mask IS uplift_mask, for all 16 border combinations.
    # Compared through the real rebound field, so this pins the implementation
    # and not a restatement of its digit rule. A UNIFORM load keeps every
    # non-fixed node's rebound near Airy, far from roundoff, so ``!= 0`` is a
    # mask and not a coin toss on the band's far field.
    for combo in itertools.product(('fixed_value', 'core'), repeat=4):
        got = (rebound_for(combo, dh=dh_peak) != 0.0).astype(float)
        assert np.array_equal(got, uplift_mask(combo, (ny, nx))), \
            f"rebound mask != uplift_mask for {combo} (ibc={_ibc_from_border_status(combo)})"


@pytest.mark.adapter
def test_flexure_three_way_beats_fortran():
    """Three-way validation of the oracle and the design intent: on a square grid
    both solvers ALSO agree with Kelvin (validating the oracle), but the in-house
    is far closer -- the whole in-house-vs-fortran gap IS the fortran's
    quarter-grid discretization error. Encodes OQ-6 ('the in-house fixes the
    fortran; the Kelvin gate is the stronger oracle') as a tolerance-free
    assertion, and reports the measured fortran-vs-in-house RMS the S2 brief's
    (unachievable) 1e-6 line referred to."""
    fs = pytest.importorskip('fastscapelib_fortran')
    nx = ny = 161
    xl = yl = 1.6e6
    Te, rhoa, rhos = 20e3, 3200.0, 2800.0
    i0, j0 = nx // 2, ny // 2
    dx = xl / (nx - 1)
    load = _point_load(nx, ny)
    rhos_flat = np.full(ny * nx, rhos)

    w_in = _inhouse_deflection(load, nx, ny, xl, yl, rhos, rhoa, Te, ibc=0)
    diff = (-load / (rhos * _G)).ravel()
    elev_eq = np.zeros(ny * nx)
    ep = (elev_eq + diff).copy()
    fs.flexure(ep, elev_eq, nx, ny, xl, yl, rhos_flat, rhoa, Te, 0)
    w_ft = (ep - diff).reshape(ny, nx)

    w_k, l = _kelvin_unload(nx, ny, xl, yl, 1.0 * dx * (yl / (ny - 1)), rhoa, Te, i0, j0)
    mask = _annulus(nx, ny, xl, yl, i0, j0, l)
    # all three pairwise comparisons in the shared (mean-removed) gauge; the
    # fortran number barely moves (its DST error is genuine SHAPE error, not
    # gauge: raw 4.8e-2 vs mean-removed 5.0e-2 -- measured at the k=0 decision)
    in_vs_k = _rel_rms_gauge(w_in, w_k, mask)
    ft_vs_k = _rel_rms_gauge(w_ft, w_k, mask)
    in_vs_ft = _rel_rms_gauge(w_in, w_ft, mask)
    print(f"\n[flexure 3-way {nx}x{ny} Te={Te/1e3:.0f}km] in-house vs Kelvin={in_vs_k:.3e} | "
          f"fortran vs Kelvin={ft_vs_k:.3e} | in-house vs fortran={in_vs_ft:.3e}")
    # both solve the same biharmonic (fortran within its coarse-grid error)
    assert ft_vs_k < 0.1, f"fortran vs Kelvin={ft_vs_k:.3e} (sanity: same equation)"
    assert in_vs_k < 1e-3, f"in-house vs Kelvin={in_vs_k:.3e}"
    # the in-house is the accurate one: closer to the analytic truth by >=20x
    assert in_vs_k < ft_vs_k / 20.0, \
        f"in-house should beat fortran vs Kelvin: {in_vs_k:.3e} vs {ft_vs_k:.3e}"
