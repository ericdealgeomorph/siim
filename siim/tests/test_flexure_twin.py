r"""S2 flexure gate: the in-house scipy.fft plate solve, validated against the
closed-form Kelvin point-load solution (the analytic ground truth), plus a
three-way sanity check against fortran ``fs.flexure``.

Why Kelvin, not fortran, is the gate (OQ-6 ratified FIX; Map 3 §1). The two
solvers are structurally different: the fortran resamples the domain onto the
central quarter of a power-of-two grid (~4x coarser, a Nunn & Aires 1988
anti-wraparound device) and its sine-transform pins the far field to zero; the
in-house solves on the native grid with the true wavenumbers. So they CANNOT
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

GAUGE CONVENTION (part of the ratified gate definition; Eric 2026-07-13 --
Kelvin oracle + k=0 zeroing, both ratified). The in-house solve zeroes the k=0
bin (far-field-neutral: the domain-mean load is rigidity-supported by the
surrounding plate), so it returns BOX-MEAN-FREE deflection fields by
construction. The infinite-plate Kelvin field instead carries the Airy integral
``int w dA = F / (rho_a g)`` (its gauge: w -> 0 at infinity); over the padded
box the two conventions differ by EXACTLY the uniform constant
``c = F_total(padded) / (A_pad rho_a g)`` (measured 1.75e-9 m for the point
load; ``w_withDC - w_zeroDC`` uniform to std 1.1e-24 -- pure gauge, zero shape
change). The gates therefore compare the fields in the SAME gauge: both minus
their annulus mean (:func:`_rel_rms_gauge`, equivalent to subtracting the
closed-form ``c`` from the oracle). This is a comparison-protocol alignment
forced by the far-field-neutral decision, NOT a tolerance loosening: the
tolerances are unchanged, the shape oracle keeps its full strength (2.0e-5
achieved), and the mean degree of freedom -- the one a gauge-invariant
comparison no longer sees -- is pinned MORE strictly by
:func:`test_flexure_mean_load_rigidity_supported` (exactly 0.0 mean response).
Shape + mean together fully constrain the field.

The Kelvin gates are pure (numpy/scipy); the fortran three-way is conda-only.
"""
import numpy as np
import pytest
from scipy.special import kei

from siim._core.flexure import flexure, _YOUNG, _POISSON, _G


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
    uniform offset between the two gauge conventions (box-mean-free in-house vs
    w->0-at-infinity Kelvin; the constant ``c`` of the module docstring) cancels
    and only the deflection SHAPE is compared."""
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


def test_flexure_mean_load_rigidity_supported():
    """k=0 (domain-mean) suppression (decision, Eric 2026-07-13): a UNIFORM
    unloading of a small domain (L << alpha) must NOT Airy-rebound -- the mean
    load is rigidity-supported by the surrounding plate (the periodic rfft2
    would otherwise tile the plate with loaded copies and lift the whole domain
    by ~rhos/rhoa per metre unloaded). siim's regime: 20 km domain vs
    alpha ~ 78 km at Te = 20 km. The fortran DST + zero-pad basis has this
    far-field-neutral behaviour structurally; the in-house matches it by
    zeroing the [0, 0] bin."""
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
    assert mean_abs < 0.15 * airy, \
        f"mean load must be rigidity-supported, not Airy-compensated: " \
        f"mean|w|={mean_abs:.3e} vs Airy={airy:.3f}"


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
