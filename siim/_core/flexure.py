r"""In-house spectral flexure, the standalone replacement for
fastscapelib-fortran's ``fs.flexure`` (fastscape ``Flexure`` / siim
``GlacialFlexure``).

Thin elastic plate on an inviscid asthenosphere,

.. math::

    D\,\nabla^4 w + \rho_a\,g\,w = q,
    \qquad D = \frac{E\,T_e^3}{12\,(1-\nu^2)},

solved spectrally on the **native (ny, nx) grid** (no quarter-grid resample) in a
**type-2 sine basis (DST-II)** on the padded box: transfer
:math:`\hat w = \hat q / (\rho_a g + D|k|^4)` with :math:`|k|^2 = k_x^2 + k_y^2`,
:math:`k_x = \pi j/(N_x\,d_x)` and :math:`k_y = \pi i/(N_y\,d_y)` for
modes :math:`i, j = 1 \ldots N`, evaluated with ``scipy.fft`` dstn/idstn
(``type=2``, ``workers=-1``). Type 2 rather than type 1 because scipy computes
DST-I through a real FFT of length ``2(N+1)``, near-prime for a box sized as a
fast FFT length, while DST-II runs at ``N`` itself: same fidelity, 2-3.5x
faster (measurements in the 2026-09-16 decision record). Hardcoded
``E = 1e11 Pa``, ``nu = 0.25``, ``g = 9.81`` -- the 9.81 MATCHES fastscape's
flexure (``flexure2D.f90``), deliberately NOT siim's ``constants.GRAVITY = 9.8``
(which the shared step's ice-column term uses); this cross-constant seam is
intentional and documented.

The sine basis clamps ``w = 0`` at the padded-box edge (half a sample outside
it in type 2, one sample in the fortran's type 1; invisible at the gates'
resolution) -- the ``flexure2D.f90`` basis, here on the native grid -- and
needs no regime switch
(decision, Eric 2026-09-11, superseding the 2026-07-13 domain-mean zeroing):

* domains ``L << alpha`` (siim's valleys; flexural parameter
  ``alpha = (4D/(rho_a g))**(1/4)``, ~55 km at Te = 20 km vs ~20 km domains):
  a uniform load is RIGIDITY-SUPPORTED by the plate held flat at the padded-box
  edge, not Airy-compensated -- 0.004 m per metre of uniform unloading on a
  20 km all-free box, matching the fortran's far-field-neutral behaviour (full
  Airy would be 0.875).
* domains ``>> alpha``: a localized load compensates LOCALLY at Airy and its
  deflection decays to zero well inside the box -- the correct large-domain
  limit. The retired periodic (rfft2) solve zeroed the domain-mean ``[0, 0]``
  bin, which removed the Airy mean of a localized load and re-emitted it as a
  uniform uplift of the WHOLE box (+3.5 m per 100 kyr wave step on a
  2500 x 250 km domain, an untouched far plateau rising 290 -> 1460 m).

Key differences from the fortran (both by design):

* **native grid.** The fortran resamples the domain onto the central quarter of
  a power-of-two grid (~4x coarser), a Nunn & Aires (1988) anti-wraparound
  device; the native-grid solve is finer and closer to the analytic Kelvin
  solution. So the two agree only to a tolerance, not bit-for-bit (twin-gated on
  square grids; the anisotropic gate is vs the Kelvin oracle).
* **anisotropy fixed (OQ-6).** The fortran builds the y-wavenumber with the
  x-spacing (``pihy = pi/hx``, ``flexure2D.f90:88,145``), correct only for square
  cells. Here ``k_y`` uses the true y-spacing ``dy``, so on ``dx != dy`` grids
  the in-house diverges from fortran BY DESIGN and instead tracks the
  closed-form point-load solution.

Same call signature as the fortran seam
``flexure(elev_post, elev_eq, nx, ny, xl, yl, rhos, rhoa, Te, ibc)`` -- mutates
``elev_post`` in place (adds the deflection ``w``) -- so it drops into the S1
injection seam (:func:`siim._core.step.glacial_flexure_step`) unchanged.

numpy/scipy only -- no fastscape/xsimlab imports (framework-free core).
"""
from functools import lru_cache

import numpy as np
from scipy import fft

# Fixed plate constants (match flexure2D.f90 lines 83-85).
_YOUNG = 1.0e11        # Young's modulus [Pa]
_POISSON = 0.25        # Poisson ratio
_G = 9.81              # gravity [m s^-2] -- fortran's flexure g (NOT constants.GRAVITY)


def _pad_load(load, free_top, free_bottom, free_left, free_right):
    """Embed ``load`` (ny, nx) in a >=2x, fast-FFT-length grid: mirror-reflect
    the load across each FREE (no-reflection at fixed) edge -- a symmetric
    extension that gives the zero-gradient (free) plate edge the fortran ``addw``
    reflection imposes -- and zero-pad the fixed edges (clamped far field) plus
    the anti-wraparound buffer. Returns ``(padded, off_y, off_x)`` where the
    native block sits at ``padded[off_y:off_y+ny, off_x:off_x+nx]``.

    WARNING -- since the sine basis the pad RATIO is physics, not just a
    numerical buffer: the padded box's outer edge is where the sine basis clamps
    ``w = 0``, so it sets how far from the load the plate is held flat, which IS
    the small-domain rigidity support. Measured on the 31x31 / 20 km all-free
    config (uniform 1 m unload), ``mean|w|`` / Airy runs 0.005 at the current
    >=2x, 0.028 at 3x, 0.081 at 4x, 0.35 at 6x. Do not retune the RATIO for
    speed without re-pinning :func:`test_flexure_mean_load_rigidity_supported`."""
    ny, nx = load.shape
    a = load
    off_y = 0
    off_x = 0
    for axis, (n, free_lo, free_hi) in enumerate(
            ((ny, free_top, free_bottom), (nx, free_left, free_right))):
        big = fft.next_fast_len(2 * n)
        pad = big - n
        lo = pad // 2
        hi = pad - lo
        # reflect (mirror about the boundary node) on free sides
        rlo = lo if free_lo else 0
        rhi = hi if free_hi else 0
        if rlo or rhi:
            width = [(0, 0), (0, 0)]
            width[axis] = (rlo, rhi)
            a = np.pad(a, width, mode='reflect')
        # zero-fill the remaining pad (fixed sides + anti-wraparound buffer)
        zlo = lo - rlo
        zhi = hi - rhi
        if zlo or zhi:
            width = [(0, 0), (0, 0)]
            width[axis] = (zlo, zhi)
            a = np.pad(a, width, mode='constant')
        if axis == 0:
            off_y = lo
        else:
            off_x = lo
    return a, off_y, off_x


@lru_cache(maxsize=2)
def _transfer(Ny, Nx, dy, dx, D, rhoa):
    """``rho_a g + D |k|^4`` on the padded (Ny, Nx) box: DST-II mode j has
    wavenumber ``pi*j/(N*d)``, j = 1..N (module docstring). Cached because it is
    about a quarter of a solve and identical every step of a run; read-only so a
    cached array can never be corrupted in place."""
    ky = (np.pi * np.arange(1, Ny + 1) / (Ny * dy))[:, None]
    kx = (np.pi * np.arange(1, Nx + 1) / (Nx * dx))[None, :]
    t = rhoa * _G + D * (kx ** 2 + ky ** 2) ** 2
    t.setflags(write=False)
    return t


def flexure(elev_post, elev_eq, nx, ny, xl, yl, rhos, rhoa, Te, ibc):
    """Flexural deflection of the elastic plate under the load implied by
    ``elev_post - elev_eq`` (the per-step loading increment the caller stacked
    into ``elev_post``); the deflection ``w`` is ADDED to ``elev_post`` in place
    (so ``rebound = elev_post - elev_eq_pre`` on return). Signature matches the
    fortran ``fs.flexure`` seam.

    ``elev_post`` / ``elev_eq`` / ``rhos`` are flat length-``nx*ny`` arrays (row-
    major, x fastest, as fastscape passes them); ``rhoa`` asthenospheric density;
    ``Te`` effective elastic thickness; ``ibc`` the fastscapelib boundary code.
    ``dx = xl/(nx-1)``, ``dy = yl/(ny-1)``."""
    dx = xl / (nx - 1)
    dy = yl / (ny - 1)
    D = _YOUNG * Te ** 3 / (12.0 * (1.0 - _POISSON ** 2))

    diff = (np.asarray(elev_post, dtype=np.float64)
            - np.asarray(elev_eq, dtype=np.float64)).reshape(ny, nx)
    rhos2 = np.broadcast_to(np.asarray(rhos, dtype=np.float64).ravel(),
                            (ny * nx,)).reshape(ny, nx)
    # Load PRESSURE [Pa] = rock-column weight per area; sign +ve for unloading
    # (elev_post below elev_eq -> upward rebound). dx*dy is NOT applied: scipy's
    # type-2 dstn/idstn pair at the default norm is an EXACT inverse, so the
    # amplitude convention cancels around the per-mode transfer and a
    # distributed pressure needs no cell-area factor (as for the rfft2/irfft2
    # pair this replaced; the fortran's dx*dy force + 4/hx/hy DST norm cancel
    # the same way). Measured, not asserted: the Kelvin point-load gate checks
    # the central deflection MAGNITUDE (tests/test_flexure_twin.py).
    load = -diff * rhos2 * _G

    cbc = f"{int(ibc):04d}"     # cbc[0]=top row0, cbc[2]=bottom rowN, cbc[3]=left col0, cbc[1]=right colN
    free_top = cbc[0] == '0'
    free_bottom = cbc[2] == '0'
    free_left = cbc[3] == '0'
    free_right = cbc[1] == '0'

    lp, oy, ox = _pad_load(load, free_top, free_bottom, free_left, free_right)
    Ny, Nx = lp.shape
    # scalar Te / rhoa only (the cache key); per-cell lithos density stays in rhos2
    w_hat = fft.dstn(lp, type=2, workers=-1) / _transfer(Ny, Nx, dy, dx, float(D), float(rhoa))
    w = fft.idstn(w_hat, type=2, workers=-1)[oy:oy + ny, ox:ox + nx]

    elev_post += w.ravel()
