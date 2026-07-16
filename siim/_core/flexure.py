r"""In-house spectral flexure, the standalone replacement for
fastscapelib-fortran's ``fs.flexure`` (fastscape ``Flexure`` / siim
``GlacialFlexure``).

Thin elastic plate on an inviscid asthenosphere,

.. math::

    D\,\nabla^4 w + \rho_a\,g\,w = q,
    \qquad D = \frac{E\,T_e^3}{12\,(1-\nu^2)},

solved spectrally on the **native (ny, nx) grid** (no quarter-grid resample):
transfer :math:`\hat w = \hat q / (\rho_a g + D|k|^4)` with
:math:`|k|^2 = k_x^2 + k_y^2`, evaluated with ``scipy.fft`` rfft2/irfft2
(``workers=-1``). Hardcoded ``E = 1e11 Pa``, ``nu = 0.25``, ``g = 9.81`` -- the
9.81 MATCHES fastscape's flexure (``flexure2D.f90``), deliberately NOT siim's
``constants.GRAVITY = 9.8`` (which the shared step's ice-column term uses); this
cross-constant seam is intentional and documented.

Key differences from the fortran (both by design):

* **native grid.** The fortran resamples the domain onto the central quarter of
  a power-of-two grid (~4x coarser), a Nunn & Aires (1988) anti-wraparound
  device; the native-grid solve is finer and closer to the analytic Kelvin
  solution. So the two agree only to a tolerance, not bit-for-bit (twin-gated on
  square grids; the anisotropic gate is vs the Kelvin oracle).
* **anisotropy fixed (OQ-6).** The fortran builds the y-wavenumber with the
  x-spacing (``pihy = pi/hx``, ``flexure2D.f90:88,145``), correct only for square
  cells. Here ``k_y = 2*pi*fftfreq(ny, dy)`` uses the true y-spacing, so on
  ``dx != dy`` grids the in-house diverges from fortran BY DESIGN and instead
  tracks the closed-form point-load solution.
* **k=0 (domain-mean) mode ZEROED (decision, Eric 2026-07-13).** An rfft2 solve
  is periodic, so keeping k=0 would implicitly tile the plate with loaded
  copies of the domain and Airy-compensate the domain-mean load. siim's domains
  are isolated loaded patches with ``L << alpha`` (the flexural parameter
  ``alpha = (4D/(rho_a g))**(1/4)``, ~55 km at Te = 20 km, vs ~20 km domains),
  where the mean load is RIGIDITY-SUPPORTED by the surrounding plate -- the
  fortran DST + zero-pad basis encodes exactly that far-field-neutral
  assumption, and the in-house deliberately matches it (staying near
  established fastscape semantics). Regime caveat: for domains ``>> alpha``
  the full Airy compensation of the mean load becomes the correct limit --
  revisit the ``[0, 0]`` bin if such domains ever appear.

Same call signature as the fortran seam
``flexure(elev_post, elev_eq, nx, ny, xl, yl, rhos, rhoa, Te, ibc)`` -- mutates
``elev_post`` in place (adds the deflection ``w``) -- so it drops into the S1
injection seam (:func:`siim._core.step.glacial_flexure_step`) unchanged.

numpy/scipy only -- no fastscape/xsimlab imports (framework-free core).
"""
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
    native block sits at ``padded[off_y:off_y+ny, off_x:off_x+nx]``."""
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
    # (elev_post below elev_eq -> upward rebound). dx*dy is NOT applied: the
    # rfft2/irfft2 pair is norm-preserving, so a distributed pressure needs no
    # cell-area factor (the fortran's dx*dy force + 4/hx/hy DST norm cancel).
    load = -diff * rhos2 * _G

    cbc = f"{int(ibc):04d}"     # cbc[0]=top row0, cbc[2]=bottom rowN, cbc[3]=left col0, cbc[1]=right colN
    free_top = cbc[0] == '0'
    free_bottom = cbc[2] == '0'
    free_left = cbc[3] == '0'
    free_right = cbc[1] == '0'

    lp, oy, ox = _pad_load(load, free_top, free_bottom, free_left, free_right)
    Ny, Nx = lp.shape
    ky = (2.0 * np.pi * fft.fftfreq(Ny, d=dy))[:, None]
    kx = (2.0 * np.pi * fft.rfftfreq(Nx, d=dx))[None, :]
    k4 = (kx ** 2 + ky ** 2) ** 2
    w_hat = fft.rfft2(lp, workers=-1) / (rhoa * _G + D * k4)
    # Zero the k=0 (domain-mean) bin: the mean load is rigidity-supported by
    # the surrounding plate, not Airy-compensated (see the module docstring;
    # decision note, Eric 2026-07-13). ONLY the single mean bin -- the ky=0 row
    # and kx=0 column otherwise stay.
    w_hat[0, 0] = 0.0
    w = fft.irfft2(w_hat, s=lp.shape, workers=-1)
    w = w[oy:oy + ny, ox:ox + nx]

    elev_post += w.ravel()
