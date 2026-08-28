r"""In-house hillslope diffusion (ADI), the standalone replacement for
fastscapelib-fortran's ``fs.diffusion`` (stock ``LinearDiffusion``).

Linear hillslope diffusion :math:`\partial z/\partial t = \nabla\cdot(k_d\nabla z)`
advanced one step by the **alternating-direction-implicit** (ADI) scheme,
matching fastscapelib-fortran ``Diffusion.f90`` (v2.8.4): two ``dt/2``
half-steps (x-implicit then y-implicit), arithmetic-mean face diffusivities, a
Thomas tridiagonal solve per row/column, and the exact Dirichlet (fixed) /
one-sided no-flux (free) edge branches keyed to the same ``ibc`` digit map. The
Thomas solver is independently expressed from the standard row recurrence. The
scheme is unconditionally stable and, for spatially-uniform ``k_d`` (siim's
scalar ``D``), reproduces ``fs.diffusion`` bit-for-bit (twin-gated,
``rtol`` ~1e-12).

numpy/numba only -- no fastscape/xsimlab imports (framework-free core). This is
the HILLSLOPE diffuser (topography); distinct from :mod:`siim._core.diffusion`,
the ice-thickness FD diffuser, which is not a migration target.

Edge/``ibc`` map (single source: ``fs.processes.boundary`` +
``Diffusion.f90``). ``cbc = f"{ibc:04d}"`` and digit ``'1'`` = fixed_value
(Dirichlet), ``'0'`` = free (no-flux)::

    row j=0      -> cbc[0]      (top)
    row j=ny-1   -> cbc[2]      (bottom)
    col i=0      -> cbc[3]      (left)
    col i=nx-1   -> cbc[1]      (right)

The outer boundary COLUMNS (i=0, i=nx-1) are left unchanged by the scheme (the
y-pass writes only interior columns) -- a structural quirk of the fortran ADI
that is reproduced exactly for the bit-for-bit gate.
"""
import numpy as np
import numba


@numba.njit(cache=True)
def _solve_tridiagonal(lower, diagonal, upper, rhs, solution, size):
    """Solve a tridiagonal system with the standard Thomas recurrence.

    ``lower[i]``, ``diagonal[i]``, and ``upper[i]`` multiply the unknowns at
    indices ``i-1``, ``i``, and ``i+1``. The result is written into the
    preallocated ``solution`` array. This implementation was independently
    derived from that row equation; it performs no pivoting.
    """
    modified_upper = np.empty(size)

    pivot = diagonal[0]
    solution[0] = rhs[0] / pivot

    for i in range(1, size):
        modified_upper[i - 1] = upper[i - 1] / pivot
        pivot = diagonal[i] - lower[i] * modified_upper[i - 1]
        solution[i] = (rhs[i] - lower[i] * solution[i - 1]) / pivot

    for i in range(size - 2, -1, -1):
        solution[i] = solution[i] - modified_upper[i] * solution[i + 1]


@numba.njit(cache=True)
def _adi_step(z, kd, dt, dx, dy, fix_r0, fix_rN, fix_c0, fix_cN):
    """One ADI diffusion step (two ``dt/2`` half-steps) on ``z`` (ny, nx), face
    diffusivities from the arithmetic mean of ``kd`` (ny, nx). ``fix_*`` are the
    fixed-value (Dirichlet) flags for the four edges (row 0 / row ny-1 / col 0 /
    col nx-1); a False flag is the one-sided no-flux branch. Serial (no prange) --
    grids are small and the tridiagonal solves are sequential; order-independent
    by construction. Returns the diffused (ny, nx) array. The x-boundary columns
    are frozen (never written by the y-pass), matching the fortran ADI."""
    ny, nx = z.shape
    h2dt = 0.5 * dt
    zint = z.copy()          # first-pass (x-implicit) result
    zintp = z.copy()         # second-pass (y-implicit) result

    # --- first pass: implicit in x, explicit in y (interior rows only) --------
    diag = np.empty(nx)
    sup = np.zeros(nx)
    inf = np.zeros(nx)
    f = np.empty(nx)
    res = np.empty(nx)
    for r in range(1, ny - 1):
        for c in range(1, nx - 1):
            axp = 0.5 * (kd[r, c + 1] + kd[r, c]) * h2dt / (dx * dx)
            axm = 0.5 * (kd[r, c - 1] + kd[r, c]) * h2dt / (dx * dx)
            ayp = 0.5 * (kd[r + 1, c] + kd[r, c]) * h2dt / (dy * dy)
            aym = 0.5 * (kd[r - 1, c] + kd[r, c]) * h2dt / (dy * dy)
            diag[c] = 1.0 + axp + axm
            sup[c] = -axp
            inf[c] = -axm
            f[c] = (zintp[r, c] + ayp * zintp[r + 1, c]
                    - (ayp + aym) * zintp[r, c] + aym * zintp[r - 1, c])
        # left edge (col 0)
        if fix_c0:
            diag[0] = 1.0
            sup[0] = 0.0
            f[0] = zintp[r, 0]
        else:
            axp = 0.5 * (kd[r, 1] + kd[r, 0]) * h2dt / (dx * dx)
            ayp = 0.5 * (kd[r + 1, 0] + kd[r, 0]) * h2dt / (dy * dy)
            aym = 0.5 * (kd[r - 1, 0] + kd[r, 0]) * h2dt / (dy * dy)
            diag[0] = 1.0 + axp
            sup[0] = -axp
            f[0] = (zintp[r, 0] + ayp * zintp[r + 1, 0]
                    - (ayp + aym) * zintp[r, 0] + aym * zintp[r - 1, 0])
        # right edge (col nx-1)
        if fix_cN:
            diag[nx - 1] = 1.0
            inf[nx - 1] = 0.0
            f[nx - 1] = zintp[r, nx - 1]
        else:
            axm = 0.5 * (kd[r, nx - 2] + kd[r, nx - 1]) * h2dt / (dx * dx)
            ayp = 0.5 * (kd[r + 1, nx - 1] + kd[r, nx - 1]) * h2dt / (dy * dy)
            aym = 0.5 * (kd[r - 1, nx - 1] + kd[r, nx - 1]) * h2dt / (dy * dy)
            diag[nx - 1] = 1.0 + axm
            inf[nx - 1] = -axm
            f[nx - 1] = (zintp[r, nx - 1] + ayp * zintp[r + 1, nx - 1]
                         - (ayp + aym) * zintp[r, nx - 1] + aym * zintp[r - 1, nx - 1])
        _solve_tridiagonal(inf, diag, sup, f, res, nx)
        for c in range(nx):
            zint[r, c] = res[c]

    # --- second pass: implicit in y, explicit in x (interior columns only) ----
    diag = np.empty(ny)
    sup = np.zeros(ny)
    inf = np.zeros(ny)
    f = np.empty(ny)
    res = np.empty(ny)
    for c in range(1, nx - 1):
        for r in range(1, ny - 1):
            axp = 0.5 * (kd[r, c + 1] + kd[r, c]) * h2dt / (dx * dx)
            axm = 0.5 * (kd[r, c - 1] + kd[r, c]) * h2dt / (dx * dx)
            ayp = 0.5 * (kd[r + 1, c] + kd[r, c]) * h2dt / (dy * dy)
            aym = 0.5 * (kd[r - 1, c] + kd[r, c]) * h2dt / (dy * dy)
            diag[r] = 1.0 + ayp + aym
            sup[r] = -ayp
            inf[r] = -aym
            f[r] = (zint[r, c] + axp * zint[r, c + 1]
                    - (axp + axm) * zint[r, c] + axm * zint[r, c - 1])
        # row 0 edge
        if fix_r0:
            diag[0] = 1.0
            sup[0] = 0.0
            f[0] = zint[0, c]
        else:
            axp = 0.5 * (kd[0, c + 1] + kd[0, c]) * h2dt / (dx * dx)
            axm = 0.5 * (kd[0, c - 1] + kd[0, c]) * h2dt / (dx * dx)
            ayp = 0.5 * (kd[1, c] + kd[0, c]) * h2dt / (dy * dy)
            diag[0] = 1.0 + ayp
            sup[0] = -ayp
            f[0] = (zint[0, c] + axp * zint[0, c + 1]
                    - (axp + axm) * zint[0, c] + axm * zint[0, c - 1])
        # row ny-1 edge
        if fix_rN:
            diag[ny - 1] = 1.0
            inf[ny - 1] = 0.0
            f[ny - 1] = zint[ny - 1, c]
        else:
            axp = 0.5 * (kd[ny - 1, c + 1] + kd[ny - 1, c]) * h2dt / (dx * dx)
            axm = 0.5 * (kd[ny - 1, c - 1] + kd[ny - 1, c]) * h2dt / (dx * dx)
            aym = 0.5 * (kd[ny - 2, c] + kd[ny - 1, c]) * h2dt / (dy * dy)
            diag[ny - 1] = 1.0 + aym
            inf[ny - 1] = -aym
            f[ny - 1] = (zint[ny - 1, c] + axp * zint[ny - 1, c + 1]
                         - (axp + axm) * zint[ny - 1, c] + axm * zint[ny - 1, c - 1])
        _solve_tridiagonal(inf, diag, sup, f, res, ny)
        for r in range(ny):
            zintp[r, c] = res[r]

    return zintp


def _fixed_flags(ibc):
    """Decode ``ibc`` -> (fix_row0, fix_rowN, fix_col0, fix_colN) booleans."""
    cbc = f"{int(ibc):04d}"
    return cbc[0] == '1', cbc[2] == '1', cbc[3] == '1', cbc[1] == '1'


def diffuse(elevation, diffusivity, dt, nx, ny, xl, yl, ibc):
    """Diffuse ``elevation`` one step (dt) by the ADI scheme; return the diffused
    surface, same 2D shape (ny, nx). ``elevation`` / ``diffusivity`` may be flat
    or 2D; ``diffusivity`` is a scalar ``D`` or an (ny, nx) field. ``ibc`` is the
    fastscapelib boundary code. Mirrors the stock ``LinearDiffusion`` step
    (``fs.diffusion`` on ``fs_context['h']``), the caller derives
    ``erosion = elevation - diffuse(...)``. ``dx = xl/(nx-1)``, ``dy = yl/(ny-1)``."""
    z = np.ascontiguousarray(np.asarray(elevation, dtype=np.float64).reshape(ny, nx))
    kd = np.ascontiguousarray(
        np.broadcast_to(np.asarray(diffusivity, dtype=np.float64), (ny, nx)))
    dx = xl / (nx - 1)
    dy = yl / (ny - 1)
    fix_r0, fix_rN, fix_c0, fix_cN = _fixed_flags(ibc)
    return _adi_step(z, kd, float(dt), dx, dy, fix_r0, fix_rN, fix_c0, fix_cN)
