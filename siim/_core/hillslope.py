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

On a non-looped x axis the outer boundary COLUMNS (i=0, i=nx-1) are left
unchanged by the scheme (the y-pass writes only interior columns) -- a structural quirk of the fortran ADI
that is reproduced exactly for the bit-for-bit gate.

A LOOPED axis (``wrap_y`` / ``wrap_x``) has no edge: its first and last rows
(columns) are neighbours across the seam, so they join the implicit solve as
interior cells and that axis's tridiagonal systems become cyclic
(:func:`_solve_cyclic_tridiagonal`). The fortran-twin claims above hold for
grids with no looped axis.
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
def _solve_cyclic_tridiagonal(lower, diagonal, upper, rhs, solution, size):
    """Solve a cyclic tridiagonal system: the rows of :func:`_solve_tridiagonal`
    plus the corner couplings ``lower[0]`` (row 0 on unknown ``size-1``) and
    ``upper[size-1]`` (row ``size-1`` on unknown 0). Sherman-Morrison: two
    plain Thomas solves of the matrix with the corners folded into its first
    and last diagonal entries, then a rank-one correction. ``size >= 3``."""
    alpha = lower[0]
    beta = upper[size - 1]
    gamma = -diagonal[0]
    diag_mod = diagonal.copy()
    diag_mod[0] = diagonal[0] - gamma
    diag_mod[size - 1] = diagonal[size - 1] - alpha * beta / gamma
    _solve_tridiagonal(lower, diag_mod, upper, rhs, solution, size)
    u = np.zeros(size)
    u[0] = gamma
    u[size - 1] = beta
    q = np.empty(size)
    _solve_tridiagonal(lower, diag_mod, upper, u, q, size)
    fact = ((solution[0] + alpha * solution[size - 1] / gamma)
            / (1.0 + q[0] + alpha * q[size - 1] / gamma))
    for i in range(size):
        solution[i] -= fact * q[i]


@numba.njit(cache=True)
def _adi_step(z, kd, dt, dx, dy, fix_r0, fix_rN, fix_c0, fix_cN, wrap_y, wrap_x):
    """One ADI diffusion step (two ``dt/2`` half-steps) on ``z`` (ny, nx), face
    diffusivities from the arithmetic mean of ``kd`` (ny, nx). ``fix_*`` are the
    fixed-value (Dirichlet) flags for the four edges (row 0 / row ny-1 / col 0 /
    col nx-1); a False flag is the one-sided no-flux branch. On a looped axis
    (``wrap_y`` / ``wrap_x``) there is no edge: every row (column) is interior,
    its neighbours taken across the seam, and that axis's systems are cyclic.
    A non-looped x axis keeps the fortran quirk that the boundary columns are
    frozen (never written by the y-pass). With neither axis looped the float
    operations are the fortran twin's, bit-for-bit. Serial (no prange)."""
    ny, nx = z.shape
    h2dt = 0.5 * dt
    zint = z.copy()
    zintp = z.copy()
    r_lo, r_hi = (0, ny) if wrap_y else (1, ny - 1)
    c_lo, c_hi = (0, nx) if wrap_x else (1, nx - 1)

    # --- first pass: implicit in x, explicit in y ------------------------------
    diag = np.empty(nx)
    sup = np.zeros(nx)
    inf = np.zeros(nx)
    f = np.empty(nx)
    res = np.empty(nx)
    for r in range(r_lo, r_hi):
        rp = r + 1 if r < ny - 1 else 0
        rm = r - 1 if r > 0 else ny - 1
        for c in range(c_lo, c_hi):
            cp = c + 1 if c < nx - 1 else 0
            cm = c - 1 if c > 0 else nx - 1
            axp = 0.5 * (kd[r, cp] + kd[r, c]) * h2dt / (dx * dx)
            axm = 0.5 * (kd[r, cm] + kd[r, c]) * h2dt / (dx * dx)
            ayp = 0.5 * (kd[rp, c] + kd[r, c]) * h2dt / (dy * dy)
            aym = 0.5 * (kd[rm, c] + kd[r, c]) * h2dt / (dy * dy)
            diag[c] = 1.0 + axp + axm
            sup[c] = -axp
            inf[c] = -axm
            f[c] = (zintp[r, c] + ayp * zintp[rp, c]
                    - (ayp + aym) * zintp[r, c] + aym * zintp[rm, c])
        if wrap_x:
            _solve_cyclic_tridiagonal(inf, diag, sup, f, res, nx)
        else:
            if fix_c0:
                diag[0] = 1.0
                sup[0] = 0.0
                f[0] = zintp[r, 0]
            else:
                axp = 0.5 * (kd[r, 1] + kd[r, 0]) * h2dt / (dx * dx)
                ayp = 0.5 * (kd[rp, 0] + kd[r, 0]) * h2dt / (dy * dy)
                aym = 0.5 * (kd[rm, 0] + kd[r, 0]) * h2dt / (dy * dy)
                diag[0] = 1.0 + axp
                sup[0] = -axp
                f[0] = (zintp[r, 0] + ayp * zintp[rp, 0]
                        - (ayp + aym) * zintp[r, 0] + aym * zintp[rm, 0])
            if fix_cN:
                diag[nx - 1] = 1.0
                inf[nx - 1] = 0.0
                f[nx - 1] = zintp[r, nx - 1]
            else:
                axm = 0.5 * (kd[r, nx - 2] + kd[r, nx - 1]) * h2dt / (dx * dx)
                ayp = 0.5 * (kd[rp, nx - 1] + kd[r, nx - 1]) * h2dt / (dy * dy)
                aym = 0.5 * (kd[rm, nx - 1] + kd[r, nx - 1]) * h2dt / (dy * dy)
                diag[nx - 1] = 1.0 + axm
                inf[nx - 1] = -axm
                f[nx - 1] = (zintp[r, nx - 1] + ayp * zintp[rp, nx - 1]
                             - (ayp + aym) * zintp[r, nx - 1] + aym * zintp[rm, nx - 1])
            _solve_tridiagonal(inf, diag, sup, f, res, nx)
        for c in range(nx):
            zint[r, c] = res[c]

    # --- second pass: implicit in y, explicit in x ------------------------------
    diag = np.empty(ny)
    sup = np.zeros(ny)
    inf = np.zeros(ny)
    f = np.empty(ny)
    res = np.empty(ny)
    for c in range(c_lo, c_hi):
        cp = c + 1 if c < nx - 1 else 0
        cm = c - 1 if c > 0 else nx - 1
        for r in range(r_lo, r_hi):
            rp = r + 1 if r < ny - 1 else 0
            rm = r - 1 if r > 0 else ny - 1
            axp = 0.5 * (kd[r, cp] + kd[r, c]) * h2dt / (dx * dx)
            axm = 0.5 * (kd[r, cm] + kd[r, c]) * h2dt / (dx * dx)
            ayp = 0.5 * (kd[rp, c] + kd[r, c]) * h2dt / (dy * dy)
            aym = 0.5 * (kd[rm, c] + kd[r, c]) * h2dt / (dy * dy)
            diag[r] = 1.0 + ayp + aym
            sup[r] = -ayp
            inf[r] = -aym
            f[r] = (zint[r, c] + axp * zint[r, cp]
                    - (axp + axm) * zint[r, c] + axm * zint[r, cm])
        if wrap_y:
            _solve_cyclic_tridiagonal(inf, diag, sup, f, res, ny)
        else:
            if fix_r0:
                diag[0] = 1.0
                sup[0] = 0.0
                f[0] = zint[0, c]
            else:
                axp = 0.5 * (kd[0, cp] + kd[0, c]) * h2dt / (dx * dx)
                axm = 0.5 * (kd[0, cm] + kd[0, c]) * h2dt / (dx * dx)
                ayp = 0.5 * (kd[1, c] + kd[0, c]) * h2dt / (dy * dy)
                diag[0] = 1.0 + ayp
                sup[0] = -ayp
                f[0] = (zint[0, c] + axp * zint[0, cp]
                        - (axp + axm) * zint[0, c] + axm * zint[0, cm])
            if fix_rN:
                diag[ny - 1] = 1.0
                inf[ny - 1] = 0.0
                f[ny - 1] = zint[ny - 1, c]
            else:
                axp = 0.5 * (kd[ny - 1, cp] + kd[ny - 1, c]) * h2dt / (dx * dx)
                axm = 0.5 * (kd[ny - 1, cm] + kd[ny - 1, c]) * h2dt / (dx * dx)
                aym = 0.5 * (kd[ny - 2, c] + kd[ny - 1, c]) * h2dt / (dy * dy)
                diag[ny - 1] = 1.0 + aym
                inf[ny - 1] = -aym
                f[ny - 1] = (zint[ny - 1, c] + axp * zint[ny - 1, cp]
                             - (axp + axm) * zint[ny - 1, c] + axm * zint[ny - 1, cm])
            _solve_tridiagonal(inf, diag, sup, f, res, ny)
        for r in range(ny):
            zintp[r, c] = res[r]

    return zintp


def _fixed_flags(ibc):
    """Decode ``ibc`` -> (fix_row0, fix_rowN, fix_col0, fix_colN) booleans."""
    cbc = f"{int(ibc):04d}"
    return cbc[0] == '1', cbc[2] == '1', cbc[3] == '1', cbc[1] == '1'


def diffuse(elevation, diffusivity, dt, nx, ny, xl, yl, ibc, wrap_y=False,
            wrap_x=False):
    """Diffuse ``elevation`` one step (dt) by the ADI scheme; return the diffused
    surface, same 2D shape (ny, nx). ``elevation`` may be flat or 2D;
    ``diffusivity`` is a scalar ``D`` or an (ny, nx) field (not flat). ``ibc`` is the
    fastscapelib boundary code. Mirrors the stock ``LinearDiffusion`` step
    (``fs.diffusion`` on ``fs_context['h']``), the caller derives
    ``erosion = elevation - diffuse(...)``. ``dx = xl/(nx-1)``, ``dy = yl/(ny-1)``.
    ``wrap_y`` / ``wrap_x`` mark looped (periodic) axes; with neither, the
    fortran twin, bit-for-bit."""
    z = np.ascontiguousarray(np.asarray(elevation, dtype=np.float64).reshape(ny, nx))
    kd = np.ascontiguousarray(
        np.broadcast_to(np.asarray(diffusivity, dtype=np.float64), (ny, nx)))
    dx = xl / (nx - 1)
    dy = yl / (ny - 1)
    fix_r0, fix_rN, fix_c0, fix_cN = _fixed_flags(ibc)
    if (wrap_y and ny < 3) or (wrap_x and nx < 3):
        raise ValueError("a looped axis needs at least 3 cells for the cyclic "
                         f"ADI solve, got ny={ny}, nx={nx}")
    return _adi_step(z, kd, float(dt), dx, dy, fix_r0, fix_rN, fix_c0, fix_cN,
                     bool(wrap_y), bool(wrap_x))
