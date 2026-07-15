"""Explicit H diffusion for the siim 2D model (CFL-substepped 5-point FD).

Consolidated into the numerical core in the pre-v1.0 rewrite. numpy/numba
only -- no model/fastscape imports.
"""
import math

import numpy as np
import numba

@numba.njit(cache=True)
def _diffuse_H_2d(H_flat, ny, nx, dx, dy, D, dt, wrap_y=False, wrap_x=False):
    """Explicit 5-point FD diffusion on H, sub-stepped for CFL: per sub-step
    ax + ay <= 0.4 with ax = D*dt_sub/dx², ay = D*dt_sub/dy² (the 2D FTCS
    stability limit is ax + ay <= 0.5). Reduces to the previous square-grid
    behaviour exactly when dx == dy.

    A looped axis (``wrap_y`` / ``wrap_x``) wraps the stencil across the seam so
    the seam cells diffuse as the interior cells they physically are (matching
    the fill / facet-scan / carve seam-awareness; audit m16); a non-looped axis
    holds its outer ring fixed. Bit-for-bit with the old kernel when both are
    False. No-op when D <= 0. Clamps H >= 0 after each sub-step.
    """
    if D <= 0.0 or dt <= 0.0:
        return
    ax = D * dt / (dx * dx)
    ay = D * dt / (dy * dy)
    n_sub = max(1, int(math.ceil((ax + ay) / 0.4)))
    ax_sub = ax / n_sub
    ay_sub = ay / n_sub
    j0 = 0 if wrap_y else 1
    j1 = ny if wrap_y else ny - 1
    i0 = 0 if wrap_x else 1
    i1 = nx if wrap_x else nx - 1
    work = np.empty(ny * nx)
    for _ in range(n_sub):
        for k in range(ny * nx):
            work[k] = H_flat[k]
        for j in range(j0, j1):
            jm = (j - 1) % ny if wrap_y else j - 1
            jp = (j + 1) % ny if wrap_y else j + 1
            for i in range(i0, i1):
                im = (i - 1) % nx if wrap_x else i - 1
                ip = (i + 1) % nx if wrap_x else i + 1
                k = j * nx + i
                H_new = work[k] + (
                    ay_sub * (work[jm * nx + i] + work[jp * nx + i] - 2.0 * work[k])
                    + ax_sub * (work[j * nx + im] + work[j * nx + ip] - 2.0 * work[k])
                )
                if H_new < 0.0:
                    H_new = 0.0
                H_flat[k] = H_new
