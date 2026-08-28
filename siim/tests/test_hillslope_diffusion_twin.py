"""S2 twin gate: in-house ADI hillslope diffusion == fortran ``fs.diffusion``.

The in-house numba ADI (:func:`siim._core.hillslope.diffuse`) matches the
fastscapelib-fortran ``Diffusion.f90`` (v2.8.4) stencil while using an
independently expressed Thomas recurrence. For the spatially-uniform ``kd``
(scalar ``D``) siim uses, the two agree bit-for-bit. This drives the fortran
through the exact context ops the stock ``LinearDiffusion`` process uses (set
``kd`` / ``kdsed=-1`` / ``h`` / ``dt`` in the fortran context, call
``fs.diffusion()``, read ``h`` back) and diffs against the in-house solve.

Conda-only (needs fastscapelib-fortran).
"""
import numpy as np
import pytest

fs = pytest.importorskip('fastscapelib_fortran')
pytestmark = pytest.mark.adapter

from siim._core.hillslope import diffuse       # noqa: E402


def _ibc(left, right, top, bottom):
    """fastscapelib ibc code from fixed-value flags (fastscape BorderBoundary)."""
    a = [1 if x else 0 for x in (left, right, top, bottom)]
    return a[0] * 1 + a[1] * 100 + a[2] * 1000 + a[3] * 10


# The four fixed/free edge mixes (left, right, top, bottom).
_MIXES = {
    'all_fixed': _ibc(1, 1, 1, 1),
    'lr_fixed':  _ibc(1, 1, 0, 0),   # x-fixed, y-free
    'tb_fixed':  _ibc(0, 0, 1, 1),   # y-fixed, x-free
    'all_free':  _ibc(0, 0, 0, 0),   # pure no-flux
}


def _fortran_diffuse(z0, D, dt, nsteps, nx, ny, xl, yl, ibc):
    """Diffuse ``z0`` (ny, nx) ``nsteps`` times through the fortran, driving the
    context exactly as ``LinearDiffusion.run_step`` does. Init/destroy the global
    fortran context inside a try/finally so the singleton is left clean."""
    fs.fastscape_init()
    try:
        fs.fastscape_set_nx_ny(nx, ny)
        fs.fastscape_setup()
        fs.fastscape_set_xl_yl(xl, yl)
        fs.fastscape_set_bc(int(ibc))
        ctx = fs.fastscapecontext
        ctx.kd = np.broadcast_to(float(D), (ny, nx)).flatten().astype(float)
        ctx.kdsed = -1.0
        ctx.dt = float(dt)
        z = np.ascontiguousarray(z0, dtype=float)
        for _ in range(nsteps):
            ctx.h = z.flatten()
            fs.diffusion()
            z = np.array(ctx.h, dtype=float).reshape(ny, nx).copy()
        return z
    finally:
        fs.fastscape_destroy()


@pytest.mark.parametrize('nsteps', [1, 20], ids=['1step', 'many'])
@pytest.mark.parametrize('grid', [(31, 31, 2.0e4, 2.0e4), (27, 31, 1.7e4, 2.0e4)],
                         ids=['square', 'anisotropic'])
@pytest.mark.parametrize('mix', list(_MIXES), ids=list(_MIXES))
def test_hillslope_diffusion_twin(mix, grid, nsteps):
    """in-house ADI == fortran fs.diffusion bit-for-bit (rtol 1e-12 minimum) for
    scalar kd, every ibc mix, 1 and many steps, square and anisotropic grids."""
    ny, nx, yl, xl = grid[1], grid[0], grid[3], grid[2]
    ibc = _MIXES[mix]
    D, dt = 1e-3, 200.0
    rng = np.random.RandomState(3)
    z0 = rng.rand(ny, nx) * 100.0 + np.linspace(0, 300, ny)[:, None]

    zf = _fortran_diffuse(z0, D, dt, nsteps, nx, ny, xl, yl, ibc)
    zi = z0.copy()
    for _ in range(nsteps):
        zi = diffuse(zi, D, dt, nx, ny, xl, yl, ibc)

    d = np.abs(zi - zf)
    rel = float(np.max(d / np.maximum(np.abs(zf), 1e-30)))
    ulp = float(np.max(d / np.maximum(np.spacing(np.abs(zf)), 1e-300)))
    print(f"\n[diffusion twin {mix} {grid[0]}x{grid[1]} nsteps={nsteps}] "
          f"max|d|={d.max():.3e}  max_rel={rel:.3e}  max_ulp={ulp:.2f}")
    assert np.allclose(zi, zf, rtol=1e-12, atol=0.0), \
        f"in-house ADI diverged from fortran: max_rel={rel:.3e} max_ulp={ulp:.1f}"
