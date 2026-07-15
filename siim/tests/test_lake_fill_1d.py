"""Bit-for-bit gate for the njit'd 1D lake fill (audit m55).

``siim1d.lake_fill_1d`` — on the default mode-B hot path (~33% of the step) —
moved its two monotone-fill loops into ``_core.skeleton._lake_fill_1d``
(@njit, cache=True; measured ~80x faster). This pins the njit kernel bit-for-bit
against the original pure-Python loops across representative beds, including an
autogenic-cycling mode-B run.
"""
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from siim._core.skeleton import _lake_fill_1d           # noqa: E402
from siim.siim1d import siim as siim1d                  # noqa: E402
from siim.forcing import ela_sawtooth                   # noqa: E402


def _ref_fill(zb, didx_l, didx_r, nx):
    """The original pure-Python monotone fill (both flanks)."""
    zb = zb.copy()
    if didx_l >= 0:
        for i in range(1, didx_l + 1):
            if zb[i] < zb[i - 1]:
                zb[i] = zb[i - 1]
    if didx_r < nx:
        for i in range(nx - 2, didx_r - 1, -1):
            if zb[i] < zb[i + 1]:
                zb[i] = zb[i + 1]
    return zb


def test_m55_lake_fill_1d_bit_for_bit_random():
    rng = np.random.default_rng(1)
    for _ in range(200):
        nx = int(rng.integers(10, 600))
        zb = rng.normal(0.0, 100.0, nx)                 # rough bed with basins
        didx_l = int(rng.integers(-1, nx))              # incl. one-sided flanks
        didx_r = didx_l + 1 if didx_l >= 0 else 0
        got = zb.copy()
        _lake_fill_1d(got, didx_l, didx_r, nx)
        np.testing.assert_array_equal(got, _ref_fill(zb, didx_l, didx_r, nx))


def test_m55_lake_fill_1d_bit_for_bit_autogenic_beds():
    """The kernel matches the reference on beds from a live autogenic-cycling
    mode-B run (overdeepened, non-monotone flanks — the case the fill exists
    for)."""
    nt, T = 600, 3e5
    _, zELA = ela_sawtooth(T, nt, ela_high=500, ela_low=200,
                           period=1e5, buildup_frac=0.88)
    m = siim1d(dict(
        U=1e-3, zELA=zELA, beta=1e-2, P=1, Ko=1e-6, n=1, ce=5e-5, nu=2,
        L=3e4, dx=150, xo=50, T=T, nt=nt, nt_out=None,
        sliding_law='coulomb', tau_c=1.2e5, k=0.65,
        left_bc='base_level', right_bc='reflecting',
        mode='bedrock+ice_thickness', cap_ice_accumulation=False,
        progress_bar=False))
    m.run()
    for frame in range(0, m.zb_out.shape[1], 40):
        zb = m.zb_out[:, frame]
        got = zb.copy()
        _lake_fill_1d(got, m.didx_l, m.didx_r, m.nx)
        np.testing.assert_array_equal(got, _ref_fill(zb, m.didx_l, m.didx_r, m.nx))
