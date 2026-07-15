"""Smoke test for the 1D limit-cycle plotters (audit m44).

``limit_cycle`` / ``limit_cycle_phase`` (~290 lines, paper-figure-bound) had no
test coverage. This exercises construction + one call of each on a small forced
sawtooth-ELA coulomb mode-B run (persistently glaciated, so the terminus series
oscillates cleanly), asserting each returns a figure — no numerical pins.
"""
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib                                       # noqa: E402
matplotlib.use('Agg')                                  # headless
import matplotlib.pyplot as plt                        # noqa: E402

from siim.siim1d import siim as siim1d                 # noqa: E402
from siim.forcing import ela_sawtooth                  # noqa: E402


def test_m44_limit_cycle_and_phase_smoke():
    nt, T = 1000, 4e5
    # ELA band high enough that the terminus retreats inland each cycle: under
    # the zero-gradient (slaved) outflow border the boundary column carries the
    # full interior surface, so the old mild band (500-200, cap=False) kept the
    # terminus pinned at the border (Lt constant -> no cycle to detect).
    _, zELA = ela_sawtooth(T, nt, ela_high=1600, ela_low=600,
                           period=1e5, buildup_frac=0.88)
    m = siim1d(dict(
        U=1e-3, zELA=zELA, beta=1e-2, P=1, Ko=1e-6, n=1, ce=5e-5, nu=2,
        L=3e4, dx=150, xo=50, T=T, nt=nt, nt_out=None,
        sliding_law='coulomb', tau_c=1.2e5, k=0.65,
        left_bc='base_level', right_bc='reflecting',
        mode='bedrock+ice_thickness', cap_ice_accumulation=True,
        progress_bar=False))
    m.run()

    fig, info = m.plot.limit_cycle()
    assert fig is not None
    assert info['period'] > 0 and info['amplitude'] > 0
    plt.close(fig)

    fig2, res = m.plot.limit_cycle_phase()
    assert fig2 is not None
    assert np.asarray(res['x']).size > 0 and np.asarray(res['y']).size > 0
    plt.close(fig2)
