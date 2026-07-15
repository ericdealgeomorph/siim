"""Regression tests for time-varying (nt, ny, nx) field forcing (audit B6).

Documented ``U`` and ``border_bed_uplift`` as ``(nt, ny, nx)`` arrays crashed
on the pinned stack: xsimlab slices the clock axis per step but xarray no
longer squeezes groupby, so each per-step slice keeps a leading size-1 dim and
``np.broadcast_to((1, ny, nx), (ny, nx))`` raised. The uplift path additionally
crashed inside fastscape's own BlockUplift; siim now fills that slot with a
squeeze-tolerant subclass (GlacialBlockUplift).
"""

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from siim.siim2d import siim as siim2d  # noqa: E402

_NT, _NY, _NX = 11, 21, 21


def _cfg(**overrides):
    return {
        'P': 2, 'beta': 1e-3,
        'Ko': 1e-6, 'n': 1, 'ce': 1e-4, 'nu': 2,
        'Ac': 2e-24, 'lambda_p': 5e2, 'lambda_c': 1e2, 'alpha_g': 8,
        'sliding_law': 'eff-exp',
        'zELA': 1000, 'T': 1e5,
        'Lx': 3e4, 'Ly': 3e4, 'nx': _NX, 'ny': _NY,
        'nt': _NT, 'nt_out': 6,
        'D': 1e-3, 'seed': 7,
        'boundary_status': ['fixed_value'] * 4,
        'initial_max_elevation': 500, 'noise_amplitude': 10,
        'k': 1, 'width_hack_k': 1.0, 'width_hack_p': 0.5,
        'flow_routing': 'single', 'progress_bar': False,
        **overrides,
    }


def test_time_varying_U_field_runs_and_matches_scalar():
    """A uniform (nt, ny, nx) U must run to completion and, being uniform,
    reproduce the equivalent scalar-U run exactly (i.e. the uplift is applied,
    not silently dropped)."""
    m_scalar = siim2d(_cfg(U=1e-3))
    m_scalar.run()

    m_field = siim2d(_cfg(U=np.full((_NT, _NY, _NX), 1e-3)))
    m_field.run()

    assert np.all(np.isfinite(m_field.z_out[-1]))
    assert np.array_equal(m_field.z_out[-1], m_scalar.z_out[-1]), \
        "uniform (nt,ny,nx) U diverges from scalar U -> uplift not applied"


def test_time_varying_border_bed_uplift_field_runs():
    """A (nt, ny, nx) border_bed_uplift must run to completion (mode-B default)."""
    m = siim2d(_cfg(U=1e-3, border_bed_uplift=np.full((_NT, _NY, _NX), 5e-4)))
    m.run()
    assert np.all(np.isfinite(m.z_out[-1]))
    assert np.all(np.isfinite(m.H_out[-1]))
