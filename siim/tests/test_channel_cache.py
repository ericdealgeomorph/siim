"""Regression test for the profile-channel cache invalidation on re-run (audit B9).

run() must clear the cache the plotting layer actually reads
(``_profile_channel`` / ``_profile_channel_key`` in plotting/profiles.py),
so re-running the same model instance re-extracts the channel instead of
re-plotting the previous run's data.
"""

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from siim.siim2d import siim as siim2d  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _run_under_both_drivers(both_drivers):
    """S3 (Map 4 §1 PARAM): every test in this file runs under BOTH drivers --
    the conftest ``both_drivers`` fixture patches ``constants.DRIVER_DEFAULT``,
    so the existing assertions gate the in-house driver too."""



def _cache_config(**overrides):
    return {
        'U': 1e-3, 'P': 2, 'beta': 1e-3,
        'Ko': 1e-6, 'n': 1, 'ce': 1e-4, 'nu': 2,
        'Ac': 2e-24, 'lambda_p': 5e2, 'lambda_c': 1e2, 'alpha_g': 8,
        'sliding_law': 'eff-exp',
        'zELA': 1000, 'T': 2e5,
        'Lx': 5e4, 'Ly': 5e4, 'nx': 41, 'ny': 41,
        'nt': 101, 'nt_out': 11,
        'D': 1e-3, 'seed': 111,
        'boundary_status': ['fixed_value'] * 4,
        'initial_max_elevation': 500, 'noise_amplitude': 10,
        'k': 1, 'width_hack_k': 1.0, 'width_hack_p': 0.5,
        'flow_routing': 'single', 'progress_bar': False,
        **overrides,
    }


def test_channel_cache_invalidated_on_rerun():
    """Re-running the same instance must invalidate the profile-channel cache
    and re-extract on the next plot, rather than serve the first run's channel."""
    m = siim2d(_cache_config())
    m.run()

    # Populate the cache via the exact path the plotting layer uses.
    ch1 = m.plot._get_channel(ref=-1, basin_rank=0)
    assert m._profile_channel is ch1
    assert m._profile_channel_key == (-1, 0)
    z_first = np.array(ch1.z, copy=True)

    # Re-run the SAME instance with a different initial-noise seed -> a
    # different drainage network -> a genuinely different channel.
    m.seed = 222
    m.run()

    # The fix: the cache the plotting layer reads is cleared by run().
    assert m._profile_channel is None
    assert m._profile_channel_key is None

    # Next extraction rebuilds from the NEW run, not the stale object.
    ch2 = m.plot._get_channel(ref=-1, basin_rank=0)
    assert ch2 is not ch1
    assert not np.array_equal(ch2.z, z_first), \
        "re-extracted channel is identical to the first run's -> stale cache"
