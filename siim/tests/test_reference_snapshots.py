"""Reference-battery reproduce-gate — the frozen battery must reproduce bit-for-bit.

Reruns every battery config ON THE SHIPPED STANDALONE DEFAULTS (in-house driver
+ inhouse_d8 router + in-house numerics — sanctioned regeneration #4, S5) and
asserts ``np.array_equal`` vs the committed ``.npz`` in ``data/reference/`` —
under ``parallel_erode`` ON and OFF (``constants.PARALLEL_ERODE``;
serial==parallel is bit-for-bit, ``test_parallel_erode.py``). This is the
standing determinism tripwire for the standalone default path, and it is
PIP-capable (no fastscape/xsimlab import — it runs in the ``test-standalone``
CI job).

The 3 fortran-SFR router references are gated separately (``adapter`` marker,
conda): live fastscapelib-fortran is re-driven on the FROZEN surfaces and must
reproduce the raw-integer refs bit-for-bit — certifying that the immutable S4
baselines ``test_d8_receiver_parity`` (PIP) diffs against still match live
fortran.

The battery legs skip cleanly when either

  (a) the frozen ``.npz`` are not present yet — CI writes them with
      ``--if-absent`` (``capture_snapshots.py``) before the suite runs; or
  (b) the references do not match the running platform: the committed battery
      is PLATFORM-BOUND (frozen in CI on linux/x86-64; cross-platform floats
      amplify to O(1) through the autogenic dynamics — plan doc, S0 platform
      addendum). The gate runs in CI (which froze the committed refs) or when
      ``SIIM_S0_REF_DIR`` points at a battery frozen on THIS machine.

``basin`` arrays are frozen from nothing (excluded at capture); this gate never
compares basin labels (protocol rule 5).
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

if not (os.environ.get('SIIM_S0_REF_DIR') or os.environ.get('CI')):
    pytest.skip(
        "frozen reference battery is platform-bound (CI linux/x86-64 floats); "
        "freeze a local battery first — "
        "SIIM_S0_REF_DIR=siim/tests/data/reference_local python "
        "siim/tests/data/reference/capture_snapshots.py --write --if-absent — "
        "then rerun the suite with SIIM_S0_REF_DIR set",
        allow_module_level=True)

# Load the capture script by file path — it is not a declared setuptools package
# (siim.tests is test-only), so a package import is not guaranteed under strict
# editable installs. The script imports siim only inside functions, so loading it
# here is cheap and stack-free.
_CAP_PATH = Path(__file__).resolve().parent / 'data' / 'reference' / 'capture_snapshots.py'
_spec = importlib.util.spec_from_file_location('siim_s0_capture_snapshots', _CAP_PATH)
cap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cap)


def _require_frozen(key):
    path = cap.npz_path(key)
    if not path.exists():
        pytest.skip(
            f"frozen reference {key}.npz not generated yet "
            "(capture_snapshots.py --write); CI freezes it before this gate")
    return path


@pytest.mark.parametrize('parallel_erode', [True, False],
                         ids=['par_on', 'par_off'])
@pytest.mark.parametrize('key', list(cap.matrix()))
def test_reference_reproduces_bit_for_bit(key, parallel_erode):
    """Every frozen config reproduces bit-for-bit on a rerun, for both
    ``parallel_erode`` settings (the determinism + serial==parallel gate),
    on the shipped standalone default path."""
    ref = np.load(_require_frozen(key))
    got = cap.run_config(cap.matrix()[key], parallel_erode=parallel_erode)
    cap._assert_equal(key, ref, got)


@pytest.mark.adapter
@pytest.mark.parametrize('name', list(cap.ROUTER_SURFACE_NAMES))
def test_router_reference_frozen(name):
    """The frozen fortran-SFR router references reproduce BIT-FOR-BIT — surface,
    raw-integer receivers/stack, lengths, and meta — when live fortran is
    re-driven on the FROZEN surface (never a regenerated one; plan decision
    note 12). This is the conda (adapter) leg that re-certifies the immutable
    S4 baselines against LIVE fortran; the PIP parity gate
    (``test_d8_receiver_parity``) consumes the same refs without fortran."""
    pytest.importorskip('fastscapelib_fortran')
    key = cap.router_ref_key(name)
    ref = np.load(_require_frozen(key))
    got = cap.compute_router_ref(name, cap.frozen_router_surface(name))
    cap.assert_router_ref(key, ref, got)
