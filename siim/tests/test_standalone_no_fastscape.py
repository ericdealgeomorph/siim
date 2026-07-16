"""S5 headline gate — the standalone 2D model runs with NO fastscape/xsimlab.

Runs ONLY in a genuinely fastscape-free env (the pip-only ``test-standalone``
CI job / a fresh pip venv) — in the conda adapter env the stack is importable,
so the absence gate is vacuous there and the tests SKIP with a directed
message. Two-step failure signature (orchestrator gate addition, plan decision
note 18): the import is asserted BEFORE any model run, so a missed lazy-import
site fails the import test while a physics regression fails the run test.
"""
import importlib.util

import numpy as np
import pytest

_STACK_ABSENT = (importlib.util.find_spec('fastscape') is None
                 and importlib.util.find_spec('xsimlab') is None)

pytestmark = pytest.mark.skipif(
    not _STACK_ABSENT,
    reason="fastscape/xsimlab importable — the standalone-absence gate is "
           "vacuous here; it runs in the pip-only test-standalone job")


def test_stack_truly_absent():
    """The env contract of this gate: neither fastscape nor xsimlab resolves."""
    assert importlib.util.find_spec('fastscape') is None
    assert importlib.util.find_spec('xsimlab') is None


def test_import_siim2d_without_stack():
    """``import siim.siim2d`` (and escarpment + the adapter-free plotting layer)
    succeeds with the stack absent — every stack import in the standalone path
    is either gone or lazy."""
    import siim.siim2d          # noqa: F401
    import siim.escarpment      # noqa: F401


def test_full_modeC_run_without_stack():
    """A full default (mode C: carve + trunk-surface + routing-relax standard)
    run — in-house driver, in-house D8 router, in-house flexure — completes and
    produces finite outputs with the stack absent."""
    from siim.siim2d import siim as siim2d
    m = siim2d(dict(
        U=1e-3, zELA=150, beta=1e-2, P=2, alpha_g=12, Ko=2e-6, n=1, ce=1e-4,
        nu=2, sliding_law='power', lambda_p=500, k=0.9,
        T=3e4, nt=31, nt_out=5, Lx=2e4, Ly=2e4, nx=31, ny=31, seed=7,
        flexure=True, ice_load=True, e_thickness=20e3,
        initial_max_elevation=800, progress_bar=False))
    assert m.mode == 'B' and m.carve_width          # the mode-C default
    m.run()
    for name in ('z_out', 'zb_out', 'H_out', 'rebound_out'):
        arr = getattr(m, name)
        assert np.all(np.isfinite(arr)), name
    assert m.H_out.max() > 1.0, "config must glaciate"

    # the adapter raises its directed ImportError, not a bare ModuleNotFound
    with pytest.raises(ImportError, match="(?i)optional fastscape/xsimlab adapter"):
        import siim.fastscape   # noqa: F401
