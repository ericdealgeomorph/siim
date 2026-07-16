"""The ``parallel_erode`` toggle — level-scheduled parallel mode-B erosion.

The parallel eroders (:mod:`siim._core.eroders` ``_erode_modeb_*_levels``)
bucket the flow graph into topological levels (every node's receivers live in
strictly lower levels) and run each level under ``prange``: writes are
disjoint and the per-node arithmetic replicates the serial eroders
expression-for-expression, so the toggle must be BIT-FOR-BIT with the serial
walk at any thread count. That exact equality is the whole contract — pin it
over a multi-step glaciating run for both routings and both nonlinear laws.
"""
import numpy as np
import pytest

from siim.siim2d import siim as siim2d


@pytest.fixture(autouse=True)
def _run_under_both_drivers(both_drivers):
    """S3 (Map 4 §1 PARAM): every test in this file runs under BOTH drivers --
    the conftest ``both_drivers`` fixture patches ``constants.DRIVER_DEFAULT``,
    so the existing assertions gate the in-house driver too."""



def _cfg(**ov):
    """Glaciating mode-C config (mirrors test_mode_c / test_dinf_routing)."""
    base = dict(
        U=1e-3, zELA=200, beta=1e-2, P=1, alpha_g=8, Ko=2e-6, n=1, ce=1e-4,
        nu=2, sliding_law='power', lambda_p=500, k=0.9,
        T=6e4, nt=61, nt_out=13, Lx=2e4, Ly=2e4, nx=41, ny=41, seed=7,
        boundary_status=['fixed_value'] * 4, initial_max_elevation=800,
        progress_bar=False)
    base.update(ov)
    return base


@pytest.mark.parametrize("routing", ['single', 'dinf'])
@pytest.mark.parametrize("law", ['power', 'coulomb'])
def test_parallel_erode_bit_for_bit(routing, law):
    cfg = _cfg(flow_routing=routing)
    if law == 'coulomb':
        cfg.update(sliding_law='coulomb', tau_c=1.2e5, k=0.65)
    # Both arms explicit: parallel_erode defaults ON (constants.PARALLEL_ERODE),
    # so the serial reference must be pinned, not left to the default.
    ref = siim2d(dict(cfg, parallel_erode=False)); ref.run()
    par = siim2d(dict(cfg, parallel_erode=True)); par.run()
    assert ref.H_out.max() > 10.0, "precondition: run must glaciate"
    for name in ('zb_out', 'z_out', 'H_out', 'denudation_out',
                 'erosion_rate_out'):
        np.testing.assert_array_equal(
            getattr(par, name), getattr(ref, name),
            err_msg=f"{name} diverged from the serial eroder "
                    f"(routing={routing}, law={law})")
