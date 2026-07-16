"""Shared pytest fixtures for the siim suite.

``both_drivers`` (standalone-migration scaffold, S3) parametrizes the 2D
time-loop backend so the 13 Map-4 §1 PARAM files run their existing
bit-for-bit/behavioral assertions under BOTH orchestrators:

* ``'inhouse'`` — the standalone default (PIP; runs everywhere).
* ``'xsimlab'`` — the optional fastscape/xsimlab adapter. This arm carries the
  ``adapter`` marker (applied on the param, so every fanned test id like
  ``[xsimlab]`` is marked) and an importorskip guard: in CI it runs only in
  the conda ``test-adapter`` job (``-m adapter``), and it skips cleanly in a
  bare pip venv.

A test (or file, via a one-line autouse fixture) opts in by requesting the
fixture; its value is the backend key. ``constants.DRIVER_DEFAULT`` is
monkeypatched (read at ``run()`` call time; an explicit ``driver=`` kwarg
still wins).

``both_routers`` (S4) was DELETED at S5 with the fortran router arm it
parametrized: no test ever opted in whole-file (checkpoint ruling — the two
routers were not bitwise-equivalent, so pinned-value files would have broken
spuriously), and its ``'fortran'`` param is no longer a legal backend. The
in-house router coverage lives in the explicit gate files
(test_d8_receiver_parity / test_d8_dinf_cardinal_parity /
test_analytical_oracle_router / test_router_contract) + the reference battery.
"""
import pytest


@pytest.fixture(params=[
    pytest.param('inhouse'),
    pytest.param('xsimlab', marks=pytest.mark.adapter),
])
def both_drivers(request, monkeypatch):
    """Driver-backend selector (S3). Parametrizes over siim's in-house driver
    (:mod:`siim._core.driver`, the standalone default) and the xsimlab-
    orchestrated adapter loop (``adapter``-marked arm, conda-only),
    monkeypatching ``constants.DRIVER_DEFAULT`` so every ``.run()`` inside the
    test resolves to the selected backend. Returns the backend key."""
    if request.param == 'xsimlab':
        pytest.importorskip('xsimlab')
        pytest.importorskip('fastscape')
    from siim import constants
    monkeypatch.setattr(constants, 'DRIVER_DEFAULT', request.param)
    return request.param
