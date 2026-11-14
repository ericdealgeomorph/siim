"""The numba threading layer siim picks, and what follows it (ED-168).

``import siim`` sets numba's layer priority to TBB, else workqueue, without
importing numba; an explicit user layer or priority wins. Under MFD routing
(mode D's default) the ``parallel_erode`` default follows the layer numba runs
(on under TBB); under another router it stays on, and an explicit value wins. The level eroder runs at the performance-core thread
count, never above the caller's. numba picks its layer once per process, so
the layer cases run in fresh ones.
"""
import os
import platform
import subprocess
import sys

import numba
import pytest

from siim._core import threads

_PROBE = """
import os, sys
import siim
assert 'numba' not in sys.modules, 'import siim imported numba'
import numba
from siim.siim2d import siim as siim2d
from siim.tests.test_mode_d import _D, _D_KNOBS
print(os.environ.get('NUMBA_THREADING_LAYER_PRIORITY'))
print(siim2d(_D).parallel_erode, numba.threading_layer())
print(siim2d(dict(_D, parallel_erode=True)).parallel_erode,
      siim2d(dict(_D, parallel_erode=False)).parallel_erode)
mfd_c = {k: v for k, v in dict(_D, mode='C', **_D_KNOBS).items() if k != 'parallel_erode'}
print(siim2d(mfd_c).parallel_erode, siim2d(dict(_D, mode='C', flow_routing='single')).parallel_erode)
"""


def _run(code, env=()):
    """``code``'s stdout from a fresh interpreter with no layer variable set
    but those in ``env``."""
    clean = {k: v for k, v in os.environ.items()
             if not k.startswith('NUMBA_THREADING_LAYER')}
    return subprocess.run([sys.executable, '-c', code], env={**clean, **dict(env)},
                          check=True, capture_output=True, text=True).stdout


@pytest.mark.pin
@pytest.mark.parametrize('env, priority, layers', [
    ({}, 'tbb workqueue omp', ('tbb', 'workqueue')),
    ({'NUMBA_THREADING_LAYER': 'workqueue'}, 'tbb workqueue omp', ('workqueue',)),
    ({'NUMBA_THREADING_LAYER_PRIORITY': 'workqueue tbb omp'}, 'workqueue tbb omp',
     ('workqueue',)),
], ids=['default', 'user-layer', 'user-priority'])
def test_layer_priority_and_mfd_parallel_erode(env, priority, layers):
    out = _run(_PROBE, env)
    got_priority, (default, layer), explicit, by_router = [
        line.split() for line in out.strip().splitlines()[-4:]]
    assert ' '.join(got_priority) == priority
    assert layer in layers
    assert default == str(layer == 'tbb')     # the default follows the layer
    assert explicit == ['True', 'False']      # an explicit value wins
    # it is MFD's default, not mode D's: mode C under MFD follows the layer
    # too, and mode C on single-flow routing keeps the parallel eroder (mode D
    # refuses single flow since 2026-11-13)
    assert by_router == [default, 'True']


def test_the_priority_reaches_a_numba_imported_first():
    # numba reads its environment at import; its own priority puts OpenMP
    # ahead of workqueue
    out = _run("import numba, siim; print(*numba.config.THREADING_LAYER_PRIORITY)")
    assert out.strip().splitlines()[-1] == 'tbb workqueue omp'
    # and what the user then sets on numba.config survives numba's next compile
    # (numba re-reads its configuration there whenever the environment changed)
    out = _run("import numba, siim\n"
               "numba.config.THREADING_LAYER = 'workqueue'\n"
               "numba.njit(lambda x: x + 1)(1)\n"
               "print(numba.config.THREADING_LAYER)")
    assert out.strip().splitlines()[-1] == 'workqueue'


@pytest.mark.skipif(sys.platform != 'darwin' or platform.machine() != 'arm64',
                    reason='only Apple silicon reports a performance-core count')
def test_performance_cores_reads_the_real_count():
    assert 0 < threads.performance_cores() <= os.cpu_count()


def test_level_threads_caps_and_restores(monkeypatch):
    n = numba.get_num_threads()
    if n < 2:
        pytest.skip('needs two numba threads')
    monkeypatch.setattr(threads, 'performance_cores', lambda: n - 1)
    with threads.level_threads():
        assert numba.get_num_threads() == n - 1
    assert numba.get_num_threads() == n
    with threads.level_threads(False):        # the serial eroder: untouched
        assert numba.get_num_threads() == n
    with monkeypatch.context() as patch:      # and numba is not asked (that starts its pool)
        patch.setattr(numba, 'get_num_threads', lambda: pytest.fail('asked numba'))
        with threads.level_threads(False):
            pass
        patch.setattr(threads, 'performance_cores', lambda: None)
        with threads.level_threads():         # nor where no P-core count exists
            pass
    with pytest.raises(RuntimeError), threads.level_threads():
        raise RuntimeError
    assert numba.get_num_threads() == n       # restored on an exception
    monkeypatch.setattr(threads, 'performance_cores', lambda: n + 1)
    with threads.level_threads():             # never above the caller's count
        assert numba.get_num_threads() == n
