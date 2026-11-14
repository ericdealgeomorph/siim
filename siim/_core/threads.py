"""Numba threading for siim's parallel kernels: the layer they run on and the
thread count of the level-scheduled eroder.

``siim/__init__.py`` sets numba's layer priority (TBB, else workqueue) before
numba is imported; numba picks the layer when its thread pool launches.
No helper here changes a result: every parallel kernel is bit-for-bit at any
layer and thread count (``test_parallel_erode``).
"""

import contextlib
import ctypes
import functools
import sys

import numba


def threading_layer():
    """The numba threading layer this process runs: ``'tbb'``, ``'omp'`` or
    ``'workqueue'``. numba names it only once its thread pool has launched,
    which ``numba.get_num_threads`` does first."""
    numba.get_num_threads()
    return numba.threading_layer()


@functools.cache
def performance_cores():
    """The performance-core count of a hybrid CPU, or None where the system
    reports none.

    macOS on Apple silicon reports it as ``hw.perflevel0.physicalcpu`` (Intel
    Macs have no such key). It is read through libc's ``sysctlbyname``, which
    answers even in a sandbox that blocks the ``sysctl`` command; a failed read
    is None. Linux and Windows are None: all cores count alike."""
    if sys.platform != 'darwin':
        return None
    n = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(n))
    rc = ctypes.CDLL(None).sysctlbyname(
        b'hw.perflevel0.physicalcpu', ctypes.byref(n), ctypes.byref(size),
        None, ctypes.c_size_t(0))
    return n.value if rc == 0 and n.value > 0 else None


@contextlib.contextmanager
def level_threads(active=True):
    """Run the level-scheduled eroder (``parallel_erode``) with as many threads
    as the CPU has performance cores (a count, not a pinning; the system
    schedules them); ``active=False`` (the serial eroder) leaves it alone.

    The eroder opens one parallel region per drainage level, each a few hundred
    nodes, so its fork/join cost grows with the thread count and the slower
    efficiency cores hold every level back (ED-168: 4 threads beat 8 and 10 on
    the 101 x 101 grid, on both layers, on a 4 P + 6 E core laptop). Every
    other parallel kernel keeps the caller's count. The count is never raised
    above the caller's (``NUMBA_NUM_THREADS`` or ``numba.set_num_threads``) and
    is restored on exit."""
    p = performance_cores() if active else None
    # the serial eroder never asks numba: that would launch its thread pool
    n = None if p is None else numba.get_num_threads()
    if p is None or p >= n:
        yield
        return
    numba.set_num_threads(p)
    try:
        yield
    finally:
        numba.set_num_threads(n)
