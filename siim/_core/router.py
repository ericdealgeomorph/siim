"""The in-house flow-routing backend for the standalone driver (S4).

:class:`InhouseRouter` is a per-step callable the driver injects as
``cfg.route`` (``with router as route:`` context-manager protocol) producing
the router-contract bundle
(``stack, receivers, nb_receivers, weights, lengths, basin``) with NO fortran /
xsimlab dependency — the numba D8 single-flow producer (:func:`route_d8`) under
``routing='single'`` and the Tarboton D-inf producer (:func:`route_dinf`) under
``'dinf'``. The standalone default (and, since the 0.9.1 flip, only) backend;
the retired fortran ``FortranRouter`` counterpart followed the same protocol.

``numpy``-only; imports only from :mod:`siim._core.step`, so the standalone
driver stays framework-free (Map 3 §3-§4, ``docs/dev/router_contract.md``).
"""
import numpy as np

from .step import route_d8, route_dinf


class RoutingArrays:
    """The router-contract array bundle the accumulator + kernels + carve
    consume: ``stack`` (outlet-first), ``receivers`` (1D SFR / ``(n, 2)`` D-inf),
    ``nb_receivers``, ``weights``, ``lengths``, and the diagnostic ``basin``
    ``(ny, nx)`` int. The duck type any routing backend hands the driver
    (the router-contract seam)."""

    __slots__ = ("stack", "receivers", "nb_receivers", "weights", "lengths",
                 "basin")

    def __init__(self, stack, receivers, nb_receivers, weights, lengths, basin):
        self.stack = stack
        self.receivers = receivers
        self.nb_receivers = nb_receivers
        self.weights = weights
        self.lengths = lengths
        self.basin = basin


class InhouseRouter:
    """Framework-free in-house router for the standalone driver.

    Produces, per step, the router-contract bundle via the in-house D8
    single-flow producer (``routing='single'``) or the in-house Tarboton D-inf
    producer (``'dinf'``) — no fortran. Used as a context manager (the backend
    protocol; there is no context to set up / tear down here, so ``__enter__``
    / ``__exit__`` are no-ops)::

        with InhouseRouter(shape, border_status, 'single', dx, dy) as route:
            for k in range(nt - 1):
                r = route(elevation)   # RoutingArrays(stack, receivers, ...)
    """

    def __init__(self, shape, border_status, routing, dx, dy):
        self.shape = (int(shape[0]), int(shape[1]))
        self.border_status = list(np.broadcast_to(border_status, 4))
        if routing not in ('single', 'dinf'):
            raise ValueError(f"routing must be 'single' or 'dinf', got {routing!r}")
        self.routing = routing
        self.dx = float(dx)
        self.dy = float(dy)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __call__(self, elevation):
        if self.routing == 'single':
            (receivers, weights, lengths, nb_receivers,
             stack, basin) = route_d8(
                elevation, self.shape, self.dx, self.dy, self.border_status)
        else:
            (receivers, weights, lengths, nb_receivers,
             stack, basin) = route_dinf(
                elevation, self.shape, self.dx, self.dy, self.border_status)
        return RoutingArrays(stack, receivers, nb_receivers, weights, lengths,
                             basin)
