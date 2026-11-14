"""The in-house flow-routing backend for the standalone driver (S4).

:class:`InhouseRouter` is a per-step callable the driver injects as
``cfg.route`` (``with router as route:`` context-manager protocol) producing
the router-contract bundle
(``stack, receivers, nb_receivers, weights, lengths, basin``) with NO fortran /
xsimlab dependency — the numba D8 single-flow producer (:func:`route_d8`) under
``routing='single'``, the Tarboton D-inf producer (:func:`route_dinf`) under
``'dinf'`` and the multiple-flow-direction producer (:func:`route_mfd`) under
``'mfd'``. The standalone default (and, since the 0.9.1 flip, only) backend;
the retired fortran ``FortranRouter`` counterpart followed the same protocol.

``numpy``-only; imports only from :mod:`siim._core.step` and
:mod:`siim.constants`, so the standalone driver stays framework-free. See
``docs/guides/concepts.md`` for the public routing overview.
"""
import numpy as np

from .. import constants as _constants
from ..constants import (MFD_EXPONENT as _MFD_EXPONENT,
                         resolve_mfd_exponent as _resolve_mfd_exponent,
                         INNER_PASSES as _INNER_PASSES,
                         INNER_RELAX as _INNER_RELAX)
from .step import route_d8, route_dinf, route_mfd

# Routers whose ``stack`` is DONOR-FIRST (donors precede their receivers, so a
# forward walk runs head->outlet). D8 single-flow is the odd one out: its stack
# is outlet-first and must be walked in reverse. Consumers that dispatch on
# router IDENTITY rather than on receiver arity -- e.g. the Strahler-order walk
# in ``siim2d.channel_statistics`` -- read this set, so a new multi-receiver
# router must be added here or its walk silently collapses to order 1.
DONOR_FIRST_ROUTERS = frozenset({'dinf', 'mfd'})

# Routers whose multi-receiver reduction needs the gamma effective-slope
# normalisation (``constants.MFD_GAMMA_FLOOR``) -- i.e. every MULTI-receiver
# router. D8 is the only exclusion: one receiver has gamma == 1 identically, so
# the flag is a no-op there. The single source for the `slope_normalise=None`
# default resolved by ``siim.siim2d.siim`` (and mirrored on the adapter's
# receiver arity); an explicit user value always wins, except that MFD without
# the normalisation is rejected (17-24 % low by construction at p = 1).
#
# D-inf JOINED this set on 2026-09-05 (Eric's call, docs/dev/mfd_router_plan.md
# section 6 open decision 1, recorded in docs/dev/dinf_slope_normalisation.md):
# its measured 7.6 % mid-facet projection bias (up to +5 % H, period-pi/4
# texture) is a discretisation artefact, not physics. That flip CHANGES D-inf
# results wherever flow is non-cardinal; the old behaviour is reachable with
# ``slope_normalise=False``.
SLOPE_NORMALISED_ROUTERS = frozenset({'dinf', 'mfd'})

# Every routing kind the in-house producers accept.
ROUTING_KINDS = ('single', 'dinf', 'mfd')


def _require_square(knob, dx, dy, why):
    """Raise ``ValueError`` for a knob that is only defined on square cells;
    ``dx``/``dy`` left ``None`` are not checked."""
    if dx is not None and dy is not None and float(dx) != float(dy):
        raise ValueError(f"{knob} requires square cells (dx == dy): {why}. "
                         f"Got dx={dx!r}, dy={dy!r}.")


def check_mfd_config(routing, *, mode=None, closure_flux=None, dx=None,
                     dy=None, slope_normalise=None):
    """Raise ``ValueError`` for any configuration the MFD router does not
    support. A no-op for every other routing kind.

    Single-sourced so ``siim.siim2d.siim``, ``siim.fastscape.glacial_processes``
    and the adapter processes raise the SAME errors — direct
    adapter construction would otherwise bypass the model-level checks
    (docs/dev/mfd_router_plan.md §2.6). An argument left ``None`` is NOT checked:
    each caller passes only what it owns (the facade knows the mode, the router
    process knows the grid spacing, the erosion process knows the closure).

    MFD runs in modes B, C and D (mode D accepts no other routing), with the carve and the fabricated trunk
    surface, whose dip needs k >= 1 for a zero axis leak
    (:func:`resolve_trunk_dip_k`). It requires:

    * mode 'B' or 'C' — mode A has only the one-cell closure, which the next
      rule excludes;
    * ``closure_flux != 'cell'`` — MFD + the cell closure is incoherent (the
      trunk's own routed flux is half the section it should close on);
    * ``dx == dy`` — the gamma effective-slope normalisation
      (``constants.MFD_GAMMA_FLOOR``) is exact only on a square stencil; on
      rectangular cells it errs 4-8 % at p = 1;
    * ``slope_normalise`` not an explicit ``False`` — the raw MFD reduction is
      17-24 % low by construction.
    """
    if routing != 'mfd':
        return
    # mode arrives normalised ('C' -> 'B' + carve), so 'B' admits both.
    if mode is not None and str(mode).upper() not in ('B', 'C'):
        raise ValueError(
            "flow_routing='mfd' requires mode 'B' or 'C': mode 'A' has only "
            "the one-cell closure, and MFD needs a transect closure_flux. "
            f"Got mode={mode!r}.")
    if closure_flux is not None and closure_flux == 'cell':
        raise ValueError(
            "flow_routing='mfd' requires closure_flux 'transect' or "
            "'transect_min' (MFD + the cell closure is incoherent: the trunk's "
            "own routed flux is ~0.48 of the cross-section it must close on).")
    if slope_normalise is not None and not slope_normalise:
        raise ValueError(
            "flow_routing='mfd' requires slope_normalise (the raw MFD "
            "reduction is 17-24 % low by construction at p = 1). Use "
            "slope_normalise=None (the default) or True.")
    _require_square("flow_routing='mfd'", dx, dy,
                    "the gamma effective-slope normalisation is exact only on "
                    "a square 8-neighbour stencil")


def resolve_trunk_dip_k(value, routing):
    """Resolve the ``trunk_dip_k`` sentinel and return the dip as a float.

    ``None`` is router-derived: ``constants.TRUNK_DIP_K_MFD`` under ``'mfd'``
    (the axis leak is zero from k = 1), ``constants.TRUNK_DIP_K`` under
    ``'single'`` and ``'dinf'``. An explicit value wins and must be a finite
    float >= 0 (a bool is not one here; 0 is allowed, the un-dipped surface).
    """
    if value is None:
        return float(_constants.TRUNK_DIP_K_MFD if routing == 'mfd'
                     else _constants.TRUNK_DIP_K)
    k = float(value)
    if isinstance(value, (bool, np.bool_)) or not (np.isfinite(k) and k >= 0.0):
        raise ValueError("trunk_dip_k must be a finite float >= 0 (or None "
                         f"for the router-derived default), got {value!r}.")
    return k


def check_inner_loop_config(inner_passes, inner_relax, *, mode=None):
    """Validate the within-step damped-iteration knobs and return them coerced
    as ``(int n, float omega)``. Single-sourced (like :func:`check_mfd_config`)
    so ``siim.siim2d.siim`` and the ``siim.fastscape`` facade raise the SAME
    errors. ``mode`` is checked only when supplied.

    ``inner_passes`` is an integer >= 1 (a bool is not an integer here) and
    ``inner_relax`` a finite omega in (0, 1]. Non-defaults need mode 'B' or 'C'
    (mode A has no closure path to iterate; ``constants.INNER_PASSES``)."""
    n = float(inner_passes)
    if (isinstance(inner_passes, (bool, np.bool_))
            or not (np.isfinite(n) and n >= 1 and n == int(n))):
        raise ValueError(f"inner_passes must be an integer >= 1, got "
                         f"{inner_passes!r}.")
    n = int(n)
    omega = float(inner_relax)
    if (isinstance(inner_relax, (bool, np.bool_))
            or not (np.isfinite(omega) and 0.0 < omega <= 1.0)):
        raise ValueError(f"inner_relax (omega) must be in (0, 1], got "
                         f"{inner_relax!r}.")
    default = (n, omega) == (_INNER_PASSES, _INNER_RELAX)
    # mode arrives normalised (C -> B, as for check_mfd_config); the message
    # names both because both are what the user typed.
    if not default and mode is not None and str(mode).upper() not in ('B', 'C'):
        raise ValueError(
            "inner_passes > 1 / inner_relax < 1 require mode 'B' or 'C' (mode "
            "'A' has no glacial closure path to iterate). Got "
            f"mode={mode!r}.")
    return n, omega


def check_routing_filter_config(routing_filter_factor, *, mode=None,
                                trunk_surface=None, dx=None, dy=None,
                                regime_switch=False):
    """Validate the routing-surface spatial filter factor and return it as a
    float. Sibling of :func:`check_inner_loop_config`, single-sourced for the
    same reason; an argument left ``None`` is not checked.

    ``f`` is a finite, nonnegative float (a bool is not one here). A positive f
    additionally needs mode 'B' or 'C' (mode A has no reconstructed ice surface
    to filter), ``trunk_surface=False`` outside mode D (there the fabricated
    trunk surface is the opposite-regime routing-surface device and the two are
    never combined; mode D's valley <-> sheet map fades each where the other
    applies — ``constants.HC_SHEET``),
    and square cells (the window is sized in cells from dx alone). Windows wrap
    on looped axes, so filtering commutes with a seam shift
    (``docs/dev/sheet_knobs_plan_review_astra.md`` §(c))."""
    f = float(routing_filter_factor)
    if (isinstance(routing_filter_factor, (bool, np.bool_))
            or not (np.isfinite(f) and f >= 0.0)):
        raise ValueError("routing_filter_factor must be a finite float >= 0, "
                         f"got {routing_filter_factor!r}.")
    if f == 0.0:
        return f
    # mode arrives normalised ('C' -> 'B'), as for check_inner_loop_config.
    if mode is not None and str(mode).upper() not in ('B', 'C'):
        raise ValueError(
            "routing_filter_factor > 0 requires mode 'B' or 'C' (mode 'A' has "
            f"no reconstructed ice surface to filter). Got mode={mode!r}.")
    if trunk_surface and not regime_switch:
        raise ValueError(
            "routing_filter_factor > 0 requires trunk_surface=False (or "
            "mode='D', which fades each device where the other applies): the "
            "fabricated trunk surface is the other routing-surface device, "
            "built for the opposite (valley) regime.")
    _require_square("routing_filter_factor > 0", dx, dy,
                    "the filter window is sized in cells from dx alone")
    return f


class RoutingArrays:
    """The router-contract array bundle the accumulator + kernels + carve
    consume: ``stack`` (outlet-first for single-flow routing, donor-first for
    D-inf and MFD), ``receivers`` (1D single-flow / ``(n, 2)`` D-inf /
    ``(n, 8)`` MFD),
    ``nb_receivers``, ``weights``, ``lengths``, and the diagnostic ``basin``
    ``(ny, nx)`` int — handed over either as the array or as a zero-argument
    callable run on the first read (the multi-receiver routers' D8 labelling
    pass, which the driver reads only on the steps a frame saves). The duck type
    any routing backend hands the driver (the router-contract seam)."""

    __slots__ = ("stack", "receivers", "nb_receivers", "weights", "lengths",
                 "_basin")

    def __init__(self, stack, receivers, nb_receivers, weights, lengths, basin):
        self.stack = stack
        self.receivers = receivers
        self.nb_receivers = nb_receivers
        self.weights = weights
        self.lengths = lengths
        self._basin = basin

    @property
    def basin(self):
        if callable(self._basin):
            self._basin = self._basin()
        return self._basin


class InhouseRouter:
    """Framework-free in-house router for the standalone driver.

    Produces, per step, the router-contract bundle via the in-house D8
    single-flow producer (``routing='single'``), the in-house Tarboton D-inf
    producer (``'dinf'``) or the multiple-flow-direction producer (``'mfd'``) —
    no fortran. Used as a context manager (the backend protocol; there is no
    context to set up / tear down here, so ``__enter__`` / ``__exit__`` are
    no-ops)::

        with InhouseRouter(shape, border_status, 'single', dx, dy) as route:
            for k in range(nt - 1):
                r = route(elevation)   # RoutingArrays(stack, receivers, ...)
    """

    def __init__(self, shape, border_status, routing, dx, dy,
                 mfd_exponent=_MFD_EXPONENT):
        self.shape = (int(shape[0]), int(shape[1]))
        self.border_status = list(np.broadcast_to(border_status, 4))
        if routing not in ROUTING_KINDS:
            raise ValueError("routing must be one of "
                             f"{ROUTING_KINDS}, got {routing!r}")
        self.routing = routing
        self.dx = float(dx)
        self.dy = float(dy)
        # MFD spreading exponent (constants.MFD_EXPONENT); inert for the
        # single-receiver / D-inf branches.
        # resolved here, so a value above the bound warns once (route_mfd then
        # sees inf)
        self.mfd_exponent = _resolve_mfd_exponent(mfd_exponent)
        if routing == 'mfd':
            check_mfd_config(routing, dx=self.dx, dy=self.dy)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __call__(self, elevation):
        if self.routing == 'single':
            (receivers, weights, lengths, nb_receivers,
             stack, basin) = route_d8(
                elevation, self.shape, self.dx, self.dy, self.border_status)
        elif self.routing == 'mfd':
            (receivers, weights, lengths, nb_receivers,
             stack, basin) = route_mfd(
                elevation, self.shape, self.dx, self.dy, self.border_status,
                self.mfd_exponent, lazy_basin=True)
        else:
            (receivers, weights, lengths, nb_receivers,
             stack, basin) = route_dinf(
                elevation, self.shape, self.dx, self.dy, self.border_status,
                lazy_basin=True)
        return RoutingArrays(stack, receivers, nb_receivers, weights, lengths,
                             basin)
