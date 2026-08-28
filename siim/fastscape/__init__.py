"""siim.fastscape — siim's glacial processes as composable fastscape citizens.

This is the public surface for using siim's glacial-erosion physics inside your
own fastscape/xsimlab model. The names exported here are ``@xs.process`` classes
that drop into fastscape's ``basic_model`` once its stock
``'spl'`` and ``'drainage'`` slots are removed. Build the override dict with
:func:`glacial_processes`, or get a ready-to-run model from :func:`glacial_model`::

    from fastscape.models import basic_model
    from siim.fastscape import glacial_processes

    model = basic_model.drop_processes(['spl', 'drainage']).update_processes(
        glacial_processes(mode='B'))

For the full coupled model with forcing, I/O and plotting, use
:class:`siim.siim2d.siim` instead. Its default in-house driver calls the shared
framework-free numerical core directly; its optional ``driver='xsimlab'`` path
assembles these process classes. The two paths share the same law parameters
and step kernels and are kept aligned by driver-conformance tests.

Implementation lives in submodules: the glacial ``@xs.process`` classes in
:mod:`siim.fastscape.processes`, and two reusable generic forcing processes
(``WaveUplift``, ``PlateauSurface``) in :mod:`siim.fastscape.forcing`.

API stability
-------------
siim is pre-1.0 (0.x). This surface is *intended* to stay stable, but may still
evolve before 1.0 — it is not a frozen contract yet. Modes A, B and C are
settled: the sub-grid width-carving path (``carve=True`` → :class:`GlacialSPLModeC`)
is a mode-B citizen too (bed + ice thickness + carve), and is siim's flagship
carved mode.
"""
try:
    from fastscape.models import basic_model
except ImportError as e:
    raise ImportError(
        "siim.fastscape is the OPTIONAL fastscape/xsimlab adapter — the "
        "standalone 2D model (siim.siim2d) needs none of it. To use the "
        "adapter: `pip install siim[fastscape]` provides fastscape + xsimlab "
        "from PyPI, but fastscapelib-fortran (the fortran backend fastscape's "
        "stock processes call at run time) has no PyPI wheel — use the conda "
        "env instead: `conda env create -f environment.yml`."
    ) from e

from .. import constants as _constants
from .processes import (
    D8FlowRouter,           # in-house numba D8 single-flow router (S4)
    DinfFlowRouter,
    GlacialBlockUplift,
    GlacialFlexure,
    GlacialFlowAccumulator,
    GlacialLaw,
    GlacialSPLModeA,
    GlacialSPLModeB,
    GlacialSPLModeC,        # mode B + sub-grid carve, citizen semantics
    GlacialSurfaceToErode,
    HillslopeDiffusion,     # in-house ADI hillslope diffuser (numerics_backend='inhouse')
    InitialTopography,      # siim-specific forcing — internal, not exported
    SedimentTracker,
    TrunkSurfaceToErode,    # fabricated trunk-surface routing (opt-in)
)
from .forcing import PlateauSurface, WaveUplift

# The intended-stable public process surface: modes A, B and C + the auxiliary
# processes + the two generic forcing processes. GlacialSPLBase (a base class)
# and InitialTopography (siim-specific forcing — bring your own) stay internal.
__all__ = [
    "GlacialLaw",
    "GlacialFlowAccumulator",
    "GlacialSPLModeA",
    "GlacialSPLModeB",
    "GlacialSPLModeC",
    "GlacialSurfaceToErode",
    "TrunkSurfaceToErode",
    "SedimentTracker",
    "GlacialFlexure",
    "DinfFlowRouter",
    # generic fastscape forcing processes (reusable in any fastscape model)
    "WaveUplift",
    "PlateauSurface",
    "glacial_processes",
    "glacial_model",
]


def glacial_processes(*, mode='B', carve=None, routing='single',
                      router_backend=None,
                      flexure=False, sediment=False, trunk_surface=False,
                      numerics_backend=_constants.NUMERICS_BACKEND):
    """Build the slot→process-class override dict for fastscape's ``basic_model``.

    Apply the returned dict after dropping the stock fluvial slots::

        basic_model.drop_processes(['spl', 'drainage']).update_processes(
            glacial_processes(...))

    or use :func:`glacial_model`, which does exactly that. The optional
    ``driver='xsimlab'`` path of :class:`siim.siim2d.siim` builds its model from
    this function; the default in-house driver consumes the same shared law
    records and numerical kernels without xsimlab.

    This facade stays EXPLICIT: every flag defaults off, with no mode-resolved
    magic (standalone fastscape users opt in by hand). The mode-C standard —
    ``trunk_surface=True`` and ``routing_relax`` = ``constants.MODE_C_ROUTING_RELAX``
    for a carved mode-B run, ``widening_rate`` = ``constants.DEFAULT_WIDENING_RATE``
    — is applied only by the :class:`siim.siim2d.siim` wrapper.

    Parameters
    ----------
    mode : {'B', 'A', 'C'}
        One-breath taxonomy: ``'A'`` = ice-surface state (carved troughs heal
        instantly); ``'B'`` = bedrock ``zb`` + ice-thickness ``H`` state (bed
        memory; siim's native regime); ``'C'`` = B + sub-grid width carving
        (equivalent to ``mode='B', carve=True`` — implies ``carve``, and an
        explicit ``carve=False`` contradicts it). Case-insensitive.
    carve : bool, optional
        Mode-B sub-grid glacier-width carving — wires :class:`GlacialSPLModeC`
        (the citizen bed-and-thickness class plus the carve). Ignored for
        ``mode='A'``.
        Defaults off here so the advertised default is the plain Mode-B citizen.
        :class:`siim.siim2d.siim` defaults to Mode C, which enables carving;
        explicit Mode B leaves it off unless requested. ``mode='C'`` is the
        alias for turning it on.
    trunk_surface : bool, optional
        Fabricated trunk-surface routing (mode B/C) — swaps in
        :class:`TrunkSurfaceToErode` so routing + accumulation converge trunk
        flow onto the centerline (the centerline's raw flux is then the full
        cross-section). Off by default here; :class:`siim.siim2d.siim` turns it
        on as part of the mode-C standard. Ignored for ``mode='A'``.
    routing : {'single', 'dinf'}, optional
        D8 single-flow (``'single'``, default) or D-infinity (``'dinf'``).
    router_backend : {'inhouse_d8'}, optional
        Single-flow routing backend (``constants.ROUTER_DEFAULT``).
        ``'inhouse_d8'`` — siim's numba :class:`D8FlowRouter` — is the only
        accepted value since the 0.9.1 standalone flip (the fortran
        ``SingleFlowRouter`` wiring was retired); the parameter is the
        router-contract plug point for
        future backends. Only affects ``routing='single'`` — the D-inf
        directions + mask + basin ride the same in-house contract.
    flexure, sediment : bool, optional
        Opt in to glacial-isostatic flexure / sediment tracking.
    numerics_backend : {'inhouse'}, optional
        Flexure + hillslope-diffusion backend (``constants.NUMERICS_BACKEND``).
        ``'inhouse'`` — :class:`HillslopeDiffusion` (siim's numba ADI) in the
        stock ``LinearDiffusion`` slot and :class:`GlacialFlexure` on siim's
        scipy.fft plate solve — is the only accepted value since the 0.9.1
        standalone flip. Does not affect routing.

    Returns
    -------
    dict
        Maps xsimlab slot name → process class, ready for ``update_processes``.
    """
    mode = _constants.normalize_mode(str(mode).upper())  # 'A'/'B'/'C'; single-source vocab
    if mode == 'C':
        # C is the B + sub-grid-carve alias: imply carve, and reject an explicit
        # carve=False (defaulted None passes through). Not a new dynamical mode.
        if carve is False:
            raise ValueError("mode='C' is the mode-B + sub-grid-carve alias; "
                             "carve=False contradicts it (use mode='B').")
        mode, carve = 'B', True
    if routing not in ('single', 'dinf'):
        raise ValueError(f"routing must be 'single' or 'dinf', got {routing!r}")
    if numerics_backend != 'inhouse':
        raise ValueError("numerics_backend must be 'inhouse' (the retired "
                         "'fortran' backend was removed at the 0.9.1 "
                         f"standalone flip), got {numerics_backend!r}")
    if router_backend is None:
        router_backend = _constants.ROUTER_DEFAULT
    if router_backend != 'inhouse_d8':
        raise ValueError("router_backend must be 'inhouse_d8' (the retired "
                         "'fortran' SFR was removed at the 0.9.1 standalone "
                         f"flip), got {router_backend!r}")

    # Flow slot: both routings are in-house producers on the router contract.
    flow_cls = DinfFlowRouter if routing == 'dinf' else D8FlowRouter
    overrides = {
        'glacial_flow': GlacialFlowAccumulator,
        'law': GlacialLaw,
        'flow': flow_cls,
        'init_topography': InitialTopography,
        # Squeeze-tolerant BlockUplift so a documented (nt, ny, nx) uplift rate
        # doesn't crash on the pinned xarray/xsimlab stack (audit B6).
        'uplift': GlacialBlockUplift,
        # In-house hillslope diffusion (numba ADI) replaces the stock fortran
        # LinearDiffusion slot.
        'diffusion': HillslopeDiffusion,
    }
    if mode == 'A':
        overrides['glacial_spl'] = GlacialSPLModeA           # stock surf2erode
    else:
        # Mode B is a fastscape citizen (topography IS the bed) whether or not
        # it carves: ModeC = citizen + sub-grid carve, ModeB = citizen alone.
        # Both route on the reconstructed ice surface zb + hc*H.
        overrides['glacial_spl'] = GlacialSPLModeC if carve else GlacialSPLModeB
        # Routing surface: the fabricated trunk surface (flow converges to the
        # centerline) or the plain reconstructed ice surface zb + hc*H.
        overrides['surf2erode'] = (TrunkSurfaceToErode if trunk_surface
                                   else GlacialSurfaceToErode)
    if flexure:
        overrides['flexure'] = GlacialFlexure
    if sediment:
        overrides['sediment'] = SedimentTracker
    return overrides


def glacial_model(*, mode='B', carve=None, routing='single', router_backend=None,
                  flexure=False, sediment=False, trunk_surface=False,
                  numerics_backend=_constants.NUMERICS_BACKEND):
    """Return a ready-to-run xsimlab ``Model``: fastscape's ``basic_model`` with
    its ``'spl'``/``'drainage'`` slots replaced by siim's glacial processes.

    Convenience wrapper over :func:`glacial_processes` (same parameters). For the
    full coupled model with forcing, I/O and plotting, use
    :class:`siim.siim2d.siim` instead.
    """
    # The frozen xsimlab stack emits an xarray ``Dataset.drop`` deprecation
    # while composing the model. Keep third-party warning suppression local to
    # this SIIM-owned call instead of changing the process-wide warnings state.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="dropping variables using",
            category=FutureWarning, module="xsimlab")
        return (basic_model
                .drop_processes(['spl', 'drainage'])
                .update_processes(glacial_processes(
                    mode=mode, carve=carve, routing=routing,
                    router_backend=router_backend,
                    flexure=flexure, sediment=sediment,
                    trunk_surface=trunk_surface,
                    numerics_backend=numerics_backend)))
