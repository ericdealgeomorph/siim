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
        "adapter: `pip install siim-lem[fastscape]` provides fastscape + xsimlab "
        "from PyPI, but fastscapelib-fortran (the fortran backend fastscape's "
        "stock processes call at run time) has no PyPI wheel — use the conda "
        "env instead: `conda env create -f environment.yml`."
    ) from e

from .. import constants as _constants
from .._core.router import (check_inner_loop_config as _check_inner_loop_config,
                            check_mfd_config as _check_mfd_config,
                            check_routing_filter_config
                            as _check_routing_filter_config)
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
    InitialTopography,      # initial surface + tie-breaking noise (fixed edges kept clean)
    MfdFlowRouter,          # multiple-flow-direction router
    SedimentTracker,
    TrunkSurfaceToErode,    # fabricated trunk-surface routing (the mode-C standard)
    _MfdTrunkSurfaceToErode,  # its MFD-dip-default variant (routing='mfd')
)
from .forcing import PlateauSurface, WaveUplift


def _refuse_sheet_knobs(inner_passes, inner_relax, routing_filter_factor):
    """Validate the sheet knobs, then reject any NON-default: this adapter
    routes once per model step through the process graph, so there is no seam
    for a second pass and none for a separately filtered router surface.
    Standalone-only (``siim.siim2d.siim`` on its default in-house driver).
    Adapter policy, so it lives here and not in ``_core``."""
    n, omega = _check_inner_loop_config(inner_passes, inner_relax)
    if (n, omega) != (_constants.INNER_PASSES, _constants.INNER_RELAX):
        raise ValueError(
            "inner_passes / inner_relax are STANDALONE-only (the in-house "
            "driver iterates the surf2erode -> route -> accumulate -> kernel "
            "chain within a step; the xsimlab adapter routes once per step "
            f"through the process graph). Got inner_passes={inner_passes!r}, "
            f"inner_relax={inner_relax!r}; use driver='inhouse'. (siim2d "
            f"damps modes B, C and D with inner_damping="
            f"{_constants.MODE_BCD_INNER_DAMPING!r} by default; pass "
            f"inner_passes={_constants.INNER_PASSES}, "
            f"inner_relax={_constants.INNER_RELAX} for the model this adapter "
            "runs.)")
    if _check_routing_filter_config(routing_filter_factor) != \
            _constants.ROUTING_FILTER_FACTOR:
        raise ValueError(
            "routing_filter_factor is STANDALONE-only (the filtered surface "
            "feeds the router alone, and no xsimlab surf2erode provider builds "
            "it: the adapter's router-only surface is the fabricated trunk V). "
            f"Got routing_filter_factor={routing_filter_factor!r}; "
            "use driver='inhouse'.")


# The intended-stable public process surface: modes A, B and C + the auxiliary
# processes + the two generic forcing processes + every process the helpers wire
# in, so a model can be composed by hand from this surface alone. GlacialSPLBase
# (a base class) stays internal.
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
    "MfdFlowRouter",
    # the rest of what glacial_processes wires in
    "D8FlowRouter",
    "GlacialBlockUplift",
    "HillslopeDiffusion",
    "InitialTopography",
    # generic fastscape forcing processes (reusable in any fastscape model)
    "WaveUplift",
    "PlateauSurface",
    "glacial_processes",
    "glacial_model",
]


def glacial_processes(*, mode='B', carve=None, routing='single',
                      router_backend=None,
                      flexure=False, sediment=False, trunk_surface=None,
                      numerics_backend=_constants.NUMERICS_BACKEND,
                      inner_passes=_constants.INNER_PASSES,
                      inner_relax=_constants.INNER_RELAX,
                      routing_filter_factor=_constants.ROUTING_FILTER_FACTOR):
    """Build the slot→process-class override dict for fastscape's ``basic_model``.

    Apply the returned dict after dropping the stock fluvial slots::

        basic_model.drop_processes(['spl', 'drainage']).update_processes(
            glacial_processes(...))

    or use :func:`glacial_model`, which does exactly that. The optional
    ``driver='xsimlab'`` path of :class:`siim.siim2d.siim` builds its model from
    this function; the default in-house driver consumes the same shared law
    records and numerical kernels without xsimlab.

    Every flag defaults off, except that mode C (``mode='C'``, or ``mode='B'``
    with ``carve=True``) resolves ``trunk_surface`` to the mode-C standard
    exactly as :class:`siim.siim2d.siim` does
    (:func:`siim.constants.mode_c_standard`), so ``glacial_model(mode='C')``
    equals siim2d mode C run with ``inner_passes=1, inner_relax=1.0`` (undamped).
    siim2d's default damping (``inner_damping='adaptive'``) and its
    ``'omega'`` passes need a commit this facade cannot run (one routing pass
    per model step).

    Parameters
    ----------
    mode : {'B', 'A', 'C'}
        (``'D'`` raises: mode D is standalone-only.) One-breath taxonomy: ``'A'`` = ice-surface state (carved troughs heal
        instantly); ``'B'`` = bedrock ``zb`` + ice-thickness ``H`` state (bed
        memory; siim's native regime); ``'C'`` = B + sub-grid width carving
        (equivalent to ``mode='B', carve=True`` — implies ``carve``, and an
        explicit ``carve=False`` contradicts it). Case-insensitive; the
        legible names ``'ice_surface'`` (A) and ``'bedrock+ice_thickness'`` (B)
        are accepted too.
    carve : bool, optional
        Mode-B sub-grid glacier-width carving — wires :class:`GlacialSPLModeC`
        (the citizen bed-and-thickness class plus the carve). Ignored for
        ``mode='A'``.
        Defaults off here so the advertised default is the plain Mode-B citizen.
        :class:`siim.siim2d.siim` defaults to Mode C, which enables carving;
        explicit Mode B leaves it off unless requested. ``mode='C'`` is the
        alias for turning it on.
    trunk_surface : bool or None, optional
        Fabricated trunk-surface routing (mode B/C) — swaps in
        :class:`TrunkSurfaceToErode`, whose V builds the router graph alone
        (FIREWALL: see ``siim._core.driver``), so flow converges onto the
        centerline (the centerline's raw flux is then the full cross-section).
        ``None`` (default) resolves to the mode-C standard: on for mode C, off
        for plain mode B. An explicit ``True`` / ``False`` wins. Ignored for
        ``mode='A'``. The default of the dip input ``surf2erode__trunk_dip_k``
        follows ``routing`` (see :func:`siim._core.router.resolve_trunk_dip_k`);
        an explicit value wins.
    routing : {'single', 'multiple', 'infinite'}, optional
        D8 single-flow (``'single'``, default), multiple-flow-direction
        (``'multiple'``, code ``'mfd'``) or D-infinity (``'infinite'``, code
        ``'dinf'``); the codes are accepted as aliases. ``'mfd'`` raises
        for ``mode='A'`` and runs in modes B and C, with the carve and the
        trunk surface; the remaining guards (a transect
        ``closure_flux``, ``dx == dy``) are enforced by the processes that own
        those inputs (:class:`GlacialSPLBase`, :class:`MfdFlowRouter`). Set the
        spreading exponent through ``flow__mfd_exponent``. The gamma
        effective-slope normalisation (``glacial_spl__slope_normalise``) is
        router-derived on both sides of the seam: its ``None`` default resolves
        ON for any MULTI-receiver router (``'dinf'`` since 2026-09-05, and
        ``'mfd'``) and OFF for ``'single'``, matching
        :data:`siim._core.router.SLOPE_NORMALISED_ROUTERS`. Pass ``False``
        explicitly for the pre-2026-09-05 unnormalised D-inf reduction.
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
    inner_passes, inner_relax : int, float, optional
        Accepted only at their defaults (``constants.INNER_PASSES`` = 1,
        ``constants.INNER_RELAX`` = 1.0). The within-step damped iteration is
        STANDALONE-only — xsimlab routes once per model step through the process
        graph, so there is no seam for a second pass. They exist here (and are
        forwarded by :class:`siim.siim2d.siim`'s ``driver='xsimlab'`` path) so a
        non-default is REFUSED rather than silently dropped.
    routing_filter_factor : float, optional
        Accepted only at its default (``constants.ROUTING_FILTER_FACTOR`` = 0).
        The routing-surface spatial filter is STANDALONE-only: no process here
        builds the filtered router surface (the adapter's ``router_surface``
        seam carries the trunk V alone).

    Returns
    -------
    dict
        Maps xsimlab slot name → process class, ready for ``update_processes``.
    """
    mode = _constants.normalize_mode(mode)   # -> 'A'/'B'/'C'/'D' (single-source vocab)
    if mode == 'D':
        raise ValueError(
            "mode 'D' (the channel/sheet reading choosing valley or sheet rules) is "
            "STANDALONE-only: the xsimlab process graph carries no per-cell "
            "datum row; use siim2d with driver='inhouse'.")
    if mode == 'C':
        # C is the B + sub-grid-carve alias: imply carve, and reject an explicit
        # carve=False (defaulted None passes through). Not a new dynamical mode.
        if carve is False:
            raise ValueError("mode='C' is the mode-B + sub-grid-carve alias; "
                             "carve=False contradicts it (use mode='B').")
        mode, carve = 'B', True
    if trunk_surface is None:
        # The mode-C standard, resolved by siim2d's rule (explicit value wins).
        trunk_surface = _constants.mode_c_standard(mode == 'B' and bool(carve))
    routing = _constants.normalize_flow_routing(routing, 'routing')
    # MFD coherence guards, single-sourced with siim2d and the processes: this
    # facade owns the mode (the direct-adapter bypass the review flagged).
    # ``mode`` is already resolved ('C' -> 'B' + carve above).
    _check_mfd_config(routing, mode=mode)
    # The within-step damped iteration has no xsimlab seam (one routing pass per
    # model step): validate, then refuse any non-default at BOTH entry points —
    # here for a direct glacial_processes() call, and via siim2d's
    # _process_overrides for a configured run(driver='xsimlab').
    _refuse_sheet_knobs(inner_passes, inner_relax, routing_filter_factor)
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

    # Flow slot: all three routings are in-house producers on the router contract.
    flow_cls = {'dinf': DinfFlowRouter, 'mfd': MfdFlowRouter}.get(
        routing, D8FlowRouter)
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
        # centerline; under MFD the variant with the MFD dip default) or the
        # plain reconstructed ice surface zb + hc*H.
        trunk_cls = (_MfdTrunkSurfaceToErode if routing == 'mfd'
                     else TrunkSurfaceToErode)
        overrides['surf2erode'] = (trunk_cls if trunk_surface
                                   else GlacialSurfaceToErode)
    if flexure:
        overrides['flexure'] = GlacialFlexure
    if sediment:
        overrides['sediment'] = SedimentTracker
    return overrides


def glacial_model(*, mode='B', carve=None, routing='single', router_backend=None,
                  flexure=False, sediment=False, trunk_surface=None,
                  numerics_backend=_constants.NUMERICS_BACKEND):
    """Return a ready-to-run xsimlab ``Model``: fastscape's ``basic_model`` with
    its ``'spl'``/``'drainage'`` slots replaced by siim's glacial processes.

    Convenience wrapper over :func:`glacial_processes`, with the same parameters
    except the three STANDALONE-only knobs ``inner_passes``, ``inner_relax``
    and ``routing_filter_factor``, which this wrapper does not accept at all
    (passing them is a ``TypeError``; the adapter routes once per step through
    the process graph). For the full coupled
    model with forcing, I/O and plotting, use :class:`siim.siim2d.siim` instead.
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
