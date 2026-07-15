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
:class:`siim.siim2d.siim` instead — it assembles its model from exactly this
:func:`glacial_processes` call, so the two never diverge.

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
from fastscape.models import basic_model
from fastscape.processes import SingleFlowRouter

from .. import constants as _constants
from .processes import (
    DinfFlowRouter,
    GlacialBlockUplift,
    GlacialFlexure,
    GlacialFlowAccumulator,
    GlacialLaw,
    GlacialSPLModeA,
    GlacialSPLModeB,
    GlacialSPLModeC,        # mode B + sub-grid carve, citizen semantics
    GlacialSurfaceToErode,
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


def _silence_xarray_did_you_mean():
    """Neuter xarray's fuzzy 'did you mean ...?' KeyError suggestion builder.

    xsimlab 0.5.0's per-step driver probes the run dataset with ``ds[key]`` for
    ~35 keys per step that are NOT in the dataset. Each miss hits
    ``Dataset.__getitem__``'s ``except KeyError`` path, which eagerly builds a
    :func:`difflib.get_close_matches` suggestion *before* re-raising — ~0.24
    ms/step (0.2-2.4% of a real run; audit N33). xsimlab immediately catches
    and discards that KeyError, so the suggestion string is never surfaced on
    the hot path.

    We replace only ``xarray.core.utils.did_you_mean`` with a no-op returning
    ``""``. Effect is confined to the *content* of a KeyError message for a
    genuinely-missing variable: it still says "No variable named X" (and, per
    xarray's own fallback, lists the available variables), just without the
    fuzzy "did you mean?" hint. Idempotent and reversible; a no-op if xarray's
    internals move. Measured ~22% faster per step on a small grid.
    """
    try:
        from xarray.core import utils as _xr_utils
    except Exception:
        return
    if getattr(_xr_utils, '_siim_did_you_mean_silenced', False):
        return
    _xr_utils._siim_did_you_mean_orig = _xr_utils.did_you_mean
    _xr_utils.did_you_mean = lambda *args, **kwargs: ""
    _xr_utils._siim_did_you_mean_silenced = True


_silence_xarray_did_you_mean()


def _silence_xsimlab_drop_futurewarning():
    """Suppress the ``dropping variables using `drop` is deprecated``
    FutureWarning that xsimlab 0.5.0 (frozen upstream) triggers on every run
    (``stores.write_input_xr_dataset`` + the step-clock setup in ``drivers``
    both call xarray's long-deprecated ``Dataset.drop``). Modern xarray
    escalated the warning to FutureWarning ahead of removing the method —
    third-party noise, not siim's to fix; ``environment.yml`` freezes xarray
    below the removal (the same quarantine class as numpy<2). Scoped to
    warnings raised FROM xsimlab so siim's own deprecations still surface.
    Sibling of the ``did_you_mean`` neutering above (audit N33)."""
    import warnings
    warnings.filterwarnings(
        "ignore", message="dropping variables using",
        category=FutureWarning, module="xsimlab")


_silence_xsimlab_drop_futurewarning()


def glacial_processes(*, mode='B', carve=None, routing='single',
                      flexure=False, sediment=False, trunk_surface=False):
    """Build the slot→process-class override dict for fastscape's ``basic_model``.

    Apply the returned dict after dropping the stock fluvial slots::

        basic_model.drop_processes(['spl', 'drainage']).update_processes(
            glacial_processes(...))

    or use :func:`glacial_model`, which does exactly that. This is siim's single
    source of truth for process assembly — :class:`siim.siim2d.siim` builds its
    model from this same function.

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
        (the citizen bed+H class plus the carve). Ignored for ``mode='A'``.
        Defaults off here so the advertised default is the plain Mode-B citizen;
        note :class:`siim.siim2d.siim` defaults carving *on*. ``mode='C'`` is the
        alias for turning it on.
    trunk_surface : bool, optional
        Fabricated trunk-surface routing (mode B/C) — swaps in
        :class:`TrunkSurfaceToErode` so routing + accumulation converge trunk
        flow onto the centerline (the centerline's raw flux is then the full
        cross-section). Off by default here; :class:`siim.siim2d.siim` turns it
        on as part of the mode-C standard. Ignored for ``mode='A'``.
    routing : {'single', 'dinf'}, optional
        D8 single-flow (``'single'``, default) or D-infinity (``'dinf'``).
    flexure, sediment : bool, optional
        Opt in to glacial-isostatic flexure / sediment tracking.

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

    overrides = {
        'glacial_flow': GlacialFlowAccumulator,
        'law': GlacialLaw,
        'flow': DinfFlowRouter if routing == 'dinf' else SingleFlowRouter,
        'init_topography': InitialTopography,
        # Squeeze-tolerant BlockUplift so a documented (nt, ny, nx) uplift rate
        # doesn't crash on the pinned xarray/xsimlab stack (audit B6).
        'uplift': GlacialBlockUplift,
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


def glacial_model(*, mode='B', carve=None, routing='single',
                  flexure=False, sediment=False, trunk_surface=False):
    """Return a ready-to-run xsimlab ``Model``: fastscape's ``basic_model`` with
    its ``'spl'``/``'drainage'`` slots replaced by siim's glacial processes.

    Convenience wrapper over :func:`glacial_processes` (same parameters). For the
    full coupled model with forcing, I/O and plotting, use
    :class:`siim.siim2d.siim` instead.
    """
    return (basic_model
            .drop_processes(['spl', 'drainage'])
            .update_processes(glacial_processes(
                mode=mode, carve=carve, routing=routing,
                flexure=flexure, sediment=sediment,
                trunk_surface=trunk_surface)))
