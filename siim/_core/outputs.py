"""Pack the in-house driver's step buffers into the exact ``ds_out`` contract.

The standalone driver (:mod:`siim._core.driver`) fills per-step numpy buffers;
this module allocates them to the exact dtypes and wraps them in the
``xarray.Dataset`` :meth:`siim.siim2d.siim._unpack_outputs` reads — same
variable names, dims (``(time, y, x)`` for every per-node field, ``(time,
side)`` for the per-domain-edge sediment totals), coords
(``time``/``x``/``y``/``tstep``, plus ``side``),
dtypes (float64 fields; int32 for ``basin_ids``/``receivers_2d``/``stack_2d``)
and the mode/flag-conditional rows. ``_unpack_outputs`` / ``save`` / ``load``
consume the exact keys constructed here; see
``docs/guides/outputs_and_io.md`` for the public output contract.

xarray is the only heavy import; it stays function-local so
``import siim._core.outputs`` costs nothing until a dataset is built.
"""
import warnings

import numpy as np

from ..constants import SAVE_CLOSURE_PAIR

_FLOAT = np.float64
_INT = np.int32

_RASTER: tuple[str, ...] = ('time', 'y', 'x')   # the per-node fields
_EDGE: tuple[str, ...] = ('time', 'side')       # the per-domain-edge totals

#: The ``side`` coordinate: domain edges in ``border_status`` order. Index 2
#: ('bottom') is row 0, i.e. y = 0, and index 3 ('top') is row ny-1 — the
#: fastscape ibc digit naming calls those same two edges the other way round.
SIDES = ('left', 'right', 'bottom', 'top')

# The always-present rows, shared by the in-house and xsimlab drivers.
_BASE_SPEC = (
    ('topography__elevation', _FLOAT),
    ('glacial_spl__ice_thickness', _FLOAT),
    ('glacial_flow__ice_flux', _FLOAT),
    ('glacial_flow__water_flux', _FLOAT),
    ('glacial_flow__area', _FLOAT),
    ('glacial_flow__basin_ids', _INT),
    ('glacial_flow__receivers_2d', _INT),
    ('glacial_flow__stack_2d', _INT),
    ('glacial_spl__erosion_rate', _FLOAT),
    ('glacial_spl__denudation', _FLOAT),
    ('uplift__uplift', _FLOAT),
)


def output_spec(mode, flexure, sediment, sediment_edge, regime_switch=False,
                closure_flux='cell', inner_relax=1.0, save_closure_pair=SAVE_CLOSURE_PAIR):
    """The active ``(name, dtype, dims)`` rows for this run.
    ``bedrock_surface`` iff mode A (keys the mode-A branch of siim2d's
    ``_unpack_outputs``), ``sediment__*`` iff ``sediment``, ``sediment__edge_*`` (the
    per-edge totals, dims ``(time, side)``) iff ``sediment_edge``,
    ``flexure__rebound`` iff ``flexure``, ``glacial_spl__hc_cell`` (the per-cell
    channel-floor datum of mode D's valley <-> sheet map,
    ``constants.HC_SHEET``) and ``glacial_spl__channel`` (mode D's binary
    class read on the frame's committed H: 1 on icy cells read channel, 0 on
    sheet cells AND off the ice, ``H <= constants.REGIME_ICY_H``) iff
    ``regime_switch``.

    The closure pair, iff ``save_closure_pair`` (``constants.SAVE_CLOSURE_PAIR``),
    each row only where it differs from its stored counterpart (mode B/C): ``glacial_spl__closure_ice_flux`` (the flux the
    kernel's closure solved H from, last pass) iff ``closure_flux != 'cell'``,
    and ``glacial_spl__closure_ice_thickness`` (the kernel's own thickness F,
    last pass, before the ``inner_relax`` blend) iff ``inner_relax < 1``."""
    spec = [(name, dt, _RASTER) for name, dt in _BASE_SPEC]
    if mode == 'A':
        spec.append(('glacial_spl__bedrock_surface', _FLOAT, _RASTER))
    if sediment:
        spec.append(('sediment__flux', _FLOAT, _RASTER))
        spec.append(('sediment__cumulative', _FLOAT, _RASTER))
    if sediment_edge:
        spec.append(('sediment__edge_flux', _FLOAT, _EDGE))
        spec.append(('sediment__edge_cumulative', _FLOAT, _EDGE))
    if flexure:
        spec.append(('flexure__rebound', _FLOAT, _RASTER))
    if regime_switch:
        spec.append(('glacial_spl__hc_cell', _FLOAT, _RASTER))
        spec.append(('glacial_spl__channel', _INT, _RASTER))
    if save_closure_pair and mode == 'B' and closure_flux != 'cell':
        spec.append(('glacial_spl__closure_ice_flux', _FLOAT, _RASTER))
    if save_closure_pair and mode == 'B' and inner_relax < 1.0:
        spec.append(('glacial_spl__closure_ice_thickness', _FLOAT, _RASTER))
    return spec


def closure_pair(m, i):
    """``(H, Qg)`` the thickness closure solved at output ``i`` of model ``m``
    (a run, or any object with its ``*_out`` arrays): the closure's own rows
    where the run stored them (``closure_H_out``, ``closure_Qg_out``), else
    ``H_out`` / ``Qg_out``. The pair :func:`siim._core.solvers.ice_speeds`
    inverts."""
    H = getattr(m, 'closure_H_out', None)
    Q = getattr(m, 'closure_Qg_out', None)
    return ((m.H_out if H is None else H)[i], (m.Qg_out if Q is None else Q)[i])


def warn_if_closure_pair_missing(m):
    """Warn that the speed of model ``m`` is approximate when its settings need
    the closure's own (flux, thickness) rows and it did not store them
    (``save_closure_pair`` off, the default). Skipped for objects without the
    settings (the animation workers' shim). One warning per session: it is
    attributed to this line, whichever caller reaches it."""
    flux, relax = getattr(m, 'closure_flux', None), getattr(m, 'inner_relax', None)
    if flux is None or relax is None:
        return
    if ((flux != 'cell' and getattr(m, 'closure_Qg_out', None) is None)
            or (relax < 1.0 and getattr(m, 'closure_H_out', None) is None)):
        warnings.warn(
            "this run did not store the closure's own (flux, thickness) pair "
            f"(closure_flux={flux!r}, inner_relax={relax:g}); the ice speed is "
            "approximate (save_closure_pair=True stores the pair)", stacklevel=1)


def allocate_buffers(spec, nt_out, ny, nx):
    """One output buffer per spec row, at the row's dtype and dims (the raster
    rows ``(nt_out, ny, nx)``, the per-edge rows ``(nt_out, 4)``)."""
    size = {'time': nt_out, 'y': ny, 'x': nx, 'side': len(SIDES)}
    return {name: np.empty(tuple(size[d] for d in dims), dtype=dt)
            for name, dt, dims in spec}


def build_dataset(buffers, spec, t_out, x, y, tstep):
    """Wrap the filled step buffers into the ``ds_out`` xarray Dataset: each
    row at its spec dims; coords ``time=t_out``, ``x``, ``y``, ``tstep``, plus
    ``side`` when a per-edge row is active. Reads only variable values by name
    downstream (``_unpack_outputs``), so this in-memory build is
    interchangeable with the retired zarr path."""
    import xarray as xr
    data_vars = {name: (dims, buffers[name]) for name, _dt, dims in spec}
    coords = {'time': t_out, 'x': x, 'y': y, 'tstep': tstep}
    if any(dims == _EDGE for _n, _dt, dims in spec):
        coords['side'] = list(SIDES)
    return xr.Dataset(data_vars, coords=coords)


# --- receiver-network views of receivers_out (shared by the model's data
# methods and the plotting layer) ----------------------------------------------
def _sanitized_receivers(rec_in):
    """Flat int receiver array with NaNs mapped to SELF — the zarr round-trip
    artefact at boundary nodes, which every walk over ``receivers_out`` guards
    against (``nan_to_num``'s 0 is the corner cell, which would make every such
    node a phantom donor of the corner and grow a bogus path from it)."""
    rec_flat = np.asarray(rec_in, dtype=float).ravel()
    return np.where(np.isnan(rec_flat), np.arange(rec_flat.size),
                    rec_flat).astype(np.int64)


def _donor_lists(rec):
    """Each node's donors, in ascending node order, from a flat receiver
    array (``_sanitized_receivers``); a self-receiving node donates to no
    one."""
    donors = [[] for _ in range(rec.size)]
    for j in range(rec.size):
        r = rec[j]
        if r != j:
            donors[r].append(j)
    return donors
