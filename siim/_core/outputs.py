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
import numpy as np

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


def output_spec(mode, flexure, sediment, sediment_edge):
    """The active ``(name, dtype, dims)`` rows for this run.
    ``bedrock_surface`` iff mode A (keys the ``_unpack`` mode-A branch,
    siim2d.py:965), ``sediment__*`` iff ``sediment``, ``sediment__edge_*`` (the
    per-edge totals, dims ``(time, side)``) iff ``sediment_edge``,
    ``flexure__rebound`` iff ``flexure``."""
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
    return spec


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
