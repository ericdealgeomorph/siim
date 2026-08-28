"""Pack the in-house driver's step buffers into the exact ``ds_out`` contract.

The standalone driver (:mod:`siim._core.driver`) fills per-step numpy buffers;
this module allocates them to the exact dtypes and wraps them in the
``xarray.Dataset`` :meth:`siim.siim2d.siim._unpack_outputs` reads — same
variable names, dims ``(time, y, x)``, coords (``time``/``x``/``y``/``tstep``),
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


def output_spec(mode, flexure, sediment):
    """The active ``(name, dtype)`` rows for this run. ``bedrock_surface`` iff
    mode A (keys the ``_unpack`` mode-A branch, siim2d.py:965), ``sediment__*``
    iff ``sediment``, ``flexure__rebound`` iff ``flexure``."""
    spec = list(_BASE_SPEC)
    if mode == 'A':
        spec.append(('glacial_spl__bedrock_surface', _FLOAT))
    if sediment:
        spec.append(('sediment__flux', _FLOAT))
        spec.append(('sediment__cumulative', _FLOAT))
    if flexure:
        spec.append(('flexure__rebound', _FLOAT))
    return spec


def allocate_buffers(spec, nt_out, ny, nx):
    """One ``(nt_out, ny, nx)`` output buffer per spec row, at the row's dtype."""
    return {name: np.empty((nt_out, ny, nx), dtype=dt) for name, dt in spec}


def build_dataset(buffers, t_out, x, y, tstep):
    """Wrap the filled step buffers into the ``ds_out`` xarray Dataset:
    dims ``(time, y, x)``; coords ``time=t_out``, ``x``, ``y``, ``tstep``. Reads
    only variable values by name downstream (``_unpack_outputs``), so this
    in-memory build is interchangeable with the retired zarr path."""
    import xarray as xr
    data_vars = {name: (('time', 'y', 'x'), buf) for name, buf in buffers.items()}
    return xr.Dataset(
        data_vars,
        coords={'time': t_out, 'x': x, 'y': y, 'tstep': tstep})
