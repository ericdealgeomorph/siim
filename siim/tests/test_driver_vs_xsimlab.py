"""S3 headline gate: the in-house driver == the xsimlab driver, bit-for-bit.

Identical setups run through (a) the xsimlab adapter orchestration and (b)
siim's in-house time loop (:mod:`siim._core.driver`), BOTH on the same default
backend (in-house flexure/diffusion + fortran routing at S3). Assert
``np.array_equal`` on every compared output at EVERY output time — this
comparison is same-process/same-platform, so bitwise is legitimate; any
mismatch is a real ordering/state bug (never tolerance it; plan S3 escalation
rule).

Compared: ``z_out, zb_out, H_out`` (+ ``rebound_out`` where flexure,
``sediment_*`` where tracked) plus the deterministic float fields
``Qg/Qf/area/erosion_rate/denudation`` — a strictly stronger gate at zero
risk. EXCLUDED: ``basin_out`` (nondeterministic fortran labels, Map 4 §4) and
``receivers/stack`` (the xsimlab store's integer NaN-decode artifact, the S0
misdiagnosis — raw-int gating is S4's job).

Matrix: {A,B,C} x {SFR,dinf} x {flexure off/on} (carve rides mode C) + a
bl(t)-step + sediment case + an escarpment (wave uplift + plateau) case.
Fresh save->load round-trip under the in-house driver rides at the bottom
(OQ-3: no old-pickle gate; forward round-trip only).
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from siim.siim2d import siim as siim2d            # noqa: E402
from siim.escarpment import siim_escarpment       # noqa: E402

_NT = 61

_BASE = dict(U=1e-3, zELA=150, beta=1e-2, P=2, alpha_g=12, Ko=2e-6, n=1,
             ce=1e-4, nu=2, sliding_law='power', lambda_p=500, k=0.9,
             T=6e4, nt=_NT, nt_out=7, Lx=2e4, Ly=2e4, nx=31, ny=31, seed=7,
             boundary_status=['fixed_value', 'fixed_value', 'looped', 'looped'],
             initial_max_elevation=800, progress_bar=False)

_FIELDS = ('z_out', 'zb_out', 'H_out', 'Qg_out', 'Qf_out', 'area_out',
           'erosion_rate_out', 'denudation_out')


def _assert_pair(params, cls=siim2d):
    """Run identical setups through both drivers; np.array_equal everywhere.
    Adapter-env only (the xsimlab arm needs the stack) — every caller carries
    the ``adapter`` marker."""
    pytest.importorskip('xsimlab')
    pytest.importorskip('fastscape')
    a = cls(params); a.run(driver='xsimlab')
    b = cls(params); b.run(driver='inhouse')
    fields = list(_FIELDS)
    if params.get('flexure'):
        fields.append('rebound_out')
    if params.get('track_sediment'):
        fields += ['sediment_flux_out', 'eroded_volume_out']
    for name in fields:
        xa, xb = getattr(a, name), getattr(b, name)
        assert xa.shape == xb.shape, f"{name}: shape {xa.shape} vs {xb.shape}"
        if not np.array_equal(xa, xb):
            first = next(i for i in range(xa.shape[0])
                         if not np.array_equal(xa[i], xb[i]))
            raise AssertionError(
                f"{name} differs (first differing output frame {first}, "
                f"t={a.output_times[first]:g}); max|d|="
                f"{np.max(np.abs(np.asarray(xa, float) - np.asarray(xb, float))):.3e}")
    # coords + frame count of the packed dataset match the contract
    assert np.array_equal(a.output_times, b.output_times)
    return a, b


@pytest.mark.adapter
@pytest.mark.parametrize('flexure', [False, True], ids=['flexoff', 'flexon'])
@pytest.mark.parametrize('routing', ['single', 'dinf'])
@pytest.mark.parametrize('mode', ['A', 'B', 'C'])
def test_driver_matrix(mode, routing, flexure):
    """{A,B,C} x {SFR,dinf} x {flexure off/on}: bit-for-bit at every frame.
    Mode C = B + carve + the mode-C standard (trunk_surface, routing_relax)."""
    _assert_pair(dict(_BASE, mode=mode, flow_routing=routing,
                      flexure=flexure, ice_load=flexure))


@pytest.mark.adapter
@pytest.mark.parametrize('routing', ['single', 'dinf'])
def test_driver_bl_step_sediment(routing):
    """bl(t) step drop (the S0 battery's blstep analog) + sediment tracking:
    the per-step bl slicing and the driver-owned _cum both match bitwise."""
    bl = np.zeros(_NT)
    bl[_NT // 2:] = -60.0
    _assert_pair(dict(_BASE, mode='B', flow_routing=routing, bl=bl,
                      track_sediment=True))


@pytest.mark.adapter
def test_driver_escarpment_wave_plateau():
    """Escarpment: WaveUplift (midpoint-sampled) + PlateauSurface through the
    subclass driver seams == the xsimlab shells, incl. the snapshotted
    uplift__uplift wave field."""
    p = dict(_BASE, zELA=800, mode='C',
             uplift_type='wave', delta_h=1200.0, wave_width=1e4,
             wave_velocity=0.2, x_escarpment=1e4,
             init_type='plateau', plateau_zo=900.0)
    a, b = _assert_pair(p, cls=siim_escarpment)
    assert np.array_equal(a.ds_out['uplift__uplift'].values,
                          b.ds_out['uplift__uplift'].values)


@pytest.mark.adapter
def test_driver_U_series():
    """(nt,) time-varying U: the driver's direct arr[k] indexing == xsimlab's
    size-1 clock slice (Map 2 §4)."""
    _assert_pair(dict(_BASE, mode='C', U=np.linspace(5e-4, 2e-3, _NT)))


def test_nt_out_exceeding_nt_rejected():
    """Degenerate cadence nt_out > nt is rejected at construction (driver-
    independent): duplicate out_idx entries made xsimlab emit NaN frames while
    the in-house driver duplicated real frames — now a directed error naming
    both values."""
    with pytest.raises(ValueError, match=r"nt_out.*8.*nt=5"):
        siim2d(dict(_BASE, nt=5, nt_out=8))


@pytest.mark.adapter
@pytest.mark.parametrize('nt_out', [1, 5], ids=['ntout1', 'ntout_eq_nt'])
def test_driver_cadence_edges(nt_out):
    """The legal cadence edges stay bit-for-bit across drivers: nt_out == nt
    (every-step output) and nt_out == 1 (a single frame-0 snapshot; measured
    identical under xsimlab too — the determination keeping the lower bound
    at 1)."""
    _assert_pair(dict(_BASE, mode='C', nt=5, nt_out=nt_out))


def test_inhouse_run_rejects_hooks():
    """OQ-2: the standalone driver DROPS the xsimlab hooks= kwarg (directed
    error); the xsimlab arm keeps native hooks."""
    m = siim2d(_BASE)
    with pytest.raises(ValueError, match="hooks"):
        m.run(hooks=object(), driver='inhouse')


def test_inhouse_ds_out_contract():
    """The packed ds_out carries the exact contract rows: names, (time,y,x)
    dims, int32 topology vars, float64 fields, time/x/y/tstep coords, and the
    bedrock_surface-present-iff-mode-A invariant (Map 2 §1)."""
    m = siim2d(dict(_BASE, mode='C', track_sediment=True, flexure=True))
    m.run(driver='inhouse')
    ds = m.ds_out
    for name in ('topography__elevation', 'glacial_spl__ice_thickness',
                 'glacial_flow__ice_flux', 'glacial_flow__water_flux',
                 'glacial_flow__area', 'glacial_spl__erosion_rate',
                 'glacial_spl__denudation', 'uplift__uplift',
                 'sediment__flux', 'sediment__cumulative', 'flexure__rebound'):
        assert name in ds, name
        assert ds[name].dims == ('time', 'y', 'x'), name
        assert ds[name].dtype == np.float64, name
    for name in ('glacial_flow__basin_ids', 'glacial_flow__receivers_2d',
                 'glacial_flow__stack_2d'):
        assert ds[name].dims == ('time', 'y', 'x'), name
        assert ds[name].dtype == np.int32, name
    assert 'glacial_spl__bedrock_surface' not in ds       # mode B/C
    assert np.array_equal(ds.time.values, m.t_out)
    assert np.array_equal(ds.tstep.values, m.t)
    assert np.array_equal(ds.x.values, np.linspace(0, m.Lx, m.grid_nx))
    assert np.array_equal(ds.y.values, np.linspace(0, m.Ly, m.grid_ny))

    ma = siim2d(dict(_BASE, mode='A'))
    ma.run(driver='inhouse')
    assert 'glacial_spl__bedrock_surface' in ma.ds_out    # keys the _unpack branch


def test_inhouse_save_load_round_trip(tmp_path, monkeypatch):
    """Fresh save->load round-trip under the in-house driver (OQ-3): the loaded
    model's unpacked attrs equal the saver's."""
    monkeypatch.chdir(tmp_path)   # save() writes ./model_outputs/saved_models
    m = siim2d(dict(_BASE, mode='C', flexure=True, ice_load=True))
    m.run(driver='inhouse')
    path = m.save('s3_roundtrip')
    m2 = siim2d.load(path)
    for name in _FIELDS + ('rebound_out',):
        assert np.array_equal(getattr(m, name), getattr(m2, name)), name
    assert np.array_equal(m.output_times, m2.output_times)
    assert np.array_equal(m.basin_out, m2.basin_out)      # same pickle, safe here
