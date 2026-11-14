"""siim2d constructor knobs that replaced instance-level seams (Linear ED-163).

- ``ce = 0`` / ``Ko = 0``: a frozen bed, built directly (the 1D analytical
  reference, which divides by both, is skipped: ``m.analytical is None``).
- ``hc_over_H``: the channel-floor datum, validated (finite, > 0).
- ``zELA`` as an ``(ny, nx)`` field or an ``(nt, ny, nx)`` field series.

Each knob must reproduce, bit for bit, the run that set the same value on the
instance after construction (what the assessment suites did before).
"""
import numpy as np
import pytest

from siim.siim2d import siim as siim2d
from siim.tests.test_sheet_knobs import _PHYSICS, _assert_same_outputs

_NT, _NY, _NX = 11, 21, 21
_STATE = ('z_out', 'zb_out', 'H_out')   # what the two drivers share bit for bit


def _cfg(**overrides):
    """A small glaciated mode-B run (tent 800 m, ELA 150 m: ice from step 1)."""
    return {**dict(_PHYSICS, T=1e4, nt=_NT, nt_out=6, Lx=2e4, Ly=2e4, nx=_NX, ny=_NY,
                   boundary_status=['fixed_value'] * 4, mode='B', flow_routing='single'),
            **overrides}


def _run(cfg, **attrs):
    m = siim2d(cfg)
    for k, v in attrs.items():
        setattr(m, k, v)
    m.run()
    return m


def _zELA_field():
    return 100.0 + 100.0 * np.linspace(0.0, 1.0, _NX)[None, :] * np.ones((_NY, 1))


@pytest.mark.pin
def test_zero_ce_ko_matches_construct_then_zero():
    """ce = Ko = 0 at construction is the old construct-then-zero run; there is
    no analytical reference, and channel extraction and the profile overlay
    survive its absence."""
    m = _run(_cfg(ce=0, Ko=0))
    ref = _run(_cfg(), ce=0.0, Ko=0.0)
    _assert_same_outputs(m, ref)
    assert m.analytical is None
    m.extract_channel()
    assert m.analytical is None and m.plot._analytical_overlay(True) is None


@pytest.mark.parametrize('bad', [0.0, -1.0, np.inf, np.nan])
def test_hc_over_H_rejects_nonpositive_or_nonfinite(bad):
    with pytest.raises(ValueError, match='hc_over_H'):
        siim2d(_cfg(hc_over_H=bad))


def test_hc_over_H_matches_instance_override(both_drivers):
    """The knob is the old instance override, and it reaches the run."""
    m = _run(_cfg(hc_over_H=1.0))
    _assert_same_outputs(m, _run(_cfg(), hc_over_H=1.0), names=_STATE)   # drivers differ in receiver layout
    assert not np.array_equal(m.H_out, _run(_cfg()).H_out)


def test_hc_over_H_mode_A_sets_only_the_reported_bed():
    """Mode A's dynamics are hc-free: only the reported bed, the kernel's
    z - hc_over_H*H, moves."""
    m = _run(_cfg(mode='A', hc_over_H=1.0))
    ref = _run(_cfg(mode='A'))
    assert np.array_equal(m.z_out, ref.z_out) and np.array_equal(m.H_out, ref.H_out)
    assert not np.array_equal(m.zb_out, ref.zb_out)


@pytest.mark.parametrize('kind', ['field', 'series'])
def test_spatial_zELA_matches_written_series(kind):
    """A static (ny, nx) field and an (nt, ny, nx) series reproduce the old
    write of the driver's _zELA_series; the scalar reference and the per-frame
    output are spatial means (a series' scalar: the minimum over time)."""
    field = _zELA_field()
    shift = np.linspace(0.0, 50.0, _NT)[:, None, None]
    series = field[None] + (0.0 if kind == 'field' else shift)
    m = _run(_cfg(zELA=field if kind == 'field' else series))
    ref = _run(_cfg(), _zELA_series=np.broadcast_to(series, (_NT, _NY, _NX)))
    _assert_same_outputs(m, ref)
    assert m.H_out.max() > 0
    assert m.zELA == pytest.approx(field.mean())
    first, last = (field.mean(), field.mean()) if kind == 'field' else (field.mean(), field.mean() + 50.0)
    np.testing.assert_allclose(m._zELA_output[[0, -1]], [first, last])


@pytest.mark.parametrize('shape', [(_NY + 1, _NX), (_NT - 1, _NY, _NX), (_NT, _NY), (_NT - 1,)])
def test_zELA_wrong_shape_raises(shape):
    with pytest.raises(ValueError, match='zELA must be'):
        siim2d(_cfg(zELA=np.full(shape, 150.0)))


def test_spatial_zT_with_P_series_is_per_cell():
    """zT given as a field with a P series: zELA(t) = zT - P(t)/beta per cell."""
    zT, P = _zELA_field() + 200.0, np.linspace(3.0, 1.0, _NT)
    m = siim2d(_cfg(zELA=None, zT=zT, P=P))
    assert m._zELA_series is not None
    np.testing.assert_array_equal(m._zELA_series, zT[None] - P[:, None, None] / 1e-2)


@pytest.mark.adapter
def test_static_zELA_field_adapter_matches_inhouse():
    pytest.importorskip('xsimlab')
    pytest.importorskip('fastscape')
    one = dict(inner_passes=1, inner_relax=1.0)   # the adapter's one pass
    m = siim2d(_cfg(zELA=_zELA_field(), **one))
    m.run(driver='xsimlab')
    _assert_same_outputs(m, _run(_cfg(zELA=_zELA_field(), **one)), names=_STATE)   # drivers differ in receiver layout


def test_zELA_field_series_refused_by_adapter():
    m = siim2d(_cfg(zELA=np.broadcast_to(_zELA_field(), (_NT, _NY, _NX))))
    with pytest.raises(NotImplementedError, match='in-house-driver only'):
        m.run(driver='xsimlab')


def test_save_load_round_trips_the_knobs(tmp_path):
    m = _run(_cfg(ce=0, Ko=0, hc_over_H=1.0, zELA=_zELA_field()))
    loaded = siim2d.load(m.save(tmp_path / 'knobs'))
    assert loaded.analytical is None and loaded.hc_over_H == 1.0
    assert np.array_equal(loaded._zELA_field, _zELA_field())
    _assert_same_outputs(loaded, m)
