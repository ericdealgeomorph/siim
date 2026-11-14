"""The landscape display reads mode D's PER-CELL column datum (``hc_cell_out``,
1 on sheet cells, 1.5 on valley cells) where it derives a column depth or an
ice surface, and stays bit-for-bit on the scalar ``hc_over_H`` path outside
mode D."""

import os
import shutil
import tempfile
from types import SimpleNamespace

import matplotlib.animation as anm
import numba

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from siim.constants import HC_OVER_H
from siim.plotting._render import _footprint_ice_surface, _trunk_ribbons
from siim.plotting.landscape import _run_global_H_max


def _one_source():
    """9x9 flat bed, one 50 m ice source draining to its east neighbour."""
    ny = nx = 9
    zb = np.full((ny, nx), 100.0)
    H = np.zeros((ny, nx))
    H[4, 4] = 50.0
    rec = np.arange(ny * nx, dtype=float).reshape(ny, nx)
    rec[4, 4] = 4 * nx + 5
    return zb, H, rec


def test_footprint_fill_reads_array_datum():
    zb, H, rec = _one_source()
    args = (H, zb, rec, 2.0, 100.0, 100.0, False, False)
    z1, _, d1 = _footprint_ice_surface(*args, hc_over_H=np.ones_like(H))
    assert z1[4, 4] == zb[4, 4] + H[4, 4] and d1[4, 4] == H[4, 4]
    scalar = _footprint_ice_surface(*args, hc_over_H=HC_OVER_H)
    uniform = _footprint_ice_surface(*args, hc_over_H=np.full_like(H, HC_OVER_H))
    for a, b in zip(scalar, uniform):
        assert np.array_equal(a, b)


def _trunk():
    """One 60 m trunk along row 4 flowing east (area growing downstream)."""
    ny = nx = 9
    rec = np.arange(ny * nx, dtype=float).reshape(ny, nx)
    rec[4, :-1] += 1
    H = np.zeros((ny, nx))
    H[4, :-1] = 60.0
    area = np.ones((ny, nx))
    area[4] = 1e6 * np.arange(1, nx + 1)
    zb = np.full((ny, nx), 100.0)
    return rec, H, area, zb, H > 0, nx, ny


def test_trunk_ribbons_per_cell_datum():
    rec, H, area, zb, cells, nx, ny = _trunk()
    common = (cells, nx, ny, 8e3, 8e3, 12.0, 2)

    def ribbons(hc):
        return _trunk_ribbons(rec, H, area, zb + HC_OVER_H * H, *common, hc_over_H=hc)

    d_s, z_s = ribbons(HC_OVER_H)
    assert d_s.max() > 0.0
    d_u, z_u = ribbons(np.full_like(H, HC_OVER_H))
    assert np.array_equal(d_s, d_u) and np.array_equal(z_s, z_u)
    d_1, _ = ribbons(np.ones_like(H))
    np.testing.assert_allclose(HC_OVER_H * d_1, d_s, rtol=1e-12, atol=1e-9)


def test_run_global_H_max_reads_datum():
    H = np.random.default_rng(0).uniform(0, 100, (3, 4, 5))
    hc = np.random.default_rng(1).uniform(1.0, 1.5, H.shape)
    m = SimpleNamespace(H_out=H, hc_over_H=HC_OVER_H, hc_cell_out=hc)
    assert _run_global_H_max(m) == float(np.nanmax(hc * H))
    m.hc_cell_out = None
    assert _run_global_H_max(m) == float(HC_OVER_H * np.nanmax(H))


@pytest.fixture(scope='module')
def switched():
    from siim.siim2d import siim as siim2d
    from siim.tests.test_mode_d import _D      # a mixed channel/sheet mode-D run
    m = siim2d(dict(_D, nt_out=4))
    m.run()
    assert m.hc_cell_out is not None and (m.hc_cell_out[-1] < HC_OVER_H).any()
    return m


@pytest.mark.parametrize('ice_extent', ['cells', 'footprint'])
def test_switched_landscape_renders(switched, ice_extent, monkeypatch):
    def raster():
        fig, ax = switched.plot.landscape(
            field='bedrock+ice', ice_extent=ice_extent, ice_shading='veil',
            trunk_display='ribbons', cross_section=6, H_max=300.0)
        rgb = np.asarray(fig.axes[0].get_images()[0].get_array(), dtype=float)
        plt.close(fig)
        return rgb

    per_cell = raster()
    monkeypatch.setattr(switched, 'hc_cell_out', None)
    assert not np.array_equal(per_cell, raster())   # the datum is read


def test_switched_animate_parallel(switched, tmp_path, monkeypatch):
    if not anm.FFMpegWriter.isAvailable():
        pytest.skip("ffmpeg not available")
    from siim.plotting import landscape as ls
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    # keep the worker bundle so the shim it builds can be inspected
    monkeypatch.setattr(tempfile, 'mkdtemp', lambda **kw: str(bundle))
    monkeypatch.setattr(shutil, 'rmtree', lambda *a, **kw: None)
    out = switched.plot.animate_landscape(path='sw_anim', n_workers=2,
                                          field='bedrock+ice', oversample=2,
                                          trunk_display='ribbons')
    assert out.endswith('.mp4') and os.path.getsize(out) > 0
    monkeypatch.setattr(numba, 'set_num_threads', lambda n: None)  # in-process init
    ls._animate_worker_init(str(bundle), 8, {})
    assert np.array_equal(ls._ANIM_WORKER['plot'].model.hc_cell_out,
                          switched.hc_cell_out)
    (bundle / 'hc_cell_out.npy').unlink()      # a switch-off bundle
    ls._animate_worker_init(str(bundle), 8, {})
    assert ls._ANIM_WORKER['plot'].model.hc_cell_out is None
    ls._ANIM_WORKER.clear()
