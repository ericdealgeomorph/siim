"""The ablation footprint options of siim2d (``footprint_from_H``, ``footprint_width_factor``), EXPERIMENTAL and off.

Below the ELA a cell ablates over the footprint ``max(W*L, cell_area)``. The default width is the Hack width
``W = width_hack_k*A**width_hack_p``; ``footprint_from_H`` replaces it with the valley width ``alpha_g*H`` on the
lagged pass thickness, and ``footprint_width_factor`` f multiplies either form. Both act on the footprint alone.
"""
import numpy as np
import pytest

from siim import constants
from siim._core import driver as _driver
from siim._core.step import accumulate_glacial_flow
from siim.siim2d import siim as siim2d

SHAPE = (6, 9)
CELL = 1e4                                                  # m^2
LENGTH = 100.0                                              # m
K, P = 0.5, 0.5                                             # width_hack_k, width_hack_p


def _args(rec):
    """A random surface around the ELA (2200 m) on routing ``rec`` (flat SFR receivers, receiver-first stack)."""
    surface = 2000.0 + 500.0 * np.random.default_rng(3).random(SHAPE)
    n = surface.size
    return (surface, 0.0, 2200.0, 1e-2, 1.0, CELL, K, P, SHAPE, np.arange(n), rec, None, None,
            np.full(n, LENGTH), None)


def _hack(area):
    return K * area ** P


def _same(ref, out):
    """ice flux, water flux and drainage area equal, bit for bit"""
    for a, b in zip(ref[:3], out[:3]):
        assert np.array_equal(a, b)


def _rows():
    """Each row drains to the row below: drainage areas grow downstream."""
    ny, nx = SHAPE
    return np.r_[np.arange(nx), np.arange(nx, ny * nx) - nx]


def test_width_replaces_the_hack_width_in_the_footprint_only():
    """``width`` equal to the Hack width is the default, bit for bit; any ``width`` sets the below-ELA source to
    beta*(z - zELA)*max(width*L, cell_area) (capped at the precipitation flux), leaves every above-ELA source and the
    drainage area alone. Self-receiving cells make the pre-clamp flux the source itself."""
    ref = accumulate_glacial_flow(*_args(_rows()))
    _same(ref, accumulate_glacial_flow(*_args(_rows()), width=_hack(ref[2])))

    args = _args(np.arange(SHAPE[0] * SHAPE[1]))
    surface = args[0]
    W = 50.0 + 400.0 * np.random.default_rng(4).random(SHAPE)        # an H-form width, alpha_g*H
    src_hack, src_W = np.empty(surface.size), np.empty(surface.size)
    hack = accumulate_glacial_flow(*args, raw_out=src_hack)
    wide = accumulate_glacial_flow(*args, width=W, raw_out=src_W)
    below = (surface < 2200.0).ravel()
    assert below.any() and (~below).any()
    assert np.array_equal(src_W[~below], src_hack[~below])
    expect = np.minimum(1e-2 * (surface - 2200.0) * np.maximum(W * LENGTH, CELL), CELL).ravel()
    assert np.allclose(src_W[below], expect[below], rtol=1e-14, atol=0.0)
    assert not np.array_equal(src_W[below], src_hack[below])
    assert np.array_equal(wide[2], hack[2])


def test_width_factor_scales_the_footprint_width_only():
    """f*Hack equals ``width`` = f*k_w*A^p, f*``width`` equals ``width`` = f*W, exactly; f = 1 is the default; the
    drainage area never moves."""
    args = _args(_rows())
    ref = accumulate_glacial_flow(*args)
    W = 40.0 + 30.0 * np.random.default_rng(5).random(SHAPE)
    for f in (0.4, 3.2):
        hack = accumulate_glacial_flow(*args, width_factor=f)
        assert np.array_equal(hack[0], accumulate_glacial_flow(*args, width=f * _hack(ref[2]))[0])
        assert not np.array_equal(hack[0], ref[0]) and np.array_equal(hack[2], ref[2])
        assert np.array_equal(accumulate_glacial_flow(*args, width=W, width_factor=f)[0],
                              accumulate_glacial_flow(*args, width=f * W)[0])
    _same(ref, accumulate_glacial_flow(*args, width_factor=1.0))


BASE = dict(nx=21, ny=17, Lx=20e3, Ly=16e3, T=40e3, nt=41, nt_out=3, seed=5, zELA=900.0,
            initial_max_elevation=2000.0, progress_bar=False)


def test_the_driver_passes_alpha_g_H_and_f(monkeypatch):
    """On, the accumulator gets width = alpha_g*x, x the pass iterate the routing surface was built on, and the
    factor f; off, width None and f = 1. alpha_g = 4 (not the default) so the test tells them apart."""
    seen = []
    orig_route, orig_acc = _driver.routing_surface, _driver.accumulate_glacial_flow

    def route(zb, hc, x):
        seen.append(['x', x.copy()])
        return orig_route(zb, hc, x)

    def acc(*a, width=None, width_factor=1.0, **kw):
        seen[-1][0] = (width, width_factor)
        return orig_acc(*a, width=width, width_factor=width_factor, **kw)

    monkeypatch.setattr(_driver, 'routing_surface', route)
    monkeypatch.setattr(_driver, 'accumulate_glacial_flow', acc)
    for opts in (dict(), dict(footprint_from_H=True, footprint_width_factor=3.2)):
        seen.clear()
        m = siim2d(BASE | dict(mode='C', flow_routing='mfd', alpha_g=4.0, nt=11) | opts)
        m.run()
        assert seen and max(float(x.max()) for _, x in seen) > 10.0
        for (width, f), x in seen:
            if opts:
                assert f == 3.2 and np.array_equal(width, 4.0 * x)
            else:
                assert width is None and f == 1.0


@pytest.mark.pin
def test_options_off_are_the_default_model():
    """footprint_from_H=False and footprint_width_factor=1.0 passed explicitly are the default model, bit for bit,
    and each option on changes it (mode C on MFD, mode A on D8, mode B on D-inf, glaciated)."""
    assert constants.FOOTPRINT_FROM_H is False and constants.FOOTPRINT_WIDTH_FACTOR == 1.0
    H = 'glacial_spl__ice_thickness'
    for extra in (dict(mode='C', flow_routing='mfd'), dict(mode='A', flow_routing='single'),
                  dict(mode='B', flow_routing='dinf')):
        runs = []
        for opts in ({}, dict(footprint_from_H=False, footprint_width_factor=1.0),
                     # the on arms widen the footprint past the 1 km cell: at the default width_hack_k 0.2 a
                     # narrower footprint (or alpha_g*H at f = 1) floors to the cell and changes nothing here
                     dict(footprint_from_H=True, footprint_width_factor=3.2), dict(footprint_width_factor=4.0)):
            m = siim2d(BASE | extra | opts)
            m.run()
            runs.append(m.ds_out)
        assert float(runs[0][H].max()) > 10.0
        for k in runs[0].data_vars:
            if 'basin' not in k:
                assert np.array_equal(runs[0][k].values, runs[1][k].values), k
        for on in runs[2:]:
            assert not np.array_equal(runs[0][H].values, on[H].values), extra


def test_options_validated_and_refused_on_the_xsimlab_driver():
    cfg = dict(nx=11, ny=11, nt=3, nt_out=2, T=2e3, progress_bar=False)
    with pytest.raises(ValueError, match='True or False'):
        siim2d(cfg | dict(footprint_from_H='yes'))
    for bad in (True, np.full((11, 11), 3.0)):
        with pytest.raises(ValueError, match='footprint_width_factor must be a scalar'):
            siim2d(cfg | dict(footprint_width_factor=bad))
    for bad in (0.0, -1.0, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='footprint_width_factor must be finite and > 0'):
            siim2d(cfg | dict(footprint_width_factor=bad))
    for opts in (dict(footprint_from_H=True), dict(footprint_width_factor=3.2)):
        m = siim2d(cfg | opts)
        with pytest.raises(NotImplementedError, match='in-house-driver'):
            m.run(driver='xsimlab')
