"""Restarting siim2d from a run's end state (``state=m.restart_state()``) and
from a supplied ice thickness (``initial_ice_thickness``).

1. a restart continues the run: T, then T more from its end state (through
   ``np.savez``), equals 2T in one go, every stored row of every frame, in
   modes A, B (single and multiple flow), C (with the plate) and D (with the
   plate), and the loop leaves the start state it was handed untouched;
2. ``initial_ice_thickness`` is that restart wherever the run carries no
   state beyond the bed and the ice (no adaptive damping, no plate, not D);
3. the trust boundary: shapes, finite and in-range values, the state's mode,
   grid and boundaries, its groups whole, the drivers that take it.
"""
import io

import numpy as np
import pytest

from siim._core import driver as drv
from siim.siim2d import siim as siim2d
from siim.tests.test_mode_d import _D
from siim.tests.test_sheet_knobs import _PHYSICS

_N, _DT = 6, 1000.0        # steps per half, a step whole years long (same dt both clocks)
_NXY = 21
_GRID = dict(Lx=1e4, Ly=1e4, nx=_NXY, ny=_NXY)
_PLATE = dict(flexure=True, e_thickness=20e3)
_CASES = {
    'A': dict(_PHYSICS, **_GRID, mode='A'),
    'B_single': dict(_PHYSICS, **_GRID, mode='B'),
    'B_mfd': dict(_PHYSICS, **_GRID, mode='B', flow_routing='mfd'),
    'C_plate': dict(_PHYSICS, **_GRID, mode='C', **_PLATE),
    'D_plate': dict(_D, **_PLATE),          # mode D's fixture: a tent cut by four valleys
}


def _model(cfg, steps, **ov):
    """``steps`` steps of ``_DT`` from ``ov.get('to', 0)``, a frame every step."""
    to = ov.pop('to', 0.0)
    return siim2d(dict(cfg, **dict(dict(to=to, T=to + steps * _DT, nt=steps + 1,
                                        nt_out=steps + 1), **ov)))


def _run(cfg, steps, **ov):
    m = _model(cfg, steps, **ov)
    m.run()
    return m


def _assert_continues(full, rest, same_clock=True):
    """``rest``'s frames are ``full``'s from step ``_N`` on, every stored row
    (the basin labels aside, which the battery never gates either)."""
    rows = [v for v in full.ds_out.data_vars if 'basin' not in v]
    assert rows == [v for v in rest.ds_out.data_vars if 'basin' not in v]
    for v in rows:
        a, b = full.ds_out[v].values[_N:], rest.ds_out[v].values
        assert np.array_equal(a, b), f"{v} differs after the restart"
    if same_clock:
        np.testing.assert_array_equal(full.ds_out.time.values[_N:], rest.ds_out.time.values)


def _savez_round_trip(state):
    buf = io.BytesIO()
    np.savez(buf, **state)
    buf.seek(0)
    return dict(np.load(buf))


@pytest.mark.pin
@pytest.mark.parametrize('case', list(_CASES))
def test_restart_continues_the_run_exactly(case):
    """T, then a restart from its end state for T, equals 2T in one go, bit for
    bit: the frames after the restart, the ice volume with them, and the state
    the two runs end on. The state goes through ``np.savez`` (0-d mode, array
    grid and boundaries); one case restarts on its own clock (``to`` = 0, the
    clock entering only through the step size), one saves a single frame (so
    the end state cannot be read off the frames), and the loop must leave the
    start state it was handed as it was (it rebinds, never writes into it)."""
    cfg = _CASES[case]
    full = _run(cfg, 2 * _N)
    first = _run(cfg, _N, **(dict(nt_out=1) if case == 'B_mfd' else {}))
    state = first.restart_state()
    assert state['mode'] == case[0] and state['time'] == _N * _DT
    assert ('regime_s' in state) == (case[0] == 'D')
    assert ('damping_weight' in state) == (case[0] != 'A')     # the adaptive default
    own_clock = case == 'C_plate'
    rest = _model(cfg, _N, state=_savez_round_trip(state),
                  **({} if own_clock else dict(to=state['time'])))
    handed = {k: np.copy(v) for k, v in rest._start_state.items()}
    rest.run()
    for k, v in handed.items():
        assert np.array_equal(v, rest._start_state[k]), f"the loop wrote into {k}"
    assert rest.H_out[0].sum() > 0.0, "the fixture carried no ice across the restart"
    _assert_continues(full, rest, same_clock=not own_clock)
    end_full, end_rest = full.restart_state(), rest.restart_state()
    assert end_full.keys() == end_rest.keys()
    for k in end_full:
        if k != 'time' or not own_clock:
            assert np.array_equal(end_full[k], end_rest[k]), k


@pytest.mark.parametrize('case, damping', [
    ('B_single', dict(inner_relax=1.0, inner_passes=1)),
    ('C_plate', dict(inner_relax=0.5, inner_passes=2))])
def test_initial_ice_thickness_is_the_restart_without_carried_devices(case, damping):
    """Without adaptive damping (omega damping at any relax and pass count
    carries nothing across steps), without the plate and outside mode D, the
    bed and the ice are the whole state, so ``initial_ice_thickness`` on the
    end bed (no noise) is the exact restart."""
    cfg = dict(_CASES[case], inner_damping='omega', flexure=False, **damping)
    full = _run(cfg, 2 * _N)
    state = _run(cfg, _N).restart_state()
    assert set(state) == {'mode', 'grid', 'boundary_status', 'time', 'topography',
                          'ice_thickness'}
    rest = _run(cfg, _N, initial_topography=state['topography'], noise_amplitude=0.0,
                initial_ice_thickness=state['ice_thickness'], to=state['time'])
    _assert_continues(full, rest)


def _bad(cfg, **ov):
    with pytest.raises(ValueError) as err:
        siim2d(dict(cfg, **ov))
    return str(err.value)


def test_start_state_validation():
    """Shapes, finite and in-range values, the mode, grid and boundaries the
    state was made on, whole groups and known keys raise at construction (the
    driver checks the groups too); the noise must be off under
    ``initial_ice_thickness``; the xsimlab driver refuses both parameters, an
    unrun model has no state, and a state is not stashed for save()."""
    cfg = dict(_CASES['B_single'], T=2 * _DT, nt=3, nt_out=3, noise_amplitude=0.0)
    shape = (_NXY, _NXY)
    ice = np.zeros(shape)
    assert 'shape' in _bad(cfg, initial_ice_thickness=np.zeros((3, 3)))
    assert 'initial_ice_thickness must be finite and >= 0' in _bad(
        cfg, initial_ice_thickness=np.full(shape, -1.0))
    assert 'finite' in _bad(cfg, initial_ice_thickness=np.full(shape, np.nan))
    assert 'mode A' in _bad(dict(cfg, mode='A'), initial_ice_thickness=ice)
    assert 'noise_amplitude=0' in _bad(dict(cfg, noise_amplitude=100.0),
                                       initial_ice_thickness=ice)
    m = siim2d(cfg)
    with pytest.raises(RuntimeError, match='run'):
        m.restart_state()
    m.run()
    state = m.restart_state()
    assert 'state' not in siim2d(dict(cfg, state=state))._user_params
    assert 'both' in _bad(cfg, state=state, initial_ice_thickness=ice)
    assert ('mode B run and cannot start a mode C run (mode C is mode B with '
            'carve_width=True)') in _bad(cfg, state=state, mode='C')
    assert 'mode B run and cannot start a mode A run' in _bad(cfg, state=state, mode='A')
    assert 'grid' in _bad(dict(cfg, Lx=2e4), state=state)
    assert 'boundary_status' in _bad(dict(cfg, boundary_status=['fixed_value'] * 4),
                                     state=state)
    assert 'dict' in _bad(cfg, state=[state])
    assert 'unknown' in _bad(cfg, state=dict(state, extra=1))
    assert 'missing' in _bad(cfg, state={k: v for k, v in state.items() if k != 'grid'})
    assert "'adaptive' group" in _bad(cfg, state={k: v for k, v in state.items()
                                                  if k != 'damping_weight'})
    for key, value, rule in (('ice_thickness', -1.0, '>= 0'), ('flexure_column', -1.0, '>= 0'),
                             ('damping_weight', 0.1, '>= 0.2 and <= 1'),
                             ('damping_weight', 1.5, '>= 0.2 and <= 1')):
        assert f"state[{key!r}] must be finite and {rule}" in _bad(
            cfg, state=dict(state, **{key: np.full(shape, value)}))
    regime = dict(regime_s=np.ones(shape), regime_icy=np.zeros(shape, dtype=bool),
                  regime_s_conf=np.ones(shape), regime_s_datum=np.ones(shape))
    for key in ('regime_s', 'regime_s_conf', 'regime_s_datum'):
        assert f"state[{key!r}] must be finite and >= 0 and <= 1" in _bad(
            cfg, state=dict(state, **dict(regime, **{key: np.full(shape, 1.5)})))
    assert 'boolean' in _bad(cfg, state=dict(state, **dict(regime, regime_icy=ice)))
    assert 'finite' in _bad(cfg, state=dict(state, topography=np.full(shape, np.inf)))
    assert 'shape' in _bad(cfg, state=dict(state, flexure_column=np.zeros(3)))
    for ov in (dict(state=state), dict(initial_ice_thickness=ice)):
        with pytest.raises(NotImplementedError, match='in-house-driver only'):
            siim2d(dict(cfg, **ov)).run(driver='xsimlab')
    loop_cfg = m._build_driver_config()
    loop_cfg.initial_state = {'damping_weight': np.ones(shape)}
    with pytest.raises(ValueError, match="'adaptive' group"):
        drv.run_loop(loop_cfg)
