"""MFD in valleys, the xsimlab-adapter half (plan-2026-09-22-mfd-guard-lift.md).

The in-house routers route on the ``router_surface`` group's member (the trunk
V of :class:`~siim.fastscape.TrunkSurfaceToErode`) when a process publishes one
and on ``elevation`` otherwise; the accumulator and the diffusion keep
``elevation``. xsimlab captures process methods at class decoration, so every
test here EXECUTES a model; nothing monkeypatches a process.
"""
import numpy as np
import pytest

pytest.importorskip('fastscape')
xs = pytest.importorskip('xsimlab')
pytestmark = pytest.mark.adapter

from siim import constants as _constants
from siim._core.router import InhouseRouter
from siim._core.step import routing_surface
from siim.fastscape import MfdFlowRouter, glacial_model
from siim.siim2d import siim as siim2d

from .test_adapter_conformance import _direct_inputs
from .test_driver_vs_xsimlab import _assert_pair
from .test_mfd_routing import _VALLEY_RUN as _VALLEY

_SHORT = dict(_VALLEY, T=4e3, nt=5, nt_out=2)


def _direct_run(m, model, **over):
    """Run ``model`` (a DIRECT composition) through xsimlab on the inputs an
    external user would write for the configured siim2d ``m``.
    ``surf2erode__trunk_dip_k`` is never passed."""
    iv = _direct_inputs(m)
    iv.update(over)
    ds_in = xs.create_setup(
        model=model, clocks={'time': m.t_out, 'tstep': m.t},
        master_clock='tstep', input_vars=iv,
        output_vars={'topography__elevation': 'time',
                     'glacial_spl__ice_thickness': 'time'})
    return ds_in.xsimlab.run(model=model)


# --- acceptance 2: the guards the processes own raise at RUN time --------------

@pytest.mark.parametrize("over,msg", [
    ({'glacial_spl__closure_flux': 'cell'}, "closure_flux"),
    ({'grid__length': [2 * _SHORT['Ly'], _SHORT['Lx']]}, "square cells"),
    ({'glacial_spl__slope_normalise': False}, "slope_normalise"),
], ids=['cell_closure', 'rectangular_cells', 'unnormalised'])
def test_mfd_process_guards_raise_when_the_model_runs(over, msg):
    """siim2d and the facade cannot see these inputs on a direct composition,
    so the erosion process (closure) and the router process (grid spacing)
    refuse them when the model executes."""
    m = siim2d(dict(_SHORT))
    with pytest.raises(ValueError, match=msg):
        _direct_run(m, glacial_model(mode='B', routing='mfd'), **over)


def test_mfd_mode_a_raises_when_a_hand_built_model_runs():
    """glacial_processes refuses MFD + mode A up front; a model composed around
    it (mode A with the MFD router swapped in) is refused by the erosion
    process when it runs, the cell closure being mode A's only one."""
    m = siim2d(dict(_SHORT, mode='A', flow_routing='single'))
    model = glacial_model(mode='A').update_processes({'flow': MfdFlowRouter})
    with pytest.raises(ValueError, match="closure_flux"):
        _direct_run(m, model)


@pytest.mark.parametrize("mode,over,msg", [
    ('B', {'glacial_spl__closure_flux': 'CELL'}, "closure_flux"),
    ('B', {'glacial_spl__carve_width': True}, "never carves"),
    ('A', {'glacial_spl__closure_flux': 'transect'}, "closure_flux"),
    ('C', {'surf2erode__trunk_dip_k': float('nan')}, "trunk_dip_k"),
    ('C', {'surf2erode__trunk_dip_k': -1.0}, "trunk_dip_k"),
], ids=['closure_typo', 'carve_on_mode_b', 'closure_on_mode_a', 'dip_nan', 'dip_negative'])
def test_the_adapter_refuses_what_siim2d_refuses(mode, over, msg):
    """A direct composition used to run these silently: an unknown closure
    string ran the transect closure, the carve on a non-carving class and a
    closure on mode A were ignored, a NaN dip poisoned the router surface and a
    negative one turned the V into a ridge (audit #45, #46, #49)."""
    m = siim2d(dict(_SHORT, flow_routing='single', mode=mode,
                    trunk_surface=mode == 'C'))
    with pytest.raises(ValueError, match=msg):
        _direct_run(m, glacial_model(mode=mode, routing='single'), **over)


def test_the_adapter_refuses_lake_depth():
    """flow__lake_depth read a fortran field no in-house router fills, so it
    came back all zeros; it now says so (audit #48)."""
    m = siim2d(dict(_SHORT, flow_routing='single', mode='B'))
    model = glacial_model(mode='B', routing='single')
    ds_in = xs.create_setup(
        model=model, clocks={'time': m.t_out, 'tstep': m.t}, master_clock='tstep',
        input_vars=_direct_inputs(m), output_vars={'flow__lake_depth': 'time'})
    with pytest.raises(NotImplementedError, match="lake_depth"):
        ds_in.xsimlab.run(model=model)


# --- acceptance 2 + 3 + B4b: construct and run every provider x router ---------

_SMOKE = [(name, params, routing)
          for name, params in [('A', dict(mode='A')),
                               ('B', dict(mode='B')),
                               ('B_trunk', dict(mode='B', trunk_surface=True)),
                               ('C', dict(mode='C', trunk_surface=True))]
          for routing in ('single', 'dinf', 'mfd')
          if not (name == 'A' and routing == 'mfd')]   # refused by the facade


@pytest.mark.parametrize('name,params,routing', _SMOKE,
                         ids=[f'{n}-{r}' for n, _, r in _SMOKE])
def test_every_provider_runs_under_every_router(name, params, routing):
    """Direct composition, a few steps: A, plain B, B + trunk surface and C
    under D8, D-inf and MFD. The dip is never passed, so the facade-built model
    runs on the router's default: 1.0 under MFD, 0.6 otherwise. A composition
    built fully by hand, bypassing glacial_processes, gets the process default
    0.6 under MFD too — a facade-bypass gap of the same class as the mode-A one
    the plan deferred."""
    m = siim2d(dict(_SHORT, flow_routing=routing, **params))
    model = glacial_model(mode=params['mode'], routing=routing,
                          trunk_surface=m.trunk_surface)
    ds = _direct_run(m, model)
    H = ds['glacial_spl__ice_thickness'].values
    assert np.isfinite(ds['topography__elevation'].values).all()
    assert np.isfinite(H).all() and (H >= 0.0).all() and H[-1].max() > 0.0
    if m.trunk_surface:
        want = (_constants.TRUNK_DIP_K_MFD if routing == 'mfd'
                else _constants.TRUNK_DIP_K)
        assert float(ds['surf2erode__trunk_dip_k']) == want


# --- acceptance 3: the facade-built model gets the RESOLVED dip ----------------

@pytest.mark.parametrize('routing,dip', [('mfd', 1.0), ('single', 0.6)])
def test_siim2d_passes_the_resolved_dip_and_matches_inhouse(routing, dip):
    """siim2d's xsimlab path hands the trunk-surface process the resolved dip
    (read back from the process input) and the two drivers agree byte-for-byte
    on this newly admitted mode-C configuration."""
    a, _ = _assert_pair(dict(_VALLEY, mode='C', flow_routing=routing,
                             trunk_surface=None))
    assert a.trunk_surface and a.trunk_dip_k == dip
    assert float(a.ds_out['surf2erode__trunk_dip_k']) == dip


# --- acceptance 4: the adapter FIREWALL, instrumented --------------------------

class _SurfaceSpy(xs.RuntimeHook):
    """Records, per step, what surf2erode published and what the router made
    of it. ``pre`` snapshots the step-start bed and the LAGGED H (glacial_spl
    overwrites ice_thickness within run_step); ``post`` reads the step's
    surfaces, uplift and flow graph once every process has run."""

    def __init__(self, hc, routing):
        super().__init__()
        self.hc, self.routing = hc, routing
        self.router = self.wiring = None
        self.dynamic_exact, self.n_router_only = [], []
        self.uplifted = []
        self.graph_from_router_input, self.graph_from_elevation = [], []

    @xs.runtime_hook('run_step', 'model', 'pre')
    def pre(self, model, context, state):
        self.zb0 = np.array(state[('topography', 'elevation')])
        self.H_lag = np.array(state[('glacial_spl', 'ice_thickness')])

    @xs.runtime_hook('run_step', 'model', 'post')
    def post(self, model, context, state):
        if self.wiring is None:
            self.wiring = {(p, v): model[p].__xsimlab_state_keys__.get(v)
                           for p, v in (('flow', 'router_surface'),
                                        ('flow', 'elevation'),
                                        ('glacial_flow', 'surface'),
                                        ('diffusion', 'elevation'))}
            self.router = InhouseRouter(
                state[('grid', 'shape')], state[('boundary', 'border_status')],
                self.routing, state[('grid', 'dx')], state[('grid', 'dy')])
        elev = state[('surf2erode', 'elevation')]
        router_in = state.get(('surf2erode', 'routing_elevation'), elev)
        uplift = state[('tectonics', 'surface_upward')]
        self.uplifted.append(bool(np.max(uplift) > 0.0))
        zb = self.zb0 + uplift                              # post-uplift bed
        self.dynamic_exact.append(
            np.array_equal(elev, routing_surface(zb, self.hc, self.H_lag)))
        self.n_router_only.append(int((router_in != elev).sum()))
        rec = state[('flow', 'receivers')]
        self.graph_from_router_input.append(
            np.array_equal(self.router(router_in).receivers, rec))
        self.graph_from_elevation.append(
            np.array_equal(self.router(elev).receivers, rec))


@pytest.mark.parametrize('routing', ['single', 'dinf', 'mfd'])
@pytest.mark.parametrize('trunk', [True, False], ids=['trunk', 'no_trunk'])
def test_router_input_is_the_V_and_the_accumulator_input_is_dynamic(
        routing, trunk):
    """On an ACTIVE footprint (alpha_g*H/2 > dx), with uplift: the router's
    input (the ``router_surface`` member) differs from the accumulator's and
    the diffusion's (``elevation``) on the footprint, ``elevation`` is exactly
    the dynamic ``zb + hc*H`` on every cell at every step, and the router's
    graph is the one its own input produces (not the dynamic surface's). With
    the trunk surface off the group is empty and the router reads
    ``elevation``."""
    params = dict(_VALLEY, mode='C', flow_routing=routing, trunk_surface=trunk,
                  inner_passes=1, inner_relax=1.0, D=1.0)   # the adapter's one pass
    m = siim2d(params)
    spy = _SurfaceSpy(m.hc_over_H, routing)
    m.run(driver='xsimlab', hooks=[spy])
    dynamic = ('surf2erode', 'elevation')
    assert spy.wiring == {
        ('flow', 'router_surface'):
            [('surf2erode', 'routing_elevation')] if trunk else [],
        ('flow', 'elevation'): dynamic,
        ('glacial_flow', 'surface'): dynamic,
        ('diffusion', 'elevation'): dynamic}
    # gate: routing_relax (H_eff != raw H) was removed 2026-11-04
    assert all(spy.uplifted)
    assert all(spy.dynamic_exact)
    assert all(spy.graph_from_router_input)
    if trunk:
        assert sum(spy.n_router_only) > 0, spy.n_router_only
        assert not all(spy.graph_from_elevation)
    else:
        assert spy.n_router_only == [0] * len(spy.n_router_only)
