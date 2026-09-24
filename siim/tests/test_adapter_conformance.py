"""S5 adapter-conformance gate — siim stays composable with fastscape (conda).

The ``siim.fastscape`` ``@xs.process`` shells, composed the way an EXTERNAL
fastscape user would — ``basic_model.drop_processes(['spl', 'drainage'])
.update_processes(glacial_processes(...))`` via :func:`glacial_model`, driven
by ``xs.create_setup(...).xsimlab.run`` directly (NOT through
``siim2d._run_xsimlab``) — must reproduce a standalone in-house-driver run.

Since the S5 flip the adapter model runs the SAME in-house numerics + router
the standalone driver uses (only the orchestration differs), and the S3 gate
proved the two orchestrations bit-for-bit on identical configs — so this gate
asserts ``np.array_equal`` (the strongest that holds; the plan's rtol=1e-9
floor is subsumed) on elevation + ice thickness at every output frame.
"""
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip('fastscape')
pytest.importorskip('xsimlab')
pytestmark = pytest.mark.adapter

from siim.siim2d import siim as siim2d

# A plain mode-B config (no carve, no trunk surface, routing_relax 0 — siim2d's
# mode-C standard is pinned OFF so both arms describe the same physics).
_P = dict(
    U=1e-3, zELA=150, beta=1e-2, P=2, alpha_g=12, Ko=2e-6, n=1, ce=1e-4,
    nu=2, sliding_law='power', lambda_p=500, k=0.9,
    T=3e4, nt=31, nt_out=5, Lx=2e4, Ly=2e4, nx=31, ny=31, seed=7,
    mode='B', trunk_surface=False, routing_relax=0.0,
    boundary_status=['fixed_value', 'fixed_value', 'looped', 'looped'],
    initial_max_elevation=800, progress_bar=False)


def test_adapter_import_does_not_patch_xarray_process_wide(tmp_path):
    """Importing the optional adapter must not rewrite private xarray APIs."""
    code = """
from xarray.core import utils

original = utils.did_you_mean
import siim.fastscape  # noqa: F401
assert utils.did_you_mean is original
"""
    result = subprocess.run(
        [sys.executable, '-c', code], cwd=tmp_path,
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('routing', ['single', 'dinf'])
def test_shells_in_real_fastscape_model_match_standalone(routing):
    """glacial_model() composed + run by hand through xsimlab == the standalone
    in-house run, bit-for-bit at every output frame."""
    from siim.fastscape import glacial_model

    # --- standalone arm (the shipped default path) --------------------------
    m = siim2d(dict(_P, flow_routing=routing))
    m.run(driver='inhouse')

    # --- adapter arm: an external user's composition ------------------------
    model = glacial_model(mode='B', routing=routing)
    zb_adapter, H_adapter = _run_adapter(model, m, {
        'glacial_spl__carve_width': m.carve_width,
        'glacial_spl__widening_rate': m.widening_rate,
        'surf2erode__routing_relax': m.routing_relax,
    })

    np.testing.assert_array_equal(zb_adapter, m.zb_out)
    np.testing.assert_array_equal(H_adapter, m.H_out)
    assert m.H_out.max() > 1.0, "config must glaciate for the gate to bite"


def _adapter_inputs(m):
    """The xsimlab inputs every adapter arm needs, taken from siim2d run ``m``
    (no carve / widening / routing-relax / trunk inputs: callers add those)."""
    return {
        'grid__shape': [m.grid_ny, m.grid_nx],
        'grid__length': [m.Ly, m.Lx],
        'boundary__status': m.boundary_status,
        'init_topography__elevation_init': m._make_initial_topo(),
        'init_topography__seed': m.seed,
        'init_topography__noise_amplitude': m.noise_amplitude,
        'uplift__rate': m._make_uplift_field(),
        'glacial_flow__runoff': m.P,
        'glacial_flow__beta': m.beta,
        'glacial_flow__zELA': m.zELA,
        'glacial_flow__width_hack_k': m.width_hack_k,
        'glacial_flow__width_hack_p': m.width_hack_p,
        'law__n': m.n, 'law__nu': m.nu, 'law__Ko': m.Ko, 'law__Ac': m.Ac,
        'law__lambda_p': m.lambda_p, 'law__lambda_c': m.lambda_c,
        'law__tau_c': m.tau_c, 'law__coulomb_clamp': m.coulomb_clamp,
        'law__m': m.m, 'law__mu': m.mu, 'law__alpha_g': m.alpha_g,
        'law__sliding_law': m.sliding_law, 'law__ce': m.ce,
        'law__H_diffusivity': m.H_diffusivity,
        'law__hc_over_H': m.hc_over_H,
        'glacial_spl__border_bed_uplift': m._make_border_bed_uplift(),
        'glacial_spl__bl': m.bl,
        'glacial_spl__flotation_gate': m.flotation_gate,
        'glacial_spl__flotation_ramp': m.flotation_ramp,
        'glacial_spl__parallel_erode': m.parallel_erode,
        'glacial_spl__ice_thickness': np.zeros((m.grid_ny, m.grid_nx)),
        'diffusion__diffusivity': m.D,
    }


def _run_adapter(model, m, extra_inputs=None):
    """Run ``model`` through xsimlab by hand on ``m``'s clocks and inputs (plus
    ``extra_inputs``). Returns (topography, H) at every output frame."""
    import xsimlab as xs
    ds_in = xs.create_setup(
        model=model,
        clocks={'time': m.t_out, 'tstep': m.t},
        master_clock='tstep',
        input_vars={**_adapter_inputs(m), **(extra_inputs or {})},
        output_vars={'topography__elevation': 'time',
                     'glacial_spl__ice_thickness': 'time'},
    )
    with model:
        ds_out = ds_in.xsimlab.run(
            encoding={'topography__elevation': {'compressor': None},
                      'glacial_spl__ice_thickness': {'compressor': None}})
    return (ds_out['topography__elevation'].values,       # mode B/C: bed IS topo
            ds_out['glacial_spl__ice_thickness'].values)


def _run_facade_mode_c(m, **model_kw):
    """glacial_model(mode='C', **model_kw) with no mode-C inputs, so every
    mode-C-standard knob is the facade's own default. Returns (bed, H)."""
    from siim.fastscape import glacial_model
    return _run_adapter(glacial_model(mode='C', **model_kw), m)


@pytest.mark.pin
def test_facade_mode_C_is_native_mode_C():
    """glacial_model(mode='C') with no mode-C inputs == siim2d mode C (the
    in-house default driver), bit-for-bit at every output frame: the facade
    resolves the mode-C standard (trunk surface, routing_relax 0.5,
    widening_rate 3) exactly as siim2d does."""
    from siim import constants

    params = {k: v for k, v in _P.items()
              if k not in ('mode', 'trunk_surface', 'routing_relax')}
    m = siim2d(dict(params, mode='C'))
    m.run(driver='inhouse')
    assert (m.carve_width, m.trunk_surface, m.routing_relax) == (
        True, constants.MODE_C_TRUNK_SURFACE, constants.MODE_C_ROUTING_RELAX)

    zb, H = _run_facade_mode_c(m)
    np.testing.assert_array_equal(zb, m.zb_out)
    np.testing.assert_array_equal(H, m.H_out)
    assert m.H_out.max() > 1.0, "config must glaciate for the gate to bite"
    # The gate bites on the standard: the old facade composition (carve, no
    # trunk surface, relax 0) diverges on this config.
    zb_old, _ = _run_facade_mode_c(m, trunk_surface=False)
    assert not np.array_equal(zb_old, m.zb_out)
