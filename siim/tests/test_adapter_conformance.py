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

from siim.siim2d import siim as siim2d     # noqa: E402

# A plain mode-B config (the facade's explicit-default semantics: no carve, no
# trunk surface, routing_relax 0 — siim2d's mode-C magic is pinned OFF so both
# arms describe the same physics).
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
    import xsimlab as xs
    from siim.fastscape import glacial_model

    # --- standalone arm (the shipped default path) --------------------------
    m = siim2d(dict(_P, flow_routing=routing))
    m.run(driver='inhouse')

    # --- adapter arm: an external user's composition ------------------------
    model = glacial_model(mode='B', routing=routing)
    ds_in = xs.create_setup(
        model=model,
        clocks={'time': m.t_out, 'tstep': m.t},
        master_clock='tstep',
        input_vars={
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
            'glacial_spl__carve_width': m.carve_width,
            'glacial_spl__widening_rate': m.widening_rate,
            'glacial_spl__bl': m.bl,
            'glacial_spl__flotation_gate': m.flotation_gate,
            'glacial_spl__flotation_ramp': m.flotation_ramp,
            'glacial_spl__parallel_erode': m.parallel_erode,
            'glacial_spl__ice_thickness': np.zeros((m.grid_ny, m.grid_nx)),
            'surf2erode__routing_relax': m.routing_relax,
            'diffusion__diffusivity': m.D,
        },
        output_vars={'topography__elevation': 'time',
                     'glacial_spl__ice_thickness': 'time'},
    )
    with model:
        ds_out = ds_in.xsimlab.run(
            encoding={'topography__elevation': {'compressor': None},
                      'glacial_spl__ice_thickness': {'compressor': None}})

    zb_adapter = ds_out['topography__elevation'].values     # mode B: bed IS topo
    H_adapter = ds_out['glacial_spl__ice_thickness'].values
    np.testing.assert_array_equal(zb_adapter, m.zb_out)
    np.testing.assert_array_equal(H_adapter, m.H_out)
    assert m.H_out.max() > 1.0, "config must glaciate for the gate to bite"
