"""Sanity checks for the public fastscape process surface (``siim.fastscape``).

Intentionally light. siim is pre-1.0, so this guards against *accidental*
removal/breakage of the advertised names and against the assembly helper
drifting from siim2d — NOT against deliberate evolution. When you intentionally
change the surface, update ``ADVERTISED`` below.
"""
import pytest

pytest.importorskip('fastscape')
pytest.importorskip('xsimlab')
pytestmark = pytest.mark.adapter

from fastscape.models import basic_model

import siim.fastscape as F

# The process classes promoted as the intended-stable public surface (modes A,
# B and C + auxiliaries + every class glacial_processes wires in). Subset check:
# adding names is fine, dropping one breaks.
ADVERTISED = {
    "GlacialLaw", "GlacialFlowAccumulator", "GlacialSPLModeA", "GlacialSPLModeB",
    "GlacialSPLModeC", "GlacialSurfaceToErode", "TrunkSurfaceToErode",
    "SedimentTracker", "GlacialFlexure", "DinfFlowRouter",
    "D8FlowRouter", "GlacialBlockUplift", "HillslopeDiffusion", "InitialTopography",
}


def test_all_names_importable():
    for name in F.__all__:
        assert hasattr(F, name), f"{name!r} is in __all__ but not importable"


def test_advertised_surface_present():
    assert ADVERTISED <= set(F.__all__)
    assert {"glacial_processes", "glacial_model"} <= set(F.__all__)


@pytest.mark.parametrize("mode,carve,citizen_slot", [
    ('A', False, False),   # mode A -> stock surf2erode
    ('B', False, True),    # mode B citizen -> adds surf2erode
    ('B', True, True),     # mode B + carve (ModeC citizen) -> adds surf2erode
])
def test_glacial_processes_assembles(mode, carve, citizen_slot):
    ov = F.glacial_processes(mode=mode, carve=carve)
    # The always-present slots.
    assert ov['glacial_flow'] is F.GlacialFlowAccumulator
    assert ov['law'] is F.GlacialLaw
    assert 'glacial_spl' in ov and 'init_topography' in ov
    assert ('surf2erode' in ov) is citizen_slot
    # And it builds a real xsimlab model the way siim2d does.
    model = basic_model.drop_processes(['spl', 'drainage']).update_processes(ov)
    assert 'glacial_spl' in model and 'glacial_flow' in model


def test_routing_and_optional_slots():
    assert F.glacial_processes(routing='dinf')['flow'] is F.DinfFlowRouter
    assert 'flexure' in F.glacial_processes(flexure=True)
    assert 'sediment' in F.glacial_processes(sediment=True)


def test_case_insensitive_mode_and_validation():
    assert F.glacial_processes(mode='b') == F.glacial_processes(mode='B')
    # 'C' is now the B + sub-grid-carve alias (implies carve=True); it used to
    # be rejected. A genuinely-unknown mode still raises.
    assert F.glacial_processes(mode='C')['glacial_spl'] is F.GlacialSPLModeC
    with pytest.raises(ValueError):
        F.glacial_processes(mode='nonsense')
    with pytest.raises(ValueError):
        F.glacial_processes(routing='nonsense')


def test_glacial_model_builds():
    m = F.glacial_model(mode='B', carve=False)
    assert 'glacial_spl' in m and 'surf2erode' in m


def test_siim2d_uses_the_helper():
    """siim2d's private assembly must equal the public helper for the same
    config — the single-source-of-truth guarantee."""
    from siim.siim2d import siim
    m = siim({'mode': 'B', 'carve_width': False, 'flow_routing': 'single'})
    assert m._process_overrides() == F.glacial_processes(
        mode='B', carve=False, routing='single')


@pytest.mark.parametrize("mode,relax_name", [
    ('C', 'MODE_C_ROUTING_RELAX'),   # TrunkSurfaceToErode redeclares the default
    ('B', 'ROUTING_RELAX'),          # GlacialSurfaceToErode keeps the plain one
])
def test_surf2erode_routing_relax_default_seen_by_xsimlab(mode, relax_name):
    """The routing_relax default an un-set input resolves to, read through
    xsimlab itself (create_setup's default fill), not the class attribute —
    the subclass redeclaration must reach the model."""
    import xsimlab as xs
    from siim import constants
    ds = xs.create_setup(model=F.glacial_model(mode=mode), clocks={'t': [0, 1]})
    assert ds['surf2erode__routing_relax'].item() == getattr(constants, relax_name)



@pytest.mark.parametrize("kw,surf2erode", [
    (dict(mode='C'), 'TrunkSurfaceToErode'),                     # the mode-C standard
    (dict(mode='C', trunk_surface=False), 'GlacialSurfaceToErode'),  # explicit opt-out
    (dict(mode='B'), 'GlacialSurfaceToErode'),                   # plain B: standard off
    (dict(mode='B', carve=True), 'TrunkSurfaceToErode'),         # B + carve IS mode C
    (dict(mode='B', trunk_surface=True), 'TrunkSurfaceToErode'),  # explicit opt-in
])
def test_glacial_processes_resolves_mode_C_standard(kw, surf2erode):
    """The facade resolves trunk_surface=None by siim2d's rule (on for mode B +
    carve, off otherwise); an explicit value wins."""
    assert F.glacial_processes(**kw)['surf2erode'] is getattr(F, surf2erode)


def test_glacial_processes_mode_A_ignores_trunk_surface():
    """Mode A keeps fastscape's stock surf2erode whatever trunk_surface says."""
    assert 'surf2erode' not in F.glacial_processes(mode='A')
    assert F.glacial_processes(mode='A', trunk_surface=True) == F.glacial_processes(mode='A')
