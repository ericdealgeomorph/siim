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

from fastscape.models import basic_model      # noqa: E402

import siim.fastscape as F                    # noqa: E402

# The process classes promoted as the intended-stable public surface (modes A
# and B + auxiliaries). Subset check: adding names is fine, dropping one breaks.
ADVERTISED = {
    "GlacialLaw", "GlacialFlowAccumulator", "GlacialSPLModeA", "GlacialSPLModeB",
    "GlacialSPLModeC", "GlacialSurfaceToErode", "TrunkSurfaceToErode",
    "SedimentTracker", "GlacialFlexure", "DinfFlowRouter",
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
