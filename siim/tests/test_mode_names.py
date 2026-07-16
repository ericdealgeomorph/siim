"""User-facing `mode` names normalize to the internal A/B codes.

The legible names 'ice_surface' / 'bedrock+ice_thickness' are what users pass;
the kernels run on 'A' / 'B', which stay accepted as permanent aliases.
"""
import numpy as np
import pytest

from siim.constants import normalize_mode, DEFAULT_MODE, DEFAULT_MODE_2D
from siim.siim1d import siim as siim1d
from siim.siim2d import siim as siim2d


@pytest.fixture(autouse=True)
def _run_under_both_drivers(both_drivers):
    """S3 (Map 4 §1 PARAM): every test in this file runs under BOTH drivers --
    the conftest ``both_drivers`` fixture patches ``constants.DRIVER_DEFAULT``,
    so the existing assertions gate the in-house driver too."""



# --- normalize_mode (the single source) ------------------------------------

@pytest.mark.parametrize("value,code", [
    ('ice_surface', 'A'),
    ('bedrock+ice_thickness', 'B'),
    ('A', 'A'), ('B', 'B'),                 # short aliases (exact match)
    ('C', 'C'), ('c', 'C'),                 # B + sub-grid-carve alias (case-insensitive)
])
def test_normalize_mode_maps_to_internal_code(value, code):
    assert normalize_mode(value) == code


def test_default_mode_is_two_state():
    # Shared default (1D) is plain mode B; 2D defaults to mode C (carving on).
    assert DEFAULT_MODE == 'bedrock+ice_thickness'
    assert normalize_mode(DEFAULT_MODE) == 'B'
    assert DEFAULT_MODE_2D == 'C'


@pytest.mark.parametrize("bad", ['surface', 'bedrock', '', None, 2])
def test_normalize_mode_rejects_unknown(bad):
    with pytest.raises(ValueError):
        normalize_mode(bad)


# --- 1D model accepts the names at the user boundary -----------------------

@pytest.mark.parametrize("value,code", [
    ('ice_surface', 'A'),
    ('bedrock+ice_thickness', 'B'),
    ('A', 'A'), ('B', 'B'),
])
def test_siim1d_accepts_mode_names(value, code):
    m = siim1d({'mode': value, 'progress_bar': False})
    assert m.mode == code
    assert m._mode_B_active == (code == 'B')


def test_siim1d_default_is_two_state():
    assert siim1d({'progress_bar': False}).mode == 'B'


def test_siim1d_rejects_unknown_mode():
    with pytest.raises(ValueError):
        siim1d({'mode': 'surface', 'progress_bar': False})


def test_siim1d_rejects_mode_C_as_2d_only():
    # Mode C (sub-grid width carving) has no 1D analogue — the 1D model has no
    # width machinery, so it raises rather than silently degrading to mode A/B.
    with pytest.raises(ValueError, match="2D-only"):
        siim1d({'mode': 'C', 'progress_bar': False})


# --- 2D model: names normalize, and the carve opt-out still keys off them ---

_SMALL_2D = {'nx': 11, 'ny': 11, 'Lx': 1e4, 'Ly': 1e4, 'T': 1e4, 'nt': 11,
             'nt_out': 3, 'progress_bar': False}


def test_siim2d_default_is_two_state():
    # The 2D default is mode C: internal mode 'B' with carving ON (Eric,
    # 2026-07-07 — carving is the flagship default, but only for the DEFAULT).
    m = siim2d(_SMALL_2D)
    assert m.mode == 'B'
    assert m.carve_width is True


def test_siim2d_ice_surface_coerces_carving_off():
    # 'ice_surface' (single-state, mode A) has no bed memory, so carving is off
    # (carving is opt-in and mode A cannot carve — no raise without explicit True).
    m = siim2d({**_SMALL_2D, 'mode': 'ice_surface'})
    assert m.mode == 'A'
    assert m.carve_width is False


def test_siim2d_explicit_carve_with_ice_surface_still_raises():
    with pytest.raises(ValueError):
        siim2d({**_SMALL_2D, 'mode': 'ice_surface', 'carve_width': True})


def test_siim2d_explicit_mode_B_does_not_carve():
    # An UNQUALIFIED mode='B' is plain bed memory: NO carving (Eric, 2026-07-07).
    # Carving is opt-in — via mode='C' or an explicit carve_width=True.
    assert siim2d({**_SMALL_2D, 'mode': 'B'}).carve_width is False
    assert siim2d({**_SMALL_2D, 'mode': 'B', 'carve_width': True}).carve_width is True


def test_siim2d_legacy_alias_still_works():
    assert siim2d({**_SMALL_2D, 'mode': 'B'}).mode == 'B'


# --- Mode 'C' = the B + sub-grid-carve alias (2D only) ----------------------

def test_siim2d_mode_C_resolves_to_B_with_carve_on():
    m = siim2d({**_SMALL_2D, 'mode': 'C'})
    assert m.mode == 'B'                    # not a new dynamical mode
    assert m.carve_width is True


@pytest.mark.adapter
def test_siim2d_mode_C_wires_modeC_slot():
    from siim.fastscape import GlacialSPLModeC
    m = siim2d({**_SMALL_2D, 'mode': 'C'})
    assert m._process_overrides()['glacial_spl'] is GlacialSPLModeC


def test_siim2d_mode_C_rejects_explicit_carve_false():
    with pytest.raises(ValueError):
        siim2d({**_SMALL_2D, 'mode': 'C', 'carve_width': False})


def test_siim2d_mode_C_equals_bare_default_bit_for_bit():
    # C IS the 2D default configuration (mode B + carve_width on), just named.
    # Pin the seed so the initial-topography noise matches, then assert identity
    # between explicit mode='C' and the bare default.
    cfg = {**_SMALL_2D, 'seed': 7}
    m_c = siim2d({**cfg, 'mode': 'C'}); m_c.run()
    m_b = siim2d(cfg); m_b.run()            # bare default == mode C (carve on)
    np.testing.assert_array_equal(m_c.zb_out, m_b.zb_out)
    np.testing.assert_array_equal(m_c.z_out, m_b.z_out)


# --- Mode-C standard defaults (trunk_surface / routing_relax / widening_rate) --

def test_mode_C_standard_defaults_resolve():
    """Mode C (mode B + carve — including the bare 2D default) turns the mode-C
    standard ON: trunk_surface, routing_relax 0.5, widening_rate 3.0."""
    for cfg in ({**_SMALL_2D, 'mode': 'C'}, _SMALL_2D):   # explicit C and bare default
        m = siim2d(cfg)
        assert m.mode == 'B' and m.carve_width is True
        assert m.trunk_surface is True
        assert m.routing_relax == 0.5
        assert m.widening_rate == 3.0


def test_plain_mode_B_keeps_standard_off():
    """Plain mode B (no carve) and mode A keep trunk_surface off and
    routing_relax 0.0 (the mode-C standard applies only to carved runs)."""
    b = siim2d({**_SMALL_2D, 'mode': 'B', 'carve_width': False})
    assert b.trunk_surface is False and b.routing_relax == 0.0
    a = siim2d({**_SMALL_2D, 'mode': 'A'})
    assert a.trunk_surface is False and a.routing_relax == 0.0


def test_mode_C_explicit_flags_win():
    """An explicit user value overrides the mode-C standard, in both directions."""
    off = siim2d({**_SMALL_2D, 'mode': 'C', 'trunk_surface': False,
                  'routing_relax': 0.0, 'widening_rate': 1.0})
    assert off.trunk_surface is False and off.routing_relax == 0.0
    assert off.widening_rate == 1.0
    # plain mode B can still opt IN explicitly.
    on = siim2d({**_SMALL_2D, 'mode': 'B', 'carve_width': False,
                 'trunk_surface': True, 'routing_relax': 0.3})
    assert on.trunk_surface is True and on.routing_relax == 0.3


def test_mode_A_rejects_explicit_standard_flags():
    with pytest.raises(ValueError):
        siim2d({**_SMALL_2D, 'mode': 'A', 'trunk_surface': True})
    with pytest.raises(ValueError):
        siim2d({**_SMALL_2D, 'mode': 'A', 'routing_relax': 0.5})


# --- glacial_processes front door (fastscape citizens) ----------------------

@pytest.mark.adapter
def test_glacial_processes_mode_C_wires_modeC():
    from siim.fastscape import glacial_processes, GlacialSPLModeC
    assert glacial_processes(mode='C')['glacial_spl'] is GlacialSPLModeC
    assert glacial_processes(mode='c')['glacial_spl'] is GlacialSPLModeC  # case-insensitive


@pytest.mark.adapter
def test_glacial_processes_mode_C_rejects_explicit_carve_false():
    from siim.fastscape import glacial_processes
    with pytest.raises(ValueError):
        glacial_processes(mode='C', carve=False)
