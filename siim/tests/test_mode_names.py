"""User-facing `mode` names normalize to the internal A/B codes.

The legible names 'ice_surface' / 'bedrock+ice_thickness' are what users pass;
the kernels run on 'A' / 'B', which stay accepted as permanent aliases.
"""
import numpy as np
import pytest

from siim.constants import (normalize_mode, normalize_flow_routing, DEFAULT_MODE,
                            DEFAULT_MODE_2D)
from siim.siim1d import siim as siim1d
from siim.siim2d import siim as siim2d


@pytest.fixture(autouse=True)
def _run_under_both_drivers(both_drivers):
    """S3 (Map 4 §1 PARAM): every test in this file runs under BOTH drivers --
    the conftest ``both_drivers`` fixture patches ``constants.DRIVER_DEFAULT``
    (and, on the xsimlab arm, the B/C damping defaults to the adapter's one
    pass), so the existing assertions gate the in-house driver too."""



# --- normalize_mode (the single source) ------------------------------------

@pytest.mark.parametrize("value,code", [
    ('ice_surface', 'A'),
    ('bedrock+ice_thickness', 'B'),
    ('A', 'A'), ('B', 'B'),                 # short aliases
    ('C', 'C'), ('c', 'C'),                 # B + sub-grid-carve alias (case-insensitive)
    ('D', 'D'), ('d', 'D'),                 # C + the channel/sheet reading (case-insensitive)
    ('a', 'A'), ('b', 'B'), ('Ice_Surface', 'A'),   # every name in any case
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


@pytest.mark.parametrize("value,code", [
    ('single', 'single'), ('multiple', 'mfd'), ('infinite', 'dinf'),
    ('mfd', 'mfd'), ('dinf', 'dinf'),       # the router codes stay aliases
])
def test_normalize_flow_routing_maps_to_router_code(value, code):
    assert normalize_flow_routing(value) == code


def test_flow_routing_names_cover_the_routers():
    from siim._core.router import ROUTING_KINDS
    names = ('single', 'multiple', 'infinite')
    assert sorted(map(normalize_flow_routing, names)) == sorted(ROUTING_KINDS)


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
    m = siim2d({**_SMALL_2D, 'mode': 'C', 'inner_passes': 1, 'inner_relax': 1.0})   # the adapter's one pass
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


# --- Mode-C standard defaults (trunk_surface / widening_rate) and the damping --

def test_mode_C_standard_defaults_resolve(both_drivers):
    """Mode C (mode B + carve — including the bare 2D default) turns the mode-C
    standard ON: trunk_surface, widening_rate 3.0, and the mode-B/C damping
    (the adaptive preset)."""
    if both_drivers == 'xsimlab':
        pytest.skip('construction-time defaults; the xsimlab arm patches the B/C damping to the adapter\'s')
    for cfg in ({**_SMALL_2D, 'mode': 'C'}, _SMALL_2D):   # explicit C and bare default
        m = siim2d(cfg)
        assert m.mode == 'B' and m.carve_width is True
        assert m.trunk_surface is True
        # gate: the B/C default damping moved from routing_relax 0.5 (removed) to the
        # adaptive_capped preset (decisions/2026-11-04-bc-damping-omega-passes.md, private),
        # then to the adaptive preset (Eric, 2026-11-05)
        assert (m.inner_damping, m.inner_passes) == ('adaptive', 1)
        assert m.inner_relax == pytest.approx(0.2)
        assert m.widening_rate == 3.0


@pytest.mark.pin
@pytest.mark.parametrize('kw, want', [
    (dict(mode='B'), ('adaptive', 1, 0.2)),
    (dict(mode='C'), ('adaptive', 1, 0.2)),
    (dict(mode='B', inner_passes=3), ('adaptive', 3, 0.2)),           # passes compose
    (dict(mode='C', inner_damping='adaptive_capped'), ('adaptive_capped', 1, 1 / 3)),
    (dict(mode='C', inner_damping='omega'), ('omega', 2, 0.5)),       # the omega dials
    (dict(mode='C', inner_relax=0.25), ('omega', 2, 0.25)),           # inner_relax picks omega
    (dict(mode='B', inner_relax=1.0, inner_passes=1), ('omega', 1, 1.0)),   # undamped
    (dict(mode='C', inner_relax=1.0, inner_passes=1), ('omega', 1, 1.0)),
    (dict(mode='A'), ('omega', 1, 1.0)),
], ids=['B', 'C', 'B-passes', 'C-capped', 'C-omega', 'C-relax', 'B-undamped', 'C-undamped', 'A'])
def test_mode_BC_damping_defaults_resolve(kw, want, both_drivers):
    """Modes B and C default to MODE_BCD_INNER_DAMPING, whose floor rides in
    inner_relax; an explicit inner_relax picks 'omega', which fills each
    unset dial with (MODE_BC_INNER_PASSES, MODE_BC_INNER_RELAX), so an
    explicit (1, 1.0) is the undamped model. Mode A keeps the plain defaults
    (mode D's own: test_mode_d)."""
    if both_drivers == 'xsimlab':
        pytest.skip('construction-time defaults; the xsimlab arm patches the B/C damping to the adapter\'s')
    m = siim2d({**_SMALL_2D, **kw})
    # gate: the default became the adaptive_capped preset (Eric, 2026-11-04), then the
    # adaptive one (Eric, 2026-11-05), so the pin reads it too; the explicit-preset case
    # now asks for the capped one
    assert (m.inner_damping, m.inner_passes) == want[:2]
    assert m.inner_relax == pytest.approx(want[2])


@pytest.mark.parametrize('kw, match', [
    (dict(inner_damping='adaptive', inner_relax=0.5), "'omega' preset's dial"),
    (dict(mode='A', inner_damping='adaptive_capped'), "requires mode"),
    (dict(inner_damping='fast'), "inner_damping must be one of"),
])
def test_inner_damping_rejects(kw, match):
    with pytest.raises(ValueError, match=match):
        siim2d({**_SMALL_2D, **kw})


def test_plain_mode_B_keeps_standard_off():
    """Plain mode B (no carve) and mode A keep trunk_surface off (the mode-C
    standard applies only to carved runs)."""
    b = siim2d({**_SMALL_2D, 'mode': 'B', 'carve_width': False})
    # gate: routing_relax was removed (Eric, 2026-11-04)
    assert b.trunk_surface is False
    a = siim2d({**_SMALL_2D, 'mode': 'A'})
    assert a.trunk_surface is False


def test_mode_C_explicit_flags_win():
    """An explicit user value overrides the mode-C standard, in both directions."""
    off = siim2d({**_SMALL_2D, 'mode': 'C', 'trunk_surface': False,
                  'widening_rate': 1.0})
    # gate: routing_relax was removed (Eric, 2026-11-04)
    assert off.trunk_surface is False
    assert off.widening_rate == 1.0
    # plain mode B can still opt IN explicitly.
    on = siim2d({**_SMALL_2D, 'mode': 'B', 'carve_width': False,
                 'trunk_surface': True})
    assert on.trunk_surface is True


def test_mode_A_rejects_explicit_standard_flags():
    with pytest.raises(ValueError):
        siim2d({**_SMALL_2D, 'mode': 'A', 'trunk_surface': True})


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


def test_resolved_damping_round_trips():
    """A model's resolved damping, passed back as user parameters, rebuilds
    the same model (the adaptive floor in inner_relax is accepted as itself)."""
    m = siim2d({**_SMALL_2D, 'mode': 'C'})
    again = siim2d({**_SMALL_2D, 'mode': 'C', 'inner_damping': m.inner_damping,
                    'inner_relax': m.inner_relax, 'inner_passes': m.inner_passes})
    assert (again.inner_damping, again.inner_relax, again.inner_passes) == (
        m.inner_damping, m.inner_relax, m.inner_passes)
