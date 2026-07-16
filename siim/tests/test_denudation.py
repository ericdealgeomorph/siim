"""Denudation sourcing for sediment + flexure.

The mass-accounting consumers — `SedimentTracker` and `GlacialFlexure` — read
`GlacialSPLBase.denudation`, the mode-aware TRUE rock removed:

  * mode B -> delta-zb (tracked bed lowering, incl. sub-grid carve), and
  * mode A -> delta-zs (the surface lowering; mode A is hc-invariant, so the
    derived bed change delta-zs - hc*delta-H must NOT be used).

How `erosion` (the elevation-update channel) relates to `denudation`:

  * mode A (GlacialSPLModeA): `erosion` is the ice-SURFACE change delta-zs (a
    surface-replace), so it differs from `denudation` (Δzb) by hc*delta-H under
    evolving ice.
  * citizen mode B / C (GlacialSPLModeB, GlacialSPLModeC = B + carve): `erosion`
    == `denudation` == Δzb (a genuine rock-removal height); fastscape composes
    zb_new = zb + uplift - erosion. The surface change delta-zs is then carried
    by the reconstructed z_out = zb + hc*H.

These pin that split, that carve volume reaches the sediment budget, and that
flexure is wired to `GlacialFlexure`.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from siim.siim2d import siim as siim2d                                  # noqa: E402


@pytest.fixture(autouse=True)
def _run_under_both_drivers(both_drivers):
    """S3 (Map 4 §1 PARAM): every test in this file runs under BOTH drivers --
    the conftest ``both_drivers`` fixture patches ``constants.DRIVER_DEFAULT``,
    so the existing assertions gate the in-house driver too."""



def _glac(**ov):
    """Strongly glaciated transient (low zELA, fine grid) where ice grows
    (dH/dt != 0, so delta-zb and delta-zs diverge) AND the sub-grid carve bites
    — the footprint R = alpha_g*H/2 spans several cells at this resolution.
    Mirrors the carve-active config in test_dinf_routing, on SFR."""
    base = dict(
        U=1e-3, zELA=150, beta=1e-2, P=2, alpha_g=12, Ko=2e-6, n=1, ce=1e-4,
        nu=2, sliding_law='power', lambda_p=500, k=0.9,
        T=6e4, nt=61, nt_out=13, Lx=2e4, Ly=2e4, nx=41, ny=41, seed=7,
        boundary_status=['fixed_value'] * 4, initial_max_elevation=800,
        mode='B', flow_routing='single', progress_bar=False)
    base.update(ov)
    return base


def _has_ice(m):
    return m.H_out.max() > 1.0


def test_modeA_denudation_is_surface_change():
    """Mode A: denudation == the ice-surface erosion delta-zs (= erosion_rate*dt)
    EXACTLY, even with ice present (dH/dt != 0). It must NOT be the derived
    delta-zb, which would break mode A's hc-invariance."""
    m = siim2d(_glac(mode='A'))
    m.run()
    assert _has_ice(m), "precondition: run must glaciate"
    assert np.allclose(m.denudation_out, m.erosion_rate_out * m.dt,
                       rtol=1e-6, atol=1e-9)


def test_modeB_no_ice_denudation_equals_surface():
    """Mode B with no ice anywhere (dH/dt == 0): delta-zb == delta-zs, so
    denudation == erosion_rate*dt."""
    m = siim2d(_glac(mode='B', carve_width=False, zELA=1e5))
    m.run()
    assert not _has_ice(m), "precondition: this run must stay ice-free"
    assert np.allclose(m.denudation_out, m.erosion_rate_out * m.dt,
                       rtol=1e-6, atol=1e-9)


def test_modeB_citizen_erosion_is_denudation():
    """Citizen mode B (Fork B; no carve): the erosion-group height IS the true
    rock denudation delta-zb. fastscape composes zb_new = zb + uplift - erosion,
    so erosion == denudation == erosion_rate*dt EXACTLY (no surface-replace) —
    the genuine erosion height that lets flexure/sediment read the bed and admits
    a second erosion-group member. (Legacy mode B set erosion = the surface
    change delta-zs, which differed from denudation by hc*delta-H.)"""
    m = siim2d(_glac(mode='B', carve_width=False))
    m.run()
    assert _has_ice(m), "precondition: run must glaciate"
    assert np.allclose(m.denudation_out, m.erosion_rate_out * m.dt,
                       rtol=1e-6, atol=1e-9)


def test_modeB_citizen_surface_change_diverges_from_bed():
    """Citizen mode B with ice: the SURFACE change delta-zs (= diff of the
    reconstructed z_out = zb + hc*H) diverges from the bed change delta-zb
    (denudation) by hc*delta-H where the ice thickness evolves — the physics is
    intact, just carried by z_out now that erosion tracks the bed, not the
    surface."""
    m = siim2d(_glac(mode='B', carve_width=False))
    m.run()
    assert _has_ice(m), "precondition: run must glaciate"
    # delta-zs per output interval vs delta-zb per interval (denudation is the
    # per-STEP bed drop; compare like-for-like by summing denudation is harder,
    # so contrast the cumulative bed vs surface drop, which differ by hc*delta-H).
    dz_surface = m.z_out[0] - m.z_out[-1]      # net surface lowering
    dz_bed = m.zb_out[0] - m.zb_out[-1]        # net bed lowering
    gap = np.abs(dz_surface - dz_bed)
    assert gap.max() > 1e-2, \
        "delta-zs and delta-zb should diverge by hc*delta-H where ice changed"


def test_sediment_carve_adds_to_yield():
    """Sub-grid carve removes extra rock; that volume must enter the sediment
    budget. With carve ON, both the denudation source and the OUTLET-DELIVERED
    sediment (flux integrated over the fixed borders) exceed the carve-OFF
    twin.

    The routed quantity is border-integrated, not domain-summed (S5, plan
    decision note 20): the domain+time sum is path-length weighted (each
    eroded volume counted once per traversed cell), so it can flip when the
    twins self-organise different-but-equivalent networks (routing
    multistability) — measured under inhouse_d8: source ratio 1.0306 ==
    border-flux ratio 1.0306 (conservation exact) while the domain-summed
    proxy read 0.99."""
    # Isolate the carve: pin the mode-C standard flags off so the carve-on /
    # carve-off twins differ ONLY in carve_width.
    base = _glac(mode='B', track_sediment=True, trunk_surface=False,
                 routing_relax=0.0)
    on = siim2d({**base, 'carve_width': True, 'widening_rate': 0.0})
    on.run()
    off = siim2d({**base, 'carve_width': False})
    off.run()
    assert _has_ice(on) and _has_ice(off), "precondition: both runs must glaciate"

    # Source: total positive rock removed over the run.
    den_on = np.maximum(on.denudation_out, 0.0).sum()
    den_off = np.maximum(off.denudation_out, 0.0).sum()
    assert den_on > den_off, (den_on, den_off)

    # Routed: the tracker delivers that extra rock to the outlets (all four
    # borders are fixed_value in _glac, so this is the total leaving the
    # domain).
    def border_flux(m):
        s = m.sediment_flux_out
        return (s[:, 0, :].sum() + s[:, -1, :].sum()
                + s[:, 1:-1, 0].sum() + s[:, 1:-1, -1].sum())
    assert np.isfinite(on.sediment_flux_out).all()
    assert border_flux(on) > border_flux(off), (border_flux(on), border_flux(off))


# (The fortran flexure/diffusion arms died at the 0.9.1 standalone flip — the
# in-house solve is the only backend; the S2 Kelvin twin remains its oracle.)

def test_flexure_acts_on_topography():
    """flexure=True still measurably reshapes the steady landscape. Pinned to
    erosion-only (ice_load=False) so it stays a pure erosional-flexure
    regression."""
    p = _glac(U=2e-3)
    off = siim2d(p)
    off.run()
    on = siim2d({**p, 'flexure': True, 'e_thickness': 20e3, 'ice_load': False})
    on.run()
    assert np.all(np.isfinite(on.z_out[-1])), "non-finite elevation with flexure on"
    assert np.abs(on.z_out[-1] - off.z_out[-1]).max() > 1.0, \
        "flexure=True produced no change — not coupling into the topography"


@pytest.mark.adapter
def test_flexure_slot_is_glacial_flexure():
    """flexure=True wires GlacialFlexure (load from denudation) into the
    adapter's flexure slot (the Map-4 class-identity assert; conda)."""
    from siim.fastscape import GlacialFlexure
    on = siim2d({**_glac(U=2e-3), 'flexure': True, 'e_thickness': 20e3,
                 'ice_load': False})
    assert on._process_overrides()['flexure'] is GlacialFlexure


@pytest.mark.adapter
def test_facade_flexure_backend_chain_inhouse():
    """The facade's flexure chain resolves in-house end to end: the installed
    class, the variable DEFAULT an un-set input var resolves to, and the
    plate-solve callable that default resolves to. (The fortran arm of this
    chain died at the 0.9.1 standalone flip.)"""
    import attr
    from types import SimpleNamespace
    from siim import constants as _constants
    from siim._core.flexure import flexure as inhouse_flexure
    from siim.fastscape import GlacialFlexure, glacial_processes

    assert glacial_processes(
        flexure=True, numerics_backend='inhouse')['flexure'] is GlacialFlexure
    assert attr.fields_dict(
        GlacialFlexure)['numerics_backend'].default == _constants.NUMERICS_BACKEND
    assert GlacialFlexure._flexure_solve(
        SimpleNamespace(numerics_backend='inhouse')) is inhouse_flexure


@pytest.mark.parametrize('bad', ['fortan', 'fortran'])
def test_flexure_backend_invalid_raises(bad):
    """An invalid numerics_backend raises the clear ValueError at the siim2d
    entry point — both the typo ('fortan', the historical regression) and the
    RETIRED 'fortran' option (deleted at the 0.9.1 standalone flip; its
    rejection is the deletion's regression pin)."""
    with pytest.raises(ValueError, match="numerics_backend"):
        siim2d(_glac(numerics_backend=bad))


@pytest.mark.adapter
@pytest.mark.parametrize('bad', ['fortan', 'fortran'])
def test_flexure_backend_invalid_raises_facade(bad):
    """The same rejection at the adapter facade and the GlacialFlexure solve
    resolution (a typo must not silently run in-house)."""
    from types import SimpleNamespace
    from siim.fastscape import GlacialFlexure, glacial_processes

    with pytest.raises(ValueError, match="numerics_backend"):
        GlacialFlexure._flexure_solve(SimpleNamespace(numerics_backend=bad))
    with pytest.raises(ValueError, match="numerics_backend"):
        glacial_processes(numerics_backend=bad)


@pytest.mark.parametrize('bad', ['fortan', 'fortran'])
def test_router_backend_invalid_raises(bad):
    """An invalid router_backend raises the clear ValueError at the siim2d entry
    point — both a typo and the RETIRED 'fortran' option (deleted at the 0.9.1
    standalone flip). The twin of ``test_flexure_backend_invalid_raises``; the
    deletion's regression pin for the router arm (S5 review, note 21)."""
    with pytest.raises(ValueError, match="router_backend"):
        siim2d(_glac(router_backend=bad))


@pytest.mark.adapter
@pytest.mark.parametrize('bad', ['fortan', 'fortran'])
def test_router_backend_invalid_raises_facade(bad):
    """The same router_backend rejection at the adapter facade
    (``glacial_processes``) — a retired/typo backend must not silently wire the
    fortran slot that no longer exists."""
    from siim.fastscape import glacial_processes

    with pytest.raises(ValueError, match="router_backend"):
        glacial_processes(router_backend=bad)


# ---------------------------------------------------------------------------
# Ice load in flexure (true glacial isostatic adjustment).
#
# The ice load's DIRECT flexural deflection is isolated via the per-step
# `rebound` output: summing rebound over the run for the ice_load=True vs
# =False twins, their difference is the ice-load contribution to the total
# deflection — clean of the coupled-divergence noise that contaminates a raw
# zb comparison. A thin plate (e_thickness=5 km) gives a ~1 m signal at this
# domain size; full per-step output (nt_out=nt) makes the sum exact.
# ---------------------------------------------------------------------------

def _glac_flex(**ov):
    """`_glac` with flexure on, a thin (flexible) plate, and full per-step
    output so the cumulative rebound sums exactly."""
    base = dict(flexure=True, e_thickness=5e3, U=2e-3, T=3e4, nt=31, nt_out=31)
    base.update(ov)
    return _glac(**base)


def _ice_deflection(on, off):
    """Ice-load contribution to the total flexural deflection: the difference
    of the summed per-step rebounds between the ice-load and erosion-only twins."""
    return on.rebound_out.sum(axis=0) - off.rebound_out.sum(axis=0)


def test_ice_load_subsides_under_ice():
    """The ice load deflects the plate DOWN under the ice. Across the glaciated
    footprint, the ice-load contribution to the cumulative deflection is
    negative (subsidence) at essentially every iced cell."""
    base = _glac_flex()
    on = siim2d({**base, 'ice_load': True})
    on.run()
    off = siim2d({**base, 'ice_load': False})
    off.run()
    assert _has_ice(on) and _has_ice(off), "precondition: both runs must glaciate"

    icy = on.H_out[-1] > 50.0
    assert icy.sum() > 10, "precondition: a meaningful ice footprint"
    defl = _ice_deflection(on, off)[icy]
    assert defl.mean() < -0.1, defl.mean()                # net subsidence under ice
    assert (defl < 0).mean() > 0.9, (defl < 0).mean()     # ~all iced cells subside


def test_ice_load_reversible_on_deglaciation():
    """Linear-elastic reversibility: grow ice then melt it off (zELA ramped far
    above the terrain). Once deglaciated, the ice-load deflection telescopes back
    to ~0 (sum of per-step dH = H_final - H_initial = 0), so the ice contribution
    to the total deflection vanishes — far smaller than at peak glaciation."""
    nt = 41
    zELA = np.concatenate([np.full(nt // 2, 150.0),
                           np.full(nt - nt // 2, 5000.0)])
    base = _glac_flex(T=4e4, nt=nt, nt_out=nt, zELA=zELA)
    on = siim2d({**base, 'ice_load': True})
    on.run()
    off = siim2d({**base, 'ice_load': False})
    off.run()
    assert on.H_out[nt // 2 - 1].max() > 50.0, "precondition: glaciates first"
    assert on.H_out[-1].max() < 1.0, "precondition: fully deglaciated by the end"

    peak = np.abs(on.rebound_out[:nt // 2].sum(0) - off.rebound_out[:nt // 2].sum(0)).max()
    final = np.abs(_ice_deflection(on, off)).max()
    assert peak > 0.2, peak                 # a real ice-load deflection at peak
    assert final < 0.2 * peak, (final, peak)  # recovered once the ice is gone


def test_ice_load_finite_stable_and_default_on():
    """Adding the ice load keeps the run finite and bounded, and ice_load
    defaults to True (full GIA when flexure=True)."""
    assert siim2d(_glac_flex()).ice_load is True
    m = siim2d(_glac_flex(ice_load=True))
    m.run()
    assert _has_ice(m)
    assert np.all(np.isfinite(m.z_out)) and np.all(np.isfinite(m.zb_out))
    assert np.all(np.isfinite(m.H_out)) and np.all(np.isfinite(m.rebound_out))
    assert np.abs(m.zb_out[-1]).max() < 1e5, "deflection must stay geologically sane"
