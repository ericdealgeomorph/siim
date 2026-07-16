"""Mode C — the citizen mode-B bed+H class plus the sub-grid glacier-width
carve (:class:`siim.fastscape.GlacialSPLModeC`), the ``carve=True`` default path.

Stage-1 port gates (citizen-refactor methodology):

* **Gate A** — ModeC with carving OFF is :class:`GlacialSPLModeB` bit-for-bit
  (its non-carve path must not diverge from the parent citizen); both routings.
* **Gate B** — ModeC with carving ON conserves rock at U=0: the tracked bed
  moves by exactly the reported ``denudation`` (nothing leaks into the
  sediment/flexure source), and the carve genuinely deepens the bed vs the
  no-carve twin.

The one-off machine-precision agreement with the retired surface-replace class
lived in the port verification (equal to it bit-for-bit on the first U=0 step;
they then diverge under mode-B cycling + that class's surface-hc*H round-trip,
exactly as the citizen refactor documented). It cannot be a permanent test — the
class it compared against is deleted.
"""
import numpy as np
import pytest

from siim.siim2d import siim as siim2d


@pytest.fixture(autouse=True)
def _run_under_both_drivers(both_drivers):
    """S3 (Map 4 §1 PARAM): every test in this file runs under BOTH drivers --
    the conftest ``both_drivers`` fixture patches ``constants.DRIVER_DEFAULT``,
    so the existing assertions gate the in-house driver too."""



class _ForceModeC(siim2d):
    """siim2d that swaps :class:`GlacialSPLModeC` into the ``glacial_spl`` slot
    regardless of ``carve_width`` — the only way to reach 'ModeC with carve
    OFF', since the public wiring maps carve-off to :class:`GlacialSPLModeB`.
    Keeps the citizen ``surf2erode`` (both mode-B citizens route on zb + hc*H)."""

    def _process_overrides(self):
        # Adapter-only seam (xsimlab driver); lazy import keeps this file
        # PIP-collectable. Under the in-house driver the override is inert
        # (the driver reads cfg.carve), so gate A's forced-ModeC arm is
        # exercised on the xsimlab (adapter) driver arm.
        from siim.fastscape import GlacialSPLModeC, GlacialSurfaceToErode
        ov = super()._process_overrides()
        ov['glacial_spl'] = GlacialSPLModeC
        ov.setdefault('surf2erode', GlacialSurfaceToErode)
        return ov


def _cfg(**ov):
    """A glaciating, carve-biting config (fine grid + wide channels so the
    footprint R = alpha_g*H/2 spans several cells — see the carve-needs-a-fine-
    grid note). Mirrors test_dinf_routing / test_carve_width."""
    base = dict(
        U=1e-3, zELA=200, beta=1e-2, P=1, alpha_g=8, Ko=2e-6, n=1, ce=1e-4,
        nu=2, sliding_law='power', lambda_p=500, k=0.9,
        T=1e5, nt=101, nt_out=25, Lx=2e4, Ly=2e4, nx=41, ny=41, seed=7,
        boundary_status=['fixed_value'] * 4, initial_max_elevation=800,
        mode='B', progress_bar=False,
        # Pin the mode-C standard flags off: these gates isolate the citizen /
        # carve behaviour, so the carve-on and carve-off twins differ only in
        # carve_width (Gate A is already carve-off; Gate B compares carve twins).
        trunk_surface=False, routing_relax=0.0)
    base.update(ov)
    return base


@pytest.mark.parametrize("routing", ['single', 'dinf'])
def test_gateA_modeC_carve_off_equals_modeB(routing):
    """Gate A: ModeC with carve OFF == the citizen ModeB, BIT-FOR-BIT, over a
    multi-step run (both SFR and D-inf). ModeC.run_step gates the carve on
    ``self._carve``, so carve_width=False must fall straight through to the
    parent citizen behaviour with no drift."""
    cfg = _cfg(carve_width=False, flow_routing=routing)
    ref = siim2d(cfg); ref.run()             # GlacialSPLModeB (citizen)
    got = _ForceModeC(cfg); got.run()        # GlacialSPLModeC, carve gated off
    assert ref._citizen_mode_b and got._citizen_mode_b
    for name in ('zb_out', 'z_out', 'H_out', 'denudation_out', 'erosion_rate_out'):
        np.testing.assert_array_equal(
            getattr(got, name), getattr(ref, name),
            err_msg=f"{name} diverged from ModeB (routing={routing})")


@pytest.mark.parametrize("routing", ['single', 'dinf'])
def test_gateB_modeC_carve_on_conserves_at_U0(routing):
    """Gate B: ModeC with carve ON at U=0 (no uplift, no hillslope diffusion)
    accounts for every metre of bed change in ``denudation`` — the sediment /
    flexural-unloading source is exact, carve deepening included. With
    ``topo_new = topo + uplift - erosion``, ``erosion = denudation`` and
    U=0/D=0, the per-step bed change is exactly ``-denudation``:

        ``zb_out[k+1] - zb_out[k] == -denudation_out[k]``   (machine precision).

    And the carve must genuinely bite (bed pulled below the no-carve twin)."""
    cfg = _cfg(U=0.0, border_bed_uplift=0.0, D=0.0, zELA=150, alpha_g=12,
               initial_max_elevation=1200, T=6e4, nt=61, nt_out=61,
               carve_width=True, widening_rate=0.0, flow_routing=routing)
    m = siim2d(cfg); m.run()
    assert m._citizen_mode_b
    assert m.H_out.max() > 50.0, "precondition: run must glaciate"

    # Conservation: the whole per-step bed change is the reported denudation.
    dzb = np.diff(m.zb_out, axis=0)
    np.testing.assert_array_equal(dzb, -m.denudation_out[:-1])

    # The carve genuinely deepens the bed somewhere (footprint wider than the
    # channel) vs the no-carve twin at the same U=0 config.
    m0 = siim2d({**cfg, 'carve_width': False}); m0.run()
    assert (m.zb_out[-1] - m0.zb_out[-1]).min() < -1.0, \
        "carve must deepen the bed below the no-carve twin"
