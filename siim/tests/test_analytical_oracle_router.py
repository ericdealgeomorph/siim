"""S4 gate — the analytical steady-state oracle under the in-house router.

History (S4, plan decision notes 3/8): the plan's literal "2D SS vs analytical,
1e-9" is unachievable independent of router correctness — the 2D↔1D-analytical
geometry gap is inherently ~25% and siim's 2D landscapes are chaotic/multistable
under routing tie-breaks ("compare attractors, not snapshots"). The ratified
reading was ROUTER-INVARIANCE OF THE UNIQUE FIXED-POINT: on a mode-A, tie-free,
1D-reducible config the in-house D8 reproduced the fortran router's steady
state BIT-FOR-BIT (measured exactly 0.0 — surf_rms, bed_rms, and H), plus the
TRANSITIVE ANCHOR below. The fortran arm of that invariance test died with the
fortran router at the S5 flip (plan decision note 17) — its result is recorded
here and in the plan; what remains live is the transitive analytical anchor,
which pins the in-house router against the analytical solution at the existing
``test_baseline`` tolerance (and is now PIP: no stack import).

Mode B is EXCLUDED from any SS snapshot gate: it autogenic-cycles (MISI-style
limit cycles); the regenerated reference battery pins its full in-house
dynamics bit-for-bit instead.
"""
from siim.siim2d import siim as siim2d


def test_existing_analytical_oracle_config_inhouse_router():
    """The TRANSITIVE ANCHOR (S4 orchestrator checkpoint ruling): the EXISTING
    analytical-oracle config — ``test_baseline.test_1d_2d_parity_glacial``, the
    :206-region channel-vs-analytical comparison — run under the in-house
    router, asserted at its EXISTING tolerance (25%). Measured at S4: 14.64%
    in-house vs 14.66% fortran (fluvial arm 23.79% under both backends) —
    holding transitively from the bit-for-bit fixed-point router-invariance
    recorded in the plan's S4 notes."""
    from siim.siim1d import siim as siim1d
    from .test_baseline import (_small_2d, _params_1d_from_2d_channel,
                                _channel_rms_pct)
    p2d = _small_2d(zELA=1000, T=1e6, nt=501, nt_out=26,
                    initial_max_elevation=500, mode='A')
    m2 = siim2d(p2d)
    m2.run()
    ch = m2.extract_channel(basin_rank=0)
    m1 = siim1d(_params_1d_from_2d_channel(p2d, ch))
    m1.run()
    rms = _channel_rms_pct(ch.z[-1], m1, ch)
    assert rms < 25.0, (
        f"in-house router: glacial 1D-2D channel RMS = {rms:.1f}% "
        "(existing test_baseline tolerance 25%)")
