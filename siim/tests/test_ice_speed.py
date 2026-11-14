"""Post-hoc ice speed (``siim._core.solvers.ice_speeds``, ``siim.ice_speed``)
and the ``landscape(field='bedrock+speed')`` layer that draws it.

The speed is not stored; it is the flux closure run backwards on the stored
(H, Qg) pair. The round trip below drives the model's own scalar closure
forward (Qg, S -> H) and checks that the inversion returns the closure's
U = Qg/(alpha_g H^2) and its deformation speed (2A/5) kt (rho g H S)^3 H.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from siim import constants as C
from siim._core.solvers import (LAW_COULOMB, LAW_EFFEXP, LAW_POWER,
                                _modeb_closure, ice_speeds)

ALPHA, AC, LAMBDA_P, LAMBDA_C, TAU_C = 6.0, 2.5e-24, 300.0, 10.0, 1.2e5


@pytest.mark.parametrize('law, code', [('power', LAW_POWER),
                                       ('coulomb', LAW_COULOMB),
                                       ('eff-exp', LAW_EFFEXP)])
def test_speed_inverts_the_flux_closure(law, code):
    rng = np.random.default_rng(3)
    Q = 10 ** rng.uniform(4, 10, 400)          # m^3/yr
    S = 10 ** rng.uniform(-3.5, -0.7, 400)     # surface slope
    cg = C.cg_prefactor(ALPHA, AC)
    H = np.array([_modeb_closure(code, True, s, q, 0.0, 1.0, cg, LAMBDA_P,
                                 LAMBDA_C, TAU_C, C.RHO_ICE_G,
                                 C.COULOMB_CLAMP) for s, q in zip(S, Q)])
    assert np.all(H > 0)
    us, U = ice_speeds(H, Q, law, ALPHA, AC, LAMBDA_P, LAMBDA_C, TAU_C)
    np.testing.assert_allclose(ALPHA * H ** 2 * U, Q, rtol=1e-12)
    if law == 'eff-exp':                       # no sliding split
        np.testing.assert_array_equal(us, U)
        return
    ud_true = 0.4 * AC * C.KT * (C.RHO_ICE_G * H * S) ** 3 * H
    # us = U + ud/4; compare ud on the scale of U (sliding can dominate)
    assert np.max(np.abs(4 * (us - U) - ud_true) / U) < 1e-8
    assert np.all((U <= us) & (us <= 1.25 * U))


def test_speed_is_zero_off_the_ice_and_at_outlets():
    H, Q = np.array([0.0, 100.0, 50.0, 50.0]), np.array([1e6, 0.0, 1e6, 1e6])
    us, U = ice_speeds(H, Q, 'coulomb', ALPHA, AC, LAMBDA_P, LAMBDA_C, TAU_C)
    assert us[0] == us[1] == U[0] == U[1] == 0.0 and us[2] > 0 and us[3] > 0
    # cell 3 drains to itself: an outlet, whose H is copied, not solved
    us, U = ice_speeds(H, Q, 'coulomb', ALPHA, AC, LAMBDA_P, LAMBDA_C, TAU_C,
                       receivers=np.array([1, 2, 3, 3]))
    assert us[2] > 0 and us[3] == U[3] == 0.0


_COULOMB = dict(
    U=1e-3, P=1, beta=1e-2, Ko=1e-6, n=1, ce=1e-5, nu=2,
    sliding_law='coulomb', alpha_g=8, zELA=400.0, T=1e5,
    Lx=3e4, Ly=3e4, nx=31, ny=31, nt=9, nt_out=4, D=1e-3, seed=7,
    boundary_status=['fixed_value'] * 4, initial_max_elevation=1200,
    noise_amplitude=60, mode='C', flow_routing='single',
    inner_passes=1, inner_relax=1.0,   # the undamped pass (B/C damp by default); _RELAX is the damped arm
    save_closure_pair=True,            # the opt-in rows that make the speed exact
    progress_bar=False)


@pytest.fixture(scope='module')
def _coulomb_model():
    from siim.siim2d import siim as siim2d
    m = siim2d(dict(_COULOMB))
    m.run()
    return m


def test_landscape_speed_field_renders(_coulomb_model):
    m = _coulomb_model
    us, U = m.ice_speed(-1)
    assert (m.H_out[-1] > 0).any() and np.all(us >= U)
    for kw in (dict(style='smooth'), dict(style='raw'),
               dict(trunk_display='ribbons')):    # opt-in speed ribbons
        fig, _ = m.plot.landscape(field='bedrock+speed', **kw)
        labels = [a.get_ylabel() for a in fig.axes]
        assert 'Ice surface speed (m/yr)' in labels
        plt.close(fig)
    # gate: the speed field's own ice_time_avg refusal went with the knob
    # (removed 0.10.0); test_plotting_review pins the refusal for every field
    plt.close('all')


@pytest.mark.parametrize('n_workers', [1, 2])
def test_animate_landscape_speed_field(_coulomb_model, n_workers, tmp_path,
                                       monkeypatch):
    """Serial and parallel paths; the workers need Qg_out and the law
    constants shipped for this field only."""
    import matplotlib.animation as anm
    if not anm.FFMpegWriter.isAvailable():
        pytest.skip("ffmpeg not available")
    monkeypatch.chdir(tmp_path)
    out = _coulomb_model.plot.animate_landscape(
        path=f'speed{n_workers}', n_workers=n_workers, field='bedrock+speed',
        oversample=2)
    assert out.endswith('.mp4') and os.path.getsize(out) > 0
    plt.close('all')


_RELAX = dict(inner_passes=2, inner_relax=0.5)


@pytest.mark.parametrize('kw, rows', [
    (dict(), ('Qg',)),                                   # transect_min, omega 1
    (dict(_RELAX, closure_flux='cell'), ('H',)),
    (dict(_RELAX), ('Qg', 'H')),                         # transect + omega < 1
    (dict(closure_flux='cell'), ()),                     # the stored pair is the closure's
], ids=['transect', 'relax', 'both', 'cell'])
def test_ice_speed_inverts_the_closure_pair_the_kernel_solved(kw, rows,
                                                              monkeypatch):
    """At every output frame ice_speed reads the (flux, thickness) the kernel's
    closure solved in that step's last pass, captured here by a spy on the
    kernel: U*alpha_g*H**2 returns that flux to 1e-12 at every icy cell and no
    warning fires. A transect closure_flux closes on a flux that is not
    Qg_out and inner_relax < 1 commits a blend that is not the closure's H,
    so the run stores the closure's own row exactly where it differs."""
    import warnings
    from siim._core import driver as drv
    from siim._core.step import run_modeB_kernel
    from siim.siim2d import siim as siim2d
    calls = []

    def spy(zb_flat, H_flat, Q, *a, **k):
        out = run_modeB_kernel(zb_flat, H_flat, Q, *a, **k)
        calls.append((np.array(Q, dtype=float).reshape(-1), H_flat.copy()))
        return out
    monkeypatch.setattr(drv, 'run_modeB_kernel', spy)
    m = siim2d(dict(_COULOMB, **kw))
    m.run()
    assert (m.closure_Qg_out is not None) == ('Qg' in rows)
    assert (m.closure_H_out is not None) == ('H' in rows)
    passes = m.inner_passes
    for j, k in enumerate(np.searchsorted(m.t, m.t_out)):   # frame -> step
        Q, F = (a.reshape(m.H_out.shape[1:]) for a in
                calls[min(k, m.nt - 2) * passes + passes - 1])
        # (frame 0 closes on the lagged H = 0, where the transect is the cell)
        if j and 'Qg' in rows:
            assert not np.allclose(m.Qg_out[j], Q), "vacuous: Qg_out is the closure flux"
        if 'H' in rows:
            assert not np.allclose(m.H_out[j], F), "vacuous: H_out is the closure H"
        # the pair ice_speed reads is the kernel's, bit for bit
        assert np.array_equal(
            (m.Qg_out if m.closure_Qg_out is None else m.closure_Qg_out)[j], Q)
        assert np.array_equal(
            (m.H_out if m.closure_H_out is None else m.closure_H_out)[j], F)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            us, U = m.ice_speed(j)
        icy = U > 0
        rec = m.receivers_out[j]
        assert np.array_equal(icy, (F > 0) & (Q > 0)
                              & (rec != np.arange(rec.size).reshape(rec.shape)))
        if j:
            assert icy.any()
        np.testing.assert_allclose(U[icy] * m.alpha_g * F[icy] ** 2, Q[icy],
                                   rtol=1e-12)
        assert np.all(us >= U)


def test_ice_speed_warns_on_a_run_saved_without_the_closure_pair(_coulomb_model):
    """A transect run whose dataset lacks the closure-flux row (saved without it,
    or before the row existed) falls back to the committed Qg_out, and says so."""
    m = _coulomb_model
    ds = m.ds_out
    try:
        m.ds_out = ds.drop_vars('glacial_spl__closure_ice_flux')
        m._unpack_outputs()
        assert m.closure_Qg_out is None
        with pytest.warns(UserWarning, match='approximate'):
            m.ice_speed(-1)
    finally:
        m.ds_out = ds
        m._unpack_outputs()


def test_closure_pair_rows_are_opt_in():
    """By default (save_closure_pair False) a transect, damped run stores
    neither closure row, and ice_speed falls back to Qg_out / H_out with a
    warning that names the flag."""
    from siim.siim2d import siim as siim2d
    kw = {k: v for k, v in _COULOMB.items() if k != 'save_closure_pair'}
    m = siim2d(dict(kw, **_RELAX))
    m.run()
    assert m.closure_Qg_out is None and m.closure_H_out is None
    assert not any(v.startswith('glacial_spl__closure') for v in m.ds_out.data_vars)
    with pytest.warns(UserWarning, match='save_closure_pair=True'):
        us, U = m.ice_speed(-1)
    us0, U0 = ice_speeds(m.H_out[-1], m.Qg_out[-1], m.sliding_law, m.alpha_g, m.Ac,
                         m.lambda_p, m.lambda_c, m.tau_c, receivers=m.receivers_out[-1])
    np.testing.assert_array_equal(U, U0)       # the fallback is the committed pair
    with pytest.warns(UserWarning, match='save_closure_pair=True'):
        fig, _ = m.plot.landscape(i=-1, field='bedrock+speed', style='raw')
    plt.close(fig)
    with pytest.raises(ValueError, match='save_closure_pair'):
        siim2d(dict(kw, save_closure_pair='False'))


def test_landscape_speed_reads_the_closure_pair(_coulomb_model, monkeypatch):
    """field='bedrock+speed' inverts the same pair as ice_speed."""
    from siim.plotting import landscape as ls
    m, seen, real = _coulomb_model, [], ls.ice_speeds
    monkeypatch.setattr(ls, 'ice_speeds', lambda H, Q, *a, **k: (
        seen.append((H, Q)), real(H, Q, *a, **k))[1])
    fig, _ = m.plot.landscape(i=-1, field='bedrock+speed', style='raw')
    plt.close(fig)
    (H, Q), = seen
    assert np.array_equal(Q, m.closure_Qg_out[-1])
    assert np.array_equal(H, m.H_out[-1])
