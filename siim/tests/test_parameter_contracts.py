"""Public parameter-contract tests that do not require a model run."""

import warnings

import numpy as np
import pytest

from siim import constants

from siim.escarpment import siim_escarpment
from siim.siim1d import siim as siim1d
from siim.siim2d import siim as siim2d


def _construct(model_cls, **params):
    grid = dict(nx=21) if model_cls is siim1d else dict(nx=5, ny=5)
    return model_cls({**dict(T=1.0, nt=2, nt_out=2, progress_bar=False), **grid, **params})


@pytest.mark.parametrize('model_cls', [siim1d, siim2d], ids=['1d', '2d'])
@pytest.mark.parametrize('sliding_law', ['power', 'coulomb'])
def test_exact_law_mu_override_is_retained_but_warns(model_cls, sliding_law):
    with pytest.warns(UserWarning, match="retained for the analytical"):
        model = _construct(model_cls, sliding_law=sliding_law, mu=0.4)

    assert model.mu == 0.4
    assert model.phi == pytest.approx(0.4 / model.nu)


@pytest.mark.parametrize('model_cls', [siim1d, siim2d], ids=['1d', '2d'])
def test_effexp_mu_override_has_no_exact_law_warning(model_cls):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = _construct(model_cls, sliding_law='eff-exp', mu=0.4)

    assert model.mu == 0.4
    assert not any("exact numerical sliding law" in str(w.message)
                   for w in caught)



@pytest.mark.parametrize('params, match', [
    (dict(nt=1, nt_out=1), 'nt must be >= 2'),
    (dict(T=0.0), 'T must be finite and later than to'),
    (dict(T=-3e3), 'T must be finite and later than to'),
    (dict(nx=2), 'nx and ny must be >= 3'),
    (dict(Ly=0.0), 'Ly must be finite and > 0'),
    (dict(coulomb_clamp=1.0), 'coulomb_clamp must be finite and > 0 and < 1'),
    (dict(coulomb_clamp=-0.5), 'coulomb_clamp must be'),
    (dict(D=-50.0), 'D must be finite and >= 0'),
    (dict(D=np.nan), 'D must be finite and >= 0'),
    (dict(D=10 ** 400), 'D must be finite and >= 0; got an int too large'),   # was an OverflowError
    (dict(flotation_ramp=np.nan), 'flotation_ramp'),
    (dict(H_diffusivity=np.nan), 'H_diffusivity must be finite'),
    (dict(flexure=True, e_thickness=-1e3), 'e_thickness must be'),
    (dict(flexure=True, asthen_density=0.0), 'asthen_density must be'),
    (dict(flexure=True, lithos_density=np.where(np.eye(5, dtype=bool), np.nan, 2800.0)),
     'lithos_density must be finite and > 0; got 5 of 25 cells'),
    (dict(flexure=True, e_thickness=np.full((5, 5), 35e3)), 'must be scalars'),
    (dict(nu=-1.0), 'nu must be'),
    (dict(ell=0.0), 'ell must be'),
    (dict(lambda_p=0.0), 'lambda_p must be'),
    (dict(lambda_c=-1e3), 'lambda_c must be'),
    (dict(tau_c=0.0), 'tau_c must be'),
])
def test_2d_rejects_bad_inputs_at_construction(params, match):
    """Each of these used to construct, then run silently wrong (no ice, a NaN
    bed, anti-diffusion, time running backwards) or die later on an opaque
    KeyError or ZeroDivisionError (audit #3, #4, #8, #29, #38, #40)."""
    with pytest.raises(ValueError, match=match):
        _construct(siim2d, **params)


@pytest.mark.parametrize('params', [dict(wave_width=0.0), dict(wave_velocity=0.0),
                                    dict(wave_velocity=-15e-3)])
def test_escarpment_rejects_a_degenerate_wave(params):
    """A zero width made the wave's rate 0/0; a zero or negative velocity
    deposited nothing or subsided instead of raising the plateau (audit #29)."""
    with pytest.raises(ValueError, match='wave_'):
        _construct(siim_escarpment, uplift_type='wave', delta_h=800, **params)


def test_2d_accepts_the_closed_ends():
    """The inclusive bounds are real settings: D = 0 turns hillslope diffusion
    off, Te = 0 is Airy isostasy, flotation_ramp = 0 the hard gate."""
    m = _construct(siim2d, D=0.0, e_thickness=0.0, flotation_ramp=0.0, H_diffusivity=0.0, flexure=True)
    assert (m.D, m.e_thickness, m.flotation_ramp, m.H_diffusivity) == (0.0, 0.0, 0.0, 0.0)


def test_1d_rms_vs_analytical_uses_the_models_datum():
    """With hc_over_H overridden, a model planted exactly on the analytical
    reported a 226 m bed rms: the model bed used hc_over_H, the analytical's
    1.5 (audit #60). The analytical is now built on the model's datum
    (ED-183), so both sides sit on it with no re-datum."""
    m = _construct(siim1d, mode='A', hc_over_H=1.0)
    a = m.analytical
    H = (a.surface - a.bed) / a.hc_over_H          # the analytical's bed is on 1.0 now (ED-183)
    m.z_out, m.H_out = a.surface[:, None], H[:, None]
    assert m.rms_vs_analytical() == pytest.approx((0.0, 0.0), abs=1e-9)


def test_1d_analytical_reference_is_built_on_the_models_datum():
    """siim1d forwards hc_over_H, so the reference it overlays and measures
    against reports its bed on the run's datum (ED-183): zb = z - hc_over_H*H,
    with the solved surface and thickness unchanged (the solve is in (z, H))."""
    from siim.analytical import SteadyStateProfile
    m2 = _construct(siim2d, hc_over_H=1.0)
    assert m2.analytical is None or m2.analytical.hc_over_H == 1.0   # 2D forwards it too
    m = _construct(siim1d, mode='B', hc_over_H=1.0)
    a, ref = m.analytical, SteadyStateProfile(m._analytical_user_params() | {'hc_over_H': None})
    assert a.hc_over_H == m.hc_over_H == 1.0 and ref.hc_over_H == constants.HC_OVER_H
    np.testing.assert_allclose(a.surface, ref.surface, rtol=0, atol=1e-9)
    H = (ref.surface - ref.bed) / constants.HC_OVER_H
    np.testing.assert_allclose(a.bed, a.surface - 1.0 * H, rtol=0, atol=1e-9)
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match='hc_over_H'):
            SteadyStateProfile({'hc_over_H': bad})


def test_1d_keeps_its_own_copy_and_refuses_nan():
    """siim1d aliased its input arrays and let NaN past its flotation_ramp and
    H_diffusivity guards (the siim2d fixes, audit #35, #38)."""
    U = np.full((21, 2), 1e-3)                  # the (nx, nt) grid siim1d stored as given
    m = _construct(siim1d, U=U)
    U[:] = 5e-3
    assert np.all(m.U_matrix == 1e-3)
    for bad in (dict(flotation_ramp=np.nan), dict(H_diffusivity=np.nan)):
        with pytest.raises(ValueError, match="finite"):
            _construct(siim1d, **bad)


def test_2d_takes_a_0d_P_or_bl_as_a_scalar_and_refuses_beta_0():
    """A 0-d P or bl died on a TypeError inside the f-string of the ValueError
    meant for a bad series, and beta = 0 on a bare ZeroDivisionError (audit #41)."""
    m = _construct(siim2d, P=np.array(2.0), bl=np.array(5.0))
    assert (m.P, m.bl, m._P_series, m._bl_series) == (2.0, 5.0, None, None)
    assert m.zT == _construct(siim2d, P=2.0).zT
    with pytest.raises(ValueError, match='beta must be finite and > 0'):
        _construct(siim2d, beta=0.0)


@pytest.mark.parametrize('model_cls', [siim1d, siim2d], ids=['1d', '2d'])
def test_a_0d_array_is_its_scalar(model_cls):
    """Every scalar-or-array parameter reads a 0-d array as the scalar it is
    (P in 1D, zELA and U in 2D refused one; audit #41)."""
    scalars = dict(P=2.0, zELA=1200.0, U=2e-3, bl=0.0)
    m = _construct(model_cls, **{k: np.array(v) for k, v in scalars.items()})
    ref = _construct(model_cls, **scalars)
    assert (m.P, m.zELA, m.U, m.bl, m.zT) == (ref.P, ref.zELA, ref.U, ref.bl, ref.zT)


def test_1d_refuses_beta_0():
    """siim1d divides by beta in the same conversions (audit #41)."""
    with pytest.raises(ValueError, match='beta must be finite and > 0'):
        _construct(siim1d, beta=0.0)


@pytest.mark.parametrize('model_cls', [siim1d, siim2d], ids=['1d', '2d'])
def test_a_zT_given_alone_sets_zELA(model_cls):
    """zELA defaulted to ZELA, so a zT passed without zELA=None was dropped
    ({'zT': 3000} ran at zELA 1500, zT 1600). zELA now defaults to None, as in
    SteadyStateProfile, and the analytical reference follows (audit #65)."""
    m = _construct(model_cls, zT=3000.0)
    assert (m.zELA, m.zT) == (3000.0 - constants.P / constants.BETA, 3000.0)
    assert m.analytical.zELA == m.zELA
    assert _construct(model_cls).zELA == constants.ZELA      # neither given: the default
