"""Public parameter-contract tests that do not require a model run."""

import warnings

import pytest

from siim.siim1d import siim as siim1d
from siim.siim2d import siim as siim2d


def _construct(model_cls, **params):
    common = dict(T=1.0, nt=2, nt_out=2, progress_bar=False, **params)
    if model_cls is siim1d:
        common['nx'] = 21
    else:
        common.update(nx=5, ny=5)
    return model_cls(common)


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
