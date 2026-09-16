"""Public summary API: reference selection, data-only calls and composition."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from siim.siim2d import siim


@pytest.fixture(scope='module')
def model():
    result = siim(dict(U=1e-3, zELA=300, T=1e5, nt=11, nt_out=4,
                       nx=31, ny=31, Lx=3e4, Ly=3e4, seed=7,
                       initial_max_elevation=800, progress_bar=False,
                       track_sediment=True, boundary_status=['fixed_value'] * 4))
    result.run()
    return result


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


METHODS = ('hacks_law', 'largest_basins', 'largest_basins_history',
           'sediment_history', 'steady_state')


@pytest.mark.parametrize('method', METHODS)
def test_data_only_calls_create_no_figures_or_text(model, monkeypatch, capsys, method):
    fig, ax = plt.subplots()
    sentinel = ax.plot([0, 1], [1, 2])[0]
    def fail(*args, **kwargs):
        pytest.fail('A data-only call tried to create a figure')
    monkeypatch.setattr(plt, 'figure', fail)
    capsys.readouterr()
    result = getattr(model.plot, method)(plot=False)
    assert result is not None
    assert plt.get_fignums() == [fig.number]
    assert sentinel in ax.lines
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('method', ('largest_basins_history', 'sediment_history'))
def test_reference_keyword_and_legacy_alias_select_the_same_outlets(model, method):
    fn = getattr(model.plot, method)
    current = fn(n_basins=2, ref=0, plot=False)
    with pytest.warns(DeprecationWarning, match='use ref'):
        old = fn(n_basins=2, i_ref=0, plot=False)
    positional = fn(2, 0, plot=False)
    field = 'L' if method == 'largest_basins_history' else 'series'
    for result in (old, positional):
        np.testing.assert_array_equal(result.outlets, current.outlets)
        np.testing.assert_array_equal(getattr(result, field), getattr(current, field))
    with pytest.raises(ValueError, match='disagree'):
        fn(ref=1, i_ref=0, plot=False)


def test_hacks_law_separates_snapshot_and_reference_channel(model):
    data = model.plot.hacks_law(i=1, ref=0, basin_rank=1, plot=False)
    channel = model.extract_channel(i=0, basin_rank=1)
    np.testing.assert_array_equal(data.area_km2, channel.area[1] / 1e6)
    np.testing.assert_array_equal(data.distance_km, channel.distance / 1e3)
    assert (data.k_h, data.d, data.xo, data.L) == (channel.k_h, channel.d, channel.xo, channel.L)
    assert data.time == model.output_times[1]
    assert data.ref_time == model.output_times[0]
    ax = model.plot.hacks_law(i=1, ref=0, basin_rank=1)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), data.area_km2)
    np.testing.assert_array_equal(ax.lines[1].get_ydata(), data.fit_area_km2)


@pytest.mark.parametrize('method,count', [
    ('hacks_law', 1), ('largest_basins', 2), ('largest_basins_history', 9),
    ('sediment_history', 2), ('steady_state', 2),
])
def test_summary_plots_compose_without_replacing_other_content(model, capsys, method, count):
    fig, panels = plt.subplots(3, 4)
    axes = panels.ravel()
    sentinel = axes[-1].plot([0, 1], [1, 2])[0]
    title = fig.suptitle('Caller-owned figure')
    if method == 'steady_state':
        from matplotlib.ticker import FuncFormatter
        formatter = FuncFormatter(lambda value, _: f'{value:.2g}')
        axes[1].yaxis.set_major_formatter(formatter)
    supplied = axes[0] if count == 1 else axes[:count]
    capsys.readouterr()
    result = getattr(model.plot, method)(ax=supplied)
    assert result is not None
    assert plt.get_fignums() == [fig.number]
    assert len(fig.axes) == 12
    assert sentinel in axes[-1].lines
    assert title.get_text() == 'Caller-owned figure'
    assert capsys.readouterr().out == ''
    if method == 'steady_state':
        assert axes[1].yaxis.get_major_formatter() is formatter
    if method in ('largest_basins_history', 'sediment_history', 'steady_state'):
        assert axes[count - 1].get_xlabel() == 'Time (kyr)'


@pytest.mark.parametrize('method,text', [
    ('hacks_law', "Hack's law:"), ('largest_basins_history', 'Stats over'),
    ('steady_state', 'Steady state'),
])
def test_text_summaries_are_opt_in_and_independent_of_figures(model, capsys, method, text):
    capsys.readouterr()
    getattr(model.plot, method)(plot=False, verbose=True)
    assert text in capsys.readouterr().out
    assert not plt.get_fignums()


@pytest.mark.parametrize('options', [
    {'t_start': 3, 't_end': 1}, {'t_start': 100}, {'n_samples': 0}, {'n_basins': 0},
])
def test_history_rejects_empty_selections_before_plotting(model, options):
    with pytest.raises(ValueError):
        model.plot.largest_basins_history(plot=False, **options)
    assert not plt.get_fignums()


def test_history_window_is_inclusive_and_accepts_negative_indices(model):
    data = model.plot.largest_basins_history(t_start=-3, t_end=-1, plot=False)
    np.testing.assert_array_equal(data.steps, [1, 2, 3])
    np.testing.assert_array_equal(data.t, model.output_times[1:])


def test_sediment_series_can_use_a_single_caller_axis(model):
    fig, ax = plt.subplots()
    data = model.plot.sediment_history(n_basins=2, quantity='flux', ax=ax)
    assert fig.axes == [ax]
    for line, values in zip(ax.lines, data.series):
        np.testing.assert_array_equal(line.get_ydata(), values)
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), data.t / 1e3)


def test_summary_axes_reject_wrong_counts_and_mixed_figures(model):
    _, first = plt.subplots()
    _, second = plt.subplots()
    with pytest.raises(ValueError, match='2 Matplotlib axes'):
        model.plot.steady_state(ax=first)
    with pytest.raises(ValueError, match='one figure'):
        model.plot.largest_basins(ax=[first, second])
    assert not first.lines and not second.lines
