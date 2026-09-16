"""Shared visual roles and axis treatment; never changes global rcParams."""

import numpy as np

COLORS = {
    'bed': '#555a60', 'ice': '#23658a', 'ice_fill': '#8ec4e3',
    'water': '#56b4e9', 'ela': '#a56d18', 'uplift': '#42464b',
    'erosion': '#b34f36', 'ice_flux': '#0072b2', 'water_flux': '#d55e00',
    'shear': '#9467a6', 'velocity': '#00876c', 'alternative': '#90969b',
}
# A basin keeps its color across maps, profiles and histories.
BASIN_COLORS = ('#0072b2', '#d55e00', '#009e73', '#cc79a7',
                '#56b4e9', '#a56d18', '#6f63a5', '#42464b')


def _time_unit(years):
    return (1e6, 'Myr') if abs(years) >= 1e6 else (
        (1e3, 'kyr') if abs(years) >= 1e3 else (1, 'yr'))


def time_label(years):
    scale, unit = _time_unit(years)
    return f't = {years / scale:.4g} {unit}'


def time_axis(years):
    """One readable time unit for the entire displayed series."""
    years = np.asarray(years)
    finite = years[np.isfinite(years)]
    scale, unit = _time_unit(np.max(np.abs(finite)) if finite.size else 0)
    return years / scale, f'Time ({unit})'


def summary_axes(ax, shape, fig_width, aspect, sharex=False):
    """Create a constrained summary layout or validate caller-owned axes."""
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    if ax is None:
        fig, axes = plt.subplots(*shape, figsize=(fig_width, fig_width * aspect),
                                 layout='constrained', sharex=sharex, squeeze=False)
        return fig, axes.ravel()
    axes = np.asarray(ax, dtype=object).ravel()
    if len(axes) != np.prod(shape) or not all(isinstance(a, Axes) for a in axes):
        raise ValueError(f'ax must contain {np.prod(shape)} Matplotlib axes')
    fig = axes[0].figure
    if any(a.figure is not fig for a in axes):
        raise ValueError('all supplied axes must belong to one figure')
    return fig, axes


def style_axes(ax, legend=True, scientific=False):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#a0a5aa')
    ax.tick_params(labelsize=9, color='#a0a5aa')
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.15, linewidth=0.6)
    if scientific:
        from matplotlib.ticker import ScalarFormatter
        if isinstance(ax.yaxis.get_major_formatter(), ScalarFormatter):
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4), useMathText=True)
    if legend and ax.get_legend_handles_labels()[1]:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                  borderaxespad=0, frameon=False, fontsize=8)
