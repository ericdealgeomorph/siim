"""Public basin plots, with computation delegated to ``_basin_data``."""

import warnings
from types import SimpleNamespace

import numpy as np

from ._basin_data import (largest_basins_data, basin_history_data,
                         sediment_history_data, hacks_law_data)
from ._render import _add_colorbar, _ice_ramp, _node_extent, _water_flux_ramp
from ._style import (BASIN_COLORS, COLORS, style_axes, time_label, time_axis,
                     summary_axes)


# which -> (ds var, scale, colormap or its factory, colorbar label, title). Ice
# takes the glacier ramp of every depth-coded ice map, water a ramp up to the
# profile panels' water-flux hue.
_FLUX_SPEC = {
    'ice':   ('glacial_flow__ice_flux',   1e9, _ice_ramp,        r'Ice flux (km$^3$/yr)',   'Ice flux'),
    'water': ('glacial_flow__water_flux', 1e9, _water_flux_ramp, r'Water flux (km$^3$/yr)', 'Water flux'),
    'area':  ('glacial_flow__area',       1e6, 'Greens',         r'Area (km$^2$)',          'Catchment area'),
}


def _km_raster(ax, model, values, **kwargs):
    """A node-valued (y, x) raster on the km grid map() and landscape() use,
    so a caller's km overlay lands where it should."""
    im = ax.imshow(values, origin='lower', interpolation='nearest',
                   extent=_node_extent(model.Lx, model.Ly, model.grid_nx, model.grid_ny),
                   **kwargs)
    # map()'s frame: the domain, not the half-cell border the node extent adds
    ax.set_xlim(0, model.Lx / 1e3)
    ax.set_ylim(0, model.Ly / 1e3)
    return im


def _draw_basin_map(ax, model, channels, idx):
    """One subdued map and basin-color assignment for all comparison families."""
    from matplotlib.colors import ListedColormap
    ids = np.asarray(model.ds_out.glacial_flow__basin_ids.isel(time=idx).values)
    _km_raster(ax, model, model.z_out[idx], cmap='Greys', alpha=0.35)
    for rank, channel in enumerate(channels):
        color = BASIN_COLORS[rank % len(BASIN_COLORS)]
        basin_id = ids.flat[channel.nodes[-1]]
        _km_raster(ax, model, np.where(ids == basin_id, 1.0, np.nan),
                   cmap=ListedColormap([color]), alpha=0.15)
        x_km, y_km = channel.x_coord / 1e3, channel.y_coord / 1e3
        ax.plot(x_km, y_km, color=color, lw=1.5, label=f'Basin {rank}')
        ax.plot(x_km[-1], y_km[-1], 'o', color=color, ms=4)
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_aspect('equal')
    ax.set_title(time_label(model.output_times[idx]), loc='right')
    style_axes(ax, legend=False)
    ax.grid(False)


class BasinsMixin:
    """Channel/basin analysis methods."""

    def flux(self, which='ice', i=-1, ax=None):
        """Single log-scale map of a flux field at step ``i``, on the km grid
        of ``map`` (6 in wide when it makes the figure).

        ``which`` ∈ {``'ice'``, ``'water'``, ``'area'``}."""
        import matplotlib.pyplot as plt
        if which not in _FLUX_SPEC:
            raise ValueError(f"which must be one of {sorted(_FLUX_SPEC)}")
        var, scale, cmap, label, title = _FLUX_SPEC[which]
        values = np.asarray(self.model.ds_out[var].isel(time=i).values, dtype=float)
        pos = values[values > 0]
        from matplotlib.colors import LogNorm
        low = float(pos.min()) / scale if pos.size else 1.0
        high = float(pos.max()) / scale if pos.size else 10.0
        norm = LogNorm(vmin=low, vmax=high if high > low else low * 10)
        if ax is None:
            _, ax = plt.subplots(figsize=self._field_figsize(6), layout='constrained')
        im = _km_raster(ax, self.model, np.where(values > 0, values / scale, np.nan),
                        cmap=cmap() if callable(cmap) else cmap, norm=norm)
        _add_colorbar(im, ax, label=label)
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_aspect('equal')
        ax.set_title(title, loc='left')
        ax.set_title(time_label(self.model.output_times[i]), loc='right')
        return ax

    def hacks_law(self, i=-1, *, ref=-1, basin_rank=0, plot=True,
                  verbose=False, ax=None, fig_width=6):
        """Main-channel area and its Hack's-law fit.

        ``i`` selects the area snapshot; ``ref`` selects the fitted reference
        channel and defaults to the final output. Returns an Axes, or a data
        namespace with ``plot=False``. Scatter/curve coordinates in that
        namespace are named ``distance_km``/``area_km2`` and
        ``fit_distance_km``/``fit_area_km2``; fit coefficients retain SI units.
        Only ``verbose=True`` prints the fit summary.
        """
        channel = self._get_channel(ref, basin_rank)
        data = hacks_law_data(channel, self.model.output_times, i, ref, basin_rank)
        if verbose:
            print(_format_hacks_law(data))
        return _draw_hacks_law(data, ax, fig_width) if plot else data

    def largest_basins(self, n_basins=4, i=-1, channel_threshold=1e5,
                       plot=True, z_max=None, *, ax=None, fig_width=12):
        """Extract the largest basins and optionally draw a map/profile pair.

        ``i`` selects both the basin ranking and the displayed snapshot.
        Returns a namespace (channels, k_h, d, L, zo, xt, sigma), also when
        ``plot=False``. Distances and elevations are meters. ``ax`` accepts a
        pair of axes; ``fig_width`` controls newly created figures in inches.
        """
        data = largest_basins_data(self.model, n_basins, i, channel_threshold)
        if plot:
            _draw_largest_basins(self.model, data, i, z_max, ax, fig_width)
        return data

    def largest_basins_history(self, n_basins=4, ref=-1, t_start=0, t_end=None,
                               n_samples=None, channel_threshold=1e5, plot=True,
                               *, i_ref=None, verbose=False, ax=None, fig_width=12):
        """Trace reference outlets through an inclusive output-step window.

        ``ref`` fixes basin identity, matching ``profile(ref=...)``. ``i_ref``
        is a deprecated keyword alias (removal in 1.0). ``t_start``/``t_end`` are output-step
        indices, not years; negative indices count from the final output.
        Returns a namespace of (basin, sample) arrays and ``stats``. Missing
        channels/fits are NaN; stats omit them and include valid counts in
        ``stats[metric]['n']``. ``plot=False`` creates no figure. Printing is
        opt-in with ``verbose=True``. ``ax`` accepts nine axes in row order.
        """
        ref = _resolve_reference(ref, i_ref)
        data = basin_history_data(self.model, n_basins, ref, t_start, t_end,
                                  n_samples, channel_threshold)
        if plot:
            _draw_basin_history(data, ax, fig_width)
        if verbose:
            print(_format_basin_history(data))
        return data

    def sediment_history(self, n_basins=4, ref=-1, quantity='volume', ax=None,
                         *, i_ref=None, plot=True, fig_width=12):
        """Outlet sediment output for basins selected at ``ref``.

        Needs ``track_sediment=True``. ``quantity`` is ``'volume'`` (cumulative
        km³) or ``'flux'`` (km³/yr). Returns a namespace with t (years), outlets,
        series and quantity, including with ``plot=False``.
        The default figure has a map and time series; ``ax`` accepts either a
        single time-series axis or a map/series pair. ``i_ref`` is deprecated
        (removal in 1.0).
        """
        ref = _resolve_reference(ref, i_ref)
        data = sediment_history_data(self.model, n_basins, ref, quantity)
        if plot:
            _draw_sediment_history(self.model, data, ax, fig_width)
        return SimpleNamespace(t=data.t, outlets=data.outlets, series=data.series,
                               quantity=data.quantity)


def _resolve_reference(ref, i_ref):
    if i_ref is None:
        return ref
    if ref != -1 and ref != i_ref:
        raise ValueError('ref and the deprecated i_ref alias disagree; pass only ref')
    warnings.warn('i_ref is deprecated and will be removed in 1.0; use ref instead.',
                  DeprecationWarning, stacklevel=3)
    return i_ref


def _format_hacks_law(data):
    return (f"Hack's law: k_h = {data.k_h:.2f}, d = {data.d:.2f}, "
            f"xo = {data.xo:.2f}, L = {data.L:.2f}")


def _draw_hacks_law(data, ax, fig_width):
    _, axes = summary_axes(ax, (1, 1), fig_width, 0.5)
    ax = axes[0]
    color = BASIN_COLORS[data.basin_rank % len(BASIN_COLORS)]
    ax.plot(data.distance_km, data.area_km2, '.', color=color, ms=4,
            alpha=0.7, label='Channel samples')
    ax.plot(data.fit_distance_km, data.fit_area_km2, '--', color=COLORS['bed'],
            lw=1.1, label=f"Hack's-law fit\n{time_label(data.ref_time)}")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Distance from divide (km)')
    ax.set_ylabel('Upstream area (km$^2$)')
    ax.set_title(f'Basin {data.basin_rank}', loc='left')
    ax.set_title(time_label(data.time), loc='right')
    style_axes(ax)
    return ax


def _draw_largest_basins(model, data, i, z_max, ax, fig_width):
    _, axes = summary_axes(ax, (1, 2), fig_width, 0.4)
    _draw_basin_map(axes[0], model, data.channels, i)
    for rank, channel in enumerate(data.channels):
        color = BASIN_COLORS[rank % len(BASIN_COLORS)]
        axes[1].plot(channel.distance / 1e3, channel.z[i], color=color, lw=1.5,
                     label=f'Basin {rank}')
        axes[1].plot(channel.distance / 1e3, channel.zb[i], color=color,
                     lw=1, alpha=0.45)
    axes[1].set_xlabel('Distance from divide (km)')
    axes[1].set_ylabel('Elevation (m)')
    axes[1].set_title('Channel profiles (faint = bedrock)')
    if z_max is not None:
        axes[1].set_ylim(top=z_max)
    style_axes(axes[1])


def _draw_basin_history(data, ax, fig_width):
    owns_fig = ax is None
    fig, axes = summary_axes(ax, (3, 3), fig_width, 0.74, sharex=True)
    time, xlabel = time_axis(data.t)
    panels = ((data.xt_over_L, r'$x_t / L$'),
              (data.zo_over_zELA, r'$z_o / z_{ELA}$'),
              (None, r'$\sigma$ (landscape)'),
              (data.k_h, r'$k_h$'), (data.d, r'$d$'),
              (data.H_max, r'$H_{max}$ (m)'), (data.zo, r'$z_o$ (m)'),
              (data.xt / 1e3, r'$x_t$ (km)'), (data.z_term, r'$z_{term}$ (m)'))
    for panel, (values, ylabel) in zip(axes, panels):
        if values is None:
            panel.plot(time, data.sigma, color=COLORS['bed'], lw=1.5)
        else:
            for rank, series in enumerate(values):
                panel.plot(time, series, color=BASIN_COLORS[rank % len(BASIN_COLORS)],
                           lw=1.5, label=f'Basin {rank}')
        panel.set_ylabel(ylabel)
        style_axes(panel, legend=False)
    for panel in axes[-3:]:
        panel.set_xlabel(xlabel)
    if owns_fig:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside upper center',
                   ncols=min(4, len(labels)), frameon=False)
    else:
        style_axes(axes[0])


def _format_basin_history(data):
    n_basins, n_steps = data.L.shape
    lines = [f'Stats over {n_steps} steps ({time_label(data.t[0])} to '
             f'{time_label(data.t[-1])}):']
    header = '  '.join(f'Basin {rank}'.rjust(16) for rank in range(n_basins))
    lines.append(f"  {'':10s}  {header}")
    for name, stats in data.stats.items():
        if name == 'sigma':
            row = f"{stats['mean']:8.3g} ± {stats['std']:7.2g} (landscape)"
        else:
            row = '  '.join(f"{mean:8.3g} ± {std:7.2g}"
                            for mean, std in zip(stats['mean'], stats['std']))
        lines.append(f'  {name:10s}  {row}')
    return '\n'.join(lines)


def _draw_sediment_history(model, data, ax, fig_width):
    single_axis = ax is not None and np.asarray(ax, dtype=object).size == 1
    _, axes = summary_axes(ax, (1, 1 if single_axis else 2), fig_width, 0.4)
    ax_ts = axes[-1]
    if len(axes) == 2:
        _draw_basin_map(axes[0], model, data.channels, data.ref)
    time, xlabel = time_axis(data.t)
    for rank, series in enumerate(data.series):
        ax_ts.plot(time, series, color=BASIN_COLORS[rank % len(BASIN_COLORS)],
                   lw=1.5, label=f'Basin {rank}')
    ylabel = ('Cumulative eroded volume (km$^3$)' if data.quantity == 'volume'
              else 'Sediment flux (km$^3$/yr)')
    ax_ts.set_xlabel(xlabel)
    ax_ts.set_ylabel(ylabel)
    ax_ts.set_title(f'Outlet sediment {data.quantity}')
    if len(time) > 1:
        ax_ts.set_xlim(time[0], time[-1])
    finite = data.series[np.isfinite(data.series)]
    ymax = 1.1 * float(finite.max()) if finite.size else 0
    ax_ts.set_ylim(0, ymax if ymax > 0 else 1)
    style_axes(ax_ts, scientific=data.quantity == 'flux')
