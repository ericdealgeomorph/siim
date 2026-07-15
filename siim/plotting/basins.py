"""Channel + basin analysis: ``flux`` / ``hacks_law`` / ``largest_basins`` /
``largest_basins_history`` / ``sediment_history``. These need only model methods
(``extract_channel`` / ``strahler_order``) + ``_add_colorbar`` — no glacier
rasterizers. See docs/dev/plotting_plan.md.
"""

import warnings
from types import SimpleNamespace

import numpy as np

try:  # numpy >= 1.25
    from numpy.exceptions import RankWarning as _RankWarning
except ImportError:  # numpy < 1.25 (np.RankWarning removed in numpy 2)
    _RankWarning = np.RankWarning

from ._render import _add_colorbar


def _nanstat_quiet(fn, arr):
    """np.nanmean/np.nanstd along axis=1, silencing the all-NaN-row
    'Mean of empty slice' RuntimeWarning (the NaN result is correct: a basin
    had no ice at its head across every sampled step)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(arr, axis=1)

# which -> (ds var, scale, cmap, colorbar label, title)
_FLUX_SPEC = {
    'ice':   ('glacial_flow__ice_flux',   1e9, 'Blues',  r'$\log_{10}$ Ice flux (km$^3$/yr)',   'Ice flux'),
    'water': ('glacial_flow__water_flux', 1e9, 'YlOrRd', r'$\log_{10}$ Water flux (km$^3$/yr)', 'Water flux'),
    'area':  ('glacial_flow__area',       1e6, 'Greens', r'$\log_{10}$ Area (km$^2$)',          'Catchment area'),
}


class BasinsMixin:
    """Channel/basin analysis methods."""

    def flux(self, which='ice', i=-1, ax=None):
        """Single log-scale map of a flux field at step ``i``.

        ``which`` ∈ {``'ice'``, ``'water'``, ``'area'``}."""
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        if which not in _FLUX_SPEC:
            raise ValueError(f"which must be one of {sorted(_FLUX_SPEC)}")
        var, scale, cmap, label, title = _FLUX_SPEC[which]
        field = self.model.ds_out[var].isel(time=i)
        pos = field.values[field.values > 0]
        floor = np.percentile(pos, 5) if pos.size > 0 else 1.0
        if ax is None:
            _, ax = plt.subplots(figsize=(6.5, 5))
        im = np.log10((field / scale).where(field > floor)).plot(
            ax=ax, cmap=cmap, add_colorbar=False)
        _add_colorbar(im, ax, label=label)
        km = FuncFormatter(lambda v, _: f'{v / 1e3:.0f}')
        ax.xaxis.set_major_formatter(km)
        ax.yaxis.set_major_formatter(km)
        ax.set_aspect('equal')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_title(title)
        return ax

    def hacks_law(self, i=-1):
        """Hack's-law fit (channel length vs upstream area) for the main
        channel; the fit (k_h, d, xo, L) is from the last step, the scatter
        is area at step ``i``."""
        import matplotlib.pyplot as plt
        c = self._get_channel(-1, 0)
        print(f"Hack's law: k_h = {c.k_h:.2f}, d = {c.d:.2f}, "
              f"xo = {c.xo:.2f}, L = {c.L:.2f}")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(c.distance / 1e3, c.area[i] / 1e6, '.')
        x = np.logspace(np.log10(c.xo / 1e3), np.log10(c.L / 1e3), 100)
        ax.plot(x, c.k_h * (x * 1e3) ** c.d / 1e6, 'k--', label="Hack's law")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Distance along channel (km)')
        ax.set_ylabel('Area along channel (km$^2$)')
        ax.legend(loc='upper left')
        return ax

    def largest_basins(self, n_basins=4, i=-1, channel_threshold=1e5,
                       plot=True, z_max=None):
        """Extract the n largest basins, their main channels + summary stats.

        For each basin (ranked by node count, 0 = largest) follows the longest
        path via ``extract_channel`` and fits Hack's law. ``zo`` = max channel
        elevation at step ``i``; ``xt`` = distance to the first ice break;
        ``sigma`` = landscape Strahler bifurcation fit. Returns a
        SimpleNamespace (channels, k_h, d, L, zo, xt, sigma)."""
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        m = self.model

        channels = [m.extract_channel(i=i, basin_rank=r) for r in range(n_basins)]
        k_h = np.array([ch.k_h for ch in channels])
        d   = np.array([ch.d   for ch in channels])
        L   = np.array([ch.L   for ch in channels])
        zo  = np.array([ch.z[i].max() for ch in channels])

        xt_vals = []
        for ch in channels:
            H_t = ch.H[i]
            if H_t[0] <= 0:
                xt_vals.append(0.0)
            else:
                broken = np.where(H_t <= 0)[0]
                xt_vals.append(ch.L if broken.size == 0 else ch.distance[broken[0]])
        xt = np.array(xt_vals)

        strahler = m.strahler_order(i=i, channel_threshold=channel_threshold)
        sigma = float(np.mean(strahler.sigma))

        if plot:
            colors = ['r', 'orange', 'magenta', 'lime', 'cyan', 'yellow',
                      'white', 'pink']
            fig, ax = plt.subplots(ncols=2, figsize=(14, 5))
            m.ds_out.glacial_flow__basin_ids.isel(time=i).plot(
                ax=ax[0], cmap="tab20", add_colorbar=False)
            for r, ch in enumerate(channels):
                ax[0].plot(ch.x_coord, ch.y_coord, '-',
                           color=colors[r % len(colors)], lw=2, label=f'Basin {r}')
            ax[0].legend(loc='upper right', fontsize='small')
            ax[0].set_aspect('equal')
            ax[0].set_title(f"Largest {n_basins} basins "
                            f"(t = {m.output_times[i]/1e3:.1f} kyr)")
            km_fmt = FuncFormatter(lambda v, _: f'{v/1e3:.0f}')
            ax[0].xaxis.set_major_formatter(km_fmt)
            ax[0].yaxis.set_major_formatter(km_fmt)
            ax[0].set_xlabel('x (km)')
            ax[0].set_ylabel('y (km)')
            for r, ch in enumerate(channels):
                c = colors[r % len(colors)]
                ax[1].plot(ch.distance / 1e3, ch.z[i],  '-',  color=c, lw=1.5,
                           label=f'Basin {r}')
                ax[1].plot(ch.distance / 1e3, ch.zb[i], '--', color=c, lw=1.0)
            ax[1].set_xlabel('Distance from divide (km)')
            ax[1].set_ylabel('Elevation (m)')
            ax[1].set_title('Channel profiles (solid = ice surface, '
                            'dashed = bedrock)')
            if z_max is not None:
                ax[1].set_ylim(0, z_max)
            ax[1].legend(loc='upper right', fontsize='small')
            plt.tight_layout()

        return SimpleNamespace(channels=channels, k_h=k_h, d=d, L=L,
                               zo=zo, xt=xt, sigma=sigma)

    def largest_basins_history(self, n_basins=4, i_ref=-1, t_start=0, t_end=None,
                               n_samples=None, channel_threshold=1e5, plot=True):
        """Time-resolved ``largest_basins`` metrics over a step window.

        Basin identity is fixed at ``i_ref`` (each basin's largest-channel outlet
        flat-index); at every sampled step the channel is re-traced from that
        fixed outlet on the current flow graph. Headline benchmarks: ``xt/L`` and
        ``zo/zELA``; ``k_h``/``d``/``sigma`` are stationarity diagnostics.
        Returns a SimpleNamespace of (n_basins, n_steps) arrays + stats."""
        import matplotlib.pyplot as plt
        m = self.model
        nt_out = len(m.output_times)

        t_start = nt_out + t_start if t_start < 0 else t_start
        if t_end is None:
            t_end = nt_out - 1
        elif t_end < 0:
            t_end = nt_out + t_end
        if n_samples is None:
            steps = np.arange(t_start, t_end + 1)
        else:
            steps = np.unique(np.round(np.linspace(t_start, t_end, n_samples)).astype(int))

        outlets = np.empty(n_basins, dtype=int)
        xo_ref  = np.empty(n_basins, dtype=float)
        for r in range(n_basins):
            ch_ref = m.extract_channel(i=i_ref, basin_rank=r)
            outlets[r] = int(ch_ref.nodes[-1])
            xo_ref[r]  = float(ch_ref.xo)

        n_steps = len(steps)
        L_ts     = np.zeros((n_basins, n_steps))
        k_h_ts   = np.zeros((n_basins, n_steps))
        d_ts     = np.zeros((n_basins, n_steps))
        zo_ts    = np.zeros((n_basins, n_steps))
        xt_ts    = np.zeros((n_basins, n_steps))
        zterm_ts = np.zeros((n_basins, n_steps))
        Hmax_ts  = np.zeros((n_basins, n_steps))
        sigma_ts = np.zeros(n_steps)

        for k, t in enumerate(steps):
            sigma_ts[k] = float(np.mean(
                m.strahler_order(i=t, channel_threshold=channel_threshold).sigma))
            rec_t = m.receivers_out[t].flatten()
            rec_t = np.where(np.isnan(rec_t), np.arange(rec_t.size), rec_t).astype(int)
            area_t = m.area_out[t].flatten()
            lengths_t = m.lengths_out[t].flatten()
            z_flat = m.z_out[t].flatten()
            H_flat = m.H_out[t].flatten()
            nn = rec_t.size

            donor_list = [[] for _ in range(nn)]
            for j in range(nn):
                r_node = rec_t[j]
                if r_node != j:
                    donor_list[r_node].append(j)

            for r, outlet in enumerate(outlets):
                path = [int(outlet)]
                node = int(outlet)
                while True:
                    donors = donor_list[node]
                    if not donors:
                        break
                    best = max(donors, key=lambda dd: area_t[dd])
                    path.append(best)
                    node = best
                nodes = np.array(path[::-1])

                dist = np.zeros(len(nodes))
                for j in range(1, len(nodes)):
                    dist[j] = dist[j-1] + lengths_t[nodes[j-1]]
                L_ts[r, k] = dist[-1]

                valid = dist > 0
                log_x = np.log(dist[valid])
                log_a = np.log(area_t[nodes][valid])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", _RankWarning)
                    d_fit, log_kh = np.polyfit(log_x, log_a, 1)
                d_ts[r, k] = d_fit
                k_h_ts[r, k] = np.exp(log_kh)

                z_ch = z_flat[nodes]
                H_ch = H_flat[nodes]
                zo_ts[r, k] = z_ch.max()
                Hmax_ts[r, k] = H_ch.max()

                if H_ch[0] <= 0:
                    xt_ts[r, k] = 0.0
                    zterm_ts[r, k] = np.nan
                else:
                    broken = np.where(H_ch <= 0)[0]
                    if broken.size == 0:
                        xt_ts[r, k] = dist[-1]
                        zterm_ts[r, k] = z_ch[-1]
                    else:
                        xt_ts[r, k] = dist[broken[0]]
                        zterm_ts[r, k] = z_ch[broken[0]]

        t_vals = m.output_times[steps]
        zELA_at = m._zELA_output[steps]

        if plot:
            colors = ['r', 'orange', 'magenta', 'lime', 'cyan', 'yellow',
                      'white', 'pink']
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), sharex=True)
            ax = axes.flatten()
            t_kyr = t_vals / 1e3

            def per_basin(a, data, ylabel, legend=False):
                for r in range(n_basins):
                    a.plot(t_kyr, data[r], '-', color=colors[r % len(colors)],
                           lw=1.5, label=f'Basin {r}' if legend else None)
                a.set_ylabel(ylabel)

            per_basin(ax[0], xt_ts / L_ts, r'$x_t / L$', legend=True)
            ax[0].legend(loc='upper right', fontsize='small')
            per_basin(ax[1], zo_ts / zELA_at[None, :], r'$z_o / z_{ELA}$')
            ax[2].plot(t_kyr, sigma_ts, '-k', lw=1.5)
            ax[2].set_ylabel(r'$\sigma$ (landscape)')
            per_basin(ax[3], k_h_ts, r'$k_h$')
            per_basin(ax[4], d_ts, r'$d$')
            per_basin(ax[5], Hmax_ts, r'$H_{max}$ (m)')
            per_basin(ax[6], zo_ts, r'$z_o$ (m)')
            per_basin(ax[7], xt_ts / 1e3, r'$x_t$ (km)')
            per_basin(ax[8], zterm_ts, r'$z_{term}$ (m)')
            for a in ax[-3:]:
                a.set_xlabel('Time (kyr)')
            plt.tight_layout()

        xt_over_L = xt_ts / L_ts
        zo_over_zELA = zo_ts / zELA_at[None, :]
        stats = {
            'L':       {'mean': L_ts.mean(axis=1),     'std': L_ts.std(axis=1)},
            'k_h':     {'mean': k_h_ts.mean(axis=1),   'std': k_h_ts.std(axis=1)},
            'd':       {'mean': d_ts.mean(axis=1),     'std': d_ts.std(axis=1)},
            'zo':      {'mean': zo_ts.mean(axis=1),    'std': zo_ts.std(axis=1)},
            'xt':      {'mean': xt_ts.mean(axis=1),    'std': xt_ts.std(axis=1)},
            'z_term':  {'mean': _nanstat_quiet(np.nanmean, zterm_ts),
                        'std':  _nanstat_quiet(np.nanstd,  zterm_ts)},
            'H_max':   {'mean': Hmax_ts.mean(axis=1),  'std': Hmax_ts.std(axis=1)},
            'xt/L':    {'mean': xt_over_L.mean(axis=1), 'std': xt_over_L.std(axis=1)},
            'zo/zELA': {'mean': zo_over_zELA.mean(axis=1),
                        'std':  zo_over_zELA.std(axis=1)},
            'sigma':   {'mean': float(sigma_ts.mean()), 'std': float(sigma_ts.std())},
        }

        print(f"Stats over {n_steps} steps "
              f"(t = {t_vals[0]/1e3:.1f}–{t_vals[-1]/1e3:.1f} kyr):")
        header = "  ".join([f"Basin {r}".rjust(16) for r in range(n_basins)])
        print(f"  {'':10s}  {header}")
        for name in ('L', 'k_h', 'd', 'zo', 'xt', 'z_term', 'H_max', 'xt/L', 'zo/zELA'):
            row = "  ".join(
                f"{stats[name]['mean'][r]:8.3g} ± {stats[name]['std'][r]:7.2g}"
                for r in range(n_basins))
            print(f"  {name:10s}  {row}")
        print(f"  {'sigma':10s}  "
              f"{stats['sigma']['mean']:8.3g} ± {stats['sigma']['std']:7.2g} (landscape)")

        return SimpleNamespace(
            t=t_vals, steps=steps, outlets=outlets, xo_ref=xo_ref,
            L=L_ts, k_h=k_h_ts, d=d_ts, sigma=sigma_ts,
            zo=zo_ts, xt=xt_ts, z_term=zterm_ts, H_max=Hmax_ts, stats=stats)

    def sediment_history(self, n_basins=4, i_ref=-1, quantity='volume', ax=None):
        """Outlet sediment output for the n largest catchments through time.

        Outlets fixed at ``i_ref``; the sediment series at those nodes plotted
        across the run. ``quantity`` ∈ {'volume' (cumulative km^3), 'flux'
        (km^3/yr)}. Needs ``track_sediment=True``. Two panels (map + series)
        unless an Axes/pair is passed. Returns a SimpleNamespace."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.ticker import FuncFormatter
        m = self.model

        if quantity not in ('volume', 'flux'):
            raise ValueError("quantity must be 'volume' or 'flux'")
        attr = 'eroded_volume_out' if quantity == 'volume' else 'sediment_flux_out'
        if not hasattr(m, attr):
            raise RuntimeError("No sediment outputs found — re-run the model "
                               "with track_sediment=True.")

        data = getattr(m, attr)
        nt = data.shape[0]
        channels = [m.extract_channel(i=i_ref, basin_rank=r) for r in range(n_basins)]
        outlets = np.array([int(ch.nodes[-1]) for ch in channels], dtype=int)
        series = data.reshape(nt, -1)[:, outlets].T
        t = m.output_times

        if quantity == 'volume':
            series = series / 1e9
            ylabel = 'Cumulative eroded volume (km$^3$)'
        else:
            dt = float(m.t[1] - m.t[0])
            series = series / dt / 1e9
            ylabel = 'Sediment flux (km$^3$ / yr)'

        owns_fig = ax is None
        if owns_fig:
            fig, (ax_map, ax_ts) = plt.subplots(1, 2, figsize=(15, 5))
        else:
            try:
                ax_map, ax_ts = ax
            except (TypeError, ValueError):
                ax_map, ax_ts = None, ax

        if ax_map is not None:
            # m.z_out is the ice SURFACE for every mode (mode-aware _unpack
            # reconstructs zs = zb + hc*H for the citizen mode B whose
            # topography__elevation IS the bed) — plot that, not the raw
            # topography state, so the basin backdrop is the surface in all
            # modes. Reuse the topography DataArray's (x, y) coords, just swap
            # in the surface values, so the km axes + channel overlays still line up.
            elev = m.ds_out.topography__elevation.isel(time=i_ref).copy(
                data=m.z_out[i_ref])
            bids = m.ds_out.glacial_flow__basin_ids.isel(time=i_ref)
            elev.plot(ax=ax_map, cmap='gray', add_colorbar=False)
            for r, ch in enumerate(channels):
                bid = int(bids.values.flat[outlets[r]])
                bids.where(bids == bid).plot(
                    ax=ax_map, add_colorbar=False, alpha=0.35,
                    cmap=mcolors.ListedColormap([f'C{r}']))
                ax_map.plot(ch.x_coord, ch.y_coord, '-', color=f'C{r}', lw=1.2)
                ax_map.plot(ch.x_coord[-1], ch.y_coord[-1], 'o', color=f'C{r}',
                            mec='k', mew=0.8, ms=8, label=f'Basin {r}')
            ax_map.set_aspect('equal')
            km_fmt = FuncFormatter(lambda v, _: f'{v/1e3:.0f}')
            ax_map.xaxis.set_major_formatter(km_fmt)
            ax_map.yaxis.set_major_formatter(km_fmt)
            ax_map.set_xlabel('x (km)')
            ax_map.set_ylabel('y (km)')
            ax_map.set_title(f'Basins & outlets (t = {t[i_ref]/1e3:.1f} kyr)')
            ax_map.legend(loc='upper right', fontsize='small')

        for r in range(n_basins):
            ax_ts.plot(t / 1e3, series[r], '-', color=f'C{r}', lw=1.5,
                       label=f'Basin {r}')
        ax_ts.set_xlabel('Time (kyr)')
        ax_ts.set_ylabel(ylabel)
        ax_ts.set_title(f'Outlet sediment {quantity}')
        ax_ts.set_xlim(t[0] / 1e3, t[-1] / 1e3)
        ymax = 1.1 * float(series.max())
        ax_ts.set_ylim(0, ymax if ymax > 0 else 1.0)
        ax_ts.legend(loc='upper left', fontsize='small')

        if owns_fig:
            plt.tight_layout()

        return SimpleNamespace(t=t, outlets=outlets, series=series, quantity=quantity)
