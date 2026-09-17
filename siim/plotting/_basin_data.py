"""Basin measurements without figure creation, styling or printed summaries."""

import operator
import warnings
from types import SimpleNamespace

import numpy as np

try:
    from numpy.exceptions import RankWarning as _RankWarning
except ImportError:
    _RankWarning = np.RankWarning


def _basin_count(value):
    count = operator.index(value)
    if count < 1:
        raise ValueError('n_basins must be a positive integer')
    return count


def _history_steps(count, start, end, samples):
    if count == 0:
        raise ValueError('No simulation output; run the model first.')
    start = operator.index(start)
    end = count - 1 if end is None else operator.index(end)
    start = count + start if start < 0 else start
    end = count + end if end < 0 else end
    if not 0 <= start <= end < count:
        raise ValueError('t_start and t_end must select a nonempty output-step window')
    if samples is None:
        return np.arange(start, end + 1)
    samples = operator.index(samples)
    if samples < 1:
        raise ValueError('n_samples must be a positive integer')
    return np.unique(np.round(np.linspace(start, end, samples)).astype(int))


def _nanstat_quiet(fn, arr, axis=1):
    # An all-missing basin correctly yields NaN, without an empty-slice warning.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return fn(arr, axis=axis)


def hacks_law_data(channel, times, i, ref, basin_rank):
    """Scatter and fitted curve in displayed units; fit coefficients remain SI."""
    x = np.logspace(np.log10(channel.xo / 1e3), np.log10(channel.L / 1e3), 100)
    return SimpleNamespace(
        distance_km=channel.distance / 1e3, area_km2=channel.area[i] / 1e6,
        fit_distance_km=x, fit_area_km2=channel.k_h * (x * 1e3) ** channel.d / 1e6,
        k_h=channel.k_h, d=channel.d, xo=channel.xo, L=channel.L,
        time=times[i], ref_time=times[ref], basin_rank=basin_rank,
    )


def largest_basins_data(m, n_basins, i, channel_threshold):
    """Channel metrics at one output frame; no figure or text output."""
    n_basins = _basin_count(n_basins)
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
    sigma = float(np.mean(strahler.sigma)) if len(strahler.sigma) else np.nan

    return SimpleNamespace(channels=channels, k_h=k_h, d=d, L=L,
                           zo=zo, xt=xt, sigma=sigma)


def basin_history_data(m, n_basins, ref, t_start, t_end, n_samples, channel_threshold):
    """Trace fixed outlets through time and summarize their available samples."""
    n_basins = _basin_count(n_basins)
    steps = _history_steps(len(m.output_times), t_start, t_end, n_samples)
    outlets = np.empty(n_basins, dtype=int)
    xo_ref  = np.empty(n_basins, dtype=float)
    for r in range(n_basins):
        ch_ref = m.extract_channel(i=ref, basin_rank=r)
        outlets[r] = int(ch_ref.nodes[-1])
        xo_ref[r]  = float(ch_ref.xo)

    n_steps = len(steps)
    L_ts     = np.full((n_basins, n_steps), np.nan)
    k_h_ts   = np.full((n_basins, n_steps), np.nan)
    d_ts     = np.full((n_basins, n_steps), np.nan)
    zo_ts    = np.full((n_basins, n_steps), np.nan)
    xt_ts    = np.full((n_basins, n_steps), np.nan)
    zterm_ts = np.full((n_basins, n_steps), np.nan)
    Hmax_ts  = np.full((n_basins, n_steps), np.nan)
    sigma_ts = np.zeros(n_steps)

    for k, t in enumerate(steps):
        sigma_values = m.strahler_order(i=t, channel_threshold=channel_threshold).sigma
        sigma_ts[k] = float(np.mean(sigma_values)) if len(sigma_values) else np.nan
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
                if best in path:
                    break
                path.append(best)
                node = best
            nodes = np.array(path[::-1])
            if len(nodes) < 2:
                continue  # The reference outlet has no channel at this step.

            dist = np.zeros(len(nodes))
            for j in range(1, len(nodes)):
                dist[j] = dist[j-1] + lengths_t[nodes[j-1]]
            L_ts[r, k] = dist[-1]

            valid = (dist > 0) & (area_t[nodes] > 0) & np.isfinite(area_t[nodes])
            if valid.sum() >= 2 and np.ptp(dist[valid]) > 0:
                log_x = np.log(dist[valid])
                log_a = np.log(area_t[nodes][valid])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', _RankWarning)
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
    xt_over_L = np.divide(xt_ts, L_ts, out=np.full_like(xt_ts, np.nan), where=L_ts > 0)
    zo_over_zELA = np.divide(zo_ts, zELA_at[None, :], out=np.full_like(zo_ts, np.nan),
                             where=zELA_at[None, :] != 0)

    metrics = {'L': L_ts, 'k_h': k_h_ts, 'd': d_ts, 'zo': zo_ts, 'xt': xt_ts,
               'z_term': zterm_ts, 'H_max': Hmax_ts, 'xt/L': xt_over_L,
               'zo/zELA': zo_over_zELA}
    stats = {
        name: {'mean': _nanstat_quiet(np.nanmean, values),
               'std': _nanstat_quiet(np.nanstd, values),
               'n': np.isfinite(values).sum(axis=1)}
        for name, values in metrics.items()
    }
    stats['sigma'] = {
        'mean': float(_nanstat_quiet(np.nanmean, sigma_ts, axis=None)),
        'std': float(_nanstat_quiet(np.nanstd, sigma_ts, axis=None)),
        'n': int(np.isfinite(sigma_ts).sum()),
    }

    return SimpleNamespace(
        t=t_vals, steps=steps, outlets=outlets, xo_ref=xo_ref,
        L=L_ts, k_h=k_h_ts, d=d_ts, sigma=sigma_ts,
        zo=zo_ts, xt=xt_ts, z_term=zterm_ts, H_max=Hmax_ts, stats=stats,
        xt_over_L=xt_over_L, zo_over_zELA=zo_over_zELA)


def sediment_history_data(m, n_basins, ref, quantity):
    """Outlet volume/flux series in km³ or km³/yr, plus reference channels."""
    n_basins = _basin_count(n_basins)
    if quantity not in ('volume', 'flux'):
        raise ValueError("quantity must be 'volume' or 'flux'")
    attr = 'eroded_volume_out' if quantity == 'volume' else 'sediment_flux_out'
    if not hasattr(m, attr):
        raise RuntimeError("No per-node sediment outputs found — re-run the "
                           "model with track_sediment=True (or 'both', which "
                           "keeps the per-edge totals as well).")

    data = getattr(m, attr)
    nt = data.shape[0]
    channels = [m.extract_channel(i=ref, basin_rank=r) for r in range(n_basins)]
    outlets = np.array([int(ch.nodes[-1]) for ch in channels], dtype=int)
    series = data.reshape(nt, -1)[:, outlets].T
    t = m.output_times

    if quantity == 'volume':
        series = series / 1e9
    else:
        dt = float(m.t[1] - m.t[0])
        series = series / dt / 1e9

    return SimpleNamespace(t=t, outlets=outlets, series=series, quantity=quantity,
                           channels=channels, ref=ref)
