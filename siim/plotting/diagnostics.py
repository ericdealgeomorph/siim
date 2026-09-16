"""Steady-state measurements and their public plotting wrapper."""

from types import SimpleNamespace

import numpy as np

from ._profiles import uplift_at_outputs
from ._style import COLORS, style_axes, time_label, time_axis, summary_axes


def _steady_state_data(m, tol_E, tol_z, window):
    """Compute the convergence series and first matching output, without plotting."""
    if not np.isfinite(tol_E) or not np.isfinite(tol_z) or tol_E < 0 or tol_z < 0:
        raise ValueError('steady-state tolerances must be finite and nonnegative')
    # Interior mask: exclude fixed_value edges from spatial means (pinned z
    # there biases mean(E)/mean(U) at coarse resolution).
    bs = m.boundary_status  # [left, right, bottom, top]
    ny, nx = m.grid_ny, m.grid_nx
    interior = np.ones((ny, nx), dtype=bool)
    if bs[0] == 'fixed_value': interior[:,  0] = False
    if bs[1] == 'fixed_value': interior[:, -1] = False
    if bs[2] == 'fixed_value': interior[ 0, :] = False
    if bs[3] == 'fixed_value': interior[-1, :] = False

    if not interior.any():
        raise ValueError('steady_state needs at least one interior node')

    t = m.output_times
    if len(t) == 0:
        raise ValueError('No simulation output; run the model first.')
    z_max_t  = m.z_out.max(axis=(1, 2))
    z_mean_t = m.z_out[:, interior].mean(axis=1)
    er_mean_t = m.erosion_rate_out[:, interior].mean(axis=1)

    if int(window) != window or window < 1:
        raise ValueError('window must be a positive integer')
    U_mean_t = uplift_at_outputs(m, np.flatnonzero(interior)).mean(axis=1)
    Uref = np.maximum(np.abs(U_mean_t), 1e-30)
    cond_E = np.abs(er_mean_t - U_mean_t) / Uref < tol_E
    # One output cannot establish an elevation trend.
    dz_dt = np.gradient(z_mean_t, t) if len(t) > 1 else np.full(1, np.nan)
    width = min(int(window), len(t))
    if width > 1:
        weights = np.ones(width)
        dz_dt = (np.convolve(dz_dt, weights, mode='same')
                 / np.convolve(np.ones(len(t)), weights, mode='same'))
    cond_z = np.abs(dz_dt) / Uref < tol_z
    cond = cond_E & cond_z
    ss_idx = int(np.argmax(cond)) if cond.any() else -1

    return SimpleNamespace(t=t, surface_max=z_max_t, surface_mean=z_mean_t,
                           erosion_mean=er_mean_t, uplift_mean=U_mean_t,
                           ela=m._zELA_output, elevation_trend=dz_dt,
                           converged=cond, index=ss_idx)


def _format_steady_state(data):
    if data.index < 0:
        return 'Steady state not reached within simulated time'
    return (f'Steady state reached at {time_label(data.t[data.index])} '
            f'(step {data.index}/{len(data.t) - 1})')


def _draw_steady_state(data, z_max, ax, fig_width):
    owns_fig = ax is None
    fig, axes = summary_axes(ax, (1, 2), fig_width, 0.35)
    time, xlabel = time_axis(data.t)
    axes[0].plot(time, data.surface_max, ':', color=COLORS['ice'], lw=1.2,
                 label='Maximum surface')
    axes[0].plot(time, data.surface_mean, color=COLORS['ice'], lw=1.5,
                 label='Mean surface')
    axes[0].plot(time, data.ela, color=COLORS['ela'], lw=1, label='ELA')
    axes[0].set_ylabel('Elevation (m)')
    if z_max is not None:
        axes[0].set_ylim(top=z_max)
    axes[1].plot(time, data.erosion_mean, color=COLORS['erosion'], lw=1.5,
                 label='Mean erosion rate')
    axes[1].plot(time, data.uplift_mean, '--', color=COLORS['uplift'], lw=1,
                 label='Mean uplift')
    axes[1].set_ylabel('Rate (m/yr)')
    for panel in axes:
        panel.set_xlabel(xlabel)
        if data.index >= 0:
            panel.axvline(time[data.index], color=COLORS['alternative'], ls=':', lw=1)
        style_axes(panel, scientific=panel is axes[1])
    if owns_fig:
        fig.suptitle(_format_steady_state(data), fontsize=11)


class DiagnosticsMixin:
    """Convergence diagnostics."""

    def steady_state(self, tol_E=0.05, tol_z=0.1, window=5, z_max=None, *,
                     plot=True, verbose=False, ax=None, fig_width=10):
        """First output where erosion and elevation trend satisfy the tolerances.

        Spatial-mean erosion must match mean uplift within ``tol_E`` (relative),
        and the rolling mean-elevation slope must be below ``tol_z * abs(U)``.
        Fixed-value boundaries are excluded from spatial means. Returns the
        first matching output-step index, or -1 (including single-output runs).
        ``plot=False`` creates no figure; printing requires ``verbose=True``.
        ``ax`` accepts a pair of axes for the elevation and erosion panels.
        """
        data = _steady_state_data(self.model, tol_E, tol_z, window)
        if plot:
            _draw_steady_state(data, z_max, ax, fig_width)
        if verbose:
            print(_format_steady_state(data))
        return data.index
