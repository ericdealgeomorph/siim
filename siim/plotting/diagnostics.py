"""``steady_state`` — detect and plot when the 2D landscape reaches steady
state. See docs/dev/plotting_plan.md."""

import numpy as np


class DiagnosticsMixin:
    """Convergence diagnostics."""

    def steady_state(self, tol_E=0.05, tol_z=0.1, window=5, z_max=None):
        """Detect + plot steady state.

        SS criterion: spatial-mean erosion matches mean uplift within ``tol_E``
        (relative) AND the rolling slope of mean elevation is below
        ``tol_z * mean(U)``. Left panel: max/mean ice-surface elevation + ELA;
        right panel: mean erosion vs mean uplift. Returns the output-step index
        at which SS is first satisfied, or -1 if never reached.
        """
        import matplotlib.pyplot as plt
        m = self.model

        # Interior mask: exclude fixed_value edges from spatial means (pinned z
        # there biases mean(E)/mean(U) at coarse resolution).
        bs = m.boundary_status  # [left, right, bottom, top]
        ny, nx = m.grid_ny, m.grid_nx
        interior = np.ones((ny, nx), dtype=bool)
        if bs[0] == 'fixed_value': interior[:,  0] = False
        if bs[1] == 'fixed_value': interior[:, -1] = False
        if bs[2] == 'fixed_value': interior[ 0, :] = False
        if bs[3] == 'fixed_value': interior[-1, :] = False

        t = m.output_times
        z_max_t  = m.z_out.max(axis=(1, 2))
        z_mean_t = m.z_out[:, interior].mean(axis=1)
        er_mean_t = m.erosion_rate_out[:, interior].mean(axis=1)

        U_field = m._make_uplift_field()
        if isinstance(U_field, tuple):
            U_full = U_field[1]
            out_idx = np.round(np.linspace(0, m.nt - 1, len(t))).astype(int)
            U_mean_t = U_full[out_idx][:, interior].mean(axis=1)
        else:
            U_mean_t = np.full(len(t), float(U_field[interior].mean()))

        Uref = np.maximum(U_mean_t, 1e-30)
        cond_E = np.abs(er_mean_t - U_mean_t) / Uref < tol_E
        dz_dt = np.gradient(z_mean_t, t)
        if window > 1:
            dz_dt = np.convolve(dz_dt, np.ones(window) / window, mode='same')
        cond_z = np.abs(dz_dt) / Uref < tol_z
        cond = cond_E & cond_z
        ss_idx = int(np.argmax(cond)) if cond.any() else -1

        fig, ax = plt.subplots(ncols=2, figsize=(14, 4.5))
        ax[0].plot(t, z_max_t, 'g--', label='ice surface max')
        ax[0].plot(t, z_mean_t, 'g-',  label='ice surface mean')
        zELA_t = m._zELA_output
        if np.allclose(zELA_t, zELA_t[0]):
            ax[0].axhline(zELA_t[0], c='k', ls='--', lw=0.5, label='ELA')
        else:
            ax[0].plot(t, zELA_t, 'k--', lw=0.5, label='ELA')
        ax[0].set_xlabel('Time (yr)')
        ax[0].set_ylabel('Elevation (m)')
        if z_max is not None:
            ax[0].set_ylim(0, z_max)
        ax[0].legend(loc='upper right')

        ax[1].plot(t, er_mean_t, 'firebrick', label='mean erosion rate')
        if np.allclose(U_mean_t, U_mean_t[0]):
            ax[1].axhline(U_mean_t[0], c='k', ls='--', lw=0.8, label='mean uplift')
        else:
            ax[1].plot(t, U_mean_t, 'k--', lw=0.8, label='mean uplift')
        ax[1].set_xlabel('Time (yr)')
        ax[1].set_ylabel('Rate (m/yr)')
        ax[1].legend(loc='upper right')

        if ss_idx >= 0:
            for a in ax:
                a.axvline(t[ss_idx], c='gray', ls=':', lw=1)
            msg = (f"Steady state reached at t = {t[ss_idx]:.3g} yr "
                   f"(step {ss_idx}/{len(t)-1})")
        else:
            msg = "Steady state not reached within simulated time"
        fig.suptitle(msg)
        print(msg)
        plt.tight_layout()
        return ss_idx
