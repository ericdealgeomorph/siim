"""``profile`` / ``view_profile`` / ``animate_profile`` — the extracted-channel
profile vs the analytical steady state, plus ``view_xsection`` (raster
cross-section) and the ``analytical`` overlay helper.

Unified call shape (identical in 1D and 2D):

    profile(fields='elevation', i=-1, field_min=None, field_max=None,
            basin_rank=0, ref=-1, analytical=True, bistable=True, ax=None)

``fields`` is a str, a list (stacked panels), or a ``{field: (min, max)|None}``
dict; ``None`` (or a ``None`` tuple side) means auto = the global, all-frame
stable limit (so the axis is constant across a view/animation). ``field_min`` /
``field_max`` are the single-field shorthand. 5 fields: elevation,
ice_thickness, erosion_rate, ice_flux, water_flux. See
``docs/guides/outputs_and_io.md`` for the public plotting and output contract.
"""


import numpy as np

from ._render import _slider_view, _profile_slider, output_path

_FIELDS = ('elevation', 'ice_thickness', 'erosion_rate', 'ice_flux', 'water_flux')


def _normalize_fields(fields):
    """str | list | dict -> ordered ``{field: (min, max)|None}``."""
    if fields is None:
        fields = 'elevation'
    if isinstance(fields, str):
        norm = {fields: None}
    elif isinstance(fields, dict):
        norm = dict(fields)
    else:
        norm = {f: None for f in fields}
    bad = [f for f in norm if f not in _FIELDS]
    if bad:
        raise ValueError(f"unknown profile field(s) {bad}; allowed: {list(_FIELDS)}")
    return norm


class ProfileMixin:
    """profile / view_profile / animate_profile + view_xsection + analytical."""

    # --- channel + limits + analytical (computed once; reused per frame) ---
    def _get_channel(self, ref, basin_rank):
        """Extract (and cache on the model) the basin-``basin_rank`` channel at
        step ``ref``. Also sets the analytical grid (a side effect of
        ``extract_channel``), so call before reading the analytical overlay."""
        m = self.model
        key = (ref, basin_rank)
        if (getattr(m, '_profile_channel_key', None) != key
                or getattr(m, '_profile_channel', None) is None):
            m._profile_channel = m.extract_channel(i=ref, basin_rank=basin_rank)
            m._profile_channel_key = key
        return m._profile_channel

    @staticmethod
    def _compute_field_ylims(ch, fields, smooth_n=10):
        """Stable global y-limits per field: rolling-mean-smoothed per-step
        min/max across all output steps. Returns ``{field: (ymin, ymax)}``."""
        fmap = {'elevation': ('zb', 'z'), 'ice_thickness': ('H', 'H'),
                'erosion_rate': ('erosion_rate', 'erosion_rate'),
                'ice_flux': ('Qg', 'Qg'), 'water_flux': ('Qf', 'Qf')}

        def _rolling_mean(arr, n):
            if len(arr) < n:
                return arr
            return np.convolve(arr, np.ones(n) / n, mode='same')

        ylims = {}
        for field in fields:
            if field == 'ice_thickness':
                hi = float(np.nanmax(ch.H))
                ylims[field] = (0, max(500.0, hi * 1.05))
                continue
            lo_attr, hi_attr = (('zb', 'z') if field == 'elevation'
                                else (fmap[field][0], fmap[field][0]))
            lo_s = np.array([np.nanmin(getattr(ch, lo_attr)[t])
                             for t in range(len(getattr(ch, lo_attr)))])
            hi_s = np.array([np.nanmax(getattr(ch, hi_attr)[t])
                             for t in range(len(getattr(ch, hi_attr)))])
            ymin = float(np.nanmin(_rolling_mean(lo_s, smooth_n)))
            ymax = float(np.nanmax(_rolling_mean(hi_s, smooth_n)))
            if field == 'elevation':
                ymin = 0.0
            elif field == 'erosion_rate':
                ymin = ymax / 100
            elif ymin > 0:
                ymin = 0.0
            pad = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
            ylims[field] = (ymin, ymax + pad)
        return ylims

    def _analytical_overlay(self, bistable):
        """Analytical SS reference (or None) as a SimpleNamespace; assumes the
        analytical grid was set by a prior ``_get_channel``."""
        from types import SimpleNamespace
        m = self.model
        surface, bed = m.analytical._analytical_profiles()
        if surface is None or bed is None:
            return None
        return SimpleNamespace(
            x=(m.L - m.x) / 1e3, surface=surface, bed=bed,
            surface_alt=(m.analytical.surface_alt
                         if bistable and m.analytical.glacier_flag in (4, 5)
                         else None),
            H=m.analytical._analytical_ice_thickness(),
            Qg=m.analytical._analytical_ice_flux(),
            Qf=m.analytical._analytical_water_flux())

    def _profile_setup(self, fields, field_min, field_max, basin_rank, ref,
                       analytical, bistable):
        fields_norm = _normalize_fields(fields)
        if len(fields_norm) != 1 and (field_min is not None or field_max is not None):
            raise ValueError(
                "field_min/field_max apply to a single field; use the "
                "{field: (min, max)} dict form for per-field limits")
        ch = self._get_channel(ref, basin_rank)
        auto = self._compute_field_ylims(ch, list(fields_norm))
        single = len(fields_norm) == 1
        resolved = {}
        for field, override in fields_norm.items():
            lo_a, hi_a = auto[field]
            omin = omax = None
            if override is not None:
                omin, omax = override
            if single:
                omin = field_min if field_min is not None else omin
                omax = field_max if field_max is not None else omax
            resolved[field] = (omin if omin is not None else lo_a,
                               omax if omax is not None else hi_a)
        an = self._analytical_overlay(bistable) if analytical else None
        return ch, resolved, an

    # --- the one render core (frame / view / animate all call this) -------
    def _draw_profile(self, axes, ch, idx, resolved, an):
        m = self.model
        x = ch.distance / 1e3
        for ax, (field, (lo, hi)) in zip(axes, resolved.items()):
            ax.cla()
            if field == 'elevation':
                ax.plot(x, ch.z[idx], 'navy', lw=1.5, label='2D ice surface')
                ax.plot(x, ch.zb[idx], 'dimgray', lw=1, label='2D bedrock')
                ax.fill_between(x, ch.zb[idx], ch.z[idx], color='blue', alpha=0.1)
                if an is not None:
                    ax.plot(an.x, an.surface, 'b--', lw=1, label='Analytical ice surface')
                    ax.plot(an.x, an.bed, 'k--', lw=0.5, label='Analytical bedrock')
                    ax.axhline(m.zELA, c='goldenrod', ls='--', lw=1, label='Analytical ELA')
                    if an.surface_alt is not None:
                        ax.plot(an.x, an.surface_alt, '--', color='gray', alpha=0.5,
                                lw=1, label='Analytical fluvial alt')
                ax.axhline(m._zELA_output[idx], c='goldenrod', ls='-', lw=1, label='ELA')
                ax.set_ylabel('Elevation (m)')
            elif field == 'ice_thickness':
                ax.plot(x, ch.H[idx], 'steelblue', lw=1.5, label='2D mean ice thickness')
                if an is not None and an.H is not None:
                    ax.plot(an.x, an.H, 'b--', lw=1,
                            label='Analytical mean ice thickness')
                # H is the WIDTH-MEAN thickness the physics consumes, not the
                # column depth landscape() paints (hc_over_H * H at the floor).
                ax.set_ylabel('Mean ice thickness (m)')
            elif field == 'erosion_rate':
                ax.plot(x, ch.erosion_rate[idx], 'firebrick', lw=1.5, label='Erosion rate')
                ax.axhline(m.U, color='black', linestyle='--', lw=1, label='Uplift rate')
                ax.set_yscale('log')
                ax.set_ylabel('Erosion rate (m/yr)')
            elif field == 'ice_flux':
                ax.plot(x, ch.Qg[idx], 'dodgerblue', lw=1.5, label='2D ice flux')
                if an is not None and an.Qg is not None:
                    ax.plot(an.x, an.Qg, 'b--', lw=1, label='Analytical ice flux')
                ax.set_ylabel('Ice flux (m$^3$/yr)')
            elif field == 'water_flux':
                ax.plot(x, ch.Qf[idx], 'orangered', lw=1.5, label='2D water flux')
                if an is not None and an.Qf is not None:
                    ax.plot(an.x, an.Qf, 'r--', lw=1, label='Analytical water flux')
                ax.set_ylabel('Water flux (m$^3$/yr)')
            ax.set_ylim(lo, hi)
            ax.legend(loc='upper right')
        axes[-1].set_xlabel('Distance (km)')

    def _profile_axes(self, nf, ax):
        import matplotlib.pyplot as plt
        if ax is not None:
            if nf != 1:
                raise ValueError("ax= is only valid for a single field")
            return ax.figure, np.array([ax])
        fig, axes = plt.subplots(nf, 1, figsize=(12, 3 * nf), sharex=True,
                                 squeeze=False)
        return fig, axes[:, 0]

    # --- public triad -----------------------------------------------------
    def profile(self, fields='elevation', i=-1, field_min=None, field_max=None,
                basin_rank=0, ref=-1, analytical=True, bistable=True, ax=None):
        """Extracted-channel profile at step ``i`` vs the analytical SS.

        ``fields`` str/list/dict (see module docstring); ``basin_rank`` picks
        the nth-largest basin (0 = largest); ``ref`` is the extraction/fit
        reference step. Returns ``(fig, axes)``."""
        ch, resolved, an = self._profile_setup(
            fields, field_min, field_max, basin_rank, ref, analytical, bistable)
        fig, axes = self._profile_axes(len(resolved), ax)
        self._draw_profile(axes, ch, i, resolved, an)
        if ax is None:
            fig.tight_layout()
        return fig, axes

    def view_profile(self, fields='elevation', field_min=None, field_max=None,
                     basin_rank=0, ref=-1, analytical=True, bistable=True,
                     fig_width=12, aspect=0.27, legend=False):
        """Interactive slider over output steps (see ``profile``).

        Uses ipympl for a smooth live canvas (managed locally, so your other
        plots stay on the default backend). ``fig_width`` (inches) sets the
        on-screen size and ``aspect`` the per-panel height/width ratio;
        ``legend`` shows a static legend (off by default).
        """
        ch, resolved, an = self._profile_setup(
            fields, field_min, field_max, basin_rank, ref, analytical, bistable)
        times = np.asarray(self.model.output_times)

        def frame(axes, idx):
            self._draw_profile(axes, ch, idx, resolved, an)

        return _profile_slider(frame, len(times), len(resolved), times,
                               fig_width=fig_width, aspect=aspect, legend=legend)

    def animate_profile(self, fields='elevation', path=None, run_id=None,
                        field_min=None, field_max=None, basin_rank=0, ref=-1,
                        analytical=True, bistable=True, fps=20, interval=42):
        """MP4 over output steps (see ``profile``). Returns the path."""
        import matplotlib.pyplot as plt
        import matplotlib.animation as anm
        ch, resolved, an = self._profile_setup(
            fields, field_min, field_max, basin_rank, ref, analytical, bistable)
        nf = len(resolved)
        times = np.asarray(self.model.output_times)
        fig, axes = plt.subplots(nf, 1, figsize=(12, 3 * nf), sharex=True,
                                 squeeze=False)
        axes = axes[:, 0]

        def update(idx):
            self._draw_profile(axes, ch, idx, resolved, an)
            fig.suptitle(f"t = {times[idx] / 1e3:.1f} kyr")

        update(0)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        ani = anm.FuncAnimation(fig, update, frames=range(len(times)),
                                interval=interval)
        name = f"{run_id}_profile" if run_id else (path or "profile")
        out = output_path(name, 'movies')
        ani.save(filename=out + ".mp4", writer="ffmpeg", fps=fps, dpi=150)
        plt.close(fig)
        return out + ".mp4"

    # --- standalone overlay + cross-section viewer ------------------------
    def analytical(self, ax=None, bistable=True):
        """Plot the analytical SS reference profile (distance axis). Assumes a
        prior ``profile``/channel extraction set the analytical grid."""
        import matplotlib.pyplot as plt
        m = self.model
        an = self._analytical_overlay(bistable)
        if an is None:
            print('No analytical solution available.')
            return
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(an.x, an.surface, 'b--', lw=1, label='Analytical ice surface')
        ax.plot(an.x, an.bed, 'k-', lw=0.5, label='Analytical bedrock')
        if an.surface_alt is not None:
            ax.plot(an.x, an.surface_alt, '--', color='gray', alpha=0.5, lw=1,
                    label='Analytical fluvial alt')
        ax.set_xlim(0, 1.05 * m.L / 1e3)
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Elevation (m)')
        ax.legend(loc='upper right')
        return ax

    def view_xsection(self, fields='elevation', figsize=None, y_km=None,
                      z_min=None, z_max=None, H_max=None, fig_width=12, aspect=0.27):
        """Interactive x-direction cross-section viewer: two sliders (time,
        y-row). The cross-section is the raster row at the chosen y — no
        flow graph / analytical. ``fields`` ⊂ {elevation, ice_thickness,
        erosion_rate}. Uses ipympl for a smooth live canvas (managed locally,
        so your other plots stay on the default backend)."""
        m = self.model
        if isinstance(fields, str):
            fields = [fields]
        allowed = {'elevation', 'ice_thickness', 'erosion_rate'}
        bad = [f for f in fields if f not in allowed]
        if bad:
            raise ValueError(f"unsupported field(s) {bad}; allowed: {sorted(allowed)}")

        nt, ny, nx = m.z_out.shape
        x_km = np.linspace(0, m.Lx, nx) / 1e3
        y_m = np.linspace(0, m.Ly, ny)
        iy0 = ny // 2 if y_km is None else int(
            np.clip(round(y_km * 1e3 / m.Ly * (ny - 1)), 0, ny - 1))

        ylims = {}
        for field in fields:
            if field == 'elevation':
                hi = float(np.nanmax(m.z_out))
                ylims[field] = (0.0, hi + (hi * 0.05 if hi > 0 else 1.0))
            elif field == 'ice_thickness':
                ylims[field] = (0.0, max(500.0, float(np.nanmax(m.H_out)) * 1.05))
            elif field == 'erosion_rate':
                hi = float(np.nanmax(m.erosion_rate_out))
                ylims[field] = (max(hi / 100.0, 1e-12), hi * 1.5)
        if z_min is not None and 'elevation' in ylims:
            ylims['elevation'] = (z_min, ylims['elevation'][1])
        if z_max is not None and 'elevation' in ylims:
            ylims['elevation'] = (ylims['elevation'][0], z_max)
        if H_max is not None and 'ice_thickness' in ylims:
            ylims['ice_thickness'] = (0.0, H_max)

        nf = len(fields)

        def make_draw(fig):
            axes = fig.subplots(nf, 1, sharex=True, squeeze=False)[:, 0]

            def draw(t_idx, iy):
                for ax, field in zip(axes, fields):
                    ax.cla()
                    if field == 'elevation':
                        ax.plot(x_km, m.z_out[t_idx, iy, :], 'navy', lw=1.5, label='Ice surface')
                        ax.plot(x_km, m.zb_out[t_idx, iy, :], 'dimgray', lw=1, label='Bedrock')
                        ax.fill_between(x_km, m.zb_out[t_idx, iy, :], m.z_out[t_idx, iy, :],
                                        color='blue', alpha=0.1)
                        ax.axhline(m._zELA_output[t_idx], c='goldenrod', ls='-', lw=1, label='ELA')
                        ax.set_ylabel('Elevation (m)')
                    elif field == 'ice_thickness':
                        ax.plot(x_km, m.H_out[t_idx, iy, :], 'steelblue', lw=1.5, label='Mean ice thickness')
                        ax.set_ylabel('Mean ice thickness (m)')
                    elif field == 'erosion_rate':
                        ax.plot(x_km, m.erosion_rate_out[t_idx, iy, :], 'firebrick', lw=1.5, label='Erosion rate')
                        ax.axhline(m.U, color='black', linestyle='--', lw=1, label='Uplift rate')
                        ax.set_yscale('log')
                        ax.set_ylabel('Erosion rate (m/yr)')
                    ax.set_ylim(*ylims[field])
                    ax.spines[['top', 'right']].set_visible(False)
                axes[-1].set_xlabel('x (km)')
                axes[0].set_title(f"{m.output_times[t_idx] / 1e3:.0f} kyr,  "
                                  f"y = {y_m[iy] / 1e3:.2f} km", loc='right')
            return draw

        figsize = figsize or (fig_width, fig_width * aspect * nf)
        return _slider_view(make_draw, [(nt, 'Snapshot', 0), (ny, 'y row', iy0)],
                            figsize=figsize)
