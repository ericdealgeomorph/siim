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

from ._render import _slider_view
from ._profiles import (ProfileMethods, PROFILE_FIELDS, profile_data, data_limits,
                        normalize_fields, resolve_limits, uplift_at_outputs, draw_panel)
from ._style import COLORS, style_axes, time_label

_FIELDS = tuple(name for name in PROFILE_FIELDS
                if name not in ('shear_stress', 'sliding_velocity'))


class ProfileMixin(ProfileMethods):
    """2D channel extraction and references for the common profile renderer."""

    _PROFILE_FIELDS = _FIELDS

    def _get_channel(self, ref, basin_rank):
        m = self.model
        key = (ref, basin_rank)
        if (getattr(m, '_profile_channel_key', None) != key
                or getattr(m, '_profile_channel', None) is None):
            m._profile_channel = m.extract_channel(i=ref, basin_rank=basin_rank)
            m._profile_channel_key = key
        ch = m._profile_channel
        geometry = (ch.L, ch.xo, ch.k_h, ch.d)
        if geometry != (m.L, m.xo, m.k_h, m.d):
            m._set_analytical_grid(*geometry)
        return ch

    def _profile_source(self, ref=-1, basin_rank=0):
        m = self.model
        ch = self._get_channel(ref, basin_rank)
        return profile_data(
            **{name: getattr(ch, name) for name in ('z', 'zb', 'H', 'Qg', 'Qf', 'erosion_rate')},
            x=ch.distance / 1e3, times=m.output_times, ela=m._zELA_output,
            uplift=uplift_at_outputs(m, ch.nodes), context=f'Basin {basin_rank}',
        )

    @staticmethod
    def _compute_field_ylims(ch, fields, smooth_n=None):
        """Stable finite extrema; smooth_n is accepted for old internal callers."""
        return data_limits(ch, fields)

    def _analytical_overlay(self, bistable):
        import warnings
        m = self.model
        surface, bed = m.analytical._analytical_profiles()
        if surface is None or bed is None:
            return None
        if getattr(m, '_analytical_bl_note', None):
            warnings.warn(m._analytical_bl_note, UserWarning, stacklevel=2)
        return profile_data(
            x=(m.analytical.L - m.analytical.x) / 1e3,
            surface=surface, bed=bed, ela=m.zELA,
            surface_alt=(m.analytical.surface_alt
                         if bistable and m.analytical.glacier_flag in (4, 5) else None),
            H=m.analytical._analytical_ice_thickness(),
            Qg=m.analytical._analytical_ice_flux(),
            Qf=m.analytical._analytical_water_flux(),
        )

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
            _, ax = plt.subplots(1, 1, figsize=(8, 3.04), layout='constrained')
        ax.plot(an.x, an.surface, '--', color=COLORS['ice'], lw=1.1, label='Analytical ice surface')
        ax.plot(an.x, an.bed, '--', color=COLORS['bed'], lw=1, label='Analytical bedrock')
        if an.surface_alt is not None:
            ax.plot(an.x, an.surface_alt, '--', color='gray', alpha=0.5, lw=1,
                    label='Analytical fluvial alt')
        ax.set_xlim(0, 1.05 * m.L / 1e3)
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Elevation (m)')
        style_axes(ax)
        return ax

    def view_xsection(self, fields='elevation', figsize=None, y_km=None,
                      z_min=None, z_max=None, H_max=None, fig_width=8, aspect=0.38):
        """Interactive x-direction cross-section viewer: two sliders (time,
        y-row). The cross-section is the raster row at the chosen y — no
        flow graph / analytical. ``fields`` ⊂ {elevation, ice_thickness,
        erosion_rate}. Uses ipympl for a smooth live canvas (managed locally,
        so your other plots stay on the default backend)."""
        m = self.model
        fields = normalize_fields(fields, ('elevation', 'ice_thickness', 'erosion_rate'))
        nt, ny, nx = m.z_out.shape
        x = np.linspace(0, m.Lx / 1e3, nx)
        iy0 = ny // 2 if y_km is None else int(
            np.clip(round(y_km * 1e3 / m.Ly * (ny - 1)), 0, ny - 1))
        uplift = uplift_at_outputs(m).reshape(nt, ny, nx)
        data = profile_data(z=m.z_out, zb=m.zb_out, H=m.H_out,
                            erosion_rate=m.erosion_rate_out, ela=m._zELA_output,
                            uplift=uplift)
        limits = resolve_limits(fields, data_limits(data, fields))
        if 'elevation' in limits:
            lo, hi = limits['elevation']
            limits['elevation'] = (lo if z_min is None else z_min, hi if z_max is None else z_max)
        if H_max is not None and 'ice_thickness' in limits:
            limits['ice_thickness'] = (0, H_max)

        def make_draw(fig):
            axes = fig.subplots(len(fields), 1, sharex=True, squeeze=False)[:, 0]

            def draw(idx, iy):
                row = profile_data(
                    **{name: getattr(data, name)[:, iy, :]
                       for name in ('z', 'zb', 'H', 'erosion_rate', 'uplift')},
                    x=x, ela=data.ela,
                )
                for ax, field in zip(axes, fields):
                    ax.clear()
                    draw_panel(ax, field, row, idx)
                    ax.set_ylim(*limits[field])
                    style_axes(ax, legend=False)
                axes[-1].set_xlabel('x (km)')
                axes[0].set_title(
                    f'{time_label(m.output_times[idx])} · y = {iy * m.Ly / (ny - 1) / 1e3:g} km',
                    loc='right', fontsize=10)
            return draw

        return _slider_view(make_draw, [(nt, 'Snapshot', 0), (ny, 'y row', iy0)],
                            figsize=figsize or (fig_width, fig_width * aspect * len(fields)))
