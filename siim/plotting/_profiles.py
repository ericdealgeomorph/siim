"""Common profile presentation; model plotters supply data and reference grids."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from ._animation import movie_path, save_animation
from ._style import COLORS, style_axes, time_label


@dataclass(frozen=True)
class ProfileField:
    attribute: str
    label: str
    color: str
    scale: float = 1.0
    logarithmic: bool = False


PROFILE_FIELDS = {
    'elevation': ProfileField('z', 'Elevation (m)', COLORS['ice']),
    'ice_thickness': ProfileField('H', 'Mean ice thickness (m)', COLORS['ice']),
    'erosion_rate': ProfileField('erosion_rate', 'Erosion rate (m/yr)', COLORS['erosion'], logarithmic=True),
    'ice_flux': ProfileField('Qg', 'Ice flux (m$^3$/yr)', COLORS['ice_flux']),
    'water_flux': ProfileField('Qf', 'Water flux (m$^3$/yr)', COLORS['water_flux']),
    'shear_stress': ProfileField('tau', 'Basal shear stress (kPa)', COLORS['shear'], scale=1e-3),
    'sliding_velocity': ProfileField('ub', 'Sliding velocity (m/yr)', COLORS['velocity']),
}


def normalize_fields(fields, allowed):
    """Normalize names and per-field overrides consistently across models."""
    if fields is None:
        fields = 'elevation'
    items = fields.items() if isinstance(fields, dict) else (
        [(fields, None)] if isinstance(fields, str) else [(field, None) for field in fields]
    )
    result = {}
    for field, limits in items:
        if not isinstance(field, str) or field.lower() not in allowed:
            raise ValueError(f'unknown profile field {field!r}; allowed: {list(allowed)}')
        name = field.lower()
        if name in result:
            raise ValueError(f'duplicate profile field {name!r}')
        if limits is not None and (not isinstance(limits, (tuple, list)) or len(limits) != 2):
            raise ValueError(f'{name} limits must be a (min, max) pair or None')
        result[name] = limits
    if not result:
        raise ValueError('fields must contain at least one profile field')
    return result


def field_limits(field, *arrays):
    """All-frame finite extrema in display units, including supplied references."""
    spec = PROFILE_FIELDS[field]
    lo, hi = np.inf, -np.inf
    for array in arrays:
        if array is None:
            continue
        values = np.asarray(array, dtype=float)
        keep = np.isfinite(values)
        if spec.logarithmic:
            keep &= values > 0
        if keep.any():
            lo = min(lo, float(np.min(values, where=keep, initial=np.inf)) * spec.scale)
            hi = max(hi, float(np.max(values, where=keep, initial=-np.inf)) * spec.scale)
    if spec.logarithmic:
        return (lo / 1.2, hi * 1.2) if np.isfinite(lo) else (1e-12, 1.0)
    lo = min(0.0, lo)
    hi = max(0.0, hi)
    if field == 'ice_thickness':
        return lo, max(500.0, hi * 1.05)
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    return (lo - pad if lo < 0 else lo, hi + pad)


def data_limits(data, fields, analytical=None):
    result = {}
    for field in fields:
        attr = PROFILE_FIELDS[field].attribute
        arrays = [getattr(data, attr)]
        if field == 'elevation':
            arrays.extend([data.zb, getattr(data, 'ela', None), getattr(data, 'water_level', None)])
            if analytical is not None:
                arrays.extend([analytical.surface, analytical.bed, analytical.surface_alt,
                               getattr(analytical, 'ela', None)])
        elif field == 'erosion_rate':
            arrays.append(getattr(data, 'uplift', None))
        elif field == 'shear_stress':
            arrays.append(getattr(data, 'tau_c', None))
        if analytical is not None and field != 'elevation':
            arrays.append(getattr(analytical, attr, None))
        result[field] = field_limits(field, *arrays)
    return result


def resolve_limits(fields, auto, field_min=None, field_max=None):
    if len(fields) != 1 and (field_min is not None or field_max is not None):
        raise ValueError('field_min/field_max apply to a single field; use the '
                         '{field: (min, max)} dict form for per-field limits')
    result = {}
    for field, override in fields.items():
        lo, hi = (None, None) if override is None else override
        if len(fields) == 1:
            lo = field_min if field_min is not None else lo
            hi = field_max if field_max is not None else hi
        lo = auto[field][0] if lo is None else lo
        hi = auto[field][1] if hi is None else hi
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError(f'{field} limits must be finite with min < max')
        if PROFILE_FIELDS[field].logarithmic and lo <= 0:
            raise ValueError(f'{field} uses a log axis; its lower limit must be positive')
        result[field] = (lo, hi)
    return result


def uplift_at_outputs(model, nodes=None):
    """2D forcing at saved steps; supports scalar, spatial, time and time-space U.

    Keep uniform series compact; broadcast only after selecting requested nodes.
    """
    spec = model._make_uplift_field()
    nt = len(model.output_times)
    if isinstance(spec, tuple):
        values = np.asarray(spec[1])
        steps = np.round(np.linspace(0, model.nt - 1, nt)).astype(int)
        values = values[steps]
        if values.ndim == 1:
            values = values[:, None]
        else:
            values = values.reshape(nt, -1)
    else:
        values = np.asarray(spec).reshape(1, -1)
    count = model.grid_nx * model.grid_ny
    if nodes is not None:
        nodes = np.asarray(nodes)
        values = values if values.shape[1] == 1 else values[:, nodes]
        count = len(nodes)
    return np.broadcast_to(values, (nt, count))


def draw_panel(ax, field, data, idx, analytical=None):
    """Draw equivalent scientific quantities with identical semantic styles."""
    spec = PROFILE_FIELDS[field]
    x = data.x
    if field == 'elevation':
        ice = data.H[idx] > 0
        ax.fill_between(x, data.zb[idx], data.z[idx], where=ice,
                        color=COLORS['ice_fill'], alpha=0.3)
        ax.plot(x, data.zb[idx], color=COLORS['bed'], lw=1.3, label='Bedrock')
        ax.plot(x, np.where(ice, data.z[idx], np.nan), color=COLORS['ice'],
                lw=1.5, label='Ice surface')
        if analytical is not None:
            ax.plot(analytical.x, analytical.surface, '--', color=COLORS['ice'],
                    lw=1.1, label='Analytical ice surface')
            if not np.allclose(analytical.bed, analytical.surface, equal_nan=True):
                ax.plot(analytical.x, analytical.bed, '--', color=COLORS['bed'],
                        lw=1, label='Analytical bedrock')
            if analytical.surface_alt is not None:
                ax.plot(analytical.x, analytical.surface_alt, ':',
                        color=COLORS['alternative'], lw=1, label='Analytical fluvial alt')
            if analytical.ela != data.ela[idx]:
                ax.axhline(analytical.ela, color=COLORS['ela'], ls='--', lw=1,
                           label='Analytical ELA')
        ax.axhline(data.ela[idx], color=COLORS['ela'], lw=1, label='ELA')
    else:
        values = getattr(data, spec.attribute)[idx] * spec.scale
        label = {'shear_stress': 'Basal shear stress', 'ice_thickness': 'Mean ice thickness'}.get(
            field, field.replace('_', ' ').capitalize())
        if spec.logarithmic:
            values = np.ma.masked_less_equal(values, 0)
            ax.set_yscale('log')
        ax.plot(x, values, color=spec.color, lw=1.5, label=label)
        if field == 'ice_thickness':
            ax.fill_between(x, 0, values, color=COLORS['ice_fill'], alpha=0.25)
        if analytical is not None:
            ref = getattr(analytical, spec.attribute, None)
            if ref is not None:
                ax.plot(analytical.x, np.asarray(ref) * spec.scale, '--',
                        color=spec.color, lw=1.1, label='Analytical reference')
        if field == 'erosion_rate':
            uplift = np.ma.masked_less_equal(data.uplift[idx], 0)
            ax.plot(x, uplift, '--', color=COLORS['uplift'], lw=1, label='Uplift rate')
            if not np.any(np.asarray(getattr(data, spec.attribute)[idx]) > 0):
                ax.text(0.02, 0.05, 'No positive erosion in this snapshot',
                        transform=ax.transAxes, fontsize=8, color=COLORS['erosion'])
        if field == 'shear_stress' and data.tau_c is not None:
            ax.axhline(data.tau_c * spec.scale, color=COLORS['uplift'], ls='--',
                       lw=1, label=r'$\tau_c$')
    ax.set_ylabel(spec.label)
    ax.set_xlim(float(x[0]), float(x[-1]))


class ProfileMethods:
    """Common public methods. Subclasses supply _profile_source and reference data."""

    def _profile_setup(self, fields, field_min, field_max, basin_rank, ref,
                       analytical, bistable):
        normalized = normalize_fields(fields, self._PROFILE_FIELDS)
        data = self._profile_source(ref, basin_rank)
        reference = self._analytical_overlay(bistable) if analytical else None
        resolved = resolve_limits(normalized, data_limits(data, normalized, reference),
                                  field_min, field_max)
        return data, resolved, reference

    def _draw_profile_extra(self, ax, data, idx, field):
        pass

    def _draw_profile(self, axes, data, idx, resolved, reference, legend=True):
        for ax, (field, limits) in zip(axes, resolved.items()):
            draw_panel(ax, field, data, idx, reference)
            self._draw_profile_extra(ax, data, idx, field)
            ax.set_ylim(*limits)
            style_axes(ax, legend=legend)
        axes[-1].set_xlabel('Distance (km)')
        title = time_label(data.times[idx])
        if data.context:
            title += ' · ' + data.context
        axes[0].set_title(title, loc='right', fontsize=10)

    @staticmethod
    def _profile_axes(nf, ax=None, fig_width=8, aspect=0.38):
        import matplotlib.pyplot as plt
        if ax is not None:
            if nf != 1:
                raise ValueError('ax= is only valid for a single field')
            return ax.figure, np.array([ax])
        fig, axes = plt.subplots(nf, 1, sharex=True, squeeze=False,
                                 figsize=(fig_width, fig_width * aspect * nf),
                                 layout='constrained')
        return fig, axes[:, 0]

    def profile(self, fields='elevation', i=-1, field_min=None, field_max=None,
                basin_rank=0, ref=-1, analytical=True, bistable=True, ax=None, *,
                fig_width=8, aspect=0.38, legend=True):
        """Snapshot profile. Fields: name, list or {name: (min, max)|None}.

        Automatic limits cover finite data and references across all outputs.
        ``i`` selects the snapshot; ``ref`` selects the channel extraction frame
        and ``basin_rank`` its basin in 2D. In 1D those two arguments are accepted
        for call compatibility and have no effect. Figure width is in inches;
        aspect is panel height/width. Supplied axes retain existing artists.
        Coordinates follow the source order (descending in 1D, ascending along
        extracted 2D channels). Returns ``(fig, axes)``.
        """
        data, resolved, reference = self._profile_setup(
            fields, field_min, field_max, basin_rank, ref, analytical, bistable)
        fig, axes = self._profile_axes(len(resolved), ax, fig_width, aspect)
        self._draw_profile(axes, data, i, resolved, reference, legend)
        return fig, axes

    def view_profile(self, fields='elevation', field_min=None, field_max=None,
                     basin_rank=0, ref=-1, analytical=True, bistable=True,
                     fig_width=8, aspect=0.38, legend=False):
        """Notebook slider with the same fields, limits and styles as profile()."""
        from ._render import _profile_slider
        data, resolved, reference = self._profile_setup(
            fields, field_min, field_max, basin_rank, ref, analytical, bistable)

        def frame(axes, idx):
            for ax in axes:
                ax.clear()
            self._draw_profile(axes, data, idx, resolved, reference, legend=False)

        return _profile_slider(frame, len(data.times), len(resolved), data.times,
                               fig_width=fig_width, aspect=aspect, legend=legend)

    def animate_profile(self, fields='elevation', path=None, run_id=None,
                        field_min=None, field_max=None, basin_rank=0, ref=-1,
                        analytical=True, bistable=True, fps=20, interval=42, *,
                        fig_width=8, aspect=0.38, legend=True, frames=None):
        """MP4 with profile() styling. Explicit fps wins over interval (ms/frame).

        Pass fps=None to derive the encoded frame rate from interval.
        ``frames`` encodes a subset of the saved frames: a slice, a range or a
        sequence of indices (negatives count from the end).
        """
        data, resolved, reference = self._profile_setup(
            fields, field_min, field_max, basin_rank, ref, analytical, bistable)
        fig, axes = self._profile_axes(len(resolved), fig_width=fig_width, aspect=aspect)

        def update(idx):
            for ax in axes:
                ax.clear()
            self._draw_profile(axes, data, idx, resolved, reference, legend)
            return axes

        default = getattr(self, '_PROFILE_MOVIE', 'profile')
        # Preserve existing run_id names: <id>_1d and <id>_profile.
        target = movie_path(path, run_id, '1d' if run_id and default == 'profile_1d' else default)
        return save_animation(fig, update, len(data.times), target, fps=fps,
                              interval=interval, frames=frames)


def profile_data(**values):
    """A lightweight namespace: arrays are views, always ordered (time, space)."""
    return SimpleNamespace(**values)
