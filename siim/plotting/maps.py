"""``map`` / ``view_map`` / ``animate_map`` — a straight 2D raster of a
stored model field (bedrock, ice, ...).

Registry-driven: adding a field is one entry in ``FIELD_REGISTRY`` (the data is
always stored). No smoothing / hillshade / sub-grid ice — that is
``landscape``. See ``docs/guides/outputs_and_io.md`` for the public plotting
contract.

("field" here is *which stored variable* to draw — the ``field=`` argument and
``FIELD_REGISTRY`` — distinct from the ``map`` method that draws it.)
"""

import numpy as np

from ._render import _slider_view, _add_colorbar, _node_extent
from ._style import time_label
from ._animation import movie_path, save_animation

# field name -> (model attribute holding the (time, y, x) array, cmap, label)
# 'ice' is H_out: the WIDTH-MEAN thickness the physics consumes, NOT the local
# column depth landscape() paints (hc_over_H * H at the thalweg, deeper on
# carved flanks) — the labels say which.
FIELD_REGISTRY = {
    'bedrock': ('zb_out', 'gist_earth', 'Bedrock elevation (m)'),
    'ice':     ('H_out',  'Blues',      'Mean ice thickness H̄ (m)'),
    # future, free to add: 'erosion': ('erosion_rate_out', 'magma', 'Erosion rate (m/yr)')
}

# Ice-free ground under the 'ice' field: masked out and painted a neutral bare
# tone, so H = 0 does not read as a pale-blue skin of ice.
BARE_GROUND_COLOR = '#eeeae4'
_MASKED_FIELDS = ('ice',)


class MapMixin:
    """Straight raster of a stored 2D field: ``map`` / ``view_map`` /
    ``animate_map`` (frame / interactive / mp4 over the same render)."""

    # --- shared helpers ---------------------------------------------------
    def _field_data(self, field):
        if field not in FIELD_REGISTRY:
            raise ValueError(
                f"unknown field {field!r}; options: {sorted(FIELD_REGISTRY)}")
        attr, cmap, label = FIELD_REGISTRY[field]
        arr = getattr(self.model, attr, None)
        if arr is None:
            raise RuntimeError(
                f"field {field!r} ({attr}) is not stored on this run")
        return np.asarray(arr), cmap, label

    def _field_extent(self):
        m = self.model
        return _node_extent(m.Lx, m.Ly, m.grid_nx, m.grid_ny)

    def _field_figsize(self, fig_width):
        """Figure sized from the DOMAIN aspect: with ``aspect='equal'`` a
        hard-coded square figure boxes a non-square domain in whitespace."""
        m = self.model
        return (fig_width, fig_width * (m.Ly / m.Lx))

    @staticmethod
    def _mask_field(arr, field):
        """Mask the ice-free ground out of the 'ice' field (its colormap paints
        the masked cells ``BARE_GROUND_COLOR``), else pass the frame through."""
        return (np.ma.masked_less_equal(arr, 0.0)
                if field in _MASKED_FIELDS else arr)

    @staticmethod
    def _field_cmap(field, cmap):
        import matplotlib.pyplot as plt
        cmap_obj = plt.get_cmap(cmap)
        if field in _MASKED_FIELDS:
            cmap_obj = cmap_obj.with_extremes(bad=BARE_GROUND_COLOR)
        return cmap_obj

    @staticmethod
    def _auto_clim(arr, field_min, field_max):
        # global over ALL frames so the color scale is constant across a
        # view/animation (and a static frame is directly comparable)
        vmin = float(np.nanmin(arr)) if field_min is None else field_min
        vmax = float(np.nanmax(arr)) if field_max is None else field_max
        return vmin, vmax

    def _setup_map(self, field, field_min, field_max, cmap, ax, fig_width):
        import matplotlib.pyplot as plt
        arr, default_cmap, label = self._field_data(field)
        vmin, vmax = self._auto_clim(arr, field_min, field_max)
        if ax is None:
            _, ax = plt.subplots(figsize=self._field_figsize(fig_width), layout='constrained')
        im = ax.imshow(self._mask_field(arr[0], field), origin='lower',
                       extent=self._field_extent(), interpolation='nearest',
                       cmap=self._field_cmap(field, cmap or default_cmap),
                       vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_xlim(0, self.model.Lx / 1e3)
        ax.set_ylim(0, self.model.Ly / 1e3)
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.tick_params(labelsize=9)
        _add_colorbar(im, ax, label=label)
        return arr, im

    def _update_map(self, im, arr, field, idx):
        im.set_data(self._mask_field(arr[idx], field))
        title = im.axes.set_title(time_label(self.model.output_times[idx]),
                                  loc='right', fontsize=10)
        return im, title

    def map(self, field='bedrock', i=-1, field_min=None, field_max=None,
            cmap=None, ax=None, fig_width=6):
        """Stored node-valued raster at snapshot i; returns the Axes.

        Ice is width-mean thickness, with ice-free ground masked. Color limits
        cover all outputs unless field_min/field_max are supplied.
        """
        field = field.lower()
        arr, im = self._setup_map(field, field_min, field_max, cmap, ax, fig_width)
        self._update_map(im, arr, field, i)
        return im.axes

    def view_map(self, field='bedrock', field_min=None, field_max=None,
                 cmap=None, fig_width=6):
        """Notebook slider; update the same raster and colorbar in place."""
        field = field.lower()

        def make_draw(fig):
            arr, im = self._setup_map(field, field_min, field_max, cmap,
                                       fig.add_subplot(111), fig_width)

            def draw(idx):
                self._update_map(im, arr, field, idx)
            return draw

        return _slider_view(make_draw, [(len(self.model.output_times), 'Snapshot', -1)],
                            figsize=self._field_figsize(fig_width))

    def animate_map(self, field='bedrock', path=None, run_id=None,
                    field_min=None, field_max=None, cmap=None,
                    fps=20, interval=42, fig_width=6, frames=None):
        """MP4 of map(). Explicit fps wins; fps=None uses interval (ms/frame).

        ``frames`` encodes a subset of the saved frames: a slice, a range or a
        sequence of indices (negatives count from the end).
        """
        field = field.lower()
        arr, im = self._setup_map(field, field_min, field_max, cmap, None, fig_width)

        def update(idx):
            return self._update_map(im, arr, field, idx)

        return save_animation(im.figure, update, len(arr),
                              movie_path(path, run_id, f'map_{field}'),
                              fps=fps, interval=interval, frames=frames)
