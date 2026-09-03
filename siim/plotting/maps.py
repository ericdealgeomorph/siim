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

from ._render import _slider_view, _add_colorbar, output_path

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
        return [0.0, m.Lx / 1e3, 0.0, m.Ly / 1e3]

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
            cmap_obj = cmap_obj.copy()
            cmap_obj.set_bad(BARE_GROUND_COLOR)
        return cmap_obj

    @staticmethod
    def _auto_clim(arr, field_min, field_max):
        # global over ALL frames so the color scale is constant across a
        # view/animation (and a static frame is directly comparable)
        vmin = float(np.nanmin(arr)) if field_min is None else field_min
        vmax = float(np.nanmax(arr)) if field_max is None else field_max
        return vmin, vmax

    # --- public triad -----------------------------------------------------
    def map(self, field='bedrock', i=-1, field_min=None, field_max=None,
            cmap=None, ax=None, fig_width=6):
        """Raster of a stored 2D ``field`` at output step ``i`` (default last).

        ``field`` is a key of ``FIELD_REGISTRY`` (``'bedrock'``, ``'ice'``).
        ``field_min`` / ``field_max`` override the global (all-frame) color
        limits. ``'ice'`` draws ``H_out``, the WIDTH-MEAN thickness — not the
        local column depth ``landscape`` paints — with the ice-free ground
        masked to a neutral bare tone. Returns the Axes.
        """
        import matplotlib.pyplot as plt
        arr, default_cmap, label = self._field_data(field)
        vmin, vmax = self._auto_clim(arr, field_min, field_max)
        if ax is None:
            _, ax = plt.subplots(figsize=self._field_figsize(fig_width))
        im = ax.imshow(self._mask_field(arr[i], field), origin='lower',
                       extent=self._field_extent(),
                       cmap=self._field_cmap(field, cmap or default_cmap),
                       vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_title(f"{np.asarray(self.model.output_times)[i]:.3g} yr",
                     loc='right')
        _add_colorbar(im, ax, label=label)
        return ax

    def view_map(self, field='bedrock', field_min=None, field_max=None,
                 cmap=None, fig_width=6):
        """Interactive slider over output steps (see ``map``).

        Uses ipympl for a smooth live canvas (managed locally, so your other
        plots stay on the default backend).
        """
        times = np.asarray(self.model.output_times)

        def make_draw(fig):
            def draw(idx):
                fig.clear()                       # drop the previous colorbar axes
                ax = fig.add_subplot(111)
                self.map(field=field, i=idx, field_min=field_min,
                         field_max=field_max, cmap=cmap, ax=ax)
                ax.set_title(f"{times[idx]:.3g} yr", loc='right')
            return draw

        return _slider_view(make_draw, [(len(times), 'Snapshot', -1)],
                            figsize=(fig_width, fig_width))

    def animate_map(self, field='bedrock', path=None, run_id=None,
                    field_min=None, field_max=None, cmap=None,
                    fps=20, interval=42, fig_width=6):
        """MP4 over output steps (see ``map``). Written under
        ``model_outputs/movies/``; returns the path."""
        import matplotlib.pyplot as plt
        import matplotlib.animation as anm
        arr, default_cmap, label = self._field_data(field)
        vmin, vmax = self._auto_clim(arr, field_min, field_max)
        times = np.asarray(self.model.output_times)

        fig, ax = plt.subplots(figsize=self._field_figsize(fig_width))
        im = ax.imshow(self._mask_field(arr[0], field), origin='lower',
                       extent=self._field_extent(),
                       cmap=self._field_cmap(field, cmap or default_cmap),
                       vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        _add_colorbar(im, ax, label=label)
        ttl = ax.set_title('')

        def update(idx):
            im.set_data(self._mask_field(arr[idx], field))
            ttl.set_text(f"t = {times[idx]:.3g} yr")
            return im, ttl

        ani = anm.FuncAnimation(fig, update, frames=range(len(arr)),
                                interval=interval, blit=False)
        name = f"{run_id}_map_{field}" if run_id else (path or f"map_{field}")
        out = output_path(name, 'movies')
        ani.save(filename=out + ".mp4", writer="ffmpeg", fps=fps, dpi=150)
        plt.close(fig)
        return out + ".mp4"
