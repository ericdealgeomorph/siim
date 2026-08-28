"""siim plotting package — ``siim_plotter`` assembled from per-area mixins.

Split out of the old ``siim2d_plotting.py`` for file-size hygiene and a trimmed
v1.0 keeper set; the public API (``m.plot.<method>``) is unchanged. See
``docs/guides/outputs_and_io.md`` for the public plotting contract.

``siim_plotter`` is assembled from the per-area mixins below; ``siim2d.py``
imports it from here, and the old ``siim2d_plotting.py`` is gone.
"""

from .maps import MapMixin
from .profiles import ProfileMixin
from .landscape import LandscapeMixin
from .basins import BasinsMixin
from .diagnostics import DiagnosticsMixin


class siim_plotter(MapMixin, ProfileMixin, LandscapeMixin,
                   BasinsMixin, DiagnosticsMixin):
    """Plotting facade bound to a run model instance (``m.plot``).

    This is the 2D facade (``siim2d.py`` binds it). ``siim1d.py`` carries a
    separate class of the same name — the 1D plotter (``profile`` /
    ``view_profile`` / ``limit_cycle`` …); the two share only the ``_render``
    helpers. Merging them into this package is a deferred post-1.0 cleanup.
    """

    def __init__(self, model):
        self.model = model
