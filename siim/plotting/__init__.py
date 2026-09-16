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

    The 1D facade remains in ``siim1d.py``. Both inherit the same public
    profile methods and share field metadata, visual styles and serial movie
    handling. The 1D limit-cycle and phase-portrait helpers are private
    research tools, outside the public plotting API.
    """

    def __init__(self, model):
        self.model = model
