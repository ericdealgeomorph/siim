"""siim plotting package — the 2D ``siim_plotter`` (``m.plot``), assembled
from per-area mixins: raw maps, channel profiles, the landscape renderer, basin
summaries and convergence diagnostics. See ``docs/guides/outputs_and_io.md``
for the public plotting contract.
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
