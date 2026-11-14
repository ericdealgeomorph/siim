"""The shared colour roles, free of matplotlib.

``siim.plotting`` (through ``siim.plotting._style``, which re-exports them) and
the analytical package's ``plot`` methods both read ``COLORS``, so a bed, an ice
column or an ELA has one colour everywhere while ``import siim.analytical``
stays numpy/scipy-only.
"""

COLORS = {
    'bed': '#555a60', 'ice': '#23658a', 'ice_fill': '#8ec4e3',
    'water': '#56b4e9', 'ela': '#a56d18', 'uplift': '#42464b',
    'erosion': '#b34f36', 'ice_flux': '#0072b2', 'water_flux': '#d55e00',
    'shear': '#9467a6', 'velocity': '#00876c', 'alternative': '#90969b',
}
