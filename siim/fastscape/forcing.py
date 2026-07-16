"""Generic fastscape forcing processes.

Two reusable ``@xs.process`` forcing classes — not glacial, not siim-specific —
that drop into any fastscape model. They came out of the escarpment work
(:mod:`siim.escarpment`) but depend only on fastscape's grid/boundary/surface
processes:

  - :class:`WaveUplift` — a moving Gaussian uplift wave (a drop-in replacement
    for fastscape's ``BlockUplift``).
  - :class:`PlateauSurface` — an arctan-smoothed plateau initial topography
    (a drop-in replacement for an initial-elevation process).
"""
try:
    import xsimlab as xs
    from fastscape.processes import RasterGrid2D, SurfaceTopography, BorderBoundary
except ImportError as e:
    raise ImportError(
        "siim.fastscape is the OPTIONAL fastscape/xsimlab adapter — the "
        "standalone 2D model (siim.siim2d) needs none of it. To use the "
        "adapter: `pip install siim[fastscape]` provides fastscape + xsimlab "
        "from PyPI, but fastscapelib-fortran (the fortran backend fastscape's "
        "stock processes call at run time) has no PyPI wheel — use the conda "
        "env instead: `conda env create -f environment.yml`."
    ) from e

from .._core.step import uplift_mask, wave_uplift, plateau_surface


@xs.process
class WaveUplift:
    """Moving Gaussian uplift wave (replaces fastscape's BlockUplift).

    rate(x, t) = U_inf + U0 * exp(-(x - wave_center(t))^2 / wave_width^2),
    wave_center(t) = x[0] + x_escarpment + t * wave_velocity,
    U0 = wave_calibration * delta_h * wave_velocity / (wave_width * sqrt(pi)).
    """

    delta_h = xs.variable(description='target integrated uplift Δh as the wave passes (m)')
    wave_width = xs.variable(description='1/e half-width of wave in space (m)')
    wave_velocity = xs.variable(description='wave propagation velocity (m/yr)')
    x_escarpment = xs.variable(description='initial wave-center position relative to left edge (m)')
    wave_calibration = xs.variable(default=1.0, description='calibration factor on U0 (1.0 integrates to exactly delta_h as the wave passes)')
    U_inf = xs.variable(default=0.0, description='steady background uplift (m/yr)')

    x = xs.foreign(RasterGrid2D, 'x')
    y = xs.foreign(RasterGrid2D, 'y')
    shape = xs.foreign(RasterGrid2D, 'shape')
    status = xs.foreign(BorderBoundary, 'border_status')

    uplift = xs.variable(
        dims=[(), ('y', 'x')],
        intent='out',
        groups=['bedrock_forcing_upward', 'surface_forcing_upward'],
        description='Imposed vertical uplift this step (m)',
    )

    def initialize(self):
        # zero uplift on 'fixed_value' borders, mirroring fastscape's
        # BlockUplift — SPL never erodes self-receiving base-level nodes, so
        # uplifting them would let the boundary drift upward over the run.
        self._mask = uplift_mask(self.status, self.shape)

    @xs.runtime(args=['step_delta', 'step_start'])
    def run_step(self, dt, t):
        # Midpoint-sampled moving-Gaussian wave — body extracted framework-free
        # (siim._core.step.wave_uplift) so the shell and the in-house driver
        # share one implementation.
        self.uplift = wave_uplift(
            self.x, self.y, self.shape, self._mask, dt, t, self.delta_h,
            self.wave_width, self.wave_velocity, self.x_escarpment,
            self.wave_calibration, self.U_inf)


@xs.process
class PlateauSurface:
    """Arctan-smoothed plateau initial topography (replaces InitialTopography).

    Arctan ramp from ~0 on the low-x side to plateau_zo on the high-x side,
    centered at x_escarpment = (1 - plateau_frac) * Lx with transition width
    plateau_w. A small slope plateau_dz across the plateau seeds the divide.
    Uniform noise is added for D8 tie-breaking and zeroed on 'fixed_value'
    edges (matching InitialTopography, so the boundary z stays put across the
    run).
    """

    seed = xs.variable(default=None, description='Random seed')
    noise_amplitude = xs.variable(
        default=None,
        description='Noise amplitude (m); default 0.1 * max(elevation).'
    )

    plateau_zo = xs.variable(description='plateau elevation (m)')
    plateau_dz = xs.variable(default=1.0, description='slope across plateau (m); seeds divide')
    plateau_frac = xs.variable(default=0.8, description='fraction of x occupied by plateau')
    plateau_w = xs.variable(default=10e3, description='escarpment transition width (m)')

    x = xs.foreign(RasterGrid2D, 'x')
    y = xs.foreign(RasterGrid2D, 'y')
    shape = xs.foreign(RasterGrid2D, 'shape')
    elevation = xs.foreign(SurfaceTopography, 'elevation', intent='out')
    border_status = xs.foreign(BorderBoundary, 'border_status')

    def initialize(self):
        # Body extracted framework-free (siim._core.step.plateau_surface) so the
        # shell and the in-house driver share one implementation.
        self.elevation = plateau_surface(
            self.x, self.y, self.shape, self.border_status, self.seed,
            self.noise_amplitude, self.plateau_zo, self.plateau_dz,
            self.plateau_frac, self.plateau_w)
