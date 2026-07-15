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
import numpy as np
import xsimlab as xs
from fastscape.processes import RasterGrid2D, SurfaceTopography, BorderBoundary


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
        self._mask = np.ones(self.shape)
        _all = slice(None)
        slices = [(_all, 0), (_all, -1), (0, _all), (-1, _all)]
        for st, border in zip(self.status, slices):
            if st == 'fixed_value':
                self._mask[border] = 0.0

    @xs.runtime(args=['step_delta', 'step_start'])
    def run_step(self, dt, t):
        # Midpoint evaluation: rate is applied over [t, t+dt], so sampling the
        # wave at t + dt/2 integrates the Gaussian passage to second order and
        # leaves no systematic one-step position bias (the old t + dt sampling
        # shifted the wave forward by one step).
        time = t + 0.5 * dt
        wave_center = float(self.x[0]) + self.x_escarpment + time * self.wave_velocity
        X, _ = np.meshgrid(self.x, self.y)
        # U0 = delta_h * v / (w * sqrt(pi)) deposits exactly delta_h at every
        # point the full wave passes: integral of exp(-(x - v t')^2/w^2) dt'
        # over the passage is w*sqrt(pi)/v.
        U0 = (self.wave_calibration * self.delta_h * self.wave_velocity
              / (self.wave_width * np.sqrt(np.pi)))
        rate = self.U_inf + U0 * np.exp(-(X - wave_center) ** 2 / self.wave_width ** 2)
        self.uplift = rate * self._mask * dt


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
        X, Y = np.meshgrid(self.x, self.y)
        Lx = float(self.x[-1] - self.x[0])
        x_esc = (1 - self.plateau_frac) * Lx
        x_or = X - float(self.x[0])
        ramp = (1 + 2 / np.pi * np.arctan((x_or - x_esc) / self.plateau_w)) / 2
        slope = self.plateau_dz * np.clip(x_or - x_esc, 0.0, None) / max(Lx - x_esc, 1e-9)
        topo = ramp * (self.plateau_zo - slope)

        if self.seed is not None and not (isinstance(self.seed, float) and np.isnan(self.seed)):
            seed = int(self.seed)
        else:
            seed = None
        rs = np.random.RandomState(seed=seed)
        if self.noise_amplitude is None:
            noise_scale = 0.1 * np.max(topo)
        else:
            noise_scale = float(self.noise_amplitude)
        noise = noise_scale * rs.rand(*self.shape)
        bs = list(self.border_status)
        if bs[0] == "fixed_value": noise[:,  0] = 0.0
        if bs[1] == "fixed_value": noise[:, -1] = 0.0
        if bs[2] == "fixed_value": noise[0,  :] = 0.0
        if bs[3] == "fixed_value": noise[-1, :] = 0.0
        self.elevation = topo + noise
