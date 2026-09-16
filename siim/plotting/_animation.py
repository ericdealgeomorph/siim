"""Serial movie ownership, timing and output naming for every plot family."""

from .._output import output_path


def movie_path(path, run_id, default):
    name = f'{run_id}_{default}' if run_id else (path or default)
    name = str(name)
    if name.lower().endswith('.mp4'):
        name = name[:-4]
    return output_path(name, 'movies') + '.mp4'


def save_animation(fig, update, nframes, path, *, fps=None, interval=42,
                   close=True, dpi=150):
    """Encode frames; explicit fps wins, otherwise interval is milliseconds/frame.

    Only figures created by the plotter are closed, including on encoder errors.
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        if nframes < 1:
            raise ValueError('No simulation output; run the model first.')
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError('interval must be positive and finite')
        rate = 1000.0 / interval if fps is None else fps
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError('fps must be positive and finite')
        movie = animation.FuncAnimation(
            fig, update, frames=range(nframes), interval=interval,
            blit=False, repeat=False,
        )
        movie.save(filename=path, writer='ffmpeg', fps=rate, dpi=dpi)
        return path
    finally:
        if close:
            plt.close(fig)
