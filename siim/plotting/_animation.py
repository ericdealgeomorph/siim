"""Serial movie ownership, timing and output naming for every plot family."""

from .._output import output_path


def movie_path(path, run_id, default):
    name = f'{run_id}_{default}' if run_id else (path or default)
    name = str(name)
    if name.lower().endswith('.mp4'):
        name = name[:-4]
    return output_path(name, 'movies') + '.mp4'


def frame_indices(nframes, frames):
    """Which saved frames to encode: None is all of them; a slice, range or
    sequence of indices selects, with negatives counting from the end.

    Empty is an error, not an empty movie: the encoder happily writes an
    unplayable 262-byte file from no frames and reports success.
    """
    if frames is None:
        selected = list(range(nframes))
    elif isinstance(frames, slice):
        selected = list(range(nframes)[frames])
    else:
        selected = [range(nframes)[i] for i in frames]
    if not selected:
        raise ValueError('No simulation output to encode; run the model '
                         'first, or widen the frames selection.')
    return selected


def save_animation(fig, update, nframes, path, *, fps=None, interval=42,
                   frames=None, close=True, dpi=150):
    """Encode frames; explicit fps wins, otherwise interval is milliseconds/frame.

    ``frames`` selects a subset of the ``nframes`` saved frames (see
    :func:`frame_indices`); None encodes every one, in order.

    Only figures created by the plotter are closed, including on encoder errors.
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError('interval must be positive and finite')
        rate = 1000.0 / interval if fps is None else fps
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError('fps must be positive and finite')
        movie = animation.FuncAnimation(
            fig, update, frames=frame_indices(nframes, frames),
            interval=interval, blit=False, repeat=False,
        )
        movie.save(filename=path, writer='ffmpeg', fps=rate, dpi=dpi)
        return path
    finally:
        if close:
            plt.close(fig)
