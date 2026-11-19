"""Regression tests for the landscape() ice/lake display layer.

Opens with the ice_cmap-on-an-ice-free-frame guard (audit B8); the tests after
the phantom-ice one pin the 2026-09-02 display-layer sweep (unreachable lake
layer, silently ice-free 'field' smoothing, per-frame animation colour scales,
the section's own y-limits, raster/overlay registration, the section bed line,
and the bare-bed colorbar label).

With ``ice_cmap`` set, an ice-free frame has an all-False ice mask, and
``H_col[ice_mask].max()`` raised 'zero-size array to reduction operation
maximum which has no identity' -- which also aborted animate_landscape mid-run
on runs that start or cycle ice-free. An ice-free frame must render (no ice
drawn), not raise.
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')            # headless; no display needed
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from siim.siim2d import siim as siim2d
from siim.plotting._render import _priority_flood
from siim.plotting.landscape import _frozen_scale_kwargs
import pytest

from siim.plotting._style import COLORS
# gate: lake_color's default is now siim's one standing-water colour, COLORS['water']
LAKE_RGB = np.asarray(matplotlib.colors.to_rgb(COLORS['water']))   # lake_color default
ICE_RGB = np.asarray(matplotlib.colors.to_rgb('#e5e8ed'))    # ice_color default


def _map_rgb(ax):
    """The rendered map raster (ny_sub, nx_sub, 3)."""
    return np.asarray(ax.get_images()[0].get_array(), dtype=float)


def _section_axes(fig):
    """The cross-section panel of a landscape(cross_section=...) figure."""
    return [a for a in fig.axes if a.get_ylabel() == 'Elevation (m)'][0]


def _section_lines(ax_cs, n):
    """The section's profile/bedrock lines (the n-column ones, in draw order:
    bedrock then surface when ice is drawn, surface alone otherwise)."""
    return [np.asarray(ln.get_ydata()) for ln in ax_cs.lines
            if len(ln.get_ydata()) == n]


@pytest.fixture(autouse=True)
def _run_under_both_drivers(both_drivers):
    """S3 (Map 4 §1 PARAM): every test in this file runs under BOTH drivers --
    the conftest ``both_drivers`` fixture patches ``constants.DRIVER_DEFAULT``,
    so the existing assertions gate the in-house driver too."""



def test_landscape_ice_cmap_ice_free_frame():
    """A completed run with a huge ELA (never any ice) must render with
    ice_cmap set instead of crashing on the empty-mask reduction."""
    cfg = dict(
        U=1e-3, P=2, beta=1e-3, Ko=1e-6, n=1, ce=1e-4, nu=2, Ac=2e-24,
        lambda_p=5e2, lambda_c=1e2, alpha_g=8, sliding_law='eff-exp',
        zELA=1e5, T=1e5, Lx=3e4, Ly=3e4, nx=31, ny=31, nt=11, nt_out=4,
        D=1e-3, seed=7, boundary_status=['fixed_value'] * 4,
        initial_max_elevation=500, noise_amplitude=10, k=1,
        width_hack_k=1.0, width_hack_p=0.5, flow_routing='single',
        progress_bar=False,
    )
    m = siim2d(cfg)
    m.run()
    # Precondition: the run really is ice-free (so ice_mask is all-False).
    assert float(np.max([h.max() for h in m.H_out])) == 0.0

    # Must not raise on the empty ice mask.
    m.plot.landscape(field='bedrock+ice', ice_cmap='Blues', i=-1, colorbar=True)
    plt.close('all')


def test_cross_section_paints_no_phantom_ice():
    """The smooth-style cross-section must not paint ice in ice-free bed
    notches (the phantom-ice bug, 2026-07-16). The section previously filled
    everything between the UNsmoothed bed and the Gaussian-SMOOTHED display
    terrain as ice, so a relict 1-cell-wide carved slot rendered as a
    permanently ice-filled valley (~60% of notch depth painted, H = 0
    beneath). The section now reads the pre-smoothing composite: an ice-free
    frame must paint exactly zero ice pixels, notches included."""
    ny, nx = 31, 31
    zb0 = np.full((ny, nx), 500.0)
    zb0[5:26, 15] = 200.0            # 1-cell-wide, 300 m-deep relict slot
    cfg = dict(
        U=1e-3, zELA=1e5, T=1e3, nt=2, nt_out=2,
        Lx=3e4, Ly=3e4, nx=nx, ny=ny, seed=7,
        initial_topography=zb0, noise_amplitude=0,
        boundary_status=['fixed_value'] * 4, progress_bar=False,
    )
    m = siim2d(cfg)
    m.run()
    # Preconditions: truly ice-free, and the slot survived the (1-step) run —
    # otherwise the assertion below could pass vacuously.
    assert float(np.max([h.max() for h in m.H_out])) == 0.0
    assert float(m.zb_out[-1][:, 15].min()) < 350.0

    fig, _ = m.plot.landscape(field='bedrock+ice', i=-1, cross_section=15,
                              H_threshold=0)
    # The section's ice layer is the only zorder-2 image (bed image is 1,
    # lakes 2.5, the map is a single composite imshow).
    ice_imgs = [im for a in fig.axes for im in a.get_images()
                if im.get_zorder() == 2]
    assert ice_imgs, "cross-section ice layer not found"
    painted = sum(float(np.asarray(im.get_array())[..., 3].sum())
                  for im in ice_imgs)
    assert painted == 0.0, \
        f"phantom ice painted in an ice-free section (alpha sum {painted})"
    plt.close('all')


def _tiny_run(nt_out):
    m = siim2d(dict(U=1e-3, zELA=300, T=1e5, nt=11, nt_out=nt_out,
                    nx=31, ny=31, Lx=3e4, Ly=3e4, seed=7,
                    initial_max_elevation=800, progress_bar=False,
                    boundary_status=['fixed_value'] * 4))
    m.run()
    return m


def test_animate_parallel_renders_mp4(tmp_path, monkeypatch):
    """The parallel animate path (audit N35): workers reload the pickled run
    and render frames via the unchanged landscape(); ffmpeg assembles.
    n_workers forced to 2 to exercise the machinery regardless of the
    adaptive auto gate (which picks serial for a job this small)."""
    import os
    import matplotlib.animation as anm
    import pytest
    if not anm.FFMpegWriter.isAvailable():
        pytest.skip("ffmpeg not available")
    monkeypatch.chdir(tmp_path)   # model_outputs/ + movie land in tmp
    m = _tiny_run(4)
    # cross_section exercises the worker shim's meta attrs (_zELA_output etc.)
    swaps = []
    real_replace = os.replace

    def replace(src, dst):
        swaps.append((src, dst))
        real_replace(src, dst)

    monkeypatch.setattr(os, 'replace', replace)
    out = m.plot.animate_landscape(path='par_anim', n_workers=2,
                                   field='bedrock+ice', oversample=2,
                                   cross_section=15)
    assert out.endswith('.mp4') and os.path.getsize(out) > 10_000
    # the finished movie is swapped in from beside its target (one filesystem,
    # so the swap is atomic), and nothing is left there
    (src, dst), = [s for s in swaps if s[1] == out]
    assert os.path.dirname(src) == os.path.dirname(dst)
    assert os.listdir(os.path.dirname(out)) == [os.path.basename(out)]


def test_ice_cmap_auto_top_shows_no_max_arrow():
    """With ``ice_cmap`` and no ``H_max`` a still's colour scale tops out at
    its own thickest drawn column, so nothing saturates and the bar has no
    'max' arrow, though the model's unsmoothed column is thicker: by a hair (a
    last-bit rounding, in some frames) or by far (ice the display drops, such
    as a lone cell the smoothed mask does not draw). A pinned ``H_max`` below
    the ice still shows it."""
    # gate: same three expectations (no arrow on any auto-top frame, a 'max'
    # arrow under a pinned H_max, a non-vacuous frame); the vacuity guard moved
    # from a last-bit coin flip over the run's frames (ED-270 flipped it) to a
    # constructed frame with a 100 m margin, and the bar reads go through one
    # helper
    import siim.plotting.landscape as landscape_module
    m = siim2d(dict(U=1e-3, zELA=300, T=1e5, nt=11, nt_out=6, nx=41, ny=41,
                    Lx=4e4, Ly=4e4, seed=7, initial_max_elevation=800,
                    progress_bar=False, boundary_status=['fixed_value'] * 4))
    m.run()
    hc = getattr(m, 'hc_cell_out', None)

    def column_and_bar(i, **kw):
        """Frame i's model column maximum, drawn bar top and bar arrows."""
        fig, _ = m.plot.landscape(i=i, ice_cmap='viridis', **kw)
        try:
            bar, = [a for a in fig.axes if a.get_ylabel() == 'Ice column depth (m)']
            native = landscape_module._ice_column(
                m.H_out[i], m.hc_over_H if hc is None else hc[i]).max()
            return native, bar.get_ylim()[1], bar._colorbar.extend
        finally:
            plt.close(fig)

    arrows = [column_and_bar(i)[2] for i in range(1, len(m.output_times))]
    # a frame whose model column tops the drawn one by far more than rounding:
    # frame 1 plus one interior cell with no icy neighbour, made thicker than
    # every glacier; ice_extent='cells' smooths the cell mask, which does not
    # draw a lone cell
    k = 1
    H_k = np.array(m.H_out[k])
    near = np.logical_or.reduce([np.roll(H_k > 0.0, (a, b), (0, 1))
                                 for a in (-1, 0, 1) for b in (-1, 0, 1)])
    near[[0, -1], :] = True
    near[:, [0, -1]] = True
    H_lone = H_k.copy()
    H_lone[tuple(np.argwhere(~near)[0])] = 1.5 * H_k.max()
    m.H_out[k] = H_lone
    try:
        native, top, arrow = column_and_bar(k, ice_extent='cells')
    finally:
        m.H_out[k] = H_k
    assert native > top + 100.0, "vacuous: the model column does not top the drawn one"
    arrows.append(arrow)
    assert set(arrows) == {'neither'}
    assert column_and_bar(-1, H_max=100.0)[2] == 'max'


def test_animate_parallel_from_a_script_without_a_main_guard(tmp_path, both_drivers):
    """Spawned workers re-import the launching script; one with no
    ``if __name__ == '__main__':`` guard ran the model again in every worker
    and then broke the pool. The script's path is hidden from them."""
    import subprocess
    import matplotlib.animation as anm
    if both_drivers != 'inhouse':
        pytest.skip("the script runs in its own process, beyond the driver patch")
    if not anm.FFMpegWriter.isAvailable():
        pytest.skip("ffmpeg not available")
    script = tmp_path / 'unguarded.py'
    script.write_text(
        "import os\n"
        "from siim.tests.test_landscape_plot import _tiny_run\n"
        "print('RUN', flush=True)\n"
        "m = _tiny_run(4)\n"
        "out = m.plot.animate_landscape(path='unguarded', n_workers=2, style='raw')\n"
        "print('MOVIE', os.path.getsize(out), __file__ == %r, flush=True)\n"
        % str(script))
    done = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
                          capture_output=True, text=True, timeout=600,
                          env={**os.environ, 'PYTHONPATH': _REPO_ROOT})
    assert done.returncode == 0, done.stderr[-2000:]
    assert '(2 workers)' in done.stderr     # the parallel path, by its progress bar
    lines = done.stdout.split()
    assert lines.count('RUN') == 1          # the model ran once, not per worker
    movie = lines.index('MOVIE')
    assert int(lines[movie + 1]) > 0 and lines[movie + 2] == 'True'   # __file__ is back


def test_animate_parallel_keeps_the_script_for_a_keyword_it_defines(tmp_path, monkeypatch):
    """The launching script is hidden from the workers unless a keyword's
    pickle names ``__main__`` (an object of a class the script defines): those
    workers must import the script to rebuild it."""
    import concurrent.futures
    import contextlib
    import siim.plotting.landscape as landscape_module

    class Hidden(Exception):
        pass

    class Started(Exception):
        pass

    @contextlib.contextmanager
    def hidden():
        raise Hidden
        yield

    def pool(*args, **kwargs):
        raise Started

    monkeypatch.setattr(landscape_module, '_launching_script_hidden', hidden)
    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', pool)
    monkeypatch.chdir(tmp_path)
    m = _tiny_run(4)
    with pytest.raises(Hidden):
        m.plot.animate_landscape(path='a', n_workers=2, style='raw')
    class Shade:                        # as a script's own class pickles
        pass

    Shade.__module__, Shade.__qualname__ = '__main__', 'Shade'
    monkeypatch.setattr(sys.modules['__main__'], 'Shade', Shade, raising=False)
    with pytest.raises(Started):
        m.plot.animate_landscape(path='a', n_workers=2, style='raw',
                                 smoothed_path_color=Shade())


class _FedFrames:
    """An ffmpeg stdin stand-in that records each frame written to it."""

    def __init__(self, pipe):
        self.pipe, self.md5 = pipe, []

    def write(self, data):
        import hashlib
        self.md5.append(hashlib.md5(data).hexdigest())
        return self.pipe.write(data)

    def close(self):
        return self.pipe.close()


def _tap_ffmpeg(monkeypatch, seen):
    """Spy on the parallel path's ffmpeg: its command and the frames fed."""
    import subprocess
    real_popen = subprocess.Popen

    def spy_popen(cmd, **kwargs):
        seen['cmd'] = cmd
        seen['framerate'] = cmd[cmd.index('-framerate') + 1]
        proc = real_popen(cmd, **kwargs)
        tap = _FedFrames(proc.stdin)
        setattr(proc, 'stdin', tap)     # the parent writes the frames through it
        seen.setdefault('fed', []).append(tap.md5)
        return proc

    monkeypatch.setattr(subprocess, 'Popen', spy_popen)


@pytest.mark.pin
def test_animate_parallel_renders_only_the_selected_frames(tmp_path, monkeypatch):
    """``frames`` reaches the parallel path too: the workers render just the
    selected saved frames, ffmpeg receives them in the order given, and an
    explicit fps becomes the encoder's -framerate."""
    import os
    import matplotlib.animation as anm
    if not anm.FFMpegWriter.isAvailable():
        pytest.skip("ffmpeg not available")
    monkeypatch.chdir(tmp_path)
    m = _tiny_run(8)
    assert len(m.output_times) == 8, "test setup: need 8 frames to subsample 4"
    seen = {}
    _tap_ffmpeg(monkeypatch, seen)
    out = m.plot.animate_landscape(path='sel_anim', n_workers=2, style='raw',
                                   frames=slice(None, None, 2), fps=3)
    m.plot.animate_landscape(path='sel_back', n_workers=2, style='raw',
                             frames=[6, 4, 2, 0], fps=3)
    # all 8 frames, both ways (the same frames, so the same frozen scales):
    # more than the 2-worker render window of 4, so later frames are
    # submitted as earlier ones are fed
    m.plot.animate_landscape(path='all_fwd', n_workers=2, style='raw', fps=3)
    m.plot.animate_landscape(path='all_back', n_workers=2, style='raw', fps=3,
                             frames=range(7, -1, -1))
    # gate: the parallel path streams raw frames into ffmpeg in selection order
    # (ED-168) instead of numbering PNGs for a %05d sequence; the pin is the
    # same: exactly the selected frames, in the order given, at the given fps
    forward, backward, all_forward, all_backward = seen['fed']
    assert len(forward) == 4 and len(set(forward)) == 4
    assert backward == forward[::-1]
    assert len(set(all_forward)) == 8 and all_backward == all_forward[::-1]
    assert seen['framerate'] == '3'
    assert 'yuv420p' in seen['cmd']         # matplotlib's pixel format at h264
    assert '-b' not in seen['cmd']          # no bitrate unless rcParams set one
    assert out.endswith('.mp4') and os.path.getsize(out) > 0


def test_animate_parallel_failure_keeps_the_previous_movie(tmp_path, monkeypatch):
    """ffmpeg writes inside its temporary directory, so a failed encode leaves
    a movie already at ``path`` as it was, never half-overwritten."""
    import matplotlib.animation as anm
    if not anm.FFMpegWriter.isAvailable():
        pytest.skip("ffmpeg not available")
    monkeypatch.chdir(tmp_path)
    m = _tiny_run(4)
    previous = m.plot.animate_landscape(path='kept', n_workers=2, style='raw')
    before = open(previous, 'rb').read()
    import subprocess
    real_popen = subprocess.Popen
    monkeypatch.setattr(subprocess, 'Popen', lambda cmd, **kwargs: real_popen(
        cmd[:-1] + ['-c:v', 'no_such_encoder', cmd[-1]], **kwargs))
    with pytest.raises(RuntimeError, match='ffmpeg encoding failed'):
        m.plot.animate_landscape(path='kept', n_workers=2, style='raw')
    assert open(previous, 'rb').read() == before


def test_animate_parallel_failed_frame_skips_the_queued_frames(tmp_path, monkeypatch):
    """A frame that raises in its worker stops the movie: the parent raises a
    flag in the temporary directory, and a worker that finds it returns without
    rendering, so the frames already queued are skipped (it waited for the
    whole queue, 2 x n_workers frames, which cancel_futures cannot reach). The
    temporary directory goes and the running ffmpeg is killed."""
    import shutil
    import subprocess
    import matplotlib.animation as anm
    import siim.plotting.landscape as landscape_module
    if not anm.FFMpegWriter.isAvailable():
        pytest.skip("ffmpeg not available")
    monkeypatch.chdir(tmp_path)
    m = _tiny_run(8)
    removed, flagged, encoders = [], [], []
    real_rmtree, real_popen = shutil.rmtree, subprocess.Popen

    def rmtree(path, ignore_errors=False):
        removed.append(path)
        flagged.append('cancelled' in os.listdir(path))
        real_rmtree(path, ignore_errors=ignore_errors)

    def popen(cmd, **kwargs):
        encoders.append(real_popen(cmd, **kwargs))
        return encoders[-1]

    monkeypatch.setattr(shutil, 'rmtree', rmtree)
    monkeypatch.setattr(subprocess, 'Popen', popen)
    bad = np.asarray(m.output_times).astype(object)
    bad[1] = 'not a time'                    # its title raises in the worker
    monkeypatch.setattr(m, 'output_times', bad)
    with pytest.raises(TypeError):           # frame 0 has started ffmpeg by then
        m.plot.animate_landscape(path='kept', n_workers=2, style='raw')
    assert flagged == [True] and not os.path.exists(removed[0])
    assert encoders and all(proc.poll() is not None for proc in encoders)
    # the worker's side: with the flag up, a queued frame returns unrendered
    flag, raw = tmp_path / 'cancelled', tmp_path / 'frame.rgba'
    flag.touch()
    monkeypatch.setitem(landscape_module._ANIM_WORKER, 'cancelled', str(flag))
    assert landscape_module._animate_render_frame((0, 0, str(raw), 150)) is None
    assert not raw.exists()


def test_animate_parallel_workers_take_the_parents_rcparams(tmp_path, monkeypatch):
    """A spawned worker starts from matplotlib's defaults; the parallel path
    hands it the calling process's rcParams, so a style set in a script or a
    notebook reaches the frames as it does on the serial path."""
    import matplotlib.animation as anm
    if not anm.FFMpegWriter.isAvailable():
        pytest.skip("ffmpeg not available")
    monkeypatch.chdir(tmp_path)
    m = _tiny_run(4)
    seen = {}
    _tap_ffmpeg(monkeypatch, seen)
    m.plot.animate_landscape(path='plain', n_workers=2, style='raw')
    m.plot.animate_landscape(path='again', n_workers=2, style='raw')
    monkeypatch.setitem(matplotlib.rcParams, 'font.size', 20.0)
    m.plot.animate_landscape(path='styled', n_workers=2, style='raw')
    plain, again, styled = seen['fed']
    assert again == plain and set(styled).isdisjoint(plain)


def test_animate_parallel_names_the_codec(tmp_path, monkeypatch):
    """The parallel path's ffmpeg names its codec as matplotlib's writer
    does, from ``rcParams['animation.codec']``, instead of relying on ffmpeg's
    default mp4 encoder (byte-identical movies at the default h264), and
    passes the user's ``rcParams['animation.ffmpeg_args']`` as the serial
    path does."""
    import matplotlib.animation as anm
    if not anm.FFMpegWriter.isAvailable():
        pytest.skip("ffmpeg not available")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(matplotlib.rcParams, 'animation.codec', 'mpeg4')
    monkeypatch.setitem(matplotlib.rcParams, 'animation.ffmpeg_args',
                        ['-metadata', 'comment=siim'])
    monkeypatch.setitem(matplotlib.rcParams, 'animation.bitrate', 800)
    m = _tiny_run(4)
    seen = {}
    _tap_ffmpeg(monkeypatch, seen)
    out = m.plot.animate_landscape(path='codec', n_workers=2, style='raw',
                                   quality='draft')
    cmd = seen['cmd']
    assert cmd[cmd.index('-vcodec') + 1] == 'mpeg4'
    assert cmd[cmd.index('-metadata') + 1] == 'comment=siim'
    assert cmd[cmd.index('-b') + 1] == '800k'      # rcParams['animation.bitrate']
    assert 'yuv420p' not in cmd            # h264's pixel format, not forced on others
    assert '-preset' not in cmd            # nor is draft's x264 preset
    assert os.path.getsize(out) > 0
    # at h264 the user's own pixel format stands, as with matplotlib's writer
    monkeypatch.setitem(matplotlib.rcParams, 'animation.codec', 'h264')
    monkeypatch.setitem(matplotlib.rcParams, 'animation.ffmpeg_args',
                        ['-pix_fmt', 'yuv444p'])
    out = m.plot.animate_landscape(path='pix', n_workers=2, style='raw')
    cmd = seen['cmd']
    # gate: the libx264 case of 4ecf0d3 is gone with its rule (h264 alone is
    # x264 to matplotlib's writer, so the two render paths agree)
    assert 'yuv444p' in cmd and 'yuv420p' not in cmd
    assert os.path.getsize(out) > 0


def test_animate_parallel_bundles_only_the_selected_frames(tmp_path, monkeypatch):
    """The workers get the selected frames alone, each once, and read them by
    row: a repeated, reordered selection encodes the very frames a plain one
    of the same frames does, from a bundle of just those rows, and the RAM
    cap on the worker count counts that bundle, not the whole run."""
    import pickle
    import shutil
    import tempfile
    import matplotlib.animation as anm
    if not anm.FFMpegWriter.isAvailable():
        pytest.skip("ffmpeg not available")
    monkeypatch.chdir(tmp_path)
    m = _tiny_run(8)
    shipped = ('z_out', 'zb_out', 'H_out', 'area_out', 'receivers_out')
    row = sum(np.asarray(getattr(m, name))[0].nbytes for name in shipped)
    workers = []
    real_sysconf = os.sysconf
    with monkeypatch.context() as patch:
        # RAM for two workers of the 4-frame bundle, for one of the whole run
        patch.setattr(os, 'sysconf', lambda name: {
            'SC_PAGE_SIZE': 1, 'SC_PHYS_PAGES': 32 * row}.get(name) or real_sysconf(name))
        patch.setattr(type(m.plot), '_animate_parallel',
                      lambda self, path, interval, fig_width, n_workers, *a, **kw:
                      workers.append(n_workers))
        m.plot.animate_landscape(path='capped', n_workers=2, style='raw',
                                 frames=[7, 3, 7, 1, 5])
    assert workers == [2]
    seen = {}
    _tap_ffmpeg(monkeypatch, seen)
    m.plot.animate_landscape(path='plain', n_workers=2, style='raw',
                             frames=[1, 3, 5, 7])
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    # keep the worker bundle so it can be inspected
    monkeypatch.setattr(tempfile, 'mkdtemp', lambda **kw: str(bundle))
    monkeypatch.setattr(shutil, 'rmtree', lambda *a, **kw: None)
    m.plot.animate_landscape(path='mixed', n_workers=2, style='raw',
                             frames=[7, 3, 7, 1, 5])
    plain, mixed = seen['fed']              # both took the parallel path
    f1, f3, f5, f7 = plain
    assert mixed == [f7, f3, f7, f1, f5]
    kept = [1, 3, 5, 7]
    for name in shipped:
        # gate: equal_nan, as the xsimlab driver's receivers_out holds NaN
        assert np.array_equal(np.load(bundle / f'{name}.npy'),
                              np.asarray(getattr(m, name))[kept],
                              equal_nan=True), name
    with open(bundle / 'meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    for name in ('output_times', '_zELA_output'):
        assert np.array_equal(meta[name], np.asarray(getattr(m, name))[kept]), name


def test_animate_quality_presets(_basin_model, tmp_path, monkeypatch):
    """quality='draft' renders at most at oversample 2 (raw stays at 1) with
    x264's veryfast preset; 'high' (the default) keeps the still's oversample
    and matplotlib's encoder defaults; an explicit oversample wins; anything
    else is refused."""
    import siim.plotting.landscape as landscape_module
    m = _basin_model
    seen = []
    real_landscape = m.plot.landscape

    def landscape(*args, **kwargs):
        seen.append(kwargs.get('oversample'))
        return real_landscape(*args, **kwargs)

    def save(fig, update, nframes, path, *, extra_args=None, **_):
        seen.append(extra_args)
        plt.close(fig)
        return path

    monkeypatch.setattr(m.plot, 'landscape', landscape)
    monkeypatch.setattr(landscape_module, 'save_animation', save)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(matplotlib.rcParams, 'animation.codec', 'h264')
    veryfast = ['-preset', 'veryfast']
    for quality, kwargs, expected in (('high', {}, [None, None]),
                                      ('draft', {}, [2, veryfast]),
                                      ('draft', {'oversample': 3}, [3, veryfast]),
                                      ('draft', {'style': 'raw'}, [1, veryfast])):
        seen.clear()
        m.plot.animate_landscape(path='q', n_workers=1, quality=quality, **kwargs)
        assert seen == expected, quality
    with pytest.raises(ValueError, match='quality'):
        m.plot.animate_landscape(path='q', n_workers=1, quality='best')
    # the preset is x264's: with another codec draft keeps only its oversample
    monkeypatch.setitem(matplotlib.rcParams, 'animation.codec', 'mpeg4')
    seen.clear()
    m.plot.animate_landscape(path='q', n_workers=1, quality='draft')
    # gate: back to the one mpeg4 case; the libx264 loop went with its rule
    assert seen == [2, None]


def test_animate_draft_keeps_the_users_ffmpeg_args(_basin_model, tmp_path, monkeypatch):
    """matplotlib's writer reads ``rcParams['animation.ffmpeg_args']`` only
    when no ``extra_args`` are given, so the serial draft preset appends its
    x264 args to the user's instead of dropping them."""
    import subprocess
    import matplotlib.animation as anm
    if not anm.FFMpegWriter.isAvailable():
        pytest.skip("ffmpeg not available")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(matplotlib.rcParams, 'animation.ffmpeg_args',
                        ['-tune', 'animation'])
    monkeypatch.setitem(matplotlib.rcParams, 'animation.codec', 'h264')
    commands = []
    real_popen = subprocess.Popen

    def popen(cmd, **kwargs):
        commands.append(' '.join(cmd))
        return real_popen(cmd, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', popen)
    for quality, args in (('high', '-tune animation -y'),
                          ('draft', '-tune animation -preset veryfast -y')):
        commands.clear()
        _basin_model.plot.animate_landscape(path=quality, n_workers=1,
                                            quality=quality, style='raw')
        assert len(commands) == 1 and args in commands[0], quality


def test_landscape_contours_use_the_serial_algorithm(_basin_model, monkeypatch):
    """landscape's contours trace with contourpy's 'serial' algorithm, 1.6-1.7x
    faster than matplotlib's default (Eric, 2026-11-06; ED-168)."""
    import contourpy
    names = []
    real = contourpy.contour_generator

    def spy(*args, **kwargs):
        names.append(kwargs.get('name'))
        return real(*args, **kwargs)

    monkeypatch.setattr(contourpy, 'contour_generator', spy)
    fig, _ = _basin_model.plot.landscape(i=-1)
    plt.close(fig)
    assert names and set(names) == {'serial'}


def test_hypsometry_counts_match_the_edge_array_histogram():
    """The hypsometry counts exactly what np.histogram does with the edge
    array: values on the edges, outside them and NaN included, with even
    levels and with the partial end bins z_min / z_max add."""
    from siim.plotting.landscape import _hypsometry_counts
    rng = np.random.default_rng(3)
    values = np.concatenate([rng.uniform(-50.0, 1850.0, 10_000),
                             np.arange(0.0, 1801.0, 100.0), [1777.7, np.nan]])
    for edges in (np.arange(0.0, 1801.0, 100.0),
                  np.concatenate(([-37.3], np.arange(0.0, 1701.0, 100.0), [1777.7])),
                  np.array([0.0, 50.0, 400.0, 1800.0])):
        assert np.array_equal(_hypsometry_counts(values, edges),
                              np.histogram(values, bins=edges)[0])
    with pytest.raises(ValueError, match='must increase'):   # a search needs sorted edges
        _hypsometry_counts(values, np.linspace(1800.0, 0.0, 21))


def test_place_axes_draws_when_a_layout_engine_is_set():
    """A layout engine moves axes at draw time, so _place_axes draws then."""
    from siim.plotting.landscape import _place_axes
    fig, axes = plt.subplots(1, 2, layout='constrained')
    for a in axes:
        a.set_aspect('equal')
    _place_axes(fig)
    placed = [a.get_position().bounds for a in axes]
    fig.canvas.draw()
    assert [a.get_position().bounds for a in axes] == placed
    plt.close(fig)


def test_section_placement_needs_no_draw(_basin_model):
    """landscape settles its axes without rendering the frame (ED-168): the
    positions it leaves are those a full draw gives, with the section under
    the map."""
    fig, ax = _basin_model.plot.landscape(i=-1, cross_section=15)
    placed = [a.get_position().bounds for a in fig.axes]
    fig.canvas.draw()
    assert [a.get_position().bounds for a in fig.axes] == placed
    section = _section_axes(fig).get_position()
    assert section.x0 == pytest.approx(ax.get_position().x0)
    assert section.width == pytest.approx(ax.get_position().width)
    plt.close(fig)


# --- the 2026-09-02 display-layer sweep -------------------------------------

@pytest.fixture(scope='module')
def _basin_model():
    """Short ice-free run over a bed with ONE closed interior basin and one
    1-cell peak — the two features the lake layer and the section y-limit
    need."""
    ny = nx = 31
    zb0 = np.full((ny, nx), 500.0)
    zb0[13:18, 13:18] = 200.0        # closed basin, 300 m deep
    zb0[15, 8] = 1200.0              # 1-cell peak ON the y = 15 km section row
    cfg = dict(U=1e-3, zELA=1e5, T=1e3, nt=2, nt_out=2,
               Lx=3e4, Ly=3e4, nx=nx, ny=ny, seed=7,
               initial_topography=zb0, noise_amplitude=0,
               boundary_status=['fixed_value'] * 4, progress_bar=False)
    m = siim2d(cfg)
    m.run()
    return m


@pytest.fixture(scope='module')
def _iced_model():
    """Small glacial mode-C run (ice on every output frame). The bed is
    deliberately rough (noise_amplitude=60): the section's bed line is drawn
    against the ice fill, and a bicubic upsample only overshoots it where the
    bed has sharp relief."""
    cfg = dict(
        U=1e-3, P=1, beta=1e-2, Ko=1e-6, n=1, ce=1e-5, nu=2,
        sliding_law='power', lambda_p=300.0, alpha_g=8,
        zELA=400.0, T=3e5, Lx=3e4, Ly=3e4, nx=31, ny=31, nt=11, nt_out=4,
        D=1e-3, seed=7, boundary_status=['fixed_value'] * 4,
        initial_max_elevation=800, noise_amplitude=60, mode='C',
        flow_routing='single', progress_bar=False,
    )
    m = siim2d(cfg)
    m.run()
    return m


def test_lakes_field_renders_and_floods_the_true_surface(_basin_model):
    """field='bedrock+lakes' must render at all (the boundary_status list was
    bound only inside the footprint-ice branch, so every lake render raised
    UnboundLocalError: 'bs'), and the flood must run on the TRUE composite —
    not on the Gaussian-smoothed display terrain the section cannot meet (the
    phantom-ice split, 2026-07-16)."""
    from scipy.ndimage import gaussian_filter
    m = _basin_model
    zb = np.asarray(m.zb_out[-1])
    assert zb[15, 15] < 350.0, "test setup: the basin did not survive the run"
    # oversample=1 -> the subgrid IS the native grid, so the rendered composite
    # is this bed and its lake mask can be recomputed here exactly.
    true_lake = (_priority_flood(zb) - zb) > 0.5
    zs = gaussian_filter(zb, sigma=1.0)                  # sigma_cells = oversample
    smooth_lake = (_priority_flood(zs) - zs) > 0.5
    assert true_lake.any(), "test setup: no lake on the true bed"
    assert smooth_lake.sum() != true_lake.sum(), \
        "test setup: smoothing does not change the flood here"

    for extra in (dict(), dict(cross_section=15.0)):
        fig, ax = m.plot.landscape(field='bedrock+lakes', i=-1, oversample=1,
                                   hillshade=False, **extra)
        painted = np.abs(_map_rgb(ax) - LAKE_RGB).max(axis=-1) < 1e-9
        np.testing.assert_array_equal(painted, true_lake)
        if extra:
            # the section's water band (zorder 2.5) is painted from the same
            # flood, so it must be wet where the bed row is.
            wet = sum(float(np.asarray(im.get_array())[..., 3].sum())
                      for im in _section_axes(fig).get_images()
                      if im.get_zorder() == 2.5)
            assert wet > 0.0, "section paints no water through the basin"
        plt.close(fig)


def test_field_smoothing_requires_a_positive_threshold(_basin_model):
    """ice_smoothing='field' clips H to 2*H_threshold before thresholding, so
    at H_threshold <= 0 it draws NO ice at all. style='raw' resolves
    H_threshold to 0, which made landscape(style='raw', ice_smoothing='field')
    a silently ice-free render — it must raise instead."""
    with pytest.raises(ValueError, match='H_threshold'):
        _basin_model.plot.landscape(style='raw', ice_smoothing='field', i=-1)
    with pytest.raises(ValueError, match='H_threshold'):
        _basin_model.plot.landscape(field='bedrock+ice', ice_extent='cells',
                                    ice_smoothing='field', H_threshold=0.0,
                                    i=-1)
    plt.close('all')


def test_animate_freezes_the_auto_colour_scales(_iced_model, tmp_path,
                                                monkeypatch):
    """landscape() autoscales z_max and the ice norm per frame, so an animation
    put every frame on its own scale (bed top swinging with the topography, the
    ice bar collapsing to 0-1 m on ice-free frames). animate_landscape must
    resolve both ONCE over the run and pass them down."""
    import matplotlib.animation as anm
    m = _iced_model
    mdl = m.plot.model
    kwargs = _frozen_scale_kwargs(mdl, dict(field='bedrock+ice',
                                            ice_cmap='Blues'))
    assert kwargs['z_max'] == float(np.nanmax(np.asarray(mdl.z_out)))
    assert kwargs['H_max'] == float(mdl.hc_over_H
                                    * np.nanmax(np.asarray(mdl.H_out)))
    # explicit values still win
    assert _frozen_scale_kwargs(mdl, dict(z_max=7.0))['z_max'] == 7.0

    def limits(i, **kw):
        fig, _ = m.plot.landscape(i=i, oversample=2, **kw)
        got = {a.get_ylabel(): tuple(a.get_ylim()) for a in fig.axes
               if a.get_ylabel() in ('Surface elevation (m)',
                                     'Ice column depth (m)')}
        plt.close(fig)
        return got

    assert set(limits(0, **kwargs)) == {'Surface elevation (m)',
                                        'Ice column depth (m)'}, \
        "test setup: the bed and ice colorbars were not both found"
    assert limits(0, **kwargs) == limits(-1, **kwargs)
    auto = dict(field='bedrock+ice', ice_cmap='Blues')
    assert limits(0, **auto) != limits(-1, **auto), \
        "test setup: the per-frame autoscale does not move on this run"

    # ...and animate_landscape actually hands the frozen values to landscape().
    seen = []
    plotter = type(m.plot)
    original = plotter.landscape

    def spy(self, **kw):
        seen.append(kw)
        return original(self, **kw)

    monkeypatch.setattr(plotter, 'landscape', spy)
    monkeypatch.setattr(anm.FuncAnimation, 'save', lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():           # the no-op save() never renders
        warnings.simplefilter('ignore', UserWarning)
        m.plot.animate_landscape(path='frozen', n_workers=1,
                                 field='bedrock+ice', ice_cmap='Blues')
    assert seen and all(kw['z_max'] == kwargs['z_max']
                        and kw['H_max'] == kwargs['H_max'] for kw in seen)
    plt.close('all')


def test_animate_freezes_the_scales_over_the_selected_frames(_iced_model, tmp_path,
                                                             monkeypatch):
    """The freeze must follow ``frames``: resolved over the whole run, a movie
    of a late window is scaled to relief (and ice) it never shows."""
    import matplotlib.animation as anm
    m = _iced_model
    mdl = m.plot.model
    z, H = np.asarray(mdl.z_out), np.asarray(mdl.H_out)
    windows = ([0, 1], [2, 3])
    expect = [(float(np.nanmax(z[w])), float(mdl.hc_over_H * np.nanmax(H[w])))
              for w in windows]
    assert expect[0] != expect[1], \
        "test setup: the two windows must not share their scales"

    seen = []
    plotter = type(m.plot)
    original = plotter.landscape

    def spy(self, **kw):
        seen.append((kw.get('z_max'), kw.get('H_max')))
        return original(self, **kw)

    monkeypatch.setattr(plotter, 'landscape', spy)
    monkeypatch.setattr(anm.FuncAnimation, 'save', lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    got = []
    for window in windows:
        seen.clear()
        with warnings.catch_warnings():       # the no-op save() never renders
            warnings.simplefilter('ignore', UserWarning)
            m.plot.animate_landscape(path=f'window{window[0]}', n_workers=1,
                                     frames=window, field='bedrock+ice',
                                     oversample=1, hillshade=False)
        assert seen, "no frame was rendered for this window"
        got.append(seen[0])
    assert got == expect
    plt.close('all')


def test_cross_section_ylim_contains_its_own_profile(_basin_model):
    """The auto z_max came from the SMOOTHED map field while the section draws
    the unsmoothed composite against the same limit, so a peak sharper than the
    Gaussian ran off the top of the panel."""
    m = _basin_model
    fig, ax = m.plot.landscape(field='bedrock', i=-1, cross_section=15.0)
    ax_cs = _section_axes(fig)
    profile = _section_lines(ax_cs, _map_rgb(ax).shape[1])[-1]
    assert profile.max() - np.median(profile) > 300.0, \
        "test setup: the 1-cell peak did not survive the run"
    assert ax_cs.get_ylim()[1] >= profile.max()
    plt.close(fig)


def test_map_raster_registers_with_the_node_coordinates(_basin_model):
    """imshow places pixel CENTRES, but the rendered array is node-valued: with
    extent [0, Lx] every pixel sat half a subgrid pixel off the contour /
    trimline / section-line coordinates (249 m at oversample=1)."""
    m = _basin_model
    mdl = m.plot.model
    for ov in (1, 4):
        fig, ax = m.plot.landscape(field='bedrock', i=-1, oversample=ov,
                                   hillshade=False)
        im = ax.get_images()[0]
        ny_sub, nx_sub = _map_rgb(ax).shape[:2]
        x0, x1, y0, y1 = im.get_extent()
        # the coordinates the overlays use
        x_axis = np.linspace(0, mdl.Lx / 1e3, nx_sub)
        y_axis = np.linspace(0, mdl.Ly / 1e3, ny_sub)
        np.testing.assert_allclose(
            x0 + (np.arange(nx_sub) + 0.5) * (x1 - x0) / nx_sub, x_axis,
            atol=1e-9)
        np.testing.assert_allclose(
            y0 + (np.arange(ny_sub) + 0.5) * (y1 - y0) / ny_sub, y_axis,
            atol=1e-9)
        # the axes still show exactly the domain
        assert ax.get_xlim() == (0.0, mdl.Lx / 1e3)
        assert ax.get_ylim() == (0.0, mdl.Ly / 1e3)
        plt.close(fig)


def test_section_paints_a_band_under_every_masked_ice_column(_iced_model):
    """The section's bed line was the BICUBIC zb_sub against the BILINEAR
    filled ice surface, so wherever the bicubic overshot above the fill the
    band collapsed to nothing on columns the map paints as ice."""
    m = _iced_model
    fig, ax = m.plot.landscape(field='bedrock+ice', i=-1, oversample=4,
                               H_threshold=0.0, ice_sigma_cells=0,
                               hillshade=False, contour_interval=0,
                               ice_shading='flat',   # locate ice by ice_color
                               cross_section=1.5)
    rgb = _map_rgb(ax)
    ny_sub, nx_sub = rgb.shape[:2]
    j_cs = int(round(1.5e3 / (m.plot.model.Ly / (ny_sub - 1))))
    icy = np.abs(rgb[j_cs] - ICE_RGB).max(axis=-1) < 1e-9
    assert icy.any(), "test setup: no ice on the section row"
    bedrock, profile = _section_lines(_section_axes(fig), nx_sub)
    assert (profile[icy] - bedrock[icy] > 0).all(), (
        f"{int((profile[icy] - bedrock[icy] <= 0).sum())} of {int(icy.sum())} "
        "masked ice columns paint no section band")
    plt.close(fig)


def test_bare_bed_colorbar_is_labelled_bedrock(_basin_model):
    """Without ice the composite IS the bare bed, so the bar must not claim to
    be the surface (map() calls it 'Bedrock elevation (m)')."""
    m = _basin_model
    for field, want in (('bedrock', 'Bedrock elevation (m)'),
                        ('bedrock+lakes', 'Bedrock elevation (m)'),
                        ('bedrock+ice', 'Surface elevation (m)')):
        fig, _ = m.plot.landscape(field=field, i=-1, hillshade=False)
        assert want in [a.get_ylabel() for a in fig.axes]
        plt.close(fig)


def test_two_colorbars_do_not_collide_at_paper_width(_iced_model):
    """The veil adds a second colorbar, and its pad is in INCHES: it has to
    clear the elevation bar's tick labels + rotated title at every figure
    width. Measured, that is a typographic width (~0.7 in), so the pad is
    floored there and only grows on a figure wider than 14 in — a purely
    relative pad overlaps the two bars below ~14 in."""
    m = _iced_model
    for fw in (5, 7, 14, 20):
        fig, ax = m.plot.landscape(field='bedrock+ice', i=-1, fig_width=fw)
        fig.canvas.draw()
        bars = {a.get_ylabel(): a for a in fig.axes}
        bed, ice = bars['Surface elevation (m)'], bars['Ice column depth (m)']
        # tightbbox: the bar plus its tick labels and title, i.e. what must fit
        bed_x1 = (bed.get_tightbbox(fig.canvas.get_renderer())
                  .transformed(fig.transFigure.inverted()).x1)
        assert ice.get_position().x0 > bed_x1, \
            f"the two colorbars collide at fig_width={fw}"
        # gate: the text margins grow with the font sizes (2026-11-05); at the
        # matplotlib defaults a 5 in figure keeps 47 % for the map (was 50 %+)
        assert ax.get_position().width > 0.45, \
            f"the map is squeezed to {ax.get_position().width:.2f} at fig_width={fw}"
        plt.close(fig)


def test_section_names_its_ela_and_states_the_exaggeration(_iced_model):
    """The section's ELA was a black dash indistinguishable from the contour
    grid, and the panel's strong vertical exaggeration was unstated."""
    from siim.plotting._style import COLORS
    m = _iced_model
    fig, _ = m.plot.landscape(field='bedrock+ice', i=-1, cross_section=15.0)
    ax_cs = _section_axes(fig)
    texts = [t.get_text() for t in ax_cs.texts]
    assert 'ELA' in texts
    assert any(t.startswith('VE ') for t in texts), texts
    zela = float(m.plot.model._zELA_output[-1])
    ela_lines = [ln for ln in ax_cs.lines
                 if ln.get_linestyle() == '--'
                 and np.allclose(ln.get_ydata(), zela)]
    assert ela_lines, "no dashed ELA line in the section"
    # gate: the landscape module's ELA_COLOR alias is gone (Phase C cruft);
    # the same colour is read from its one home, COLORS['ela'].
    assert matplotlib.colors.to_hex(ela_lines[0].get_color()) == COLORS['ela']
    plt.close(fig)


# --- averaged cross-sections (0.9.8) ----------------------------------------

def _band_bounds(band):
    """The lower/upper edge of a fill_between band, per unique x (the polygon
    carries both edges, endpoints doubled)."""
    verts = band.get_paths()[0].vertices
    xs = np.round(verts[:, 0], 9)
    edges = [(verts[xs == x, 1].min(), verts[xs == x, 1].max())
             for x in np.unique(xs)]
    return np.asarray([lo for lo, _ in edges]), np.asarray([hi for _, hi in edges])


def _quartile_lines(ax_cs, n):
    """The mean section's dashed bed-quartile lines (the ELA dash spans two
    points, so the column count separates them)."""
    return [ln for ln in ax_cs.lines
            if ln.get_linestyle() == '--' and len(ln.get_ydata()) == n]


# gate: 'median' removed by decision 2026-09-17 (mean + IQR only)
def test_averaged_cross_section_reduces_every_row(_basin_model):
    """cross_section='mean' replaces the single row with the mean over the
    rendered rows: at oversample=1 the surface line is that mean of the stored
    bed (this run is ice-free, so the composite IS the bed), the map carries no
    locator line, and no ice is painted."""
    m = _basin_model
    fig, ax = m.plot.landscape(field='bedrock+ice', i=-1, oversample=1,
                               cross_section='mean', hillshade=False,
                               contour_interval=0)
    nx_sub = _map_rgb(ax).shape[1]
    profile = _section_lines(_section_axes(fig), nx_sub)[-1]
    np.testing.assert_allclose(profile, np.mean(np.asarray(m.zb_out[-1]), axis=0),
                               atol=1e-6)
    assert len(ax.lines) == 0, "an averaged section has no single y to mark"
    # Same zero-alpha check as the phantom-ice guard: the section's ice layer
    # is the only zorder-2 image.
    ice_imgs = [im for a in fig.axes for im in a.get_images()
                if im.get_zorder() == 2]
    assert ice_imgs, "cross-section ice layer not found"
    painted = sum(float(np.asarray(im.get_array())[..., 3].sum())
                  for im in ice_imgs)
    assert painted == 0.0, f"ice painted on an ice-free average (alpha {painted})"
    plt.close(fig)

    # control: a y in km still gets its locator line
    fig, ax = m.plot.landscape(field='bedrock+ice', i=-1, oversample=1,
                               cross_section=15.0, hillshade=False,
                               contour_interval=0)
    assert len(ax.lines) == 1
    plt.close(fig)


# gate: shorter labels so the one-row legend fits narrow figures (Eric, 2026-11-05)
SECTION_LEGEND = ['ice IQR', 'bed IQR']


def _section_legend(ax_cs):
    """The averaged section's legend labels ([] when there is no legend)."""
    legend = ax_cs.get_legend()
    return [t.get_text() for t in legend.get_texts()] if legend else []


# gate: bed IQR is now two dashed lines, decision 2026-09-17
def test_mean_cross_section_shades_the_interquartile_range(_iced_model):
    """A mean alone cannot say whether the rows it averaged agree, so the
    section shows their spread: the ice thickness's 25-75% band shaded on the
    mean bed, the bed's quartiles dashed (a second fill read as more ice, not
    as topography). On a
    bare bed at oversample=1 the composite IS the stored bed, so the dashed
    pair is exactly the stored rows' quartiles."""
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgba
    from siim.plotting._style import COLORS
    m = _iced_model
    fig, ax = m.plot.landscape(field='bedrock', i=-1, oversample=1,
                               cross_section='mean', hillshade=False,
                               contour_interval=0)
    nx_sub = _map_rgb(ax).shape[1]
    ax_cs = _section_axes(fig)
    # gate: bed IQR is now two dashed lines, decision 2026-09-17
    assert not [c for c in ax_cs.collections if isinstance(c, PolyCollection)], \
        "a bare bed has no ice to shade"
    want = np.percentile(np.asarray(m.zb_out[-1]), [25, 75], axis=0)
    # gate: interior columns only; on the shipped-default state (ED-171 dropped the
    # fixture's closure/damping pins) a fixed_value border column can be flat
    assert (want[1] - want[0])[1:-1].min() > 1.0, "test setup: the rows barely spread"
    dashed = _quartile_lines(ax_cs, nx_sub)
    assert len(dashed) == 2, "the bed quartiles are two dashed lines"
    np.testing.assert_allclose([ln.get_ydata() for ln in dashed], want, atol=1e-6)
    assert all(ln.get_color() == 'black' for ln in dashed)
    # gate: legend replaces the text note (Eric, 2026-09-21)
    assert _section_legend(ax_cs) == SECTION_LEGEND[1:]
    plt.close(fig)

    # with ice drawn, the ice THICKNESS carries the only band. With the cell
    # extent, no trunk ribbons and a crisp mask, the section's rows at
    # oversample=1 are the stored surface and bed, so the band is exactly
    # mean bed + quartiles of the drawn column hc*H.
    fig, _ = m.plot.landscape(field='bedrock+ice', i=-1, oversample=1,
                              ice_extent='cells', ice_sigma_cells=0,
                              trunk_display='none', cross_section='mean',
                              hillshade=False, contour_interval=0)
    ax_cs = _section_axes(fig)
    assert _section_legend(ax_cs) == SECTION_LEGEND
    bands = [c for c in ax_cs.collections if isinstance(c, PolyCollection)]
    assert len(bands) == 1
    np.testing.assert_allclose(bands[0].get_facecolor()[0],
                               to_rgba(COLORS['ice'], 0.22), atol=1e-9)
    dashed = _quartile_lines(ax_cs, nx_sub)
    assert len(dashed) == 2
    # the section's ice fill is OPAQUE (zorder 2): anything under it is invisible
    assert bands[0].get_zorder() > 2
    assert min(ln.get_zorder() for ln in dashed) > bands[0].get_zorder()
    zb = np.asarray(m.zb_out[-1])
    q = np.percentile(m.hc_over_H * np.asarray(m.H_out[-1]), [25, 75], axis=0)
    assert (q[1] - q[0]).max() > 1.0, "test setup: no ice column spreads anywhere"
    # gate: the band is the ice THICKNESS IQR hung on the mean bed, not the
    # surface's (decision 2026-09-22-section-band-ice-thickness-iqr)
    np.testing.assert_allclose(_band_bounds(bands[0]), zb.mean(axis=0) + q, atol=1e-6)
    plt.close(fig)

    # the default style repaints the surface (smoothed mask), but the
    # band still hangs on the drawn mean bed: its lower edge is bed + H25 >= bed
    fig, _ = m.plot.landscape(field='bedrock+ice', i=-1, oversample=1,
                              cross_section='mean', hillshade=False,
                              contour_interval=0)
    ax_cs = _section_axes(fig)
    band = [c for c in ax_cs.collections if isinstance(c, PolyCollection)][0]
    bed_line = [np.asarray(ln.get_ydata()) for ln in ax_cs.lines
                if ln.get_linestyle() == '-' and len(ln.get_ydata()) == nx_sub][0]
    assert (_band_bounds(band)[0] >= bed_line - 1e-6).all()
    plt.close(fig)

    # a single row is one landscape: no band, no quartile lines, no legend
    fig, _ = m.plot.landscape(field='bedrock+ice', i=-1, oversample=1,
                              cross_section=15.0, hillshade=False,
                              contour_interval=0)
    ax_cs = _section_axes(fig)
    assert not [c for c in ax_cs.collections if isinstance(c, PolyCollection)]
    assert not _quartile_lines(ax_cs, nx_sub)
    assert _section_legend(ax_cs) == []
    plt.close(fig)


@pytest.mark.parametrize('bad', ['bogus', 'median'])
def test_cross_section_rejects_an_unknown_statistic(_basin_model, bad):
    """Any other string is a typo, not a y — caught with the other argument
    checks, before the map exists, so a typo leaks no figure."""
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="'mean'"):
        _basin_model.plot.landscape(field='bedrock', i=-1, cross_section=bad)
    assert plt.get_fignums() == before, "the rejected render left a figure open"
    plt.close('all')


def test_animate_landscape_accepts_an_averaged_section(_basin_model, monkeypatch,
                                                       tmp_path):
    """The string is frame-independent and picklable, so movies inherit the
    averaged profile through the usual landscape_kwargs pass-through."""
    m = _basin_model
    drawn = []

    def save(movie, *args, **kwargs):
        movie._draw_was_started = True
        for idx in movie.new_frame_seq():
            movie._func(idx)
            drawn.append(idx)
        # gate: legend replaces the text note; the parallel-worker block that
        # re-asserted the same landscape() call was dropped (Eric, 2026-09-21)
        assert _section_legend(_section_axes(movie._fig)) == SECTION_LEGEND

    monkeypatch.setattr('matplotlib.animation.Animation.save', save)
    out = m.plot.animate_landscape(path=str(tmp_path / 'averaged.mp4'),
                                   n_workers=1, style='raw',
                                   cross_section='mean')
    assert out.endswith('averaged.mp4')
    assert drawn == list(range(len(m.output_times)))
    plt.close('all')


def test_ribbon_surface_never_lifts_a_bed_below_zero(_iced_model, monkeypatch):
    """The ribbon layer's surface is an unset 0 off its ribbons, so the terrain
    may read it only where the ribbon depth is positive, and never below the
    bed (audit #69, whose test went with ice_time_avg). With frame i's bed and
    surface lowered 2 km below 0 m, every drawn-ice pixel of the composite
    stays below 0 m, in both extents."""
    import scipy.ndimage
    m = _iced_model
    i = -1
    zb0, z0 = np.array(m.zb_out[i]), np.array(m.z_out[i])
    seen = []
    smooth = scipy.ndimage.gaussian_filter

    def spy(a, **kw):
        seen.append(np.array(a))
        return smooth(a, **kw)
    monkeypatch.setattr(scipy.ndimage, 'gaussian_filter', spy)
    try:
        m.zb_out[i], m.z_out[i] = zb0 - 2000.0, z0 - 2000.0
        assert (z0 - 2000.0).max() < -500.0, "test setup: frame i must sit below 0 m"
        for extent in ('footprint', 'cells'):
            fig, ax = m.plot.landscape(
                field='bedrock+ice', i=i, ice_extent=extent, oversample=1,
                H_threshold=0.0, hillshade=False, ice_shading='flat',
                trunk_display='ribbons', sigma_cells=1.0, z_min=-2500.0,
                contour_interval=0)
            terrain = seen[-1]                      # the composite, pre-smoothing
            drawn = np.abs(_map_rgb(ax) - ICE_RGB).max(axis=-1) < 1e-9
            plt.close(fig)
            assert drawn.sum() > 50, f"test setup: {extent} draws little ice"
            assert terrain[drawn].max() < 0.0
    finally:
        m.zb_out[i], m.z_out[i] = zb0, z0
    plt.close('all')
