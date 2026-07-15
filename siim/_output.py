"""Default output locations for siim plotters and saved model states.

siim writes into ``model_outputs/<kind>/<name>`` under the **current working
directory** (created on demand), so an installed package drops results where
the user is working rather than inside the package install. ``kind`` is one of
``'images'``, ``'movies'``, ``'saved_models'``. Pass an absolute ``name`` to
write somewhere specific instead.
"""
from pathlib import Path


def output_path(name, kind):
    """Map an output name to ``./model_outputs/<kind>/<name>`` under the CWD.

    A relative ``name`` is placed under that directory; an absolute path is
    returned unchanged so callers can still write anywhere explicitly. The
    target directory is created. Returns a ``str`` (no extension is added — the
    caller appends ``.mp4`` etc.). The working directory is resolved at call
    time, so it tracks any ``os.chdir`` between import and use.
    """
    p = Path(name)
    out = p if p.is_absolute() else Path.cwd() / 'model_outputs' / kind / p
    out.parent.mkdir(parents=True, exist_ok=True)
    return str(out)
