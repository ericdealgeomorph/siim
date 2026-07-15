"""Output-path policy: results land under ./model_outputs/<kind>/ in the CWD
(never inside the installed package), with absolute names passed through."""
from pathlib import Path

import siim._output as _output
from siim._output import output_path


def test_relative_names_land_under_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pkg_dir = Path(_output.__file__).resolve().parent
    for kind in ('images', 'movies', 'saved_models'):
        out = Path(output_path(f'run.{kind}', kind))
        assert out == tmp_path / 'model_outputs' / kind / f'run.{kind}'
        assert out.parent.is_dir()            # created on demand
        assert pkg_dir not in out.parents     # never inside the package


def test_absolute_name_passes_through(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / 'elsewhere' / 'fig.png'
    assert Path(output_path(str(target), 'images')) == target
    assert target.parent.is_dir()                      # parent created
    assert not (tmp_path / 'model_outputs').exists()   # default root untouched
