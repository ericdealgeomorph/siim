# Sphinx configuration for the siim documentation.
#
# Local build (from an environment with docs/requirements.txt installed):
#   python -m sphinx -W -b html docs docs/_build/html
# The standard 1D/2D stack is installed on Read the Docs. Optional adapter or
# presentation dependencies that are absent are mocked so autodoc can still
# import their public modules.
import importlib.util
import os
import sys

sys.path.insert(0, os.path.abspath('..'))  # repo root (editable installs too)

project = 'siim'
author = 'Eric Deal'
copyright = '2026, Eric Deal'

extensions = [
    'myst_nb',                    # markdown + (optional) notebook pages
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.bibtex',
]

# Mock only optional adapter/presentation dependencies that are actually
# missing. NumPy/SciPy are hard requirements and are never mocked.
autodoc_mock_imports = [
    m for m in ('numba', 'xsimlab', 'fastscape', 'matplotlib', 'mpl_toolkits',
                'pandas', 'tqdm')
    if importlib.util.find_spec(m) is None
]

autosummary_generate = False      # module pages are hand-written (docs/api/)
autodoc_member_order = 'bysource'

napoleon_google_docstring = False
napoleon_numpy_docstring = True

myst_enable_extensions = ['dollarmath', 'amsmath']
# Execute notebook pages (markdown notebooks carrying a kernelspec — e.g. the
# 1D getting-started walkthrough) at build time so the examples stay tested.
# The quick 1D notebook page executes during the build. The larger 2D example
# remains plain Markdown so documentation builds do not run a landscape model.
nb_execution_mode = 'auto'
nb_execution_raise_on_error = True   # a broken example fails the build
nb_execution_timeout = 300           # allow for numba JIT on the first run
nb_output_stderr = 'remove'          # keep tqdm/progress bars out of the render

bibtex_bibfiles = ['references.bib']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

templates_path = []
# Internal development notes are not part of the rendered public documentation;
# exclude that tree and common generated files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**/__pycache__', 'dev/**']

html_theme = 'furo'
html_title = 'siim'
