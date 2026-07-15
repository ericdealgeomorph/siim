# Sphinx configuration for the siim documentation.
#
# Local build (miniforge base env has the full model stack):
#   python -m sphinx -W -b html docs docs/_build/html
# On Read the Docs the compiled model stack (fastscapelib-fortran) is not
# installable; whatever heavy dependency is missing gets mocked below so
# autodoc can still import every module.
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

# Mock only what is actually missing (nothing, locally; the heavy 2D stack
# on Read the Docs). numpy/scipy are hard requirements and never mocked.
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
# Pages that need the 2D fastscape stack stay plain (no-kernelspec) markdown and
# are not executed: Read the Docs can't compile fastscapelib-fortran.
nb_execution_mode = 'auto'
nb_execution_raise_on_error = True   # a broken example fails the build
nb_execution_timeout = 300           # allow for numba JIT on the first run
nb_output_stderr = 'remove'          # keep tqdm/progress bars out of the render

bibtex_bibfiles = ['references.bib']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

templates_path = []
# docs/dev/ holds internal dev notes (plans, handoffs) that are not part of the
# rendered docs; excluding the whole tree keeps them out of the toctree (and
# future-proofs new notes added during the ongoing refactor).
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**/__pycache__', 'dev/**']

html_theme = 'furo'
html_title = 'siim'
