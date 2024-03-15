import datetime
import re

from pystencils import __version__ as pystencils_version

project = "pystencils"
html_logo = "_static/img/logo.png"

copyright = (
    f"{datetime.datetime.now().year}, Martin Bauer, Markus Holzer, Frederik Hennig"
)
author = "Martin Bauer, Markus Holzer, Frederik Hennig"

version = re.sub(r"(\d+\.\d+)\.\d+(.*)", r"\1\2", pystencils_version)
version = re.sub(r"(\.dev\d+).*?$", r"\1", version)
# The full version, including alpha/beta/rc tags.
release = pystencils_version

language = "en"
default_role = "any"
pygments_style = "sphinx"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.8", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]

# NbSphinx configuration

nbsphinx_execute = 'never'
nbsphinx_codecell_lexer = 'python3'

#   BibTex
bibtex_bibfiles = ['pystencils.bib']
