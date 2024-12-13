import datetime
import re

from pystencils import __version__ as pystencils_version

project = "pystencils"
html_title = "pystencils Documentation"

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
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.inheritance_diagram",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "myst_nb",
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
    "cupy": ("https://docs.cupy.dev/en/stable/", None),
}

# -- Options for inheritance diagrams-----------------------------------------

inheritance_graph_attrs = {
    "bgcolor": "white",
}

# -- Options for MyST / MyST-NB ----------------------------------------------

nb_execution_mode = "off"  # do not execute notebooks by default

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/fixtables.css",
]
html_theme_options = {
    "logo": {
        "image_light": "_static/img/pystencils-logo-light.svg",
        "image_dark": "_static/img/pystencils-logo-dark.svg",
    }
}

# NbSphinx configuration

nbsphinx_execute = "never"
nbsphinx_codecell_lexer = "python3"

#   BibTex
bibtex_bibfiles = ["pystencils.bib"]
