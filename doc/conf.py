#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
import datetime
import sphinx_rtd_theme
import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from version_from_git import version_number_from_git

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinxcontrib.bibtex',
    'sphinx_autodoc_typehints',
]

add_module_names = False
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

copyright = '{}, Martin Bauer'.format(datetime.datetime.now().year)
author = 'Martin Bauer'
version = version_number_from_git()
release = version_number_from_git()
language = None
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
default_role = 'any'
pygments_style = 'sphinx'
todo_include_todos = False

# Options for HTML output

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme = 'sphinx_rtd_theme'
htmlhelp_basename = 'pystencilsdoc'
html_sidebars = {'**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']}

# NbSphinx configuration
nbsphinx_execute = 'never'
nbsphinx_codecell_lexer = 'python3'

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'python': ('https://docs.python.org/3.6', None),
                       'numpy': ('https://docs.scipy.org/doc/numpy/', None),
                       'matplotlib': ('https://matplotlib.org/', None),
                       'sympy': ('https://docs.sympy.org/latest/', None),
                       }

autodoc_member_order = 'bysource'
bibtex_bibfiles = ['sphinx/pystencils.bib']

project = 'pystencils'
html_logo = "img/logo.png"
