# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Minimum version, enforced by sphinx
needs_sphinx = '4.3'

project = 'clustersort'
copyright = '2023, Flynn OConnell'
author = 'Flynn OConnell'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.coverage',
    'numpydoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_design',
    'sphinx.ext.githubpages'
]

templates_path = ['_templates']

source_suffix = '.rst'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_context = {"default_mode": "dark"}
add_function_parentheses = False

# -----------------------------------------------------------------------------
# Autosummary/numpydoc
# -----------------------------------------------------------------------------

autosummary_generate = True
numpydoc_show_class_members = False
