# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
from os.path import abspath, dirname

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))
sys.path.insert(0, abspath("sphinxext"))
from github_link import make_linkcode_resolve

# -- Project information -----------------------------------------------------

project = "imbalanced-ensemble"
copyright = "2021, Zhining Liu"
author = "Zhining Liu"


from imbens import __version__

version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    # "sphinxcontrib.bibtex",
    "numpydoc",
    # "sphinx_issues",
    "sphinx_gallery.gen_gallery",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for autodoc ------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
}

# generate autosummary even if no references
autosummary_generate = True

# -- Options for numpydoc -----------------------------------------------------

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = False

# -- Options for sphinx-gallery -----------------------------------------------

# Generate the plot for the gallery
plot_gallery = True

# sphinx-gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "gallery_dirs": ["auto_examples"],
    "doc_module": (
        "imbens",
        "imbens.ensemble",
    ),
    "backreferences_dir": os.path.join("back_references"),
    "show_memory": True,
    "reference_url": {"imbalanced-ensemble": None},
    "nested_sections": False,
}

# # -- Options for github link for what's new -----------------------------------

# Config for sphinx_issues
# issues_uri = "https://github.com/ZhiningLiu1998/imbalanced-ensemble/issues/{issue}"
# issues_github_path = "ZhiningLiu1998/imbalanced-ensemble"
# issues_user_uri = "https://github.com/{user}"

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "imbens",
    "https://github.com/ZhiningLiu1998/"
    "imbalanced-ensemble/blob/{revision}/"
    "{package}/{path}#L{lineno}",
)

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


def setup(app):
    app.add_css_file("css/my_theme.css")
