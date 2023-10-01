# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = u"quantbullet"
copyright = u"2023, Yiming Zhang"
author = u"Yiming Zhang"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx"
]
autoapi_dirs = ["../src"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/*_dev.ipynb"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = "sphinx_rtd_theme"
html_theme = "pydata_sphinx_theme"

# do not execute notebooks
# nbsphinx_execute = 'never'
# for myst-nb, the following is needed
nb_execution_mode = "off"

# -- View on Github ------------------------------------------------------

# The following is used by sphinx.ext.linkcode to provide links to github
# github_version has to be master/docs/ for the link to work
# for sphinx_rtd_theme, the following is needed
# html_context = {
#   'display_github': True,
#   'github_user': 'YimingZhang07',
#   'github_repo': 'quantbullet',
#   'github_version': 'master/docs/',
# }

# for pydata_sphinx_theme, the following is needed

html_context = {
  'default_mode': 'light',
  'github_user': 'YimingZhang07',
  'github_repo': 'quantbullet',
  'github_version': 'master',
  'doc_path': 'docs',
}

html_theme_options = {
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/YimingZhang07/quantbullet",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        },
        {
            "name": "My Website",
            "url": "https://yimingzhang.netlify.app/",
            "icon": "fa-solid fa-blog",
            "type": "fontawesome",
        }
   ]
}