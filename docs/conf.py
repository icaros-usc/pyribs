#!/usr/bin/env python
#
# ribs documentation build configuration file, created by
# sphinx-quickstart.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#
import os
import sys

import ribs
import sphinx_material

sys.path.insert(0, os.path.abspath('..'))
DEV_MODE = os.environ.get("DOCS_MODE", "regular") == "dev"

# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'autodocsumm',
    'sphinx_material',
    'sphinx_copybutton',
    'myst_nb',  # Covers both Markdown and Jupyter notebooks.
]

# Napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

# MyST NB -- exclude execution of Jupyter notebooks because they can take a
# while to run.
jupyter_execute_notebooks = "off"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = ['.rst', '.md', '.ipynb']

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'pyribs'
copyright = "2020, ICAROS Lab"
author = "ICAROS Lab"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = ribs.__version__
# The full version, including alpha/beta/rc tags.
release = ribs.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
#  pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output -------------------------------------------

html_show_sourcelink = True
html_sidebars = {
    "**": [
        "logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"
    ]
}

html_theme_path = sphinx_material.html_theme_path()
html_context = sphinx_material.get_html_context()
html_theme = "sphinx_material"
html_logo = "_static/imgs/logo.svg"
html_favicon = "_static/imgs/favicon.ico"
html_title = "pyribs v" + ribs.__version__

# material theme options (see theme.conf for more information)
html_theme_options = {
    "nav_title": "pyribs",
    "base_url": "https://ribs.readthedocs.io",
    "repo_url": "https://github.com/icaros-usc/pyribs",
    "repo_name": "pyribs",
    "google_analytics_account": None,
    "html_minify": not DEV_MODE,
    "css_minify": not DEV_MODE,
    #  "logo_icon": "&#xe869",
    "repo_type": "github",
    "globaltoc_depth": 2,
    "color_primary": "indigo",
    "color_accent": "light-blue",
    "touch_icon": "images/apple-icon-152x152.png",
    #  "theme_color": "#2196f3",
    "master_doc": False,
    "nav_links": [{
        "href": "index",
        "internal": True,
        "title": "pyribs"
    },],
    "heroes": {
        "index": "A bare-bones quality diversity optimization library."
    },
    "version_dropdown": False,
    "version_json": None,
    #  "version_info": {
    #      "Release": "https://bashtage.github.io/sphinx-material/",
    #      "Development": "https://bashtage.github.io/sphinx-material/devel/",
    #      "Release (rel)": "/sphinx-material/",
    #      "Development (rel)": "/sphinx-material/devel/",
    #  },
    "table_classes": ["plain"],
}

html_last_updated_fmt = ""

html_use_index = True
html_domain_indices = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'custom.css',
]

# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'ribsdoc'

# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'ribs.tex', 'pyribs Documentation', 'ICAROS Lab', 'manual'),
]

# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, 'ribs', 'pyribs Documentation', [author], 1)]

# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'ribs', 'pyribs Documentation', author, 'ribs',
     'One line description of project.', 'Miscellaneous'),
]

# -- Extension config -------------------------------------------------

autodoc_default_options = {
    "member-order": "bysource",
    "autosummary": True,
    "inherited-members": True,
}
