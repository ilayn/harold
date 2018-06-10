#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# harold documentation build configuration file, created by
# sphinx-quickstart on Sat Sep 19 17:38:48 2015.
#

import sys
import os
import mock
# import cloud_sptheme for themes, etc
import cloud_sptheme as csp
from harold import __version__ as release

MOCK_MODULES = ['tabulate', 'scipy.signal']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.7.4'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',

    # cloud's extensions
    'cloud_sptheme.ext.autodoc_sections',
    'cloud_sptheme.ext.relbar_links',
    'cloud_sptheme.ext.escaped_samp_literals',
    'cloud_sptheme.ext.issue_tracker',
    'cloud_sptheme.ext.table_styling',
    ]

# == CSP theme ===============================================================
# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
source_encoding = 'utf-8'

# The master toctree document.
# master_doc = 'contents'

# The frontpage document.
# index_doc = 'index'

# General information about the project.
project = "harold"
author = "Ilhan Polat"
copyright = "2015-2018, " + author

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

# release: The full version, including alpha/beta/rc tags.
# version: The short X.Y version.
version = csp.get_version(release)

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# #add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# #show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# =============================================================================
# Options for all output
# =============================================================================
todo_include_todos = True
keep_warnings = True
issue_tracker_url = "gh:ilayn/harold/issues"

# =============================================================================
# Options for HTML output
# =============================================================================

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = os.environ.get("SPHINX_THEME") or 'cloud'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = [csp.get_theme_dir()]

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "%s v%s Documentation".format(project, release)

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "%s %s Documentation".format(project, version)

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = os.path.join("_static", "masthead.png")

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# #html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# #html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = False

# Custom sidebar templates, maps document names to template names.
html_sidebars = {'**': ['searchbox.html', 'globaltoc.html']}


# ============================================================================

#
autodoc_member_order = 'bysource'

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'harold'
copyright = '2018, Ilhan Polat'
author = 'Ilhan Polat'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'setup.py']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'harolddoc'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  (master_doc, 'harold.tex', 'harold Documentation',
   'Ilhan Polat', 'manual'),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'harold', 'harold Documentation',
     [author], 1)
]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  (master_doc, 'harold', 'harold Documentation',
   author, 'harold', 'A Python Control Systems Toolbox',
   'Miscellaneous'),
]
