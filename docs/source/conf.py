# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Adjust depending on your folder structure

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'EnTiSe'
copyright = '2024, Markus Doepfert, Patrick Buchenberg'
author = 'Markus Doepfert, Patrick Buchenberg'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",       # Automatically document code
    "sphinx.ext.napoleon",      # Support Google/NumPy-style docstrings
    "sphinx.ext.viewcode",      # Add links to source code
    "myst_nb",                  # Support Jupyter Notebooks with MyST-NB
    "sphinxcontrib.mermaid"     # Support Mermaid diagrams
    # "autoapi"                 # Generate docs directly from modules (disabled for now)
]

# MyST-NB configuration
myst_enable_extensions = [
    "amsmath",                  # For math equations
    "colon_fence",              # For code blocks with colons
    "deflist",                  # For definition lists
    "dollarmath",               # For inline math with $
    "html_image",               # For HTML images
    "html_admonition",          # For HTML admonitions
    "tasklist",                 # For task lists
]

# Notebook execution settings
nb_execution_mode = "off"       # Don't execute notebooks (options: 'auto', 'force', 'cache', 'off')
nb_execution_allow_errors = True  # Continue building even if there are errors
nb_execution_timeout = 30       # Timeout for cell execution in seconds

# MyST parser settings
myst_heading_anchors = 3        # Generate anchors for headings h1-h3

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    'tum-theme.css',
]


def copy_notebooks_and_generate_index(app):
    """
    Copy notebooks from examples to docs and generate the examples index.
    """
    import os
    import sys
    # Add the source directory to the Python path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from copy_notebooks import copy_notebooks_to_docs, generate_examples_index
    copy_notebooks_to_docs()
    generate_examples_index()

def setup(app):
    # Connect the generate_method_docs function to the builder-inited event.
    # app.connect("builder-inited", generate_method_docs)

    # Connect the copy_notebooks_and_generate_index function to the builder-inited event
    app.connect("builder-inited", copy_notebooks_and_generate_index)
