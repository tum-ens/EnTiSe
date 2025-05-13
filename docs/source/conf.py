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

# AutoAPI configuration (disabled for now)
# autoapi_dirs = ['../../entise']  # Path to your code
# autoapi_root = "autoapi"
# autoapi_add_toctree_entry = True  # Add entries to the toctree
# autoapi_generate_api_docs = True  # Automatically generate API docs

# autoapi_keep_files = False  # Cleanup intermediate .rst files after generation
# autoapi_options = [
#     "members",              # Include class and module members
#     "undoc-members",        # Include undocumented members
#     "show-inheritance",     # Show inheritance diagrams
#     "show-module-summary",  # Show a summary for each module
#     "special-members",      # Include special methods like __init__
# ]

# autoapi_member_order = "bysource"  # Order members by their appearance in the source

# Add methods docs through automated script (disabled for now)
# def generate_method_docs(app):
#     """
#     Generate documentation for all methods before the Sphinx build starts.
#     """
#     # Import the generation functions from your module
#     from docs.source.generate_methods_docs import generate_docs_for_all_methods, generate_indexes, discover_method_classes
#
#     # Automatically discover all method classes in the package.
#     method_classes = discover_method_classes("entise.methods")
#
#     # Determine the absolute paths for the template and output directories.
#     template_path = os.path.join(app.confdir, "_templates", "method.rst")
#     base_output_dir = os.path.join(app.confdir, "methods")
#
#     # Generate documentation files for each method in their corresponding folders.
#     generate_docs_for_all_methods(method_classes, template_path, base_output_dir)
#
#     # Generate index files for each subfolder in the methods directory.
#     generate_indexes(base_output_dir)


def setup(app):
    # Connect the generate_method_docs function to the builder-inited event.
    # app.connect("builder-inited", generate_method_docs)
    pass
