# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))  # Adjust depending on your folder structure

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "EnTiSe"
copyright = "2026, Markus Doepfert, Hussein Geneva"
author = "Markus Doepfert, Hussein Geneva"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Automatically document code
    "sphinx.ext.napoleon",  # Support Google/NumPy-style docstrings
    "sphinx.ext.viewcode",  # Add links to source code
    "myst_nb",  # Support Jupyter Notebooks with MyST-NB
    "sphinxcontrib.mermaid",  # Support Mermaid diagrams
    # "autoapi"                 # Generate docs directly from modules (disabled for now)
]

# MyST-NB configuration
myst_enable_extensions = [
    "amsmath",  # For math equations
    "colon_fence",  # For code blocks with colons
    "deflist",  # For definition lists
    "dollarmath",  # For inline math with $
    "html_image",  # For HTML images
    "html_admonition",  # For HTML admonitions
    "tasklist",  # For task lists
]

# Notebook execution settings
nb_execution_mode = "off"  # Don't execute notebooks (options: 'auto', 'force', 'cache', 'off')
nb_execution_allow_errors = True  # Continue building even if there are errors
nb_execution_timeout = 30  # Timeout for cell execution in seconds

# MyST parser settings
myst_heading_anchors = 3  # Generate anchors for headings h1-h3

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "tum-theme.css",
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


def generate_method_docs(app):
    """
    Discover EnTiSe methods and (re)generate the method reference pages before the build starts.
    Additionally, clean the docs/source/methods directory (except the top-level index.rst)
    to ensure no stale files linger from previous builds.
    """
    import os
    import shutil
    import sys

    # Ensure we can import the generator script and the package
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

    try:
        from generate_methods_docs import (
            discover_method_classes,
            generate_docs_for_all_methods,
            generate_indexes,
        )
    except Exception as e:
        print(f"[docs] Skipping method docs generation (import error): {e}")
        return

    try:
        template_path = os.path.join(os.path.dirname(__file__), "_templates", "method.rst")
        base_output_dir = os.path.join(os.path.dirname(__file__), "methods")

        # 1) Ensure base directory exists
        os.makedirs(base_output_dir, exist_ok=True)

        # 1a) Bootstrap a minimal, valid folder tree so Sphinx has targets even if no pages are generated
        try:
            from generate_methods_docs import bootstrap_methods_tree

            bootstrap_methods_tree(base_output_dir)
        except Exception as bs_err:
            print(f"[docs] Warning during methods tree bootstrap: {bs_err}")

        # 2) Cleanup existing generated method docs to avoid stale files
        try:
            for entry in os.listdir(base_output_dir):
                path = os.path.join(base_output_dir, entry)
                # Preserve the top-level landing page 'index.rst'
                if os.path.isfile(path) and entry.lower() == "index.rst":
                    continue
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                elif os.path.isfile(path) and entry.endswith(".rst"):
                    try:
                        os.remove(path)
                    except OSError as rm_err:
                        print(f"[docs] Warning: could not remove file {path}: {rm_err}")
        except Exception as cleanup_err:
            # Never fail the build due to cleanup issues; just log
            print(f"[docs] Warning during method docs cleanup: {cleanup_err}")

        # 2a) Re-bootstrap after cleanup to recreate base indexes and folders if they were removed
        try:
            from generate_methods_docs import bootstrap_methods_tree

            bootstrap_methods_tree(base_output_dir)
        except Exception as bs2_err:
            print(f"[docs] Warning during post-clean bootstrap: {bs2_err}")

        # 3) Discover and regenerate all method docs
        method_classes = discover_method_classes("entise.methods")
        generate_docs_for_all_methods(method_classes, template_path, base_output_dir)
        generate_indexes(base_output_dir)
        print(f"[docs] Generated method docs for {len(method_classes)} classes.")
    except Exception as e:
        # Do not fail the docs build if generation has issues; just log.
        print(f"[docs] Error during method docs generation: {e}")


def setup(app):
    # Connect the generate_method_docs function to the builder-inited event.
    app.connect("builder-inited", generate_method_docs)

    # Connect the copy_notebooks_and_generate_index function to the builder-inited event
    app.connect("builder-inited", copy_notebooks_and_generate_index)
