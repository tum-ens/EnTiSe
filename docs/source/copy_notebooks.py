import glob
import os
import shutil
from pathlib import Path


def copy_notebooks_to_docs():
    """
    Copy all Jupyter notebooks from the examples directory to the docs/source/examples directory.
    Preserves the directory structure.
    """
    # Get the absolute path to the repository root
    repo_root = Path(__file__).parent.parent.parent.absolute()
    examples_dir = repo_root / "examples"
    docs_examples_dir = repo_root / "docs" / "source" / "examples"

    # Create the docs/examples directory if it doesn't exist
    os.makedirs(docs_examples_dir, exist_ok=True)

    # Find all Jupyter notebooks in the examples directory
    notebooks = glob.glob(str(examples_dir / "**" / "*.ipynb"), recursive=True)

    print(f"Found {len(notebooks)} notebooks in {examples_dir}")

    # Copy each notebook to the corresponding location in docs/source/examples
    for notebook in notebooks:
        notebook_path = Path(notebook)
        # Get the relative path from the examples directory
        rel_path = notebook_path.relative_to(examples_dir)
        # Construct the destination path
        dest_path = docs_examples_dir / rel_path

        # Create the destination directory if it doesn't exist
        os.makedirs(dest_path.parent, exist_ok=True)

        # Copy the notebook
        shutil.copy2(notebook_path, dest_path)
        print(f"Copied {notebook_path} to {dest_path}")


def generate_examples_index():
    """
    Generate the examples.rst file based on the available notebooks in the docs/source/examples directory.
    """
    # Get the absolute path to the repository root
    repo_root = Path(__file__).parent.parent.parent.absolute()
    docs_examples_dir = repo_root / "docs" / "source" / "examples"
    examples_rst_path = repo_root / "docs" / "source" / "examples.rst"

    # Find all Jupyter notebooks in the docs/source/examples directory
    notebooks = glob.glob(str(docs_examples_dir / "**" / "*.ipynb"), recursive=True)

    # Sort notebooks by directory and then by name
    notebooks.sort()

    # Generate the examples.rst file
    with open(examples_rst_path, "w") as f:
        f.write(".. _examples:\n\n")
        f.write("Examples\n")
        f.write("========\n\n")
        f.write(
            "This section provides examples of how to use EnTiSe for generating timeseries data. Each example demonstrates a\n"
        )
        f.write(
            "different aspect of the library, from basic usage to more complex scenarios. You can find the examples in the\n"
        )
        f.write("``examples`` directory of the repository.\n\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 1\n\n")

        # Add each notebook to the toctree
        for notebook in notebooks:
            notebook_path = Path(notebook)
            # Get the relative path from the docs/source directory
            rel_path = notebook_path.relative_to(repo_root / "docs" / "source")
            # Convert to a path that can be used in the toctree (without the .ipynb extension)
            toctree_path = str(rel_path.with_suffix("")).replace("\\", "/")
            f.write(f"   {toctree_path}\n")

    print(f"Generated examples index at {examples_rst_path}")


if __name__ == "__main__":
    copy_notebooks_to_docs()
    generate_examples_index()
