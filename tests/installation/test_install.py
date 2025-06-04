"""Test script to verify package installation and import."""

import os
import shutil
import subprocess
import sys
import tempfile


def test_install():
    """Test that the package can be installed and imported."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a virtual environment
        venv_dir = os.path.join(temp_dir, "venv")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])

        # Get the path to the Python executable in the virtual environment
        if os.name == "nt":  # Windows
            python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
        else:  # Unix/Linux/Mac
            python_exe = os.path.join(venv_dir, "bin", "python")

        # Install uv
        subprocess.check_call([python_exe, "-m", "pip", "install", "uv"])

        # Install the package in development mode
        subprocess.check_call([python_exe, "-m", "uv", "pip", "install", "-e", "."])

        # Test importing the package and accessing the version
        result = subprocess.check_output(
            [python_exe, "-c", "import entise; print(entise.__version__)"],
            universal_newlines=True,
        )

        print(f"Package version: {result.strip()}")
        assert result.strip() != "unknown"

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_install()
    print("Installation test passed!")
