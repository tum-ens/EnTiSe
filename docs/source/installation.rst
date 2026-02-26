.. _installation:

Installation
============

This page provides instructions on how to install the EnTiSe package on your system.

Supported Environments
----------------------

- **Python Version**: 3.10 to 3.13
- **Operating Systems**: Windows, Linux, macOS

Installation Options
--------------------

This project can be installed in two main ways depending on your needs.

For Users
~~~~~~~~~

Install the latest released version from PyPI using pip:

.. code-block:: bash

   pip install entise

If you prefer ``uv`` for faster installs, you can use:

.. code-block:: bash

   pip install uv
   uv pip install entise

Optional: Verify the installation and version:

.. code-block:: bash

   python -c "import entise; print(entise.__version__)"

For Developers
~~~~~~~~~~~~~~

If you plan to contribute or need the latest development version, install from source in editable mode.

1. Clone the repository and enter the project folder:

   .. code-block:: bash

      git clone https://github.com/tum-ens/entise.git
      cd entise

2. (Recommended) Create and activate a virtual environment:

   - Windows (PowerShell):

     .. code-block:: powershell

        python -m venv .venv
        .venv\Scripts\Activate.ps1

   - macOS/Linux (bash):

     .. code-block:: bash

        python -m venv .venv
        source .venv/bin/activate

3. Install in editable mode with development extras:

   .. code-block:: bash

      pip install -e ".[dev]"

   Using ``uv``:

   .. code-block:: bash

      uv pip install -e ".[dev]"

4. (Optional) Set up development tooling and run tests:

   .. code-block:: bash

      pre-commit install
      pytest -q

Notes
-----
- If you encounter permission issues on Windows, run the terminal as Administrator or enable script execution for virtualenv activation (``Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`` in PowerShell).
- The package supports Python 3.10–3.13 on Windows, Linux, and macOS.
