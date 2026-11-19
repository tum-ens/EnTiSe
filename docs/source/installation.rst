.. _installation:

Installation
============

This page provides instructions on how to install the EnTiSe package on your system.

Supported Environments
----------------------

- **Python Version**: 3.10 to 3.13
- **pandas**: 2.2.2 or newer, below 4, so pandas 2 and pandas 3 both work. pandas 3 needs Python 3.11 or newer, so Python 3.10 uses pandas 2.
- **Operating Systems**: Windows, Linux, macOS

Installation Options
--------------------

This project can be installed in two main ways depending on your needs.

For Users
~~~~~~~~~

Install the latest released version from PyPI using pip:

.. code-block:: bash

   pip install entise

The core includes the file loader, Ruhnau heat pump, and GeoMA and PHT
occupancy methods. Add the extras for any other methods you run:

.. code-block:: bash

   pip install "entise[pv-pvlib,hvac-1r1c]"

Each method's reference page gives its install command. A type group
installs all packaged methods of that type. For example, ``entise[hvac]``
installs the four RC methods. The other groups are ``cooling``, ``dhw``,
``electricity``, ``heating``, ``hp``, ``mobility``, ``occupancy``, ``pv``,
and ``wind``. Cooling and mobility currently provide the file loader.

.. list-table::
   :header-rows: 1

   * - Extra
     - Purpose
   * - ``all``
     - All packaged methods, the weather service, and example tools.
   * - ``weather``
     - The weather service and Open-Meteo provider.
   * - ``examples``
     - Matplotlib and Jupyter Notebook. Add the extras for the methods you run.
   * - ``numba``
     - The optional 1R1C accelerator. Install with ``hvac-1r1c`` or ``hvac``.

For the previous installation coverage, use ``pip install "entise[all]"``.
To run the PyLPG notebook, install ``entise[examples,electricity-pylpg]`` and
start it with ``python -m notebook``.

DistrictHeatingSim requires a separate manual installation with Python 3.11.
Follow the `DistrictHeatingSim installation instructions
<https://github.com/JonasPfeiffer123/DistrictHeatingSim>`_ to install its
backend alongside EnTiSe.

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

3. Install in editable mode with development extras. This includes the
   packaged method and weather dependencies needed by the test suite:

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
