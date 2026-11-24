.. _installation:

Installation
============

This page provides instructions on how to install the EnTiSe package on your system.

Supported Environments
----------------------

- **Python Version**: 3.11 to 3.14
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

.. _third-party-licenses:

Third-party licenses
--------------------

EnTiSe's own code is distributed under the `MIT License
<https://github.com/tum-ens/entise/blob/main/LICENSE>`_. Pip installs the
dependencies for the core and any selected extras as separate packages.
For the terms of a specific release, consult the license files included
in that package. Upstream references for direct dependencies are listed
below.

- The core depends on `joblib <https://github.com/joblib/joblib/blob/main/LICENSE.txt>`_,
  `NumPy <https://github.com/numpy/numpy/blob/main/LICENSE.txt>`_,
  `pandas <https://github.com/pandas-dev/pandas/blob/main/LICENSE>`_ and
  `tqdm <https://github.com/tqdm/tqdm/blob/master/LICENCE>`_.
- Methods and acceleration use `demandlib <https://github.com/oemof/oemof-demand/blob/dev/LICENSE>`_,
  `holidays <https://github.com/vacanza/holidays/blob/dev/LICENSE>`_,
  `Numba <https://github.com/numba/numba/blob/main/LICENSE>`_,
  `pvlib <https://github.com/pvlib/pvlib-python/blob/main/LICENSE>`_,
  `PyLPG <https://github.com/FZJ-IEK3-VSA/pylpg/blob/main/LICENSE>`_,
  `SciPy <https://github.com/scipy/scipy/blob/main/LICENSE.txt>`_ and
  `windpowerlib <https://github.com/wind-python/windpowerlib/blob/dev/LICENSE>`_.
  For the separately installed backend, see
  `DistrictHeatingSim's license <https://github.com/JonasPfeiffer123/DistrictHeatingSim/blob/main/LICENSE>`_.
- The weather client uses `openmeteo-requests <https://github.com/open-meteo/python-requests/blob/main/LICENSE>`_,
  `Pint <https://github.com/hgrecco/pint/blob/master/LICENSE>`_,
  `requests-cache <https://github.com/requests-cache/requests-cache/blob/main/LICENSE>`_ and
  `retry-requests <https://github.com/bustawin/retry-requests/blob/master/LICENSE>`_.
- Examples use `Matplotlib <https://matplotlib.org/stable/project/license.html>`_ and
  `Jupyter Notebook <https://github.com/jupyter/notebook/blob/main/LICENSE>`_.
- Development and documentation use `mypy <https://github.com/python/mypy/blob/master/LICENSE>`_,
  `MyST-NB <https://github.com/executablebooks/MyST-NB/blob/main/LICENSE>`_,
  `pre-commit <https://github.com/pre-commit/pre-commit/blob/main/LICENSE>`_,
  `pytest <https://github.com/pytest-dev/pytest/blob/main/LICENSE>`_,
  `pytest-cov <https://github.com/pytest-dev/pytest-cov/blob/master/LICENSE>`_,
  `pytest-mock <https://github.com/pytest-dev/pytest-mock/blob/main/LICENSE>`_,
  `Ruff <https://github.com/astral-sh/ruff/blob/main/LICENSE>`_,
  `Sphinx <https://github.com/sphinx-doc/sphinx/blob/master/LICENSE.rst>`_,
  `sphinxcontrib-mermaid <https://github.com/mgaitan/sphinxcontrib-mermaid/blob/master/LICENSE>`_,
  `sphinx-rtd-theme <https://github.com/readthedocs/sphinx_rtd_theme/blob/master/LICENSE>`_,
  `tox <https://github.com/tox-dev/tox/blob/main/LICENSE>`_ and
  uv (`MIT terms <https://github.com/astral-sh/uv/blob/main/LICENSE-MIT>`_ or
  `Apache 2.0 <https://github.com/astral-sh/uv/blob/main/LICENSE-APACHE>`_).

On first use, PyLPG can download the LoadProfileGenerator native
executable and its database. For the executable's source license, see the
`LoadProfileGenerator MIT license
<https://github.com/FZJ-IEK3-VSA/LoadProfileGenerator/blob/master/LICENSE>`_.
For the download and accompanying resources, see the
`LoadProfileGenerator download terms
<https://loadprofilegenerator.de/download/>`_
and the notices for your LPG release.
See :ref:`open-meteo-terms` for the hosted API and its data, and
:ref:`example-weather-attribution` for the sample weather in the repository.

Notes
-----
- If you encounter permission issues on Windows, run the terminal as Administrator or enable script execution for virtualenv activation (``Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`` in PowerShell).
- The package supports Python 3.11–3.14 on Windows, Linux, and macOS.
