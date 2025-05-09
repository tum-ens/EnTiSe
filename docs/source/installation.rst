.. _installation:

Installation
============

This page provides instructions on how to install the EnTiSe package on your system. EnTiSe is a timeseries generation framework for various data types like HVAC, Electricity, Mobility, and Occupancy.

Supported Environments
----------------------
- **Python Version**: 3.8 or higher
- **Operating Systems**: Windows, Linux, macOS

Installing EnTiSe
-----------------

EnTiSe can be installed using **pip** from PyPI or directly from the source.

Install from PyPI
~~~~~~~~~~~~~~~~~

To install the latest stable release of EnTiSe from the Python Package Index (PyPI), run:

.. code-block:: bash

    pip install entise

This will download and install all required dependencies automatically.

Install from Source
~~~~~~~~~~~~~~~~~~~

To install EnTiSe directly from the source code, follow these steps:

1. Clone the Git repository:

   .. code-block:: bash

       git clone https://github.com/tum-ens/entise.git

2. Navigate to the project directory:

   .. code-block:: bash

       cd entise

3. Install the package using pip:

   .. code-block:: bash

       pip install .

This will install EnTiSe and its dependencies.

Upgrading EnTiSe
----------------

To upgrade to the latest version of EnTiSe, use the following command:

.. code-block:: bash

    pip install --upgrade entise

Verifying the Installation
--------------------------

To verify that EnTiSe has been installed correctly, run the following command in a Python interpreter:

.. code-block:: python

    >>> import entise
    >>> print(entise.__version__)

You should see the installed version of EnTiSe printed on the console.

Optional Dependencies
---------------------

EnTiSe supports optional visualization tools. To include these during installation, use:

.. code-block:: bash

    pip install entise[visualization]

This will install additional packages like **matplotlib** and **networkx** for generating dependency graphs and visualizations.

Uninstallation
--------------

If you need to uninstall EnTiSe, run:

.. code-block:: bash

    pip uninstall entise

Troubleshooting
---------------

- **Problem**: ``pip`` command not found.
   - **Solution**: Make sure Python and pip are added to your system's PATH.

- **Problem**: Permission denied when installing.
   - **Solution**: Use ``--user`` flag to install locally:

     .. code-block:: bash

         pip install --user entise

- **Problem**: Missing dependencies.
   - **Solution**: Run the following command to ensure all dependencies are installed:

     .. code-block:: bash

         pip install -r requirements.txt
