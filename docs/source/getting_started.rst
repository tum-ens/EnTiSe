.. _getting_started:

Getting Started
===============

Install EnTiSe with the HVAC and PV extras used below. See the
:ref:`installation` guide for supported Python versions and other extras.

.. code-block:: bash

    pip install "entise[hvac-1r1c,pv-pvlib]"

Batch Processing
----------------

Define the building and its method in an object dictionary. The
``weather`` field names a table in the data dictionary. The input names
include units where applicable. The weather table has a timezone-aware
hourly index in UTC. In Python, you can also use the corresponding
constants from ``entise.constants``, such as ``Columns.TEMP_AIR``.

.. include:: ../../README.rst
   :start-after: .. quickstart-batch-start
   :end-before: .. quickstart-batch-end

``summary`` contains one row per object. ``series`` is keyed by object ID
and time series type. In this example, ``hvac`` holds the 24 hourly rows
for ``building1``. To run a batch, pass a list of object dictionaries or a
DataFrame to ``add_objects``.

Direct Method Access
--------------------

Call ``PVLib`` directly when you want to integrate a single method into
your own code. This example supplies synthetic irradiance, temperature
and wind for four midday hours. The result contains a summary dictionary
and a DataFrame of hourly PV power.

.. include:: ../../README.rst
   :start-after: .. quickstart-direct-start
   :end-before: .. quickstart-direct-end

Understanding the Workflow
--------------------------

- Define objects and provide input data (e.g., weather).
- EnTiSe selects methods and strategies based on your inputs.
- A dependency resolver ensures correct execution order.
- You get a summary and a dictionary of time series DataFrames.

See :ref:`workflow` for the full conceptual guide, decision charts, and troubleshooting.

Available Methods
-----------------

For a list of available methods, refer to the :ref:`methods` section in this documentation.
