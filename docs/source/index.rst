.. EnTiSe documentation master file, created by
   sphinx-quickstart on Tue Dec 17 14:37:34 2024.

.. figure:: ../img/logo_TUM.png
   :width: 200px
   :align: right
   :alt: TUM Logo

====================
EnTiSe Documentation
====================

**EnTiSe** (Energy Time Series) is a Python framework for generating synthetic time series data for energy systems research.
It provides a flexible and extensible platform for creating realistic time series for:

* **HVAC** (Heating, Ventilation, and Air Conditioning)
* **DHW** (Drinking Hot-Water)
* **Electricity** consumption and generation (under development)
* **Mobility** patterns and energy demand (under development)
* **Occupancy** profiles for buildings (under development)

EnTiSe is designed to support a wide range of research applications in the energy domain:

* **Building Energy Modeling**: Simulate thermal behavior and energy consumption of buildings
* **Renewable Energy Integration**: Model the variability of renewable energy sources and their impact on energy systems
* **Demand Response**: Analyze the potential for demand-side management and flexibility
* **Energy System Planning**: Support the design and sizing of energy systems with realistic load profiles
* **Policy Analysis**: Evaluate the impact of energy policies on system performance and emissions

The framework can be integrated with other energy modeling tools and workflows, serving as a foundation for comprehensive energy systems analysis.

Key Features
------------

* **Modular Design**: Easily extensible with new methods and strategies
* **Flexible Pipeline**: Automatic dependency resolution between methods
* **Multiple Domains**: Support for HVAC, electricity, mobility, and occupancy
* **Research-Focused**: Designed for energy systems researchers
* **Reproducible**: Generate consistent time series for scientific analysis

Quick Start
-----------

.. code-block:: python

   from entise.core.generator import TimeSeriesGenerator
   import pandas as pd

   # Initialize the generator
   gen = TimeSeriesGenerator()

   # Add objects (e.g., buildings)
   gen.add_objects({
       "id": "building1",
       "hvac": "1R1C",
       "resistance": 2.0,
       "capacitance": 1e5,
       "temp_min": 20.0,
       "temp_max": 24.0,
   })

   # Prepare input data (e.g., weather)
   data = {
       "weather": pd.DataFrame({
           "temp_out": [0.0] * 24,
       }, index=pd.date_range("2025-01-01", periods=24, freq="h"))
   }

   # Generate time series
   summary, df = gen.generate(data)

Documentation Contents
-----------------------

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   getting_started
   workflow
   methods/index
   services/index
   examples
   api/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
