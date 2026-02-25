
.. figure:: ../img/logo.png
   :width: 200px
   :align: right
   :alt: EnTiSe Logo

====================
EnTiSe Documentation
====================

**EnTiSe** (Energy Time Series) is a Python framework for generating synthetic time series data for energy systems research.
It provides a flexible and extensible platform for creating realistic time series for:

* **Drinking Hot-Water**
* **Electricity**
* **Heat Pumps**
* **HVAC**
* **Occupancy**
* **PV**
* **Wind**

EnTiSe is designed to support a wide range of research applications in the energy domain:

* **Building Energy Modeling**: Simulate thermal behavior and energy consumption of buildings
* **Renewable Energy Integration**: Model the variability of renewable energy sources and their impact on energy systems
* **Demand Response**: Analyze the potential for demand-side management and flexibility
* **Energy System Planning**: Support the design and sizing of energy systems with realistic load profiles

The framework can be integrated with other energy modeling tools and workflows, serving as a foundation for comprehensive energy systems analysis.

Key Features
------------

* **Modular Design**: Easily extensible with new methods and strategies independent of existing methods
* **Flexible Pipeline**: Automatic dependency resolution between methods
* **Multiple Domains**: Support for HVAC, electricity, and more

Quick Start
-----------

For those wanting to quickly get started with EnTiSe, here is a simple example of how to use the `Generator` to create synthetic time series data for a building's thermal behavior. We recommend having a look at the :ref:`examples` to get a better understanding of the available parameters and methods.

.. code-block:: python

   from entise import Generator
   import pandas as pd

   # Initialize the generator
   gen = Generator()

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



.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   getting_started
   workflow
   architecture
   methods/index
   services/index
   examples
   api/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
