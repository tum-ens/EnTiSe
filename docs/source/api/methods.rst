.. _api_methods:

Methods API
===========

This section provides detailed API documentation for the methods modules of EnTiSe.

Accessing Methods
------------------

EnTiSe provides two ways to access methods:

1. **Through TimeSeriesGenerator (Batch Processing)**:

   .. code-block:: python

      from entise.core.generator import TimeSeriesGenerator

      # Initialize the generator
      gen = TimeSeriesGenerator()

      # Add objects
      gen.add_objects(objects)

      # Generate timeseries
      summary, df = gen.generate(data)

2. **Direct Import (Individual Processing)**:

   .. code-block:: python

      # Import a specific method
      from entise.methods.pv import PVLib

      # Create an instance
      pvlib = PVLib()

      # Generate timeseries
      result = pvlib.generate(obj, data)

      # Access results
      summary = result["summary"]
      timeseries = result["timeseries"]

Flexible Parameter Passing
~~~~~~~~~~~~~~~~~~~~~~~~~~

When using direct method access, EnTiSe provides flexible ways to pass parameters:

1. **Using Dictionaries**:

   .. code-block:: python

      # Pass parameters as dictionaries
      obj = {"latitude[degree]": 48.1, "longitude[degree]": 11.6, "power[W]": 5000}
      data = {"weather": weather_df}
      result = pvlib.generate(obj=obj, data=data)

2. **Using Named Parameters**:

   .. code-block:: python

      # Pass parameters directly by name
      result = pvlib.generate(
          latitude=48.1,
          longitude=11.6,
          power=5000,
          weather=weather_df
      )

.. _direct_input_names:

Discovering and overriding inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call ``input_names()`` on the method class or instance to list its supported input
names. The returned dictionary maps every accepted name to its full input key.
It includes Python names without units or heights, full keys, and any aliases
from earlier releases. Use ``get_requirements()`` to see which input keys and
tables are required or optional.

.. code-block:: python

   names = PVLib.input_names()
   assert names["power"] == "power[W]"
   assert names["power[W]"] == "power[W]"

Pass a dictionary or a pandas Series row as ``obj``. You can then override
individual inputs with keywords while keeping the rest of the row. In this
example, ``power=5000`` overrides both object entries for power.

.. code-block:: python

   obj = {
       "latitude[degree]": 48.1,
       "longitude[degree]": 11.6,
       "power[W]": 3000,
       "pv:power[W]": 4000,
   }
   result = pvlib.generate(obj=obj, power=5000, weather=weather_df)

A keyword value of ``None`` leaves the corresponding object entry in place.
A table keyword accepts a DataFrame or the name of a table already in
``data``. To use a full key as a keyword, pass it through a dictionary such
as ``**{"power[W]": 5000}``. Unknown names and two non-None keyword values
for the same input raise ``TypeError``.


HVAC Methods
------------

.. automodule:: entise.methods.hvac
   :members:
   :undoc-members:
   :show-inheritance:

R1C1 Method
~~~~~~~~~~~

.. automodule:: entise.methods.hvac.R1C1
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Occupancy Methods
-----------------

.. automodule:: entise.methods.occupancy
   :members:
   :undoc-members:
   :show-inheritance:


Auxiliary Methods
-----------------

Internal Gains
~~~~~~~~~~~~~~

.. automodule:: entise.methods.auxiliary.internal.strategies
   :members:
   :undoc-members:
   :show-inheritance:

Solar Gains
~~~~~~~~~~~

.. automodule:: entise.methods.auxiliary.solar.strategies
   :members:
   :undoc-members:
   :show-inheritance:

Selectors
~~~~~~~~~

.. automodule:: entise.methods.auxiliary.internal.selector
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: entise.methods.auxiliary.solar.selector
   :members:
   :undoc-members:
   :show-inheritance:
