.. _api_methods:

Methods API
===========

This section provides detailed API documentation for the methods modules of EnTiSe.

Accessing Methods
----------------

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
~~~~~~~~~~~~~~~~~~~~~~~~~

When using direct method access, EnTiSe provides flexible ways to pass parameters:

1. **Using Dictionaries**:

   .. code-block:: python

      # Pass parameters as dictionaries
      obj = {"latitude": 48.1, "longitude": 11.6, "power": 5000}
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

3. **Combining Both Approaches**:

   .. code-block:: python

      # Use dictionaries for most parameters
      obj = {"latitude": 48.1, "longitude": 11.6}
      data = {"weather": weather_df}

      # Override specific parameters with explicit values
      result = pvlib.generate(
          obj=obj, 
          data=data,
          power=5000  # This overrides any "power" value in obj
      )

The method automatically determines whether each parameter belongs in the object 
dictionary or the data dictionary based on the method's defined `required_keys`, 
`optional_keys`, `required_timeseries`, and `optional_timeseries` attributes.

HVAC Methods
-----------

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

FixedSchedule Method
~~~~~~~~~~~~~~~~~~~

.. automodule:: entise.methods.occupancy.FixedSchedule
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
