.. _methods:

Methods
=======

This section provides details about the available timeseries generation methods in EnTiSe. Each method is tailored to handle specific data types and scenarios.

How to Read Method Pages
-----------------------

Each method page in the documentation follows a consistent structure to help you understand how to use the method effectively:

**Method Name and Key**
   At the top of each page, you'll find the method name (e.g., ``R1C1``, ``FixedSchedule``) and its method key.
   The method key is what you'll use when configuring objects for bulk generation with ``TimeSeriesGenerator``.

**Description**
   A detailed explanation of what the method does, how it works, and when to use it.
   This section may include information about the algorithm, assumptions, and limitations.

**Requirements**
   This section details what inputs the method needs to function:

   * **Required Keys**: Object attributes that must be provided in the object dictionary.
     Each key is listed with its expected data type.

   * **Required Timeseries**: Input timeseries data that must be available in the data dictionary.
     This includes weather data, occupancy data, or other timeseries inputs.

**Dependencies**
   Lists other methods that this method depends on. These dependencies are automatically
   resolved when using bulk generation with ``TimeSeriesGenerator``.

**Methods**
   Shows the source code of the methods defined in the class. This helps you understand
   how the method works internally and how to use it effectively.

.. toctree::
   :maxdepth: 2
   :glob:

   */index
