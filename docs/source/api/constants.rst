.. _api_constants:

Constants API
=============

``Types``, ``Objects``, ``Columns`` and ``Keys`` group read-only string
constants. Iterate over a namespace to get its values in declaration order.
Use ``in`` to check a value, ``len()`` to count the constants, and ``items()``
to get name-value pairs in declaration order.

.. code-block:: python

   from entise.constants import Types

   assert Types.DHW == "dhw"
   assert "dhw" in Types
   assert dict(Types.items())["DHW"] == "dhw"

These namespaces are used directly as classes.

Types
-----

.. automodule:: entise.constants.ts_types
   :members:
   :undoc-members:
   :show-inheritance:

Objects
-------

.. automodule:: entise.constants.objects
   :members:
   :undoc-members:
   :show-inheritance:

Columns
-------

.. automodule:: entise.constants.columns
   :members:
   :undoc-members:
   :show-inheritance:

Keys
----

.. automodule:: entise.constants.general
   :members:
   :undoc-members:
   :show-inheritance:
