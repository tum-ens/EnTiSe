{{ method_name }}
=========================

{% if hasattr(cls, 'name') and cls.name %}
**Method Key:** ``{{ cls.name }}``

.. note::
   This is the key required to call this method when using bulk generation with TimeSeriesGenerator.
{% else %}
**Method Key:** ``{{ method_name }}``

.. note::
   This is the class name. For auxiliary methods, the key is determined by the selector class.
{% endif %}

Description
-----------

{{ description }}

Requirements
-------------

Required Keys
~~~~~~~~~~~~~

{% if required_keys %}
.. list-table::
   :widths: auto
   :header-rows: 1

   * - Key
     - Type
   {% for key, dtype in required_keys.items() %}
   * - ``{{ key }}``
     - ``{{ dtype.__name__ }}``
   {% endfor %}
{% else %}
None
{% endif %}


Required Timeseries
~~~~~~~~~~~~~~~~~~~

{% if required_timeseries %}
{% for ts_key, schema in required_timeseries.items() %}
**Timeseries Key:** ``{{ ts_key }}``

{% if 'columns_required' in schema %}
**Required Columns**

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Column
     - Type
   {% for col, col_type in schema['columns_required'].items() %}
   * - ``{{ col }}``
     - ``{{ col_type.__name__ if hasattr(col_type, '__name__') else col_type }}``
   {% endfor %}
{% endif %}

{% if 'cols_optional' in schema %}
**Optional Columns**

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Column
     - Type
   {% for col, col_type in schema['cols_optional'].items() %}
   * - ``{{ col }}``
     - ``{{ col_type.__name__ if hasattr(col_type, '__name__') else col_type }}``
   {% endfor %}
{% endif %}

{% if 'dtype' in schema %}
**DataFrame Type:** ``{{ schema['dtype'].__name__ if hasattr(schema['dtype'], '__name__') else schema['dtype'] }}``
{% endif %}

{% endfor %}
{% else %}
None
{% endif %}



Dependencies
-------------

{% if dependencies %}
- {{ dependencies | join(", ") }}
{% else %}
- None
{% endif %}

Methods
-------

{% for method, info in methods.items() %}
**{{ method }}**:


  .. code-block:: python

     {{ info.source_code | indent(3) }}

{% endfor %}
