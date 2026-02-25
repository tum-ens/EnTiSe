{{ method_name }}
{{ '=' * (method_name|length) }}

Overview
--------

{{ description }}

Key facts
---------

- Method key: ``{{ method_key }}``
{% if supported_types %}
- Supported types:

  {% for t in supported_types %}
  - ``{{ t }}``
  {% endfor %}
{% endif %}

Requirements
------------

Required keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

{% if required_keys %}
{% for k in required_keys %}- ``{{ k }}``
{% endfor %}
{% else %}
- None
{% endif %}

Optional keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

{% if optional_keys %}
{% for k in optional_keys %}- ``{{ k }}``
{% endfor %}
{% else %}
- None
{% endif %}

Required data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

{% if required_data %}
{% for k in required_data %}- ``{{ k }}``
{% endfor %}
{% else %}
- None
{% endif %}

Optional data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

{% if optional_data %}
{% for k in optional_data %}- ``{{ k }}``
{% endfor %}
{% else %}
- None
{% endif %}

Outputs
-------

Summary metrics
~~~~~~~~~~~~~~~

{% if output_summary %}
.. list-table::
   :widths: auto
   :header-rows: 1

   * - Key
     - Description
   {% for k, v in output_summary.items() %}
   * - ``{{ k }}``
     - {{ v }}
   {% endfor %}
{% else %}
- None
{% endif %}

Timeseries columns
~~~~~~~~~~~~~~~~~~

{% if output_timeseries %}
.. list-table::
   :widths: auto
   :header-rows: 1

   * - Column
     - Description
   {% for k, v in output_timeseries.items() %}
   * - ``{{ k }}``
     - {{ v }}
   {% endfor %}
{% else %}
- None
{% endif %}

Public methods
--------------

{% for method, info in methods.items() %}
- {{ method }}

  .. code-block:: python

     {{ info.source_code | indent(3) }}
{% endfor %}
