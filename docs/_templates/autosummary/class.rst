{#
  Adapted from
  https://github.com/sphinx-doc/sphinx/blob/v3.2.1/sphinx/ext/autosummary/templates/autosummary/class.rst
  and
  https://stackoverflow.com/questions/62142793/autosummary-generated-documentation-missing-all-dunder-methods-except-for-ini
#}

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   {%- if name == "ArchiveDataFrame" %}
   :no-inherited-members:
   :members:
   {%- elif name == "DiscountArchive" %}
   :no-inherited-members:
   :members: add, init_discount_model, train_discount_model, empty, discount_model_manager, dtypes, empty_points, init_train_points, learning_rate, threshold_min
   {%- elif name == "DensityArchive" %}
   :no-inherited-members:
   :members: add, compute_density, buffer, empty, measure_dim, objective_dim, solution_dim
   {% endif %}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% if name == "ArchiveDataFrame" %}
       ~{{ name }}.get_field
       ~{{ name }}.iterelites
   {% elif name == "DiscountArchive" %}
       ~{{ name }}.add
       ~{{ name }}.init_discount_model
       ~{{ name }}.train_discount_model
   {% elif name == "DensityArchive" %}
       ~{{ name }}.add
       ~{{ name }}.compute_density
   {% else %}
     {% for item in all_methods %}
       {%- if not item.startswith('_') or item in ['__len__',
                                                   '__call__',
                                                   '__next__',
                                                   '__iter__',
                                                   '__getitem__',
                                                   '__setitem__',
                                                   '__delitem__',
                                                   ] %}
       ~{{ name }}.{{ item }}
       {%- endif -%}
     {%- endfor %}
   {% endif %}

   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% if name == "ArchiveDataFrame" %}
   {% elif name == "DiscountArchive" %}
       ~{{ name }}.discount_model_manager
       ~{{ name }}.dtypes
       ~{{ name }}.empty
       ~{{ name }}.empty_points
       ~{{ name }}.init_train_points
       ~{{ name }}.learning_rate
       ~{{ name }}.threshold_min
   {% elif name == "DensityArchive" %}
       ~{{ name }}.buffer
       ~{{ name }}.empty
       ~{{ name }}.measure_dim
       ~{{ name }}.objective_dim
       ~{{ name }}.solution_dim
   {% else %}
     {% for item in attributes %}
       ~{{ name }}.{{ item }}
     {%- endfor %}
   {% endif %}

   {% endif %}
   {% endblock %}
