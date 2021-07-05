{#
  Adapted from
  https://github.com/sphinx-doc/sphinx/blob/v3.2.1/sphinx/ext/autosummary/templates/autosummary/class.rst
  and
  https://stackoverflow.com/questions/62142793/autosummary-generated-documentation-missing-all-dunder-methods-except-for-ini
#}

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
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
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
