Usage
=====

To use pyribs in a project::

    import ribs

Config Options
--------------

ribs uses a global config object that is passed to all classes. The default
configuration is stored in :attr:`ribs.config.DEFAULT_CONFIG`, shown below.
Configuration dicts that you pass in are merged with this object. Any values
you provide will override existing values here, including nested ones.

.. literalinclude:: ../ribs/config.py
   :language: python
   :start-after: Default configuration.
   :end-before: End of default configuration
