# Configuration

Each ribs class has a corresponding config class named `*Config`; for
instance, [GridArchive](ribs.archives.GridArchive) has
[GridArchiveConfig](ribs.archives.GridArchiveConfig). These config classes
contain all the config options for their class as well as default values for
them. When creating each class, you have a few options for configuration:

1. Pass in no config (i.e. `None`), which will give the class a default value
   for its config. In the example below, the config of `GridArchive` will be
   `GridArchiveConfig()`:

   ```python
   from ribs.archives import GridArchive

   archive = GridArchive(...)
   ```

2. Pass in an instance of the config class. You can specify as many or as few
   parameters as you like; the rest have default values. For instance:

   ```python
   from ribs.archives import GridArchive, GridArchiveConfig

   config = GridArchiveConfig(seed=42)
   archive = GridArchive(..., config)
   ```

3. Pass in a dict with the same options as the config class. This dict's
   options are used to create the config class; unspecified options are left as
   defaults. This is equivalent to option 2), but it may be more convenient.
   The following example does the same as above:

   ```python
   from ribs.archives import GridArchive
   config = {"seed": 42}
   archive = GridArchive(..., config)
   ```
