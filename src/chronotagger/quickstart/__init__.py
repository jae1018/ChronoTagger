"""
ChronoTagger Quick-Start Module

Provides a GUI wizard for loading data and configuring plots
without requiring users to write Python code.
"""

# Pack 6 D8: 'config' MUST go with config.py. A star-import resolves
# every name in __all__ as a submodule attribute, so leaving it here
# makes `from chronotagger.quickstart import *` raise AttributeError --
# proved on a two-package minimal reproduction, not assumed. Both names
# below ARE submodules of this package, which is what makes them legal
# entries.
#
# Pack 8 R19: driver_export joins the public surface. It stopped being
# an internal detail of the wizard the moment the wizard began writing
# files with it -- a user who wants a driver without walking the GUI
# calls generate_driver directly, and that is a supported use.
__all__ = ['driver_export', 'wizard']
