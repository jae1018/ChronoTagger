"""
ChronoTagger Quick-Start Module

Provides a GUI wizard for loading data and configuring plots
without requiring users to write Python code.
"""

# Pack 6 D8: 'config' MUST go with config.py. A star-import resolves
# every name in __all__ as a submodule attribute, so leaving it here
# makes `from chronotagger.quickstart import *` raise AttributeError --
# proved on a two-package minimal reproduction, not assumed.
__all__ = ['wizard']
