"""
Utilities for the labeler package.

This module provides utility functions for ChronoTagger, including:
- Interactive layout builder (build_layout)
- Automatic plot function generation (generate_plot_fn)
- Plot code generation helpers

The layout builder and plot generator simplify the process of creating
TimeIntervalLabeler instances by providing visual tools and automatic
code generation.
"""

from .layout_builder import build_layout
from .plot_generator import generate_plot_fn, generate_plot_code, print_plot_code

__all__ = [
    'build_layout',
    'generate_plot_fn',
    'generate_plot_code',
    'print_plot_code',
]
