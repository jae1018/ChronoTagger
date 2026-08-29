"""
Utilities for the labeler package.

This module provides utility functions for ChronoTagger, including:
- Interactive layout builder (build_layout)
- The vertical-stack preset (vertical_stack_config)
- Automatic plot function generation (generate_plot_fn)
- Spectrogram / image panels (draw_spectrogram, attach_colorbar; the
  pure rebin core is chronotagger.labeler.utils.spectrogram.rebin)

The layout builder and plot generator simplify the process of creating
TimeIntervalLabeler instances by providing visual tools and automatic
plot-function generation.

The old `generate_plot_code` / `print_plot_code` pair -- which returned
a bare 36-line function body to stdout, had zero callers, and emitted
`ax.clear()` plus an `axs['labels']` block that raised KeyError -- is
gone. Its job is done properly by
`chronotagger.quickstart.driver_export.generate_driver`, which emits a
whole runnable file rather than a fragment (Pack 7 W1).
"""

from .layout_builder import build_layout
from .plot_generator import (
    generate_plot_fn,
    normalize_time_columns,
    validate_plot_inputs,
    vertical_stack_config,
)
from .spectrogram import attach_colorbar, draw_spectrogram

__all__ = [
    'attach_colorbar',
    'build_layout',
    'draw_spectrogram',
    'generate_plot_fn',
    'normalize_time_columns',
    'validate_plot_inputs',
    'vertical_stack_config',
]
