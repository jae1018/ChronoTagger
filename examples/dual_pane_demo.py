"""
Simple Dual-Pane Example

Demonstrates basic multi-pane functionality with two views:
- Pane 1: Sinusoidal waves (sin/cos)
- Pane 2: Random walk + its rolling spread

This is the minimal example to get started with multi-pane labeling.
For a more comprehensive example with real scientific data, see:
  examples/multi_pane_magnetosphere.py
For what makes a plot_fn cheap to redraw, see:
  docs/fast-plot-fn.md

Usage:
    python examples/dual_pane_demo.py

Keyboard shortcuts:
    Ctrl+Tab        - Switch to next pane
    Ctrl+1 / Ctrl+2 - Jump directly to pane 1 or 2
    Right-click tab - Rename or refresh pane
    F1              - Show all shortcuts
"""

import pandas as pd
import numpy as np
from chronotagger.labeler import TimeIntervalLabeler

# Create test data
dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
df = pd.DataFrame({
    'value1': np.sin(np.linspace(0, 10*np.pi, 1000)) + 0.1*np.random.randn(1000),
    'value2': np.cos(np.linspace(0, 10*np.pi, 1000)) + 0.1*np.random.randn(1000),
    'value3': np.random.randn(1000).cumsum(),
}, index=dates)

def plot_fn_tab1(axs, df_slice, t0, t1):
    """Plot for Tab 1 - Sinusoidal"""
    axs['top'].plot(df_slice.index, df_slice['value1'], 'b-')
    axs['top'].set_ylabel('Sin Wave')
    axs['top'].grid(True, alpha=0.3)

    axs['bottom'].plot(df_slice.index, df_slice['value2'], 'r-')
    axs['bottom'].set_ylabel('Cos Wave')
    axs['bottom'].grid(True, alpha=0.3)

def plot_fn_tab2(axs, df_slice, t0, t1):
    """Plot for Tab 2 - Random Walk and its rolling spread.

    Both areas below are role='time', so both panels are on the window's
    date axis and both must be drawn against df_slice.index.  See
    docs/fast-plot-fn.md.
    """
    axs['top'].plot(df_slice.index, df_slice['value3'], 'g-')
    axs['top'].set_ylabel('Random Walk')
    axs['top'].grid(True, alpha=0.3)

    # Derived here on purpose: a VECTORIZED statistic over the handed-in
    # window is free next to the redraw (rule 2).  One artist, not one
    # per bin (rule 1).  A histogram would have been neither -- 20
    # Rectangle patches, re-binned every frame, and drawn against VALUES
    # on an axis this panel's role has already pinned to [t0, t1].
    spread = df_slice['value3'].rolling(31, center=True, min_periods=2).std()
    axs['bottom'].plot(df_slice.index, spread, color='purple')
    axs['bottom'].set_ylabel('Rolling spread')
    axs['bottom'].grid(True, alpha=0.3)

# Layout specification
layout_spec = {
    'nrows': 3,
    'ncols': 1,
    'hspace': 0.05,
    'areas': [
        {'key': 'top', 'row': 0, 'col': 0, 'role': 'time'},
        {'key': 'bottom', 'row': 1, 'col': 0, 'role': 'time'},
        {'key': 'labels', 'row': 2, 'col': 0, 'role': 'labels'},
    ],
}

# Create panes
panes = [
    {'title': 'Sinusoidal', 'plot_fn': plot_fn_tab1, 'layout_spec': layout_spec},
    {'title': 'Random Walk', 'plot_fn': plot_fn_tab2, 'layout_spec': layout_spec},
]

# Create labeler with multi-pane mode
print('Creating multi-pane labeler...')
print('='*70)
print('MULTI-PANE DEMO')
print('='*70)

labeler = TimeIntervalLabeler(
    df=df,
    panes=panes,
    window=pd.Timedelta('2h'),
    step=pd.Timedelta('30min'),
    classes=['Normal', 'Anomaly', 'Unknown'],
)

print(f'Multi-pane mode: {labeler.multi_pane_mode}')
print(f'Number of panes: {len(labeler.panes)}')
print(f'Pane titles: {[p.title for p in labeler.panes]}')
print()
print('Multi-pane shortcuts:')
print('  - Ctrl+Tab / Ctrl+Shift+Tab - Switch tabs')
print('  - Ctrl+1 / Ctrl+2 - Jump to specific tab')
print('  - Right-click tab - Rename/refresh')
print('  - F1 - Show all shortcuts')
print('='*70)
print('Launching GUI...')

labeler.run()
