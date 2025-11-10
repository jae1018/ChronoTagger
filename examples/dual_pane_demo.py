"""Test multi-pane mode with two tabs."""

import pandas as pd
import numpy as np
from src.chronotagger.labeler import TimeIntervalLabeler

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
    """Plot for Tab 2 - Random Walk"""
    axs['top'].plot(df_slice.index, df_slice['value3'], 'g-')
    axs['top'].set_ylabel('Random Walk')
    axs['top'].grid(True, alpha=0.3)

    axs['bottom'].hist(df_slice['value3'], bins=20, alpha=0.7, color='purple')
    axs['bottom'].set_xlabel('Value')
    axs['bottom'].set_ylabel('Count')

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
print('Launching GUI...')

labeler.run()
