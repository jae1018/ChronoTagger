#!/usr/bin/env python3
"""
Simple Example: Load CSV → Layout Wizard → Labeler

This is the minimal example showing how to use the layout wizard
with your own CSV data.

Usage:
    python examples/layout_wizard_simple.py path/to/your/data.csv
    
Or run with sample data:
    python examples/layout_wizard_simple.py
"""

import sys
import pandas as pd
import numpy as np

from chronotagger.labeler import TimeIntervalLabeler
from chronotagger.labeler.utils import build_layout, generate_plot_fn


def load_data(csv_path: str = None) -> pd.DataFrame:
    """
    Load data from CSV or create sample data.
    
    Args:
        csv_path: Path to CSV file (optional)
    
    Returns:
        DataFrame with datetime index
    """
    if csv_path:
        print(f"📂 Loading data from: {csv_path}")
        # Load CSV - assumes first column is datetime
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    else:
        print("📊 Creating sample data...")
        # Create simple sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        df = pd.DataFrame({
            'Temperature': 20 + 5 * np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100),
            'Humidity': 60 + 10 * np.cos(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100),
            'Pressure': 1013 + np.random.randn(100)
        }, index=dates)
        print(f"✓ Created sample data with {len(df)} rows")
    
    return df


def main():
    """Main function."""
    # Get CSV path from command line (optional)
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Load data
    df = load_data(csv_path)
    print()
    
    # Launch layout builder
    print("🎨 Opening Layout Builder...")
    print("   Build your layout visually, then click 'Done'")
    print()
    
    layout_spec, plot_config = build_layout(df)
    
    # Check if user cancelled
    if layout_spec is None:
        print("Cancelled by user.")
        return
    
    print("✓ Layout created!")
    print()
    
    # Generate plot function and launch labeler
    print("🚀 Starting labeler...")
    plot_fn = generate_plot_fn(plot_config)
    
    app = TimeIntervalLabeler(
        df=df,
        plot_fn=plot_fn,
        layout_spec=layout_spec
    )
    
    app.run()
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
