#!/usr/bin/env python3
"""
Example Driver: Interactive Layout Builder → TimeIntervalLabeler

This script demonstrates the complete workflow:
1. Load sample data
2. Launch interactive layout builder (visual grid editor)
3. Generate plot function from user's layout choices
4. Start TimeIntervalLabeler with the custom layout

Usage:
    python examples/layout_wizard_demo.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import ChronoTagger components
from chronotagger.labeler import TimeIntervalLabeler
from chronotagger.labeler.utils import build_layout, generate_plot_fn


def create_sample_data() -> pd.DataFrame:
    """
    Create sample time-series data for demonstration.
    
    Returns:
        DataFrame with datetime index and multiple numeric columns
    """
    print("📊 Creating sample data...")
    
    # Create datetime index (1 hour intervals for 7 days)
    dates = pd.date_range('2024-01-01', periods=168, freq='h')
    
    # Generate sample data with interesting patterns
    np.random.seed(42)
    
    # Column A: Sinusoidal pattern with noise
    A = 10 + 5 * np.sin(np.linspace(0, 4*np.pi, 168)) + np.random.normal(0, 0.5, 168)
    
    # Column B: Linear trend with noise
    B = np.linspace(0, 20, 168) + np.random.normal(0, 1, 168)
    
    # Column C: Step function with noise
    C = np.where(np.arange(168) < 84, 5, 15) + np.random.normal(0, 0.5, 168)
    
    # Column D: Random walk
    D = np.cumsum(np.random.normal(0, 0.5, 168)) + 50
    
    # Column E: Periodic spikes
    E = 10 + 5 * (np.sin(np.linspace(0, 20*np.pi, 168)) > 0.8) + np.random.normal(0, 0.3, 168)
    
    # Create DataFrame
    df = pd.DataFrame({
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        'E': E
    }, index=dates)
    
    print(f"✓ Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    print(f"  Columns: {', '.join(df.columns)}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def main():
    """Main driver function."""
    print("=" * 60)
    print("ChronoTagger - Interactive Layout Builder Demo")
    print("=" * 60)
    print()
    
    # Step 1: Create or load data
    df = create_sample_data()
    print()
    
    # Step 2: Launch interactive layout builder
    print("🎨 Launching Layout Builder...")
    print("   • Click and drag on grid cells to create panels")
    print("   • Click existing panels to select and edit them")
    print("   • Use 'Preview' to see your layout")
    print("   • Click 'Done' when satisfied with your layout")
    print()
    
    layout_spec, plot_config = build_layout(df)
    
    # Check if user cancelled
    if layout_spec is None or plot_config is None:
        print("❌ Layout builder cancelled by user")
        print("   Exiting...")
        return
    
    print()
    print("✓ Layout created successfully!")
    print(f"  Grid: {layout_spec['nrows']} rows × {layout_spec['ncols']} columns")
    print(f"  Panels: {len(layout_spec['areas'])} panels configured")
    print()
    
    # Step 3: Generate plot function from config
    print("🔧 Generating plot function from layout...")
    plot_fn = generate_plot_fn(plot_config)
    print("✓ Plot function generated")
    print()
    
    # Step 4: Launch TimeIntervalLabeler
    print("🚀 Launching TimeIntervalLabeler...")
    print("   • Use the labeling interface to mark time intervals")
    print("   • Press 'q' to quit when done")
    print()
    
    # Create and run labeler
    app = TimeIntervalLabeler(
        df=df,
        plot_fn=plot_fn,
        layout_spec=layout_spec,
        # Optional: Provide output path for labels
        # output_path='labels.json'
    )
    
    # Run the labeler
    app.run()
    
    print()
    print("=" * 60)
    print("Session Complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        print("   Exiting...")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
