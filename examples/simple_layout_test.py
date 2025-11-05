"""
Simple standalone test for layout builder.

Run this from command line:
    python examples/simple_layout_test.py

This will test if the layout builder works.
"""

import numpy as np
import pandas as pd

def main():
    print("Simple Layout Builder Test")
    print("="*50)
    print()
    
    # Create simple test data
    df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100),
    }, index=pd.date_range('2024-01-01', periods=100, freq='1h'))
    
    print(f"Created test DataFrame: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print()
    print("Launching layout builder...")
    print("(A dialog window should appear)")
    print()
    
    # Import and launch layout builder
    from chronotagger.labeler.utils import build_layout
    
    layout_spec, plot_config = build_layout(df)
    
    # Check results
    if layout_spec is None:
        print("\nLayout builder was canceled.")
    else:
        print("\n" + "="*50)
        print("SUCCESS! Layout created:")
        print("="*50)
        print("\nlayout_spec:")
        import pprint
        pprint.pprint(layout_spec)
        print("\nplot_config:")
        pprint.pprint(plot_config)
        print("\n✓ Layout builder works!")

if __name__ == "__main__":
    main()
