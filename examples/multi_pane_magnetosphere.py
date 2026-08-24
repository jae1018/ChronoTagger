"""
Multi-Pane Magnetosphere Data Labeling Example

Demonstrates using multiple panes for complex space physics data:
- Pane 1: Overview (density, temperature, pressure)
- Pane 2: Magnetic field components
- Pane 3: Velocity components
- Pane 4: Position plots (GSE coordinates)

This setup helps identify magnetospheric regions (plasmasheet,
magnetosheath, solar wind, etc.) by viewing complementary parameters.

Keyboard shortcuts:
    Ctrl+Tab        - Switch to next pane
    Ctrl+1-4        - Jump directly to panes 1-4
    Right-click tab - Rename or refresh pane
    F1              - Show all shortcuts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chronotagger import TimeIntervalLabeler

# ============================================================================
# GENERATE SAMPLE MAGNETOSPHERE DATA
# ============================================================================

def generate_sample_data():
    """Generate realistic-looking magnetosphere time series."""
    times = pd.date_range('2024-01-01', periods=2000, freq='30s')

    # Simulate orbital trajectory
    t = np.linspace(0, 20*np.pi, len(times))

    df = pd.DataFrame({
        # Plasma parameters
        'n': np.abs(5 + 3*np.sin(t/3) + np.random.randn(len(times))*0.5),
        'T': 1e6 * np.abs(1 + 0.5*np.cos(t/2) + np.random.randn(len(times))*0.1),
        'P': np.abs(5 + 2*np.sin(t/4) + np.random.randn(len(times))*0.3),

        # Magnetic field
        'Bx': 10*np.cos(t/5) + np.random.randn(len(times))*2,
        'By': 5*np.sin(t/4) + np.random.randn(len(times))*2,
        'Bz': 3*np.sin(t/6) + np.random.randn(len(times))*1,

        # Velocity
        'Vx': -400 + 50*np.sin(t/3) + np.random.randn(len(times))*20,
        'Vy': 30*np.cos(t/4) + np.random.randn(len(times))*10,
        'Vz': 20*np.sin(t/5) + np.random.randn(len(times))*10,

        # Position (GSE - Earth)
        'X_gse': -20 + 15*np.cos(t/10),
        'Y_gse': 15*np.sin(t/10),
        'Z_gse': 3*np.sin(t/8),
    }, index=times)

    return df


# ============================================================================
# PLOT FUNCTIONS
# ============================================================================

def plot_overview(axs, df, t0, t1):
    """Overview: density, temperature, pressure."""
    axs['n'].plot(df.index, df['n'], 'b-', linewidth=1)
    axs['n'].set_ylabel('n (cm⁻³)')
    axs['n'].grid(alpha=0.3)

    axs['T'].semilogy(df.index, df['T'], 'r-', linewidth=1)
    axs['T'].set_ylabel('T (K)')
    axs['T'].grid(alpha=0.3)

    axs['P'].plot(df.index, df['P'], 'g-', linewidth=1)
    axs['P'].set_ylabel('P (nPa)')
    axs['P'].grid(alpha=0.3)


def plot_magnetic_field(axs, df, t0, t1):
    """Magnetic field components."""
    axs['B'].plot(df.index, df['Bx'], 'r-', label='Bx', linewidth=1)
    axs['B'].plot(df.index, df['By'], 'g-', label='By', linewidth=1)
    axs['B'].plot(df.index, df['Bz'], 'b-', label='Bz', linewidth=1)
    axs['B'].set_ylabel('B (nT)')
    axs['B'].legend(loc='upper right', framealpha=0.8)
    axs['B'].grid(alpha=0.3)
    axs['B'].axhline(0, color='k', linewidth=0.5, linestyle='--')

    # |B| total
    B_mag = np.sqrt(df['Bx']**2 + df['By']**2 + df['Bz']**2)
    axs['B_mag'].plot(df.index, B_mag, 'k-', linewidth=1.5)
    axs['B_mag'].set_ylabel('|B| (nT)')
    axs['B_mag'].grid(alpha=0.3)


def plot_velocity(axs, df, t0, t1):
    """Velocity components."""
    axs['V'].plot(df.index, df['Vx'], 'r-', label='Vx', linewidth=1)
    axs['V'].plot(df.index, df['Vy'], 'g-', label='Vy', linewidth=1)
    axs['V'].plot(df.index, df['Vz'], 'b-', label='Vz', linewidth=1)
    axs['V'].set_ylabel('V (km/s)')
    axs['V'].legend(loc='upper right', framealpha=0.8)
    axs['V'].grid(alpha=0.3)
    axs['V'].axhline(0, color='k', linewidth=0.5, linestyle='--')

    # |V| total
    V_mag = np.sqrt(df['Vx']**2 + df['Vy']**2 + df['Vz']**2)
    axs['V_mag'].plot(df.index, V_mag, 'k-', linewidth=1.5)
    axs['V_mag'].set_ylabel('|V| (km/s)')
    axs['V_mag'].grid(alpha=0.3)


def plot_positions(axs, df, t0, t1):
    """Position plots in GSE coordinates."""
    # Geocentric distance vs time -- this pane's role="time" panel
    r_gse = np.sqrt(df['X_gse']**2 + df['Y_gse']**2 + df['Z_gse']**2)
    axs['r_gse'].plot(df.index, r_gse, 'k-', linewidth=1)
    axs['r_gse'].set_ylabel('R (RE)')
    axs['r_gse'].grid(alpha=0.3)

    # XY GSE (Earth-centered)
    axs['xy_gse'].scatter(df['X_gse'], df['Y_gse'], s=2, c='blue', alpha=0.5)
    axs['xy_gse'].set_xlabel('X (RE)')
    axs['xy_gse'].set_ylabel('Y (RE)')
    axs['xy_gse'].set_title('GSE X-Y Coordinates')
    axs['xy_gse'].grid(alpha=0.3)
    axs['xy_gse'].set_aspect('equal')

    # Add Earth
    circle_earth = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
    axs['xy_gse'].add_patch(circle_earth)


# ============================================================================
# LAYOUT SPECIFICATIONS
# ============================================================================

layout_overview = {
    "nrows": 4,
    "ncols": 1,
    "hspace": 0.05,
    "areas": [
        {"key": "n", "row": 0, "col": 0, "role": "time"},
        {"key": "T", "row": 1, "col": 0, "role": "time"},
        {"key": "P", "row": 2, "col": 0, "role": "time"},
        {"key": "labels", "row": 3, "col": 0, "role": "labels"},
    ]
}

layout_field = {
    "nrows": 3,
    "ncols": 1,
    "hspace": 0.05,
    "areas": [
        {"key": "B", "row": 0, "col": 0, "role": "time"},
        {"key": "B_mag", "row": 1, "col": 0, "role": "time"},
        {"key": "labels", "row": 2, "col": 0, "role": "labels"},
    ]
}

layout_velocity = {
    "nrows": 3,
    "ncols": 1,
    "hspace": 0.05,
    "areas": [
        {"key": "V", "row": 0, "col": 0, "role": "time"},
        {"key": "V_mag", "row": 1, "col": 0, "role": "time"},
        {"key": "labels", "row": 2, "col": 0, "role": "labels"},
    ]
}

layout_positions = {
    # Every pane needs at least one role="time" area (canvas.py:151).
    # This one had only a cross-plot and a labels strip, so building the
    # fourth tab raised ValueError and took the whole example down with
    # it. Widened to two columns: a time panel on the left, which the
    # labels strip lines up under, and the GSE cross-plot on the right.
    "nrows": 2,
    "ncols": 2,
    "hspace": 0.1,
    "wspace": 0.15,
    "areas": [
        {"key": "r_gse", "row": 0, "col": 0, "role": "time"},
        {"key": "xy_gse", "row": 0, "col": 1, "role": "not-time"},
        {"key": "labels", "row": 1, "col": 0, "role": "labels"},
    ]
}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Generate sample data
    print("Generating sample magnetosphere data...")
    df = generate_sample_data()
    print(f"Created {len(df)} time points spanning {df.index[0]} to {df.index[-1]}")

    # Define panes
    panes = [
        {
            "title": "Overview (n, T, P)",
            "plot_fn": plot_overview,
            "layout_spec": layout_overview,
        },
        {
            "title": "Magnetic Field",
            "plot_fn": plot_magnetic_field,
            "layout_spec": layout_field,
        },
        {
            "title": "Velocity",
            "plot_fn": plot_velocity,
            "layout_spec": layout_velocity,
        },
        {
            "title": "Position",
            "plot_fn": plot_positions,
            "layout_spec": layout_positions,
        },
    ]

    # Define label classes
    classes = [
        "UNKNOWN",
        "SolarWind",
        "Magnetosheath",
        "PlasmaSheet",
        "LobesBoundary",
    ]

    # Create labeler
    print("\nStarting multi-pane labeler...")
    print("="*70)
    print("MULTI-PANE FEATURES:")
    print("  • Ctrl+Tab / Ctrl+Shift+Tab - Switch tabs")
    print("  • Ctrl+1-4 - Jump to specific tab")
    print("  • Right-click tab - Rename/refresh")
    print("  • F1 - Show all shortcuts")
    print("="*70)

    labeler = TimeIntervalLabeler(
        df=df,
        panes=panes,
        classes=classes,
        window=pd.Timedelta("2h"),
        step=pd.Timedelta("30min"),
        # There is no autosave_path parameter and never has been -- this
        # raised TypeError. The autosave file NAME is derived from a
        # dataset fingerprint; the caller chooses only the folder.
        autosave_folder=".",
    )

    labeler.run()
