#!/usr/bin/env python3
"""
Quick launcher for the Streamlit dashboard.

Usage:
    python run_dashboard.py
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "src" / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"[ERROR] Dashboard not found at {dashboard_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("Starting Strategic Planning Dashboard")
    print("=" * 80)
    print(f"\nDashboard will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard\n")
    print("=" * 80)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_path)
        ])
    except KeyboardInterrupt:
        print("\n[DASHBOARD] Stopped by user")
    except FileNotFoundError:
        print("\n[ERROR] Streamlit not found. Install with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to start dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
