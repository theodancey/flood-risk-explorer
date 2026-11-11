"""
check_setup.py - confirm the project's Python environment works.

Run it from the project folder with:
    .venv\\Scripts\\python.exe src\\check_setup.py

It does three things:
  1. Prints which Python version is running.
  2. Imports every library the project needs and prints its version.
  3. Makes sure the data folders exist (data/ is not stored in Git, so a
     fresh copy of the project won't have them).

If any library fails to import, the script says which one and exits with
an error code, so it's easy to tell when something is broken.
"""

import importlib
import sys
from pathlib import Path

# The libraries we need. Left side: the name we import in Python code.
# Right side: the name used for `pip install` (they differ for a couple of them).
LIBRARIES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "requests": "requests",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",
    "sklearn": "scikit-learn",
    "statsmodels": "statsmodels",
    "xgboost": "xgboost",
    "shap": "shap",
    # 'jupyter' is a bundle with no version of its own; jupyter_core is its main piece,
    # so the version printed here is jupyter_core's, not the bundle's.
    "jupyter_core": "jupyter-core",
    "ipykernel": "ipykernel",
    "streamlit": "streamlit",
}

# The project folder is one level above this file (src/ -> project root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDERS = [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "data" / "processed"]


def check_libraries():
    """Try to import each library. Return a list of the ones that failed."""
    failed = []
    for import_name, pip_name in LIBRARIES.items():
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"  OK    {pip_name:<14} {version}")
        except Exception as error:  # catch anything so one failure doesn't hide the rest
            print(f"  FAIL  {pip_name:<14} {type(error).__name__}: {error}")
            failed.append(pip_name)
    return failed


def ensure_data_folders():
    """Create data/raw and data/processed if they are missing."""
    for folder in DATA_FOLDERS:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"  OK    {folder.relative_to(PROJECT_ROOT)}")


def main():
    print(f"Python {sys.version.split()[0]} at {sys.executable}\n")

    print("Libraries:")
    failed = check_libraries()

    print("\nData folders:")
    ensure_data_folders()

    if failed:
        print(f"\nSetup problem: {len(failed)} library(ies) failed: {', '.join(failed)}")
        sys.exit(1)
    print(f"\nAll {len(LIBRARIES)} libraries imported successfully. Setup looks good!")


if __name__ == "__main__":
    main()
