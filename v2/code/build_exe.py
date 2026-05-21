"""
build_exe.py
============
Bundle the manager app + forecast pipeline into a single standalone .exe.
The manager double-clicks ForecastManager.exe → browser opens → done.
No Python installation required on the target machine.

Run once on YOUR machine:
    python v2\\code\\build_exe.py

Result:
    dist\\ForecastManager.exe   (~80–120 MB, self-contained)
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # pfe_plan/
CODE = ROOT / "v2" / "code"

# Make sure PyInstaller is available
try:
    import PyInstaller  # noqa: F401
except ImportError:
    print("[setup] installing pyinstaller...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

# Files that must travel inside the .exe so subprocess can find them
DATA_FILES = [
    ("v2/code/forecast_pipeline.py", "v2/code"),
    ("v2/code/02_extract_budget_data.py", "v2/code"),
    ("v2/code/01_convert_xls_to_xlsx.py", "v2/code"),
]

# Build PyInstaller args
args = [
    sys.executable, "-m", "PyInstaller",
    "--name", "ForecastManager",
    "--onefile",
    "--noconsole" if "--windowed" in sys.argv else "--console",
    "--clean",
    "--noconfirm",
    # Hidden imports PyInstaller may miss
    "--hidden-import", "sklearn.utils._typedefs",
    "--hidden-import", "sklearn.neighbors._partition_nodes",
    "--hidden-import", "openpyxl",
    "--hidden-import", "xlrd",
    "--hidden-import", "xgboost",
    "--collect-all", "sklearn",
    "--collect-all", "xgboost",
    str(CODE / "manager_app.py"),
]

# Attach the pipeline scripts as data so subprocess.run can locate them
for src, dst in DATA_FILES:
    src_abs = ROOT / src
    if src_abs.exists():
        args += ["--add-data", f"{src_abs};{dst}"]
    else:
        print(f"[warn] missing data file: {src_abs}")

print("[build] running:", " ".join(args))
subprocess.check_call(args, cwd=str(ROOT))
print("\n[done] -> dist/ForecastManager.exe")
print("       Copy this single file to the manager's desktop. Double-click to run.")
