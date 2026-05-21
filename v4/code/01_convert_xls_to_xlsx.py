"""Convert .xls -> .xlsx using Excel itself (preserves ALL formatting).

Requires: Microsoft Excel installed + pywin32.
Excel does a real "Save As" so merged cells, column widths, fonts,
borders, number formats, etc. are kept exactly as in the original.
"""
from pathlib import Path
import win32com.client as win32

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "01_raw"
OUT_DIR = BASE_DIR / "02_cleaned"
xlOpenXMLWorkbook = 51  # .xlsx file format

xls_files = sorted(SRC_DIR.rglob("*.xls"))
if not xls_files:
    raise SystemExit(f"No .xls files found in {SRC_DIR}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

excel = win32.DispatchEx("Excel.Application")
excel.Visible = False
excel.DisplayAlerts = False

try:
    for xls_path in xls_files:
        # Mirror subfolder structure from SRC_DIR into OUT_DIR
        rel = xls_path.relative_to(SRC_DIR).with_suffix(".xlsx")
        xlsx_path = OUT_DIR / rel
        xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Converting {xls_path.name} -> {xlsx_path.relative_to(BASE_DIR)}")

        wb = excel.Workbooks.Open(str(xls_path))
        wb.SaveAs(str(xlsx_path), FileFormat=xlOpenXMLWorkbook)
        wb.Close(SaveChanges=False)
finally:
    excel.Quit()

print("\nDone.")
