"""Convert .xls -> .xlsx using Excel itself (preserves ALL formatting).

Requires: Microsoft Excel installed + pywin32.
Excel does a real "Save As" so merged cells, column widths, fonts,
borders, number formats, etc. are kept exactly as in the original.
"""
from pathlib import Path
import win32com.client as win32

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "01_raw"
xlOpenXMLWorkbook = 51  # .xlsx file format

xls_files = sorted(DATA_DIR.rglob("*.xls"))
if not xls_files:
    raise SystemExit(f"No .xls files found in {DATA_DIR}")

excel = win32.DispatchEx("Excel.Application")
excel.Visible = False
excel.DisplayAlerts = False

try:
    for xls_path in xls_files:
        xlsx_path = xls_path.with_suffix(".xlsx")
        print(f"Converting {xls_path.name} -> {xlsx_path.name}")

        wb = excel.Workbooks.Open(str(xls_path))
        wb.SaveAs(str(xlsx_path), FileFormat=xlOpenXMLWorkbook)
        wb.Close(SaveChanges=False)
finally:
    excel.Quit()

print("\nDone.")
