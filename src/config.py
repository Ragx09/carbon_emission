from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "CO2 Emissions_Canada.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COL = "CO2 Emissions(g/km)"
