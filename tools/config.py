from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT
SRC = DATA_ROOT / "machine_leg"
DATA = DATA_ROOT / "machine_leg_roi"

TOP = 0.65
ROI_LEFT = 0.18
ROI_RIGHT = 0.82
SIZE = 960
CONF = 0.25
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
