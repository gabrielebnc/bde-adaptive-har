"""Live, paired-window demonstration of the selected Adaptive HAR checkpoint."""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
