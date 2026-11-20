"""Make `src` importable when running pytest from any folder in the repo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
