"""Hace que `src` sea importable al correr pytest desde cualquier carpeta del repo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
