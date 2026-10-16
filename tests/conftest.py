import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gateway" / "receiver"))
sys.path.insert(0, str(ROOT / "edge_ai" / "fusion" / "src"))
sys.path.insert(0, str(ROOT / "tests"))
