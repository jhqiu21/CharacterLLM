import os
import sys
from pathlib import Path

# Tests run on CPU so they never compete with a training run for the GPU (override with JAX_PLATFORMS=cuda).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
