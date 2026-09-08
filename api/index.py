import sys
from pathlib import Path

# Vercel runs api/index.py with cwd=repo root. Backend lives in backend/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.main import app  # noqa: E402

# Vercel Python runtime looks for `app` (ASGI). No Mangum needed.
