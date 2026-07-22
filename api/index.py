"""Vercel Python Serverless Function entry point.

Routes /api/* and everything else to the FastAPI app in ../backend/main.py.
"""

import sys
import os

# Make 'backend/' importable as the root package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from main import app  # noqa: F401  — Vercel looks for `app` in this module
