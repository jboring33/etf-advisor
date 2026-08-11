"""
config/portfolio.py
===================
Portfolio configuration and baseline ETF scan pools with disk persistence.
"""

import os
import json

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PERSISTENT_FILE = os.path.join(DATA_DIR, "user_universe.json")

DEFAULT_SCAN_POOL = {
    "Core Equity": ["SPY", "QQQ", "DIA", "SCHD", "VFLO"],
    "Income & Credit": ["SCYB", "JPST", "JAAA"],
    "Tactical / Value": ["VFLO"],
}

def load_universe() -> dict:
    """Loads ETF scan pool from disk if present, else defaults."""
    if os.path.exists(PERSISTENT_FILE):
        try:
            with open(PERSISTENT_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_SCAN_POOL.copy()
    return DEFAULT_SCAN_POOL.copy()

def save_universe(universe: dict) -> None:
    """Persists ETF scan pool changes directly to disk."""
    with open(PERSISTENT_FILE, "w") as f:
        json.dump(universe, f, indent=4)

DYNAMIC_SCAN_POOL = load_universe()
