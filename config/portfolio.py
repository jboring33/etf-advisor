"""
config/portfolio.py
===================
Portfolio configuration with dynamic defaults and disk persistence.
"""

import os
import json

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
USER_FILE = os.path.join(DATA_DIR, "user_universe.json")
DEFAULT_FILE = os.path.join(DATA_DIR, "default_universe.json")

# Hardcoded fallback ONLY used if the default JSON is completely missing
HARDCODED_FALLBACK = {
    "Core Equity": ["SPY", "QQQ", "DIA", "SCHD", "VFLO"],
    "Income & Credit": ["SCYB", "JPST", "JAAA"],
    "Tactical / Value": ["VFLO"],
}

def load_defaults() -> dict:
    """Loads defaults from disk. Creates the default file if it is missing."""
    if os.path.exists(DEFAULT_FILE):
        try:
            with open(DEFAULT_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
            
    # Write the hardcoded fallback to disk so it becomes the dynamic default
    with open(DEFAULT_FILE, "w") as f:
        json.dump(HARDCODED_FALLBACK, f, indent=4)
    return HARDCODED_FALLBACK.copy()

def load_universe() -> dict:
    """Loads active user edits, or falls back to disk defaults if no edits exist."""
    if os.path.exists(USER_FILE):
        try:
            with open(USER_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return load_defaults()

def save_universe(universe: dict, as_default: bool = False) -> None:
    """
    Saves ETF changes to disk. 
    If as_default is True, it overwrites the baseline defaults.
    """
    target_file = DEFAULT_FILE if as_default else USER_FILE
    with open(target_file, "w") as f:
        json.dump(universe, f, indent=4)

def restore_defaults() -> dict:
    """Wipes active user edits and returns to the disk-saved baseline."""
    if os.path.exists(USER_FILE):
        os.remove(USER_FILE)
    return load_defaults()

DYNAMIC_SCAN_POOL = load_universe()
