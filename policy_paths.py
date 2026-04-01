# policy_paths.py
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"
COMPARE_DIR = STORAGE_DIR / "compare_prod"

POLICY_A_DIR = DATA_DIR / "policy_a"
POLICY_B_DIR = DATA_DIR / "policy_b"

for p in [DATA_DIR, STORAGE_DIR, COMPARE_DIR, POLICY_A_DIR, POLICY_B_DIR]:
    p.mkdir(parents=True, exist_ok=True)