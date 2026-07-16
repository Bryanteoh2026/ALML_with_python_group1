"""Run the complete project pipeline in the correct order."""

import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
SCRIPTS = [
    "01_data_cleaning.py",
    "02_model_training_analysis.py",
    "03_customer_segmentation.py",
]


for script in SCRIPTS:
    print(f"\nRunning {script}...")
    subprocess.run([sys.executable, str(BASE_DIR / script)], check=True)

print("\nAll project steps completed successfully.")
