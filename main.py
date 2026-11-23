
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

scripts = [
    "data_ingestion.py",
    "data_cleaning.py",
    "data_analysis.py",
    "demand_prediction.py",
    "model_evaluation.py",
    "prep_recommendation.py",
]

for script in scripts:
    script_path = PROJECT_DIR / "src" / script

    print(f"\n{'=' * 50}")
    print(f"Running: {script}")
    print("=" * 50)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_DIR,
        check=True
    )

print("\nAll project steps completed successfully!")
