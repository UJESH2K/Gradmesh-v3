from pathlib import Path
import runpy
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
runpy.run_path(str(REPOSITORY_ROOT / "worker.py"), run_name="__main__")
