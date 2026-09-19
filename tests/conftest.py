import sys
from pathlib import Path

# The observations package lives beside this tests directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
