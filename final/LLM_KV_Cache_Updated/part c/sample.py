"""Legacy entry point; delegates to the corrected root sample.py."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sample import main

if __name__ == "__main__":
    main()
