"""Generate Supplementary Figure 1a from any working directory."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    subprocess.run([sys.executable, str(HERE / "panel_a/supplementary_figure_1a.py")], check=True)


if __name__ == "__main__":
    main()
