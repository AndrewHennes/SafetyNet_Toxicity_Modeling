"""Compute Figure 1b molecular chemical space using the shared implementation."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from chemical_space import export_panel


def main():
    export_panel("figure_1", output=HERE)


if __name__ == "__main__":
    main()
