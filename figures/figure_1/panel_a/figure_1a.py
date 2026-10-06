"""Record the schematic status of Figure 1a."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    status = {"panel": "a", "status": "schematic", "requires_data_analysis": False,
              "reference": str(HERE.parent / "figure.svg"),
              "note": "The screening workflow is conceptual artwork, not a numerical analysis."}
    (HERE / "status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
