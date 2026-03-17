"""Quick-start entry point — delegates to scripts/train.py.

For full control use the individual scripts:
    python scripts/train.py --config configs/default.yaml
    python scripts/evaluate.py --run outputs/experiments/<run>
    python scripts/predict.py --run outputs/experiments/<run> --data data.csv
    python scripts/preprocess.py --config configs/default.yaml
"""

import subprocess
import sys


def main():
    """Run the training script with default config."""
    cmd = [sys.executable, "scripts/train.py"] + sys.argv[1:]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
