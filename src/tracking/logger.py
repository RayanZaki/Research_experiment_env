"""Lightweight experiment logger.

Saves experiment configs, metrics, and artifacts into a timestamped run
directory.  Works standalone or as a thin wrapper you can later swap for
MLflow / Weights & Biases.
"""

import json
import os
import shutil
from datetime import datetime


class ExperimentLogger:
    """Log configs, metrics, and artifacts for a single experiment run."""

    def __init__(self, experiment_name: str, base_dir: str = "outputs/experiments"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{experiment_name}_{timestamp}"
        self.run_dir = os.path.join(base_dir, self.run_name)
        self.metrics_path = os.path.join(self.run_dir, "metrics.json")
        self.config_path = os.path.join(self.run_dir, "config.yaml")
        self.artifacts_dir = os.path.join(self.run_dir, "artifacts")

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)

        self._metrics: dict[str, list] = {}
        self._params: dict = {}
        print(f"[ExperimentLogger] Run directory: {self.run_dir}")

    # -- config --
    def log_config(self, cfg: dict):
        """Save the full experiment config (dict or OmegaConf)."""
        self._params = cfg
        with open(self.config_path, "w") as f:
            # Use json for dicts; if you use OmegaConf pass OmegaConf.to_yaml(cfg)
            json.dump(cfg, f, indent=2, default=str)

    # -- metrics --
    def log_metric(self, key: str, value: float, step: int | None = None):
        """Append a metric value (optionally with a step number)."""
        if key not in self._metrics:
            self._metrics[key] = []
        entry = {"value": value}
        if step is not None:
            entry["step"] = step
        self._metrics[key].append(entry)
        self._flush_metrics()

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """Log multiple metrics at once."""
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def _flush_metrics(self):
        with open(self.metrics_path, "w") as f:
            json.dump(self._metrics, f, indent=2)

    # -- artifacts --
    def log_artifact(self, filepath: str):
        """Copy a file into the run's artifacts directory."""
        dest = os.path.join(self.artifacts_dir, os.path.basename(filepath))
        shutil.copy2(filepath, dest)
        print(f"[ExperimentLogger] Artifact saved: {dest}")

    # -- summary --
    def summary(self) -> dict:
        """Return the latest value of every metric."""
        return {k: v[-1]["value"] for k, v in self._metrics.items()}

    def __repr__(self):
        return f"ExperimentLogger(run={self.run_name!r})"


if __name__ == "__main__":
    print("tracking.logger module ready")
