"""Weights & Biases experiment tracker.

Thin wrapper around wandb that mirrors the ExperimentLogger interface,
so both can be used side-by-side in the training loop.
"""

import os

try:
    import wandb
except ImportError:
    wandb = None


class WandbTracker:
    """Log configs, metrics, and artifacts to Weights & Biases."""

    def __init__(
        self,
        project: str,
        experiment_name: str,
        config: dict | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        mode: str = "online",
    ):
        if wandb is None:
            raise ImportError(
                "wandb is not installed. Install it with: pip install wandb"
            )

        self.run = wandb.init(
            project=project,
            name=experiment_name,
            config=config or {},
            tags=tags,
            notes=notes,
            mode=mode,  # "online", "offline", or "disabled"
        )

    def log_config(self, cfg: dict):
        """Update the run config."""
        self.run.config.update(cfg, allow_val_change=True)

    def log_metric(self, key: str, value: float, step: int | None = None):
        """Log a single metric."""
        payload = {key: value}
        if step is not None:
            payload["step"] = step
        self.run.log(payload, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """Log multiple metrics at once."""
        self.run.log(metrics, step=step)

    def log_artifact(self, filepath: str, artifact_type: str = "model"):
        """Upload a file as a wandb artifact."""
        name = os.path.splitext(os.path.basename(filepath))[0]
        artifact = wandb.Artifact(name, type=artifact_type)
        artifact.add_file(filepath)
        self.run.log_artifact(artifact)

    def summary(self) -> dict:
        """Return wandb run summary."""
        return dict(self.run.summary)

    def finish(self):
        """End the wandb run."""
        self.run.finish()

    def __repr__(self):
        return f"WandbTracker(run={self.run.name!r})"
