"""Tests for experiment tracking."""

import os
import tempfile

from src.tracking.logger import ExperimentLogger


def test_experiment_logger_creates_run_dir():
    with tempfile.TemporaryDirectory() as tmp:
        logger = ExperimentLogger("test_exp", base_dir=tmp)
        assert os.path.isdir(logger.run_dir)
        assert os.path.isdir(logger.artifacts_dir)


def test_experiment_logger_logs_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        logger = ExperimentLogger("test_exp", base_dir=tmp)
        logger.log_metric("loss", 0.5, step=1)
        logger.log_metric("loss", 0.3, step=2)
        summary = logger.summary()
        assert summary["loss"] == 0.3
        assert os.path.exists(logger.metrics_path)
