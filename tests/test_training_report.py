import importlib.util
from pathlib import Path

import pytest


def test_training_report_keeps_axes_and_missing_values_separate():
    spec = importlib.util.spec_from_file_location(
        "training_report", Path(__file__).parents[1] / "scripts" / "report_training_diagnostics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    points = [
        {"checkpoint_step": 0, "metrics": {"diag/grid/multi/pass_k1_t2": 0, "train/loss": None}},
        {"checkpoint_step": 1, "metrics": {"diag/grid/multi/pass_k1_t2": 0.5, "train/loss": -2}},
    ]
    page = module.render(points)
    assert "step 0: 0, step 1: 0.5" in page
    assert "step 1: -2" in page
    assert "step 0: -2" not in page
    assert "No network resources" in page
    assert "<script" not in page
    with pytest.raises(ValueError, match="increasing"):
        module.render(points[::-1])
