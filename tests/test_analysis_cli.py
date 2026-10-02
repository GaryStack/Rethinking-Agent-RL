import json

from agent_rl_diagnostics.cli import main


def test_cpu_compare_and_likelihood_cli(tmp_path):
    run = tmp_path / "demo"
    assert main(["demo", "--offline", "--output", str(run)]) == 0
    output = tmp_path / "comparison"
    assert (
        main(["compare", "--input", str(run), "--labels", "Synthetic", "--output", str(output)])
        == 0
    )
    summary = json.loads((output / "summary.json").read_text())
    assert len({p["label"] for p in summary["series"]}) >= 2
    assert any(x["x_unit"] == "T_minus_exact_minimum" for x in summary["series"])
    assert (
        main(
            [
                "likelihood",
                "analyze",
                "--base",
                "examples/likelihood/base-scores.json",
                "--rl",
                "examples/likelihood/rl-scores.json",
                "--token-caliper",
                "0.2",
                "--step-caliper",
                "1",
                "--output",
                str(tmp_path / "likelihood"),
            ]
        )
        == 0
    )
