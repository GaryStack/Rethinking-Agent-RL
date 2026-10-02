import copy
from pathlib import Path

import pytest

from agent_rl_diagnostics.artifacts.io import read_json
from agent_rl_diagnostics.contracts.schema import SchemaError
from agent_rl_diagnostics.training.readiness import paper_asset_readiness


def ledger():
    return read_json(Path(__file__).parents[1] / "docs/reproduction/asset-ledger.json")


def test_actual_ledger_yields_nine_pending_experiments_not_paper_acceptance():
    report = paper_asset_readiness(ledger())
    assert report["pending_experiments"] == 9
    assert not report["paper_reproduction_verified"]
    assert all(
        "resolve_redistribution_permissions" in r["acquisition_tasks"]
        for r in report["experiments"]
    )


def test_metadata_claims_cannot_certify_reproduction():
    value = copy.deepcopy(ledger())
    for entry in value["entries"]:
        entry.update(
            missing_fields=[],
            raw_artifacts=[{"path": "unverified-claim"}],
            verification_command="claimed-command",
            redistribution_status="approved",
            verification_status="reproduced",
        )
    report = paper_asset_readiness(value)
    assert report["status"] == "requires_independent_verification"
    assert not report["paper_reproduction_verified"]


def test_duplicate_experiment_ids_rejected():
    value = ledger()
    value["entries"].append(value["entries"][0])
    with pytest.raises(SchemaError):
        paper_asset_readiness(value)
