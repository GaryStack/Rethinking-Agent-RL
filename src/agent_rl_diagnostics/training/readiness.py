"""Convert a provenance ledger into acquisition tasks, never a result certificate."""

from ..contracts.schema import SchemaError, text


def paper_asset_readiness(ledger):
    if not isinstance(ledger, dict) or ledger.get("schema_version") != "ard-asset-ledger-v1":
        raise SchemaError("unsupported asset ledger")
    entries = ledger.get("entries")
    if not isinstance(entries, list) or not entries:
        raise SchemaError("nonempty experiment ledger required")
    seen, records = set(), []
    for entry in entries:
        experiment = text(entry.get("experiment_id"), "experiment_id")
        if experiment in seen:
            raise SchemaError("duplicate experiment identity")
        seen.add(experiment)
        missing = entry.get("missing_fields")
        if not isinstance(missing, list) or any(
            not isinstance(v, str) or not v.strip() for v in missing
        ):
            raise SchemaError("explicit missing-field list required")
        tasks = list(missing)
        if not entry.get("raw_artifacts"):
            tasks.append("acquire_raw_artifacts")
        if not entry.get("verification_command"):
            tasks.append("provide_independent_aggregation_command")
        if entry.get("redistribution_status") != "approved":
            tasks.append("resolve_redistribution_permissions")
        if entry.get("verification_status") != "reproduced":
            tasks.append("execute_and_audit_paper_recipe")
        records.append(
            {
                "experiment_id": experiment,
                "paper_location": text(entry.get("paper_location"), "paper_location"),
                "ledger_evidence_kind": entry.get("evidence_kind"),
                "acquisition_tasks": list(dict.fromkeys(tasks)),
                "executable_paper_recipe_verified": False,
            }
        )
    pending = sum(bool(r["acquisition_tasks"]) for r in records)
    return {
        "schema_version": "ard-paper-readiness-v1",
        "scope": "ledger_metadata_acquisition_checklist",
        "experiments": records,
        "pending_experiments": pending,
        "status": "acquisition_pending" if pending else "requires_independent_verification",
        "paper_reproduction_verified": False,
        "warning": "Presence and metadata are not verified raw results; no assets are redistributed.",
    }
