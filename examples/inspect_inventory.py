"""Execute real local tools via Inspect; scripted offline model is clearly labeled."""

import argparse
import json
import sqlite3
from pathlib import Path

from agent_rl_diagnostics.artifacts.io import new_output_directory, write_json
from agent_rl_diagnostics.external_traces import compare_traces
from agent_rl_diagnostics.integrations.inspect_logs import import_inspect


def run(output, model_name=None):
    from inspect_ai import Task, eval
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageTool, ModelOutput, ModelUsage, get_model
    from inspect_ai.scorer import match
    from inspect_ai.solver import generate, use_tools
    from inspect_ai.tool import ToolError, tool
    from inspect_ai.util import store

    @tool
    def stock(retry=False):
        async def execute(product: str) -> str:
            """Look up inventory. A temporary error can be retried.

            Args:
                product: Product name, pen or notebook.
            """
            for attempt in range(2 if retry else 1):
                # Per-sample store isolates the controlled first-request fault.
                calls = store().get("inventory_calls", 0) + 1
                store().set("inventory_calls", calls)
                if product == "pen" and calls == 1:
                    if retry and attempt == 0:
                        continue
                    raise ToolError("Temporary inventory service error; retry allowed.")
                # Only authored, parameterized SQL executes; model supplies a value.
                with sqlite3.connect(":memory:") as database:
                    database.execute("CREATE TABLE inventory (name TEXT, stock INTEGER)")
                    database.executemany(
                        "INSERT INTO inventory VALUES (?, ?)", [("pen", 7), ("notebook", 4)]
                    )
                    row = database.execute(
                        "SELECT stock FROM inventory WHERE name = ?", (product,)
                    ).fetchone()
                return str(row[0]) if row else "unknown product"
            raise ToolError("Inventory retry limit exceeded")

        return execute

    def scripted(messages, tools, tool_choice, config):
        results = [m for m in messages if isinstance(m, ChatMessageTool)]
        if results:
            content = "unknown" if results[-1].error else results[-1].text
            output = ModelOutput.from_content("mockllm/inventory", content)
        else:
            prompt = next(m.text for m in messages if m.role == "user")
            product = "notebook" if "notebook" in prompt else "pen"
            output = ModelOutput.for_tool_call("mockllm/inventory", "stock", {"product": product})
        # No LLM tokens are consumed in scripted mode; avoid a tokenizer download.
        output.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        return output

    mode = "live_model" if model_name else "scripted_model_real_tools"
    design = dict(
        task_ids=["pen", "notebook"],
        samples_per_task=1,
        protocol_id="inventory-tool-loop-v1",
        scorer_id="inspect-match-v1",
        dataset_id="authored-inventory-v1",
        budget_unit="assistant_turns",
        budget_limit=3,
        cost_unit="USD",
    )
    with new_output_directory(output) as staging:
        write_json(staging / "design.json", design)
        native_logs = {}
        for label, retry in [("base", False), ("retry", True)]:
            model = (
                model_name
                if model_name
                else get_model("mockllm/inventory", custom_outputs=scripted, memoize=False)
            )
            task = Task(
                name="inventory_lookup",
                dataset=[
                    Sample(
                        id=p,
                        input=f"Use stock to find the inventory of {p}. "
                        "Reply with only the integer. Retry temporary errors if needed.",
                        target=n,
                    )
                    for p, n in [("pen", "7"), ("notebook", "4")]
                ],
                solver=[use_tools(stock(retry=retry)), generate()],
                scorer=match(),
                metadata={"case_mode": mode, "tool_internal_retry": retry},
            )
            logs = eval(
                task,
                model=model,
                epochs=1,
                turn_limit=3,
                max_samples=1,
                log_dir=str(staging / f"native-{label}"),
                display="none",
                log_format="eval",
                fail_on_error=True,
            )
            if len(logs) != 1 or logs[0].status != "success":
                detail = logs[0].error.message if logs and logs[0].error else "no error detail"
                raise ValueError(f"Inspect {label} run did not complete: {detail}")
            source = Path(logs[0].location)
            native_logs[label] = str(source.relative_to(staging))
            import_inspect(
                source, scorer="match", design=design, label=label, output=staging / label
            )
        result = compare_traces(staging / "base", staging / "retry", staging / "comparison")
        write_json(
            staging / "case.json",
            {
                "mode": mode,
                "native_logs": native_logs,
                "paired_success_delta": result["paired_success_delta"],
                "scope": "Authored two-task fault-injection case; not a business benchmark or efficacy claim.",
            },
        )
    return {
        "output": str(output),
        "mode": mode,
        "report": str(output / "comparison" / "report.html"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", help="Optional Inspect model name; omitted uses offline scripted outputs"
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.model), indent=2))
