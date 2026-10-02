"""Authored integration maps, not redistributed paper benchmark tasks."""

from ..contracts.schema import SchemaError

TRAIN_GRID_TASKS = (
    {"task_id": "train-right-one", "desc": ["SG"]},
    {"task_id": "train-left-one", "desc": ["GS"]},
    {"task_id": "train-down-one", "desc": ["S", "G"]},
    {"task_id": "train-up-one", "desc": ["G", "S"]},
)


def check_grid_holdout(validation_tasks):
    if {t["task_id"] for t in TRAIN_GRID_TASKS} & {t["task_id"] for t in validation_tasks}:
        raise SchemaError("training and validation task IDs overlap")
    if {tuple(t["desc"]) for t in TRAIN_GRID_TASKS} & {tuple(t["desc"]) for t in validation_tasks}:
        raise SchemaError("training and validation map contents overlap")
