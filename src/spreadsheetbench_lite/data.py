from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Task:
    """One SpreadsheetBench V1 instruction from dataset.json."""

    id: str
    instruction: str
    spreadsheet_path: str
    instruction_type: str
    answer_position: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class TestCase:
    """One input workbook for a task."""

    index: int
    input_path: Path
    answer_path: Path | None
    output_path: Path


def load_tasks(dataset_dir: Path) -> list[Task]:
    dataset_file = dataset_dir / "dataset.json"
    with dataset_file.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

    tasks: list[Task] = []
    for record in records:
        tasks.append(
            Task(
                id=str(record["id"]),
                instruction=str(record["instruction"]),
                spreadsheet_path=str(record["spreadsheet_path"]),
                instruction_type=str(record.get("instruction_type", "")),
                answer_position=str(record.get("answer_position", "")),
                raw=record,
            )
        )
    return tasks


def task_spreadsheet_dir(dataset_dir: Path, task: Task) -> Path:
    path = Path(task.spreadsheet_path)
    if path.is_absolute():
        return path
    return dataset_dir / path


def discover_test_cases(dataset_dir: Path, outputs_dir: Path, task: Task) -> list[TestCase]:
    """Find all official `*_input.xlsx` files for a task and map output paths."""

    source_dir = task_spreadsheet_dir(dataset_dir, task)
    task_output_dir = outputs_dir / "spreadsheet" / task.id
    input_files = sorted(
        source_dir.glob(f"*_{task.id}_input.xlsx"),
        key=lambda path: _case_index(path.name),
    )

    cases: list[TestCase] = []
    for input_path in input_files:
        index = _case_index(input_path.name)
        answer_path = input_path.with_name(input_path.name.replace("_input.xlsx", "_answer.xlsx"))
        if not answer_path.exists():
            answer_path = None
        output_path = task_output_dir / input_path.name.replace("_input.xlsx", "_output.xlsx")
        cases.append(TestCase(index=index, input_path=input_path, answer_path=answer_path, output_path=output_path))
    return cases


def _case_index(file_name: str) -> int:
    match = re.match(r"^(\d+)_", file_name)
    return int(match.group(1)) if match else 10**9
