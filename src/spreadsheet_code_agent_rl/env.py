from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spreadsheetbench_lite.data import Task, discover_test_cases, load_tasks
from spreadsheetbench_lite.evaluate_v1 import compare_workbooks
from spreadsheetbench_lite.excel_preview import workbook_preview
from spreadsheet_code_agent_v0.inspection import inspect_workbook, render_inspection
from spreadsheet_code_agent_v0.preflight import run_preflight
from spreadsheet_code_agent_v0.runtime import run_transform_code

from .reward import reward_from_record


@dataclass
class StepResult:
    observation: dict[str, Any]
    reward: dict[str, Any]
    done: bool
    info: dict[str, Any]


class SpreadsheetCodeEnv:
    """Deterministic code-as-action environment for SpreadsheetBench tasks.

    The policy action is a full Python program defining transform(wb). The env
    handles preflight, execution, verification, and evaluator reward.
    """

    def __init__(self, dataset_dir: Path, outputs_dir: Path, task: Task, timeout: int = 240, rows: int = 3, reward_profile: str = "hybrid", max_cases: int | None = None):
        self.dataset_dir = dataset_dir.resolve()
        self.outputs_dir = outputs_dir.resolve()
        self.task = task
        self.timeout = timeout
        self.rows = rows
        self.reward_profile = reward_profile
        self.max_cases = max_cases
        self.cases = discover_test_cases(self.dataset_dir, self.outputs_dir, self.task)
        if self.max_cases is not None:
            self.cases = self.cases[: self.max_cases]
        if not self.cases:
            raise ValueError(f"no test cases for task {self.task.id}")
        self.preview = workbook_preview(self.cases[0].input_path, rows=self.rows)
        self.inspection = inspect_workbook(self.cases[0].input_path, None, self.task.answer_position, rows=max(self.rows, 5))

    @classmethod
    def from_task_id(cls, dataset_dir: Path, outputs_dir: Path, task_id: str, **kwargs):
        tasks = {task.id: task for task in load_tasks(dataset_dir)}
        if task_id not in tasks:
            raise KeyError(f"task id not found: {task_id}")
        return cls(dataset_dir, outputs_dir, tasks[task_id], **kwargs)

    def reset(self) -> dict[str, Any]:
        return {
            "id": self.task.id,
            "instruction": self.task.instruction,
            "instruction_type": self.task.instruction_type,
            "answer_position": self.task.answer_position,
            "preview": self.preview,
            "inspection": self.inspection,
        }

    def step_code(self, code: str, attempt: int = 0, action_text: str | None = None) -> StepResult:
        preflight = run_preflight(code, self.inspection).to_dict()
        if not preflight.get("ok"):
            record = self._preflight_record(preflight)
            reward = reward_from_record(record, self.reward_profile, action_text=action_text)
            return StepResult({"preflight": preflight, "record": record}, reward, True, {"preflight_failed": True})

        record = self._execute(code, attempt)
        reward = reward_from_record(record, self.reward_profile, action_text=action_text)
        return StepResult({"preflight": preflight, "record": record}, reward, True, {"preflight_failed": False})

    def _execute(self, code: str, attempt: int) -> dict[str, Any]:
        case_results = []
        eval_rows = []
        for case in self.cases:
            result = run_transform_code(
                code,
                case.input_path,
                case.output_path,
                self.outputs_dir / "rl_exec" / self.task.id / f"attempt_{attempt}" / str(case.index),
                self.timeout,
                self.task.answer_position,
            )
            case_results.append(
                {
                    "case": case.index,
                    "input": str(case.input_path),
                    "output": str(case.output_path),
                    "ok": result.ok,
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "error_category": result.error_category,
                    "stdout": result.stdout[-4000:],
                    "stderr": result.stderr[-4000:],
                    "verification": result.verification,
                }
            )
            try:
                ok, message = compare_workbooks(case.answer_path, case.output_path, self.task.answer_position)
            except Exception as exc:
                ok, message = False, f"evaluation error: {type(exc).__name__}: {exc}"
            eval_rows.append({"case": case.index, "ok": int(ok), "message": message})
        return self._record(case_results, eval_rows)

    def _preflight_record(self, preflight: dict[str, Any]) -> dict[str, Any]:
        message = "preflight failed: " + str((preflight.get("errors") or [])[:3])[:1000]
        case_results = [
            {
                "case": case.index,
                "input": str(case.input_path),
                "output": str(case.output_path),
                "ok": False,
                "returncode": None,
                "timed_out": False,
                "error_category": "preflight_error",
                "stdout": "",
                "stderr": message,
                "verification": {},
            }
            for case in self.cases
        ]
        eval_rows = [{"case": case.index, "ok": 0, "message": message} for case in self.cases]
        return self._record(case_results, eval_rows)

    def _record(self, case_results: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
        hard = int(bool(eval_rows) and all(row["ok"] for row in eval_rows))
        soft = sum(row["ok"] for row in eval_rows) / len(eval_rows) if eval_rows else 0.0
        return {
            "id": self.task.id,
            "instruction": self.task.instruction,
            "instruction_type": self.task.instruction_type,
            "answer_position": self.task.answer_position,
            "sheet_names": self.inspection.get("sheet_names") or [],
            "target_ranges": self.inspection.get("target_ranges") or [],
            "ok": all(case.get("ok") for case in case_results),
            "cases": case_results,
            "eval": {"test_case_results": [row["ok"] for row in eval_rows], "soft_restriction": soft, "hard_restriction": hard, "messages": eval_rows},
        }
