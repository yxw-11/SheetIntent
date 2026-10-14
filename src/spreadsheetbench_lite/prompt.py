from __future__ import annotations

from .data import Task


SYSTEM_PROMPT = """You write compact, executable Python for Excel spreadsheet manipulation.
Use openpyxl only. The code must generalize to all test cases for the same task."""


def build_prompt(task: Task, preview: str, cot: bool = False) -> str:
    """Build the single-generation prompt for one SpreadsheetBench V1 task."""

    reasoning_instruction = _cot_instruction() if cot else _direct_instruction()
    return f"""Solve this SpreadsheetBench V1 spreadsheet task.

Instruction:
{task.instruction}

Instruction type: {task.instruction_type}
Evaluator checks: {task.answer_position}

Workbook preview from the first test case:
{preview}

Reasoning mode:
{reasoning_instruction}

Output requirements:
- Return one Python code block only.
- Define solve(input_path: str, output_path: str) -> None.
- Use openpyxl only; do not use pandas or unavailable third-party packages.
- Keep code concise; avoid long explanations and long comments inside code.
- Preserve unrelated cells, sheets, formulas, and formatting unless the instruction requires changing them.
- Do not read answer files, do not hard-code preview values, and do not assume exact test-case values.
"""


def _direct_instruction() -> str:
    return "Do not show reasoning. Output the final code directly."


def _cot_instruction() -> str:
    return "Think briefly internally, avoid repeated analysis, then output only the final code. Do not print the chain of thought."
