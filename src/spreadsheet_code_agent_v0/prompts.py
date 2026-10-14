from __future__ import annotations

import json
import re
from typing import Any

from spreadsheetbench_lite.data import Task
from spreadsheetbench_lite.executor import extract_code


UNDERSTANDING_SCHEMA = {
    "where": "task-specific sheet/range to modify",
    "from": "task-specific source data/ranges/information to read",
    "what": "task-specific operation to perform",
    "write_mode": "static_values | formulas | formatting | structure | unknown",
    "risks": [],
    "api_needs": [],
}


SPREADSHEET_SKILL_CARD = """Spreadsheet rules:
- Inspect workbook structure before editing.
- Write exactly to target_ranges unless the instruction says otherwise.
- Prefer static values for value/table outputs; write formulas only when formulas are explicitly requested.
- Preserve unrelated sheets, cells, formulas, and styles.
- For formatting tasks, copy styles without changing values.
- Normalize numeric strings and skip blanks before arithmetic/comparisons.
- Avoid unsupported openpyxl shortcuts.
"""


def build_understanding_prompt(task: Task, preview: str) -> str:
    return f"""Read the spreadsheet task and return compact JSON only.
Do not write code. Do not include chain-of-thought.

Instruction:
{task.instruction}

Instruction type: {task.instruction_type}
Evaluator checks: {task.answer_position}

Workbook preview:
{preview}

Return JSON with exactly these keys. Fill lists with task-specific items only; leave them empty when unsure.
{json.dumps(UNDERSTANDING_SCHEMA, ensure_ascii=False)}
"""


def parse_understanding(text: str) -> dict[str, Any]:
    obj_text = _first_json_object(text)
    if not obj_text:
        return _fallback_understanding(text)
    try:
        data = json.loads(obj_text)
    except json.JSONDecodeError:
        return _fallback_understanding(text)
    return {
        "where": _short_text(data.get("where")),
        "from": _short_text(data.get("from")),
        "what": _short_text(data.get("what")),
        "write_mode": _normalize_write_mode(data.get("write_mode")),
        "risks": _string_list(data.get("risks")),
        "api_needs": _string_list(data.get("api_needs")),
    }


def build_generation_prompt(
    task: Task,
    preview: str,
    understanding: dict[str, Any] | None,
    inspection: str | None,
    api_refs: str | None,
    mode: str,
) -> str:
    parts = [
        "Generate Python/openpyxl spreadsheet transformation code.",
        "Return one Python code block only.",
        "Define exactly: transform(wb) -> None.",
        "Do not load workbooks, save workbooks, read files, or define input_path/output_path handling.",
        "The harness will run: wb = load_workbook(input_path); transform(wb); wb.save(output_path).",
        "Use only openpyxl and Python standard library.",
        "Keep code compact and robust across all test cases for the same task.",
        "Avoid unsupported APIs such as ws.clear(), ws.max_col, cell.bold, delete_rows(columns=...).",
        "When copying cells, copy values/styles explicitly; do not append Cell objects from another sheet.",
        "Handle None, text numbers, dates, and merged cells conservatively.",
        "The checked answer cells are the target_ranges; make sure transform(wb) writes the requested result there when the task asks for values/formulas/tables.",
        "If the output is a value/table result, compute it in Python and write static values; openpyxl cannot calculate newly written Excel formulas for evaluation.",
        "If creating or filtering tables, clear stale target output cells before writing new rows, but keep unrelated sheets/cells intact.",
        "If Focused workbook inspection is provided, trust its sheet_names and target_ranges over malformed raw answer-position text.",
        "Do not invent sheet names; choose from the workbook sheet_names unless creating a new sheet is explicitly required.",
        "",
        SPREADSHEET_SKILL_CARD.strip(),
        "",
        f"Mode: {mode}",
        f"Instruction type: {task.instruction_type}",
        f"Evaluator checks: {task.answer_position}",
        "",
        "Instruction:",
        task.instruction,
        "",
        "Workbook preview:",
        preview,
    ]
    if understanding:
        parts += ["", "Structured understanding JSON:", json.dumps(understanding, ensure_ascii=False, indent=2)]
    if inspection:
        parts += ["", "Focused workbook inspection:", inspection]
    if api_refs:
        parts += ["", "Relevant openpyxl API reference:", api_refs]
    return "\n".join(parts)


def build_repair_prompt(
    task: Task,
    current_code: str,
    diagnostic: dict[str, Any],
    understanding: dict[str, Any] | None,
    inspection: str | None,
    api_refs: str | None,
) -> str:
    parts = [
        "Repair the current Python/openpyxl transform(wb) program.",
        "Do not re-plan from scratch. Fix only the localized cause of the diagnostic.",
        "If the diagnostic is verification_error, fix workbook integrity, Excel error values, formula cache risks, or target sheet/range selection without changing unrelated logic.",
        "For formula_cache_risk in value/table tasks, compute and write static values instead of uncached Excel formulas.",
        "Return one Python code block only containing the full repaired program.",
        "Keep the same contract: define transform(wb) -> None; do not load/save workbooks.",
        "",
        SPREADSHEET_SKILL_CARD.strip(),
        "",
        f"Instruction type: {task.instruction_type}",
        f"Evaluator checks: {task.answer_position}",
        "",
        "Instruction:",
        task.instruction,
        "",
        "Diagnostic JSON:",
        json.dumps(diagnostic, ensure_ascii=False, indent=2)[:6000],
        "",
        "Current code:",
        f"```python\n{current_code}\n```",
    ]
    if understanding:
        parts += ["", "Structured understanding:", json.dumps(understanding, ensure_ascii=False, indent=2)]
    if inspection:
        parts += ["", "Focused workbook context:", inspection[:6000]]
    if api_refs:
        parts += ["", "Relevant API reference:", api_refs[:5000]]
    return "\n".join(parts)


def extract_transform_code(response: str) -> str:
    return extract_code(response).strip()


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for pos in range(start, len(text)):
        ch = text[pos]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    return None


def _fallback_understanding(text: str) -> dict[str, Any]:
    return {"where": "", "from": "", "what": _short_text(text), "write_mode": "unknown", "risks": [], "api_needs": []}


def _normalize_write_mode(value: Any) -> str:
    text = _short_text(value, 80).lower()
    allowed = {"static_values", "formulas", "formatting", "structure", "unknown"}
    if text in allowed:
        return text
    if "formula" in text:
        return "formulas"
    if "format" in text or "style" in text:
        return "formatting"
    if "sheet" in text or "row" in text or "column" in text or "structure" in text:
        return "structure"
    if "value" in text or "table" in text:
        return "static_values"
    return "unknown"


def _short_text(value: Any, limit: int = 500) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


def _string_list(value: Any, limit: int = 12) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = []
    out = []
    for item in items:
        text = _short_text(item, 120)
        if text and text not in out:
            out.append(text)
    return out[:limit]
