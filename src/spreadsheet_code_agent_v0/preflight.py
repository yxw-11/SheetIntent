from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

from openpyxl.utils.cell import get_column_letter, range_boundaries


@dataclass
class PreflightIssue:
    severity: str
    category: str
    message: str
    line: int | None = None
    snippet: str = ""


@dataclass
class PreflightResult:
    ok: bool
    errors: list[PreflightIssue] = field(default_factory=list)
    warnings: list[PreflightIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [issue.__dict__ for issue in self.errors],
            "warnings": [issue.__dict__ for issue in self.warnings],
        }


def run_preflight(code: str, inspection: dict[str, Any] | None = None) -> PreflightResult:
    errors: list[PreflightIssue] = []
    warnings: list[PreflightIssue] = []
    lines = code.splitlines()
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        errors.append(PreflightIssue("error", "syntax_error", str(exc), exc.lineno, _line(lines, exc.lineno)))
        return PreflightResult(False, errors, warnings)

    sheet_names = set((inspection or {}).get("sheet_names") or [])
    merged_cells = _merged_cells(inspection)
    mixed_type_cols = _mixed_type_columns(inspection)
    if not any(isinstance(n, ast.FunctionDef) and n.name == "transform" for n in tree.body):
        errors.append(PreflightIssue("error", "missing_transform", "Code must define transform(wb) -> None."))

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr == "max_col":
                errors.append(_issue("error", "bad_openpyxl_attribute", "Worksheet has max_column, not max_col.", node, lines))
            elif node.attr == "clear":
                errors.append(_issue("error", "bad_openpyxl_attribute", "Worksheet has no clear(); delete rows/cols or clear cell values explicitly.", node, lines))
            elif node.attr == "bold" and not _is_font_bold(node):
                errors.append(_issue("error", "bad_cell_style_attribute", "Cell has no bold attribute; use cell.font or Font(bold=True).", node, lines))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            _check_call(node, lines, errors)
        if isinstance(node, ast.Subscript):
            _check_subscript(node, lines, sheet_names, errors)
        if isinstance(node, ast.Assign):
            _check_merged_assignment(node, lines, merged_cells, warnings)
        if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add) and mixed_type_cols:
            warnings.append(_issue("warning", "mixed_type_risk", "Workbook has mixed numeric/string columns; normalize values before +=.", node, lines))
        if isinstance(node, ast.Compare) and mixed_type_cols:
            warnings.append(_issue("warning", "mixed_type_compare_risk", "Workbook has mixed numeric/string/date columns; guard comparisons.", node, lines))
    return PreflightResult(not errors, errors, warnings[:20])


def _check_call(node: ast.Call, lines: list[str], errors: list[PreflightIssue]) -> None:
    name = node.func.attr
    keywords = {kw.arg for kw in node.keywords if kw.arg}
    if name == "delete_rows" and "columns" in keywords:
        errors.append(_issue("error", "bad_openpyxl_signature", "delete_rows(idx, amount=1) does not accept columns=.", node, lines))
    if name == "delete_cols" and "rows" in keywords:
        errors.append(_issue("error", "bad_openpyxl_signature", "delete_cols(idx, amount=1) does not accept rows=.", node, lines))
    if name in {"insert_rows", "delete_rows", "insert_cols", "delete_cols"} and keywords & {"column", "columns", "row", "rows"}:
        errors.append(_issue("error", "suspicious_row_col_keyword", f"{name} expects idx/amount integer arguments.", node, lines))


def _check_subscript(node: ast.Subscript, lines: list[str], sheet_names: set[str], errors: list[PreflightIssue]) -> None:
    value = _constant_string(node.slice)
    if not value:
        return
    if isinstance(node.value, ast.Name) and node.value.id == "wb":
        if sheet_names and value not in sheet_names:
            errors.append(_issue("error", "sheet_error", f"Workbook has no sheet named {value!r}.", node, lines))
        return
    if _looks_like_cell_or_range(value):
        try:
            range_boundaries(value)
        except Exception as exc:
            errors.append(_issue("error", "range_error", f"Invalid Excel coordinate/range {value!r}: {exc}", node, lines))



def _looks_like_cell_or_range(value: str) -> bool:
    return bool(re.fullmatch(r"[$]?[A-Za-z]{1,3}[$]?\d+(:[$]?[A-Za-z]{1,3}[$]?\d+)?", value.strip()))

def _check_merged_assignment(node: ast.Assign, lines: list[str], merged_cells: set[str], warnings: list[PreflightIssue]) -> None:
    for target in node.targets:
        cell = _assigned_cell_coordinate(target)
        if cell and cell in merged_cells:
            warnings.append(_issue("warning", "merged_cell_risk", f"Assignment targets merged non-anchor cell {cell}; unmerge or write top-left cell.", target, lines))


def _assigned_cell_coordinate(target: ast.AST) -> str | None:
    if not isinstance(target, ast.Attribute) or target.attr != "value":
        return None
    sub = target.value
    if not isinstance(sub, ast.Subscript):
        return None
    value = _constant_string(sub.slice)
    if value and re.fullmatch(r"[A-Za-z]{1,3}\d+", value):
        return value.upper()
    return None


def _merged_cells(inspection: dict[str, Any] | None) -> set[str]:
    cells = set()
    for sheet in (inspection or {}).get("sheets", []):
        for rng in sheet.get("merged_ranges", []):
            try:
                min_col, min_row, max_col, max_row = range_boundaries(str(rng))
            except Exception:
                continue
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    if (row, col) != (min_row, min_col):
                        cells.add(f"{get_column_letter(col)}{row}")
    return cells


def _mixed_type_columns(inspection: dict[str, Any] | None) -> set[str]:
    mixed = set()
    for sheet in (inspection or {}).get("sheets", []):
        for col in sheet.get("type_profile", []):
            types = {k for k, v in (col.get("types") or {}).items() if v and k != "blank"}
            if len(types) >= 2 and ({"number", "numeric_string", "string", "date"} & types):
                mixed.add(f"{sheet.get('name')}!{col.get('col')}")
    return mixed


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip()
    return None


def _is_font_bold(node: ast.Attribute) -> bool:
    return isinstance(node.value, ast.Attribute) and node.value.attr == "font"


def _issue(severity: str, category: str, message: str, node: ast.AST, lines: list[str]) -> PreflightIssue:
    lineno = getattr(node, "lineno", None)
    return PreflightIssue(severity, category, message, lineno, _line(lines, lineno))


def _line(lines: list[str], lineno: int | None) -> str:
    if not lineno or lineno < 1 or lineno > len(lines):
        return ""
    return lines[lineno - 1].strip()[:240]
