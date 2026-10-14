from __future__ import annotations
import re
from pathlib import Path
from typing import Any
import openpyxl
from spreadsheetbench_lite.evaluate_v1 import compare_value, expand_cell_range, parse_answer_positions

def score_record(
    record: dict[str, Any],
    action_text: str,
    strategy: str = "outcome",
    progress: dict[str, Any] | None = None,
    grounding: float = 0.0,
) -> dict[str, Any]:
    """Score one deterministic harness record without exposing oracle data."""

    eval_info = record.get("eval") or {}
    cases = record.get("cases") or []
    soft = float(eval_info.get("soft_restriction") or 0.0)
    hard = float(eval_info.get("hard_restriction") or 0.0)
    oracle = 0.75 * soft + 0.25 * hard
    exec_rate = _mean(bool(case.get("ok")) for case in cases)
    quality = _artifact_quality(cases)
    stage, error_penalty = _failure_stage(cases, exec_rate, oracle)
    format_bonus = _format_bonus(action_text) if exec_rate > 0 else 0.0
    semantic_progress = float((progress or {}).get("score", oracle))
    semantic_exact = bool((progress or {}).get("exact", hard >= 1.0))
    format_quality = float(format_bonus > 0.0)

    if strategy == "outcome":
        # Wrong-but-executable code is capped at 0.05; correctness dominates.
        scalar = 0.90 * oracle + 0.06 * exec_rate * oracle + 0.04 * quality * oracle
    elif strategy == "diagnostic":
        # Dense stage feedback helps when every sample in a GRPO group is wrong.
        if oracle > 0:
            scalar = 0.82 * oracle + 0.10 * exec_rate + 0.06 * quality + format_bonus
        elif exec_rate > 0:
            scalar = 0.08 + 0.04 * quality + format_bonus
        else:
            scalar = error_penalty
    elif strategy == "diagnostic_aligned":
        # Keep the strongest diagnostic recipe, but judge correctness only on
        # the evaluator's target ranges. Partial cell matches densify reward.
        target_oracle = 0.75 * semantic_progress + 0.25 * float(semantic_exact)
        stage, error_penalty = _failure_stage(cases, exec_rate, target_oracle)
        if semantic_exact and exec_rate >= 1.0:
            scalar = 1.0
            stage = "correct"
        elif target_oracle > 0:
            scalar = 0.82 * target_oracle + 0.10 * exec_rate + 0.06 * quality + format_bonus
            scalar = min(scalar, 0.99)
            stage = "target_progress"
        elif exec_rate > 0:
            scalar = 0.08 + 0.04 * quality + format_bonus
            stage = "executable_wrong"
        else:
            scalar = error_penalty
    elif strategy == "semantic_aligned":
        # Correctness uses the benchmark evaluator; auxiliary signals remain
        # too small for an executable-but-wrong rollout to look competitive.
        if semantic_exact and hard >= 1.0 and exec_rate >= 1.0:
            scalar = 1.0
            stage = "correct"
        elif oracle > 0.0 or semantic_progress > 0.0:
            scalar = (
                0.70 * oracle
                + 0.15 * semantic_progress
                + 0.08 * exec_rate
                + 0.05 * grounding
                + 0.02 * format_quality
            )
            scalar = min(scalar, 0.99)
            stage = "semantic_progress"
        elif exec_rate > 0.0:
            scalar = 0.02 * exec_rate + 0.01 * grounding + 0.005 * format_quality
            stage = "executable_wrong"
        else:
            scalar = error_penalty
    elif strategy == "ours_semantic":
        # Execution is already reliable; most reward should measure whether the
        # required edits are correct and whether unrelated edits were avoided.
        if semantic_exact and exec_rate >= 1.0:
            scalar = 1.0
            stage = "correct"
        elif semantic_progress > 0:
            scalar = 0.94 * semantic_progress + 0.02 * exec_rate + min(format_bonus, 0.005)
            stage = "semantic_progress"
        elif exec_rate > 0:
            scalar = 0.01 + min(format_bonus, 0.005)
            stage = "executable_wrong"
        else:
            scalar = 0.5 * error_penalty
    elif strategy == "binary_outcome":
        # Controlled Spreadsheet-RL-style ablation: exact final state or zero.
        scalar = float(semantic_exact and exec_rate >= 1.0)
        stage = "correct" if scalar else ("executable_wrong" if exec_rate > 0 else stage)
    else:
        raise ValueError(f"unknown reward strategy: {strategy}")

    scalar = round(max(-0.15, min(1.0, float(scalar))), 6)
    reported_oracle = (
        float(semantic_exact)
        if strategy
        in {
            "ours_semantic",
            "binary_outcome",
            "diagnostic_aligned",
            "semantic_aligned",
            "repair_incremental",
            "repair_delta_gated",
        }
        else oracle
    )
    return {
        "strategy": strategy,
        "scalar": scalar,
        "oracle": round(reported_oracle, 6),
        "stage": stage,
        "components": {
            "legacy_eval_oracle": round(oracle, 6),
            "eval_soft": soft,
            "eval_hard": hard,
            "exec_rate": round(exec_rate, 6),
            "artifact_quality": round(quality, 6),
            "semantic_progress": round(semantic_progress, 6),
            "semantic_exact": semantic_exact,
            "expected_deltas": int((progress or {}).get("expected", 0)),
            "predicted_deltas": int((progress or {}).get("predicted", 0)),
            "matched_deltas": int((progress or {}).get("matched", 0)),
            "target_progress": round(semantic_progress, 6) if strategy in {"diagnostic_aligned", "semantic_aligned"} else None,
            "target_exact": semantic_exact if strategy in {"diagnostic_aligned", "semantic_aligned"} else None,
            "target_cells": int((progress or {}).get("target_cells", 0)),
            "matched_target_cells": int((progress or {}).get("matched_target_cells", 0)),
            "format_bonus": format_bonus,
            "grounding_consistency": round(grounding, 6),
            "error_penalty": error_penalty,
        },
    }


def _target_region_progress(cases: list[Any], answer_position: str) -> dict[str, Any]:
    """Measure target-only value edits without rewarding unchanged source cells."""

    case_scores: list[float] = []
    target_cells = 0
    matched_cells = 0
    expected_edits = 0
    predicted_edits = 0
    matched_edits = 0
    exact = bool(cases)

    for case in cases:
        if case.answer_path is None or not case.answer_path.exists() or not case.output_path.exists():
            case_scores.append(0.0)
            exact = False
            continue

        source_wb = answer_wb = output_wb = None
        try:
            source_wb = openpyxl.load_workbook(case.input_path, data_only=True)
            answer_wb = openpyxl.load_workbook(case.answer_path, data_only=True)
            output_wb = openpyxl.load_workbook(case.output_path, data_only=True)
            positions = _target_positions(answer_wb, answer_position)
            case_total = 0
            case_matched = 0
            case_expected = 0
            case_predicted = 0
            case_edit_matched = 0

            for sheet_name, cell_names in positions:
                case_total += len(cell_names)
                answer_ws = answer_wb[sheet_name]
                source_ws = source_wb[sheet_name] if sheet_name in source_wb.sheetnames else None
                output_ws = output_wb[sheet_name] if sheet_name in output_wb.sheetnames else None
                for cell_name in cell_names:
                    source_value = source_ws[cell_name].value if source_ws is not None else None
                    answer_value = answer_ws[cell_name].value
                    output_value = output_ws[cell_name].value if output_ws is not None else None
                    expected = not compare_value(source_value, answer_value)
                    predicted = not compare_value(source_value, output_value)
                    matched = expected and compare_value(answer_value, output_value)
                    case_expected += int(expected)
                    case_predicted += int(predicted)
                    case_edit_matched += int(matched)
                    case_matched += int(compare_value(answer_value, output_value))

            if case_expected == 0:
                score = float(case_total > 0 and case_matched == case_total)
            elif case_predicted == 0 or case_edit_matched == 0:
                score = 0.0
            else:
                precision = case_edit_matched / case_predicted
                recall = case_edit_matched / case_expected
                score = 2.0 * precision * recall / (precision + recall)
            case_scores.append(score)
            target_cells += case_total
            matched_cells += case_matched
            expected_edits += case_expected
            predicted_edits += case_predicted
            matched_edits += case_edit_matched
            exact = exact and case_total > 0 and case_matched == case_total
        except Exception:
            case_scores.append(0.0)
            exact = False
        finally:
            if source_wb is not None:
                source_wb.close()
            if answer_wb is not None:
                answer_wb.close()
            if output_wb is not None:
                output_wb.close()

    return {
        "score": sum(case_scores) / len(case_scores) if case_scores else 0.0,
        "exact": exact and len(case_scores) == len(cases),
        "target_cells": target_cells,
        "matched_target_cells": matched_cells,
        "expected": expected_edits,
        "predicted": predicted_edits,
        "matched": matched_edits,
    }


def _target_positions(workbook: openpyxl.Workbook, answer_position: str) -> list[tuple[str, list[str]]]:
    if answer_position not in ("", "__FULL_WORKBOOK__"):
        return [
            (sheet_name, expand_cell_range(cell_range, workbook[sheet_name]))
            for sheet_name, cell_range in parse_answer_positions(workbook, answer_position)
        ]

    positions = []
    for worksheet in workbook.worksheets:
        cell_range = f"A1:{worksheet.cell(worksheet.max_row, worksheet.max_column).coordinate}"
        positions.append((worksheet.title, expand_cell_range(cell_range, worksheet)))
    return positions


def _artifact_quality(cases: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for case in cases:
        verification = case.get("verification") or {}
        if not verification:
            continue
        checks = (
            bool(verification.get("openable")),
            (verification.get("changed_target_values") or 0) > 0,
            not bool(verification.get("target_all_empty")),
            not bool(verification.get("formula_cache_risk_cells")),
            not bool(verification.get("excel_error_values")),
        )
        values.append(sum(checks) / len(checks))
    return sum(values) / len(values) if values else 0.0


def _failure_stage(cases: list[dict[str, Any]], exec_rate: float, oracle: float) -> tuple[str, float]:
    if oracle >= 1.0:
        return "correct", 0.0
    if oracle > 0.0:
        return "partially_correct", 0.0
    if exec_rate > 0.0:
        return "executable_wrong", 0.0

    categories = {str(case.get("error_category") or "").lower() for case in cases}
    if "preflight_error" in categories:
        return "preflight_error", -0.12
    if "syntax_error" in categories:
        return "syntax_error", -0.10
    if categories & {"attribute_error", "sheet_error", "range_error", "merged_cell_error"}:
        return "grounding_or_api_error", -0.08
    if categories & {"type_error", "value_error", "index_error"}:
        return "data_type_error", -0.06
    return "runtime_error", -0.08


def _format_bonus(text: str) -> float:
    """Tiny, gated bonus for a compact plan plus the required action shape."""

    has_transform = bool(re.search(r"(?m)^\s*def\s+transform\s*\(\s*wb\s*\)\s*:", text))
    match = re.search(r"<analysis>\s*(.*?)\s*</analysis>", text, flags=re.DOTALL | re.IGNORECASE)
    if not has_transform or not match:
        return 0.0
    plan = re.sub(r"\s+", " ", match.group(1)).strip()
    grounded = bool(re.search(r"\b(sheet|cell|range|column|row|read|write|formula)\b", plan, re.IGNORECASE))
    return 0.01 if grounded and 1 <= len(plan.split()) <= 120 else 0.0


def _mean(values) -> float:
    items = [bool(value) for value in values]
    return sum(items) / len(items) if items else 0.0
