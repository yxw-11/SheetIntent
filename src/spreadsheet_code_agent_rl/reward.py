from __future__ import annotations

import re
from typing import Any


def reward_from_record(record: dict[str, Any], profile: str = "hybrid", action_text: str | None = None) -> dict[str, Any]:
    """Compute scalar and component rewards from one task record.

    Profiles:
    - proxy: non-leaky artifact/runtime reward, safe as policy observation.
    - oracle: final evaluator reward, for training reward only.
    - hybrid: mostly evaluator reward plus small execution/artifact shaping.
    """

    eval_info = record.get("eval") or {}
    cases = record.get("cases") or []
    case_count = len(cases)
    exec_ok = sum(1 for case in cases if case.get("ok"))
    exec_case_rate = exec_ok / case_count if case_count else 0.0
    exec_all = float(bool(cases) and exec_ok == case_count)

    verifications = [case.get("verification") or {} for case in cases]
    with_verification = [v for v in verifications if v]
    openable_rate = _mean_bool(v.get("openable") for v in with_verification)
    target_changed_rate = _mean_bool((v.get("changed_target_values") or 0) > 0 for v in with_verification)
    target_not_empty_rate = _mean_bool(not bool(v.get("target_all_empty")) for v in with_verification)
    formula_cache_ok_rate = _mean_bool(not bool(v.get("formula_cache_risk_cells")) for v in with_verification)
    excel_error_free_rate = _mean_bool(not bool(v.get("excel_error_values")) for v in with_verification)

    soft = float(eval_info.get("soft_restriction") or 0.0)
    hard = float(eval_info.get("hard_restriction") or 0.0)
    proxy = (
        0.30 * exec_case_rate
        + 0.20 * openable_rate
        + 0.15 * target_changed_rate
        + 0.15 * target_not_empty_rate
        + 0.10 * formula_cache_ok_rate
        + 0.10 * excel_error_free_rate
    )
    oracle = 0.75 * soft + 0.25 * hard
    grounded_plan = _grounded_plan_reward(record, action_text, exec_case_rate=exec_case_rate) if action_text is not None else 0.0
    if profile == "proxy":
        scalar = (0.95 * proxy + grounded_plan) if action_text is not None else proxy
    elif profile == "oracle":
        scalar = (0.95 * oracle + grounded_plan) if action_text is not None else oracle
    elif profile == "hybrid":
        scalar = (0.82 * oracle + 0.13 * proxy + grounded_plan) if action_text is not None else (0.85 * oracle + 0.15 * proxy)
    else:
        raise ValueError(f"unknown reward profile: {profile}")

    return {
        "profile": profile,
        "scalar": round(float(scalar), 6),
        "oracle": round(float(oracle), 6),
        "proxy": round(float(proxy), 6),
        "components": {
            "eval_soft": soft,
            "eval_hard": hard,
            "exec_case_rate": exec_case_rate,
            "exec_all": exec_all,
            "output_openable_rate": openable_rate,
            "target_changed_rate": target_changed_rate,
            "target_not_empty_rate": target_not_empty_rate,
            "formula_cache_ok_rate": formula_cache_ok_rate,
            "excel_error_free_rate": excel_error_free_rate,
            "grounded_plan": grounded_plan,
        },
    }


def _grounded_plan_reward(record: dict[str, Any], action_text: str | None, exec_case_rate: float) -> float:
    """Tiny gated shaping reward for a compact, grounded pre-code plan.

    This rewards useful spreadsheet grounding, not free-form chain-of-thought.
    It is deliberately capped at 0.05 and mostly disabled when the program does
    not even execute, so the model cannot win by writing nice prose only.
    """

    text = str(action_text or "")
    plan = _extract_plan_text(text)
    if not plan:
        return 0.0

    plan_l = plan.lower()
    words = re.findall(r"[A-Za-z0-9_]+", plan)
    score = 0.0

    if len(words) <= 140:
        score += 0.01

    target_tokens = _target_tokens(record)
    if target_tokens and any(token.lower() in plan_l for token in target_tokens):
        score += 0.01

    sheet_names = [str(name).lower() for name in record.get("sheet_names") or [] if str(name).strip()]
    if sheet_names and any(name in plan_l for name in sheet_names):
        score += 0.01

    instruction = str(record.get("instruction") or "").lower()
    op_terms = _operation_terms(instruction)
    if op_terms and any(term in plan_l for term in op_terms):
        score += 0.01

    if _mentions_read_write(plan_l):
        score += 0.01

    # Preflight/runtime failures should not receive meaningful prose reward.
    cases = record.get("cases") or []
    errors = {str(case.get("error_category") or "") for case in cases}
    if "preflight_error" in errors:
        return 0.0
    if exec_case_rate <= 0.0:
        return min(score, 0.01)
    return round(min(score, 0.05), 6)


def _extract_plan_text(text: str) -> str:
    match = re.search(r"<analysis>\s*(.*?)\s*</analysis>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return _compact(match.group(1))[:1200]
    fence = re.search(r"```(?:python|py)?", text, flags=re.IGNORECASE)
    prefix = text[: fence.start()] if fence else text
    transform = re.search(r"(?m)^\s*def\s+transform\s*\(", prefix)
    if transform:
        prefix = prefix[: transform.start()]
    return _compact(prefix)[:1200]


def _target_tokens(record: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for item in record.get("target_ranges") or []:
        if isinstance(item, dict):
            tokens.extend([str(item.get("sheet") or ""), str(item.get("range") or "")])
    raw = str(record.get("answer_position") or "")
    tokens.extend(re.findall(r"\b[A-Za-z]{1,3}\d+(?::[A-Za-z]{1,3}\d+)?\b", raw))
    out = []
    for token in tokens:
        token = token.strip()
        if token and token not in out:
            out.append(token)
    return out[:16]


def _operation_terms(instruction: str) -> set[str]:
    groups = {
        "sum": ("sum", "total", "subtotal"),
        "average": ("average", "mean"),
        "count": ("count",),
        "filter": ("filter", "where"),
        "sort": ("sort", "ascending", "descending"),
        "format": ("format", "style", "font", "border", "highlight", "color"),
        "delete": ("delete", "remove", "clear"),
        "insert": ("insert", "add row", "add column"),
        "copy": ("copy", "duplicate"),
        "lookup": ("lookup", "match", "vlookup", "xlookup"),
        "formula": ("formula", "calculate", "compute"),
        "date": ("date", "month", "year", "day"),
    }
    terms = set()
    for canonical, variants in groups.items():
        if any(term in instruction for term in variants):
            terms.add(canonical)
            terms.update(variants[:2])
    return terms


def _mentions_read_write(plan_l: str) -> bool:
    read_words = ("read", "source", "from", "scan", "iterate", "header", "column", "range")
    write_words = ("write", "target", "output", "set", "fill", "cell", "range")
    return any(word in plan_l for word in read_words) and any(word in plan_l for word in write_words)


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _mean_bool(values) -> float:
    items = [bool(value) for value in values]
    if not items:
        return 0.0
    return sum(1 for value in items if value) / len(items)
