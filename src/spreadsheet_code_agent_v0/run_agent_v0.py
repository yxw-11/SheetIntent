from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from tqdm import tqdm

from spreadsheetbench_lite.data import discover_test_cases, load_tasks
from spreadsheetbench_lite.evaluate_v1 import compare_workbooks
from spreadsheetbench_lite.excel_preview import workbook_preview
from spreadsheetbench_lite.llm import LLMConfig, chat_completion

from .api_lookup import lookup_openpyxl_apis, render_api_refs
from .inspection import inspect_workbook, render_inspection
from .preflight import run_preflight
from .prompts import build_generation_prompt, build_repair_prompt, build_understanding_prompt, extract_transform_code, parse_understanding
from .runtime import preflight_diagnostic, run_transform_code, runtime_diagnostic

MODES = ("direct", "understand", "api", "preflight", "repair", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training-free Spreadsheet Code Agent V0 for SpreadsheetBench V1.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--outputs-dir", type=Path, default=None)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8001/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--understanding-model", type=str, default=None, help="Optional model used only for structured understanding.")
    parser.add_argument("--understanding-base-url", type=str, default=None)
    parser.add_argument("--repair-model", type=str, default=None, help="Optional model used only for localized repair.")
    parser.add_argument("--repair-base-url", type=str, default=None)
    parser.add_argument(
        "--replay-source-dir",
        type=Path,
        default=None,
        help="Reuse understanding and *_initial.py code from a completed run; only execute/repair again.",
    )
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=12288)
    parser.add_argument("--understanding-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--enable-thinking", action="store_true", help="Enable Qwen hidden thinking for every LLM stage, including understanding.")
    parser.add_argument("--cot", action="store_true", help="Enable Qwen hidden thinking for code generation and repair only.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-exists", choices=["dedupe", "append", "fail"], default="dedupe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flags = _mode_flags(args.mode)
    dataset_dir = args.dataset_dir.resolve()
    requested_outputs_dir = (args.outputs_dir or dataset_dir / "outputs" / f"agent_v0_{args.mode}").resolve()
    outputs_dir = _resolve_outputs_dir(requested_outputs_dir, args.output_exists)
    if outputs_dir != requested_outputs_dir:
        print(json.dumps({"requested_outputs_dir": str(requested_outputs_dir), "actual_outputs_dir": str(outputs_dir)}, ensure_ascii=False))

    dirs = _make_dirs(outputs_dir)
    replay_source_dir = args.replay_source_dir.resolve() if args.replay_source_dir else None
    if replay_source_dir and not (replay_source_dir / "codes").is_dir():
        raise FileNotFoundError(f"replay source has no codes directory: {replay_source_dir}")

    # Keep understanding terse by default; COT is useful mainly for code generation
    # and localized repair, where extra reasoning may improve spreadsheet logic.
    llm_config = LLMConfig(
        args.model,
        args.base_url,
        args.api_key,
        args.temperature,
        args.max_tokens,
        disable_thinking=not (args.enable_thinking or args.cot),
    )
    understand_config = replace(
        llm_config,
        model=args.understanding_model or args.model,
        base_url=args.understanding_base_url or args.base_url,
        max_tokens=min(args.understanding_tokens, args.max_tokens),
        temperature=0.0,
        disable_thinking=not args.enable_thinking,
    )
    repair_config = replace(
        llm_config,
        model=args.repair_model or args.model,
        base_url=args.repair_base_url or args.base_url,
    )

    tasks = load_tasks(dataset_dir)[args.start :]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    result_log = outputs_dir / "run_results.jsonl"
    conversation_log = outputs_dir / "conversations.jsonl"
    processed_records = _load_existing_records(result_log) if args.resume else []
    completed_ids = {str(record.get("id")) for record in processed_records}
    if args.resume and completed_ids:
        tasks = [task for task in tasks if task.id not in completed_ids]
        print(json.dumps({"resume": True, "loaded_records": len(processed_records), "remaining_tasks": len(tasks)}, ensure_ascii=False))
    total_selected = len(processed_records) + len(tasks)

    for completed, task in enumerate(tqdm(tasks, desc=f"agent_v0:{args.mode}"), start=len(processed_records) + 1):
        cases = discover_test_cases(dataset_dir, outputs_dir, task)
        if not cases:
            record = _task_error_record(task, "no input workbooks found")
            _record(outputs_dir, dirs, result_log, processed_records, record, completed, total_selected, args.checkpoint_every)
            continue

        preview = workbook_preview(cases[0].input_path, rows=args.rows)
        log: dict[str, Any] = {
            "id": task.id,
            "mode": args.mode,
            "llm_calls": 0,
            "repairs": [],
            "stage_models": {
                "understanding": understand_config.model,
                "generation": llm_config.model,
                "repair": repair_config.model,
            },
        }
        replay = _load_replay_source(replay_source_dir, task.id, flags) if replay_source_dir else None
        if replay:
            replay_label = f"replay:{replay_source_dir.name}"
            log["replay_source"] = str(replay_source_dir)
            log["stage_models"]["understanding"] = replay_label
            log["stage_models"]["generation"] = replay_label
        understanding: dict[str, Any] | None = None
        inspection: dict[str, Any] | None = None
        inspection_text = ""
        api_refs: list[dict[str, str]] = []
        api_text = ""

        try:
            if flags["understand"]:
                if replay:
                    understanding = replay["understanding"]
                else:
                    understanding_result = chat_completion(build_understanding_prompt(task, preview), understand_config)
                    log["llm_calls"] += 1
                    understanding = parse_understanding(understanding_result.content)
                    _write_raw(dirs["raw"], task.id, "understanding", understanding_result.raw_response)
                    _append_jsonl(conversation_log, _conversation(task, "understanding", understanding_result, None))
                log["understanding"] = understanding
                stage = "understanding_replay" if replay else "understanding"
                _append_trajectory(log, stage, {"instruction": task.instruction, "answer_position": task.answer_position}, {"type": "json", "content": understanding}, {})

            if flags["inspect"]:
                inspection = inspect_workbook(cases[0].input_path, understanding, task.answer_position, rows=max(args.rows, 5))
                inspection_text = render_inspection(inspection)
                log["inspection"] = inspection

            if flags["api"]:
                api_refs = lookup_openpyxl_apis((understanding or {}).get("api_needs", []))
                api_text = render_api_refs(api_refs)
                log["api_refs"] = api_refs

            if replay:
                code = replay["initial_code"]
            else:
                gen_prompt = build_generation_prompt(task, preview, understanding, inspection_text, api_text, args.mode)
                gen_result = chat_completion(gen_prompt, llm_config)
                log["llm_calls"] += 1
                _write_raw(dirs["raw"], task.id, "generation", gen_result.raw_response)
                _append_jsonl(conversation_log, _conversation(task, "generation", gen_result, gen_prompt))
                code = extract_transform_code(gen_result.content)
            _append_trajectory(
                log,
                "generation_replay" if replay else "generation",
                {"mode": args.mode, "has_understanding": bool(understanding), "has_inspection": bool(inspection_text), "has_api_refs": bool(api_text)},
                {"type": "code", "content": code},
                {},
            )
            initial_code = code
            (dirs["codes"] / f"{task.id}_initial.py").write_text(initial_code + "\n", encoding="utf-8")

            final_record, final_code = _run_with_optional_repair(
                task,
                cases,
                code,
                args,
                flags,
                inspection,
                understanding,
                inspection_text,
                api_text,
                repair_config,
                dirs,
                conversation_log,
                log,
            )
            (dirs["codes"] / f"{task.id}.py").write_text(final_code + "\n", encoding="utf-8")
            log["initial_generated_code"] = initial_code
            log["final_code_path"] = str(dirs["codes"] / f"{task.id}.py")
            log["final_execution_status"] = final_record.get("ok")
            log["eval"] = final_record.get("eval")
            record = final_record
        except Exception as exc:
            # Infrastructure failures must not become permanently cached model failures.
            from spreadsheet_agent.sandbox import InfrastructureError
            if isinstance(exc, InfrastructureError):
                raise
            record = _task_error_record(task, f"agent error: {type(exc).__name__}: {exc}", cases)
            log["agent_error"] = record["error"]

        (dirs["logs"] / f"{task.id}.json").write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        _record(outputs_dir, dirs, result_log, processed_records, record, completed, total_selected, args.checkpoint_every)

    _write_checkpoint(outputs_dir, dirs["checkpoints"], processed_records, len(processed_records), total_selected, final=True)


def _load_replay_source(source_dir: Path, task_id: str, flags: dict[str, bool]) -> dict[str, Any]:
    code_path = source_dir / "codes" / f"{task_id}_initial.py"
    log_path = source_dir / "agent_logs" / f"{task_id}.json"
    if not code_path.is_file() or not log_path.is_file():
        raise FileNotFoundError(f"replay source is incomplete for task {task_id}")
    source_log = json.loads(log_path.read_text(encoding="utf-8"))
    understanding = source_log.get("understanding")
    if flags["understand"] and not isinstance(understanding, dict):
        raise ValueError(f"replay source has no structured understanding for task {task_id}")
    return {
        "initial_code": code_path.read_text(encoding="utf-8").strip(),
        "understanding": understanding,
    }


def _run_with_optional_repair(task, cases, code, args, flags, inspection, understanding, inspection_text, api_text, llm_config, dirs, conversation_log, log):
    max_repairs = args.max_repairs if flags["repair"] else 0
    record = None
    for attempt in range(max_repairs + 1):
        # Failed final preflight must not leave outputs from an earlier program.
        for case in cases:
            case.output_path.unlink(missing_ok=True)
        preflight_dict = None
        if flags["preflight"]:
            preflight = run_preflight(code, inspection)
            preflight_dict = preflight.to_dict()
            log.setdefault("preflight", []).append({"attempt": attempt, **preflight_dict})
            if not preflight.ok:
                if attempt < max_repairs:
                    code = _repair(task, code, preflight_diagnostic(preflight_dict), understanding, inspection_text, api_text, llm_config, dirs, conversation_log, log, attempt)
                    continue
                return _preflight_failed_record(task, cases, preflight_dict), code

        record = _execute_and_evaluate(task, cases, code, dirs, args.timeout)
        trigger = _first_repair_trigger(record, code)
        compact = _compact_record(record)
        log.setdefault("runtime_attempts", []).append({"attempt": attempt, "record": compact, "preflight": preflight_dict})
        _append_trajectory(
            log,
            f"execute_attempt_{attempt}",
            {"preflight": preflight_dict},
            {"type": "code", "content": code},
            {"runtime_eval": compact},
            _reward_proxy(record),
        )
        if trigger and attempt < max_repairs:
            code = _repair(task, code, trigger["diagnostic"], understanding, inspection_text, api_text, llm_config, dirs, conversation_log, log, attempt)
            continue
        return record, code
    return record or _task_error_record(task, "unknown agent failure", cases), code


def _repair(task, code, diagnostic, understanding, inspection_text, api_text, llm_config, dirs, conversation_log, log, attempt):
    prompt = build_repair_prompt(task, code, diagnostic, understanding, inspection_text, api_text)
    result = chat_completion(prompt, llm_config)
    repaired = extract_transform_code(result.content)
    _append_trajectory(log, f"repair_{attempt + 1}", {"diagnostic": diagnostic}, {"type": "code", "content": repaired}, {})
    log["llm_calls"] += 1
    log["repairs"].append({"attempt": attempt + 1, "diagnostic": diagnostic, "finish_reason": result.finish_reason})
    _write_raw(dirs["raw"], task.id, f"repair_{attempt + 1}", result.raw_response)
    _append_jsonl(conversation_log, _conversation(task, f"repair_{attempt + 1}", result, prompt))
    return repaired or code


def _execute_and_evaluate(task, cases, code, dirs, timeout):
    case_results = []
    eval_results = []
    runtime_objects = []
    for case in cases:
        result = run_transform_code(code, case.input_path, case.output_path, dirs["exec"] / task.id / str(case.index), timeout, task.answer_position)
        runtime_objects.append(result)
        case_results.append({
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
        })
        eval_ok, eval_message = _safe_compare(case.answer_path, case.output_path, task.answer_position)
        eval_results.append({"case": case.index, "ok": int(eval_ok), "message": eval_message})
    record = {
        "id": task.id,
        "instruction_type": task.instruction_type,
        "answer_position": task.answer_position,
        "ok": all(item["ok"] for item in case_results),
        "cases": case_results,
        "eval": _summarize_eval(eval_results),
        "_runtime_objects": runtime_objects,
    }
    return record


def _first_repair_trigger(record: dict[str, Any], code: str) -> dict[str, Any] | None:
    runtime_objects = record.pop("_runtime_objects", [])
    for case, result in zip(record.get("cases", []), runtime_objects):
        if not case.get("ok"):
            diagnostic = runtime_diagnostic(result, code)
            diagnostic["case"] = case.get("case")
            return {"kind": "runtime", "diagnostic": diagnostic}

    for case in record.get("cases", []):
        verification = case.get("verification") or {}
        # Be conservative: value-preserving edits can be correct, and some tasks
        # intentionally clear cells. Use verification to repair only structural
        # output problems that are very unlikely to be valid answers.
        formula_cache_risk = bool(verification.get("formula_cache_risk_cells"))
        structural_problem = (
            bool(verification.get("excel_error_values"))
            or bool(verification.get("open_error"))
            or bool(verification.get("position_error"))
        )
        if formula_cache_risk or structural_problem:
            if formula_cache_risk:
                message = "Program ran, but target cells contain newly written formulas without cached values. For value/table outputs, compute and write static values instead of uncached formulas."
            else:
                message = "Program ran, but output verification found a structural problem. Fix output workbook integrity, target sheet/range selection, or Excel error values."
            return {
                "kind": "verification",
                "diagnostic": {
                    "error_type": "verification_error",
                    "message": message,
                    "case": case.get("case"),
                    "verification": verification,
                    "offending_code": "No crash. Inspect transform(wb) target writes and sheet/range selection.",
                },
            }
    return None


def _preflight_failed_record(task, cases, preflight_dict):
    message = json.dumps(preflight_dict.get("errors", [])[:3], ensure_ascii=False)
    eval_results = [{"case": case.index, "ok": 0, "message": "preflight failed: " + message[:1000]} for case in cases]
    return {
        "id": task.id,
        "instruction_type": task.instruction_type,
        "answer_position": task.answer_position,
        "ok": False,
        "preflight": preflight_dict,
        "cases": [{"case": case.index, "input": str(case.input_path), "output": str(case.output_path), "ok": False, "returncode": None, "timed_out": False, "error_category": "preflight_error", "stdout": "", "stderr": message[:4000]} for case in cases],
        "eval": _summarize_eval(eval_results),
    }


def _mode_flags(mode: str) -> dict[str, bool]:
    rank = MODES.index(mode)
    return {
        "understand": rank >= MODES.index("understand"),
        "inspect": rank >= MODES.index("understand"),
        "api": rank >= MODES.index("api"),
        "preflight": rank >= MODES.index("preflight"),
        "repair": rank >= MODES.index("repair"),
    }


def _make_dirs(outputs_dir: Path) -> dict[str, Path]:
    dirs = {"codes": outputs_dir / "codes", "exec": outputs_dir / "exec", "raw": outputs_dir / "raw_responses", "logs": outputs_dir / "agent_logs", "checkpoints": outputs_dir / "checkpoints"}
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_raw(raw_dir: Path, task_id: str, stage: str, payload: dict[str, Any]) -> None:
    (raw_dir / f"{task_id}_{stage}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _conversation(task, stage: str, result, prompt: str | None) -> dict[str, Any]:
    return {"id": task.id, "stage": stage, "instruction_type": task.instruction_type, "answer_position": task.answer_position, "prompt": prompt, "response": result.content, "reasoning_content": result.reasoning_content, "finish_reason": result.finish_reason, "usage": result.usage}


def _safe_compare(answer_path: Path | None, output_path: Path, answer_position: str) -> tuple[bool, str]:
    try:
        return compare_workbooks(answer_path, output_path, answer_position)
    except Exception as exc:
        return False, f"evaluation error: {type(exc).__name__}: {exc}"


def _summarize_eval(case_scores: list[dict]) -> dict:
    hard = int(bool(case_scores) and all(item["ok"] for item in case_scores))
    soft = sum(item["ok"] for item in case_scores) / len(case_scores) if case_scores else 0.0
    return {"test_case_results": [item["ok"] for item in case_scores], "soft_restriction": soft, "hard_restriction": hard, "messages": case_scores}


def _append_trajectory(log: dict[str, Any], stage: str, observation: dict[str, Any], action: dict[str, Any], reward_proxy: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    item = {
        "step": len(log.setdefault("trajectory", [])),
        "stage": stage,
        "observation": observation,
        "action": action,
        "reward_proxy": reward_proxy,
    }
    if extra:
        item.update(extra)
    log["trajectory"].append(item)


def _reward_proxy(record: dict[str, Any]) -> dict[str, Any]:
    cases = record.get("cases", [])
    verifications = [case.get("verification") or {} for case in cases]
    formula_risks = sum(len(v.get("formula_cache_risk_cells") or []) for v in verifications)
    excel_errors = sum(len(v.get("excel_error_values") or []) for v in verifications)
    openable = [bool(v.get("openable")) for v in verifications if v]
    changed = [v.get("changed_target_values") for v in verifications if v.get("changed_target_values") is not None]
    return {
        "exec_ok": int(bool(cases) and all(case.get("ok") for case in cases)),
        "any_exec_ok": int(any(case.get("ok") for case in cases)),
        "output_openable": int(bool(openable) and all(openable)),
        "target_changed": int(any((value or 0) > 0 for value in changed)),
        "formula_cache_ok": int(formula_risks == 0),
        "excel_error_free": int(excel_errors == 0),
        "eval_soft": record.get("eval", {}).get("soft_restriction", 0.0),
        "eval_hard": record.get("eval", {}).get("hard_restriction", 0),
    }


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {"ok": record.get("ok"), "eval": record.get("eval"), "cases": [{"case": c.get("case"), "ok": c.get("ok"), "error_category": c.get("error_category"), "verification": c.get("verification")} for c in record.get("cases", [])]}


def _task_error_record(task, error: str, cases=None) -> dict[str, Any]:
    case_rows = []
    eval_rows = []
    for case in cases or []:
        case_rows.append({"case": case.index, "input": str(case.input_path), "output": str(case.output_path), "ok": False, "returncode": None, "timed_out": False, "error_category": "agent_error", "stdout": "", "stderr": error[-4000:]})
        eval_rows.append({"case": case.index, "ok": 0, "message": error})
    return {"id": task.id, "instruction_type": task.instruction_type, "answer_position": task.answer_position, "ok": False, "error": error, "cases": case_rows, "eval": _summarize_eval(eval_rows)}


def _record(outputs_dir, dirs, result_log, processed_records, record, completed, total_selected, checkpoint_every):
    processed_records.append(record)
    _append_jsonl(result_log, record)
    if checkpoint_every > 0 and completed % checkpoint_every == 0:
        _write_checkpoint(outputs_dir, dirs["checkpoints"], processed_records, completed, total_selected, final=False)


def _load_existing_records(path: Path) -> list[dict]:
    by_id = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    record = json.loads(line)
                    by_id[str(record.get("id"))] = record
    return list(by_id.values())


def _resolve_outputs_dir(path: Path, mode: str) -> Path:
    if mode == "append":
        return path
    if not path.exists() or not any(path.iterdir()):
        return path
    if mode == "fail":
        raise FileExistsError(f"output directory already contains files: {path}")
    for index in range(1, 10000):
        candidate = path.with_name(f"{path.name}_{index:03d}")
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
    raise RuntimeError(f"could not find a free output directory suffix for: {path}")


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _metrics(records: list[dict]) -> dict:
    tasks = len(records)
    cases_total = sum(len(record.get("cases", [])) for record in records)
    cases_ok = sum(1 for record in records for case in record.get("cases", []) if case.get("ok"))
    hard_sum = sum(record.get("eval", {}).get("hard_restriction", 0) for record in records)
    soft_sum = sum(record.get("eval", {}).get("soft_restriction", 0.0) for record in records)
    return {"tasks": tasks, "exec_all_cases_ok": sum(1 for record in records if record.get("ok")), "exec_cases_ok": cases_ok, "exec_cases_total": cases_total, "exec_case_success_rate": cases_ok / cases_total if cases_total else 0.0, "hard_correct": hard_sum, "hard_accuracy": hard_sum / tasks if tasks else 0.0, "soft_sum": soft_sum, "soft_accuracy": soft_sum / tasks if tasks else 0.0}


def _write_checkpoint(outputs_dir: Path, checkpoints_dir: Path, records: list[dict], completed: int, total: int, final: bool) -> None:
    summary = {"completed": completed, "total_selected": total, "final": final, "metrics": _metrics(records), "records": records, "paths": {"outputs_dir": str(outputs_dir), "run_results": str(outputs_dir / "run_results.jsonl"), "logs": str(outputs_dir / "agent_logs"), "codes": str(outputs_dir / "codes"), "spreadsheets": str(outputs_dir / "spreadsheet")}}
    name = "latest_summary.json" if final else f"checkpoint_{completed:04d}.json"
    (outputs_dir / "latest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (checkpoints_dir / name).write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with (outputs_dir / "latest_summary.tsv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(["metric", "value"])
        for key, value in summary["metrics"].items():
            writer.writerow([key, value])
    print(json.dumps({"completed": completed, **summary["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
