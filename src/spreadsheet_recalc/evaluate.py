from __future__ import annotations

import argparse
import fcntl
import json
import os
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import openpyxl

from spreadsheetbench_lite import evaluate_v1 as local
from spreadsheetbench_lite.data import discover_test_cases, load_tasks

from .client import ExcelClient, digest, validate_workbook, write_json
from .official import REVISION, download_snapshot, load_comparator

ROOT = Path(__file__).resolve().parents[2]
SCORE_KEYS = ("raw_local", "raw_official", "excel_local", "excel_official")


def formula_profile(path: Path) -> dict:
    """Count stored formulas and empty caches, not expected accuracy recoveries."""
    formulas = empty = 0
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not (name.startswith("xl/worksheets/") and name.endswith(".xml")):
                continue
            with archive.open(name) as stream:
                for _, element in ET.iterparse(stream, events=("end",)):
                    if element.tag == ns + "c":
                        if element.find(ns + "f") is not None:
                            formulas += 1
                            cached = element.find(ns + "v")
                            empty += cached is None or cached.text in (None, "")
                        element.clear()
                    elif element.tag == ns + "row":
                        element.clear()
    return {"formula_cells": formulas, "empty_cached_formula_cells": empty}


def score(comparator, answer: Path, prediction: Path, position: str) -> dict:
    ok, message = comparator(answer, prediction, position)
    return {"ok": bool(ok), "message": str(message)}


def inventory(dataset: Path, run_dir: Path, start: int, limit: int | None) -> list[dict]:
    tasks = load_tasks(dataset)[start:]
    if limit is not None:
        tasks = tasks[:limit]
    rows = []
    for task in tasks:
        if not task.id or task.id in {".", ".."} or Path(task.id).name != task.id:
            raise ValueError(f"Unsafe task ID: {task.id!r}")
        cases = discover_test_cases(dataset, run_dir, task)
        if not cases:
            raise ValueError(f"No input cases for {task.id}; refusing to shrink the denominator")
        for case in cases:
            if case.answer_path is None:
                raise ValueError(f"Missing oracle: {task.id}:{case.index}")
            rows.append({"id": task.id, "case": case.index, "answer_position": task.answer_position,
                         "answer": str(case.answer_path.resolve()), "prediction": str(case.output_path.resolve()),
                         "answer_sha256": digest(case.answer_path), "prediction_sha256": digest(case.output_path)})
    if not rows or len({(r["id"], r["case"]) for r in rows}) != len(rows):
        raise ValueError("Empty selection or duplicate task/case IDs")
    return rows


def read_records(path: Path) -> dict:
    records = {}
    if not path.exists():
        return records
    # A killed process can leave only the final JSONL line incomplete.
    with path.open("rb+") as stream:
        while line := stream.readline():
            end = stream.tell()
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if stream.read(1):
                    raise ValueError(f"Corrupt non-final journal record: {path}")
                stream.truncate(end - len(line))
                break
            if not line.endswith(b"\n"):
                stream.seek(end)
                stream.write(b"\n")
                stream.flush()
            records[(record["id"], record["case"])] = record
    return records


def append_record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def summarize(rows: list[dict], records: dict) -> dict:
    expected = Counter(row["id"] for row in rows)
    metrics, task_scores = {}, {}
    for key in SCORE_KEYS:
        grouped = {}
        for row in rows:
            result = records.get((row["id"], row["case"]), {}).get(key)
            if result is not None:
                grouped.setdefault(row["id"], []).append(int(result["ok"]))
        completed = {task: values for task, values in grouped.items() if len(values) == expected[task]}
        known_cases = sum(map(len, grouped.values()))
        complete = known_cases == len(rows)
        hard = sum(all(values) for values in completed.values())
        soft = sum(sum(values) / len(values) for values in completed.values())
        metrics[key] = {
            "complete": complete, "known_cases": known_cases, "total_cases": len(rows),
            "complete_tasks": len(completed), "total_tasks": len(expected),
            "hard_correct_complete_tasks": hard,
            "hard_accuracy": hard / len(expected) if complete else None,
            "soft_accuracy": soft / len(expected) if complete else None,
            "case_micro_accuracy": sum(map(sum, grouped.values())) / len(rows) if complete else None,
        }
        task_scores[key] = {task: int(all(values)) for task, values in completed.items()}
    changes = {}
    for label, before, after in [("recalculation_local", "raw_local", "excel_local"),
                                  ("recalculation_official", "raw_official", "excel_official"),
                                  ("evaluator_before_recalc", "raw_local", "raw_official"),
                                  ("evaluator_after_recalc", "excel_local", "excel_official")]:
        gained, lost, pairs = [], [], 0
        for row in rows:
            record = records.get((row["id"], row["case"]), {})
            a, b = record.get(before), record.get(after)
            if a is not None and b is not None:
                pairs += 1
                if a["ok"] != b["ok"]:
                    (gained if b["ok"] else lost).append(f"{row['id']}:{row['case']}")
        common = task_scores[before].keys() & task_scores[after].keys()
        changes[label] = {
            "paired_cases": pairs, "gained_cases": gained, "lost_cases": lost,
            "paired_tasks": len(common),
            "gained_tasks": sorted(t for t in common if task_scores[after][t] > task_scores[before][t]),
            "lost_tasks": sorted(t for t in common if task_scores[after][t] < task_scores[before][t]),
        }
    return {"metrics": metrics, "changes": changes,
            "statuses": dict(Counter(r["status"] for r in records.values())),
            "formula_cases": sum(r.get("formula_profile", {}).get("formula_cells", 0) > 0 for r in records.values()),
            "empty_cache_cases": sum(r.get("formula_profile", {}).get("empty_cached_formula_cells", 0) > 0
                                     for r in records.values()),
            "notes": ["Null accuracy means incomplete/unavailable, not zero. Errors are not model labels.",
                      "Formula/empty-cache counts include all sheets and do not predict recoverable accuracy.",
                      "Hard/soft aggregate by task; case_micro aggregates by workbook.",
                      "Same local case set throughout; official paper split/aggregation equivalence is not asserted.",
                      "Engine label is operator supplied, not independently verified by /health."]}


def process_case(row: dict, out: Path, official, previous: dict | None, client,
                 retry_failed: bool) -> dict:
    record = dict(previous or row)
    record.pop("error", None)
    answer, prediction = Path(row["answer"]), Path(row["prediction"])
    try:
        for key, path in [("answer", answer), ("prediction", prediction)]:
            if digest(path) != row[key + "_sha256"]:
                raise RuntimeError(f"Frozen {key} changed: {path}")
        if not record.get("baseline_ready"):
            validate_workbook(answer)
            # Invalid oracle metadata is a dataset problem, not a wrong prediction.
            for comparator in [local.compare_workbooks] + ([official] if official else []):
                valid, message = comparator(answer, answer, row["answer_position"])
                if not valid:
                    raise ValueError(f"Oracle self-check failed: {message}")
            record["raw_local"] = score(local.compare_workbooks, answer, prediction, row["answer_position"])
            if official:
                record["raw_official"] = score(official, answer, prediction, row["answer_position"])
            if not prediction.exists():
                record["unusable_prediction"] = "missing_output"
            else:
                try:
                    validate_workbook(prediction)
                    record["formula_profile"] = formula_profile(prediction)
                except Exception as exc:
                    record["unusable_prediction"] = f"invalid_output: {exc}"
            record["baseline_ready"] = True
        if client:
            if record.get("unusable_prediction"):
                for key in ("local", "official"):
                    if "raw_" + key in record:
                        record["excel_" + key] = {"ok": False, "message": record["unusable_prediction"]}
            else:
                target = out / "recalculated" / row["id"] / prediction.name
                state_path = out / "jobs" / f"{row['id']}_{row['case']}.json"
                client.recalculate(prediction, target, state_path, retry_failed=retry_failed)
                record["recalculated_sha256"] = digest(target)
                record["recalculated_path"] = str(target)
                record["excel_local"] = score(local.compare_workbooks, answer, target, row["answer_position"])
                if official:
                    record["excel_official"] = score(official, answer, target, row["answer_position"])
            record["status"] = "done"
        else:
            record["status"] = "prepared"
    except Exception as exc:
        record["status"] = "infrastructure_error"
        record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    return record


def run(args) -> int:
    dataset, source, out = args.dataset_dir.resolve(), args.run_dir.resolve(), args.out_dir.resolve()
    if not (source / "spreadsheet").is_dir():
        raise ValueError("--run-dir must be a completed run containing spreadsheet/")
    if out.is_relative_to(dataset) or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Report directory must be separate from source run/dataset")
    if out.exists() and any(out.iterdir()) and not (out / "manifest.json").exists():
        raise ValueError("Nonempty output directory has no rescore manifest; choose a new directory")
    rows = inventory(dataset, source, args.start, args.limit)
    official, official_info = None, None
    if not args.local_only:
        if args.download_official:
            download_snapshot(args.official_reward_dir)
        official, official_info = load_comparator(args.official_reward_dir)
    manifest = {"schema": 1, "dataset": str(dataset), "source_run": str(source), "cases": rows,
                "dataset_sha256": digest(dataset / "dataset.json"), "official": official_info,
                "local_evaluator_sha256": digest(Path(local.__file__)), "openpyxl": openpyxl.__version__,
                "implementation": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}}
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Frozen dataset/source/config/code changed; use a NEW --out-dir")
        write_json(path, manifest)
        journal = out / "case_results.jsonl"
        records = read_records(journal)
        client = None
        if args.mode == "run":
            if not args.excel_url or not args.engine_label:
                raise ValueError("run requires --excel-url and --engine-label (actual Excel version/build)")
            client = ExcelClient(args.excel_url, engine_label=args.engine_label, job_timeout=args.job_timeout)
        try:
            if client:
                service = {"base_url": client.base_url, "engine_label": args.engine_label}
                service_path = out / "service.json"
                if service_path.exists() and json.loads(service_path.read_text()) != service:
                    raise ValueError("Excel service/engine changed; use a NEW --out-dir")
                write_json(service_path, service)
                write_json(out / "health.json", client.health())
            for index, row in enumerate(rows, 1):
                key = (row["id"], row["case"])
                previous = records.get(key)
                if previous and previous["status"] == "done":
                    if previous.get("recalculated_path") and digest(Path(previous["recalculated_path"])) != previous["recalculated_sha256"]:
                        raise ValueError(f"Recalculated result changed: {key}")
                    continue
                if previous and previous["status"] == "prepared" and not client:
                    continue
                record = process_case(row, out, official, previous, client, args.retry_failed)
                records[key] = record
                append_record(journal, record)
                if record.get("error"):
                    print(json.dumps({"id": row["id"], "case": row["case"], "error": record["error"]}), flush=True)
                    if client:
                        break  # Do not spend hours polling a broken service for every case.
                if index % 100 == 0 or index == len(rows):
                    write_json(out / "summary.json", summarize(rows, records))
                    print(json.dumps({"cases_visited": index, "total_cases": len(rows),
                                      "mode": args.mode, "report": str(out)}), flush=True)
        finally:
            if client:
                client.close()
            summary = summarize(rows, records)
            write_json(out / "summary.json", summary)
        print(json.dumps({"metrics": summary["metrics"], "statuses": summary["statuses"],
                          "report": str(out)}, indent=2), flush=True)
        if summary["statuses"].get("infrastructure_error"):
            return 2
        return 0


def main():
    parser = argparse.ArgumentParser(description="Offline paired rescoring. Never modifies training or original workbooks.")
    parser.add_argument("mode", choices=["prepare", "run"], help="prepare needs no Excel; run uploads predictions only")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--official-reward-dir", type=Path,
                        default=ROOT / ".cache" / "spreadsheet_rl_evaluator" / REVISION)
    parser.add_argument("--download-official", action="store_true", help="Download pinned comparator only, once")
    parser.add_argument("--local-only", action="store_true", help="Explicitly omit official comparison")
    parser.add_argument("--excel-url", default=os.environ.get("EXCEL_API_URL"))
    parser.add_argument("--engine-label", default=os.environ.get("EXCEL_ENGINE_LABEL"))
    parser.add_argument("--job-timeout", type=float, default=600)
    parser.add_argument("--retry-failed", action="store_true", help="Resubmit failed/expired jobs; history is preserved")
    args = parser.parse_args()
    if args.start < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("start must be >= 0 and limit must be > 0")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
