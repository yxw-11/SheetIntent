from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the official Spreadsheet-RL train split to this project's lightweight dataset format.")
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory containing train_*.parquet and/or spreadsheets.zip.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Converted dataset output directory.")
    parser.add_argument("--parser", choices=("hermes", "qwen3_coder"), default="hermes")
    parser.add_argument("--split", default="train", help="Usually train; uses <split>_<parser>.parquet when present.")
    parser.add_argument("--workbook-root", type=Path, default=None, help="Expanded spreadsheets root. Auto-detected when omitted.")
    parser.add_argument("--unzip", action="store_true", help="Unzip spreadsheets.zip into source-dir/spreadsheets when needed.")
    parser.add_argument("--link-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    out_dir = args.out_dir.resolve()
    workbook_root = _resolve_workbook_root(source_dir, args.workbook_root, args.unzip)
    parquet = source_dir / f"{args.split}_{args.parser}.parquet"

    rows = _load_parquet_rows(parquet) if parquet.exists() else []
    row_by_id = {_row_id(row): row for row in rows if _row_id(row)}
    all_task_dirs = _find_task_dirs(workbook_root)
    task_dirs = _select_task_dirs(workbook_root, all_task_dirs, list(row_by_id)) if row_by_id else all_task_dirs
    if not task_dirs:
        raise SystemExit(f"No Spreadsheet-RL task directories with input.xlsx and target/output workbook found under {workbook_root}")
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    spreadsheet_dir = out_dir / "spreadsheet"
    spreadsheet_dir.mkdir(parents=True, exist_ok=True)
    records = []
    missing_instruction = 0
    for index, task_dir in enumerate(task_dirs, start=1):
        task_id = _safe_id(task_dir.name or f"task_{index}")
        rel_id = _task_rel_id(workbook_root, task_dir)
        row = row_by_id.get(rel_id) or row_by_id.get(task_dir.name) or row_by_id.get(task_id) or {}
        instruction, meta = _load_instruction(task_dir, row)
        if not instruction:
            missing_instruction += 1
            instruction = _instruction_from_prompt(row) or f"Complete the spreadsheet transformation for task {task_dir.name}."
        answer_position = _first_text(meta, row, ["answer_position", "target_position", "target_range", "answer_range"]) or "__FULL_WORKBOOK__"
        instruction_type = _first_text(meta, row, ["instruction_type", "ability", "category", "task_type"]) or "Spreadsheet-RL Train"

        dst_task = spreadsheet_dir / task_id
        dst_task.mkdir(parents=True, exist_ok=True)
        _link_or_copy(task_dir / "input.xlsx", dst_task / f"1_{task_id}_input.xlsx", args.link_mode)
        answer_src = _answer_source(task_dir)
        _link_or_copy(answer_src, dst_task / f"1_{task_id}_answer.xlsx", args.link_mode)
        records.append(
            {
                "id": task_id,
                "instruction": instruction,
                "spreadsheet_path": f"spreadsheet/{task_id}",
                "instruction_type": instruction_type,
                "answer_position": answer_position,
                "answer_sheet": _first_text(meta, row, ["answer_sheet", "target_sheet"]) or "",
                "data_position": _first_text(meta, row, ["data_position", "source_range"]) or "",
                "source_task_id": task_dir.name,
                "source_dir": str(task_dir),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset.json").write_text(json.dumps(records, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summary = {
        "source_dir": str(source_dir),
        "workbook_root": str(workbook_root),
        "parquet": str(parquet) if parquet.exists() else None,
        "tasks": len(records),
        "missing_instruction": missing_instruction,
        "link_mode": args.link_mode,
        "answer_position_default": "__FULL_WORKBOOK__",
    }
    (out_dir / "conversion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _resolve_workbook_root(source_dir: Path, explicit: Path | None, unzip: bool) -> Path:
    if explicit:
        return explicit.resolve()
    candidates = [source_dir / "spreadsheets", source_dir / "excelforum", source_dir]
    for candidate in candidates:
        if _has_task_dirs(candidate):
            return candidate.resolve()
    zip_path = source_dir / "spreadsheets.zip"
    if zip_path.exists() and unzip:
        target = source_dir / "spreadsheets"
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
        for candidate in [target, target / "excelforum"]:
            if _has_task_dirs(candidate):
                return candidate.resolve()
    raise SystemExit(f"Cannot find expanded Spreadsheet-RL workbooks. Try --unzip or --workbook-root. Searched under {source_dir}")


def _has_task_dirs(root: Path) -> bool:
    if not root.exists():
        return False
    return any((path / "input.xlsx").exists() and _answer_source(path, required=False) for path in root.glob("**/*") if path.is_dir())


def _find_task_dirs(root: Path, preferred_ids: set[str] | None = None) -> list[Path]:
    dirs = [path for path in root.glob("**/*") if path.is_dir() and (path / "input.xlsx").exists() and _answer_source(path, required=False)]
    if preferred_ids:
        preferred = {str(x) for x in preferred_ids}
        dirs = [path for path in dirs if path.name in preferred or _safe_id(path.name) in preferred]
    return sorted(dirs, key=lambda p: p.name)


def _select_task_dirs(root: Path, dirs: list[Path], row_ids: list[str]) -> list[Path]:
    by_rel = {_task_rel_id(root, path): path for path in dirs}
    by_name = {path.name: path for path in dirs}
    selected = []
    missing = 0
    for row_id in row_ids:
        path = by_rel.get(row_id) or by_name.get(Path(row_id).name)
        if path:
            selected.append(path)
        else:
            missing += 1
    if not selected:
        raise SystemExit(
            f"Found {len(dirs)} workbook task directories under {root}, but none matched the parquet ids. "
            "Pass --workbook-root if the extracted Spreadsheet-RL tree is elsewhere."
        )
    if missing:
        print(f"Warning: {missing}/{len(row_ids)} parquet ids did not match workbook directories.")
    return selected


def _task_rel_id(root: Path, task_dir: Path) -> str:
    try:
        return task_dir.relative_to(root).as_posix()
    except ValueError:
        return task_dir.name


def _answer_source(task_dir: Path, required: bool = True) -> Path | None:
    for name in ("target.xlsx", "output.xlsx", "answer.xlsx"):
        path = task_dir / name
        if path.exists():
            return path
    if required:
        raise FileNotFoundError(f"no target/output workbook in {task_dir}")
    return None


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except Exception as exc:
        raise SystemExit(f"pandas is required to read {path}: {exc}") from exc
    frame = pd.read_parquet(path)
    return frame.to_dict(orient="records")


def _row_id(row: dict[str, Any]) -> str:
    info = _jsonish(row.get("extra_info"))
    for obj in (info, row):
        for key in ("id", "task_id", "uid", "qid", "example_id"):
            value = obj.get(key) if isinstance(obj, dict) else None
            if value not in (None, ""):
                return str(value)
    return ""


def _load_instruction(task_dir: Path, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    meta = {}
    for name in ("instruction.json", "metadata.json", "task.json"):
        path = task_dir / name
        if path.exists():
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                break
            except Exception:
                meta = {}
    instruction = _first_text(meta, row, ["instruction", "query", "question", "task", "goal"]) or _instruction_from_prompt(row)
    return instruction, meta


def _instruction_from_prompt(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    obj = _jsonish(prompt)
    if isinstance(obj, list):
        texts = []
        for item in obj:
            if isinstance(item, dict) and item.get("role") in ("user", "human"):
                texts.append(_extract_task_text(_content_text(item.get("content"))))
        return "\n".join(text for text in texts if text).strip()
    if isinstance(obj, dict):
        return _content_text(obj.get("content") or obj.get("prompt") or obj.get("instruction"))
    return _content_text(prompt)


def _extract_task_text(text: str) -> str:
    match = re.search(r"Task:\s*(.*?)(?:\n\s*Workbook overview\b|\n\s*Target sheet\b|\Z)", text, flags=re.S)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return text


def _first_text(meta: dict[str, Any], row: dict[str, Any], keys: list[str]) -> str:
    extra = _jsonish(row.get("extra_info"))
    for obj in (meta, extra if isinstance(extra, dict) else {}, row):
        for key in keys:
            value = obj.get(key) if isinstance(obj, dict) else None
            text = _content_text(value)
            if text:
                return text
    return ""


def _content_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, list):
        return "\n".join(_content_text(x) for x in value if _content_text(x)).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "instruction", "value"):
            if key in value:
                return _content_text(value[key])
    return re.sub(r"\s+", " ", str(value)).strip()


def _jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return {}
    try:
        return json.loads(value)
    except Exception:
        return value


def _safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value[:120] or "task"


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
