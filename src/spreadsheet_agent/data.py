"""Portable, explicit preparation of training schedules and Domain evaluation data."""
import argparse
import gzip
import json
import os
import re
import shutil
from pathlib import Path
from .io import ROOT, atomic, file_hash, freeze, read


def materialize(stage, dataset, out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from spreadsheetbench_lite.data import load_tasks, discover_test_cases
    recipe = ROOT / "recipes" / f"{stage}.jsonl.gz"
    meta = read(ROOT / "provenance/training.json")[stage]
    if file_hash(recipe) != meta["recipe_sha256"]:
        raise ValueError("Frozen training recipe changed")
    rows = [json.loads(line) for line in gzip.decompress(recipe.read_bytes()).decode().splitlines()]
    tasks = {t.id: t for t in load_tasks(dataset)}
    ids = {r["extra_info"]["task_id"] for r in rows}
    for ident in ids:
        if ident not in tasks or not ident.startswith("task-job-"):
            raise ValueError(f"Missing training task or unexpected test identifier: {ident}")
        cases = discover_test_cases(dataset, out.parent / "unused", tasks[ident])
        if not cases or any(c.answer_path is None for c in cases):
            raise ValueError(f"Missing training workbooks: {ident}")
    for row in rows:
        row["extra_info"]["dataset_dir"] = str(dataset.resolve())
    manifest = dict(stage=stage, recipe_sha256=meta["recipe_sha256"],
                    dataset_sha256=file_hash(dataset / "dataset.json"), rows=len(rows), unique_tasks=len(ids))
    freeze(out.with_suffix(".manifest.json"), manifest)
    if not out.exists():
        temporary = out.with_suffix(".tmp.parquet")
        pq.write_table(pa.Table.from_pylist(rows), temporary)
        temporary.replace(out)
    elif pq.read_table(out).to_pylist() != rows:
        raise ValueError("Existing parquet differs; choose a new output path")
    print(json.dumps(manifest))


def prepare_domain(source, out, mode):
    import pyarrow.parquet as pq
    parquet = source / "test_domain_hermes.parquet"
    rows = pq.read_table(parquet).to_pylist()
    if len(rows) != 1662:
        raise ValueError(f"Expected public Domain release: 1662 tasks, got {len(rows)}")
    records, hashes = [], {}
    for row in rows:
        extra = row["extra_info"]
        ident = extra["id"]
        if not re.fullmatch(r"domain/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", ident):
            raise ValueError(f"Invalid task path: {ident}")
        path = source / "spreadsheets" / ident
        meta = read(path / "instruction.json")
        if extra["split"] != "test" or row["reward_model"]["ground_truth"] != ident:
            raise ValueError(f"Invalid split: {ident}")
        if not meta["answer_position"] or extra["answer_position"] != meta["answer_position"]:
            raise ValueError(f"Answer region mismatch: {ident}")
        if not meta["instruction"].strip() or not any(meta["instruction"].strip() in m["content"] for m in row["prompt"] if m["role"] == "user"):
            raise ValueError(f"Instruction mismatch: {ident}")
        key = ident.replace("/", "__")
        record = dict(id=key, instruction=meta["instruction"], answer_position=meta["answer_position"],
                      spreadsheet_path=f"spreadsheet/{key}", instruction_type=meta.get("type", ""),
                      official_id=ident, domain=ident.split("/")[1], split="test")
        hashes[ident] = {n: file_hash(path / n) for n in ("instruction.json", "input.xlsx", "target.xlsx")}
        for old, kind in (("input.xlsx", "input"), ("target.xlsx", "answer")):
            target = out / record["spreadsheet_path"] / f"1_{key}_{kind}.xlsx"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if file_hash(target) != hashes[ident][old]:
                    raise ValueError(f"Existing workbook changed: {target}")
            elif mode == "symlink":
                target.symlink_to((path / old).resolve())
            elif mode == "hardlink":
                os.link(path / old, target)
            else:
                shutil.copyfile(path / old, target)
        records.append(record)
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate Domain task identifiers")
    freeze(out / "provenance.json", dict(tasks=len(rows), source_sha256=file_hash(parquet), sources=hashes))
    freeze(out / "dataset.json", records)
    print(json.dumps(dict(tasks=len(records), dataset=str(out))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    train = sub.add_parser("training")
    train.add_argument("--stage", choices=["common", "generation", "review"], required=True)
    train.add_argument("--dataset-dir", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    domain = sub.add_parser("domain")
    domain.add_argument("--source-dir", type=Path, required=True)
    domain.add_argument("--out", type=Path, required=True)
    domain.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    args = p.parse_args()
    if args.action == "training":
        materialize(args.stage, args.dataset_dir.resolve(), args.out.resolve())
    else:
        prepare_domain(args.source_dir.resolve(), args.out.resolve(), args.link_mode)


if __name__ == "__main__":
    main()
