"""Maintainer export of frozen training schedules and reference checkpoint hashes."""
import argparse
import csv
import gzip
import hashlib
import json
import subprocess
from pathlib import Path
import pyarrow.parquet as pq


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    args = p.parse_args()
    source = args.source.resolve()
    root = Path(__file__).resolve().parents[1]
    stages = {
        "common": ("data/verl_spreadsheet_rl_v2/train.parquet", "data/verl_runs/verl_fast_diagnostic_20260903_194205_29912717/preserved/global_step_50/actor/lora_adapter"),
        "generation": ("experiments/semantic_transfer/runs/v1/semantic_generation/datasets/generation/train.parquet", "experiments/semantic_transfer/runs/v1/semantic_generation/grpo/checkpoints/global_step_40/actor/lora_adapter"),
        "review": ("experiments/semantic_stage2/runs/pilot_v1/correction/datasets/correction/train.parquet", "experiments/semantic_stage2/runs/pilot_v1/correction/grpo/checkpoints/global_step_40/actor/lora_adapter"),
    }
    manifest = {}
    (root / "recipes").mkdir(exist_ok=True)
    for stage, (parquet, adapter) in stages.items():
        rows = pq.read_table(source / parquet).to_pylist()
        for row in rows:
            row["extra_info"].pop("dataset_dir", None)
        content = "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows)
        if "/data/gpfs/" in content or "/home/" in content:
            raise ValueError(f"Nonportable paths in {stage} recipe")
        path = root / "recipes" / f"{stage}.jsonl.gz"
        path.write_bytes(gzip.compress(content.encode(), mtime=0))
        weights = source / adapter
        manifest[stage] = dict(source_parquet=parquet, source_sha256=sha(source / parquet),
            recipe_sha256=sha(path), rows=len(rows), unique_tasks=len({r["extra_info"]["task_id"] for r in rows}),
            adapter_source=adapter, adapter_files={n: sha(weights / n) for n in
                ("adapter_config.json", "adapter_model.safetensors") if (weights / n).is_file()})
    (root / "provenance/training.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result_file = source / "reports/experiment_updates/20261009_main_results_compact.csv"
    reference_rows = []
    with result_file.open() as f:
        for row in csv.DictReader(f):
            if "consistency" in str(row) and ("Semantic" in str(row) or "Training-free" in str(row)):
                row = {k: (v.replace(str(source) + "/", "") if isinstance(v, str) else v) for k, v in row.items()}
                reference_rows.append(row)
    (root / "provenance/reference_results.json").write_text(json.dumps(dict(
        source=str(result_file.relative_to(source)), sha256=sha(result_file),
        evaluation="archived raw/local; not fresh release measurements", rows=reference_rows), indent=2) + "\n")
    repo = source.parent / "verl_src/verl_py311_clean"
    patch = subprocess.check_output(["git", "-C", str(repo), "diff", "v0.6.0", "--", "verl/workers/fsdp_workers.py"])
    (root / "patches").mkdir(exist_ok=True)
    (root / "patches/verl-v0.6.0.patch").write_bytes(patch)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
