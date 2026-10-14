"""Optionally copy checksum-verified reference adapters from the research workspace."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-project", type=Path, required=True)
    p.add_argument("--stage", choices=["common", "generation", "review", "all"], default="all")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    meta = json.loads((root / "provenance/training.json").read_text())
    for stage, value in meta.items():
        if args.stage not in (stage, "all"):
            continue
        if len(value["adapter_files"]) != 2:
            raise RuntimeError(f"Reference adapter unavailable: {stage}")
        for name, expected in value["adapter_files"].items():
            source = args.source_project / value["adapter_source"] / name
            with source.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                    raise ValueError(f"Reference weights changed: {source}")
            out = root / "models" / stage / name
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                with out.open("rb") as f:
                    if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                        raise ValueError(f"Refusing to replace different weights: {out}")
            else:
                shutil.copyfile(source, out)
        print(f"Verified reference adapter: models/{stage}")


if __name__ == "__main__":
    main()
