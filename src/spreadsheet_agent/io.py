"""Small shared helpers; all writable state is explicitly project-local."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("SPREADSHEET_WORK_DIR", ROOT / ".runtime"))


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True, default=str) + "\n")
    tmp.replace(path)


def freeze(path, value):
    # Compare the serialized representation, including tuple-to-list conversion.
    value = json.loads(json.dumps(value, default=str))
    if Path(path).exists():
        if read(path) != value:
            raise RuntimeError(f"Inputs changed; use a new output directory: {path}")
    else:
        atomic(path, value)
