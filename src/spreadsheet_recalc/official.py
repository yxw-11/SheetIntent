"""Load an unchanged, pinned upstream comparator without its Windows server."""
from __future__ import annotations

import importlib
import hashlib
import json
import sys
import types
from pathlib import Path

import httpx

from .client import digest, write_json

REVISION = "389be8c29fed059e3b2072741c8231e86d1dee8b"
REPOSITORY = "https://github.com/Spreadsheet-RL/Spreadsheet-RL"
FILES = ["eval.py", "config.py", "inflight_cache.py", "platform.py"]


def download_snapshot(directory: Path) -> None:
    """Explicit opt-in download; no package installation or server startup."""
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "source.json").exists():
        _, source = load_comparator(directory)
        if source["source"].get("revision") != REVISION:
            raise RuntimeError("Existing official snapshot has a different revision")
        return
    base = f"https://raw.githubusercontent.com/Spreadsheet-RL/Spreadsheet-RL/{REVISION}"
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        for name in FILES + ["LICENSE"]:
            path = directory / name
            if path.exists():
                continue
            relative = name if name == "LICENSE" else f"reward/async_reward_api/{name}"
            response = client.get(f"{base}/{relative}")
            response.raise_for_status()
            temporary = path.with_suffix(path.suffix + ".download")
            temporary.write_bytes(response.content)
            temporary.replace(path)
    write_json(directory / "source.json", {
        "repository": REPOSITORY, "revision": REVISION,
        "license": "CC-BY-SA-4.0; see LICENSE", "files": {n: digest(directory / n) for n in FILES}})


def load_comparator(directory: Path):
    # A namespace isolates upstream relative imports without importing its FastAPI app.
    package = directory / "async_reward_api" if (directory / "async_reward_api").is_dir() else directory
    hashes = {name: digest(package / name) for name in FILES}
    if not all(hashes.values()):
        raise RuntimeError("Official evaluator missing; use --download-official once or specify --official-reward-dir")
    metadata = package / "source.json"
    source = json.loads(metadata.read_text()) if metadata.exists() else {"repository": REPOSITORY}
    if "files" in source and hashes != source["files"]:
        raise RuntimeError("Official evaluator cache changed; restore the snapshot or use a new directory")
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    name = "_spreadsheet_rl_eval_" + identity[:16]
    module = types.ModuleType(name)
    module.__path__ = [str(package)]
    sys.modules[name] = module
    evaluator = importlib.import_module(name + ".eval")
    return evaluator.compare_workbooks, {"source": source, "files": hashes, "path": str(package)}
