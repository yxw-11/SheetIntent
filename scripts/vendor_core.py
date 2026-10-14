"""Maintainer-only export of the validated core; never needed to run the release."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    dest = Path(__file__).resolve().parents[1]
    records = {}

    def emit(relative, text, origin):
        path = dest / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        records[relative] = {"source": origin, "sha256": hashlib.sha256(text.encode()).hexdigest()}

    for package in ("spreadsheetbench_lite", "spreadsheet_code_agent_v0", "spreadsheet_recalc"):
        for path in (source / "src" / package).glob("*.py"):
            if path.name in {"run_v1.py", "executor.py"}:
                continue
            text = path.read_text().replace('/data/gpfs/projects/punim2168/hf_dir/Qwen3.5-4B', 'Qwen/Qwen3-4B-Thinking-2507')
            if package == "spreadsheet_code_agent_v0" and path.name == "run_agent_v0.py":
                text = text.replace('        except Exception as exc:\n            record = _task_error_record',
                    '        except Exception as exc:\n'
                    '            # Infrastructure failures must not become permanently cached model failures.\n'
                    '            from spreadsheet_agent.sandbox import InfrastructureError\n'
                    '            if isinstance(exc, InfrastructureError):\n'
                    '                raise\n'
                    '            record = _task_error_record')
                text = text.replace('    for attempt in range(max_repairs + 1):\n',
                    '    for attempt in range(max_repairs + 1):\n'
                    '        # Failed final preflight must not leave outputs from an earlier program.\n'
                    '        for case in cases:\n'
                    '            case.output_path.unlink(missing_ok=True)\n')
            emit(f"src/{package}/{path.name}", text, str(path.relative_to(source)))

    def extract(path, names):
        text = (source / path).read_text()
        nodes = {n.name: n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        return "\n\n\n".join(ast.get_source_segment(text, nodes[n]) for n in names) + "\n"

    origin = "src/spreadsheetbench_lite/executor.py"
    emit("src/spreadsheetbench_lite/executor.py", '"""Legacy import path for code extraction; execution lives in the harness."""\nimport re\n\n'
         + extract(origin, ["extract_code", "_first_code_line"]), origin)

    origin = "src/spreadsheet_code_agent_rl/verl_reward_v2.py"
    names = ["score_record", "_target_region_progress", "_target_positions", "_artifact_quality", "_failure_stage", "_format_bonus", "_mean"]
    header = ('from __future__ import annotations\nimport re\nfrom pathlib import Path\nfrom typing import Any\nimport openpyxl\n'
              'from spreadsheetbench_lite.evaluate_v1 import compare_value, expand_cell_range, parse_answer_positions\n\n')
    emit("src/spreadsheet_agent/reward_terms.py", header + extract(origin, names), origin)
    origin = "experiments/semantic_stage2/core.py"
    header = ('from __future__ import annotations\nimport ast\nimport json\nimport re\nfrom collections import Counter\nimport openpyxl\n'
              'from spreadsheet_code_agent_v0.prompts import extract_transform_code\n\n')
    emit("src/spreadsheet_agent/review.py", header + extract(origin, ["visible_facts", "correction_prompt", "decode"]), origin)
    origin = "experiments/stagewise_v1/scoring.py"
    text = (source / origin).read_text().split("\ndef sanity(")[0]
    text = text.replace("from .common import HERE, atomic", "from .io import WORK as HERE, atomic")
    text = text.replace('(HERE / "scratch").mkdir(exist_ok=True)', '(HERE / "scratch").mkdir(parents=True, exist_ok=True)')
    text = text.replace('    destination.parent.mkdir(parents=True, exist_ok=True)\n',
        '    destination.parent.mkdir(parents=True, exist_ok=True)\n'
        '    (HERE / "scratch").mkdir(parents=True, exist_ok=True)\n'
        '    if destination.resolve() == Path(source).resolve():\n'
        '        raise InfrastructureError("Input and output must be different files")\n'
        '    # Repair retries reuse the destination; stale files must never count as success.\n'
        '    destination.unlink(missing_ok=True)\n')
    emit("src/spreadsheet_agent/sandbox.py", text, origin)
    origin = "experiments/selection_loss_v1/selection.py"
    header = ('from __future__ import annotations\nimport hashlib\nimport json\nimport zipfile\n'
              'from collections import Counter\nfrom datetime import date, datetime, time, timedelta\n'
              'from xml.etree import ElementTree as ET\nfrom openpyxl import load_workbook\n'
              'from openpyxl.worksheet.formula import ArrayFormula, DataTableFormula\nfrom .io import digest\n\n')
    text = header + extract(origin, ["cell_value", "fingerprint", "select"])
    text += "\n" + extract("experiments/learned_selector/core.py", ["strict_fingerprint"])
    emit("src/spreadsheet_agent/consistency.py", text, origin + " + experiments/learned_selector/core.py")
    origin = "experiments/learned_selector/core.py"
    header = ('from __future__ import annotations\nimport hashlib\nimport json\nimport zipfile\n'
              'from pathlib import Path\nfrom xml.etree import ElementTree as ET\n'
              'from .consistency import strict_fingerprint\nfrom .review import visible_facts\n\n')
    emit("src/spreadsheet_agent/observations.py", header + extract(origin, ["prompt", "evidence"]), origin)
    origin = "experiments/sheet_preflight_fix.py"
    emit("src/spreadsheet_agent/domain_preflight.py", (source / origin).read_text().split('\ndef affected(')[0], origin)
    for name in ("env", "reward", "adapt_spreadsheet_rl"):
        path = source / "src/spreadsheet_code_agent_rl" / f"{name}.py"
        emit(f"src/spreadsheet_code_agent_rl/{name}.py", path.read_text(), str(path.relative_to(source)))
    emit("src/spreadsheet_code_agent_rl/__init__.py", "", "package marker")
    atomic_path = dest / "provenance/core_sources.json"
    atomic_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_path.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Exported {len(records)} core source files")


if __name__ == "__main__":
    main()
