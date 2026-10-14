"""Optional one-pass semantic review; never enabled by the best-profile launcher."""
import argparse
import json
import shutil
from pathlib import Path
from spreadsheetbench_lite.data import load_tasks, discover_test_cases
from spreadsheetbench_lite.llm import LLMConfig, chat_completion
from spreadsheet_code_agent_v0 import run_agent_v0 as agent
from .io import atomic, file_hash, freeze
from .predict import records
from .review import visible_facts, correction_prompt, decode
from .sandbox import execute


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True, help="One candidate run, or the selected/ run")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--model", default="review")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    args.out, args.source = args.out.resolve(), args.source.resolve()
    if args.out == args.source or args.source in args.out.parents:
        p.error("Review output must be separate from the source run")
    source_records = records(args.source)
    tasks = load_tasks(args.dataset_dir)[:args.limit]
    freeze(args.out / "manifest.json", dict(dataset=file_hash(args.dataset_dir / "dataset.json"),
        source_results=file_hash(args.source / "run_results.jsonl"), model=args.model, tasks=[t.id for t in tasks]))
    config = LLMConfig(model=args.model, base_url=args.base_url, api_key="EMPTY", temperature=.2,
                       max_tokens=12288, disable_thinking=False)
    dirs = {"exec": args.out / "exec"}
    agent.run_transform_code = lambda code, inp, out, work, timeout, position: execute(code, inp, out, timeout, position)
    completed = []
    for task in tasks:
        cached = args.out / "records" / f"{task.id}.json"
        if cached.exists():
            completed.append(json.loads(cached.read_text()))
            continue
        source_cases = discover_test_cases(args.dataset_dir, args.source, task)
        cases = discover_test_cases(args.dataset_dir, args.out, task)
        source_code = args.source / "codes" / f"{task.id}.py"
        code = source_code.read_text() if source_code.exists() else ""
        action, trace = "SKIP", None
        # Trigger on observable executability, not on reference correctness.
        if code and source_cases[0].output_path.is_file():
            try:
                facts = visible_facts(source_cases[0].input_path, source_cases[0].output_path)
            except (OSError, ValueError, KeyError):
                facts = None
            if facts is not None:
                prompt = correction_prompt(task.instruction, code, facts)
                response = chat_completion(prompt, config)
                trace = dict(prompt=prompt, response=response.raw_response)
                try:
                    action, code = decode(response.content, code)
                except (ValueError, SyntaxError) as exc:
                    action, trace["format_error"] = "KEEP_INVALID_REVIEW", str(exc)
        if action == "REVISE":
            record = agent._compact_record(agent._execute_and_evaluate(task, cases, code, dirs, 300))
        else:
            record = source_records[task.id]
            for source, case in zip(source_cases, cases):
                if source.output_path.exists():
                    case.output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source.output_path, case.output_path)
            for row, case in zip(record["cases"], cases):
                row["output"] = str(case.output_path)
        (args.out / "codes").mkdir(exist_ok=True)
        (args.out / "codes" / f"{task.id}.py").write_text(code)
        atomic(args.out / "review_logs" / f"{task.id}.json", dict(action=action, trace=trace))
        atomic(cached, record)
        completed.append(record)
        print(json.dumps(dict(completed=len(completed), task=task.id, action=action)), flush=True)
    (args.out / "run_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in completed))
    atomic(args.out / "summary.json", dict(complete=True, metrics=agent._metrics(completed)))


if __name__ == "__main__":
    main()
