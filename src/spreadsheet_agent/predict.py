"""Generate independent repaired programs, then select by visible-output consistency."""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from spreadsheetbench_lite.data import load_tasks, discover_test_cases
from spreadsheetbench_lite import llm
from spreadsheet_code_agent_v0 import run_agent_v0 as agent
from .io import atomic, digest, file_hash, freeze, read
from .consistency import select
from .observations import evidence


def routes(profile):
    # Each tuple is (understanding, generation, failure-triggered repair).
    common = ("common", "common", "common")
    generation = ("base", "generation", "common")
    return {"mixed": [common, common, generation, generation],
            "free": [("base", "base", "base")] * 4,
            "common": [common] * 4}[profile]


def records(root):
    rows = [json.loads(line) for line in (root / "run_results.jsonl").read_text().splitlines()]
    result = {r["id"]: r for r in rows}
    if len(rows) != len(result):
        raise ValueError("Duplicate task results")
    return result


def select_outputs(tasks, dataset, roots, out, tokenizer):
    selected = []
    tables = [records(root) for root in roots]
    expected = {t.id for t in tasks}
    if any(set(table) != expected for table in tables):
        raise ValueError("Candidate runs must contain exactly the requested task IDs before selection")
    for task in tasks:
        observations = []
        for root in roots:
            case = discover_test_cases(dataset, root, task)[0]
            code = root / "codes" / f"{task.id}.py"
            row = dict(id=task.id, variants=[dict(variant=case.index, input_path=str(case.input_path))])
            observations.append(evidence(row, code.read_text() if code.exists() else "", root,
                                         task.instruction, tokenizer, 12288))
        choices = select(observations)
        choice = choices["consensus"]
        # Commit the label-blind decision BEFORE consulting stored evaluator outcomes.
        atomic(out / "choices" / f"{task.id}.json", dict(selected=choice, rules=choices,
            observations=[{k: v for k, v in obs.items() if k != "prompt"} for obs in observations]))
        record = copy.deepcopy(tables[choice][task.id])
        for case in discover_test_cases(dataset, roots[choice], task):
            target = out / "spreadsheet" / task.id / case.output_path.name
            if case.output_path.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(case.output_path, target)
                elif file_hash(target) != file_hash(case.output_path):
                    raise RuntimeError(f"Selected output changed: {target}")
            for r in record["cases"]:
                if r["case"] == case.index:
                    r["output"] = str(target)
        code = roots[choice] / "codes" / f"{task.id}.py"
        if code.exists():
            (out / "codes").mkdir(exist_ok=True)
            shutil.copyfile(code, out / "codes" / code.name)
        record["selection"] = dict(candidate=choice, rule="consensus")
        selected.append(record)
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / "run_results.jsonl.tmp"
    temporary.write_text("".join(json.dumps(r) + "\n" for r in selected))
    temporary.replace(out / "run_results.jsonl")
    metrics = agent._metrics(selected)
    total = sum(len(r["eval"]["test_case_results"]) for r in selected)
    metrics["average_accuracy"] = sum(sum(r["eval"]["test_case_results"]) for r in selected) / total
    atomic(out / "summary.json", dict(complete=True, tasks=len(tasks), metrics=metrics))
    print(json.dumps(metrics), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["v1", "domain"], required=True)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokenizer", required=True, help="Local base-model tokenizer")
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--profile", choices=["mixed", "free", "common"])
    p.add_argument("--seed", type=int, help="Optional request seeding; historical full runs did not set a seed")
    p.add_argument("--limit", type=int)
    p.add_argument("--select-only", action="store_true")
    p.add_argument("--unsafe-local-execution", action="store_true", help="Historical executor; NOT a sandbox")
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error("limit must be positive")
    args.profile = args.profile or ("mixed" if args.dataset == "v1" else "free")
    args.out, args.dataset_dir = args.out.resolve(), args.dataset_dir.resolve()
    tasks = load_tasks(args.dataset_dir)[:args.limit]
    if not tasks:
        raise ValueError("Empty dataset")
    if args.limit is None and len(tasks) != {"v1": 912, "domain": 1662}[args.dataset]:
        raise ValueError("Unexpected full-dataset denominator; use --limit explicitly for a subset")
    plan = routes(args.profile)
    identity = dict(dataset=file_hash(args.dataset_dir / "dataset.json"), tasks=[t.id for t in tasks],
                    routes=plan, seed=args.seed, tokenizer=args.tokenizer,
                    unsafe_local_execution=args.unsafe_local_execution)
    if not args.select_only:
        from openai import OpenAI
        available = OpenAI(base_url=args.base_url, api_key="EMPTY").models.list().model_dump(mode="json")
        aliases = {m["id"] for m in available["data"]}
        required = {alias for group in plan for alias in group}
        if not required <= aliases:
            raise RuntimeError(f"Missing served aliases: {required - aliases}")
        identity["served_models"] = sorted([m for m in available["data"] if m["id"] in required], key=lambda m: m["id"])
        # Server process timestamps are not checkpoint identities.
        for model in identity["served_models"]:
            model.pop("created", None)
            model.pop("permission", None)
        freeze(args.out / "manifest.json", identity)
    else:
        previous = read(args.out / "manifest.json")
        if any(previous.get(k) != json.loads(json.dumps(v)) for k, v in identity.items()):
            raise ValueError("Selection configuration differs from the saved candidate run")
    if not args.unsafe_local_execution:
        from .sandbox import execute
        agent.run_transform_code = lambda code, inp, out, work, timeout, position: execute(code, inp, out, timeout, position)
    if args.dataset == "domain":
        from .domain_preflight import run_preflight
        agent.run_preflight = run_preflight
    original = llm.OpenAI
    roots = []
    for index, (understanding, generation, repair) in enumerate(plan):
        root = args.out / "candidates" / str(index)
        roots.append(root)
        if args.select_only:
            continue
        def seeded_client(*a, **kw):
            client = original(*a, **kw)
            create = client.chat.completions.create
            def seeded(**request):
                if args.seed is not None:
                    request["seed"] = int(digest([args.seed, index, request])[:8], 16) % (2**31)
                return create(**request)
            client.chat.completions.create = seeded
            return client
        llm.OpenAI = seeded_client
        sys.argv = ["agent", "--dataset-dir", str(args.dataset_dir), "--outputs-dir", str(root),
                    "--model", generation, "--understanding-model", understanding, "--repair-model", repair,
                    "--base-url", args.base_url, "--mode", "full", "--cot", "--rows", "3",
                    "--max-tokens", "32768", "--understanding-tokens", "1024", "--timeout", "300",
                    "--max-repairs", "2", "--resume", "--output-exists", "append"]
        if args.limit:
            sys.argv += ["--limit", str(args.limit)]
        try:
            agent.main()
        finally:
            llm.OpenAI = original
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    select_outputs(tasks, args.dataset_dir, roots, args.out / "selected", tokenizer)


if __name__ == "__main__":
    main()
