"""File-loadable verl reward entry point; reference labels never enter policy prompts."""
import functools
import json
import uuid
from pathlib import Path
from spreadsheetbench_lite.data import load_tasks, discover_test_cases
from spreadsheet_code_agent_v0.inspection import inspect_workbook
from spreadsheet_agent.io import atomic
from spreadsheet_agent.review import decode
from spreadsheet_agent.reward_terms import score_record, _target_region_progress
from spreadsheet_agent.sandbox import score, execute


@functools.lru_cache(None)
def tasks(dataset):
    return {t.id: t for t in load_tasks(Path(dataset))}


def compute_score(data_source, solution_str, ground_truth=None, extra_info=None,
                  dataset_dir=None, rollout_root=None, timeout=180, stage="common", **kwargs):
    extra = dict(extra_info or {})
    ident = extra["task_id"]
    task = tasks(dataset_dir)[ident]
    root = Path(rollout_root) / ident / uuid.uuid4().hex
    baseline = extra.get("baseline_code") if stage == "review" else None
    try:
        if stage == "review" and not baseline:
            raise ValueError("Review state is missing baseline_code")
        action, code = decode(solution_str, baseline)
    except (ValueError, SyntaxError) as exc:
        result = dict(score=-0.15, oracle=0., ok=False, stage="format_error", action="INVALID", error=str(exc))
    else:
        if stage == "common":
            # Preserve the original common-policy evaluator behavior, but isolate execution.
            from spreadsheet_code_agent_rl import env
            env.run_transform_code = lambda code, inp, out, work, limit, position: execute(code, inp, out, limit, position)
            environment = env.SpreadsheetCodeEnv(Path(dataset_dir), root, task, timeout=int(timeout), max_cases=1)
            record = environment.step_code(code, action_text=solution_str).observation["record"]
            reward = score_record(record, solution_str, strategy="diagnostic")
            ok = record["ok"]
        else:
            inspection = json.loads(extra["inspection_json"])
            scored = score(task, dataset_dir, code, inspection, root, int(timeout))
            cases = discover_test_cases(Path(dataset_dir), root, task)
            progress = _target_region_progress(cases, task.answer_position)
            progress["exact"] = scored["benchmark_exact"]
            record = {"cases": scored["cases"], "eval": {"hard_restriction": float(scored["benchmark_exact"]),
                "soft_restriction": sum(c["benchmark_exact"] for c in scored["cases"]) / len(cases)}}
            reward = score_record(record, solution_str, strategy="diagnostic_aligned", progress=progress)
            ok = scored["exec_ok"]
        atomic(root / "reward.json", reward)
        result = dict(score=reward["scalar"], oracle=reward["oracle"], ok=ok, stage=reward["stage"], action=action)
    atomic(root / "trajectory.json", dict(task_id=ident, completion=solution_str, baseline_code=baseline, **result))
    return result
