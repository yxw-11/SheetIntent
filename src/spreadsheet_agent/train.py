"""One GRPO launcher for common, generation, and review policies."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from .io import ROOT, file_hash, freeze, read


def command(args):
    cfg = read(ROOT / "configs" / f"{args.stage}.json")
    if args.steps is not None:
        cfg["steps"] = args.steps
    if cfg["steps"] < 1 or cfg["batch"] % args.gpus:
        raise ValueError("Steps must be positive and GPU count must divide the prompt batch")
    if args.stage != "common" and not args.common_adapter:
        raise ValueError("Generation/review require --common-adapter; never start them from scratch")
    weights = {}
    if args.common_adapter:
        weights = {n: file_hash(args.common_adapter / n) for n in ("adapter_config.json", "adapter_model.safetensors")}
    manifest = dict(config=cfg, model=str(args.model.resolve()), adapter=weights,
                    train_sha256=file_hash(args.train_file), world_size=args.gpus,
                    tensor_parallel=args.tp, gpu_memory=args.gpu_memory,
                    dataset_sha256=file_hash(args.dataset_dir / "dataset.json"))
    args.out.mkdir(parents=True, exist_ok=True)
    freeze(args.out / "manifest.json", manifest)
    env = dict(os.environ, STAGE_MANIFEST=str(args.out / "manifest.json"))
    if args.common_adapter:
        env["COMMON_ADAPTER"] = str(args.common_adapter.resolve())
    values = {
        "algorithm.adv_estimator": "grpo", "algorithm.use_kl_in_reward": False,
        "data.train_files": str(args.train_file.resolve()), "data.val_files": str(args.train_file.resolve()),
        "data.train_batch_size": cfg["batch"], "data.val_batch_size": cfg["batch"],
        "data.max_prompt_length": cfg["prompt_tokens"], "data.max_response_length": cfg["response_tokens"],
        "data.filter_overlong_prompts": True, "data.truncation": "error",
        "data.shuffle": cfg["shuffle"], "data.dataloader_num_workers": 0,
        "custom_reward_function.path": str(ROOT / "src/spreadsheet_agent/reward.py"),
        "custom_reward_function.name": "compute_score",
        "+custom_reward_function.reward_kwargs.dataset_dir": str(args.dataset_dir.resolve()),
        "+custom_reward_function.reward_kwargs.rollout_root": str(args.out / "rollouts"),
        "+custom_reward_function.reward_kwargs.timeout": cfg["timeout"],
        "+custom_reward_function.reward_kwargs.stage": args.stage,
        "actor_rollout_ref.model.path": str(args.model.resolve()),
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.model.enable_activation_offload": False,
        "actor_rollout_ref.model.use_remove_padding": False,
        "+actor_rollout_ref.model.override_config.attn_implementation": "sdpa",
        "actor_rollout_ref.model.lora_rank": 16, "actor_rollout_ref.model.lora_alpha": 32,
        "actor_rollout_ref.model.target_modules": "all-linear",
        "actor_rollout_ref.actor.optim.lr": cfg["lr"],
        "actor_rollout_ref.actor.ppo_mini_batch_size": cfg["batch"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-mean",
        "actor_rollout_ref.actor.use_kl_loss": False, "actor_rollout_ref.actor.use_torch_compile": False,
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "bfloat16",
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile": False,
        "actor_rollout_ref.rollout.name": "vllm", "actor_rollout_ref.rollout.tensor_model_parallel_size": args.tp,
        "actor_rollout_ref.rollout.gpu_memory_utilization": args.gpu_memory,
        "actor_rollout_ref.rollout.n": cfg["generations"],
        "actor_rollout_ref.rollout.temperature": cfg["temperature"], "actor_rollout_ref.rollout.top_p": cfg["top_p"],
        "actor_rollout_ref.rollout.max_model_len": cfg["prompt_tokens"] + cfg["response_tokens"],
        "actor_rollout_ref.rollout.max_num_batched_tokens": 32768,
        "actor_rollout_ref.rollout.max_num_seqs": 8,
        "actor_rollout_ref.rollout.enable_chunked_prefill": True,
        "actor_rollout_ref.rollout.enable_prefix_caching": True,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.free_cache_engine": True,
        "actor_rollout_ref.rollout.load_format": "safetensors",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.ref.fsdp_config.model_dtype": "bfloat16",
        "trainer.logger": ["console"], "trainer.project_name": "grounded_spreadsheet_agent",
        "trainer.experiment_name": args.stage, "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": args.gpus, "trainer.save_freq": cfg["save_every"],
        "trainer.test_freq": -1, "trainer.val_before_train": False,
        "trainer.total_epochs": 100, "trainer.total_training_steps": cfg["steps"],
        "trainer.default_local_dir": str(args.out / "checkpoints"),
        "trainer.max_actor_ckpt_to_keep": 8, "trainer.resume_mode": "auto",
    }
    if args.stage != "common":
        values["actor_rollout_ref.model.external_lib"] = "spreadsheet_agent.bootstrap"
    # JSON literals keep spaces and lists unambiguous for Hydra's override parser.
    cmd = [sys.executable, "-m", "verl.trainer.main_ppo"]
    cmd += [f"{k}={json.dumps(v)}" for k, v in values.items()]
    return cmd, env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["common", "generation", "review"])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--common-adapter", type=Path)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-memory", type=float, default=.45)
    p.add_argument("--steps", type=int)
    p.add_argument("--check", action="store_true", help="Resolve Hydra config without training")
    args = p.parse_args()
    args.out = args.out.resolve()
    if args.gpus < 1 or args.tp < 1 or args.gpus % args.tp or not 0 < args.gpu_memory < 1:
        p.error("Invalid GPU count, tensor parallelism, or memory fraction")
    cmd, env = command(args)
    if args.check:
        cmd += ["--cfg", "job", "--resolve"]
    print(json.dumps({"stage": args.stage, "command": cmd, "output": str(args.out)}), flush=True)
    with (args.out / "train.log").open("a") as log:
        with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = proc.wait()
    if code:
        raise subprocess.CalledProcessError(code, cmd)


if __name__ == "__main__":
    main()
