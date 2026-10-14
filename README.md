# Grounded Spreadsheet Code Agent

A standalone research release for Qwen3-4B-Thinking-2507 spreadsheet agents.
The policy generates a Python `transform(wb)` function. A deterministic harness
loads the input workbook, checks the program, executes it, and saves the output.
This directory can be moved out of the original research workspace.

## 1. Create the environment

Requirements: Linux, conda, Git, an NVIDIA driver compatible with CUDA 12.8,
and allocated GPUs. Generated Python requires `bubblewrap` (`bwrap`) and
`libseccomp.so.2`, with user namespaces enabled. These are system dependencies;
ask the cluster administrator to enable them if necessary.

```bash
cd submission
# Load your cluster's conda module first, if conda is not on PATH.
bash scripts/setup_env.sh
source scripts/env.sh
python -m spreadsheet_agent.check_env --gpu
bash scripts/test.sh
```

The environment is created at `.envs/agent`, not in the user's home directory.
All download/compiler caches and the process HOME are redirected under this
project. Short-lived socket files use `/tmp/gsca_*` to avoid Unix socket path
length limits. Set `ENV_PREFIX` to use a different conda prefix. Every launcher
selects that prefix's Python automatically; no separate activation is required.

The main compatibility pins are Python 3.11, PyTorch 2.7.0/CUDA 12.8,
vLLM 0.9.1, Transformers 4.51.3, and verl v0.6.0. The installer applies the
checked-in verl patch for SDPA configuration, LoRA dtype consistency, and event
loop initialization. It does not modify an existing research environment.
Do not independently upgrade torch, vLLM, or Transformers in this environment.

Training launchers support one or two GPUs. Memory requirements depend on
sequence lengths and co-located rollout memory; an 80 GB GPU is the reference
starting point, not a guarantee that every allocation fits. Reserve adequate
host RAM as well. For two 40 GB GPUs use `--gpus 2 --tp 2` and test a separate
short run first. Do not silently shorten response budgets to make a run fit.

## 2. Obtain model and data

```bash
source scripts/env.sh
python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-4B-Thinking-2507", local_dir="models/base")'
export MODEL="$PWD/models/base"

python -c 'from huggingface_hub import snapshot_download; snapshot_download("Spreadsheet-RL/Spreadsheet-RL", repo_type="dataset", local_dir="data/upstream_spreadsheet_rl")'
```

An already downloaded local model/data directory can be used instead. Download
SpreadsheetBench **V1** from its official
[repository](https://github.com/RUCKBReasoning/SpreadsheetBench) and extract
`all_data_912_v0.1` under `data/`. Its `dataset.json` and `spreadsheet/` directory
must remain together. 
Normalize the official Spreadsheet-RL training split and the separate Domain
evaluation split:

```bash
python -m spreadsheet_code_agent_rl.adapt_spreadsheet_rl \
  --source-dir data/upstream_spreadsheet_rl --unzip \
  --out-dir data/train --parser hermes --split train --link-mode hardlink

bash scripts/prepare_data.sh domain \
  --source-dir data/upstream_spreadsheet_rl \
  --out data/domain --link-mode hardlink

for stage in common generation review; do
  bash scripts/prepare_data.sh training --stage "$stage" \
    --dataset-dir data/train --out "data/recipes/$stage.parquet"
done
```

Use `--link-mode copy` when source and destination are on different filesystems.
The Domain converter uses the released `test_domain_hermes.parquet`, original
instructions, answer regions, `input.xlsx`, and `target.xlsx`.
### Frozen training recipes

`recipes/*.jsonl.gz` contains the actual prepared policy prompts and schedules,
not generated placeholders. `provenance/training.json` records hashes, task
counts, and the original adapters' hashes. The preparation command only
rebinds dataset paths and writes portable verl Parquet files. It does not use
either evaluation benchmark to construct training prompts.

| Stage | Prepared rows | Unique training tasks | Updates | Starting policy |
|---|---:|---:|---:|---|
| Common | 1,136 | 1,136 | 50 | Base Thinking model |
| Generation | 160 | 73 | 40 | Common-50 adapter |
| Review | 160 | 117 | 40 | Common-50 adapter |

Rows are not the number of unique tasks actually visited by an interrupted or
short common-policy run. Generation and review schedules repeat selected
states. The generation recipe emphasizes executable-but-wrong states mixed
with coverage examples; review prompts contain a current program and bounded
input/output observations. Review actions are `KEEP` or `REVISE`, with a full
replacement function for `REVISE`. Reference values appear only on the reward
side, not in these policy observations.

These are frozen-data reproduction recipes. Recollecting states from a newly
trained common policy is a new experiment, not automatic regeneration of the
archived state distribution.

## 3. Train the policies

```bash
export MODEL="$PWD/models/base"

bash scripts/train_common.sh \
  --model "$MODEL" --dataset-dir data/train \
  --train-file data/recipes/common.parquet --out runs/common

export COMMON_ADAPTER="$PWD/runs/common/checkpoints/global_step_50/actor/lora_adapter"

bash scripts/train_generation.sh \
  --model "$MODEL" --dataset-dir data/train \
  --train-file data/recipes/generation.parquet \
  --common-adapter "$COMMON_ADAPTER" --out runs/generation

bash scripts/train_review.sh \
  --model "$MODEL" --dataset-dir data/train \
  --train-file data/recipes/review.parquet \
  --common-adapter "$COMMON_ADAPTER" --out runs/review
```

Generation and review are **independent branches from Common-50**, not a
sequential generation-then-review update of one adapter. The bootstrap verifies
adapter hashes and shapes before restoring weights. New branches start new
optimizer state. Re-running the same stage command resumes its own saved verl
checkpoint, including optimizer state. Keep the GPU count and recipe unchanged
on resume. Manifests reject incompatible reuse of an output directory.

Use `--check` to resolve the Hydra configuration without loading a model, or
`--steps 2 --out runs/common_smoke` for a separate two-update training smoke run.
Do not reuse a smoke output directory for the reference 50-update run.

The three configs in `configs/` expose the method's hyperparameters: four
prompts per update, four rollout completions per prompt, learning rate `8e-7`,
LoRA rank 16/alpha 32, and 12,288-token prompt and completion limits. Training
uses GRPO, no KL penalty, gradient checkpointing, and SDPA. Infrastructure
options in `train.py` control microbatching and rollout memory separately.

Common uses the original diagnostic reward: correctness, execution/artifact
quality, format feedback, and error-category penalties. Generation and review
use its target-aligned version, including partial target progress and exact
correctness anchored to the comparator. See `reward_terms.py` for the exact
formula. Training rewards do **not** invoke Microsoft Excel recalculation.

Each run saves a frozen manifest, `train.log`, reward/trajectory files under
`rollouts/`, and checkpoints every ten updates. Up to eight checkpoints are
retained. The deployment adapter is `actor/lora_adapter/`; do not point vLLM
at the distributed optimizer checkpoint directory.

### Use archived adapters without retraining

On the original research filesystem, the following optional import verifies
and copies only the three small LoRA adapters, never optimizer states:

```bash
python scripts/import_checkpoints.py --source-project ..
export COMMON_ADAPTER="$PWD/models/common"
export GENERATION_ADAPTER="$PWD/models/generation"
export REVIEW_ADAPTER="$PWD/models/review"
```

This import is not needed on a fresh machine with newly trained adapters. No
private workspace path is needed by training, prediction, or recalculation.

## 4. Serve and predict with consistency

For newly trained policies:

```bash
export COMMON_ADAPTER="$PWD/runs/common/checkpoints/global_step_50/actor/lora_adapter"
export GENERATION_ADAPTER="$PWD/runs/generation/checkpoints/global_step_40/actor/lora_adapter"
export REVIEW_ADAPTER="$PWD/runs/review/checkpoints/global_step_40/actor/lora_adapter"
MODEL="$PWD/models/base" PORT=8001 bash scripts/serve.sh
```

The server runs in the foreground. Open another shell for prediction:

```bash
cd submission
source scripts/env.sh

# Best observed V1 profile: two common + two generation-specialized candidates.
bash scripts/predict.sh --dataset v1 \
  --dataset-dir data/all_data_912_v0.1 --tokenizer models/base \
  --out runs/v1_mixed --base-url http://127.0.0.1:8001/v1

# Best observed Domain profile: four training-free candidates.
bash scripts/predict.sh --dataset domain \
  --dataset-dir data/domain --tokenizer models/base \
  --out runs/domain_free --base-url http://127.0.0.1:8001/v1
```

To evaluate the trained mixed policy on Domain, add `--profile mixed` and use a
new output directory such as `runs/domain_mixed`. `--profile common` uses four
common-policy candidates; `--profile free` uses four base-policy candidates.
For a small smoke evaluation, add `--limit 3` and use a separate output folder.
Re-run the exact command to resume. Do not change aliases to different weights
inside an existing run. Optional `--seed` seeds requests; historical full runs
did not set request seeds, and sampling is not a bitwise replay guarantee.

| Candidate type | Understanding | Generation | Failure-triggered repair |
|---|---|---|---|
| Common | Common-50 | Common-50 | Common-50 |
| Generation-specialized | Base | Generation-40 | Common-50 |
| Training-free | Base | Base | Base |

Each candidate uses structured understanding, selective workbook inspection,
API grounding, `transform(wb)` generation, preflight checks, execution, and
up to **two LLM localized repairs** when observable checks fail. Each trial
loads the original input workbook. Domain alone enables the established
sheet-creation/rename preflight fix. V1 checks are unchanged.

The final consistency selector reads **only the first visible input/output
pair** for each candidate. It groups output fingerprints containing cell
values, formula expressions, stored styles, and selected layout information.
It selects the earliest candidate in the largest group with at least two
members. Without agreement it takes the first openable candidate, falling
back to candidate zero if all fail. The same chosen program is used for all
workbook variants. The selector never consults reference answers or evaluator
labels; the choice is recorded before selected accuracy is summarized.

**No additional semantic review is enabled in the best consistency profiles.**
Executable programs are not automatically rewritten by a semantic reviewer.
The optional trained review branch below is separate from failure repair.

Outputs:

```text
runs/v1_mixed/
  manifest.json
  candidates/0..3/       # Codes, raw responses, diagnostics, workbook outputs
  selected/
    choices/             # Label-blind decisions and observations
    codes/               # Final selected transform per question
    spreadsheet/         # All output workbook variants
    run_results.jsonl
    summary.json         # Raw/local hard, soft, average and execution accuracy
```

`--select-only` recomputes selection without a running model, using the same
arguments and existing complete candidates. Candidate generation executes in
a bubblewrap sandbox by default. `--unsafe-local-execution` is an explicit
legacy compatibility option, not a security boundary; do not use it for
untrusted programs. Sandbox failure stops the run rather than marking every
question incorrect.

## 5. Optional semantic review experiment

Serve `REVIEW_ADAPTER` as above, then run a separate one-review-per-program
experiment. Triggering uses visible output availability, never oracle failure.

```bash
bash scripts/review.sh --dataset-dir data/all_data_912_v0.1 \
  --source runs/v1_mixed/selected --out runs/v1_review \
  --base-url http://127.0.0.1:8001/v1 --model review
```

The reviewer sees the instruction, current code, and bounded input/output
observations. `KEEP` preserves the program; `REVISE` executes the new program
on the original inputs. A malformed review keeps the original; a valid but
incorrect revision is not rescued using reference labels. Review logs and
predictions are separate from the source run. Replace dataset/source paths to
run the same optional experiment on Domain.

## 6. Microsoft Excel recalculation and evaluation

A separate **Windows machine with Microsoft Excel** must expose the existing
recalculation API and be reachable from the evaluation process. Linux
openpyxl does not calculate Excel formulas. An SSH tunnel may expose the
service at `127.0.0.1:15000`. No Windows server is installed by the Linux setup.

```bash
export EXCEL_API_URL="http://127.0.0.1:15000"
export EXCEL_ENGINE_LABEL="Microsoft Excel Windows"
curl --fail "$EXCEL_API_URL/health"
# Require ready=true and excel_pool_healthy=true before a full run.

bash scripts/recalculate.sh run \
  --dataset-dir data/all_data_912_v0.1 --run-dir runs/v1_mixed/selected \
  --out-dir reports/excel_rescore/v1_mixed --download-official

bash scripts/recalculate.sh run \
  --dataset-dir data/domain --run-dir runs/domain_free/selected \
  --out-dir reports/excel_rescore/domain_free --download-official
```

This is **prediction-only recalculation**: copies of predicted workbooks are
sent to Excel; supplied reference workbooks remain unchanged. Historical
predictions, training rewards, and the local evaluator are not overwritten.
The upstream comparator is downloaded explicitly at pinned revision
`389be8c29fed059e3b2072741c8231e86d1dee8b`, with source hashes and its license.
This preserves the project's implemented evaluation protocol rather than
silently replacing it with recalculation of reference workbooks.

Reports include `raw_local`, `raw_official`, `excel_local`, and `excel_official`.
Service failures remain unresolved, not automatically incorrect. Use the same
command with `--retry-failed` to retry them. `prepare` in place of `run` performs
raw comparisons without an Excel service; `--local-only` explicitly skips the
official comparator. Review `case_results.jsonl` and `summary.json` under each
report directory, including completeness and error counts.

Metrics are **hard accuracy** (all variants correct), **soft accuracy** (mean
per-question fraction correct), **average accuracy** (all correct cases / all
cases), and **execution success rate** (successful executions / all cases).
Do not label K4 selected accuracy as single-sample Pass@1.

## 7. Reference results and reproducibility

The following are archived **raw/local**, not newly measured Excel results:

| Dataset | Selected profile | Hard accuracy | Average accuracy | Execution success |
|---|---|---:|---:|---:|
| SpreadsheetBench V1 | Common + Generation, K4 consistency | 27.08% (247/912) | 32.80% (894/2,726) | 98.42% (2,683/2,726) |
| Domain-Spreadsheet | Common + Generation, K4 consistency | 29.12% (484/1,662) | 29.12% (484/1,662) | 99.52% (1,654/1,662) |

Source: the research workspace's
`reports/experiment_updates/20261009_main_results_compact.csv`. These figures
are also recorded in `provenance/reference_results.json`. They
identify the profiles retained here, not guaranteed scores of a fresh run.
The portable release adds fail-closed execution, local path configuration,
and version pins. Historical sampling/server versions and GPU scheduling can
differ. A fresh training and benchmark run is required to claim reproduced
scores. Full native-Excel reports should be reported separately from raw ones.

## Layout

```text
configs/                  Three explicit training configs
recipes/                  Compressed, frozen training prompts/schedules
provenance/               Source and adapter hashes
patches/                  Small pinned verl compatibility patch
scripts/                  Environment, data, training, serving and evaluation
src/spreadsheet_agent/    Training, reward, review and consistency orchestration
src/spreadsheet_code_agent_v0/  Reused grounding/checker/repair harness
src/spreadsheetbench_lite/     Reused data loader, model client and evaluator
src/spreadsheet_code_agent_rl/ Reused common-policy environment/data conversion
src/spreadsheet_recalc/        Preserved prediction-only Excel pipeline
tests/                    CPU regression tests
```

Maintainer export utilities are not runtime dependencies. Model weights,
workbooks, conda environments, and experiment outputs are intentionally
excluded. See `THIRD_PARTY.md` before distributing the release or its data.
