# Books full MemRec — self-host LLM baseline contract

**Status (2026-09-25):** real-LLM 30-user smoke v2 passed and its promotion
artifacts were copied locally with matching SHA-256. Full 2,000-user dev run
is the next gate; there is **no full-MemRec baseline result yet**. The 5,377
held-out labels remain sealed.

## Fixed experiment contract

| Component | Preregistered value |
|---|---|
| Primary model | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` |
| Hugging Face revision | `5a5a776300a41aaa681dd7ff0106608ef2bc90db` |
| Backend | `vllm==0.10.2`, `transformers==4.55.4`, OpenAI-compatible server, one H100, tensor parallel `1` |
| Serving | localhost only; `--gpu-memory-utilization 0.60`, `--max-model-len 16384`, `--max-num-seqs 1`, `--seed 42`, `--dtype auto` |
| Decoding | MemRec config: temperature `0`, max output `4000` tokens, strict JSON schema; no prompt/rule tuning |
| Candidate/evaluation | Original Books fixed 10 candidates, unchanged order, dev cohort first; no test-label Stage-W |
| Warm-up | smoke: 30 dev users; full dev: all 7,377 users before scoring 2,000 dev users |
| Request caps | smoke expected `150`, cap `165`; full dev expected `26,131`, cap `28,745` physical requests before any caching |

The model card is [Qwen's published checkpoint](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8); it describes a non-thinking FP8 MoE model with 30.5B total/3.3B activated parameters. This model choice is made for a stronger full-MemRec baseline, **not** because it has already shown good ranking results. The backend version and single-GPU serving mode follow [vLLM's 0.10.2 installation guide](https://docs.vllm.ai/en/v0.10.2/getting_started/installation/gpu.html) and [OpenAI-compatible server guide](https://docs.vllm.ai/en/v0.10.2/serving/openai_compatible_server.html). The backend must be installed in `memrec-hnv/envs/llm-hnv`, with all caches/checkpoint under `memrec-hnv/`; it must never touch the TTS environment.

If the pinned model/backend cannot load or does not support MemRec's JSON
schema, this contract **fails**. A technical replacement requires a new run
ID/config and a new 20–30-user smoke before looking at performance; the model
cannot be changed to rescue low development scores. Comparisons with the
method use this *same* model revision and serving contract. Paper GPT-4o-mini
numbers are only contextual because LLM conditions differ.

## Execution gates and artifacts

1. Validate code/config offline and confirm exactly one running authorized
   allocation. In its Slurm step, snapshot utilization and memory of all four
   GPUs; select the lowest-utilization sufficiently idle H100, then lowest
   memory, and expose only that GPU to the backend and client.
2. Download only the pinned checkpoint using
   `scripts/download_books_memrec_model.py` into the project's own `models/`
   directory. Record revision, file hashes/size and `git rev-parse HEAD`; do
   not duplicate the checkpoint. Install backend in its own venv, log exact
   Python/torch/vLLM versions.
3. Start one localhost server via
   `scripts/run_books_memrec_selfhost_gpu.sh` for run
   `books-memrec-llm-smoke-v2-hnv` with an
   exact PID file and bounded health wait. Run `scripts/run_train.py` serially
   with `--eval-cohort dev --n_eval_users 30 --warmup-user-scope eval` and the
   fixed benchmark YAML. Run `scripts/check_books_memrec_smoke.py` on its
   outputs; refuse promotion on any malformed score, missing stage, request
   mismatch, invalid candidate permutation, or held-out output.
4. Terminate only this run's server PID on success *or failure*, verify VRAM
   returns to baseline; only then write `promotion.json`. Retain before/after
   GPU snapshots, logs, metrics, predictions, hashes and exit status. Pull/copy artifacts locally and verify
   hashes before interpretation. **A passing GPU/cleanup gate is required in
   addition to the Python smoke gate.**
5. The smoke **writes** to an exact-input LLM-response cache but does not read
   from it, preserving the 150-physical-request gate. Full dev may read it
   only under the same pinned namespace, so identical smoke calls are not
   billed again. Full dev must verify `promotion.json` first. Note that all-user warm-up
   changes graph state/order, so the number of reusable calls may be small;
   never reuse an output if its input, schema, model or generation settings
   differ. Full dev must not start without a passing smoke and a fresh GPU
   preflight. Held-out evaluation is blocked until method/config are locked.
6. Full dev uses a SQLite per-user memory/prediction journal and an atomic,
   durable physical-request budget. A restart replays only committed Stage-W
   writes and predictions, then resumes at the next user under the same source
   commit/model/config. The cache may save identical requests, but the
   pre-request budget reservation remains charged after a crash. The full
   gate requires 7,377 warm-up journal entries, 2,000 dev predictions, the
   original candidate lists and a consistent request ledger. A completed
   run is marked only **after** server shutdown and GPU-memory release.

## Progress / results

| Gate | Status | Evidence |
|---|---|---|
| Checkpoint/backend contract | Frozen before real inference | This document; revision above |
| Offline smoke gate | Passed | `scripts/check_books_memrec_smoke.py`, 90 local tests passing |
| vLLM venv dependency check | Passed | `vllm==0.10.2`, `transformers==4.55.4` |
| Pinned checkpoint download | Passed | revision `5a5a7763…90db`, model config hashes in smoke artifact |
| Real-LLM 30-user smoke v1 | Aborted before first request | vLLM default compile cache escaped the project root; exact client/server PIDs terminated, GPU 0 returned to 1 MiB; no promotion/result |
| Real-LLM 30-user smoke v2 | **Passed and promoted** | Run `books-memrec-llm-smoke-v2-hnv`, commit `8513bd4`; 30/30 ranking, all Stage-R/RR/W warm-up calls, 150/150 physical requests, 0 schema/failure; GPU 0 returned from 1 to 4 MiB; promotion SHA-256 verified local/remote |
| Full MemRec dev | Ready for preflight | 7,377 warm-up users, 2,000 dev users, expected 26,131 requests, hard cap 28,745; resumable journal CPU dry-run 7,377+30 passed and prediction SHA matched non-journal and replay run |
| Held-out | Sealed | 5,377 user labels untouched |

The SASRec matched dev result `NDCG@5 = 0.321127` is recorded in
[BOOKS_FULL_BASELINE_PROGRESS.md](BOOKS_FULL_BASELINE_PROGRESS.md); it is **not**
a full-MemRec or self-host-LLM result.

Smoke-only NDCG@5 was `0.689081` and Hit@1 `0.533333` on **30 users with only
30-user warm-up**. These numbers are a wiring/quality sanity check, not a
baseline estimate and must not be compared with 2,000-user SASRec or paper
scores. Full warm-up changes collaborative memory for later users.
