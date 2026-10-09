"""Outcome-free structural SFT fixtures; no torch, Books loader or reward API."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

from .labels import make_labels
from .parser import parse_action, parse_direct_ranking
from .prompts import render_direct_prompt, render_step_prompt
from .provenance import file_sha256
from .request import RankRequest


def rng(seed: str, split: str, index: int, purpose: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{split}:{index}:{purpose}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def synthetic_request(seed: str, split: str, index: int) -> RankRequest:
    if split not in ("train", "holdout") or index < 0:
        raise ValueError("Explicit disjoint synthetic split required")
    context = rng(seed, split, index, "context")
    topic = context.choice(("astronomy", "history", "gardening", "poetry", "travel", "biology",
                            "architecture", "geography", "philosophy", "music", "cooking", "computing"))
    # Negative synthetic identities never originate from the positive Books IDs.
    uid = -(index + 1 + (10000 if split == "holdout" else 0))
    ids = [uid * 100 - offset for offset in range(10)]
    context.shuffle(ids)
    # All books are exchangeable under the stated preference: random ordering
    # supplies syntax, NOT a synthetic semantic ranking oracle or Books label.
    titles = [f"{context.choice(('Readings', 'Essays', 'Notes', 'Studies'))} in {topic}, volume {j + 1}"
              for j in range(10)]
    return RankRequest.from_stage_rr_inputs(user_id=uid,
        instruction="Rank the candidate books by the user's preferences.",
        candidates=[{"id": item, "title": title} for item, title in zip(ids, titles)],
        item_mems={item: f"A general-reader collection about {topic}. No relative preference among these volumes is known."
                   for item in ids},
        retrieval_bundle={"facets": [{"facet": f"Interest in general reading about {topic}", "confidence": 0.8}]},
        snapshot_id=f"synthetic-format-{split}-v1")


@dataclass(frozen=True)
class FormatExample:
    example_id: str
    mode: str
    request: RankRequest
    active_labels: tuple[str, ...]
    completion: str

    def prompt(self) -> str:
        if self.mode == "direct":
            return render_direct_prompt(self.request)
        if self.mode == "iterative":
            return render_step_prompt(self.request, self.active_labels)
        raise ValueError("Unknown structural training mode")


def format_examples(config: dict) -> list[FormatExample]:
    examples = []
    for index in range(config["train_examples"]):
        request = synthetic_request(config["data_seed"], "train", index)
        labels = list(make_labels(range(10)))
        mode = "iterative" if index % 2 == 0 else "direct"
        action_rng = rng(config["data_seed"], "train", index, "completion")
        if mode == "iterative":
            size = 2 + (index // 2) % 9
            active = sorted(rng(config["data_seed"], "train", index, "active").sample(labels, size))
            answer = action_rng.choice(active)
        else:
            active = list(labels)
            action_rng.shuffle(labels)
            answer = " ".join(labels)
        row = FormatExample(f"synthetic-train-{index:04d}", mode, request,
                            tuple(active), f"<answer>{answer}</answer>")
        valid = (parse_action(row.completion, row.active_labels).valid if mode == "iterative"
                 else parse_direct_ranking(row.completion, make_labels(range(10))).valid)
        if not valid:
            raise AssertionError("Synthetic completion violates the unchanged parser")
        examples.append(row)
    return examples


def encode_completion(tokenizer, prompt: str, completion: str, *, max_input: int, max_output: int) -> dict:
    """Identical inference prefix; loss on the appended response/EOS only."""
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    if not tokenizer.eos_token or tokenizer.eos_token_id is None:
        raise ValueError("Official tokenizer EOS required")
    suffix = tokenizer.encode(completion + tokenizer.eos_token, add_special_tokens=False)
    if (not prefix or not suffix or suffix[-1] != tokenizer.eos_token_id
            or len(prefix) > max_input or len(suffix) > max_output):
        raise ValueError("Training sequence exceeds locked caps or lacks EOS; no truncation")
    if tokenizer.decode(suffix, skip_special_tokens=False) != completion + tokenizer.eos_token:
        raise ValueError("Completion tokenization is not lossless")
    return {"input_ids": prefix + suffix, "labels": [-100] * len(prefix) + suffix,
            "prompt_tokens": len(prefix), "completion_tokens": len(suffix),
            "rendered_prompt_sha256": hashlib.sha256(json.dumps(rendered, sort_keys=True,
                ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()}


def validate_encoded_row(row: dict, config: dict) -> None:
    n = row["prompt_tokens"]
    ids, labels = row["input_ids"], row["labels"]
    if (type(n) is not int or not 0 < n <= config["max_input_tokens"]
            or len(ids) != len(labels) or len(ids) - n != row["completion_tokens"]
            or not 0 < row["completion_tokens"] <= config["max_new_tokens"]
            or labels[:n] != [-100] * n or labels[n:] != ids[n:]
            or any(type(token) is not int or token < 0 for token in ids)):
        raise ValueError("Prompt leakage, malformed response mask or token cap violation")


def load_format_config(root: Path) -> tuple[dict, Path]:
    path = root / "configs/cmirank/format_sft_v1.json"
    config = json.loads(path.read_text())
    if (config["scope"] != "synthetic_format_only_warm_start_not_ranking_training"
            or config["train_examples"] != 256 or config["smoke_train_examples"] != 20
            or config["holdout_users"] != 20 or config["gpu_index"] != 0
            or config["books_training_data_accessed"] or config["ranking_metrics_computed"]
            or config["output_repair"] or config["training_ready"]):
        raise ValueError("Not the approved bounded format-only warm start")
    for phase in ("smoke", "full"):
        count = config["smoke_train_examples"] if phase == "smoke" else config["train_examples"]
        accumulation = config[f"{phase}_gradient_accumulation"]
        if count % accumulation or count * config[f"{phase}_epochs"] // accumulation != config[f"{phase}_updates"]:
            raise ValueError("Fixed SFT budget is inconsistent")
    return config, path


def format_code_hashes(root: Path) -> dict:
    return {name: file_sha256(root / name) for name in (
        "src/cmirank/format_sft.py", "src/cmirank/prompts.py", "src/cmirank/parser.py",
        "src/cmirank/policy_smoke.py", "scripts/cmirank/15_prepare_format_sft_cpu.py",
        "scripts/cmirank/16_train_format_sft_gpu.py")}


def load_prepared_format_data(data: Path, *, commit: str, config_sha: str, code_sha: dict,
                              marker_sha: str, config: dict) -> tuple[list[dict], dict]:
    receipt = json.loads((data / "report.json").read_text())
    expected = {"status": "FORMAT_SFT_CPU_PREPARATION_PASS", "source_commit": commit,
                "config_sha256": config_sha, "code_sha256": code_sha, "checkpoint_marker_sha256": marker_sha,
                "gpu_requested": False, "model_weights_loaded": False, "books_training_data_accessed": False}
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("Exact source/config/mask/tokenizer CPU receipt required")
    for name, digest in receipt["artifact_sha256"].items():
        if name != Path(name).name or file_sha256(data / name) != digest:
            raise ValueError("Prepared synthetic data changed")
    rows = [json.loads(s) for s in (data / "train-tokenized.jsonl").read_text().splitlines()]
    if (len(rows) != config["train_examples"] or len({r["example_id"] for r in rows}) != len(rows)
            or [r["mode"] for r in rows] != ["iterative" if i % 2 == 0 else "direct" for i in range(len(rows))]):
        raise ValueError("Synthetic training recipe changed")
    for row in rows:
        validate_encoded_row(row, config)
    return rows, receipt


def verify_sft_smoke(run: Path, *, commit: str, config_sha: str, data_report_sha: str) -> dict:
    report = json.loads((run / "report.json").read_text())
    cleanup = json.loads((run / "cleanup.json").read_text())
    keeper = json.loads((run / "keeper-handback.json").read_text())
    if (report.get("status") != "FORMAT_SFT_INFRASTRUCTURE_SMOKE_PASS"
            or report.get("source_commit") != commit or report.get("config_sha256") != config_sha
            or report.get("data_report_sha256") != data_report_sha or report.get("optimizer_updates") != 5
            or not report.get("checkpoint_roundtrip_pass") or report.get("books_training_data_accessed")
            or not cleanup.get("gpu_released") or not cleanup.get("allocation_still_running")
            or keeper.get("status") != "KEEPER_GPU0_HANDBACK_VERIFIED"):
        raise ValueError("Full warm-start requires the reviewed matching twenty-example infrastructure smoke")
    return report
