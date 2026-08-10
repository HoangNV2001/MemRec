"""
M3 Part B step 2: LoRA SFT of `LM_Mem` on the selected teacher memories.

Reads the (prompt, best `M_collab`) pairs chosen by ``select_sft_data`` and
fine-tunes the policy on them. §M3 exists because RL from a base model on
structured JSON collapses the format before it learns anything; the DoD is
therefore about format and non-regression, not about beating the teacher:

    valid JSON on val >= 95%   ·   val NDCG@5(SFT) >= val NDCG@5(base)

Three choices that are easy to get wrong here:

**Loss on the completion only.** The prompt carries the neighbour table and is
~85% of every sequence. Training on it spends most of the gradient learning to
reproduce text that is *given* at inference time, and dilutes the part that
matters. Controlled by ``data.mask_prompt``.

**Thinking disabled everywhere.** Qwen3.5 is a reasoning model, and its template
opens a ``<think>`` block. Left alone the policy would emit a chain of thought
before every memory: past §6.2's 384-token budget, inflating the §9.3 length
alarm, and against the one axis this thesis competes on. The SFT targets are
bare JSON, so leaving it on would also make training and generation disagree
about what a completion looks like. ``src.rl.policy.chat_text`` handles it in
one place, shared with the reward ranker, which was bitten by the same template.

**No packing, no truncation of targets.** A truncated `M_collab` is a malformed
one, and teaching the policy to emit malformed JSON is the exact failure §9.4
warns about. Sequences over the budget are dropped and counted instead.

    python -m src.rl.sft --config configs/rl/m3_sft_books.yaml
"""
from pathlib import Path
from typing import Dict, List

import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.policy import chat_text            # noqa: E402
from src.utils import load_config              # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="M3-B: LoRA SFT for LM_Mem")
    p.add_argument("--config", default="configs/rl/m3_sft_books.yaml")
    p.add_argument("--limit", type=int, default=None, help="smoke run on N examples")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--dry_run", action="store_true",
                   help="build the dataset and report shapes, then stop")
    return p.parse_args()


def build_examples(rows: List[Dict], tokenizer, cfg) -> List[Dict]:
    """
    Tokenise into input_ids/labels with the prompt masked out.

    ``chat_text(..., add_generation_prompt=True)`` gives exactly the string the
    policy will be handed at rollout time, so the boundary between "context" and
    "what the model must produce" is the same at train and inference time. Any
    drift here shows up later as a policy that is fluent but starts its answer in
    the wrong place.
    """
    max_prompt = cfg["policy"]["max_prompt_tokens"]
    max_completion = cfg["policy"]["max_completion_tokens"]
    mask_prompt = cfg["data"].get("mask_prompt", True)
    eos = tokenizer.eos_token or ""

    out, dropped_prompt, dropped_completion = [], 0, 0
    for row in rows:
        prompt_ids = tokenizer(chat_text(tokenizer, row["prompt"]),
                               add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(row["completion"] + eos,
                                   add_special_tokens=False)["input_ids"]
        if len(prompt_ids) > max_prompt:
            dropped_prompt += 1
            continue
        # Never truncate the target: a cut-off M_collab is malformed JSON, and
        # teaching the policy to emit that is §9.4's failure mode by hand.
        if len(completion_ids) > max_completion:
            dropped_completion += 1
            continue
        input_ids = prompt_ids + completion_ids
        labels = ([-100] * len(prompt_ids) + completion_ids) if mask_prompt else list(input_ids)
        out.append({"input_ids": input_ids, "labels": labels,
                    "attention_mask": [1] * len(input_ids)})

    if dropped_prompt or dropped_completion:
        print(f"  bo qua: {dropped_prompt} prompt qua dai (>{max_prompt}), "
              f"{dropped_completion} completion qua dai (>{max_completion})")
    return out


class PadCollator:
    """Right-pad a batch; padded label positions are ignored by the loss."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch

        width = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, fill: seq + [fill] * (width - len(seq))   # noqa: E731
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
        }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.output_dir or cfg["output_dir"])

    rows = []
    with open(PROJECT_ROOT / cfg["data"]["train_file"], "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    print(f"SFT data: {len(rows)} vi du tu {cfg['data']['train_file']}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    name = cfg["policy"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = build_examples(rows, tokenizer, cfg)
    lens = sorted(len(e["input_ids"]) for e in examples)
    sup = sorted(sum(1 for x in e["labels"] if x != -100) for e in examples)
    print(f"  {len(examples)} vi du dung duoc · do dai median {lens[len(lens)//2]} "
          f"(p95 {lens[int(0.95*len(lens))]}) · token co loss median {sup[len(sup)//2]} "
          f"({100*sup[len(sup)//2]/lens[len(lens)//2]:.0f}% cua chuoi)")
    if args.dry_run:
        print("dry run: khong train")
        return

    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=getattr(torch, cfg["policy"]["dtype"]), device_map="cuda")
    model.config.use_cache = False

    from peft import LoraConfig, get_peft_model

    lcfg = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"], task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    t = cfg["train"]
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=t["epochs"],
            learning_rate=t["learning_rate"],
            per_device_train_batch_size=t["per_device_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            warmup_ratio=t["warmup_ratio"],
            lr_scheduler_type=t["lr_scheduler_type"],
            weight_decay=t["weight_decay"],
            gradient_checkpointing=t["gradient_checkpointing"],
            logging_steps=t["logging_steps"],
            save_strategy=t["save_strategy"],
            seed=t["seed"],
            bf16=cfg["policy"]["dtype"] == "bfloat16",
            report_to=[],
        ),
        train_dataset=examples,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # §10.6 requires VRAM peak and wall-clock per training run in PROGRESS.md.
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"\n✓ {out_dir}  ·  VRAM peak {peak:.1f} GB")


if __name__ == "__main__":
    main()
