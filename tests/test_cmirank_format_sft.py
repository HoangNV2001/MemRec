"""Structural SFT data/masks/promotion tests, no torch, GPU or network."""

from collections import Counter
import json
from pathlib import Path

import pytest

from src.cmirank.format_sft import (
    encode_completion, format_examples, load_format_config, load_prepared_format_data,
    synthetic_request, validate_encoded_row, verify_sft_smoke,
)
from src.cmirank.parser import parse_action, parse_direct_ranking
from src.cmirank.labels import make_labels
from src.cmirank.provenance import file_sha256

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    eos_token = "|"
    eos_token_id = ord("|")

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return "USER\n" + messages[0]["content"] + "\nASSISTANT\n"

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(map(ord, text))

    def decode(self, ids, *, skip_special_tokens):
        assert not skip_special_tokens
        return "".join(map(chr, ids))


def test_synthetic_recipe_balanced_disjoint_and_valid_without_book_labels():
    config, _ = load_format_config(ROOT)
    examples = format_examples(config)
    assert examples == format_examples(config)
    assert len(examples) == 256 and Counter(r.mode for r in examples) == {"iterative": 128, "direct": 128}
    train = {r.request.user_id for r in examples}
    heldout = [synthetic_request(config["data_seed"], "holdout", i) for i in range(20)]
    assert train.isdisjoint(r.user_id for r in heldout)
    assert all(uid < 0 for uid in train)
    assert {len(r.active_labels) for r in examples if r.mode == "iterative"} == set(range(2, 11))
    for row in examples:
        assert row.request.snapshot_id == "synthetic-format-train-v1"
        assert all(r["id"] < 0 for r in row.request.candidates)
        assert "target_item_id" not in row.prompt() and "reward" not in row.prompt()
        assert (parse_action(row.completion, row.active_labels).valid if row.mode == "iterative"
                else parse_direct_ranking(row.completion, make_labels(range(10))).valid)
    # Labels are not all a fixed first/last candidate preference heuristic.
    assert len({r.completion for r in examples if r.mode == "iterative"}) > 5
    assert len({r.completion for r in examples if r.mode == "direct"}) > 100
    base, _ = load_format_config(ROOT, version=1)
    assert examples == format_examples(base)  # Resource-only retry, not new training data.


@pytest.mark.parametrize("change", [{"learning_rate": 0.00002}, {"full_epochs": 3},
                                   {"data_seed": "different"}, {"gpu_index": 1}])
def test_sft_retry_cannot_change_scientific_fields(tmp_path, change):
    directory = tmp_path / "configs/cmirank"
    directory.mkdir(parents=True)
    for version in (1, 2):
        (directory / f"format_sft_v{version}.json").write_bytes((ROOT / f"configs/cmirank/format_sft_v{version}.json").read_bytes())
    config, path = load_format_config(tmp_path)
    path.write_text(json.dumps({**config, **change}))
    with pytest.raises(ValueError, match="scientific"):
        load_format_config(tmp_path)


def test_completion_mask_never_supervises_prompt_and_includes_eos():
    config, _ = load_format_config(ROOT)
    for example in format_examples(config)[:20]:
        row = encode_completion(Tokenizer(), example.prompt(), example.completion,
                                max_input=4096, max_output=128)
        validate_encoded_row(row, config)
        n = row["prompt_tokens"]
        assert row["labels"][:n] == [-100] * n and row["labels"][n:] == row["input_ids"][n:]
        assert row["labels"][-1] == Tokenizer.eos_token_id
        assert Tokenizer().decode(row["labels"][n:], skip_special_tokens=False) == example.completion + "|"
    with pytest.raises(ValueError, match="no truncation"):
        encode_completion(Tokenizer(), "too long", "<answer>C00</answer>", max_input=3, max_output=128)
    with pytest.raises(ValueError, match="no truncation"):
        encode_completion(Tokenizer(), "prompt", "<answer>C00</answer>", max_input=4096, max_output=1)


@pytest.mark.parametrize("mutation", ["prompt_loss", "response_missing", "wrong_token", "bad_length"])
def test_malformed_training_masks_fail_closed(mutation):
    config, _ = load_format_config(ROOT)
    row = encode_completion(Tokenizer(), "prompt", "<answer>C00</answer>", max_input=4096, max_output=128)
    if mutation == "prompt_loss":
        row["labels"][0] = row["input_ids"][0]
    elif mutation == "response_missing":
        row["labels"][-1] = -100
    elif mutation == "wrong_token":
        row["labels"][-2] = 99
    else:
        row["completion_tokens"] += 1
    with pytest.raises(ValueError, match="mask"):
        validate_encoded_row(row, config)


def test_source_hash_tokenized_data_and_masks_must_match_cpu_receipt(tmp_path):
    config, _ = load_format_config(ROOT)
    rows = []
    for example in format_examples(config):
        rows.append({"example_id": example.example_id, "mode": example.mode,
                     **encode_completion(Tokenizer(), example.prompt(), example.completion, max_input=4096, max_output=128)})
    path = tmp_path / "train-tokenized.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    receipt = {"status": "FORMAT_SFT_CPU_PREPARATION_PASS", "source_commit": "source", "config_sha256": "config",
               "code_sha256": {"code": "sha"}, "checkpoint_marker_sha256": "marker",
               "gpu_requested": False, "model_weights_loaded": False, "books_training_data_accessed": False,
               "artifact_sha256": {path.name: file_sha256(path)}}
    (tmp_path / "report.json").write_text(json.dumps(receipt))
    kwargs = dict(commit="source", config_sha="config", code_sha={"code": "sha"}, marker_sha="marker", config=config)
    assert len(load_prepared_format_data(tmp_path, **kwargs)[0]) == 256
    with pytest.raises(ValueError, match="source"):
        load_prepared_format_data(tmp_path, **{**kwargs, "commit": "newsource"})
    rows[0]["labels"][0] = 3
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="changed"):
        load_prepared_format_data(tmp_path, **kwargs)


@pytest.mark.parametrize("bad", [None, "roundtrip", "release", "source", "updates", "books"])
def test_full_schedule_requires_matching_twenty_example_smoke_and_release(tmp_path, bad):
    report = {"status": "FORMAT_SFT_INFRASTRUCTURE_SMOKE_PASS", "source_commit": "source", "config_sha256": "config",
              "data_report_sha256": "data", "optimizer_updates": 5, "checkpoint_roundtrip_pass": True,
              "books_training_data_accessed": False, "training_ready": False,
              "synthetic_functional_diagnostic": {"status": "FUNCTIONAL_SMOKE_FAILED_NO_PROMOTION"}}
    cleanup = {"gpu_released": True, "allocation_still_running": True}
    keeper = {"status": "KEEPER_GPU0_HANDBACK_VERIFIED"}
    if bad == "roundtrip":report["checkpoint_roundtrip_pass"] = False
    if bad == "release":cleanup["gpu_released"] = False
    if bad == "source":report["source_commit"] = "other"
    if bad == "updates":report["optimizer_updates"] = 6
    if bad == "books":report["books_training_data_accessed"] = True
    for name, value in (("report.json", report), ("cleanup.json", cleanup), ("keeper-handback.json", keeper)):
        (tmp_path / name).write_text(json.dumps(value))
    kwargs = dict(commit="source", config_sha="config", data_report_sha="data")
    if bad is None:
        assert not verify_sft_smoke(tmp_path, **kwargs)["training_ready"]
    else:
        with pytest.raises(ValueError):verify_sft_smoke(tmp_path, **kwargs)
