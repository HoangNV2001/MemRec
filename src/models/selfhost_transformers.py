"""Strict single-GPU Transformers client for offline/self-hosted JSON generation."""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Mapping, Sequence


def extract_json_object(text: str) -> Dict[str, Any]:
    value = (text or "").strip()
    if value.startswith("```json"):
        value = value[len("```json") :]
    elif value.startswith("```"):
        value = value[3:]
    if value.endswith("```"):
        value = value[:-3]
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("self-host model did not return a JSON object")
    parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("self-host JSON root must be an object")
    return parsed


def validate_json_schema(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        properties = schema.get("properties", {})
        for key in schema.get("required", ()):  # type: ignore[arg-type]
            if key not in value:
                raise ValueError(f"{path}.{key} is required")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise ValueError(f"{path} has unexpected properties: {sorted(extra)}")
        for key, child in properties.items():
            if key in value:
                validate_json_schema(value[key], child, f"{path}.{key}")
    elif kind == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array")
        item_schema = schema.get("items", {})
        for index, item in enumerate(value):
            validate_json_schema(item, item_schema, f"{path}[{index}]")
    elif kind == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string")
    elif kind == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{path} must be a number")
    elif kind == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{path} must be an integer")
    elif kind == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be a boolean")
    elif kind is not None:
        raise ValueError(f"unsupported local schema type: {kind}")


class TransformersJSONClient:
    """Minimal interface compatible with the temporal pilot's call journal."""

    def __init__(
        self,
        *,
        model_name: str,
        revision: str,
        dtype: str,
        seed: int,
        hard_memory_fraction: float,
        max_model_len: int,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        visible = [value for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
        if len(visible) != 1:
            raise RuntimeError("self-host runner requires CUDA_VISIBLE_DEVICES to contain exactly one physical GPU")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"self-host runner must see exactly one logical GPU, got {torch.cuda.device_count()}")
        if not 0 < hard_memory_fraction <= 0.25:
            raise ValueError("hard_memory_fraction must be in (0, 0.25] for the shared H100 allocation")
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        if dtype not in dtype_map:
            raise ValueError(f"unsupported self-host dtype: {dtype}")
        torch.cuda.set_per_process_memory_fraction(hard_memory_fraction, device=0)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.torch = torch
        self.model_name = model_name
        self.revision = revision
        self.seed = seed
        self.max_model_len = max_model_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            dtype=dtype_map[dtype],
            low_cpu_mem_usage=True,
        ).to("cuda:0")
        self.model.eval()
        resolved = getattr(self.model.config, "_commit_hash", None)
        if resolved and resolved != revision:
            raise RuntimeError(f"resolved model revision {resolved} does not match locked revision {revision}")
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0

    def generate_json(
        self,
        messages: Sequence[Mapping[str, str]],
        properties: Dict[str, Dict[str, Any]],
        temperature: float | None = 0.0,
        max_tokens: int = 256,
        debug_logger: Any = None,
        max_retries: int = 1,
    ) -> Dict[str, Any]:
        if temperature not in (0, 0.0, None):
            raise ValueError("self-host experiment is locked to greedy decoding")
        if max_retries != 1:
            raise ValueError("request retries are controlled only by the outer physical-attempt journal")
        schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
        augmented = [dict(message) for message in messages]
        if not augmented or augmented[-1].get("role") != "user":
            raise ValueError("self-host JSON prompt must end with a user message")
        augmented[-1]["content"] += (
            "\n\nReturn only one compact JSON object matching this exact schema. "
            "Do not use Markdown or add commentary:\n" + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        )
        prompt = self.tokenizer.apply_chat_template(augmented, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_tokens = int(encoded["input_ids"].shape[-1])
        if input_tokens + max_tokens > self.max_model_len:
            raise ValueError(f"prompt plus output cap exceeds max_model_len: {input_tokens}+{max_tokens}>{self.max_model_len}")
        encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, input_tokens:]
        output_tokens = int(generated.shape[-1])
        response_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        response = extract_json_object(response_text)
        validate_json_schema(response, schema)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1
        return response

    def get_token_stats(self) -> Dict[str, Any]:
        peak_bytes = int(self.torch.cuda.max_memory_allocated(0))
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_requests": self.total_requests,
            "peak_vram_bytes": peak_bytes,
            "peak_vram_gib": peak_bytes / (1024**3),
            "model": self.model_name,
            "revision": self.revision,
            "seed": self.seed,
        }
