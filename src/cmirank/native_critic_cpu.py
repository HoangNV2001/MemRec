"""CPU/meta rehearsal of the native token-classification critic interface.

Only miniature random weights use CPU memory. No official full-model weights,
CUDA, optimizer updates, generated actions or recommendation labels.
"""

from copy import deepcopy
import gc
from pathlib import Path
import tempfile

from .native_critic import (check_native_critic_gradients, freeze_native_critic,
                           native_critic_contract, native_critic_values, parameter_shapes)


def rehearse_native_critic(official, rows, private):
    import torch
    from transformers import AutoModelForImageTextToText, AutoModelForTokenClassification

    torch.set_num_threads(4)
    torch.manual_seed(42)
    critic_config = deepcopy(official)
    critic_config.num_labels = 1
    critic_config.classifier_dropout = 0.0
    with torch.device("meta"):
        actor_meta = AutoModelForImageTextToText.from_config(official, attn_implementation="sdpa", dtype=torch.bfloat16)
        critic_meta = AutoModelForTokenClassification.from_config(critic_config, attn_implementation="sdpa", dtype=torch.bfloat16)
    full_contract = native_critic_contract(parameter_shapes(actor_meta), parameter_shapes(critic_meta),
        official.text_config.hidden_size, tied_lm_head=bool(official.text_config.tie_word_embeddings), official=True)
    del actor_meta, critic_meta
    tiny = deepcopy(official.to_dict())
    tiny["text_config"].update(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=2, layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [5, 5, 6], "mrope_interleaved": True})
    tiny["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64,
        num_heads=2, out_hidden_size=64, num_position_embeddings=16)
    tiny.update(num_labels=1, classifier_dropout=0.0, id2label={0: "LABEL_0"}, label2id={"LABEL_0": 0})
    mini_config = type(official).from_dict(tiny)
    mini_config.num_labels = 1  # Override any inherited id2label mapping too.
    actor = AutoModelForImageTextToText.from_config(mini_config, attn_implementation="sdpa", dtype=torch.bfloat16)
    probes = []
    with tempfile.TemporaryDirectory(prefix="native-critic-mini-", dir=private / "cache/tmp") as tmp:
        actor.save_pretrained(tmp, safe_serialization=True)
        # Same native factory as VeRL's successful first branch; SDPA because
        # this is CPU-only. The production GPU call stays unmodified FA2.
        critic, info = AutoModelForTokenClassification.from_pretrained(tmp,
            config=deepcopy(mini_config), local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
        critic_class = type(critic).__name__
        if (set(info.get("missing_keys", [])) != {"score.weight", "score.bias"}
                or not set(info.get("unexpected_keys", [])).issubset({"lm_head.weight"})
                or info.get("mismatched_keys") or info.get("error_msgs")):
            raise ValueError("Native critic initializer changed more than the scalar value head")
        mini_contract = native_critic_contract(parameter_shapes(actor), parameter_shapes(critic), 64,
                                               tied_lm_head=bool(mini_config.text_config.tie_word_embeddings))
        freeze_native_critic(critic, torch, mini_contract)
        actor_parameters = dict(actor.named_parameters())
        for name, parameter in critic.named_parameters():
            if not name.startswith("score.") and not torch.equal(parameter, actor_parameters[name]):
                raise ValueError("Native critic did not load the identical miniature actor backbone")
        del actor, actor_parameters
        critic.eval()
        for i, row in enumerate(rows):
            tokens = torch.tensor([row["input_ids"]])
            output = critic(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
            values = native_critic_values(output, tokens, torch)
            values[:, row["prompt_tokens"]:].square().mean().backward()
            norm = check_native_critic_gradients(critic, torch)
            probes.append({"sample": i, "user_id": row["user_id"], "sequence_tokens": tokens.shape[-1],
                           "value_shape": list(values.shape), "backbone_and_head_gradient_norm": norm})
        head = {n: p.detach().clone() for n, p in critic.score.state_dict().items()}
        torch.save(head, Path(tmp) / "head.pt")
        critic.score.load_state_dict(torch.load(Path(tmp) / "head.pt", weights_only=True), strict=True)
        if not all(torch.equal(p, head[n]) for n, p in critic.score.state_dict().items()):
            raise ValueError("Native score head state roundtrip changed values")
        # CPU from_pretrained may mmap checkpoint shards on the shared NFS.
        # Release model and graph references before TemporaryDirectory unlinks
        # them, otherwise NFS retains .nfs* files and rmdir fails with ENOTEMPTY.
        del critic, output, values, tokens, head, parameter
        gc.collect()
    if torch.cuda.is_initialized():
        raise ValueError("Native critic CPU rehearsal initialized CUDA")
    return full_contract, {"status": "NATIVE_CRITIC_MINIATURE_CPU_INTERFACE_PASS_FULL_GPU_GATES_REMAIN",
        "samples": len(probes), "probes": probes, "critic_class": critic_class,
        "native_head": "score", "backbone_weight_transfer_exact": True,
        "head_state_roundtrip_exact": True, "official_full_model_meta_only": True,
        "miniature_weights_only": True, "attention_backend": "sdpa_cpu_not_production_fa2",
        "production_verl_loader_executed": False, "gpu_requested": False,
        "optimizer_updates": 0, "generation_calls": 0, "books_outcomes_accessed": False}
