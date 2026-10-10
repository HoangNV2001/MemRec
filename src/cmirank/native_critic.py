"""Strict adapters for VeRL's native Qwen token-classification critic.

No model construction, framework patch or torch import at module import time.
The actor has an LM vocabulary head; the native critic uses a scalar score head.
Derive shared-backbone shapes on CPU/meta rather than equating their totals.
"""

from math import prod


def parameter_shapes(model):
    return {name: list(parameter.shape) for name, parameter in model.named_parameters()}


def shape_scope(shapes, head_prefix="score."):
    return {
        "text_backbone_parameters": sum(prod(s) for n, s in shapes.items()
                                        if ".visual." not in n and not n.startswith(head_prefix)),
        "frozen_vision_parameters": sum(prod(s) for n, s in shapes.items() if ".visual." in n),
        "scalar_head_parameters": sum(prod(s) for n, s in shapes.items() if n.startswith(head_prefix)),
    }


def native_critic_contract(actor_shapes, critic_shapes, hidden_size, *, tied_lm_head=False, official=False):
    removed_head = set() if tied_lm_head else {"lm_head.weight"}
    if (type(tied_lm_head) is not bool
            or set(actor_shapes) - set(critic_shapes) != removed_head
            or set(critic_shapes) - set(actor_shapes) != {"score.weight", "score.bias"}
            or critic_shapes["score.weight"] != [1, hidden_size]
            or critic_shapes["score.bias"] != [1]
            or (tied_lm_head and "lm_head.weight" in actor_shapes)
            or (not tied_lm_head and (len(actor_shapes["lm_head.weight"]) != 2
                                      or actor_shapes["lm_head.weight"][1] != hidden_size))
            or any(actor_shapes[n] != critic_shapes[n] for n in set(actor_shapes) & set(critic_shapes))):
        raise ValueError("Native critic must preserve exactly the actor backbone and replace only the LM head")
    actor_scope = shape_scope(actor_shapes)
    if official and (actor_scope["text_backbone_parameters"], actor_scope["frozen_vision_parameters"]) != (4205751296, 333514240):
        raise ValueError("Official actor full-text/frozen-vision boundary changed")
    return {"class": "Qwen3_5ForTokenClassification", "head_prefix": "score.",
            "head_state_names": ["weight", "bias"], "tied_lm_head": tied_lm_head, "parameter_shapes": critic_shapes,
            "scope": shape_scope(critic_shapes), "actor_scope": actor_scope,
            "removed_actor_head": {n: actor_shapes[n] for n in removed_head}}


def freeze_native_critic(model, torch, contract):
    if type(model).__name__ != contract["class"] or parameter_shapes(model) != contract["parameter_shapes"]:
        raise ValueError("Actual unmodified native critic differs from CPU/meta shape receipt")
    for name, parameter in model.named_parameters():
        if parameter.dtype != torch.bfloat16:
            raise ValueError("Native critic parameters are not BF16")
        parameter.requires_grad_(".visual." not in name)
    if shape_scope(parameter_shapes(model)) != contract["scope"]:
        raise ValueError("Native critic backbone/head/vision scope changed")
    return contract["scope"]


def native_critic_values(output, tokens, torch):
    logits = getattr(output, "logits", None)
    if (not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (*tokens.shape, 1)
            or not torch.isfinite(logits).all()):
        raise ValueError("Native critic must return finite scalar value logits for every token")
    return logits[..., 0].float()


def check_native_critic_gradients(model, torch):
    groups = {"backbone": [], "head": []}
    for name, parameter in model.named_parameters():
        if ".visual." in name:
            if parameter.grad is not None:
                raise ValueError("Frozen vision received critic gradients")
            continue
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise ValueError("Native critic has nonfinite gradients")
            groups["head" if name.startswith("score.") else "backbone"].append(parameter.grad)
    if any(not values or not any(bool(grad.abs().max() > 0) for grad in values)
           for values in groups.values()):
        raise ValueError("Native critic needs nonzero gradients in both backbone and scalar head")
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                         1.0, error_if_nonfinite=True)
    if not torch.isfinite(norm) or norm.item() <= 0:
        raise ValueError("Missing/nonfinite/zero native critic gradients")
    model.zero_grad(set_to_none=True)
    return float(norm.item())
