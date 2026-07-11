from __future__ import annotations

from typing import Any


NO_WEIGHT_DECAY_NAME_PARTS = (
    "norm",
    "layernorm",
    "rmsnorm",
    "gate",
    "pos_encoding",
    "pos_embedding",
    "position_embedding",
    "time_embedding",
    "timestep_embedding",
)


LR_GROUP_PATTERNS = (
    ("gates", ("gate",)),
    ("flow_time_embedding", ("time_pos_enc",)),
    ("timestep_mlp", ("timestep_mlp", "time_mlp")),
    ("noisy_action_encoder", ("action_head.action_encoder",)),
    ("action_pos_embedding", ("action_head.action_encoder.pos_encoding",)),
    ("temporal_pos_embedding", ("temporal_pos_embedding", "time_embedding", "short_memory_time_embedding")),
    ("bridge_attention", ("visual_attn", "action_attn", "visual_cross_attn", "action_cross_attn")),
    ("bridge_adapter", ("bridge_adapter",)),
    ("short_memory_encoder", ("short_memory",)),
    ("short_memory_projector", ("short_memory_adapter",)),
    ("plan_projector", ("plan_adapter", "plan_slot_embeddings", "plan_src_emb")),
    ("progress_condition_projector", ("progress_condition_projector",)),
    ("source_mlp", ("state_encoder", "vlm_src_emb", "mem_src_emb", "state_src_emb")),
    ("action_expert", ("action_head.transformer_blocks",)),
    ("action_head", ("action_head.action_decoder", "norm_out")),
    ("flow_matching_action_head", ("action_head",)),
)


def build_param_groups(model: Any, weight_decay: float, *, base_lr: float, lr_groups: dict[str, Any] | None = None):
    buckets: dict[tuple[str | None, bool, float], list[Any]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group_name = _resolve_lr_group(name, lr_groups)
        group_lr = float(lr_groups[group_name]) if group_name is not None and lr_groups is not None else float(base_lr)
        no_decay = _uses_no_weight_decay(name, parameter)
        buckets.setdefault((group_name, no_decay, group_lr), []).append(parameter)

    if not buckets:
        raise ValueError("No trainable parameters found. Check finetune flags and bridge config.")

    param_groups = []
    for (group_name, no_decay, group_lr), params in buckets.items():
        group = {
            "params": params,
            "lr": group_lr,
            "weight_decay": 0.0 if no_decay else float(weight_decay),
        }
        if group_name is not None:
            group["name"] = f"{group_name}.{'no_decay' if no_decay else 'decay'}"
        param_groups.append(group)
    return param_groups


def _uses_no_weight_decay(name: str, parameter: Any) -> bool:
    lowered = name.lower()
    if name.endswith("bias") or ".bias" in name:
        return True
    if getattr(parameter, "dim", lambda: 0)() <= 1:
        return True
    return any(part in lowered for part in NO_WEIGHT_DECAY_NAME_PARTS)


def _resolve_lr_group(name: str, lr_groups: dict[str, Any] | None) -> str | None:
    if not lr_groups:
        return None
    lowered = name.lower()
    for group_name, patterns in LR_GROUP_PATTERNS:
        if group_name not in lr_groups:
            continue
        if any(pattern in lowered for pattern in patterns):
            return group_name
    return None
