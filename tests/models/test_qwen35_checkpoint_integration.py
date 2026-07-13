from __future__ import annotations

from dataclasses import replace
import os

import numpy as np
import pytest
import torch

from prism.models.config import Qwen35BackboneConfig
from prism.models.vlm import (
    Qwen35ActionQueryBackbone,
    Qwen35QueryMemoryEncoder,
    pack_two_camera_history_features,
)
from prism.serve.protocol import PolicyRequest


pytestmark = pytest.mark.skipif(
    os.environ.get("PRISM_RUN_MODEL_INTEGRATION") != "1",
    reason="set PRISM_RUN_MODEL_INTEGRATION=1 after caching Qwen/Qwen3.5-0.8B",
)


def test_real_qwen35_checkpoint_produces_accepted_query_and_memory_shapes():
    if not torch.cuda.is_available():
        pytest.skip("the real-checkpoint integration test requires CUDA")

    config = replace(Qwen35BackboneConfig(), local_files_only=True)
    backbone = Qwen35ActionQueryBackbone.from_pretrained(config).eval().to("cuda")
    encoder = Qwen35QueryMemoryEncoder(backbone).eval().to("cuda")
    camera_a = np.full((448, 448, 3), 64, dtype=np.uint8)
    camera_b = np.full((448, 448, 3), 192, dtype=np.uint8)
    request = PolicyRequest(
        benchmark="libero",
        prompt="pick up the red block",
        images_by_view={"agentview_rgb": camera_a, "eye_in_hand_rgb": camera_b},
        history_images_by_view={
            "agentview_rgb": np.stack((camera_a, camera_a)),
            "eye_in_hand_rgb": np.stack((camera_b, camera_b)),
        },
        history_step_ages=np.array([6, 3], dtype=np.int32),
        history_valid_mask=np.array([True, True]),
        state=np.zeros(8, dtype=np.float32),
        action_dim=7,
    )
    prepared = encoder.prepare_requests([request])
    assert prepared.current_inputs["image_grid_thw"].tolist() == [[1, 24, 24], [1, 24, 24]]
    assert prepared.history_inputs["image_grid_thw"].tolist() == [[1, 24, 24]] * 4
    assert (prepared.current_inputs["input_ids"] == backbone.model.config.image_token_id).sum().item() == 288

    observed = {}

    def capture_language_inputs(module, args, kwargs):
        del module, args
        observed["sequence_length"] = kwargs["inputs_embeds"].shape[1]
        observed["query_embeddings"] = kwargs["inputs_embeds"][:, -48:].detach().cpu()
        observed["query_attention"] = kwargs["attention_mask"][:, -48:].detach().cpu()

    def capture_language_outputs(module, args, kwargs, output):
        del module, args, kwargs
        observed["hidden_state_levels"] = len(output.hidden_states)

    input_hook = backbone.model.language_model.register_forward_pre_hook(
        capture_language_inputs,
        with_kwargs=True,
    )
    output_hook = backbone.model.language_model.register_forward_hook(
        capture_language_outputs,
        with_kwargs=True,
    )
    try:
        with torch.inference_mode():
            output = encoder.forward_prepared(prepared)
    finally:
        input_hook.remove()
        output_hook.remove()

    assert len(backbone.model.language_model.layers) == 16
    assert observed["hidden_state_levels"] == 17
    assert observed["sequence_length"] == prepared.current_inputs["input_ids"].shape[1] + 48
    assert observed["query_attention"].bool().all()
    torch.testing.assert_close(
        observed["query_embeddings"][0],
        backbone.action_queries.detach().to(dtype=torch.bfloat16).cpu(),
    )
    assert not hasattr(backbone.model, "lm_head")
    assert not hasattr(backbone.model, "mtp")
    assert sum(parameter.numel() for parameter in backbone.parameters()) == 686_997_632
    assert len(output.layerwise_query_features) == 16
    assert all(features.shape == (1, 48, 1024) for features in output.layerwise_query_features)
    assert all(torch.isfinite(features).all() for features in output.layerwise_query_features)
    assert output.memory.tokens.shape == (1, 24, 512)
    assert output.memory.tokens.dtype == torch.float32
    assert output.memory.valid_mask.all()
    assert torch.isfinite(output.memory.tokens).all()

    with torch.inference_mode():
        image_features = backbone.encode_images(**prepared.history_inputs)
        history_tokens, history_token_mask = pack_two_camera_history_features(image_features, batch_size=1)
    assert [features.shape for features in image_features] == [(144, 1024)] * 4
    assert history_tokens.shape == (1, 2, 288, 1024)
    assert history_token_mask.sum().item() == 576


def test_real_qwen35_processor_handles_padding_and_non_square_smart_resize():
    if not torch.cuda.is_available():
        pytest.skip("the real-checkpoint integration test requires CUDA")

    config = replace(Qwen35BackboneConfig(), local_files_only=True)
    backbone = Qwen35ActionQueryBackbone.from_pretrained(config).eval().to("cuda")
    square = np.zeros((384, 384, 3), dtype=np.uint8)
    padded = backbone.prepare_current_batch(
        [[square, square], [square, square]],
        ["move", "move the red block to the plate and then release the gripper"],
    )
    assert padded["input_ids"].shape[0] == 2
    assert padded["attention_mask"].sum(dim=1).unique().numel() == 2
    assert (padded["input_ids"] == backbone.model.config.image_token_id).sum(dim=1).tolist() == [288, 288]
    with torch.inference_mode():
        padded_output = backbone(**padded)
    assert all(features.shape == (2, 48, 1024) for features in padded_output.layerwise_query_features)
    assert all(torch.isfinite(features).all() for features in padded_output.layerwise_query_features)

    wide = np.zeros((256, 512, 3), dtype=np.uint8)
    tall = np.zeros((512, 256, 3), dtype=np.uint8)
    non_square = backbone.prepare_current_batch([[wide, tall]], ["move"])
    grids = non_square["image_grid_thw"]
    assert all(height != width for _, height, width in grids.tolist())
    expected_visual_tokens = sum(int(time * height * width // 4) for time, height, width in grids.tolist())
    actual_visual_tokens = (non_square["input_ids"] == backbone.model.config.image_token_id).sum().item()
    assert actual_visual_tokens == expected_visual_tokens
