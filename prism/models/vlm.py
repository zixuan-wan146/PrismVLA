from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from prism.models.config import Qwen35BackboneConfig
from prism.models.history_qformer import HistoryMemoryOutput, HistoryQFormer
from prism.models.query_features import gather_layerwise_action_queries

if TYPE_CHECKING:
    from prism.schema import PolicyInput


@dataclass(frozen=True)
class QueryBackboneOutput:
    layerwise_query_features: tuple[torch.Tensor, ...]
    query_valid_mask: torch.Tensor


@dataclass(frozen=True)
class QueryMemoryEncoderOutput:
    layerwise_query_features: tuple[torch.Tensor, ...]
    query_valid_mask: torch.Tensor
    memory: HistoryMemoryOutput


@dataclass(frozen=True)
class EncodedHistoryObservation:
    """Two-camera visual tokens retained between policy decisions."""

    tokens: torch.Tensor
    valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.tokens.ndim != 2 or not torch.is_floating_point(self.tokens):
            raise ValueError("history observation tokens must be floating with shape [tokens, hidden]")
        if self.valid_mask.dtype != torch.bool or self.valid_mask.shape != self.tokens.shape[:1]:
            raise ValueError("history observation valid_mask must be boolean with shape [tokens]")
        if self.tokens.shape[0] <= 0 or self.tokens.shape[1] <= 0:
            raise ValueError("history observation tokens must be non-empty")


@dataclass(frozen=True)
class PreparedQueryMemoryBatch:
    current_inputs: Mapping[str, torch.Tensor]
    history_inputs: Mapping[str, torch.Tensor]
    history_step_ages: torch.Tensor
    history_valid_mask: torch.Tensor


class Qwen35ActionQueryBackbone(nn.Module):
    """Bare, truncated Qwen3.5 multimodal backbone with learned action queries."""

    def __init__(self, model: nn.Module, processor: Any, config: Qwen35BackboneConfig) -> None:
        super().__init__()
        config.validate()
        self.model = model
        self.processor = processor
        self.config = config
        self.action_queries = nn.Parameter(torch.empty(config.num_action_queries, config.hidden_size))
        nn.init.normal_(self.action_queries, mean=0.0, std=0.02)
        self._validate_loaded_model()

    @classmethod
    def from_pretrained(
        cls,
        config: Qwen35BackboneConfig | None = None,
        *,
        local_files_only: bool | None = None,
    ) -> "Qwen35ActionQueryBackbone":
        from transformers import AutoConfig, AutoModel, AutoProcessor

        config = Qwen35BackboneConfig() if config is None else config
        config.validate()
        if local_files_only is not None and type(local_files_only) is not bool:
            raise TypeError("local_files_only override must be a boolean or null")
        load_local_only = config.local_files_only if local_files_only is None else local_files_only
        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config.torch_dtype]
        hf_config = AutoConfig.from_pretrained(config.model_name, local_files_only=load_local_only)
        if getattr(hf_config, "model_type", None) != "qwen3_5":
            raise ValueError(
                f"Expected a qwen3_5 checkpoint, got model_type={getattr(hf_config, 'model_type', None)!r}"
            )
        text_config = hf_config.text_config
        if text_config.hidden_size != config.hidden_size or text_config.num_hidden_layers < config.num_hidden_layers:
            raise ValueError(
                f"Checkpoint text config is incompatible: hidden={text_config.hidden_size}, "
                f"layers={text_config.num_hidden_layers}"
            )
        text_config.num_hidden_layers = config.num_hidden_layers
        text_config.layer_types = list(text_config.layer_types[: config.num_hidden_layers])
        if hasattr(text_config, "mtp_num_hidden_layers"):
            text_config.mtp_num_hidden_layers = 0
        processor = AutoProcessor.from_pretrained(
            config.model_name,
            local_files_only=load_local_only,
            size={"shortest_edge": config.image_size**2, "longest_edge": config.image_size**2},
        )
        model = AutoModel.from_pretrained(
            config.model_name,
            config=hf_config,
            dtype=dtype,
            local_files_only=load_local_only,
            attn_implementation="sdpa",
        )
        return cls(model=model, processor=processor, config=config)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def prepare_current_batch(
        self,
        images: Sequence[Sequence[np.ndarray]],
        instructions: Sequence[str],
    ) -> Mapping[str, torch.Tensor]:
        if len(images) != len(instructions):
            raise ValueError("images and instructions must have the same batch size")
        conversations = []
        for sample_images, instruction in zip(images, instructions):
            if len(sample_images) != 2:
                raise ValueError("Each current observation must contain exactly two ordered camera images")
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": np.asarray(sample_images[0], dtype=np.uint8)},
                            {"type": "image", "image": np.asarray(sample_images[1], dtype=np.uint8)},
                            {"type": "text", "text": str(instruction)},
                        ],
                    }
                ]
            )
        batch = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        return {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}

    def prepare_history_images(
        self,
        history_images: Sequence[Sequence[Sequence[np.ndarray]]],
    ) -> Mapping[str, torch.Tensor]:
        flat_images: list[np.ndarray] = []
        for sample_history in history_images:
            if len(sample_history) != 2:
                raise ValueError("Each sample must contain exactly two historical observations")
            for historical_observation in sample_history:
                if len(historical_observation) != 2:
                    raise ValueError("Each historical observation must contain exactly two ordered camera images")
                flat_images.extend(np.asarray(image, dtype=np.uint8) for image in historical_observation)
        return self._prepare_image_batch(flat_images)

    def prepare_history_observation(
        self,
        images: Sequence[np.ndarray],
    ) -> Mapping[str, torch.Tensor]:
        """Prepare one transient two-camera observation for background encoding."""

        if len(images) != 2:
            raise ValueError("A historical observation must contain exactly two ordered camera images")
        return self._prepare_image_batch([np.asarray(image, dtype=np.uint8) for image in images])

    def _prepare_image_batch(self, images: Sequence[np.ndarray]) -> Mapping[str, torch.Tensor]:
        if not images:
            raise ValueError("images must contain at least one image")
        batch = self.processor.image_processor(images=list(images), return_tensors="pt")
        return {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}

    def forward(self, **prepared_inputs: torch.Tensor) -> QueryBackboneOutput:
        required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        missing = sorted(required - set(prepared_inputs))
        if missing:
            raise ValueError(f"Missing prepared Qwen inputs: {missing}")
        model_device = self.device
        inputs = {key: value.to(model_device) for key, value in prepared_inputs.items()}
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is None:
            mm_token_type_ids = (input_ids == self.model.config.image_token_id).to(dtype=torch.int32)

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        image_outputs = self.model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        image_features = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask = (input_ids == self.model.config.image_token_id).unsqueeze(-1)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        batch_size = input_ids.shape[0]
        query_embeddings = self.action_queries.to(inputs_embeds.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat((inputs_embeds, query_embeddings), dim=1)
        query_attention = torch.ones(
            batch_size,
            self.config.num_action_queries,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat((attention_mask, query_attention), dim=1)
        dummy_token_id = self.model.config.text_config.eos_token_id
        dummy_ids = torch.full(
            (batch_size, self.config.num_action_queries),
            int(dummy_token_id),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        extended_input_ids = torch.cat((input_ids, dummy_ids), dim=1)
        query_types = torch.zeros(
            batch_size,
            self.config.num_action_queries,
            dtype=mm_token_type_ids.dtype,
            device=mm_token_type_ids.device,
        )
        extended_token_types = torch.cat((mm_token_type_ids, query_types), dim=1)
        position_ids, _ = self.model.get_rope_index(
            extended_input_ids,
            extended_token_types,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        language_output = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        action_query_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        action_query_mask[:, -self.config.num_action_queries :] = True
        layerwise_queries = gather_layerwise_action_queries(
            language_output.hidden_states,
            action_query_mask,
            num_backbone_layers=self.config.num_hidden_layers,
            num_action_queries=self.config.num_action_queries,
            hidden_size=self.config.hidden_size,
        )
        return QueryBackboneOutput(
            layerwise_query_features=layerwise_queries,
            query_valid_mask=torch.ones(
                batch_size,
                self.config.num_action_queries,
                dtype=torch.bool,
                device=inputs_embeds.device,
            ),
        )

    def encode_images(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self.model.get_image_features(
            pixel_values.to(self.device),
            image_grid_thw.to(self.device),
            return_dict=True,
        )
        return tuple(feature for feature in outputs.pooler_output)

    def _validate_loaded_model(self) -> None:
        language_model = getattr(self.model, "language_model", None)
        visual = getattr(self.model, "visual", None)
        if language_model is None or visual is None:
            raise ValueError("Qwen3.5 bare multimodal model must expose language_model and visual modules")
        if len(language_model.layers) != self.config.num_hidden_layers:
            raise ValueError(
                f"Loaded language model has {len(language_model.layers)} layers, expected {self.config.num_hidden_layers}"
            )
        if language_model.config.hidden_size != self.config.hidden_size:
            raise ValueError("Loaded Qwen hidden size does not match the accepted config")
        if hasattr(self.model, "lm_head") or hasattr(self.model, "mtp"):
            raise ValueError("The bare VLA backbone must not construct vocabulary or MTP heads")


class Qwen35QueryMemoryEncoder(nn.Module):
    def __init__(
        self,
        backbone: Qwen35ActionQueryBackbone,
        history_qformer: HistoryQFormer | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.history_qformer = HistoryQFormer() if history_qformer is None else history_qformer

    def prepare_current_requests(self, requests: Sequence[Any]) -> Mapping[str, torch.Tensor]:
        """Prepare only current images and prompts for runtime inference."""

        if not requests:
            raise ValueError("requests must contain at least one policy request")
        benchmark = requests[0].benchmark
        view_names = tuple(requests[0].images_by_view)
        if len(view_names) != 2:
            raise ValueError(f"The query-memory encoder requires exactly two ordered views, got {view_names}")

        current_images: list[tuple[np.ndarray, np.ndarray]] = []
        instructions: list[str] = []
        for request in requests:
            if request.benchmark != benchmark:
                raise ValueError("A query-memory batch cannot mix benchmark contracts")
            if tuple(request.images_by_view) != view_names:
                raise ValueError(f"Every request must preserve the ordered views {view_names}")
            current_images.append(tuple(np.asarray(request.images_by_view[name]) for name in view_names))
            instructions.append(request.prompt)
        return self.backbone.prepare_current_batch(current_images, instructions)

    def prepare_requests(self, requests: Sequence[PolicyInput]) -> PreparedQueryMemoryBatch:
        if not requests:
            raise ValueError("requests must contain at least one policy request")
        view_names = tuple(requests[0].images_by_view)

        history_images: list[tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]] = []
        history_step_ages: list[np.ndarray] = []
        history_valid_mask: list[np.ndarray] = []
        for request in requests:
            if tuple(request.images_by_view) != view_names or tuple(request.history_images_by_view) != view_names:
                raise ValueError(f"Every request must preserve the ordered views {view_names}")
            per_view_history = [np.asarray(request.history_images_by_view[name]) for name in view_names]
            if any(images.shape[0] != 2 for images in per_view_history):
                raise ValueError("Every request must contain exactly two historical observations per view")

            history_images.append(
                tuple(
                    tuple(per_view_history[view_index][history_index] for view_index in range(2))
                    for history_index in range(2)
                )
            )
            history_step_ages.append(np.asarray(request.history_step_ages, dtype=np.int64))
            history_valid_mask.append(np.asarray(request.history_valid_mask, dtype=np.bool_))

        return PreparedQueryMemoryBatch(
            current_inputs=self.prepare_current_requests(requests),
            history_inputs=self.backbone.prepare_history_images(history_images),
            history_step_ages=torch.from_numpy(np.stack(history_step_ages)),
            history_valid_mask=torch.from_numpy(np.stack(history_valid_mask)),
        )

    def encode_history_observation(
        self,
        images: Sequence[np.ndarray],
    ) -> EncodedHistoryObservation:
        """Encode one transient two-camera observation and retain only visual tokens."""

        prepared = self.backbone.prepare_history_observation(images)
        image_features = self.backbone.encode_images(
            prepared["pixel_values"],
            prepared["image_grid_thw"],
        )
        if len(image_features) != 2:
            raise ValueError(f"Expected two encoded camera views, got {len(image_features)}")
        first_view, second_view = image_features
        if first_view.ndim != 2 or second_view.ndim != 2 or first_view.shape[-1] != second_view.shape[-1]:
            raise ValueError("Encoded camera views must have shape [tokens, hidden] with a shared hidden size")
        tokens = torch.cat((first_view, second_view), dim=0).detach()
        return EncodedHistoryObservation(
            tokens=tokens,
            valid_mask=torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device),
        )

    def build_history_memory(
        self,
        observations: Sequence[EncodedHistoryObservation],
        history_step_ages: Sequence[int],
    ) -> HistoryMemoryOutput:
        """Compress two already encoded observations and releaseable visual-token inputs."""

        if len(observations) != self.history_qformer.config.num_history_frames:
            raise ValueError(
                f"Expected {self.history_qformer.config.num_history_frames} encoded history observations, "
                f"got {len(observations)}"
            )
        ages = tuple(int(age) for age in history_step_ages)
        if len(ages) != self.history_qformer.config.num_history_frames:
            raise ValueError("history_step_ages must match the configured history frame count")
        history_tokens, history_token_mask = pack_encoded_history_observations(observations)
        return self.history_qformer(
            history_tokens,
            torch.tensor([ages], dtype=torch.long, device=history_tokens.device),
            torch.ones(1, len(observations), dtype=torch.bool, device=history_tokens.device),
            history_token_mask,
        )

    def empty_history_memory(self, batch_size: int = 1) -> HistoryMemoryOutput:
        return self.history_qformer.empty_memory(batch_size)

    def forward_current_with_memory(
        self,
        current_inputs: Mapping[str, torch.Tensor],
        memory: HistoryMemoryOutput,
    ) -> QueryMemoryEncoderOutput:
        """Encode the current request while consuming precomputed fixed-size memory."""

        current_output = self.backbone(**current_inputs)
        batch_size = current_output.query_valid_mask.shape[0]
        expected_tokens = (
            batch_size,
            self.history_qformer.config.num_memory_tokens,
            self.history_qformer.config.hidden_size,
        )
        if memory.tokens.shape != expected_tokens or not torch.is_floating_point(memory.tokens):
            raise ValueError(f"memory tokens must be floating with shape {expected_tokens}")
        if memory.valid_mask.dtype != torch.bool or memory.valid_mask.shape != expected_tokens[:2]:
            raise ValueError(f"memory valid_mask must be boolean with shape {expected_tokens[:2]}")
        return QueryMemoryEncoderOutput(
            layerwise_query_features=current_output.layerwise_query_features,
            query_valid_mask=current_output.query_valid_mask,
            memory=memory,
        )

    def forward_prepared(self, batch: PreparedQueryMemoryBatch) -> QueryMemoryEncoderOutput:
        return self(
            current_inputs=batch.current_inputs,
            history_pixel_values=batch.history_inputs["pixel_values"],
            history_image_grid_thw=batch.history_inputs["image_grid_thw"],
            history_step_ages=batch.history_step_ages,
            history_valid_mask=batch.history_valid_mask,
        )

    def forward(
        self,
        *,
        current_inputs: Mapping[str, torch.Tensor],
        history_pixel_values: torch.Tensor,
        history_image_grid_thw: torch.Tensor,
        history_step_ages: torch.Tensor,
        history_valid_mask: torch.Tensor,
    ) -> QueryMemoryEncoderOutput:
        current_output = self.backbone(**current_inputs)
        image_features = self.backbone.encode_images(history_pixel_values, history_image_grid_thw)
        history_tokens, history_token_mask = pack_two_camera_history_features(
            image_features,
            batch_size=history_valid_mask.shape[0],
        )
        memory = self.history_qformer(
            history_tokens,
            history_step_ages.to(history_tokens.device),
            history_valid_mask.to(history_tokens.device),
            history_token_mask,
        )
        return QueryMemoryEncoderOutput(
            layerwise_query_features=current_output.layerwise_query_features,
            query_valid_mask=current_output.query_valid_mask,
            memory=memory,
        )


def pack_encoded_history_observations(
    observations: Sequence[EncodedHistoryObservation],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not observations:
        raise ValueError("observations must contain at least one encoded history observation")
    hidden_size = observations[0].tokens.shape[-1]
    device = observations[0].tokens.device
    dtype = observations[0].tokens.dtype
    if any(observation.tokens.shape[-1] != hidden_size for observation in observations):
        raise ValueError("Encoded history observations must share a hidden size")
    if any(observation.tokens.device != device or observation.tokens.dtype != dtype for observation in observations):
        raise ValueError("Encoded history observations must share device and dtype")

    max_tokens = max(observation.tokens.shape[0] for observation in observations)
    packed = torch.zeros(
        1,
        len(observations),
        max_tokens,
        hidden_size,
        dtype=dtype,
        device=device,
    )
    valid_mask = torch.zeros(
        1,
        len(observations),
        max_tokens,
        dtype=torch.bool,
        device=device,
    )
    for history_index, observation in enumerate(observations):
        token_count = observation.tokens.shape[0]
        packed[0, history_index, :token_count] = observation.tokens
        valid_mask[0, history_index, :token_count] = observation.valid_mask
    return packed, valid_mask


def pack_two_camera_history_features(
    image_features: Sequence[torch.Tensor],
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_images = int(batch_size) * 2 * 2
    if len(image_features) != expected_images:
        raise ValueError(f"Expected {expected_images} history image feature tensors, got {len(image_features)}")
    per_observation: list[torch.Tensor] = []
    for index in range(0, len(image_features), 2):
        first_view, second_view = image_features[index], image_features[index + 1]
        if first_view.ndim != 2 or second_view.ndim != 2 or first_view.shape[-1] != second_view.shape[-1]:
            raise ValueError("Each encoded camera view must have shape [tokens, hidden] with a shared hidden size")
        per_observation.append(torch.cat((first_view, second_view), dim=0))
    max_tokens = max(features.shape[0] for features in per_observation)
    hidden_size = per_observation[0].shape[-1]
    packed = per_observation[0].new_zeros(batch_size, 2, max_tokens, hidden_size)
    valid_mask = torch.zeros(batch_size, 2, max_tokens, dtype=torch.bool, device=packed.device)
    for flat_index, features in enumerate(per_observation):
        batch_index, history_index = divmod(flat_index, 2)
        packed[batch_index, history_index, : features.shape[0]] = features
        valid_mask[batch_index, history_index, : features.shape[0]] = True
    return packed, valid_mask
