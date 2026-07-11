from __future__ import annotations

# --- migrated from src/prism/model/internvl3/internvl3_embedder.py ---
from dataclasses import dataclass
import logging
from typing import List, Sequence, Union

import torch
from PIL import Image
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import to_pil_image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class InternVL3EmbeddingOutput:
    fused_tokens: torch.Tensor
    hidden_states: list[torch.Tensor]
    attention_mask: torch.Tensor
    visual_tokens: torch.Tensor | None = None
    planner_vl_summary: torch.Tensor | None = None

# === Image Transformations ===
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

# === Aspect Ratio Handling ===
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

class InternVL3Embedder(nn.Module):
    def __init__(
        self,
        model_name="OpenGVLab/InternVL3-1B",
        image_size=448,
        device="cuda",
        allow_image_token_truncation: bool = False,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.max_text_length = 1024  # InternVL3 supports up to 1024 tokens
        self.allow_image_token_truncation = bool(allow_image_token_truncation)
        self.transform = build_transform(image_size)
        self.local_files_only = bool(local_files_only)
        logging.info("Loading InternVL3 model from %s local_files_only=%s", model_name, self.local_files_only)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
            use_flash_attn=True,
            low_cpu_mem_usage=True,
            _fast_init=False,
        ).to(self.device) 
        
        if hasattr(self.model.language_model, 'model'):
            layers = self.model.language_model.model.layers

        else:
            layers = self.model.language_model.layers
        layers = layers[:14]

        if hasattr(self.model.language_model, 'model'):
            self.model.language_model.model.layers = torch.nn.ModuleList(layers)
        else:
            self.model.language_model.layers = torch.nn.ModuleList(layers)
        self.model.language_model.lm_head = torch.nn.Identity()

        if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "encoder"):
            self.model.vision_model.encoder.gradient_checkpointing = False
        

    def _preprocess_images(
        self,
        image_tensors: List[Union[Image.Image, torch.Tensor]]
    ) -> (torch.Tensor, List[int]):

        pixel_values_list = []
        for i, image in enumerate(image_tensors):
            if isinstance(image, torch.Tensor):
                image = to_pil_image(image)
            tiles = dynamic_preprocess(image, image_size=self.image_size)
            tile_tensors = torch.stack([self.transform(t) for t in tiles])  # (T_i, 3, 448, 448)
            pixel_values_list.append(tile_tensors)

        pixel_values = torch.cat(pixel_values_list, dim=0).to(dtype=torch.bfloat16, device=self.device)
        num_tiles_list = [pv.shape[0] for pv in pixel_values_list]

        return pixel_values, num_tiles_list

    def _build_multimodal_prompt(
        self,
        num_tiles_list: List[int],
        text_prompt: str
    ) -> str:

        prompt = ''
        for i in range(len(num_tiles_list)):
            prompt += f"Image-{i+1}: <image>\n"
        prompt += text_prompt.strip()

        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"

        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        for tile_count in num_tiles_list:
            token_count = self.model.num_image_token * tile_count
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * token_count + IMG_END_TOKEN
            prompt = prompt.replace("<image>", image_tokens, 1)

        return prompt
    
    def _prepare_and_fuse_embeddings(
        self,
        prompt: str,
        vit_embeds: torch.Tensor,
        image_mask: torch.Tensor,
        num_tiles_list: List[int]
    ) -> (torch.Tensor, torch.Tensor):
   
        untruncated_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        true_sequence_length = untruncated_ids.shape[1]

        if true_sequence_length > self.max_text_length:
            logging.warning(
                "Input prompt was truncated: max_length=%s, actual_length=%s, prompt_prefix=%r",
                self.max_text_length,
                true_sequence_length,
                prompt[:100],
            )

        model_inputs = self.tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_text_length).to(self.device)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

       
        img_token_mask = (input_ids == self.img_context_token_id)
     
        img_token_locations = torch.where(img_token_mask)[1]


        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)

        selected = (input_ids == self.img_context_token_id)

        vit_embeds = vit_embeds.reshape(-1, C)
        selected_count = int(selected.sum().item())
        vit_token_count = int(vit_embeds.shape[0])
        if selected_count != vit_token_count:
            message = (
                "Image/text embedding token mismatch: "
                f"selected_img_context_tokens={selected_count}, "
                f"vit_tokens={vit_token_count}, "
                f"prompt_length={true_sequence_length}, "
                f"max_text_length={self.max_text_length}, "
                f"image_count={len(num_tiles_list)}, "
                f"active_image_count={int(torch.as_tensor(image_mask).sum().item())}, "
                f"num_tiles_list={num_tiles_list}"
            )
            if not self.allow_image_token_truncation:
                raise ValueError(message)
            logging.warning("%s; applying explicit truncation fallback", message)
            selected_indices = selected.nonzero(as_tuple=False).reshape(-1)
            copy_count = min(selected_count, vit_token_count)
            if copy_count > 0:
                input_embeds[selected_indices[:copy_count]] = vit_embeds[:copy_count]
        else:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds

 
        tokens_per_tile = self.model.num_image_token 
 
        current_token_idx = 0
        for i in range(len(image_mask)):
           
            num_tiles_for_this_image = num_tiles_list[i]
            num_tokens_for_this_image = num_tiles_for_this_image * tokens_per_tile
       
            if not image_mask[i]:
                
                start_idx = img_token_locations[current_token_idx]
                end_idx = start_idx + num_tokens_for_this_image
               
                attention_mask[0, start_idx:end_idx] = 0
    
            current_token_idx += num_tokens_for_this_image

        input_embeds = input_embeds.reshape(B, N, C)
        return input_embeds, attention_mask


    def get_fused_image_text_embedding_from_tensor_images(
        self,
        image_tensors: list[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        text_prompt: str,
        return_cls_only: bool = True,
        return_hidden_states: bool = False,
        selected_layers: Sequence[int | str] | None = None,
    ):

   
        pixel_values, num_tiles_list = self._preprocess_images(image_tensors)

       
        if pixel_values.shape[0] == 0:
            logging.warning("No valid images to process after masking.")

        vit_embeds = self.model.extract_feature(pixel_values)
        fused_embeds = vit_embeds  
        prompt = self._build_multimodal_prompt(num_tiles_list, text_prompt)
        inputs_embeds, attention_mask = self._prepare_and_fuse_embeddings(prompt, fused_embeds, image_mask, num_tiles_list)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        all_hidden_states = [hidden_state.to(torch.float32) for hidden_state in outputs.hidden_states]
        fused_hidden = all_hidden_states[-1]
        fused_tokens = fused_hidden[:, 0, :] if return_cls_only else fused_hidden
        planner_vl_summary = last_valid_token_summary(fused_hidden, attention_mask)

        if return_hidden_states:
            return InternVL3EmbeddingOutput(
                fused_tokens=fused_tokens,
                hidden_states=select_hidden_states(all_hidden_states, selected_layers),
                attention_mask=attention_mask,
                visual_tokens=_flatten_active_visual_tokens(vit_embeds, image_mask, num_tiles_list),
                planner_vl_summary=planner_vl_summary,
            )

        return fused_tokens


def _flatten_active_visual_tokens(
    vit_embeds: torch.Tensor,
    image_mask: torch.Tensor,
    num_tiles_list: Sequence[int],
) -> torch.Tensor:
    active = image_mask.to(device=vit_embeds.device).bool().reshape(-1)
    parts = []
    cursor = 0
    for image_index, tile_count in enumerate(num_tiles_list):
        tile_count = int(tile_count)
        image_tokens = vit_embeds[cursor : cursor + tile_count].reshape(-1, vit_embeds.shape[-1])
        if image_index < int(active.numel()) and bool(active[image_index].item()):
            parts.append(image_tokens)
        cursor += tile_count
    if not parts:
        return vit_embeds.new_zeros(1, 0, vit_embeds.shape[-1])
    return torch.cat(parts, dim=0).unsqueeze(0)


def last_valid_token_summary(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if hidden_state.ndim != 3:
        raise ValueError(f"hidden_state must have shape [B, N, D], got {tuple(hidden_state.shape)}")
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must have shape [B, N], got {tuple(attention_mask.shape)}")
    if tuple(hidden_state.shape[:2]) != tuple(attention_mask.shape):
        raise ValueError(
            f"hidden_state sequence shape {tuple(hidden_state.shape[:2])} does not match "
            f"attention_mask shape {tuple(attention_mask.shape)}"
        )
    positions = torch.arange(hidden_state.shape[1], device=hidden_state.device).unsqueeze(0)
    last_indices = (attention_mask.to(device=hidden_state.device).long() * positions).max(dim=1).values
    return hidden_state[torch.arange(hidden_state.shape[0], device=hidden_state.device), last_indices].to(torch.float32)


def select_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    selected_layers: Sequence[int | str] | None = None,
) -> list[torch.Tensor]:
    if not hidden_states:
        raise ValueError("hidden_states must not be empty")
    if selected_layers is None:
        selected_layers = ("mid", "deep")

    resolved = [_resolve_layer_index(layer, len(hidden_states)) for layer in selected_layers]
    return [hidden_states[index] for index in resolved]


def _resolve_layer_index(layer: int | str, num_layers: int) -> int:
    if isinstance(layer, int):
        index = layer if layer >= 0 else num_layers + layer
    elif layer == "shallow":
        index = max(0, num_layers // 4)
    elif layer == "mid":
        index = max(0, num_layers // 2)
    elif layer in {"deep", "last"}:
        index = num_layers - 1
    else:
        raise ValueError(f"Unsupported hidden-state layer selector: {layer!r}")

    if index < 0 or index >= num_layers:
        raise ValueError(f"Hidden-state layer index {layer!r} resolved to {index}, outside 0..{num_layers - 1}")
    return index

