from __future__ import annotations

import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import Block, Attention
from typing import Optional

from .prompt import PrefixTuningPrompt


def _attention_forward_with_prompt(
    self: Attention, x: torch.Tensor, prompt: Optional[torch.Tensor] = None
) -> torch.Tensor:
    B, N, C = x.shape
    qkv = self.qkv(x)
    qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    if prompt is not None:
        prompt_k, prompt_v = prompt.unbind(1)
        prompt_k = prompt_k.transpose(1, 2)
        prompt_v = prompt_v.transpose(1, 2)
        k = torch.cat([prompt_k, k], dim=2)
        v = torch.cat([prompt_v, v], dim=2)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)

    return x


def _block_forward_with_prompt(
    self: Block,
    x: torch.Tensor,
    prompt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), prompt=prompt)))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x


class PromptedViT(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        feature_layer: int = 5,
        prompt_length: int = 1,
        num_prompt_layers: int = 12,
        seed: int = 0,
    ):
        super().__init__()

        self.feature_layer = feature_layer
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        for param in self.vit.parameters():
            param.requires_grad = False

        self.embed_dim = self.vit.embed_dim
        self.num_heads = self.vit.blocks[0].attn.num_heads
        self.grid_size: tuple[int, int] = self.vit.patch_embed.grid_size

        self.prompt_module = PrefixTuningPrompt(
            num_layers=num_prompt_layers,
            prompt_length=prompt_length,
            num_heads=self.num_heads,
            embed_dim=self.embed_dim,
            seed=seed,
        )

        for i, block in enumerate(self.vit.blocks):
            if i < num_prompt_layers:
                bound_attn_forward = _attention_forward_with_prompt.__get__(block.attn, Attention)
                block.attn.forward = bound_attn_forward

                bound_block_forward = _block_forward_with_prompt.__get__(block, Block)
                block.forward = bound_block_forward

    def reset_prompt(self):
        self.prompt_module.reset_prompt()

    def get_prompt_state(self) -> torch.Tensor:
        return self.prompt_module.get_prompt_state()

    def set_prompt_state(self, state: torch.Tensor):
        self.prompt_module.set_prompt_state(state)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.vit.patch_embed(x)
            x = self.vit._pos_embed(x)
            x = self.vit.patch_drop(x)
            x = self.vit.norm_pre(x)

            for i, block in enumerate(self.vit.blocks):
                if i < self.prompt_module.num_layers:
                    x = block(x, prompt=None)
                else:
                    x = block(x)

                if i == self.feature_layer:
                    return x[:, 1:, :]

            raise ValueError(f"feature_layer {self.feature_layer} is out of bounds")

    def extract_features_with_prompt(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        batched_prompts = self.prompt_module.get_batched_prompt(B)

        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        for i, block in enumerate(self.vit.blocks):
            if i < self.prompt_module.num_layers:
                prompt = batched_prompts[i]
                x = block(x, prompt=prompt)
            else:
                x = block(x)

            if i == self.feature_layer:
                return x[:, 1:, :]

        raise ValueError(f"feature_layer {self.feature_layer} is out of bounds")
