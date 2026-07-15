import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer, Block, Attention
from typing import Optional, List, Dict, Any

from .prompt import PrefixTuningPrompt


def _attention_forward_with_prompt(self: Attention, x: torch.Tensor, prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Custom forward pass for timm's Attention module that supports prefix-tuning.
    
    Args:
        x: Input tensor of shape (B, N, C)
        prompt: Prefix prompt tensor of shape (B, 2, prompt_length, num_heads, head_dim)
                where 2 corresponds to (key_prompt, value_prompt).
    """
    B, N, C = x.shape
    
    # Standard qkv projection: (B, N, 3*C)
    qkv = self.qkv(x)
    
    # Reshape and permute to: (3, B, num_heads, N, head_dim)
    qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    
    if prompt is not None:
        # prompt shape: (B, 2, prompt_length, num_heads, head_dim)
        # prompt_k, prompt_v shape: (B, prompt_length, num_heads, head_dim)
        prompt_k, prompt_v = prompt.unbind(1)
        
        # Reshape to match k, v: (B, num_heads, prompt_length, head_dim)
        prompt_k = prompt_k.transpose(1, 2)
        prompt_v = prompt_v.transpose(1, 2)
        
        # Prepend prefix prompts to keys and values
        k = torch.cat([prompt_k, k], dim=2)
        v = torch.cat([prompt_v, v], dim=2)
        
    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    
    x = attn @ v
    
    # Reshape back to (B, N, C)
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    
    return x


def _block_forward_with_prompt(self: Block, x: torch.Tensor, prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Custom forward pass for timm's Block module that passes the prompt to Attention."""
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), prompt=prompt)))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x


class PromptedViT(nn.Module):
    """
    Wrapper around timm VisionTransformer that adds prefix-tuning support and 
    extracts intermediate feature maps for anomaly detection.
    """
    def __init__(
        self, 
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        feature_layer: int = 5,
        prompt_length: int = 1,
        num_prompt_layers: int = 12
    ):
        super().__init__()
        
        self.feature_layer = feature_layer
        
        # Load pre-trained ViT
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
        )
        
        # Freeze all backbone parameters
        for param in self.vit.parameters():
            param.requires_grad = False
            
        self.embed_dim = self.vit.embed_dim
        self.num_heads = self.vit.blocks[0].attn.num_heads
        
        # Initialize prompt module
        self.prompt_module = PrefixTuningPrompt(
            num_layers=num_prompt_layers,
            prompt_length=prompt_length,
            num_heads=self.num_heads,
            embed_dim=self.embed_dim
        )
        
        # Monkey-patch the forward methods of blocks and attention layers
        for i, block in enumerate(self.vit.blocks):
            if i < num_prompt_layers:
                # Bind the custom attention forward method
                bound_attn_forward = _attention_forward_with_prompt.__get__(block.attn, Attention)
                block.attn.forward = bound_attn_forward
                
                # Bind the custom block forward method
                bound_block_forward = _block_forward_with_prompt.__get__(block, Block)
                block.forward = bound_block_forward

    def get_prompt_state(self) -> torch.Tensor:
        return self.prompt_module.get_prompt_state()
        
    def set_prompt_state(self, state: torch.Tensor):
        self.prompt_module.set_prompt_state(state)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the frozen ViT without any prompt (used for Keys).
        Returns features of shape (B, Np, C), excluding the class token.
        """
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
                    # Return patch tokens (exclude cls token at index 0)
                    return x[:, 1:, :]
                    
            raise ValueError(f"feature_layer {self.feature_layer} is out of bounds")

    def extract_features_with_prompt(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using the current prompt state (used for training and Knowledge).
        Returns features of shape (B, Np, C), excluding the class token.
        """
        B = x.shape[0]
        
        # Get batched prompts: (num_layers, B, 2, prompt_length, num_heads, head_dim)
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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for compatibility, returning both features.
        Usually, methods will explicitly call `extract_features` or `extract_features_with_prompt`.
        """
        return {
            "key_features": self.extract_features(x),
            "prompted_features": self.extract_features_with_prompt(x)
        }
