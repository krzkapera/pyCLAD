import torch
import torch.nn as nn
from typing import Optional


class PrefixTuningPrompt(nn.Module):
    """
    Continual Prompting Module (CPM) based on prefix-tuning.
    
    This prompt module creates learnable prefix tokens that are prepended 
    to the keys and values in the Multi-Head Attention layers of a ViT.
    
    Args:
        num_layers: Number of ViT layers where prompt is injected.
        prompt_length: Number of tokens in the prompt per layer.
        num_heads: Number of attention heads in the ViT backbone.
        embed_dim: Total embedding dimension of the ViT backbone.
    """
    def __init__(
        self, 
        num_layers: int = 12, 
        prompt_length: int = 1, 
        num_heads: int = 12, 
        embed_dim: int = 768
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.prompt_length = prompt_length
        self.num_heads = num_heads
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.head_dim = embed_dim // num_heads
        
        # Shape: (num_layers, 2 [key, value], prompt_length, num_heads, head_dim)
        prompt_shape = (num_layers, 2, prompt_length, num_heads, self.head_dim)
        
        # Initialize with uniform distribution as in original UCAD
        self.prompt = nn.Parameter(torch.randn(prompt_shape))
        nn.init.uniform_(self.prompt, -1, 1)

    def get_prompt_state(self) -> torch.Tensor:
        """Returns the current prompt state (detached)."""
        return self.prompt.detach().clone()

    def set_prompt_state(self, state: torch.Tensor):
        """Sets the prompt to a specific state (e.g., from memory bank)."""
        if state.shape != self.prompt.shape:
            raise ValueError(f"Expected prompt shape {self.prompt.shape}, got {state.shape}")
        
        self.prompt = nn.Parameter(state.to(self.prompt.device))

    def get_batched_prompt(self, batch_size: int) -> torch.Tensor:
        """
        Expands the prompt for a specific batch size.
        
        Returns:
            Tensor of shape (num_layers, batch_size, 2, prompt_length, num_heads, head_dim)
        """
        # (num_layers, 2, prompt_length, num_heads, head_dim)
        # -> (num_layers, 1, 2, prompt_length, num_heads, head_dim)
        # -> expand to batch_size
        return self.prompt.unsqueeze(1).expand(-1, batch_size, -1, -1, -1, -1)

