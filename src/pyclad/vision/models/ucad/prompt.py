import torch
import torch.nn as nn


class PrefixTuningPrompt(nn.Module):
    def __init__(
        self,
        num_layers: int = 12,
        prompt_length: int = 1,
        num_heads: int = 12,
        embed_dim: int = 768,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.prompt_length = prompt_length
        self.num_heads = num_heads

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.head_dim = embed_dim // num_heads

        prompt_shape = (num_layers, 2, prompt_length, num_heads, self.head_dim)
        self.prompt = nn.Parameter(torch.empty(prompt_shape))
        nn.init.uniform_(self.prompt, -1, 1)

    def get_prompt_state(self) -> torch.Tensor:
        return self.prompt.detach().clone()

    def set_prompt_state(self, state: torch.Tensor):
        if state.shape != self.prompt.shape:
            raise ValueError(f"Expected prompt shape {self.prompt.shape}, got {state.shape}")

        with torch.no_grad():
            self.prompt.copy_(state)

    def get_batched_prompt(self, batch_size: int) -> torch.Tensor:
        return self.prompt.unsqueeze(1).expand(-1, batch_size, -1, -1, -1, -1)
