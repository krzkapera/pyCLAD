from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field


class UCADConfig(BaseModel):
    """Configuration for the UCAD model."""
    
    # Backbone settings
    vit_model_name: str = Field(default="vit_base_patch16_224", description="Name of the timm ViT model")
    feature_layer: int = Field(default=5, description="ViT layer index to extract features from")
    input_size: tuple[int, int] = Field(default=(224, 224), description="Input image size (H, W)")
    
    # Prompt settings
    prompt_length: int = Field(default=1, description="Length of the prefix prompt per layer")
    num_prompt_layers: int = Field(default=12, description="Number of ViT layers to inject prompts into")
    
    # Memory settings
    max_tasks: int = Field(default=15, description="Maximum number of concepts/tasks to store")
    knowledge_size: int = Field(default=196, description="Size of the coreset target for knowledge bank")
    key_size: int = Field(default=196, description="Size of the coreset target for task key")
    
    # Contrastive learning settings
    scl_temperature: float = Field(default=0.5, description="Temperature for structure-based contrastive loss")
    
    # Training settings
    training_epochs: int = Field(default=25, description="Number of training epochs per concept")
    learning_rate: float = Field(default=5e-4, description="Learning rate for prompt tuning")
    grad_clip: float = Field(default=1.0, description="Gradient clipping max norm")
    batch_size: int = Field(default=8, description="Batch size for training")
    pretrained: bool = Field(default=True, description="Whether to load pretrained backbone weights")
    
    # SAM settings
    sam_model: str = Field(default="facebook/sam2-hiera-small", description="SAM2 model identifier or path")
    sam_masks_dir: Optional[Path] = Field(default=None, description="Path to pre-computed SAM masks (offline mode)")
    
    # Hardware
    device: Optional[str] = Field(default=None, description="Device to use ('cuda', 'cpu', or None for auto)")
