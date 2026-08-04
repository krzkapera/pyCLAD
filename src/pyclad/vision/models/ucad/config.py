from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field


class UCADConfig(BaseModel):
    # Backbone settings
    vit_model_name: str = Field(default="vit_base_patch16_224", description="Name of the timm ViT model")
    feature_layer: int = Field(default=5, description="ViT layer index to extract features from")
    input_size: tuple[int, int] = Field(default=(224, 224), description="Input image size (H, W)")
    resize_mode: Literal["stretch", "short_side_crop"] = Field(
        default="stretch",
        description="'stretch' fits the whole image into input_size; 'short_side_crop' scales the short side "
        "and crops the centre, as the reference implementation's loader does. Must match the resize_mode the "
        "dataset's ground-truth masks were read with",
    )

    # Prompt settings
    prompt_length: int = Field(default=1, description="Length of the prefix prompt per layer")
    num_prompt_layers: int = Field(default=12, description="Number of ViT layers to inject prompts into")

    # Memory settings
    max_tasks: int = Field(default=15, description="Maximum number of concepts/tasks to store")
    knowledge_size: int = Field(default=196, description="Size of the coreset target for knowledge bank")
    key_size: int = Field(default=196, description="Size of the coreset target for task key")
    coreset_mode: Literal["exact", "approximate"] = Field(
        default="exact",
        description="'exact' selects deterministically in the full feature space; 'approximate' seeds from "
        "random starting points and selects in a randomly projected space, as the reference UCAD implementation does",
    )

    # Contrastive learning settings
    scl_temperature: float = Field(default=0.5, description="Temperature for structure-based contrastive loss")

    anomaly_scorer_num_nn: int = Field(
        default=1,
        description="Patch score = mean distance to this many nearest knowledge vectors (1 = nearest-neighbor "
        "distance); ignored when reweighting is enabled",
    )
    reweighting_num_nn: int = Field(
        default=0,
        description="Image-score reweighting by the local density of the knowledge bank: size of the neighborhood "
        "of the nearest match used in the softmax weight (>= 2); 0 disables it (image score = max patch score)",
    )
    blur_sigma: float = Field(default=3.0, description="Gaussian smoothing applied to the upsampled anomaly map")
    reset_prompt_per_task: bool = Field(
        default=True,
        description="Re-initialize the prompt before each concept instead of continuing from the previous one",
    )
    loss_mode: Literal["linear", "exp_negatives"] = Field(
        default="exp_negatives",
        description="'linear' = positive and negative similarities enter the loss linearly, unweighted; "
        "'exp_negatives' = negative-pair similarities are exponentiated (temperature-scaled), weighting hard negatives more",
    )
    patchsize: int = Field(
        default=1,
        description="Side of the square neighborhood of patch features averaged into each embedding; 1 = no aggregation",
    )
    target_embed_dimension: int = Field(
        default=1024, description="Aggregated feature dimension (used when patchsize > 1)"
    )

    score_ensemble_epochs: int = Field(
        default=1,
        description="How many training epochs contribute a prompt/knowledge pair to the concept's score, spread "
        "evenly over training and always including the last: 1 evaluates the final epoch alone, higher values "
        "average the normalized scores of that many epochs, as the reference implementation does over all of "
        "them (it additionally picks the best epoch by test AUROC, which this does not)",
    )

    # Training settings
    training_epochs: int = Field(default=25, description="Number of training epochs per concept")
    learning_rate: float = Field(default=5e-4, description="Learning rate for prompt tuning")
    grad_clip: float = Field(default=1.0, description="Gradient clipping max norm")
    batch_size: int = Field(default=8, description="Batch size for training")
    pretrained: bool = Field(default=True, description="Whether to load pretrained backbone weights")

    # SAM2 settings
    sam_model: str = Field(default="facebook/sam2-hiera-small", description="SAM2 model identifier or path")
    sam_points_per_side: int = Field(default=32, description="SAM2 point grid density")
    sam_pred_iou_thresh: float = Field(default=0.80, description="SAM2 mask quality threshold")
    sam_stability_thresh: float = Field(default=0.95, description="SAM2 mask stability threshold")
    sam_masks_dir: Optional[Path] = Field(
        default=None, description="Directory of precomputed SAM2 masks; None generates masks online instead"
    )
    sam_images_root: Optional[Path] = Field(
        default=None, description="Dataset root that sam_masks_dir mirrors; required when sam_masks_dir is set"
    )

    # Hardware
    device: Optional[str] = Field(default=None, description="Device to use ('cuda', 'cpu', or None for auto)")
    seed: int = Field(
        default=0,
        description="Seed of the prompt initialization and of the data-loader generator; keeps both independent "
        "of the global RNG so that evaluating the model does not perturb subsequent training",
    )
