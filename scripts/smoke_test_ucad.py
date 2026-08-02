import logging
import torch
import numpy as np

from pyclad.data.concept import Concept
from pyclad.vision.models.ucad.config import UCADConfig
from pyclad.vision.models.ucad.ucad_model import UCADModel
from pyclad.vision.models.ucad.sam import MaskProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyMaskProvider(MaskProvider):
    """Provides random mask labels for smoke testing without loading SAM."""

    def get_masks(self, image_paths, target_size=(14, 14)):
        B = len(image_paths)
        H, W = target_size
        # Generate random integer labels between 1 and 5
        return torch.randint(1, 6, (B, H * W), dtype=torch.float32)


def test_ucad_smoke():
    logger.info("Starting UCAD Smoke Test")
    
    # 1. Configuration (very small for quick test)
    config = UCADConfig(
        vit_model_name="vit_tiny_patch16_224", # Smallest possible ViT
        pretrained=False, # Skip downloading weights for instant start
        feature_layer=2,
        prompt_length=1,
        num_prompt_layers=2,
        training_epochs=1,  # Just 1 epoch
        batch_size=1,
        key_size=5,
        knowledge_size=5,
        device="cpu"  # Force CPU for smoke test compatibility
    )
    
    # 2. Initialize Model
    logger.info("Initializing UCADModel...")
    model = UCADModel(config, mask_provider=DummyMaskProvider())
    
    # 3. Create Dummy Data (2 concepts/tasks)
    # Shape: (N, H, W, C) - standard pyCLAD image format
    # ViT expects 224x224 RGB
    logger.info("Creating dummy data...")
    concept1_train = Concept("task1", data=np.random.rand(4, 224, 224, 3).astype(np.float32))
    concept1_test = Concept("task1", data=np.random.rand(2, 224, 224, 3).astype(np.float32))
    
    concept2_train = Concept("task2", data=np.random.rand(4, 224, 224, 3).astype(np.float32))
    concept2_test = Concept("task2", data=np.random.rand(2, 224, 224, 3).astype(np.float32))
    
    # 4. Training (Learn)
    logger.info("Training on Task 1...")
    model.fit(concept1_train)
    assert model.memory.num_tasks == 1
    
    logger.info("Training on Task 2...")
    model.fit(concept2_train)
    assert model.memory.num_tasks == 2
    
    # 5. Prediction
    logger.info("Predicting on Task 1 test data...")
    res1 = model.predict(concept1_test)
    assert res1.anomaly_scores.shape == (2,)
    assert res1.score_maps.shape == (2, 224, 224)
    logger.info(f"Task 1 Anomaly Scores: {res1.anomaly_scores}")
    
    logger.info("Predicting on Task 2 test data...")
    res2 = model.predict(concept2_test)
    assert res2.anomaly_scores.shape == (2,)
    assert res2.score_maps.shape == (2, 224, 224)
    logger.info(f"Task 2 Anomaly Scores: {res2.anomaly_scores}")
    
    logger.info("Smoke test passed successfully!")


if __name__ == "__main__":
    test_ucad_smoke()
