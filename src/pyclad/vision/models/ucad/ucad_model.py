import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Union

from pyclad.vision.models.vision_model import VisionModel
from pyclad.vision.models.utilities.base_model import VisionPredictionResults

from .config import UCADConfig
from .vit_prompted import PromptedViT
from .memory import TaskMemoryBank
from .coreset import greedy_coreset_sampling
from .scoring import NearestNeighborScorer
from .contrastive import structure_contrastive_loss
from .sam import create_mask_provider

logger = logging.getLogger(__name__)


class UCADModel(VisionModel):
    """
    Unsupervised Continual Anomaly Detection with Contrastively-Learned Prompt (UCAD)
    
    This model implements a Continual Prompting Module (CPM) and 
    Structure-based Contrastive Learning (SCL) to perform continual 
    anomaly detection without catastrophic forgetting.
    """
    def __init__(self, config: Union[UCADConfig, dict]):
        super().__init__()
        
        if isinstance(config, dict):
            self.config = UCADConfig(**config)
        else:
            self.config = config
            
        self._init_device()
        
        # 1. Backbone: ViT with prefix-tuning prompt injection
        self.backbone = PromptedViT(
            model_name=self.config.vit_model_name,
            pretrained=self.config.pretrained,
            feature_layer=self.config.feature_layer,
            prompt_length=self.config.prompt_length,
            num_prompt_layers=self.config.num_prompt_layers
        ).to(self.device)
        
        # 2. Memory Bank: Keys, Prompts, Knowledge
        self.memory = TaskMemoryBank(max_tasks=self.config.max_tasks)
        
        # 3. Scorer: Nearest-Neighbor with re-weighting
        self.scorer = NearestNeighborScorer(n_neighbors=5, blur_sigma=3.0)
        
        # 4. SAM: Mask Provider for SCL
        self.mask_provider = create_mask_provider(self.config, device=self.device)
        
        self.current_task_id = 0

    @property
    def name(self) -> str:
        return "UCAD"

    def _init_device(self):
        if self.config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        logger.info(f"UCAD initialized on device: {self.device}")

    def _extract_all_features(self, data_loader, use_prompt: bool = False) -> torch.Tensor:
        """Helper to extract features for all images in a dataloader."""
        features = []
        for batch in tqdm(data_loader, desc="Extracting features", leave=False):
            images = self._get_images(batch).to(self.device)
            
            if use_prompt:
                feat = self.backbone.extract_features_with_prompt(images)
            else:
                feat = self.backbone.extract_features(images)
                
            features.append(feat.cpu())
            
        return torch.cat(features, dim=0)

    def _get_images(self, batch):
        """Helper to handle different dataloader output formats."""
        # pyclad dataloaders typically return a tuple of (images, labels, ...)
        # or dicts depending on the dataset. Adjust according to pyclad conventions.
        if isinstance(batch, (tuple, list)):
            return batch[0]
        elif isinstance(batch, dict) and "image" in batch:
            return batch["image"]
        return batch
        
    def _get_paths(self, batch):
        """Helper to extract image paths if available (needed for SAM masks)."""
        if isinstance(batch, dict) and "image_path" in batch:
            return batch["image_path"]
        elif hasattr(batch, "path"):
            return batch.path
        # Fallback if no paths are provided
        return [f"dummy_{i}.png" for i in range(len(self._get_images(batch)))]

    def fit(self, training_data):
        """
        Trains the model on a new task/concept.
        
        Algorithm:
        1. Extract features using frozen backbone (no prompt) -> FPS -> Task Key
        2. Train Prompt with Structure-based Contrastive Loss using SAM masks
        3. Extract features using tuned prompt -> FPS -> Task Knowledge
        4. Save (Key, Prompt, Knowledge) to memory bank
        """
        logger.info(f"Training UCAD on task {self.current_task_id}")
        
        # We need a proper PyTorch DataLoader for training
        # If pyCLAD passes a Concept, we need to wrap its data in a DataLoader
        from torch.utils.data import DataLoader, TensorDataset
        
        if hasattr(training_data, 'data'):
            # It's likely a pyCLAD Concept
            tensor_data = torch.tensor(training_data.data, dtype=torch.float32)
            # Permute to (B, C, H, W) if it's not already
            if tensor_data.ndim == 4 and tensor_data.shape[-1] == 3:
                tensor_data = tensor_data.permute(0, 3, 1, 2)
                
            # Create a simple dataset with paths (required for SAM offline)
            # In a real scenario, we'd use pyCLAD's dataset tools
            dataset = []
            for i in range(len(tensor_data)):
                dataset.append({"image": tensor_data[i], "image_path": f"concept_{self.current_task_id}_{i}.png"})
                
            train_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        else:
            # Assume it's already an iterable/dataloader
            train_loader = training_data
            
        # Step 1: Compute Task Key (from frozen backbone)
        self.backbone.eval()
        key_features = self._extract_all_features(train_loader, use_prompt=False)
        
        # Flatten (N, Np, C) to (N*Np, C) for coreset
        B, Np, C = key_features.shape
        key_flat = key_features.reshape(-1, C)
        
        logger.info(f"Computing coreset for Task Key (target: {self.config.key_size})")
        task_key = greedy_coreset_sampling(key_flat, self.config.key_size, device=self.device)
        
        # Step 2: Train Prompt using SCL
        optimizer = optim.AdamW(
            self.backbone.prompt_module.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=self.config.training_epochs, 
            T_mult=1
        )
        
        self.backbone.train()
        
        for epoch in range(self.config.training_epochs):
            total_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.training_epochs}", leave=False):
                images = self._get_images(batch).to(self.device)
                paths = self._get_paths(batch)
                
                # Get SAM masks
                # (B, 196) labels
                mask_labels = self.mask_provider.get_masks(paths, target_size=(14, 14)).to(self.device)
                
                # Forward pass through prompted ViT
                # features shape: (B, 196, 768)
                features = self.backbone.extract_features_with_prompt(images)
                
                # Compute Contrastive Loss
                loss = structure_contrastive_loss(
                    features, 
                    mask_labels, 
                    temperature=self.config.scl_temperature
                )
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.backbone.prompt_module.parameters(), 
                    self.config.grad_clip
                )
                optimizer.step()
                
                total_loss += loss.item()
                
            scheduler.step()
            logger.info(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")
            
        # Step 3: Compute Task Knowledge (using tuned prompt)
        self.backbone.eval()
        knowledge_features = self._extract_all_features(train_loader, use_prompt=True)
        knowledge_flat = knowledge_features.reshape(-1, C)
        
        logger.info(f"Computing coreset for Task Knowledge (target: {self.config.knowledge_size})")
        task_knowledge = greedy_coreset_sampling(knowledge_flat, self.config.knowledge_size, device=self.device)
        
        # Step 4: Save to Memory Bank
        prompt_state = self.backbone.get_prompt_state()
        
        self.memory.add_task(
            task_id=self.current_task_id,
            key=task_key,
            prompt_state=prompt_state,
            knowledge=task_knowledge
        )
        
        logger.info(f"Task {self.current_task_id} added to memory bank.")
        self.current_task_id += 1

    def predict(self, data) -> VisionPredictionResults:
        """
        Predicts anomaly scores for test data.
        
        Algorithm:
        1. Extract frozen features -> match to nearest Task Key -> select task
        2. Load Task Prompt and Knowledge
        3. Extract prompted features -> NN Scoring vs Knowledge
        """
        self.backbone.eval()
        
        from torch.utils.data import DataLoader
        
        if hasattr(data, 'data'):
            tensor_data = torch.tensor(data.data, dtype=torch.float32)
            if tensor_data.ndim == 4 and tensor_data.shape[-1] == 3:
                tensor_data = tensor_data.permute(0, 3, 1, 2)
            test_loader = DataLoader(tensor_data, batch_size=self.config.batch_size, shuffle=False)
        else:
            test_loader = data
            
        all_image_scores = []
        all_anomaly_maps = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting", leave=False):
                images = self._get_images(batch).to(self.device)
                B = images.shape[0]
                
                # 1. Task Selection
                # In UCAD, task is selected based on frozen backbone features
                frozen_features = self.backbone.extract_features(images)
                
                # Typically we select one task per batch for simplicity in evaluation
                # as test sets are usually evaluated per-concept.
                task_idx = self.memory.select_task(frozen_features)
                
                # 2. Load Prompt and Knowledge
                self.backbone.set_prompt_state(self.memory.get_prompt_state(task_idx))
                knowledge = self.memory.get_knowledge(task_idx).to(self.device)
                
                # 3. Extract and Score
                prompted_features = self.backbone.extract_features_with_prompt(images)
                
                img_scores, maps = self.scorer.predict(
                    test_features=prompted_features,
                    knowledge_bank=knowledge,
                    input_size=self.config.input_size
                )
                
                all_image_scores.extend(img_scores)
                all_anomaly_maps.extend(maps)
                
        # Normalize scores (Min-Max per batch for testing, or global depending on metric logic)
        # Often handled by the Strategy or Callback in pyCLAD, but we return raw scores here.
        all_image_scores = np.array(all_image_scores)
        return VisionPredictionResults(
            y_pred=np.zeros_like(all_image_scores, dtype=int),
            anomaly_scores=all_image_scores,
            score_maps=np.array(all_anomaly_maps)
        )
