import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
from typing import Union

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
    def __init__(self, config: Union[UCADConfig, dict]):
        super().__init__()

        if isinstance(config, dict):
            self.config = UCADConfig(**config)
        else:
            self.config = config

        self._init_device()

        self.backbone = PromptedViT(
            model_name=self.config.vit_model_name,
            pretrained=self.config.pretrained,
            feature_layer=self.config.feature_layer,
            prompt_length=self.config.prompt_length,
            num_prompt_layers=self.config.num_prompt_layers,
        ).to(self.device)

        self.memory = TaskMemoryBank(max_tasks=self.config.max_tasks)
        self.scorer = NearestNeighborScorer(n_neighbors=5, blur_sigma=3.0)
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
        if isinstance(batch, (tuple, list)):
            return batch[0]
        elif isinstance(batch, dict) and "image" in batch:
            return batch["image"]
        return batch

    def _get_paths(self, batch):
        if isinstance(batch, dict) and "image_path" in batch:
            return batch["image_path"]
        elif hasattr(batch, "path"):
            return batch.path
        return [f"dummy_{i}.png" for i in range(len(self._get_images(batch)))]

    def fit(self, training_data):
        logger.info(f"Training UCAD on task {self.current_task_id}")
        from torch.utils.data import DataLoader

        if hasattr(training_data, "data"):
            tensor_data = torch.tensor(training_data.data, dtype=torch.float32)
            if tensor_data.ndim == 4 and tensor_data.shape[-1] == 3:
                tensor_data = tensor_data.permute(0, 3, 1, 2)

            dataset = []
            for i in range(len(tensor_data)):
                dataset.append({"image": tensor_data[i], "image_path": f"concept_{self.current_task_id}_{i}.png"})

            train_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        else:
            train_loader = training_data

        self.backbone.eval()
        key_features = self._extract_all_features(train_loader, use_prompt=False)

        B, Np, C = key_features.shape
        key_flat = key_features.reshape(-1, C)

        logger.info(f"Computing coreset for Task Key (target: {self.config.key_size})")
        task_key = greedy_coreset_sampling(key_flat, self.config.key_size, device=self.device)

        optimizer = optim.AdamW(
            self.backbone.prompt_module.parameters(), lr=self.config.learning_rate, weight_decay=1e-4
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.config.training_epochs, T_mult=1)

        self.backbone.train()

        for epoch in range(self.config.training_epochs):
            total_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.training_epochs}", leave=False):
                images = self._get_images(batch).to(self.device)
                paths = self._get_paths(batch)

                mask_labels = self.mask_provider.get_masks(paths, target_size=(14, 14)).to(self.device)
                features = self.backbone.extract_features_with_prompt(images)

                loss = structure_contrastive_loss(features, mask_labels, temperature=self.config.scl_temperature)
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.backbone.prompt_module.parameters(), self.config.grad_clip)
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()
            logger.info(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

        self.backbone.eval()
        knowledge_features = self._extract_all_features(train_loader, use_prompt=True)
        knowledge_flat = knowledge_features.reshape(-1, C)

        logger.info(f"Computing coreset for Task Knowledge (target: {self.config.knowledge_size})")
        task_knowledge = greedy_coreset_sampling(knowledge_flat, self.config.knowledge_size, device=self.device)
        prompt_state = self.backbone.get_prompt_state()

        self.memory.add_task(
            task_id=self.current_task_id, key=task_key, prompt_state=prompt_state, knowledge=task_knowledge
        )

        logger.info(f"Task {self.current_task_id} added to memory bank.")
        self.current_task_id += 1

    def predict(self, data) -> VisionPredictionResults:
        self.backbone.eval()

        from torch.utils.data import DataLoader

        if hasattr(data, "data"):
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

                frozen_features = self.backbone.extract_features(images)
                task_idx = self.memory.select_task(frozen_features)

                self.backbone.set_prompt_state(self.memory.get_prompt_state(task_idx))
                knowledge = self.memory.get_knowledge(task_idx).to(self.device)

                prompted_features = self.backbone.extract_features_with_prompt(images)

                img_scores, maps = self.scorer.predict(
                    test_features=prompted_features,
                    knowledge_bank=knowledge,
                    input_size=self.config.input_size,
                )

                all_image_scores.extend(img_scores)
                all_anomaly_maps.extend(maps)

        all_image_scores = np.array(all_image_scores)
        return VisionPredictionResults(
            y_pred=np.zeros_like(all_image_scores, dtype=int),
            anomaly_scores=all_image_scores,
            score_maps=np.array(all_anomaly_maps),
        )
