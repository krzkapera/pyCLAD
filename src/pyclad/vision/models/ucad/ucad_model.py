import logging
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from pyclad.vision.models.vision_model import VisionModel
from pyclad.vision.prediction_results import VisionPredictionResults

from .config import UCADConfig
from .contrastive import structure_contrastive_loss
from .coreset import greedy_coreset_sampling
from .features import patchcore_aggregate
from .memory import TaskMemoryBank
from .sam import MaskProvider, create_mask_provider
from .scoring import NearestNeighborScorer
from .vit_prompted import PromptedViT

logger = logging.getLogger(__name__)


class UCADModel(VisionModel):
    def __init__(self, config: Union[UCADConfig, dict], mask_provider: Optional[MaskProvider] = None):
        super().__init__()

        self.config = UCADConfig(**config) if isinstance(config, dict) else config
        self._init_device()

        self.backbone = PromptedViT(
            model_name=self.config.vit_model_name,
            pretrained=self.config.pretrained,
            feature_layer=self.config.feature_layer,
            prompt_length=self.config.prompt_length,
            num_prompt_layers=self.config.num_prompt_layers,
        ).to(self.device)

        self.memory = TaskMemoryBank(max_tasks=self.config.max_tasks)
        self.scorer = NearestNeighborScorer(
            num_nn=self.config.anomaly_scorer_num_nn,
            reweighting_num_nn=self.config.reweighting_num_nn,
            blur_sigma=3.0,
        )
        self.mask_provider = mask_provider if mask_provider is not None else create_mask_provider(
            self.config, device=self.device
        )
        self.current_task_id = 0

    def name(self) -> str:
        return "UCAD"

    def _init_device(self):
        if self.config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        logger.info(f"UCAD initialized on device: {self.device}")

    def _aggregate(self, features: torch.Tensor) -> torch.Tensor:
        return patchcore_aggregate(
            features, self.backbone.grid_size, self.config.patchsize, self.config.target_embed_dimension
        )

    @torch.no_grad()
    def _extract_all_features(self, data_loader, use_prompt: bool = False) -> torch.Tensor:
        features = []
        for batch in tqdm(data_loader, desc="Extracting features", leave=False):
            images = self._get_images(batch).to(self.device)

            if use_prompt:
                feat = self.backbone.extract_features_with_prompt(images)
            else:
                feat = self.backbone.extract_features(images)

            features.append(self._aggregate(feat).cpu())

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
        return [f"concept_{self.current_task_id}_{i}.png" for i in range(len(self._get_images(batch)))]

    def _as_loader(self, data, shuffle: bool) -> DataLoader:
        if not hasattr(data, "data"):
            return data

        tensor_data = torch.tensor(data.data, dtype=torch.float32)
        if tensor_data.ndim == 4 and tensor_data.shape[-1] == 3:
            tensor_data = tensor_data.permute(0, 3, 1, 2)

        dataset = [
            {"image": tensor_data[i], "image_path": f"concept_{self.current_task_id}_{i}.png"}
            for i in range(len(tensor_data))
        ]
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

    def fit(self, training_data, epoch_callback: Optional[Callable[[int], None]] = None):
        logger.info(f"Training UCAD on task {self.current_task_id}")
        train_loader = self._as_loader(training_data, shuffle=True)

        self.backbone.eval()
        key_features = self._extract_all_features(train_loader, use_prompt=False)

        C = key_features.shape[-1]
        key_flat = key_features.reshape(-1, C)

        logger.info(f"Computing coreset for Task Key (target: {self.config.key_size})")
        task_key = greedy_coreset_sampling(key_flat, self.config.key_size, device=self.device)

        if self.memory.num_tasks > 0:
            self.backbone.set_prompt_state(self.memory.get_prompt_state(self.memory.num_tasks - 1))

        optimizer = optim.Adam(self.backbone.prompt_module.parameters(), lr=self.config.learning_rate)

        self.backbone.train()

        for epoch in range(self.config.training_epochs):
            total_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.training_epochs}", leave=False):
                images = self._get_images(batch).to(self.device)
                paths = self._get_paths(batch)

                mask_labels = self.mask_provider.get_masks(paths, target_size=self.backbone.grid_size).to(self.device)
                features = self.backbone.extract_features_with_prompt(images)

                loss = structure_contrastive_loss(
                    features, mask_labels, mode=self.config.loss_mode, temperature=self.config.scl_temperature
                )
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.backbone.prompt_module.parameters(), self.config.grad_clip)
                optimizer.step()

                total_loss += loss.item()

            logger.info(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

            if epoch_callback is not None:
                self.backbone.eval()
                epoch_callback(epoch)
                self.backbone.train()

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
        test_loader = self._as_loader(data, shuffle=False)

        all_image_scores = []
        all_anomaly_maps = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting", leave=False):
                images = self._get_images(batch).to(self.device)

                frozen_features = self._aggregate(self.backbone.extract_features(images))
                task_ids = self.memory.select_tasks(frozen_features)

                batch_scores = np.empty(len(images))
                batch_maps = np.empty((len(images), *self.config.input_size))

                for task_idx in task_ids.unique().tolist():
                    selected = task_ids == task_idx
                    self.backbone.set_prompt_state(self.memory.get_prompt_state(task_idx))
                    knowledge = self.memory.get_knowledge(task_idx).to(self.device)

                    prompted_features = self._aggregate(self.backbone.extract_features_with_prompt(images[selected]))
                    img_scores, maps = self.scorer.predict(
                        test_features=prompted_features,
                        knowledge_bank=knowledge,
                        input_size=self.config.input_size,
                    )

                    selected_np = selected.cpu().numpy()
                    batch_scores[selected_np] = img_scores
                    batch_maps[selected_np] = maps

                all_image_scores.append(batch_scores)
                all_anomaly_maps.append(batch_maps)

        image_scores = np.concatenate(all_image_scores)
        return VisionPredictionResults(
            y_pred=np.zeros_like(image_scores, dtype=int),
            anomaly_scores=image_scores,
            score_maps=np.concatenate(all_anomaly_maps),
        )
