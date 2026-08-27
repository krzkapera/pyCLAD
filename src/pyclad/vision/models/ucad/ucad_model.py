from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

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
from .inputs import build_dataset
from .memory import TaskMemoryBank
from .sam import MaskProvider, create_mask_provider
from .scoring import NearestNeighborScorer
from .vit_prompted import PromptedViT

logger = logging.getLogger(__name__)

PromptedBank = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class TaskTraining:
    train_loader: DataLoader
    extraction_loader: DataLoader
    key: torch.Tensor
    dimension: int
    optimizer: optim.Optimizer


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
            prompt_generator=self._seeded_generator(),
        ).to(self.device)

        self.memory = TaskMemoryBank(max_tasks=self.config.max_tasks)
        self.scorer = NearestNeighborScorer(
            num_nn=self.config.anomaly_scorer_num_nn,
            reweighting_num_nn=self.config.reweighting_num_nn,
            blur_sigma=self.config.blur_sigma,
        )
        self._mask_provider = mask_provider
        self.current_task_id = 0
        self._coreset_generator = self._seeded_generator()

    def _seeded_generator(self) -> torch.Generator:
        return torch.Generator().manual_seed(self.config.seed)

    @property
    def mask_provider(self) -> MaskProvider:
        # Built on first request rather than in the constructor: masks are read only by the
        # contrastive loss, so a run with training_epochs=0 never needs one, and the online provider
        # would otherwise load a SAM model that nothing goes on to call.
        if self._mask_provider is None:
            self._mask_provider = create_mask_provider(self.config, device=self.device)
        return self._mask_provider

    def name(self) -> str:
        return "UCAD"

    def additional_info(self) -> Dict[str, Any]:
        return {"config": self.config.model_dump(mode="json")}

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
            images = batch["image"].to(self.device)

            if use_prompt:
                feat = self.backbone.extract_features_with_prompt(images)
            else:
                feat = self.backbone.extract_features(images)

            features.append(self._aggregate(feat).cpu())

        return torch.cat(features, dim=0)

    def _as_loader(self, data: Union[DataLoader, Any], shuffle: bool) -> DataLoader:
        if isinstance(data, DataLoader):
            return data

        dataset = build_dataset(
            data, self.config.input_size, f"concept_{self.current_task_id}", self.config.resize_mode
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            generator=self._seeded_generator(),
        )

    def _sequential_view(self, loader: DataLoader) -> DataLoader:
        return DataLoader(
            loader.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            generator=self._seeded_generator(),
        )

    def fit(self, training_data):
        task = self._begin_task(training_data)

        for epoch in range(self.config.training_epochs):
            self._train_epoch(task, epoch)

        self._end_task(task, *self._snapshot(task))

    def _begin_task(self, training_data) -> TaskTraining:
        logger.info(f"Training UCAD on task {self.current_task_id}")
        train_loader = self._as_loader(training_data, shuffle=True)
        extraction_loader = self._sequential_view(train_loader)

        self.backbone.eval()
        key_features = self._extract_all_features(extraction_loader, use_prompt=False)
        dimension = key_features.shape[-1]

        logger.info(f"Computing coreset for Task Key (target: {self.config.key_size})")
        key = greedy_coreset_sampling(
            key_features.reshape(-1, dimension), self.config.key_size, self._coreset_generator,
            device=self.device, mode=self.config.coreset_mode,
        )

        if self.config.reset_prompt_per_task:
            self.backbone.reset_prompt()
        elif self.memory.num_tasks > 0:
            self.backbone.set_prompt_state(self.memory.get_task(self.memory.num_tasks - 1).prompt)

        self.backbone.train()
        return TaskTraining(
            train_loader=train_loader,
            extraction_loader=extraction_loader,
            key=key,
            dimension=dimension,
            optimizer=optim.Adam(self.backbone.prompt_module.parameters(), lr=self.config.learning_rate),
        )

    def _train_epoch(self, task: TaskTraining, epoch: int) -> None:
        self.backbone.train()
        total_loss = 0.0

        for batch in tqdm(task.train_loader, desc=f"Epoch {epoch+1}/{self.config.training_epochs}", leave=False):
            images = batch["image"].to(self.device)
            paths = batch["image_path"]

            mask_labels = self.mask_provider.get_masks(paths, target_size=self.backbone.grid_size).to(self.device)
            features = self.backbone.extract_features_with_prompt(images)

            loss = structure_contrastive_loss(
                features, mask_labels, mode=self.config.loss_mode, temperature=self.config.scl_temperature
            )
            optimizer = task.optimizer
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.backbone.prompt_module.parameters(), self.config.grad_clip)
            optimizer.step()

            total_loss += loss.item()

        logger.info(f"Epoch {epoch+1} Loss: {total_loss / len(task.train_loader):.4f}")

    def _end_task(self, task: TaskTraining, prompt: torch.Tensor, knowledge: torch.Tensor) -> None:
        self.memory.add_task(task_id=self.current_task_id, key=task.key, prompt=prompt, knowledge=knowledge)
        logger.info(f"Task {self.current_task_id} added to memory bank.")
        self.current_task_id += 1

    def _snapshot(self, task: TaskTraining) -> tuple[torch.Tensor, torch.Tensor]:
        self.backbone.eval()
        knowledge_features = self._extract_all_features(task.extraction_loader, use_prompt=True)

        logger.info(f"Computing coreset for Task Knowledge (target: {self.config.knowledge_size})")
        knowledge = greedy_coreset_sampling(
            knowledge_features.reshape(-1, task.dimension),
            self.config.knowledge_size,
            self._coreset_generator,
            device=self.device,
            mode=self.config.coreset_mode,
        )
        return self.backbone.get_prompt_state(), knowledge

    def predict(self, data) -> VisionPredictionResults:
        scores, maps = self._score_dataset(data, [(task.prompt, task.knowledge) for task in self.memory.tasks])
        return VisionPredictionResults(y_pred=np.zeros_like(scores, dtype=int), anomaly_scores=scores, score_maps=maps)

    def _score_dataset(self, data, states: Sequence[PromptedBank]) -> tuple[np.ndarray, np.ndarray]:
        self.backbone.eval()
        test_loader = self._as_loader(data, shuffle=False)
        scores: list[np.ndarray] = []
        maps: list[np.ndarray] = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting", leave=False):
                images = batch["image"].to(self.device)
                task_ids = self.memory.select_tasks(self._aggregate(self.backbone.extract_features(images)))

                batch_scores, batch_maps = self._score_batch(images, task_ids, states)
                scores.append(batch_scores)
                maps.append(batch_maps)

        return np.concatenate(scores), np.concatenate(maps)

    def _score_batch(
        self, images: torch.Tensor, task_ids: torch.Tensor, states: Sequence[PromptedBank]
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_scores = np.empty(len(images))
        batch_maps = np.empty((len(images), *self.config.input_size))

        for task_idx in task_ids.unique().tolist():
            selected = task_ids == task_idx
            prompt, knowledge = states[task_idx]
            self.backbone.set_prompt_state(prompt)

            prompted_features = self._aggregate(self.backbone.extract_features_with_prompt(images[selected]))
            img_scores, task_maps = self.scorer.predict(
                test_features=prompted_features,
                knowledge_bank=knowledge.to(self.device),
                input_size=self.config.input_size,
            )

            selected_np = selected.cpu().numpy()
            batch_scores[selected_np] = img_scores
            batch_maps[selected_np] = task_maps

        return batch_scores, batch_maps
