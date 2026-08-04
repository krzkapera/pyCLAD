import logging
from typing import Any, Dict, Optional, Union

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
from .scoring import NearestNeighborScorer, combine_members
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
            seed=self.config.seed,
        ).to(self.device)

        self.memory = TaskMemoryBank(max_tasks=self.config.max_tasks)
        self.scorer = NearestNeighborScorer(
            num_nn=self.config.anomaly_scorer_num_nn,
            reweighting_num_nn=self.config.reweighting_num_nn,
            blur_sigma=self.config.blur_sigma,
        )
        self.mask_provider = mask_provider if mask_provider is not None else create_mask_provider(
            self.config, device=self.device
        )
        self.current_task_id = 0

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
            generator=torch.Generator().manual_seed(self.config.seed),
        )

    def _sequential_view(self, loader: DataLoader) -> DataLoader:
        """Coreset selection is order-dependent, so it must not see the shuffled training order."""
        return DataLoader(
            loader.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            generator=torch.Generator().manual_seed(self.config.seed),
        )

    def fit(self, training_data):
        logger.info(f"Training UCAD on task {self.current_task_id}")
        train_loader = self._as_loader(training_data, shuffle=True)
        extraction_loader = self._sequential_view(train_loader)

        self.backbone.eval()
        key_features = self._extract_all_features(extraction_loader, use_prompt=False)

        C = key_features.shape[-1]
        key_flat = key_features.reshape(-1, C)

        logger.info(f"Computing coreset for Task Key (target: {self.config.key_size})")
        task_key = greedy_coreset_sampling(
            key_flat, self.config.key_size, device=self.device, mode=self.config.coreset_mode
        )

        if self.config.reset_prompt_per_task:
            self.backbone.reset_prompt()
        elif self.memory.num_tasks > 0:
            self.backbone.set_prompt_state(self.memory.get_states(self.memory.num_tasks - 1)[-1].prompt_state)

        optimizer = optim.Adam(self.backbone.prompt_module.parameters(), lr=self.config.learning_rate)

        self.backbone.train()
        first_snapshot_epoch = max(self.config.training_epochs - self.config.score_ensemble_epochs, 0)
        states: list[tuple[torch.Tensor, torch.Tensor]] = []

        for epoch in range(self.config.training_epochs):
            total_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.training_epochs}", leave=False):
                images = batch["image"].to(self.device)
                paths = batch["image_path"]

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

            if epoch >= first_snapshot_epoch:
                states.append(self._snapshot_state(extraction_loader, C))
                self.backbone.train()

        if not states:
            states.append(self._snapshot_state(extraction_loader, C))

        self.memory.add_task(task_id=self.current_task_id, key=task_key, states=states)

        logger.info(f"Task {self.current_task_id} added to memory bank.")
        self.current_task_id += 1

    def _snapshot_state(self, extraction_loader: DataLoader, dimension: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The current prompt together with the knowledge bank it produces."""
        self.backbone.eval()
        knowledge_features = self._extract_all_features(extraction_loader, use_prompt=True)

        logger.info(f"Computing coreset for Task Knowledge (target: {self.config.knowledge_size})")
        knowledge = greedy_coreset_sampling(
            knowledge_features.reshape(-1, dimension),
            self.config.knowledge_size,
            device=self.device,
            mode=self.config.coreset_mode,
        )
        return self.backbone.get_prompt_state(), knowledge

    def predict(self, data) -> VisionPredictionResults:
        self.backbone.eval()
        test_loader = self._as_loader(data, shuffle=False)

        members = max(len(task.states) for task in self.memory.tasks)
        member_scores: list[list[np.ndarray]] = [[] for _ in range(members)]
        member_maps: list[list[np.ndarray]] = [[] for _ in range(members)]

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting", leave=False):
                images = batch["image"].to(self.device)

                frozen_features = self._aggregate(self.backbone.extract_features(images))
                task_ids = self.memory.select_tasks(frozen_features)

                for member in range(members):
                    batch_scores = np.empty(len(images))
                    batch_maps = np.empty((len(images), *self.config.input_size))

                    for task_idx in task_ids.unique().tolist():
                        selected = task_ids == task_idx
                        states = self.memory.get_states(task_idx)
                        state = states[min(member, len(states) - 1)]
                        self.backbone.set_prompt_state(state.prompt_state)

                        prompted_features = self._aggregate(
                            self.backbone.extract_features_with_prompt(images[selected])
                        )
                        img_scores, maps = self.scorer.predict(
                            test_features=prompted_features,
                            knowledge_bank=state.knowledge.to(self.device),
                            input_size=self.config.input_size,
                        )

                        selected_np = selected.cpu().numpy()
                        batch_scores[selected_np] = img_scores
                        batch_maps[selected_np] = maps

                    member_scores[member].append(batch_scores)
                    member_maps[member].append(batch_maps)

        image_scores, anomaly_maps = combine_members(
            [np.concatenate(scores) for scores in member_scores],
            [np.concatenate(maps) for maps in member_maps],
        )
        return VisionPredictionResults(
            y_pred=np.zeros_like(image_scores, dtype=int),
            anomaly_scores=image_scores,
            score_maps=anomaly_maps,
        )
