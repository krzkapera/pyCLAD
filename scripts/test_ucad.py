import numpy as np
import pytest
import torch

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.vision.data.vision_concept import VisionConcept
from pyclad.vision.models.ucad import UCADConfig, UCADModel
from pyclad.vision.models.ucad.coreset import greedy_coreset_sampling
from pyclad.vision.models.ucad.inputs import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageArrayDataset,
    ImagePathDataset,
    build_dataset,
)
from pyclad.vision.models.ucad.memory import TaskMemoryBank, TaskState
from pyclad.vision.models.ucad.prompt import PrefixTuningPrompt
from pyclad.vision.models.ucad.sam import MaskProvider

INPUT_SIZE = (224, 224)


class ConstantMaskProvider(MaskProvider):
    def get_masks(self, image_paths, target_size=(14, 14)):
        labels = torch.arange(target_size[0] * target_size[1]) % 3
        return labels.float().repeat(len(image_paths), 1)


def _tiny_config(**overrides) -> UCADConfig:
    defaults = dict(
        vit_model_name="vit_tiny_patch16_224",
        pretrained=False,
        feature_layer=2,
        num_prompt_layers=2,
        training_epochs=1,
        batch_size=2,
        key_size=8,
        knowledge_size=8,
        max_tasks=2,
        device="cpu",
        input_size=INPUT_SIZE,
    )
    defaults.update(overrides)
    return UCADConfig(**defaults)


def _images(rng, n: int, brightness: int = 128) -> np.ndarray:
    noise = rng.random((n, 32, 32, 3)) * 40.0
    return (noise + brightness).clip(0, 255).astype(np.uint8)


def _write_images(tmp_path, rng, n: int) -> np.ndarray:
    from PIL import Image

    paths = []
    for i in range(n):
        path = tmp_path / f"img_{i}.png"
        Image.fromarray(_images(rng, 1)[0]).save(path)
        paths.append(str(path))
    return np.asarray(paths, dtype=object)


class TestInputAdapters:
    def test_path_array_is_read_as_paths(self, tmp_path):
        paths = _write_images(tmp_path, np.random.default_rng(0), 2)

        dataset = build_dataset(paths, INPUT_SIZE, "concept_0")

        assert isinstance(dataset, ImagePathDataset)
        assert dataset[0]["image_path"] == paths[0]

    def test_numpy_array_is_not_mistaken_for_a_concept(self):
        """np.ndarray exposes a .data memoryview; duck-typing on it used to crash the scenario path."""
        images = _images(np.random.default_rng(0), 2)

        dataset = build_dataset(images, INPUT_SIZE, "concept_0")

        assert isinstance(dataset, ImageArrayDataset)
        assert dataset[0]["image"].shape == (3, *INPUT_SIZE)

    def test_concept_is_unwrapped(self):
        images = _images(np.random.default_rng(0), 2)

        dataset = build_dataset(Concept("c", data=images), INPUT_SIZE, "concept_0")

        assert len(dataset) == 2

    def test_uint8_and_unit_float_inputs_agree(self):
        images = _images(np.random.default_rng(0), 2)

        from_uint8 = build_dataset(images, INPUT_SIZE, "c")[0]["image"]
        from_float = build_dataset(images.astype(np.float32) / 255.0, INPUT_SIZE, "c")[0]["image"]

        torch.testing.assert_close(from_uint8, from_float)

    def test_images_are_imagenet_normalized(self):
        white = np.full((1, 32, 32, 3), 255, dtype=np.uint8)

        image = build_dataset(white, INPUT_SIZE, "c")[0]["image"]

        expected = (1.0 - IMAGENET_MEAN) / IMAGENET_STD
        torch.testing.assert_close(image, expected.expand(3, *INPUT_SIZE).contiguous())


class TestMemoryBank:
    def test_stored_state_is_not_aliased_by_the_live_prompt(self):
        prompt = PrefixTuningPrompt(num_layers=1, prompt_length=1, num_heads=2, embed_dim=4)
        memory = TaskMemoryBank(max_tasks=1)
        memory.add_task(0, key=torch.zeros(2, 4), states=[TaskState(prompt_state=prompt.get_prompt_state(), knowledge=torch.zeros(2, 4))])
        stored = memory.get_states(0)[0].prompt_state.clone()

        prompt.reset_prompt()

        torch.testing.assert_close(memory.get_states(0)[0].prompt_state, stored)

    def test_restoring_a_task_does_not_mutate_the_bank(self):
        prompt = PrefixTuningPrompt(num_layers=1, prompt_length=1, num_heads=2, embed_dim=4)
        memory = TaskMemoryBank(max_tasks=1)
        memory.add_task(0, key=torch.zeros(2, 4), states=[TaskState(prompt_state=prompt.get_prompt_state(), knowledge=torch.zeros(2, 4))])
        stored = memory.get_states(0)[0].prompt_state.clone()

        prompt.set_prompt_state(memory.get_states(0)[0].prompt_state)
        prompt.reset_prompt()

        torch.testing.assert_close(memory.get_states(0)[0].prompt_state, stored)

    def test_bank_rejects_more_tasks_than_configured(self):
        memory = TaskMemoryBank(max_tasks=1)
        memory.add_task(0, key=torch.zeros(2, 4), states=[TaskState(prompt_state=torch.zeros(2), knowledge=torch.zeros(2, 4))])

        with pytest.raises(RuntimeError):
            memory.add_task(1, key=torch.zeros(2, 4), states=[TaskState(prompt_state=torch.zeros(2), knowledge=torch.zeros(2, 4))])


class TestCoreset:
    def test_exact_mode_is_deterministic(self):
        features = torch.randn(40, 8)

        first = greedy_coreset_sampling(features, 5, mode="exact")
        second = greedy_coreset_sampling(features, 5, mode="exact")

        torch.testing.assert_close(first, second)

    def test_both_modes_return_the_requested_size(self):
        features = torch.randn(40, 8)

        assert greedy_coreset_sampling(features, 5, mode="exact").shape == (5, 8)
        assert greedy_coreset_sampling(features, 5, mode="approximate").shape == (5, 8)

    def test_returns_input_when_smaller_than_target(self):
        features = torch.randn(3, 8)

        torch.testing.assert_close(greedy_coreset_sampling(features, 5, mode="exact"), features)


class TestModelIntegration:
    def test_predict_does_not_consume_the_global_rng(self):
        """Evaluation must not shift the RNG stream, otherwise measuring perturbs later training."""
        model = UCADModel(_tiny_config(), mask_provider=ConstantMaskProvider())
        images = _images(np.random.default_rng(0), 2)
        model.memory.add_task(
            0,
            key=torch.zeros(4, model.config.target_embed_dimension),
            states=[
                TaskState(
                    prompt_state=model.backbone.get_prompt_state(),
                    knowledge=torch.zeros(4, model.config.target_embed_dimension),
                )
            ],
        )

        torch.manual_seed(7)
        expected = torch.randn(1)
        torch.manual_seed(7)
        model.predict(images)

        torch.testing.assert_close(torch.randn(1), expected)

    def test_frozen_task_keeps_identical_scores_as_the_bank_grows(self):
        """UCAD isolates parameters per task, so an earlier task must score the same after later ones."""
        rng = np.random.default_rng(0)
        model = UCADModel(_tiny_config(), mask_provider=ConstantMaskProvider())
        first_train, first_test = _images(rng, 2, brightness=20), _images(rng, 2, brightness=20)

        model.fit(first_train)
        before = model.predict(first_test).anomaly_scores

        model.fit(_images(rng, 2, brightness=220))
        after = model.predict(first_test).anomaly_scores

        np.testing.assert_allclose(after, before)

    def test_runs_through_the_concept_incremental_scenario(self):
        rng = np.random.default_rng(0)
        train = [Concept(f"t{i}", data=_images(rng, 2)) for i in range(2)]
        test = [
            VisionConcept(f"t{i}", data=_images(rng, 2), labels=np.array([0, 1], dtype=np.int64)) for i in range(2)
        ]
        model = UCADModel(_tiny_config(), mask_provider=ConstantMaskProvider())
        callback = ConceptMetricCallback(base_metric=RocAuc(), summarized_metrics=[BackwardTransfer()])

        ConceptIncrementalScenario(
            ConceptsDataset("dummy", train_concepts=train, test_concepts=test),
            strategy=NaiveStrategy(model),
            callbacks=[callback],
        ).run()

        assert model.memory.num_tasks == 2
        assert set(callback.info()["concept_metric_callback_ROC-AUC"]["metric_matrix"]) == {"t0", "t1"}

    def test_task_key_does_not_depend_on_batch_order(self):
        """The greedy coreset seeds from the first feature, so extraction must not see the shuffled order."""
        images = _images(np.random.default_rng(0), 6)
        keys = []
        for seed in (0, 12345):
            torch.manual_seed(1)
            model = UCADModel(_tiny_config(training_epochs=0, seed=seed), mask_provider=ConstantMaskProvider())
            model.fit(images)
            keys.append(model.memory.tasks[0].key)

        torch.testing.assert_close(keys[0], keys[1])

    def test_info_carries_the_configuration(self):
        model = UCADModel(_tiny_config(patchsize=1), mask_provider=ConstantMaskProvider())

        info = model.info()

        assert info["model"]["name"] == "UCAD"
        assert info["model"]["config"]["patchsize"] == 1
