from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.vision.data.base import ImageLoadOptions
from pyclad.vision.data.benchmarks.readers import VisABenchmarkReader
from pyclad.vision.data.lazy_concepts import LazyVisionConceptList
from pyclad.vision.data.sample import VisionSample

CONTINUAL_MEGA_SCENARIOS = (1, 2, 3)
CONTINUAL_MEGA_TASK_SIZES = (5, 10, 30)
CONTINUAL_MEGA_BASE_GROUP = "base"
CONTINUAL_MEGA_IMAGE_SIZE = (336, 336)
CONTINUAL_MEGA_INTERPOLATION = "bicubic"
CONTINUAL_MEGA_ZERO_SHOT_DIRS = {"mvtec": "mvtec_anomaly_detection", "visa": "VisA_20220922"}
CONTINUAL_MEGA_TRAIN_SAMPLES = ("all", "normal")

TestConcepts = List[Tuple[str, List[VisionSample]]]


@dataclass(frozen=True)
class ConceptGroup:
    name: str
    train_samples: List[VisionSample]
    test_concepts: TestConcepts


class ContinualMegaDataset(ConceptsDataset):
    def __init__(
        self,
        data_root: Union[str, Path],
        meta_dir: Union[str, Path],
        scenario: int,
        task_size: int,
        zero_shot: bool = False,
        train_samples: str = "all",
        data_mode: str = "numpy",
        resize_to: Optional[Tuple[int, int]] = CONTINUAL_MEGA_IMAGE_SIZE,
        color_mode: str = "rgb",
        interpolation: str = CONTINUAL_MEGA_INTERPOLATION,
        zero_shot_dirs: Mapping[str, str] = CONTINUAL_MEGA_ZERO_SHOT_DIRS,
    ):
        if scenario not in CONTINUAL_MEGA_SCENARIOS:
            raise ValueError(f"scenario must be one of {CONTINUAL_MEGA_SCENARIOS}, got {scenario}")
        if task_size not in CONTINUAL_MEGA_TASK_SIZES:
            raise ValueError(f"task_size must be one of {CONTINUAL_MEGA_TASK_SIZES}, got {task_size}")
        if train_samples not in CONTINUAL_MEGA_TRAIN_SAMPLES:
            raise ValueError(f"train_samples must be one of {CONTINUAL_MEGA_TRAIN_SAMPLES}, got {train_samples!r}")
        if zero_shot and scenario == 1:
            raise ValueError("Scenario 1 trains on MVTec-AD and VisA, so it has no held-out zero-shot datasets")

        self._scenario = scenario
        self._task_size = task_size
        resolved_data_root = Path(data_root).expanduser().resolve()
        resolved_meta_dir = Path(meta_dir).expanduser().resolve()

        options = ImageLoadOptions(
            data_mode=data_mode,
            resize_to=resize_to,
            color_mode=color_mode,
            interpolation=interpolation,
            apply_exif_transpose=True,
        )
        stream_groups = read_stream_groups(
            meta_dir=resolved_meta_dir,
            data_root=resolved_data_root,
            scenario=scenario,
            task_size=task_size,
            train_samples=train_samples,
        )
        zero_shot_groups = (
            read_zero_shot_groups(
                meta_dir=resolved_meta_dir,
                data_root=resolved_data_root,
                zero_shot_dirs=zero_shot_dirs,
            )
            if zero_shot
            else []
        )

        self._group_by_concept = {
            concept_name: group.name
            for group in stream_groups + zero_shot_groups
            for concept_name, _ in group.test_concepts
        }
        self._zero_shot_groups = [group.name for group in zero_shot_groups]

        super().__init__(
            name=f"Continual-MEGA-scenario{scenario}-{task_size}classes",
            train_concepts=LazyVisionConceptList(
                samples_by_concept=[(group.name, group.train_samples) for group in stream_groups],
                options=options,
                with_labels=True,
            ),
            test_concepts=LazyVisionConceptList(
                samples_by_concept=[
                    concept for group in stream_groups + zero_shot_groups for concept in group.test_concepts
                ],
                options=options,
                with_labels=True,
            ),
        )

    def group_by_concept(self) -> Dict[str, str]:
        return dict(self._group_by_concept)

    def zero_shot_groups(self) -> List[str]:
        return list(self._zero_shot_groups)

    def additional_info(self) -> Dict[str, Any]:
        return {
            **super().additional_info(),
            "scenario": self._scenario,
            "task_size": self._task_size,
            "zero_shot_groups": self._zero_shot_groups,
        }


def read_stream_groups(
    meta_dir: Path,
    data_root: Path,
    scenario: int,
    task_size: int,
    train_samples: str = "all",
) -> List[ConceptGroup]:
    base_meta = _read_meta(meta_dir / f"scenario{scenario}_base.json")
    task_meta = _read_meta(meta_dir / f"scenario{scenario}_{task_size}classes_tasks.json")
    task_names = sorted(task_meta, key=lambda name: int(name.split("_")[1]))

    groups = [_build_group(CONTINUAL_MEGA_BASE_GROUP, base_meta, data_root, train_samples)]
    groups.extend(_build_group(name, task_meta[name], data_root, train_samples) for name in task_names)
    return groups


def read_zero_shot_groups(
    meta_dir: Path,
    data_root: Path,
    zero_shot_dirs: Mapping[str, str] = CONTINUAL_MEGA_ZERO_SHOT_DIRS,
) -> List[ConceptGroup]:
    mvtec_meta = _read_meta(meta_dir / "meta_mvtec.json")
    mvtec_root = data_root / zero_shot_dirs["mvtec"]
    mvtec_concepts = [
        (f"mvtec_{category}", _samples_from_meta(entries, "test", mvtec_root, category))
        for category, entries in mvtec_meta["test"].items()
    ]

    visa_reader = VisABenchmarkReader(root=data_root / zero_shot_dirs["visa"])
    visa_by_category: Dict[str, List[VisionSample]] = {}
    for sample in visa_reader.index_samples():
        if sample.split == "test":
            visa_by_category.setdefault(f"visa_{sample.category}", []).append(sample)

    return [
        ConceptGroup(name="zeroshot_mvtec", train_samples=[], test_concepts=mvtec_concepts),
        ConceptGroup(name="zeroshot_visa", train_samples=[], test_concepts=list(visa_by_category.items())),
    ]


def _build_group(name: str, meta: Mapping[str, Any], data_root: Path, train_samples: str) -> ConceptGroup:
    train: List[VisionSample] = []
    for category, entries in meta["train"].items():
        train.extend(_samples_from_meta(entries, "train", data_root, category))
    if train_samples == "normal":
        train = [sample for sample in train if sample.image_label == 0]

    test_concepts = [
        (category, _samples_from_meta(entries, "test", data_root, category))
        for category, entries in meta["test"].items()
    ]
    return ConceptGroup(name=name, train_samples=train, test_concepts=test_concepts)


def _samples_from_meta(
    entries: Sequence[Mapping[str, Any]],
    split: str,
    root: Path,
    category: str,
) -> List[VisionSample]:
    return [
        VisionSample(
            category=category,
            split=split,
            image_path=root / entry["img_path"],
            image_label=int(entry["anomaly"]),
            mask_path=root / entry["mask_path"] if entry["mask_path"] else None,
            defect_type=entry.get("specie_name") or None,
        )
        for entry in entries
    ]


def _read_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Continual-MEGA meta file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
