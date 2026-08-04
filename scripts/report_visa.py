"""Compares a VisA run's JSON output against the numbers UCAD reports for VisA.

Usage: report_visa.py <run.json> [<run.json> ...]

Reference values are the 'Ours' rows of Tables 3 and 4 in Liu et al., "Unsupervised Continual
Anomaly Detection with Contrastively-learned Prompt" (AAAI 2024).
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PAPER_IMAGE_AUROC = {
    "candle": 0.778, "capsules": 0.877, "cashew": 0.960, "chewinggum": 0.958,
    "fryum": 0.945, "macaroni1": 0.823, "macaroni2": 0.667, "pcb1": 0.905,
    "pcb2": 0.871, "pcb3": 0.813, "pcb4": 0.901, "pipe_fryum": 0.988,
}
PAPER_PIXEL_AUPR = {
    "candle": 0.067, "capsules": 0.437, "cashew": 0.580, "chewinggum": 0.503,
    "fryum": 0.334, "macaroni1": 0.013, "macaroni2": 0.003, "pcb1": 0.702,
    "pcb2": 0.136, "pcb3": 0.266, "pcb4": 0.106, "pipe_fryum": 0.457,
}
PAPER_AVERAGE_IMAGE_AUROC = 0.874
PAPER_AVERAGE_PIXEL_AUPR = 0.300
PAPER_FM_IMAGE = 0.039
PAPER_FM_PIXEL = 0.015

Matrix = Dict[str, Dict[str, float]]


def callback_output(run: dict, key_prefix: str) -> Optional[dict]:
    key = next((key for key in run if key.startswith(key_prefix)), None)
    return run[key] if key else None


def forgetting_measure(matrix: Matrix, order: List[str]) -> float:
    """Eq. 7: mean over the previously learned concepts of (best earlier value - final value)."""
    final = order[-1]
    drops = [
        max(matrix[learned][evaluated] for learned in order[: index + 1]) - matrix[final][evaluated]
        for index, evaluated in enumerate(order[:-1])
    ]
    return sum(drops) / len(drops) if drops else 0.0


def report(path: Path) -> None:
    run = json.loads(path.read_text())
    image = callback_output(run, "concept_metric_callback_ROC-AUC")
    pixel = callback_output(run, "pixel_concept_metric_callback_Pixel-AP")
    order = image["concepts_order"]
    final = order[-1]

    print(f"== {path.name}")
    print(f"{'class':<12} {'img AUROC':>10} {'paper':>7} {'diff':>7}   {'pix AUPR':>9} {'paper':>7} {'diff':>7}")
    for concept in order:
        ours_image = image["metric_matrix"][final][concept]
        ours_pixel = pixel["metric_matrix"][final][concept] if pixel else float("nan")
        print(
            f"{concept:<12} {ours_image:10.4f} {PAPER_IMAGE_AUROC[concept]:7.3f} "
            f"{ours_image - PAPER_IMAGE_AUROC[concept]:+7.3f}   "
            f"{ours_pixel:9.4f} {PAPER_PIXEL_AUPR[concept]:7.3f} "
            f"{ours_pixel - PAPER_PIXEL_AUPR[concept]:+7.3f}"
        )

    average_image = sum(image["metric_matrix"][final].values()) / len(order)
    average_pixel = sum(pixel["metric_matrix"][final].values()) / len(order) if pixel else float("nan")
    print(
        f"{'AVERAGE':<12} {average_image:10.4f} {PAPER_AVERAGE_IMAGE_AUROC:7.3f} "
        f"{average_image - PAPER_AVERAGE_IMAGE_AUROC:+7.3f}   "
        f"{average_pixel:9.4f} {PAPER_AVERAGE_PIXEL_AUPR:7.3f} "
        f"{average_pixel - PAPER_AVERAGE_PIXEL_AUPR:+7.3f}"
    )

    print(
        f"{'FM':<12} {forgetting_measure(image['metric_matrix'], order):10.4f} {PAPER_FM_IMAGE:7.3f} "
        f"{'':7}   "
        + (f"{forgetting_measure(pixel['metric_matrix'], order):9.4f} {PAPER_FM_PIXEL:7.3f}" if pixel else "")
    )
    print(f"summarized: image {image['metrics']}")
    if pixel:
        print(f"            pixel {pixel['metrics']}")
    print()


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        report(Path(argument))
