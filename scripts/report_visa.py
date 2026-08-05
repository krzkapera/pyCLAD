"""Compares a VisA run's JSON output against the numbers UCAD reports for VisA.

Usage: report_visa.py <run.json> [<run.json> ...]

Paper values are the 'Ours' rows of Tables 3 and 4 in Liu et al., "Unsupervised Continual Anomaly
Detection with Contrastively-learned Prompt" (AAAI 2024). The reference column is that paper's own
implementation run over the same VisA copy and masks, so the delta against it isolates our
implementation from the dataset and the mask source.
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
# The authors' code run over our VisA copy and SAM2-derived masks (Ares job 20847574).
REFERENCE_IMAGE_AUROC = {
    "candle": 0.7125, "capsules": 0.8970, "cashew": 0.9440, "chewinggum": 0.9485,
    "fryum": 0.9470, "macaroni1": 0.8120, "macaroni2": 0.5545, "pcb1": 0.9820,
    "pcb2": 0.9415, "pcb3": 0.7720, "pcb4": 0.9395, "pipe_fryum": 0.9870,
}
REFERENCE_PIXEL_AUPR = {
    "candle": 0.0921, "capsules": 0.5396, "cashew": 0.5565, "chewinggum": 0.4060,
    "fryum": 0.3335, "macaroni1": 0.0107, "macaroni2": 0.0082, "pcb1": 0.7652,
    "pcb2": 0.1779, "pcb3": 0.2372, "pcb4": 0.2051, "pipe_fryum": 0.5958,
}
REFERENCE_AVERAGE_IMAGE_AUROC = 0.8698
REFERENCE_AVERAGE_PIXEL_AUPR = 0.3273

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
    header = (
        f"{'class':<12} {'img AUROC':>10} {'paper':>7} {'vs paper':>9} {'ref':>7} {'vs ref':>8}   "
        f"{'pix AUPR':>9} {'paper':>7} {'vs paper':>9} {'ref':>7} {'vs ref':>8}"
    )
    print(header)
    for concept in order:
        ours_image = image["metric_matrix"][final][concept]
        ours_pixel = pixel["metric_matrix"][final][concept] if pixel else float("nan")
        print(
            f"{concept:<12} {ours_image:10.4f} {PAPER_IMAGE_AUROC[concept]:7.3f} "
            f"{ours_image - PAPER_IMAGE_AUROC[concept]:+9.3f} {REFERENCE_IMAGE_AUROC[concept]:7.3f} "
            f"{ours_image - REFERENCE_IMAGE_AUROC[concept]:+8.3f}   "
            f"{ours_pixel:9.4f} {PAPER_PIXEL_AUPR[concept]:7.3f} "
            f"{ours_pixel - PAPER_PIXEL_AUPR[concept]:+9.3f} {REFERENCE_PIXEL_AUPR[concept]:7.3f} "
            f"{ours_pixel - REFERENCE_PIXEL_AUPR[concept]:+8.3f}"
        )

    average_image = sum(image["metric_matrix"][final].values()) / len(order)
    average_pixel = sum(pixel["metric_matrix"][final].values()) / len(order) if pixel else float("nan")
    print(
        f"{'AVERAGE':<12} {average_image:10.4f} {PAPER_AVERAGE_IMAGE_AUROC:7.3f} "
        f"{average_image - PAPER_AVERAGE_IMAGE_AUROC:+9.3f} {REFERENCE_AVERAGE_IMAGE_AUROC:7.3f} "
        f"{average_image - REFERENCE_AVERAGE_IMAGE_AUROC:+8.3f}   "
        f"{average_pixel:9.4f} {PAPER_AVERAGE_PIXEL_AUPR:7.3f} "
        f"{average_pixel - PAPER_AVERAGE_PIXEL_AUPR:+9.3f} {REFERENCE_AVERAGE_PIXEL_AUPR:7.3f} "
        f"{average_pixel - REFERENCE_AVERAGE_PIXEL_AUPR:+8.3f}"
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
