# UCAD

Unsupervised Continual Anomaly Detection with Contrastively-learned Prompt (Liu et al., AAAI 2024).

For each concept the model prefix-tunes a prompt on a frozen ViT-B/16 with a structure-based
contrastive loss over SAM masks, then stores one prompt and one coreset-reduced bank of normal-image
features. At test time it routes each image to a concept by nearest key, scores its patches by
distance to that concept's bank, and reports the maximum as the image score and the upsampled,
smoothed field as the anomaly map. Nothing is shared between concepts, so forgetting is zero by
construction.

## Quick start

```python
from pyclad.vision.data.benchmarks.readers import read_vision_benchmark_dataset
from pyclad.vision.models.ucad import UCADConfig, UCADModel

dataset = read_vision_benchmark_dataset(
    root=MVTEC_ROOT,
    benchmark="mvtec",
    data_mode="paths",
    resize_to=(224, 224),
    resize_mode="short_side_crop",
)

model = UCADModel(
    UCADConfig(
        max_tasks=len(dataset.train_concepts()),
        sam_masks_dir=MVTEC_MASKS_ROOT,
        sam_images_root=MVTEC_ROOT,
    )
)
```

`examples/ucad_mvtec_example.py` and `examples/ucad_visa_example.py` run this through
`ConceptIncrementalScenario` with the image- and pixel-level callbacks.

## What you need besides the images

**SAM masks.** The contrastive loss draws patches within a SAM mask together and pushes the rest
apart, so it needs one mask per training image. Point `sam_masks_dir` at a directory mirroring the
dataset root and `sam_images_root` at that root; leave `sam_masks_dir` unset to generate masks online
instead, which is far slower.

Watch out: when a mask file is missing, `SAM2OfflineMaskProvider` logs a warning and substitutes
zeros. A whole run with a wrong mask directory therefore trains on a uniform mask - every patch in
one region, no negatives - and finishes without an error, several points below where it should be.
Check the warning count before trusting a run.

**Paths, not arrays.** Read the dataset with `data_mode="paths"`. The model loads images itself
because the mask provider looks masks up by image path.

## Configuration

The defaults follow the reference implementation, so an out-of-the-box run is comparable to it:

| | default | source |
|---|---|---|
| `input_size` / `resize_mode` | (224, 224) / `short_side_crop` | reference's `Resize(224)` + `CenterCrop(224)`; paper fixes 224 only |
| `training_epochs` / `batch_size` | 25 / 8 | paper and reference agree |
| `learning_rate` / `grad_clip` | 5e-4 / 1.0 | paper and reference agree |
| `knowledge_size` / `key_size` | 196 / 196 | paper's memory accounting |
| `prompt_length` / `num_prompt_layers` / `feature_layer` | 1 / 12 / 5 | reference |
| `scl_temperature` | 0.5 | reference |
| `coreset_mode` | `approximate` | reference; `greedy` is PatchCore's exact sampler and needs O(N^2) memory |
| `blur_sigma` | 4.0 | reference hardcodes 4 |
| `reweighting_num_nn` | 0, off | reference does not apply the paper's Eq. 5-6 |
| `reset_prompt_per_task` | True | differs from the reference, which carries the prompt over |
| `seed` | 0 | covers every draw a run makes |

## Two things that will surprise you

**`resize_mode` has to match the reader.** The model reloads images from paths, so it holds its own
copy of the geometry. If it disagrees with the mode the ground-truth masks were read with, the masks
and the anomaly maps are misaligned and the pixel metric measures the misalignment rather than the
model - silently, since both are valid arrays of the right shape.

**Every concept shuffles with the same permutation.** `_as_loader` builds a fresh generator seeded
from `config.seed` on each call, so concept 2 iterates its data in the same permutation sequence as
concept 1. It is deterministic and not wrong, but it is also not what "shuffle with seed s" usually
implies, and it is the place to change if you want concepts to see different orders.

## What to expect

The reference reports numbers obtained under an evaluation protocol its paper does not describe: it
averages the scores of all 25 epochs and then reports the epoch whose test-set image AUROC is
highest. pyCLAD does neither by default, so a plain run lands **around 0.70 image AUROC on VisA**
rather than the paper's 0.874. `docs/ucad_paper_reference_pyclad.md` explains the mechanisms and what
each is worth; the tables below are pyCLAD and the reference measured side by side under both.

Three seeds per side. "reference protocol" is the reference's own reading, with the epoch chosen on
the test set; "without selection" is the same 25-epoch ensemble read after the last epoch.

### VisA, twelve categories, image AUROC

| category | pyCLAD without selection | pyCLAD reference-protocol | reference without selection | reference protocol | paper |
|---|---|---|---|---|---|
| candle | 0.5013 +- 0.0229 | 0.7278 +- 0.0891 | 0.4352 +- 0.0721 | 0.7325 +- 0.0478 | 0.778 |
| capsules | 0.8668 +- 0.0228 | 0.8748 +- 0.0145 | 0.8623 +- 0.0210 | 0.8682 +- 0.0255 | 0.877 |
| cashew | 0.9523 +- 0.0195 | 0.9577 +- 0.0143 | 0.9477 +- 0.0060 | 0.9483 +- 0.0055 | 0.960 |
| chewinggum | 0.9552 +- 0.0063 | 0.9588 +- 0.0033 | 0.9470 +- 0.0015 | 0.9498 +- 0.0019 | 0.958 |
| fryum | 0.9087 +- 0.0157 | 0.9148 +- 0.0113 | 0.9427 +- 0.0095 | 0.9487 +- 0.0043 | 0.945 |
| macaroni1 | 0.7028 +- 0.0267 | 0.7922 +- 0.0158 | 0.6633 +- 0.0316 | 0.7655 +- 0.0422 | 0.823 |
| macaroni2 | 0.5790 +- 0.0348 | 0.6213 +- 0.0101 | 0.5837 +- 0.0743 | 0.6160 +- 0.0577 | 0.667 |
| pcb1 | 0.9387 +- 0.0064 | 0.9527 +- 0.0060 | 0.9625 +- 0.0226 | 0.9718 +- 0.0194 | 0.905 |
| pcb2 | 0.9398 +- 0.0093 | 0.9452 +- 0.0119 | 0.9432 +- 0.0199 | 0.9528 +- 0.0171 | 0.871 |
| pcb3 | 0.7412 +- 0.0053 | 0.7788 +- 0.0137 | 0.7545 +- 0.0105 | 0.7788 +- 0.0062 | 0.813 |
| pcb4 | 0.9350 +- 0.0241 | 0.9547 +- 0.0297 | 0.9172 +- 0.0103 | 0.9415 +- 0.0048 | 0.901 |
| pipe_fryum | 0.9815 +- 0.0114 | 0.9908 +- 0.0032 | 0.9857 +- 0.0042 | 0.9935 +- 0.0058 | 0.988 |
| **average** | **0.8335 +- 0.0087** | **0.8725 +- 0.0046** | **0.8287 +- 0.0042** | **0.8723 +- 0.0026** | **0.874** |

### VisA, pixel AUPR

| category | pyCLAD without selection | pyCLAD reference-protocol | reference without selection | reference protocol | paper |
|---|---|---|---|---|---|
| candle | 0.1229 +- 0.0107 | 0.1189 +- 0.0313 | 0.0867 +- 0.0047 | 0.0834 +- 0.0163 | 0.067 |
| capsules | 0.6152 +- 0.0127 | 0.6125 +- 0.0055 | 0.5688 +- 0.0053 | 0.5602 +- 0.0179 | 0.437 |
| cashew | 0.6356 +- 0.0088 | 0.6088 +- 0.0157 | 0.5661 +- 0.0121 | 0.5576 +- 0.0072 | 0.580 |
| chewinggum | 0.3339 +- 0.0129 | 0.3577 +- 0.0409 | 0.3354 +- 0.0180 | 0.3941 +- 0.0190 | 0.503 |
| fryum | 0.4117 +- 0.0058 | 0.4041 +- 0.0121 | 0.3408 +- 0.0071 | 0.3333 +- 0.0019 | 0.334 |
| macaroni1 | 0.0855 +- 0.0044 | 0.0698 +- 0.0032 | 0.0231 +- 0.0018 | 0.0154 +- 0.0054 | 0.013 |
| macaroni2 | 0.0296 +- 0.0049 | 0.0315 +- 0.0034 | 0.0127 +- 0.0006 | 0.0090 +- 0.0029 | 0.003 |
| pcb1 | 0.7910 +- 0.0105 | 0.7772 +- 0.0236 | 0.7455 +- 0.0248 | 0.7447 +- 0.0185 | 0.702 |
| pcb2 | 0.3118 +- 0.0092 | 0.3091 +- 0.0105 | 0.1925 +- 0.0020 | 0.1797 +- 0.0058 | 0.136 |
| pcb3 | 0.2744 +- 0.0238 | 0.2919 +- 0.0050 | 0.2572 +- 0.0022 | 0.2384 +- 0.0168 | 0.266 |
| pcb4 | 0.2079 +- 0.0117 | 0.2453 +- 0.0282 | 0.1815 +- 0.0085 | 0.2049 +- 0.0088 | 0.106 |
| pipe_fryum | 0.6881 +- 0.0155 | 0.6255 +- 0.0246 | 0.6260 +- 0.0105 | 0.6021 +- 0.0087 | 0.457 |
| **average** | **0.3756 +- 0.0008** | **0.3710 +- 0.0087** | **0.3280 +- 0.0034** | **0.3269 +- 0.0042** | **0.300** |

### MVTec AD, fifteen categories, image AUROC

| category | pyCLAD without selection | pyCLAD reference-protocol | reference without selection | reference protocol | paper |
|---|---|---|---|---|---|
| bottle* | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 1.000 |
| cable | 0.7207 +- 0.0058 | 0.7224 +- 0.0048 | 0.7216 +- 0.0055 | 0.7280 +- 0.0058 | 0.751 |
| capsule | 0.9261 +- 0.0055 | 0.9298 +- 0.0084 | 0.9166 +- 0.0179 | 0.9273 +- 0.0037 | 0.866 |
| carpet | 0.9783 +- 0.0012 | 0.9848 +- 0.0008 | 0.9819 +- 0.0028 | 0.9853 +- 0.0036 | 0.965 |
| grid | 0.9610 +- 0.0049 | 0.9749 +- 0.0055 | 0.9713 +- 0.0034 | 0.9791 +- 0.0051 | 0.944 |
| hazelnut* | 0.9948 +- 0.0062 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 0.994 |
| leather* | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 1.000 |
| metal_nut | 0.9977 +- 0.0020 | 0.9990 +- 0.0010 | 0.9977 +- 0.0003 | 0.9980 +- 0.0005 | 0.988 |
| pill | 0.9543 +- 0.0074 | 0.9574 +- 0.0070 | 0.9512 +- 0.0057 | 0.9588 +- 0.0070 | 0.894 |
| screw | 0.4947 +- 0.0457 | 0.5358 +- 0.0365 | 0.5554 +- 0.0578 | 0.6195 +- 0.0431 | 0.739 |
| tile* | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 0.998 |
| toothbrush* | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 1.000 |
| transistor | 0.9507 +- 0.0015 | 0.9568 +- 0.0054 | 0.9529 +- 0.0044 | 0.9625 +- 0.0045 | 0.874 |
| wood | 0.9959 +- 0.0010 | 0.9959 +- 0.0010 | 0.9930 +- 0.0009 | 0.9953 +- 0.0018 | 0.995 |
| zipper | 0.9623 +- 0.0044 | 0.9685 +- 0.0034 | 0.9674 +- 0.0021 | 0.9680 +- 0.0015 | 0.938 |
| **average** | **0.9291 +- 0.0029** | **0.9350 +- 0.0024** | **0.9339 +- 0.0023** | **0.9415 +- 0.0021** | **0.930** |

### MVTec AD, pixel AUPR

| category | pyCLAD without selection | pyCLAD reference-protocol | reference without selection | reference protocol | paper |
|---|---|---|---|---|---|
| bottle* | 0.8313 +- 0.0048 | 0.8337 +- 0.0053 | 0.7822 +- 0.0055 | 0.7822 +- 0.0055 | 0.752 |
| cable | 0.2344 +- 0.0153 | 0.2222 +- 0.0135 | 0.1920 +- 0.0099 | 0.1801 +- 0.0241 | 0.290 |
| capsule | 0.4323 +- 0.0074 | 0.4326 +- 0.0084 | 0.3656 +- 0.0047 | 0.3631 +- 0.0016 | 0.349 |
| carpet | 0.6804 +- 0.0083 | 0.7212 +- 0.0006 | 0.6281 +- 0.0060 | 0.6564 +- 0.0077 | 0.622 |
| grid | 0.3071 +- 0.0018 | 0.3065 +- 0.0215 | 0.2241 +- 0.0016 | 0.2241 +- 0.0029 | 0.187 |
| hazelnut* | 0.6449 +- 0.0076 | 0.6246 +- 0.0128 | 0.5162 +- 0.0074 | 0.5162 +- 0.0074 | 0.506 |
| leather* | 0.5210 +- 0.0044 | 0.4996 +- 0.0116 | 0.4188 +- 0.0055 | 0.4188 +- 0.0055 | 0.333 |
| metal_nut | 0.8103 +- 0.0039 | 0.7929 +- 0.0189 | 0.7713 +- 0.0045 | 0.7699 +- 0.0029 | 0.775 |
| pill | 0.6899 +- 0.0027 | 0.6803 +- 0.0085 | 0.6420 +- 0.0153 | 0.6314 +- 0.0239 | 0.634 |
| screw | 0.2789 +- 0.0105 | 0.1194 +- 0.0779 | 0.2121 +- 0.0039 | 0.1626 +- 0.0418 | 0.214 |
| tile* | 0.5983 +- 0.0066 | 0.5886 +- 0.0091 | 0.5226 +- 0.0062 | 0.5226 +- 0.0062 | 0.549 |
| toothbrush* | 0.5015 +- 0.0005 | 0.4984 +- 0.0116 | 0.3383 +- 0.0039 | 0.3383 +- 0.0039 | 0.298 |
| transistor | 0.5321 +- 0.0124 | 0.5145 +- 0.0260 | 0.4821 +- 0.0138 | 0.4796 +- 0.0142 | 0.398 |
| wood | 0.6199 +- 0.0033 | 0.6294 +- 0.0079 | 0.5829 +- 0.0033 | 0.5863 +- 0.0096 | 0.535 |
| zipper | 0.4876 +- 0.0054 | 0.4938 +- 0.0126 | 0.4264 +- 0.0072 | 0.4234 +- 0.0039 | 0.398 |
| **average** | **0.5447 +- 0.0010** | **0.5305 +- 0.0024** | **0.4736 +- 0.0037** | **0.4703 +- 0.0030** | **0.456** |

\* On these five categories the reference stops training as soon as the test-set image AUROC reaches
1.0, so it has no reading free of that choice and its two columns are the same number by
construction. No VisA category reaches 1.0.

pyCLAD's pixel columns are higher than the reference's throughout because the two count ground-truth
pixels differently, not because the maps are better; equalise the convention and the two agree to
0.006 on VisA and 0.003 on MVTec. Forgetting is 0.0000 on both benchmarks - every post-learning cell
of the concept matrix repeats bit for bit and task routing is correct on every test image - against
the paper's FM 0.039.

## Known limitations

**The contrastive loss does not help on these benchmarks.** After one epoch the prompted features are
99.9% cosine-identical to the frozen ones, so the model is PatchCore on a frozen ViT. After 25 epochs
they have moved a long way in the wrong direction - mean pairwise cosine falls from 0.42 to 0.18 and
effective rank from 297 to 217 - which erodes the locality nearest-neighbour scoring depends on. A
no-SCL configuration reaches 0.7833/0.2718 on VisA, matching the paper's own CPM-only row.

**MVTec's screw is out of reach.** Both implementations are near-random and neither approaches the
paper's 0.739. It is the feature grid, not the training: ViT-B/16 at 224 gives 14x14 patches, one
patch covering 73x73 pixels of the 1024x1024 original, while screw's defects are a few pixels wide.
At zero epochs screw already scores 0.5638; with `vit_base_patch8_224`, which gives 28x28 patches at
the same input size, 0.6602 at zero epochs and 0.8660 after 25.

## Reproducing the tables

- `scripts/protocol_compare.py` trains one model per concept and reports both protocols from that one
  run, plus the per-epoch trajectory. `scripts/protocol_compare.sbatch` submits it.
- `scripts/reference_ensemble.py` holds the authors' epoch ensemble as a subclass of the model. It is
  reproduction machinery and deliberately outside the library.
- `scripts/pixel_convention_effect.py` scores one run against both ground-truth mask conventions.
- `scripts/ucad_probe.py` runs the model through the full continual scenario when the concept matrix
  is what is wanted.

## See also

`docs/ucad_paper_reference_pyclad.md` - what the paper specifies, what the authors' code actually
does, and where pyCLAD departs from either.
