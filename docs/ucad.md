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
instead, which is far slower. A missing mask raises `FileNotFoundError` - without one the loss would
see a single uniform region and no negative pairs at all, so a wrong mask directory has to stop the
run rather than quietly cost it several points. With `training_epochs=0` no mask is ever read and the
provider is never built.

**Paths, not arrays.** Read the dataset with `data_mode="paths"`. The model loads the images itself
because the mask provider resolves masks by image path; with arrays the paths are synthesised names,
and the mask provider raises rather than train on nothing.

**`resize_mode` has to match the reader.** The model reloads images from paths, so it holds its own
copy of the geometry. If it disagrees with the mode the ground-truth masks were read with, the masks
and the anomaly maps are misaligned and the pixel metric measures the misalignment rather than the
model - silently, since both are valid arrays of the right shape. The option exists because the
original method resizes the short side and centre-crops.

## Configuration

| | default |
|---|---|
| `vit_model_name` / `pretrained` | `vit_base_patch16_224` / True |
| `input_size` / `resize_mode` | (224, 224) / `short_side_crop` |
| `training_epochs` / `batch_size` | 25 / 8 |
| `learning_rate` / `grad_clip` | 5e-4 / 1.0 |
| `knowledge_size` / `key_size` | 196 / 196 |
| `prompt_length` / `num_prompt_layers` / `feature_layer` | 1 / 12 / 5 |
| `scl_temperature` | 0.5 |
| `coreset_mode` | `approximate`; `greedy` is PatchCore's exact sampler and needs O(N^2) memory |
| `blur_sigma` | 4.0 |
| `reweighting_num_nn` | 0, off |
| `reset_prompt_per_task` | True |
| `seed` | 0, covers every draw a run makes |

**Which pretrained weights you get matters, by up to 0.07 image AUROC.** `pretrained=True` hands the
name to timm, which currently resolves `vit_base_patch16_224` to an ImageNet-21k checkpoint
fine-tuned on ImageNet-1k. The published method uses ImageNet-21k with no fine-tuning. Set
`vit_model_name="vit_base_patch16_224.orig_in21k"` for that one. The library keeps timm's default so
that a plain run does not depend on which timm version you have.

**`knowledge_size` is the strongest knob in the model.** It is the number of vectors kept per concept,
196 by default, which is one image's worth of patches. Raising it improves detection substantially and
monotonically at the cost of memory - the point of the method is that it stays small, so raise it
knowingly.

**Every concept shuffles with the same permutation.** `_as_loader` builds a fresh generator seeded
from `config.seed` on each call, so concept 2 iterates its data in the same permutation sequence as
concept 1. It is deterministic and not wrong, but it is also not what "shuffle with seed s" usually
implies, and it is the place to change if you want concepts to see different orders.

## Noise and spread

A single run of this method is a weak measurement, and it is worth knowing that before reading
anything into a comparison.

The prompt's quality swings from epoch to epoch rather than converging: within one run the per-epoch
image AUROC of one category moves by as much as 0.42. What the model keeps is whatever the last epoch
happened to leave, one arbitrary point of that trajectory. Across seeds that shows up as roughly
+-0.03 on the twelve-category VisA average and +-0.07 on a single category such as candle. A change
smaller than that is not a change; measure over at least three seeds before believing one.

With `seed` fixed a configuration reproduces exactly - every draw a run makes, including the coreset's
random projection, comes from it. That is convenient and also the trap: a single seed reproduces
perfectly and still tells you very little.

## What to expect

A plain run reports one model, and lands around **0.70 image AUROC on VisA** against the paper's
0.874, measured over three seeds on the per-category folder copy of the dataset. VisA's official
`split_csv/1cls.csv` division puts five times as many normal images in the test set and reads several
points higher; if you compare against a published VisA number, check which split produced it.

That gap is not an implementation difference. The authors' code reports the mean of all 25 epochs'
rescaled scores, at the epoch whose image AUROC on the test set is highest; those two undescribed
mechanisms are worth +0.129 and +0.044 respectively, and put through the same machinery this
implementation reaches 0.8725 against the reference's 0.8723.

**A configuration with `training_epochs=0` scores as well or better.** On VisA the contrastive loss
costs about 0.006 image AUROC over 25 epochs and on MVTec it changes nothing; the apparent benefit in
the published ablation comes from the reporting protocol rather than from the loss. If you want the
method's detection quality without its training cost, set `training_epochs=0`: no SAM masks are needed
and a concept is fitted in one forward pass.

Forgetting is 0.0000 on VisA and MVTec: for every concept, every cell of the concept matrix recorded
after that concept was learned repeats its just-learned value bit for bit, however many concepts join
the memory afterwards. Nothing is shared, so the only thing that could move a learned concept's score
is the key sending its images to a different concept, and that does not happen - routing is correct on
every test image of both benchmarks.

**Read forgetting off `BackwardTransfer`, not off `ForgettingMeasure`.** pyCLAD's `ForgettingMeasure`
averages over `range(learned_task + 1)`, so it includes the concept just learned and compares it
against readings taken before that concept was in memory - readings that are near-random, because the
model had nothing to score it with. For a model with per-concept memory that term is large and
negative, and it drags the result below zero: UCAD reads -0.007 to -0.027 there while its actual
forgetting is exactly 0. The definition the UCAD paper uses (Eq. 7) averages only over concepts
learned before the last one, which is also what `BackwardTransfer` reports for this model.

## Known limitations

**The contrastive loss does not help on these benchmarks.** After one epoch the prompted features are
99.9% cosine-identical to the frozen ones, so the model is PatchCore on a frozen ViT. After 25 epochs
they have moved a long way in the wrong direction - mean pairwise cosine falls from 0.42 to 0.18 and
effective rank from 297 to 217 - which erodes the locality nearest-neighbour scoring depends on.

**Small defects are limited by the patch grid, not by the training.** ViT-B/16 at 224 gives 14x14
patches, one patch covering 73x73 pixels of a 1024x1024 original, while MVTec's screw has defects a
few pixels wide and scores near-random with timm's default weights. Two things help and neither is
training: `vit_model_name="vit_base_patch8_224"` gives 28x28 patches at the same input size, and
raising `knowledge_size` in step with the finer grid - a bank of 196 vectors compresses four times
harder at 28x28 and loses what the finer grid gained.

## Where the analysis lives

The evidence behind the two statements above - that the loss contributes nothing and that the
published numbers need an undescribed reporting protocol - is not in this repository. It is in a fork
of the authors' code, where each claim can be checked against the code that produced it:

- `scripts/FINDINGS.md` - how we know the contrastive loss contributes nothing
- `scripts/REPRODUCTION.md` - reproducing the published tables, and what it takes
- `scripts/EVALUATION.md` - the reporting protocol taken apart, and honest replacements
- `scripts/MEASUREMENTS.md` - backbone checkpoints, feature grid, bank size, and open questions

https://github.com/krzkapera/ucad/tree/claude/experiment-scripts/scripts
