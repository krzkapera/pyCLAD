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
run rather than quietly cost it several points.

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

`docs/ucad_paper_reference_pyclad.md` says where each default comes from.

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

The reference implementation reports numbers obtained under an evaluation protocol its paper does not
describe: it averages the scores of all 25 epochs and then reports the epoch whose test-set image
AUROC is highest. pyCLAD does neither, so a plain run lands **around 0.70 image AUROC on VisA** rather
than the paper's 0.874. `docs/ucad_paper_reference_pyclad.md` explains both mechanisms, what each is
worth, and has the per-category tables for both implementations under both protocols.

Forgetting is 0.0000 on VisA and MVTec - every post-learning cell of the concept matrix repeats bit
for bit and task routing is correct on every test image.

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

## See also

`docs/ucad_paper_reference_pyclad.md` - what the paper specifies, what the authors' code actually
does, where pyCLAD departs from either, and the measured results.
