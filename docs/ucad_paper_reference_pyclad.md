# UCAD: the paper, the authors' code, and pyCLAD

Three things carry the name UCAD and they are not the same: the method described in Liu et al.,
AAAI 2024; the implementation the authors released; and pyCLAD's. This records where they diverge and
what each divergence is worth, so that a number can always be traced to the thing that produced it.

Measurements come from three seeds per side on VisA (12 categories) and MVTec AD (15), at the
reference's own configuration. Where a figure has no error bar it comes from a single run and is
labelled as such.

## 1. The paper against the authors' code

Everything in this section is in the released code and absent from the paper.

### Ensembling the epochs

After every epoch the code scores the whole test set, rescales that epoch's scores to 0..1 across the
test set, and averages every epoch so far. The reported number is therefore the mean opinion of 25
models, not the score of one.

The paper excludes this by its own accounting: it states a memory of "a key of size (15, 196, 1024),
a prompt of size (15, 7, 768), and knowledge of size (15, 196, 1024), with an overall size of
approximately 23.28MB". The first dimension is the 15 tasks, so one key and one bank per task; 25
banks per task would be roughly 290MB and would contradict the compactness the paper argues for.

Worth **+0.129 image AUROC** on VisA (0.7045 +- 0.0353 for a single model against 0.8335 +- 0.0087
for the ensemble), and it cuts the seed spread fourfold.

### Selecting the epoch on the test set

```python
if (auroc > pr_auroc):
    memory_feature_list[dataloader_count] = memory_feature
    prompt_list[dataloader_count] = PatchCore.prompt_model.get_cur_prompt()
```

The epoch whose cumulative ensemble scores best **on the test set** is the one reported, and its state
is what enters the concept memory. There is no validation split. Worth **+0.0436 image AUROC** to the
reference on VisA and +0.0390 to pyCLAD when the same rule is applied to its runs.

The epochs it picks are unstable across seeds - candle 2/0/3, capsules 10/24/22, cashew 15/23/24 -
which is what fitting noise looks like. The metric it does not optimise gets slightly worse: pixel
AUPR falls from 0.3280 to 0.3269.

### Stopping a category early

```python
if (auroc == 1):
    break
```

Training stops the moment the test-set image AUROC reaches exactly 1.0. On MVTec this fires on five
categories - bottle and leather after one epoch, toothbrush after one or two, tile after one to
three, hazelnut after three to eight - so for those the reference has no leak-free reading at all:
the stopping point itself was chosen with test labels. No VisA category reaches 1.0.

### Squared distances

Neighbour distances are read out of `faiss.IndexFlatL2`, which returns **squared** L2, and no square
root is taken anywhere. The image score is the maximum over patches and squaring preserves order, so
image AUROC is untouched; the anomaly map is squared before it is upsampled and smoothed, and neither
operation commutes with squaring.

### The ground-truth mask

Masks are resized bilinearly and then truncated to int, so only pixels the interpolation left at full
weight count as positive.

### Reweighting

Equations 5-6 of the paper describe reweighting the image score by the neighbourhood of the nearest
bank vector. The scoring path in the released code does not apply it; the image score is the plain
maximum over patch scores.

### The prompt shape

The paper reports a prompt of size (15, 7, 768). The code constructs the model with
`prompt_length=1`, prefix tuning on all twelve layers, which is a different shape. We did not resolve
which the reported numbers correspond to.

### Prompts across tasks

The released code never re-initialises the prompt between concepts, so task *k* starts from wherever
task *k-1* ended - and from its last epoch, not from the epoch it selected.

### A trap in the repository

`args_dict.npy` in the released repository contains `length=5` and `batch_size=24`. Neither feeds the
model: the prompt is constructed with `prompt_length=1` in `patchcore/patchcore.py`, and the data
loader takes its batch size from a click default of 8 in the entry point. The paper independently
states batch size 8. Both values in that file misled this work at some point.

## 2. The authors' code against pyCLAD

A line-by-line trace found these equivalent: the input transform (`Resize(224)` + `CenterCrop(224)` +
ImageNet normalisation, PIL bilinear), the ViT-B/16 backbone, prefix tuning of shape
(12, 2, 1, 12, 64) initialised uniform(-1, 1) on all twelve layers, block 5 patch tokens as features,
`patchsize=1` followed by `adaptive_avg_pool1d(768 -> 1024)`, the contrastive loss term for term at
temperature 0.5, Adam at 5e-4 with a constant schedule and gradient clip 1.0, a greedy coreset over a
random 128-dimensional projection from 10 starting points, 1-NN L2 scoring with the image score as
the patch maximum, bilinear rescaling to 224 followed by a Gaussian at sigma 4, and the per-epoch
min-max normalisation before averaging.

What pyCLAD does differently, and on purpose:

| | the reference | pyCLAD | effect |
|---|---|---|---|
| patch score | squared distance, from faiss without a root | Euclidean distance | pixel AUPR only: +0.0087 VisA, +0.0140 MVTec in pyCLAD's favour |
| ground-truth mask | bilinear, then truncated to int | nearest-neighbour, every nonzero pixel counts | +0.0310 VisA, +0.0540 MVTec in pyCLAD's favour |
| epoch ensembling | always on | not in the library; `scripts/reference_ensemble.py` subclasses the model for reproduction | +0.129 image on VisA when enabled |
| epoch selection | reports the best epoch by test AUROC | never; the analysis scripts compute it only to report the reference's own protocol beside a leak-free one | +0.0436 to the reference |
| early stop at AUROC 1.0 | yes | no | five MVTec categories |
| prompt between tasks | carried over | reset per concept by default (`reset_prompt_per_task`) | not isolated |
| randomness | the coreset draws from the global RNG, so a seed does not fix a run | every draw comes from `UCADConfig.seed` | two runs of one configuration now agree exactly |

The squared-distance and mask conventions are metric conventions rather than model quality: they
change what is measured, not how well the model localises. The first was measured with a temporary
scorer variant which has since been removed; the numbers above are what it produced.

## 3. What this means for the published figures

| VisA, 12 categories, image AUROC | value |
|---|---|
| the method as the paper describes it, single model | 0.7045 +- 0.0353 |
| plus the epoch ensemble, no test labels | 0.8335 +- 0.0087 |
| plus selecting the epoch on the test set | 0.8725 +- 0.0046 |
| published | 0.874 |

The published figure needs both mechanisms, and neither is in the paper. The reference's own
leak-free average, 0.8287 +- 0.0042, is 0.045 below what it reports.

pyCLAD reproduces the reference once it scores the same way: image AUROC to 0.0002 on VisA and 0.005
on MVTec, pixel AUPR to 0.006 and 0.003 after both conventions are equalised. Per-category tables are
in `ucad_visa_reproduction.md`.

## 4. Not reproduced by either implementation

MVTec's screw. Both are near-random and the paper's 0.739 is out of reach; the reference's best
leak-assisted reading is 0.6195 +- 0.0431. It is not the training - at zero epochs, which is
PatchCore on a frozen ViT, screw already scores 0.5638, and the per-epoch trajectory wanders between
0.22 and 0.86 without a trend from the first epoch on. It is the feature grid: ViT-B/16 at 224 gives
14x14 patches, one patch covering 73x73 pixels of the 1024x1024 original, while screw's defects are a
few pixels wide. Swapping the backbone for `vit_base_patch8_224` - same input, 28x28 patches - moves
screw from 0.5638 to 0.6602 at zero epochs and from 0.6268 to 0.8660 at 25. The zero-epoch pair is
the clean comparison; the trained pair is one seed of a category whose single-model spread is +-0.2.

That is a limit of UCAD as specified, since the paper fixes ViT-B/16 at 224, not of either
implementation.
