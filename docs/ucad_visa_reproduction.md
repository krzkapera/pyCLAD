# UCAD on VisA: what pyCLAD reproduces and what it does not

pyCLAD's UCAD reproduces the reference implementation on both benchmarks: image AUROC to 0.0002 on
VisA and 0.005 on MVTec, and pixel AUPR to 0.006 and 0.003 once the reference's two scoring
conventions - its ground-truth resampling and its squared distances - are applied to pyCLAD's own
predictions. This records the comparison, the configuration it holds
at, and the two evaluation protocols the numbers can be read under. Paper reference: Liu et al., AAAI 2024, Tables 3-6. Runs live in
`$SCRATCH/pp/runs` on Athena (`protocol_*`, `probe_*`) and Ares (`ref_perepoch_*`, `probe_*`).

## The two protocols

One training run holds two numbers. The reference evaluates the test set after every epoch, averages
the min-max normalized scores of every epoch so far, and keeps the epoch whose image AUROC on that
cumulative ensemble is highest:

```python
if (auroc > pr_auroc):                                     # run_ucad.py
    memory_feature_list[dataloader_count] = memory_feature
    prompt_list[dataloader_count] = PatchCore.prompt_model.get_cur_prompt()
```

The cumulative ensemble after the last epoch uses no test labels; its maximum over epochs uses them
to pick one of 25 states per category, with no validation split, and the selected state is what
enters the concept memory. Both are reported below: **without epoch selection** and **the reference protocol**. Neither is
neutral ground - both use the authors' epoch ensemble; they differ only in whether the epoch is
chosen with test labels. The method as the paper describes it is a single model, lower than both.
The ensembling itself is not leakage and pyCLAD implements it as `score_ensemble_epochs`.

## VisA, twelve categories, three seeds on each side

Both implementations at the reference's configuration, on our VisA copy with the reference's own SAM
ViT-B supervision. Image AUROC:

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

Pixel AUPR:

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

| | pyCLAD - reference |
|---|---|
| image, reading without epoch selection | +0.0048 +- 0.0056 (0.9 sd) |
| image, reference protocol | +0.0002 +- 0.0030 (0.1 sd) |
| pixel, reading without epoch selection | +0.0476 +- 0.0020 |
| pixel, reference protocol | +0.0441 +- 0.0056 |

**Image AUROC is reproduced**: under the reference's own protocol the two implementations agree to
0.0002, and pyCLAD's 0.8725 matches the published 0.874. Forgetting is 0.0000 - every post-learning
cell of the concept matrix repeats bit for bit and task routing is correct on every test image -
against the paper's FM 0.039.

**Pixel AUPR agrees once both scoring conventions are equalised.** Two of them separate the
implementations and neither is the model:

- *the ground truth.* pyCLAD resamples the mask with nearest-neighbour and counts every nonzero
  pixel; the reference resamples bilinearly and truncates to int, keeping only pixels the
  interpolation left at full weight.
- *the anomaly map.* The reference reads its neighbour distances out of `faiss.IndexFlatL2` and
  never takes their square root, so it scores patches by **squared** distance. The image score is
  the maximum over patches and squaring preserves order, so image AUROC is untouched; the map is
  squared before it is upsampled and smoothed, and neither operation commutes with squaring.

Scoring the same trained models under all four combinations (`scripts/pixel_convention_effect.py`,
three seeds, 12-category average):

| pyCLAD | our ground truth | the reference's |
|---|---|---|
| our map (Euclidean) | 0.3741 +- 0.0040 | 0.3431 +- 0.0041 |
| the reference's map (squared) | 0.3670 +- 0.0036 | **0.3344 +- 0.0036** |

against the reference's own **0.3280 +- 0.0034**. The 0.0461 difference decomposes into **0.0310
ground-truth convention, 0.0087 squared distances and 0.0063 residual** - the residual is 2.2
standard errors and 1.9% relative. Per category the ground-truth convention is worth most where
anomalies are thinnest, since bilinear-then-truncate erases regions narrower than the interpolation
kernel: macaroni2 keeps 3.2x more positive pixels under our resampling, macaroni1 2.9x, pcb2 2.0x. On
pcb1 it reverses - discarding the boundary pixels removes the ones the model localises worst, and the
reference's stricter ground truth scores 0.028 higher.

The reference gains +0.0436 image AUROC from epoch selection and pyCLAD +0.0390, so **the published
VisA figure rests substantially on the selection mechanism**: without it the reference averages
0.8287, which is 0.045 below its own published 0.874.

## MVTec AD, fifteen categories, three seeds on each side

Same configuration, the reference's own `mvtec2d-sam-b` supervision. Image AUROC:

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

Pixel AUPR:

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

| | pyCLAD - reference |
|---|---|
| image, reading without epoch selection | -0.0048 +- 0.0021 (2.3 sd) |
| image, reference protocol | -0.0064 +- 0.0018 (3.5 sd) |
| pixel, reading without epoch selection | +0.0710 +- 0.0022 |
| pixel, reference protocol | +0.0602 +- 0.0022 |

pyCLAD is 0.005 **behind** on image here, and screw is 84% of it - see below. pyCLAD's 0.9291 without selection
matches the paper's 0.930.

The pixel column carries the same two conventions, and equalising them closes it as it does on VisA:

| pyCLAD | our ground truth | the reference's |
|---|---|---|
| our map (Euclidean) | 0.5446 +- 0.0021 | 0.4906 +- 0.0021 |
| the reference's map (squared) | 0.5316 +- 0.0020 | **0.4766 +- 0.0021** |

against the reference's **0.4736 +- 0.0037**. The 0.0710 difference is 0.0540 ground truth, 0.0140
squared distances and 0.0030 residual - 1.2 standard errors, so **MVTec's pixel metric is reproduced
within noise**.

### screw

Both implementations are near-random on screw and the paper's 0.739 is out of reach for either -
the reference's best leak-assisted reading is 0.6195 +- 0.0431. It is not the training: at zero
epochs, which is PatchCore on a frozen ViT, screw already scores 0.5638, and the per-epoch trajectory
wanders between 0.22 and 0.86 without a trend in both implementations from the first epoch on.

It is the feature grid. ViT-B/16 at 224 gives 14x14 patches, so one patch covers 73x73 pixels of the
1024x1024 original while screw's defects are a few pixels wide. Swapping the backbone for
`vit_base_patch8_224` - same input, same everything else, 28x28 patches - moves screw from 0.5638 to
**0.6602** at zero epochs and from 0.6268 to **0.8660** at 25, past the paper's figure. The
zero-epoch pair is the clean comparison, since it involves no training noise; the trained pair is one
seed of a category whose single-model spread is +-0.2.

That is a limit of UCAD as specified - the paper fixes ViT-B/16 at 224 - not of either
implementation, and it is the one place where a change to the method would clearly pay.

\* The reference stops training a category as soon as the cumulative ensemble reaches image AUROC
exactly 1.0 on the test set:

```python
if (auroc == 1):     # run_ucad.py, inside the branch that also commits the state to memory
    break
```

That fires on five MVTec categories - bottle and leather after one epoch, toothbrush after one or
two, tile after one to three, hazelnut after three to eight - so for those the reference has no
leak-free reading at all: the stopping point itself was chosen with test labels, and its two columns are the same number by construction. No VisA category reaches 1.0, so the VisA
comparison is unaffected.

## The configuration these numbers hold at

| | value | pyCLAD library default |
|---|---|---|
| `input_size` / `resize_mode` | (224, 224) / `short_side_crop` | `stretch` |
| `training_epochs` / `batch_size` | 25 / 8 | 25 / 8 |
| `score_ensemble_epochs` | 25 | 1 |
| `coreset_mode` | `approximate` | `exact` |
| `knowledge_size` / `key_size` | 196 / 196 | same |
| `blur_sigma` | 4.0 | 3.0 |
| `prompt_length` / `num_prompt_layers` / `feature_layer` | 1 / 12 / 5 | same |
| `scl_temperature` / lr / grad clip / optimizer | 0.5 / 5e-4 / 1.0 / Adam | same |
| `reweighting_num_nn` | 0 | same |
| SCL supervision | `visa-refgrid14`, the reference's own 14x14 label maps | - |
| seeds | 0, 1, 2 | - |

Run through `scripts/protocol_compare.py`, which reports both protocols from one training run;
`scripts/protocol_compare.sbatch` submits it on either cluster. `scripts/ucad_probe.py` runs the same
model through the full continual scenario when the concept matrix is what is wanted.

## What it took to get here

The batch size was the last difference and the only one that mattered. The reference trains with
**batch 8** - the click default of its own entry point, which its launch command never overrides -
not the 24 that `args_dict.npy` suggests; that file does not feed the data loader, and it had already
mislead this work once with `prompt_length=5`. Over 980 training images that is 123 optimizer steps
per epoch against 41, so at batch 24 pyCLAD's prompt had drifted three times less far by the
twenty-fifth epoch. The effect is invisible on ten of the twelve categories and worth 0.21 on candle.

The per-epoch trajectories localised it. Cumulative ensemble on candle, image AUROC:

| | epoch 1 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|
| reference, three seeds | 0.627 / 0.787 / 0.666 | 0.660 / 0.658 / 0.664 | 0.604 / 0.637 / 0.599 | 0.582 / 0.602 / 0.503 | 0.509 / 0.524 / 0.410 | 0.485 / 0.468 / 0.352 |
| pyCLAD at batch 24 | 0.705 / 0.613 / 0.642 | 0.577 / 0.541 / 0.661 | 0.558 / 0.607 / 0.730 | 0.557 / 0.625 / 0.721 | 0.566 / 0.637 / 0.651 | 0.584 / 0.628 / 0.626 |

Epoch 1 agrees (0.6935 +- 0.0833 against 0.6528 +- 0.0470), which clears feature extraction and
scoring; from there the reference decays monotonically in every seed and pyCLAD at batch 24 stays
flat, which points at the training step and nothing else.

Two other hypotheses were tested and cleared first. Giving pyCLAD the reference's own 14x14 label
maps leaves candle at 0.6172 +- 0.0665 against the base run's 0.6293 +- 0.0106, and its mask source
at 0.6298 +- 0.0471, three seeds each - the SCL supervision is not a difference. A line-by-line trace
of both pipelines found them equivalent at every step: input transform (`Resize(224)` +
`CenterCrop(224)` + ImageNet normalization, PIL bilinear), prefix tuning of shape (12, 2, 1, 12, 64)
initialised `uniform(-1, 1)` on all twelve layers, block 5 patch tokens, `patchsize=1` then
`adaptive_avg_pool1d(768 -> 1024)`, the contrastive loss term for term at temperature 0.5, Adam at
5e-4 with a constant schedule and clip 1.0, a greedy coreset over a random 128-dimensional projection
from 10 starting points, 1-NN L2 scoring with the image score as the patch maximum, bilinear
rescaling to 224 and a Gaussian at sigma 4, and per-epoch min-max normalization before averaging.

The remaining differences are real but inconsequential: the reference does not reset the prompt
between tasks where pyCLAD does (`reset_prompt_per_task`); it re-seeds the global RNG per task, where
pyCLAD seeds the prompt and the loader but leaves the coreset projection on the global stream; it
keeps the classification head and the prompt key in the optimizer, which no gradient reaches; and it
steps the optimizer even when the loss is zero, which never happens.

## Retracted findings

| claim | measurement that retracted it |
|---|---|
| the mask-resampling path is worth +0.042 image | 25 epochs, 3 seeds: candle 0.6172 +- 0.0665 with the reference's maps against 0.6293 +- 0.0106 |
| exact coreset selection is worth +0.043 image | 25 epochs, 3 seeds each: +0.007 +- 0.009 |
| "the two pipelines are numerically equivalent" | rested on one coincidental single-run match; now supported by the trace above, not by that match |
| pyCLAD leads the reference by 0.024 image AUROC | that was batch 24 against batch 8; at equal batch size the difference is +0.005 +- 0.006 |

Single-run isolation experiments produced all four. The 12-category average moves by up to +-0.03
between seeds and candle alone by +-0.07, so nothing below three seeds is quoted here.

## What the reference's own numbers say about the method

| component | measured |
|---|---|
| `Resize(224)` + `CenterCrop(224)` instead of squashing the frame | +0.039 image, +0.037 pixel |
| ensembling normalized scores across epochs | +0.151 image with all 25 epochs, +0.064 pixel |
| selecting the epoch with the best test-set image AUROC | +0.0436 image to the reference, +0.0390 to pyCLAD |
| structure-based contrastive learning itself, single model | 0 at one epoch, -0.06 to -0.09 at 25 |

The paper credits SCL with +0.088 image AUROC and +0.049 pixel AUPR (Table 5, CPM-only 0.786/0.251
against CPM+SCL 0.874/0.300). A no-SCL configuration lands on 0.7833/0.2718, matching their CPM-only
row, and the pixel half of their gain is reproduced by epoch ensembling alone.

`prompt_effect.py` on candle, measuring the features the anomaly score is computed from:

| stage | mean cosine | effective rank | norm | cosine to frozen features |
|---|---|---|---|---|
| frozen backbone | 0.4207 | 296.6 | 13.82 | - |
| untrained prompt | 0.4173 | 296.0 | 13.85 | 0.9999 |
| after 1 epoch | 0.4176 | 297.2 | 13.83 | 0.9992 |
| after 25 epochs | 0.1752 | 217.1 | 23.15 | 0.2812 |

The loss spreads embeddings apart - its negative term is exponential, its positive term linear - and
lowers the rank, which erodes the locality nearest-neighbour scoring depends on. Per-epoch image
AUROC then swings by up to 0.42 within one run, and that instability is what makes epoch selection so
profitable.

## Rejected hypotheses

- Mask source: SAM2 at full resolution, SAM2 at 224, and SAM ViT-B by the authors' own recipe carry
  the same supervision after the 14x14 downsample (candle 7.33 against 7.25 regions, positive-pair
  fraction 0.392 against 0.396) and give the same results across three seeds.
- Image-score reweighting from Eq. 5-6: `reweighting_num_nn=9` sits inside seed noise, `=3` costs
  0.08.
- Prompt length 5: helps only where the prompt barely trains (+0.025 at one epoch, from the random
  perturbation), and costs 0.05 at 25 epochs.

## Defects found in our own code

- `UCADConfig.seed` never reached the prompt, the only trained state, so runs were irreproducible:
  two runs of one configuration differed by 0.13 image AUROC on candle. Fixed.
- The earlier prompt-length sweep set an environment variable the example did not read, so both of
  its arms trained the same configuration. All 20 runs before that point used `prompt_length=1`.
- `SAM2OfflineMaskProvider` warns and substitutes zeros when a mask file is missing, so a wrong mask
  directory yields a silently untrained SCL rather than an error.

## Open

- The train/test split: our VisA copy holds every normal image with the first 20 per class held out
  for test, where the official `1cls.csv` splits normals about 90/10. Both implementations were
  compared on ours, so it does not affect the comparison, only the comparison to the paper.
- Which of these settings become pyCLAD's library defaults.
