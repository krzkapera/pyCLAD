# UCAD on VisA: what pyCLAD reproduces and what it does not

pyCLAD's UCAD reproduces the reference implementation on VisA at image level and exceeds it at pixel
level. This records the comparison, the configuration it holds at, and the two evaluation protocols
the numbers can be read under. Paper reference: Liu et al., AAAI 2024, Tables 3-6. Runs live in
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
enters the concept memory. Both are reported below as the **honest** and the **reference** protocol.
The ensembling itself is not leakage and pyCLAD implements it as `score_ensemble_epochs`.

## VisA, twelve categories, three seeds on each side

Both implementations at the reference's configuration, on our VisA copy with the reference's own SAM
ViT-B supervision. Image AUROC:

| category | pyCLAD honest | pyCLAD reference-protocol | reference honest | reference protocol | paper |
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

| category | pyCLAD honest | pyCLAD reference-protocol | reference honest | reference protocol | paper |
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
| image, honest protocol | +0.0048 +- 0.0056 (0.9 sd) |
| image, reference protocol | +0.0002 +- 0.0030 (0.1 sd) |
| pixel, honest protocol | +0.0476 +- 0.0020 |
| pixel, reference protocol | +0.0441 +- 0.0056 |

**Image AUROC is reproduced**: under the reference's own protocol the two implementations agree to
0.0002, and pyCLAD's 0.8725 matches the published 0.874. Forgetting is 0.0000 - every post-learning
cell of the concept matrix repeats bit for bit and task routing is correct on every test image -
against the paper's FM 0.039.

**Pixel AUPR does not agree** and pyCLAD is 0.048 ahead. Part of that is a metric convention rather
than model quality: pyCLAD resamples the ground-truth mask with nearest-neighbour and counts every
nonzero pixel, the reference resamples bilinearly and truncates to int, which on identical
predictions was worth about +0.022 AUPR. The ratio is largest exactly where anomalies are smallest -
macaroni1 0.0855 against 0.0231, macaroni2 0.0296 against 0.0127 - which is the signature of that
convention, since bilinear-then-truncate erases thin regions the nearest-neighbour path keeps.
Whether the remainder is the model has not been measured; it needs one run scored under both
conventions.

The reference gains +0.0436 image AUROC from epoch selection and pyCLAD +0.0390, so **the published
VisA figure rests substantially on the selection mechanism**: without it the reference averages
0.8287, which is 0.045 below its own published 0.874.

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

- The pixel-AUPR difference: how much of the 0.048 is the ground-truth mask convention and how much
  is the model. One run scored under both conventions settles it.
- The train/test split: our VisA copy holds every normal image with the first 20 per class held out
  for test, where the official `1cls.csv` splits normals about 90/10. Both implementations were
  compared on ours, so it does not affect the comparison, only the comparison to the paper.
- Which of these settings become pyCLAD's library defaults.
