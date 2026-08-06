# UCAD on VisA: what pyCLAD reproduces and what it does not

All numbers are the 12-category average after the last concept, VisA, image AUROC / pixel AUPR.
Paper reference: Liu et al., AAAI 2024, Tables 3-6. Runs live in `$SCRATCH/pp/runs` on Ares
(`probe_*`, `ref_visa_*`, `effect_*`, `selection-*`) and Athena (`visa_*`).

## The reference implementation on our data

Running the authors' own code over our VisA copy with our SAM2-derived masks (job 20847574 on Ares,
3h17m, knowledge bank 196) reproduces the paper: **0.8698 image AUROC / 0.3273 pixel AUPR** against
their 0.874 / 0.300. Per class it exceeds the paper on pcb1 (+0.077), pcb2 (+0.071) and pcb4 (+0.039)
and falls short on macaroni2 (-0.113) and candle (-0.066), averaging out. Our data pipeline - the
980/20 split, SAM2 masks instead of their unreleased SAM ViT-B ones - is therefore adequate for
reproduction; what differs is the evaluation protocol.

| | image AUROC | pixel AUPR | knowledge | protocol |
|---|---|---|---|---|
| reference | 0.8698 | 0.3273 | 196 | ensemble over 25 epochs + epoch chosen on the test set |
| paper | 0.874 | 0.300 | 196 | as above |
| **pyCLAD, reference-identical configuration** | **0.8528 +- 0.005** | **0.3756 +- 0.002** | **196** | **ensemble over 25 epochs, no test labels used** |
| pyCLAD before the fixes | 0.7189 | 0.2295 | 196 | one model, last epoch |

**VisA is reproduced**, three seeds at the reference's own configuration, and forgetting stays at
0.0000. The comparison that settles what the remaining difference is appears in the next section: the
reference's number and pyCLAD's are separated by its epoch selection, not by the method.

An earlier estimate put that selection at about +0.106 image AUROC, measured against an average
epoch. Measured the way the reference actually uses it - against the same run's cumulative ensemble -
it is worth +0.0407 to the reference and +0.0105 to pyCLAD.

MVTec, same footing (196 vectors, no test labels): 0.9325 / 0.5187 with a five-epoch ensemble against
the reference's 0.9375 / 0.4715 and the paper's 0.930 / 0.456; the regression baseline without any of
the changes gives 0.9185 / 0.4727, matching earlier sessions. A 25-epoch ensemble is running.

## Both implementations under both protocols

One training run holds two numbers: the cumulative ensemble after the last epoch, which uses no test
labels, and its maximum over epochs, which the reference reports. Measured for both implementations
at the reference's own configuration (crop, batch 24, 25 epochs, approximate coreset, sigma 4, bank
196), VisA, 12-category average, image AUROC / pixel AUPR:

| | pyCLAD | reference | paper |
|---|---|---|---|
| honest protocol, no test labels | **0.8505 / 0.3769** | 0.8291 / 0.3319 | - |
| reference protocol, epoch chosen on the test set | 0.8610 / 0.3690 | 0.8698 / 0.3273 | 0.874 / 0.300 |
| what the selection is worth | +0.0105 image | +0.0407 image | - |

**With the leakage removed pyCLAD is ahead of the reference on both metrics**, by 0.021 image AUROC
and 0.045 pixel AUPR. Under the reference's own protocol it stays 0.009 behind on image and 0.042
ahead on pixel. The reference gains four times more from epoch selection because its trajectory is
the less stable one - on candle it scores 0.4850 honestly and 0.7125 after selection, a jump of
0.228. Its honest average, 0.8291, is 0.045 below the published 0.874, so the paper's VisA figure
rests substantially on the selection mechanism rather than on the method.

## Where we stand

| configuration | image AUROC | pixel AUPR |
|---|---|---|
| 25 epochs, crop, 25-epoch ensemble, batch 24 (bank 196) | **0.8701** | **0.3623** |
| 25 epochs, two seeds | 0.686 / 0.719 | 0.217 / 0.230 |
| 1 epoch, three seeds | 0.778 +- 0.006 | 0.278 |
| 0 epochs (no SCL at all) | 0.7833 | 0.2718 |
| 0 epochs + reference preprocessing | 0.8223 | 0.3085 |
| knowledge bank 8x | 0.8569 | 0.3453 |
| reference preprocessing + knowledge 4x | **0.8680** | **0.3590** |
| paper | 0.874 | 0.300 |

Forgetting is better than the paper's throughout: BackwardTransfer is exactly 0.0, every
post-learning cell of the concept matrix repeats bit for bit, and task routing picks the right
concept for all 1725 MVTec and all VisA test images. The paper reports FM 0.039 / 0.015.

## What the gap is made of

| component | measured | reproducible without test labels |
|---|---|---|
| Reference preprocessing (`Resize(224)` + `CenterCrop(224)` instead of squashing the frame) | +0.039 image, +0.037 pixel | yes - implemented as `resize_mode="short_side_crop"` |
| Ensembling normalized scores across epochs | +0.010 image with 5 of 25 epochs, +0.151 image with all 25; +0.064 pixel | yes - implemented as `score_ensemble_epochs` |
| Reference batch size 24 | +0.023 image on VisA, -0.003 on MVTec - not general, not adopted as a default | yes |
| Selecting the epoch with the best test-set image AUROC | +0.106 image over an average epoch, but not needed for the published average | **no** |
| Structure-based contrastive learning itself, single model | 0 at one epoch, -0.06 to -0.09 at 25 | - |

The paper credits SCL with +0.088 image AUROC and +0.049 pixel AUPR (Table 5, CPM-only 0.786/0.251
against CPM+SCL 0.874/0.300). Our no-SCL configuration lands on 0.7833/0.2718, matching their
CPM-only row. The pixel-level half of their gain is reproduced, with margin, by epoch ensembling
alone. The image-level half is smaller than what epoch selection is worth on our own runs.

## Why SCL does not help here

`prompt_effect.py` on candle, measuring the features the anomaly score is computed from:

| stage | mean cosine | effective rank | norm | cosine to frozen features |
|---|---|---|---|---|
| frozen backbone | 0.4207 | 296.6 | 13.82 | - |
| untrained prompt | 0.4173 | 296.0 | 13.85 | 0.9999 |
| after 1 epoch | 0.4176 | 297.2 | 13.83 | 0.9992 |
| after 25 epochs | 0.1752 | 217.1 | 23.15 | 0.2812 |
| after 25 epochs, prompt_length 5 | 0.1529 | 181.0 | 28.13 | 0.2232 |

One epoch leaves the features untouched, so our best single-model configuration is PatchCore on a
frozen ViT. Twenty-five epochs rewrite them in the opposite direction to the paper's claim: the loss
spreads embeddings apart (its negative term is exponential, its positive term linear) and lowers the
rank, which destroys the locality nearest-neighbour scoring depends on. Per-epoch image AUROC then
swings by up to 0.42 within one run - the instability that makes epoch selection so profitable.

## Implementation comparison

Everything readable in the reference agrees with ours: the contrastive loss term for term, layer 5 as
the feature source, prefix tuning of the same shape, `prompt_length=1` (the model is constructed with
1; the `length=5` in `args_dict.npy` belongs to another entry point), a frozen backbone with only the
prompt trainable, scoring from the prompted ViT, lr 5e-4, temperature 0.5, gradient clip 1.0, Adam.

Differences that remain: batch size 24 against our 8; the two evaluation mechanisms above; and the
train/test split - our VisA copy holds every normal image with the first 20 per class held out for
test, where the official `1cls.csv` splits normals about 90/10.

## Rejected hypotheses

- Mask source: SAM2 at full resolution, SAM2 at 224, and SAM ViT-B by the authors' own recipe give
  the same supervision after the 14x14 downsample (0.58 positive pairs, one region over ~70% of the
  grid; the 224 variant is bit-identical to full resolution through pyCLAD's nearest-neighbour path).
- Image-score reweighting from Eq. 5-6: `reweighting_num_nn=9` sits inside seed noise, `=3` costs
  0.08.
- The reference's `approx_greedy_coreset`: worse than our exact selection (0.774 against 0.786
  image, 0.243 against 0.279 pixel).
- Prompt length 5: helps only where the prompt barely trains (+0.025 at one epoch, from the random
  perturbation), and costs 0.05 at 25 epochs.

## Defects found in our own code

- `UCADConfig.seed` never reached the prompt, the only trained state, so runs were irreproducible:
  two runs of one configuration differed by 0.13 image AUROC on candle. Fixed.
- The earlier prompt-length sweep set an environment variable the example did not read, so both of
  its arms trained the same configuration. All 20 runs before this session used `prompt_length=1`.
- `SAM2OfflineMaskProvider` warns and substitutes zeros when a mask file is missing, so a wrong mask
  directory yields a silently untrained SCL rather than an error.
