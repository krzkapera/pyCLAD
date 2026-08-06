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
| reference | 0.8723 +- 0.0026 | 0.3269 +- 0.0042 | 196 | ensemble over 25 epochs + epoch chosen on the test set |
| paper | 0.874 | 0.300 | 196 | as above |
| **pyCLAD, reference-identical configuration** | **0.8528 +- 0.005** | **0.3756 +- 0.002** | **196** | **ensemble over 25 epochs, no test labels used** |
| pyCLAD before the fixes | 0.7189 | 0.2295 | 196 | one model, last epoch |

Forgetting stays at 0.0000. The pyCLAD row above was run at batch 24, which turned out not to be the
reference's batch size; the corrected comparison is two sections down.

An earlier estimate put that selection at about +0.106 image AUROC, measured against an average
epoch. Measured the way the reference actually uses it - against the same run's cumulative ensemble -
it is worth +0.0407 to the reference and +0.0105 to pyCLAD.

MVTec, same footing (196 vectors, no test labels): 0.9325 / 0.5187 with a five-epoch ensemble against
the reference's 0.9375 / 0.4715 and the paper's 0.930 / 0.456; the regression baseline without any of
the changes gives 0.9185 / 0.4727, matching earlier sessions. A 25-epoch ensemble is running.

## Both implementations under both protocols

One training run holds two numbers: the cumulative ensemble after the last epoch, which uses no test
labels, and its maximum over epochs, which the reference reports. Measured for both implementations
at what was believed to be the reference's own configuration - crop, **batch 24**, 25 epochs,
approximate coreset, sigma 4, bank 196 - VisA, 12-category average, image AUROC / pixel AUPR. The
next section shows the batch size was wrong, so this table compares two different configurations and
is kept only because the protocol arithmetic in it still holds:

| | pyCLAD | reference | paper |
|---|---|---|---|
| honest protocol, no test labels | **0.8528 +- 0.0047 / 0.3756 +- 0.0017** | 0.8287 +- 0.0042 / 0.3280 +- 0.0034 | - |
| reference protocol, epoch chosen on the test set | 0.8610 / 0.3690 | **0.8723 +- 0.0026** / 0.3269 +- 0.0042 | 0.874 / 0.300 |
| what the selection is worth | +0.0105 image | +0.0436 image | - |

Three seeds per column, except pyCLAD's selected row which is one. The 0.0241 image-AUROC difference
here is a batch-size difference and not an implementation one, so it says nothing about the two
implementations; what does survive is the protocol arithmetic. The reference's honest average,
0.8287, is 0.045 below the published 0.874, and selection is worth +0.0436 to it against +0.0105 to
pyCLAD, so **the paper's VisA figure rests substantially on the selection mechanism rather than on the
method** - and that conclusion is internal to the reference's own runs.

Roughly half of the pixel advantage is a metric convention rather than model quality. pyCLAD
resamples the ground-truth mask with nearest-neighbour and counts every nonzero pixel; the reference
resamples bilinearly and truncates to int. Ours yields 1.1-1.5x more positive pixels and about +0.022
AUPR on the identical predictions. The remaining ~0.026 is the model.

## The difference was the batch size

The comparison above is not what it says it is. The reference's training batch size is the click
default in its own entry point:

```python
@click.option("--batch_size", default=8, type=int, show_default=True)   # run_ucad.py:670
```

and its launch command never overrides it, so it trains with **batch 8**, not the 24 taken here from
`args_dict.npy` - the same file that earlier suggested `prompt_length=5` and does not feed the data
loader either. Over 980 training images that is 123 optimizer steps per epoch against 41: the
reference's prompt drifts three times faster per epoch.

The per-epoch trajectories show it directly. Cumulative ensemble on candle, image AUROC:

| | epoch 1 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|
| reference, three seeds | 0.627 / 0.787 / 0.666 | 0.660 / 0.658 / 0.664 | 0.604 / 0.637 / 0.599 | 0.582 / 0.602 / 0.503 | 0.509 / 0.524 / 0.410 | 0.485 / 0.468 / 0.352 |
| pyCLAD at batch 24 | 0.705 / 0.613 / 0.642 | 0.577 / 0.541 / 0.661 | 0.558 / 0.607 / 0.730 | 0.557 / 0.625 / 0.721 | 0.566 / 0.637 / 0.651 | 0.584 / 0.628 / 0.626 |

The reference decays monotonically in all three seeds; pyCLAD at batch 24 is flat. Epoch 1 agrees
(0.6935 +- 0.0833 against 0.6528 +- 0.0470), which puts feature extraction and scoring beyond
suspicion and locates the divergence in the training step alone.

At batch 8, with the reference's own supervision, both problem categories reproduce:

| VisA category, honest protocol | pyCLAD batch 24 | pyCLAD batch 8 | reference | difference |
|---|---|---|---|---|
| candle | 0.6293 +- 0.0106 | 0.4870 +- 0.0390 | 0.4352 +- 0.0721 | +0.052 +- 0.047 (1.1 sd) |
| macaroni1 | 0.7395 +- 0.0078 | 0.6978 +- 0.0471 | 0.6633 +- 0.0316 | +0.035 +- 0.033 (1.1 sd) |

Three seeds each. The 0.209 and 0.058 gaps of the previous section were a batch-size difference, not
an implementation difference; the full twelve-category comparison at batch 8 is running.

## Where the image-AUROC difference was (batch 24, superseded)

It is not spread across the benchmark. Per category, honest protocol, three seeds on each side:

| category | pyCLAD | reference | delta | share of the gap |
|---|---|---|---|---|
| candle | 0.6442 +- 0.0412 | 0.4352 +- 0.0721 | **+0.2090** | 72% |
| macaroni1 | 0.7213 +- 0.0251 | 0.6633 +- 0.0316 | **+0.0580** | 20% |
| pcb3 | 0.7768 +- 0.0119 | 0.7545 +- 0.0105 | +0.0223 | 8% |
| fryum | 0.9207 +- 0.0112 | 0.9427 +- 0.0095 | -0.0220 | -8% |
| pcb4 | 0.9373 +- 0.0153 | 0.9172 +- 0.0103 | +0.0201 | 7% |
| capsules | 0.8423 +- 0.0349 | 0.8623 +- 0.0210 | -0.0200 | -7% |
| cashew | 0.9610 +- 0.0036 | 0.9477 +- 0.0060 | +0.0133 | 5% |
| pcb1 | 0.9695 +- 0.0134 | 0.9625 +- 0.0226 | +0.0070 | 2% |
| macaroni2 | 0.5785 +- 0.0116 | 0.5837 +- 0.0743 | -0.0052 | -2% |
| pcb2 | 0.9480 +- 0.0071 | 0.9432 +- 0.0199 | +0.0048 | 2% |
| pipe_fryum | 0.9877 +- 0.0081 | 0.9857 +- 0.0042 | +0.0020 | 1% |
| chewinggum | 0.9462 +- 0.0040 | 0.9470 +- 0.0015 | -0.0008 | 0% |

Over the ten categories other than candle and macaroni1 the two implementations differ by +0.0022 on
average - they agree. The whole difference is the two categories where prompt tuning diverges, and
there pyCLAD took three times fewer optimizer steps per epoch, so it had drifted less far by the
twenty-fifth.

That also explains why the ordering flipped under the reference's protocol. Epoch selection is a
repair mechanism for exactly this divergence - on candle it lifts the reference from 0.4352 to 0.7325
- and the run that has drifted further has more to recover.

Two mechanisms were tested and cleared before the batch size was found. Giving pyCLAD the reference's
own 14x14 label maps leaves candle at 0.6172 +- 0.0665 against the base run's 0.6293 +- 0.0106, and
its mask source at 0.6298 +- 0.0471, three seeds each: the SCL supervision is not what separates the
implementations. A line-by-line trace of both pipelines - input transform, prefix tuning, feature
layer, patch aggregation, loss, optimizer, coreset, scorer, map rescaling, ensembling - found them
equivalent at every step.

## The configuration these numbers come from

Recorded so the better result can be reproduced before anything is changed on purpose. Everything in
the first block is set to the reference's value; the second block is where pyCLAD's library defaults
still differ, deliberately, because the reproduction is not closed yet.

| | value | pyCLAD default |
|---|---|---|
| `input_size` / `resize_mode` | (224, 224) / `short_side_crop` | `stretch` |
| `training_epochs` / `batch_size` | 25 / 24 | 25 / 8 |
| `score_ensemble_epochs` | 25 | 1 |
| `coreset_mode` | `approximate` | `exact` |
| `knowledge_size` / `key_size` | 196 / 196 | same |
| `blur_sigma` | 4.0 | 3.0 |
| `prompt_length` / `num_prompt_layers` / `feature_layer` | 1 / 12 / 5 | same |
| `scl_temperature` / lr / grad clip / optimizer | 0.5 / 5e-4 / 1.0 / Adam | same |
| `reweighting_num_nn` | 0 | same |
| SCL masks | `visa-sam2`, SAM2 hiera-small at full resolution | - |
| seeds | 0, 1, 2 | - |

Run as `scripts/ucad_probe.py` (single number per concept) or `scripts/protocol_compare.py` (both
protocols from one run); `scripts/protocol_compare.sbatch` submits either cluster.

## What turned out to be inside the noise

Three effects reported earlier as real did not survive replication across seeds, and are retracted:

| claim | measurement that retracted it |
|---|---|
| the mask-resampling path is worth +0.042 image | 1 epoch, 3 seeds: our masks 0.7918 +- 0.016, reference 14x14 masks 0.8023 +- 0.016, i.e. +0.010 +- 0.016 |
| exact coreset selection is worth +0.043 image | 25 epochs, ens25, 3 seeds each: exact 0.8598 +- 0.008 against approximate 0.8528 +- 0.005, i.e. +0.007 +- 0.009 |
| "the two pipelines are numerically equivalent" | that rested on one coincidental single-run match |

The single-epoch isolation runs used to attribute these effects were underpowered: the 12-class
average moves by +-0.016 between seeds at one epoch, which is larger than any of the differences they
were meant to resolve. Only paired within-run comparisons (the selection gain above) and
multi-seed means with their spread are quoted as findings.

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
- The reference's `approx_greedy_coreset`: a single-run pair put our exact selection 0.012 ahead, but
  three seeds each at the reference configuration reduce that to +0.007 +- 0.009 - inside the noise.
- Prompt length 5: helps only where the prompt barely trains (+0.025 at one epoch, from the random
  perturbation), and costs 0.05 at 25 epochs.

## Defects found in our own code

- `UCADConfig.seed` never reached the prompt, the only trained state, so runs were irreproducible:
  two runs of one configuration differed by 0.13 image AUROC on candle. Fixed.
- The earlier prompt-length sweep set an environment variable the example did not read, so both of
  its arms trained the same configuration. All 20 runs before this session used `prompt_length=1`.
- `SAM2OfflineMaskProvider` warns and substitutes zeros when a mask file is missing, so a wrong mask
  directory yields a silently untrained SCL rather than an error.
