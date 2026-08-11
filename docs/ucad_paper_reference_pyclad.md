# UCAD in pyCLAD: what you get, and how it differs from the paper and the authors' code

pyCLAD ships an implementation of UCAD (Liu et al., *Unsupervised Continual Anomaly Detection with
Contrastively-learned Prompt*, AAAI 2024). If you compare its output against the numbers in that
paper you will find them lower, and this explains why: the published figures rest on two evaluation
mechanisms that the paper does not describe and that pyCLAD does not apply by default. Neither is a
bug in either implementation - they are choices about how a run is scored, and you should know which
one you are looking at.

Read this before quoting a number.

## What pyCLAD's UCAD does

Per concept it prefix-tunes a prompt on a frozen ViT-B/16 for 25 epochs with the structure-based
contrastive loss, then stores one prompt and one coreset-reduced bank of normal-image features. At
test time it routes each image to a concept by nearest key, scores its patches by distance to that
concept's bank, and reports the maximum as the image score and the upsampled, smoothed field as the
anomaly map.

That is the method as the paper specifies it, and `UCADModel` implements nothing else. One prompt and
one bank per concept, matching the paper's own memory accounting: "a key of size (15, 196, 1024), a
prompt of size (15, 7, 768), and knowledge of size (15, 196, 1024), with an overall size of
approximately 23.28MB".

```python
from pyclad.vision.models.ucad import UCADConfig, UCADModel

model = UCADModel(UCADConfig(max_tasks=len(dataset.train_concepts()), input_size=(224, 224)))
```

Expect roughly **0.70 image AUROC on VisA**, not the 0.874 the paper reports. The gap is the two
mechanisms below.

## The authors' evaluation protocol

The released implementation does two things inside its training loop that change the reported number.

### It ensembles the epochs

After *every* epoch it scores the whole test set, rescales that epoch's scores to 0..1 across the
test set, and averages every epoch so far. The number it reports is therefore the mean opinion of 25
different models - each epoch's prompt with the bank extracted under it - not the score of the model
you would deploy.

This uses no labels, so it is legitimate as a scoring scheme; it is simply not the method the paper
describes, and the paper's memory figure rules it out. It is worth **+0.129 image AUROC** on VisA
(0.7045 +- 0.0353 for one model against 0.8335 +- 0.0087 for the ensemble over three seeds), and it
cuts the seed-to-seed spread fourfold, because a single prompt's quality swings wildly between
epochs.

If you want to reproduce the authors' numbers, `scripts/reference_ensemble.py` subclasses the model
to do this. It is deliberately not in the library.

### It picks the epoch by looking at the test labels

```python
if (auroc > pr_auroc):
    memory_feature_list[dataloader_count] = memory_feature
    prompt_list[dataloader_count] = PatchCore.prompt_model.get_cur_prompt()
```

Of the 25 cumulative ensembles - after epoch 1, after epoch 2, and so on - it reports the one whose
image AUROC on the **test set** is highest, and that state is what enters the concept memory. There
is no validation split; the choice is made on the data the result is then reported on.

This is a leak. It is worth **+0.0436 image AUROC** to the reference on VisA. Two symptoms show it is
fitting noise rather than finding a genuine stopping point: the epoch it picks is unstable across
seeds (candle 2/0/3, capsules 10/24/22, cashew 15/23/24), and the metric it does not optimise gets
slightly worse - pixel AUPR falls from 0.3280 to 0.3269.

pyCLAD never does this. The analysis scripts compute it only so that the reference's own protocol can
be reported beside a leak-free one.

### It stops a category the moment it scores perfectly

```python
if (auroc == 1):
    break
```

Training ends as soon as the test-set image AUROC reaches exactly 1.0. This is the same leak in a
sharper form: not only the reported epoch but the amount of training is chosen from the test labels.

On MVTec it fires on five of fifteen categories - bottle and leather after one epoch, toothbrush
after one or two, tile after one to three, hazelnut after three to eight. For those categories the
reference has no leak-free reading at all, and its "before selection" and "after selection" numbers
are the same by construction. No VisA category reaches 1.0, so VisA comparisons are unaffected.

### It is not evaluated on the official split

The authors' loader walks directories and cannot read VisA's `split_csv/1cls.csv`, so a run of the
released code is not evaluated on the benchmark's official train/test division. Our comparisons use
the same per-category folder copy the reference requires, which holds every normal image with the
first 20 per class held out for test, where the official split divides the normals roughly 90/10.
That does not affect the comparison between the two implementations - both see the same data - but it
does mean neither is directly comparable to a published number obtained on the official split.

### What the three protocols give

VisA, twelve categories, image AUROC, three seeds:

| | pyCLAD | the reference |
|---|---|---|
| one model, no ensembling - the paper's method | 0.7045 +- 0.0353 | - |
| ensemble of 25 epochs, no test labels | 0.8335 +- 0.0087 | 0.8287 +- 0.0042 |
| ensemble plus the epoch chosen on the test set | 0.8725 +- 0.0046 | 0.8723 +- 0.0026 |
| published | | 0.874 |

The published figure needs both mechanisms. The reference's own leak-free average is 0.045 below
what it reports.

## Where pyCLAD deliberately differs from the authors' code

A line-by-line comparison found the two equivalent in every computational step: the input transform,
the backbone, prefix tuning of the same shape and initialisation, the feature layer, patch
aggregation, the contrastive loss, the optimizer and schedule, the coreset, nearest-neighbour
scoring, and the map rescaling. What differs is deliberate:

| | the reference | pyCLAD |
|---|---|---|
| patch score | squared distance, read from `faiss.IndexFlatL2` with no square root taken | Euclidean distance |
| ground-truth mask | resized bilinearly, then truncated to int, so only pixels left at full weight count | resized with nearest-neighbour, every nonzero pixel counts |
| prompt between concepts | carried over from the previous concept | reset per concept (`reset_prompt_per_task`) |
| randomness | the coreset draws from the global RNG, so a seed does not fix a run | every draw comes from `UCADConfig.seed`; two runs of one configuration agree exactly |
| epoch ensembling and selection | always | never in the library |

The first two are metric conventions, not model quality: they change what is measured. The squared
distance leaves image AUROC untouched, since the image score is the maximum over patches and squaring
preserves order, but it changes the anomaly map, which is squared before being upsampled and
smoothed. Together the two are worth 0.040 of pixel AUPR on VisA and 0.068 on MVTec in pyCLAD's
favour; equalise them and the two implementations agree to 0.006 and 0.003 respectively.

## Things worth knowing before you tune it

The paper credits SCL with +0.088 image AUROC and +0.049 pixel AUPR (Table 5, CPM-only 0.786/0.251
against CPM+SCL 0.874/0.300). A no-SCL configuration lands on 0.7833/0.2718, matching their CPM-only
row, and the pixel half of their gain is reproduced by epoch ensembling alone.

**The contrastive loss does not help on these benchmarks.** After one epoch the prompted features are
99.9% cosine-identical to the frozen ones, so the model is PatchCore on a frozen ViT. After 25 epochs
they have moved a long way in the wrong direction - mean pairwise cosine falls from 0.42 to 0.18 and
effective rank from 297 to 217 - which erodes the locality nearest-neighbour scoring depends on.
Per-epoch image AUROC then swings by up to 0.42 within one run, and that instability is exactly what
the epoch ensemble smooths over and the epoch selection exploits.

**MVTec's screw is out of reach for the method as specified.** Both implementations are near-random
on it and neither approaches the paper's 0.739. It is not the training: with zero epochs, which is
PatchCore on a frozen backbone, screw already scores 0.5638. It is the feature grid - ViT-B/16 at 224
gives 14x14 patches, one patch covering 73x73 pixels of the 1024x1024 original, while screw's defects
are a few pixels wide. Swapping the backbone for `vit_base_patch8_224`, which gives 28x28 patches at
the same input size, moves it to 0.6602 at zero epochs and 0.8660 after 25.

**`args_dict.npy` in the authors' repository is misleading.** It contains `length=5` and
`batch_size=24`; neither reaches the model. The prompt is built with `prompt_length=1` in
`patchcore/patchcore.py`, and the batch size comes from a click default of 8, which is also what the
paper states.

**The paper's prompt shape does not match the released code.** The paper reports a prompt of size
(15, 7, 768); the code builds prefix tuning with `prompt_length=1` on twelve layers. We could not
determine which the published numbers correspond to.

## How these differences were established

Nothing here rests on a single run. The 12-category VisA average moves by up to +-0.03 between seeds
and candle alone by +-0.07, so every difference quoted above is three seeds per side, and four
earlier claims made from single runs were withdrawn when they did not survive replication:

| withdrawn claim | what retracted it |
|---|---|
| the mask-resampling path is worth +0.042 image | 25 epochs, 3 seeds: candle 0.6172 +- 0.0665 with the reference's maps against 0.6293 +- 0.0106 |
| the exact coreset is worth +0.043 image | 25 epochs, 3 seeds each: +0.007 +- 0.009 |
| the two pipelines are numerically equivalent | rested on one coincidental match; the line-by-line trace above replaced it |
| pyCLAD leads the reference by 0.024 image AUROC | that compared batch 24 against batch 8; at equal batch size it is +0.005 +- 0.006 |

The last one was the hard one to find, and it is worth recording how. At batch 24 pyCLAD scored 0.21
higher than the reference on candle and matched it on ten of twelve categories. Logging the
cumulative ensemble after every epoch localised it: epoch 1 agreed (0.6935 +- 0.0833 against 0.6528
+- 0.0470), which cleared feature extraction and scoring, and from there the reference decayed
monotonically in every seed while pyCLAD stayed flat. That pattern points at the training step, and
the training step differed only in how many optimizer steps an epoch took - 123 against 41.

Two candidate differences were tested and cleared before the batch size was found: giving pyCLAD the
reference's own 14x14 label maps left candle at 0.6172 +- 0.0665 against 0.6293 +- 0.0106, and its
mask source at 0.6298 +- 0.0471. SAM2 at full resolution, SAM2 at 224 and SAM ViT-B by the authors'
recipe all carry the same supervision after the 14x14 downsample - candle 7.33 against 7.25 regions,
positive-pair fraction 0.392 against 0.396.

Two further hypotheses about the method were rejected: the image-score reweighting of Eq. 5-6 sits
inside seed noise at `reweighting_num_nn=9` and costs 0.08 at 3, and a prompt length of 5 helps only
where the prompt barely trains, costing 0.05 at 25 epochs.

One defect on pyCLAD's side made the early numbers noisier than they should have been: `seed` reached
the prompt but not the coreset's random projection, so two processes with one seed picked different
memory vectors. Fixed; a configuration now reproduces exactly.

## Reproducing the tables

`docs/ucad.md` has the per-category results for both implementations under both protocols on VisA
and MVTec, and everything a user of the model needs. The scripts that produce them are in `scripts/`:
`protocol_compare.py` reports both protocols from one training run, `reference_ensemble.py` holds
the ensemble, and `pixel_convention_effect.py` scores one run against both ground-truth conventions.
