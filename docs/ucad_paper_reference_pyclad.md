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

## How each of the three evaluates

Three different things get called "the result" here. Continual learning is why there is room for the
confusion: the model is frozen once per concept, so every decision about *when* to freeze it is taken
twelve times on VisA and fifteen on MVTec, independently, and each of those decisions can be made
with the test set in view.

**The paper.** Per concept, prefix-tune the prompt for 25 epochs, then store one key, one prompt and
one coreset-reduced bank. At test time route each image to a concept by nearest key and score its
patches against that concept's bank. Report image AUROC and pixel AUPR averaged over concepts, plus a
forgetting measure. No ensembling and no epoch selection appear anywhere, and the memory accounting -
"a prompt of size (15, 7, 768)" for fifteen concepts - leaves room for one prompt per concept, not
twenty-five.

**The authors' code.** After each of the 25 epochs it scores the concept's whole test set, rescales
those scores to 0..1, and averages every epoch so far. It then keeps the epoch whose cumulative
average has the highest image AUROC *on that test set*, and abandons the concept early if that AUROC
reaches exactly 1.0. The kept state is what enters the concept memory. So both the reported number
and the stored model depend on test labels, once per concept.

**pyCLAD.** `fit` trains 25 epochs and keeps the state the last one left. `predict` scores with that
single state. The numbers come from the framework: `ConceptMetricCallback` and
`VisionPixelConceptMetricCallback` fill the concept matrix, `BackwardTransfer` reads forgetting off
it. No label is touched until a callback computes a metric.

### Which reading is which

| reading | ensembles epochs | test labels choose the state | who does it | VisA image AUROC |
|---|---|---|---|---|
| single model, last epoch | no | no | pyCLAD | 0.7045 |
| 25-epoch ensemble, last epoch | yes | no | neither by default; `scripts/reference_ensemble.py` | 0.8335 |
| 25-epoch ensemble, best epoch | yes | yes | the authors' code | 0.8725 |

Only the first is what `UCADModel.predict` returns, and it is the number to quote for the method as
the paper describes it. The second is the authors' machinery with the leak taken out; it exists so
that the reference has something to be compared against that does not read test labels. The tables
under "Measured results" report the second and third, not the first.

Forgetting is 0.0000 under all three readings, on both benchmarks. Nothing is shared between
concepts, so the continual metrics cannot tell these protocols apart - the whole difference lives in
how a single concept's result is read.

## The authors' evaluation protocol

The released implementation does two things inside its training loop that change the reported number.

### It ensembles the epochs

After *every* epoch it scores the whole test set, rescales that epoch's scores to 0..1 across the
test set, and averages every epoch so far. The number it reports is therefore the mean opinion of 25
different models - each epoch's prompt with the bank extracted under it - not the score of the model
you would deploy.

This uses no labels, so it is legitimate as a scoring scheme; it is simply not the method the paper
describes, and the paper's memory figure rules it out. It is worth **+0.129 image AUROC** on VisA
(0.7045 for one model against 0.8335 for the ensemble), because a single prompt's quality swings
wildly between epochs.

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
fitting noise rather than finding a genuine stopping point: the epoch it picks is unstable between
runs (candle 2/0/3, capsules 10/24/22, cashew 15/23/24), and the metric it does not optimise gets
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

VisA, twelve categories, image AUROC:

| | pyCLAD | the reference |
|---|---|---|
| one model, no ensembling - the paper's method | 0.7045 | - |
| ensemble of 25 epochs, no test labels | 0.8335 | 0.8287 |
| ensemble plus the epoch chosen on the test set | 0.8725 | 0.8723 |
| published | | 0.874 |

The published figure needs both mechanisms. The reference's own leak-free average is 0.045 below
what it reports.

## Forgetting, and why the paper's is not zero

The paper's Eq. 7 is the usual forgetting measure: `T_{l,j}` is the score on concept `j` after
concept `l` was learned, the inner `max` over `l in {1..k-1}` is the best that concept `j` ever read
during the stream, `T_{k,j}` is its reading after everything has been learned, and the average runs
over `j = 1..k-1`, every concept except the last. `k` is the number of concepts in the stream -
twelve on VisA, fifteen on MVTec - and the paper evaluates with `k` at its maximum. Positive means a
concept got worse than it once was.

pyCLAD reads 0.0000 against the paper's 0.039 on MVTec and 0.015 on VisA, and the zero is not an
artifact of averaging. Every cell of the concept matrix recorded after a concept was learned repeats
that concept's just-learned value bit for bit: `max |M[l][e] - M[e][e]|` over all concepts and all
later learning steps is exactly 0 on both benchmarks, trained and untrained.

There is only one mechanism that could move it. Nothing is shared between concepts, so a learned
concept's score can change only if the key starts routing its images somewhere else once the memory
holds more candidates. It does not: `scripts/routing_accuracy.py` puts every test image through the
full memory and gets 1440/1440 on VisA and 1725/1725 on MVTec, at zero epochs and after 25.

A nonzero forgetting measure for this architecture therefore says the reference's routing errs as
concepts accumulate. We have not measured its routing, and the released code makes it easy to miss:
the per-epoch evaluation inside the training loop scores against the current concept's own bank and
never routes at all, so routing only enters in the final pass over all concepts, which is also the
pass the reported forgetting comes from.

One caveat when quoting pyCLAD's own numbers: `ForgettingMeasure` averages over
`range(learned_task + 1)`, so it includes the concept just learned and compares it against readings
taken before that concept was in memory. For per-concept memory those readings are near-random, the
term is large and negative, and the metric returns -0.007 to -0.027 where the paper's definition
returns 0. `BackwardTransfer` agrees with Eq. 7 here.

## Where pyCLAD deliberately differs from the authors' code

A line-by-line comparison found the two equivalent in every computational step: the input transform,
the backbone, prefix tuning of the same shape and initialisation, the feature layer, patch
aggregation, the contrastive loss, the optimizer and schedule, the coreset, nearest-neighbour
scoring, and the map rescaling. What differs is deliberate:

| | the reference | pyCLAD |
|---|---|---|
| patch score | squared distance, read from `faiss.IndexFlatL2` with no square root taken | Euclidean distance |
| ground-truth mask | resized bilinearly, then truncated to int, so only pixels left at full weight count | resized with nearest-neighbour, every nonzero pixel counts |
| where randomness restarts | `fix_seeds(seed)` at the top of every concept, then the model is rebuilt, so every concept draws the same prefix and the same coreset projection | one generator per run, seeded from `UCADConfig.seed`, advancing across concepts, so each concept draws its own |
| epoch ensembling and selection | always | never in the library |

The first two are metric conventions, not model quality: they change what is measured. Together they
are worth 0.040 of pixel AUPR on VisA and 0.068 on MVTec in pyCLAD's favour; equalise them and the
two implementations agree to 0.006 and 0.003 respectively.

**Patch score.** The reference reads distances straight out of `faiss.IndexFlatL2`, which returns
squared L2, and never takes the root. That is an artifact of the index, not a modelling decision -
the paper writes a distance. Squaring leaves image AUROC untouched, since the image score is the
maximum over patches and squaring preserves order, but it changes the anomaly map, which is squared
before being upsampled and smoothed.

**Ground-truth mask.** Bilinear resizing followed by truncation to int keeps only pixels that survive
at full weight, so annotated boundary pixels are discarded before scoring. Nearest-neighbour resizing
keeps the annotation intact, which is what a pixel metric is supposed to be measured against.

**Where randomness restarts.** `run_ucad_resumable.py` calls `patchcore.utils.fix_seeds(seed)` at the
top of the concept loop and then rebuilds the model, so the reference re-draws the identical prefix
and the identical coreset projection for every concept. One unlucky draw therefore lands on all
fifteen concepts at once, which is why its seed-to-seed spread is wide - MVTec hazelnut reads 0.7514,
0.9911, 0.9979 across three seeds where pyCLAD reads 0.9918, 1.0000, 0.9900. pyCLAD seeds one
generator per run and lets it advance, so a concept's draw is its own and a bad one stays local.

**Epoch ensembling and selection.** Neither is in the paper, the memory accounting rules the ensemble
out, and the selection reads test labels. Both stay out of the library; `scripts/` reproduces them.

## Things worth knowing before you tune it

The paper credits SCL with +0.088 image AUROC and +0.049 pixel AUPR (Table 5, CPM-only 0.786/0.251
against CPM+SCL 0.874/0.300). A no-SCL configuration lands on 0.7833/0.2718, matching their CPM-only
row, and the pixel half of their gain is reproduced by epoch ensembling alone.

Its instability is what makes the two mechanisms above so valuable to the reference: per-epoch image
AUROC swings by up to 0.42 within one run, which is exactly what the ensemble smooths over and what
the epoch selection exploits. `docs/ucad.md` has the rest of what is known about the loss and about
screw, the one category neither implementation brings near its published number.

**`args_dict.npy` in the authors' repository is misleading.** It contains `length=5` and
`batch_size=24`; neither reaches the model. The prompt is built with `prompt_length=1` in
`patchcore/patchcore.py`, and the batch size comes from a click default of 8, which is also what the
paper states.

**The paper's prompt shape does not match the released code.** The paper reports a prompt of size
(15, 7, 768); the code builds prefix tuning with `prompt_length=1` on twelve layers. We could not
determine which the published numbers correspond to.

## Measured results

Three seeds per side. Both columns per implementation are ensembled readings: "without selection" is
the 25-epoch ensemble after the last epoch, "reference-protocol" is the same ensemble with the epoch
chosen on the test set. The pyCLAD columns therefore come from `scripts/reference_ensemble.py`, not
from `UCADModel.predict` - they are what our model scores when put through the authors' machinery, so
that the two implementations are compared on equal terms. The library's own single-model reading is
0.7045 image AUROC on VisA.

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

## Where the defaults come from

| | default | source |
|---|---|---|
| `input_size` / `resize_mode` | (224, 224) / `short_side_crop` | reference's `Resize(224)` + `CenterCrop(224)`; paper fixes 224 only |
| `training_epochs` / `batch_size` | 25 / 8 | paper and reference agree |
| `learning_rate` / `grad_clip` | 5e-4 / 1.0 | paper and reference agree |
| `knowledge_size` / `key_size` | 196 / 196 | paper's memory accounting |
| `prompt_length` / `num_prompt_layers` / `feature_layer` | 1 / 12 / 5 | reference |
| `scl_temperature` | 0.5 | reference |
| `coreset_mode` | `approximate` | reference |
| `blur_sigma` | 4.0 | reference hardcodes 4 |
| `reweighting_num_nn` | 0, off | reference does not apply the paper's Eq. 5-6 |
| `reset_prompt_per_task` | True | differs from the reference, which carries the prompt over |

## Reproducing the tables

- `scripts/protocol_compare.py` trains one model per concept and reports both protocols from that one
  run, plus the per-epoch trajectory. `scripts/protocol_compare.sbatch` submits it.
- `scripts/reference_ensemble.py` holds the authors' epoch ensemble as a subclass of the model. It is
  reproduction machinery and deliberately outside the library.
- `scripts/pixel_convention_effect.py` scores one run against both ground-truth mask conventions.
- `scripts/ucad_probe.py` runs the model through the full continual scenario when the concept matrix
  is what is wanted.
