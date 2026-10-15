# PyNAS 1.0 — design

PyNAS searches segmentation networks for the Intel Myriad X (OpenVINO 2022.3.1, FP16 IR, MYRIAD plugin, as
run by ArgOS on the Unibap iX5-106). Every candidate is checked against the device before it is trained, and
trained in the precision it will run in.

## 1. The experiment

```
1 teacher    a fixed, large network (ResNet-50 U-Net), trained once in FP32 on the training split;
             its per-pixel class probabilities on the training windows are stored (soft labels)
2 search     NSGA-II over the search space, P candidates per generation, G generations:
               a  feasibility first: ONNX -> FP16 IR -> compiled for MYRIAD, IR size, layer types,
                  predicted latency; an infeasible candidate is never trained and is replaced
               b  training in FP16 with distillation from the teacher; for the last part of the steps the
                  BatchNorm layers are folded into the convolutions, as the Model Optimizer folds them
               c  fitness: fresh-burn IoU of the folded network run in FP16, on the search split, scene-balanced
                  (the mean of the per-scene IoUs, so that no single scene decides it)
               d  selection on (IoU up, latency down), crossover, mutation
3 final      with [Final] finalists = k > 0, first the finalist stage: the k candidates with the best search
             fitness trained like final models (every seed, the final number of steps) and compared on the
             validation sets, which the search never saw; those within one standard error of the best mean
             validation score count as equal and the fastest is taken (section 5.1). That model (or, with
             finalists = 0, the search's winner by the rule set before the search) is trained again with
             several seeds; the seed with the best validation score is exported (FP16 IR, compiled for MYRIAD);
             baselines and ablations go through the same trainer; the test split is read once
```

Data splits, by whole scenes or groups of scenes: `train` (teacher and candidates learn), `search` (fitness),
`val` (finalist and seed choice), `test` (final numbers, read once by `final.test`).

## 2. Search space (`pynas.space`)

A network is an encoder, searched layers going down, a U-Net decoder and a head.

```
E:<encoder>:<depth>                     pretrained encoder (smp) cut after <depth> stages, or E:none:1
                                        (a 3x3 stride-2 stem of searched width)
L:<block>:<width>:<kernel>:<expansion>:<activation>:<pool>     0 or more, each one halves the resolution
D:<block>:<width>:<convs>               decoder: every stage upsamples x2 (bilinear), concatenates the
                                        skip connection and applies <convs> blocks of <width> channels
```

Genes are written `/`-separated, e.g. `E:eb0:4/L:mb:64:3:4:silu:max/D:conv:48:2`. Writing and reading are
exact inverses; the run name is a hash of the written chromosome and the training settings.

- encoders: `none`, `eb0` (efficientnet-b0), `mv2` (mobilenet_v2), `r18` (resnet18); depths 3–5;
  parts of an encoder that its feature maps do not need are removed after building (checked to leave the
  feature maps unchanged), so sizes and latencies are those of what runs.
- blocks: `conv` (conv-BN-act), `dws` (depthwise separable), `mb` (inverted residual), `mbse` (inverted
  residual with squeeze-excitation), `res` (residual basic block); pool `max` or `avg` before the block.
- every operation is one the MYRIAD plugin runs: convolution (incl. depthwise), BatchNorm (folded), ReLU,
  Swish, sigmoid, element-wise add and multiply, max/average pooling, global average pooling, concatenation,
  bilinear interpolation.
- limits: total stride of the network, number of searched layers, IR size; the decoder ends at stride 2 and
  the head (1x1 convolution) is upsampled to the input size.

Teacher and baselines are fixed specifications `smp:<Architecture>:<encoder>`, built by the same function.

## 3. Training (`pynas.train`)

One trainer for every model (candidates, final models, baselines, ablations, teacher):

- window pool on disk (memory-mapped), random windows drawn with replacement, optionally balanced by source;
  rotations, flips, gains, band misregistration and noise on the GPU;
- loss `(1 - w) CE + w KL(teacher || model)` over pixels with data, class weights in CE, optional soft Dice;
- AdamW, linear warm-up, cosine decay to 2 % of the rate, gradient clipping at norm 1;
- precision `fp16` (onboard models): mixed precision in FP16 (FP16 weights and activations in the forward
  pass, FP32 master weights, dynamic loss scaling); at `fold_at` x steps the BatchNorm layers are folded into
  the convolution before them, the outputs of the network being checked to be unchanged, and training goes on
  with the folded network, which is the one that is exported; precision `fp32` or `bf16` for the teacher;
- exponential moving average of the weights over the last phase; the averaged weights are the result (no
  checkpoint is chosen on any held-out set);
- evaluation: per class and per ground resolution confusion matrices; for onboard models the folded network
  in half precision (`precision=fp16`), as the Myriad runs it, and the same weights in FP32 for comparison.

## 4. Hardware (`pynas.hardware`)

ONNX (opset 13, IR version 7) → FP16 IR with the OpenVINO 2022.3.1 Model Optimizer (`.xml`, `.bin`,
`.mapping`) → `compile_tool` of the OpenVINO 2022.3.1 archive for MYRIAD (no device needed). Feasible: compiled,
IR size within the limit, layer types within the ArgOS list when given, predicted latency (nn-Meter, Myriad
VPU predictor) within the limit when given. A cheap size estimate rejects candidates before any compiling.
The IR can be run on the OpenVINO CPU plugin to compare its outputs with PyTorch.

## 5. Search (`pynas.search`)

Generation 0: random chromosomes, duplicates and infeasible ones replaced, until P feasible candidates.
Generation g > 0: P new feasible offspring from binary tournaments on (Pareto rank, crowding distance),
one-point crossover of the layer genes (encoder from one parent, decoder from the other), per-gene mutation,
repair to the limits; the next population is the best P of parents and offspring (NSGA-II). Objectives and
constraints are set in the configuration. Each generation's candidates are trained in the background, one
worker per GPU; the driver resumes after a stop and never trains a run twice.

### 5.1 The winner's curse and the finalist stage

A candidate's fitness is one measurement: one training seed, a shortened schedule ([Train] steps), a search split
of a few scene groups. It is the candidate's quality plus an error, and the best of P x G such measurements is
picked as much for a large error as for its quality, so its score overstates it (regression to the mean; the
winner's curse). The size of this is measured, not assumed: every finalist's search score is reported next to the
same split measured again after the final training, mean over seeds (`finalists.json`, `report.plot_finalists`).
The final choice is therefore not made on search scores. The finalists, the best k candidates by search fitness,
are trained as final models (every seed of [Final] seeds, [Final] steps) and compared on the validation sets, which
take no part in the search: the selection error then comes only from seeds and validation scenes, and is shown as
a standard error. Differences within one standard error of the best are not taken as real (the one-standard-error
rule); among those finalists the one with the lowest predicted latency is chosen. Runs already done with the same
deciding settings (the same run name), such as the search winner's final runs, are copied, with a `reused.json`
saying from where, and not trained again.

## 6. Records

Every run folder holds the effective configuration, `run.json` (architecture, seed, settings, PyNAS version,
hash of the PyNAS source, git commit when there is one, Python / PyTorch / CUDA versions, GPU), `history.csv`,
`model.pt` (the result, folded where applicable, with the list of folded layers), `scores.json` and, for
exported models, the IR files and their check. The search folder holds every candidate tried, the reason
infeasible ones were dropped, the generations and the Pareto fronts. Paper tables and figures are made from
these folders by `pynas.report`. Runs are seeded (weights, window draws, augmentation); GPU kernels are not bit-deterministic, so
the spread over seeds is reported for every final number.
