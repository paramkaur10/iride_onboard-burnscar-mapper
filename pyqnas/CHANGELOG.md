# Changelog

## 1.1.0

- Finalist stage (`[Final] finalists = k`): the k candidates with the best search fitness are trained like final
  models (every seed, [Final] steps) and compared on the validation sets; finalists within one standard error of
  the best mean validation score count as equal and the one with the lowest predicted latency is taken
  (`final.finalist_queue`, `final.pick_finalist`, `finalists.json`). The best of many single search scores
  overstates its candidate (the winner's curse); the table of finalists shows each search score next to the same
  split measured again. Finalist 1 is the search's winner (same ordering as `[Final] pick`).
- Runs already done with the same deciding settings are copied into a new stage, with `reused.json`, not trained
  again (`final.queue(..., reuse=[...])`).
- Figures: `report.plot_finalists`, `report.plot_test_iou` (target IoU per test set and resolution, mean and
  standard deviation over seeds), `report.predict_set` and `report.plot_scenes` (test scenes as mosaics: image,
  labels, each model's hits, false alarms and misses, with the IoU of that scene and resolution).

## 1.0.1

- Scene-balanced scores: besides the IoU over all pixels of a set, the mean of the per-scene IoUs (scenes with at
  least 500 labelled target pixels) and the worst scene. `[Search] fitness = scene_mean` (default) makes it the
  fitness of the search and the score the deployed seed is chosen by, so that one scene holding most of the target
  class cannot decide them alone; `pooled` gives the IoU over all pixels. Both are recorded for every run.

## 1.0.0

PyNAS rewritten around one experiment protocol (docs/DESIGN.md). Earlier versions (0.1, 0.2) are kept in the
project's archive folder; nothing of them is imported.

- Search space: encoder gene (pretrained smp encoder cut after some stages, or a stem), searched layers (pooling +
  block: conv, depthwise separable, inverted residual with or without squeeze-excitation, residual), decoder gene;
  only operations the MYRIAD plugin runs; written and read as exact inverses.
- Feasibility before training: FP16 size estimate, FP16 IR compiled for MYRIAD (parallel), IR size, ArgOS layer
  list, predicted latency; infeasible candidates are replaced, never trained, and recorded with the reason.
- One trainer: FP16 mixed precision for onboard models, BatchNorm folded into the convolutions for the last phase
  (outputs checked unchanged), weights averaged over that phase; bf16/FP32 for the teacher; distillation; soft Dice.
- Scores of onboard models computed on the folded network in FP16, with the FP32 scores and the FP16/FP32 agreement.
- NSGA-II with elitism; candidates without a score rank last on every objective; the winner rule set in the
  configuration before the search.
- Final stage in the package: seeds, baselines, ablations (no distillation; FP32 training converted afterwards),
  export for ArgOS with the IR run on the CPU plugin, the test split read once.
- Every run folder records the effective configuration, the PyNAS source hash, the git commit and the versions.
