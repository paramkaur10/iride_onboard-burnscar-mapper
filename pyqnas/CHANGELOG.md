# Changelog

## 1.1.3

- `final.deploy(..., label=, seed=)`: exports any labelled run of a queue, not only the winner (e.g. the
  chosen finalist from the finalist queue, `label="finalist 4"`); with `seed` that seed, else the seed with the
  best mean validation score in FP16. The record says which label and how the seed was picked.
  `final.deploy_pick` is that choice on its own.

## 1.1.2

- `data.batch_input(..., input_mask=)`, `metrics.evaluate(..., input_mask=)`, `report.predict_set(..., input_mask=)`:
  `"labels"` (default, as in training: pixels without a label are set to 0 in the input) or `"image"` (only the
  image's own no-data pixels, as on board, where there are no labels). The scores always count labelled pixels only.
- `space.network.n_params_used`: parameters the forward pass reaches (what the exported network holds).

## 1.1.1

- `report.scene_table`: per scene and resolution of a stored set, every model's target IoU, hits, false alarms and
  misses; `report.plot_scenes(..., tags=, labels=)` shows each scene's split in its title. For reviewing whole
  scenes of every split (notebook 10); nothing is chosen from it.

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
