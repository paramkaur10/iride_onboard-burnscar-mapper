# PyNAS

Hardware-aware neural architecture search for semantic segmentation on the Intel Myriad X (OpenVINO 2022.3.1,
FP16 IR, MYRIAD plugin, as run by ArgOS on the Unibap iX5-106).

- every candidate is checked against the device before it is trained: FP16 IR, compiled for MYRIAD, IR size,
  layer types, predicted latency;
- every candidate is trained in the precision it runs in: FP16 mixed precision, BatchNorm folded into the
  convolutions for the last phase, scored in FP16;
- distillation from a fixed teacher, NSGA-II over accuracy and latency;
- one trainer for every model (candidates, final models, baselines, ablations, teacher), one configuration file
  per experiment, a record of versions and code with every result.

The design, the protocol and the search space are in [docs/DESIGN.md](docs/DESIGN.md).

## Layout

```
src/pynas/
  config.py, config.ini   configuration: defaults, experiment file, per-run overrides
  record.py               versions, source hash, git commit, machine
  space/                  genes.py (chromosomes), blocks.py, encoders.py (smp, trimmed), network.py
  train/                  data.py (window pool), losses.py, precision.py (folding, FP16), trainer.py, metrics.py
  hardware/               myriad.py (IR, compile, IR on CPU), latency.py (nn-Meter), constraints.py, tools/
  search/                 nsga2.py, driver.py (generation loop in a background process)
  runner.py               queue of runs, one worker per GPU
  teacher.py, final.py    teacher and soft labels; final stage, export, test
  report.py               tables and figures from the run folders
tests/                    python tests/<file>.py, or pytest
```

## Use

```python
from pynas import config, teacher, final, runner
from pynas.search import driver
cfg = config.load("configs/burnscar.ini")
q = teacher.queue(cfg, "models/pynas/teacher"); runner.launch(q, gpus=[0, 1]); runner.wait(q)
teacher.write_soft_labels(q, cfg)
driver.launch("configs/burnscar.ini", "models/pynas/search")        # background; driver.status(...)
fq = final.queue(cfg, "models/pynas/search", "models/pynas/final"); runner.launch(fq, gpus=[0, 1])
final.deploy(fq, cfg, "models/onboard", "burnscar_pynas")
final.test(fq, cfg, teacher_queue="models/pynas/teacher/runs/queue.json")
```

The Myriad X tools live in environments of their own (old packages): `pynas.hardware.myriad.setup(cfg)` makes them
once per machine.
