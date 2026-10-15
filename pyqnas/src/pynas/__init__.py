"""PyNAS: hardware-aware neural architecture search for segmentation on the Intel Myriad X.

Every candidate is checked against the device before it is trained (FP16 IR of OpenVINO 2022.3.1, compiled for
MYRIAD, size, layer types, predicted latency) and trained in the precision it runs in (FP16, BatchNorm folded in
the last phase), with distillation from a fixed teacher. See docs/DESIGN.md.

    from pynas import config, teacher, final
    from pynas.search import driver
    cfg = config.load("configs/burnscar.ini")
"""
__version__ = "1.1.0"
