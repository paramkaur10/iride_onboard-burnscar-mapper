"""Hand-written quantization simulation for py-q-nas (no torch.ao).

Public API:
    QuantOpts            - scheme configuration (defaults = Hailo-8 W8A8)
    prepare_hailo_quant  - install fake-quant wrappers on a model
    read_quant_opts      - build QuantOpts from config.ini [QuantSim]
    freeze_all_ranges    - freeze activation observers
"""

from .fake_quant import QuantOpts, fake_quant_weight, ActFakeQuant, round_ste, clamp_ste
from .qmodules import QuantConv2d, QuantLinear
from .prepare import prepare_hailo_quant, read_quant_opts, freeze_all_ranges, QuantContext

__all__ = [
    "QuantOpts",
    "fake_quant_weight",
    "ActFakeQuant",
    "round_ste",
    "clamp_ste",
    "QuantConv2d",
    "QuantLinear",
    "prepare_hailo_quant",
    "read_quant_opts",
    "freeze_all_ranges",
    "QuantContext",
]