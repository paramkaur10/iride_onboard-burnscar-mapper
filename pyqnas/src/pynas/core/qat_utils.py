from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Callable
import torch
from torch.ao.quantization.qconfig import get_default_qat_qconfig
from torch.nn.utils import parametrize


# Int 8 awareness followed by float 16 awarenss

# INT 8 awareness

# Compat shim: PyTorch 1.13 vs 2.x layout

try:
    # PyTorch 1.13.x
    from torch.ao.quantization.qconfig_mapping import QConfigMapping
    from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
except Exception:
    # PyTorch 2.x fallback
    from torch.ao.quantization.fx.qconfig_mapping import QConfigMapping  # type: ignore
    from torch.ao.quantization.fx.prepare_qat_fx import prepare_qat_fx   # type: ignore
    from torch.ao.quantization.fx.convert_fx import convert_fx           # type: ignore

@dataclass
class QATOpts:
    enabled: bool = False
    backend: str = "qnnpack"        # "qnnpack" (ARM) or "fbgemm" (x86)
    warmup_epochs: int = 0          # FP32 warm-up
    finetune_epochs: int = 2        # QAT epochs
    per_channel: bool = True
    tag: str = "qat"
    onnx_opset: int = 13
    onnx_name: str = "temp_model_qat.onnx"

    # NEW:
    mode: str = "int8"             # "int8" or "fp16"
    fp16_stochastic: bool = False
    fp16_round_weights: bool = True
    fp16_act_clip: Optional[float] = None
    fp16_tag: str = "fp16aware"


def read_qat_opts(cfg) -> "QATOpts":
    g = cfg.get; gb = cfg.getboolean; gi = cfg.getint
    fp16_clip_raw = g("QAT", "fp16_act_clip", fallback="").strip()
    fp16_clip = float(fp16_clip_raw) if fp16_clip_raw not in ("", None) else None
    return QATOpts(
        enabled=gb("QAT", "enabled", fallback=False),
        backend=g("QAT", "backend", fallback="qnnpack").lower(),
        warmup_epochs=gi("QAT", "warmup_epochs", fallback=0),
        finetune_epochs=gi("QAT", "finetune_epochs", fallback=2),
        per_channel=gb("QAT", "per_channel", fallback=True),
        tag=g("QAT", "tag", fallback="qat"),
        onnx_opset=gi("QAT", "onnx_opset", fallback=13),
        onnx_name=g("QAT", "onnx_name", fallback="temp_model_qat.onnx"),
        # NEW:
        mode=g("QAT", "mode", fallback="int8").lower(),
        fp16_stochastic=gb("QAT", "fp16_stochastic", fallback=False),
        fp16_round_weights=gb("QAT", "fp16_round_weights", fallback=True),
        fp16_act_clip=fp16_clip,
        fp16_tag=g("QAT", "fp16_tag", fallback="fp16aware"),
    )

'''
def prepare_qat(model: torch.nn.Module, example_input: torch.Tensor, backend: str) -> torch.nn.Module:
    torch.backends.quantized.engine = backend
    qconfig = get_default_qat_qconfig(backend)
    qmap = QConfigMapping().set_global(qconfig)
    model.eval()  # FX capture in eval
    prepared = prepare_qat_fx(model, qmap, example_inputs=example_input)
    prepared.train()  # enable fake-quant learnable params
    return prepared
'''

def prepare_qat(model: torch.nn.Module, example_input: torch.Tensor, backend: str) -> torch.nn.Module:
    import torch
    from torch.ao.quantization.qconfig_mapping import QConfigMapping
    from torch.ao.quantization.quantize_fx import prepare_qat_fx

    # 1) Set quantized backend (validated earlier so this shouldn’t throw)
    torch.backends.quantized.engine = backend
    qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    qmap = QConfigMapping().set_global(qconfig)

    # 2) Tell decoder(s) to skip the shape guard during FX tracing
    def _set_fx_guard(m, val: bool = True):
        if hasattr(m, "_skip_shape_guard_for_fx"):
            m._skip_shape_guard_for_fx = val

    model.apply(lambda m: _set_fx_guard(m, True))   # skip the `if` during tracing
    model.eval()                                     # FX capture in eval

    # make sure example_inputs is what FX expects
    ex = example_input if isinstance(example_input, (tuple, list)) else (example_input,)

    prepared = prepare_qat_fx(model, qmap, example_inputs=ex)

    # (Optional) restore flag on the *original* modules (the prepared model is a GraphModule)
    model.apply(lambda m: _set_fx_guard(m, False))

    prepared.train()  # enable fake-quant learnable params for QAT
    return prepared

def freeze_observers(model: torch.nn.Module):
    for m in model.modules():
        if hasattr(m, "qconfig") and hasattr(m, "activation_post_process"):
            if m.activation_post_process is not None and hasattr(m.activation_post_process, "disable_observer"):
                m.activation_post_process.disable_observer()

def enable_observers(model: torch.nn.Module):
    for m in model.modules():
        if hasattr(m, "qconfig") and hasattr(m, "activation_post_process"):
            if m.activation_post_process is not None and hasattr(m.activation_post_process, "enable_observer"):
                m.activation_post_process.enable_observer()

def convert_quantized(prepared: torch.nn.Module) -> torch.nn.Module:
    prepared.eval()
    return convert_fx(prepared)

def export_qdq_onnx(model: torch.nn.Module, example_input: torch.Tensor, onnx_path: Path, opset: int = 13):
    model = model.cpu().eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example_input.cpu(),
        str(onnx_path),
        opset_version=opset,
        do_constant_folding=False,   # keep Q/DQ explicit
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}},
    )
    return onnx_path






# FLOAT 16


# -------- Options --------
@dataclass
class FP16AwareOpts:
    enabled: bool = False           # turn on FP16 awareness
    stochastic: bool = False        # stochastic rounding instead of nearest
    after_modules: Tuple[type, ...] = (torch.nn.Conv2d, torch.nn.Linear)
    round_weights: bool = True      # also round weights in forward (fake)
    act_clip: Optional[float] = None  # e.g., 6.0 or 8.0 to tame FP16 overflow
    tag: str = "fp16aware"
    finetune_epochs: int = 2

def read_fp16aware_opts(cfg) -> "FP16AwareOpts":
    g  = cfg.get
    gb = cfg.getboolean
    gi = cfg.getint
    gf = cfg.getfloat
    act_clip = g("FP16AWARE", "act_clip", fallback="").strip()
    act_clip = float(act_clip) if act_clip not in ("", None) else None
    return FP16AwareOpts(
        enabled=gb("FP16AWARE", "enabled", fallback=False),
        stochastic=gb("FP16AWARE", "stochastic", fallback=False),
        round_weights=gb("FP16AWARE", "round_weights", fallback=True),
        act_clip=act_clip,
        tag=g("FP16AWARE", "tag", fallback="fp16aware"),
        finetune_epochs=gi("FP16AWARE", "finetune_epochs", fallback=2),
    )

# -------- Rounding primitives --------
@torch.no_grad()
def _fake_fp16_round(x: torch.Tensor) -> torch.Tensor:
    return x.half().float()


class _FP16STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, stochastic: bool) -> torch.Tensor:
        if stochastic:
            # stochastic rounding to nearest fp16 neighbor
            x16 = x.half()
            # compute next-up neighbor at fp32 then cast down
            up = torch.nextafter(x16.float(), torch.tensor(float('inf'), device=x.device)).half()
            x16f, upf = x16.float(), up.float()
            denom = (upf - x16f).abs().clamp_min(1e-12)
            p = ((x - x16f) / denom).clamp(0, 1)
            mask = (torch.rand_like(p) < p).to(x.dtype)
            y = (x16f + (upf - x16f) * mask).to(x.dtype)
            return y.half().float()
        else:
            # deterministic: cast to fp16 then back
            return x.half().float()

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through: dL/dx ≈ dL/d(round_fp16(x))
        return grad_output, None  # None for 'stochastic' arg

def _ste_fp16_round(x: torch.Tensor, stochastic: bool) -> torch.Tensor:
    return _FP16STE.apply(x, stochastic)

def _maybe_clip(x: torch.Tensor, T: Optional[float]) -> torch.Tensor:
    return torch.clamp(x, -T, T) if (T is not None) else x

def fp16_opts_from_qat(q: QATOpts) -> "FP16AwareOpts":
    return FP16AwareOpts(
        enabled=True,
        stochastic=q.fp16_stochastic,
        round_weights=q.fp16_round_weights,
        act_clip=q.fp16_act_clip,
        tag=q.fp16_tag,
        finetune_epochs=q.finetune_epochs,
    )

# -------- Activation fake-FP16 via forward hooks --------
class _ActFP16Hook:
    def __init__(self, stochastic: bool, act_clip: Optional[float]):
        self.stochastic = stochastic
        self.clipT = act_clip

    def __call__(self, module, inp, out):
        def _proc(t):
            t = _maybe_clip(t, self.clipT)
            return _ste_fp16_round(t, self.stochastic)
        if isinstance(out, (tuple, list)):
            return type(out)(_proc(o) for o in out)
        return _proc(out)

# -------- Weight fake-FP16 via parametrization --------
class _FP16WeightParam(torch.nn.Module):
    def __init__(self, stochastic: bool):
        super().__init__()
        self.stochastic = stochastic

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return _ste_fp16_round(w, self.stochastic)


# Keep track of handles to remove later
class FP16AwareContext:
    def __init__(self):
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.weight_parametrized: List[Tuple[torch.nn.Module, str]] = []

    def close(self):
        for h in self.hooks:
            try: h.remove()
            except Exception: pass
        for m, pname in self.weight_parametrized:
            try: parametrize.remove_parametrizations(m, pname, leave_parametrized=False)
            except Exception: pass
        self.hooks.clear()
        self.weight_parametrized.clear()


def prepare_fp16_aware(model: torch.nn.Module,
                       opts: FP16AwareOpts,
                       target_types: Tuple[type, ...] = (torch.nn.Conv2d, torch.nn.Linear)) -> FP16AwareContext:
    ctx = FP16AwareContext()
    act_hook = _ActFP16Hook(stochastic=opts.stochastic, act_clip=opts.act_clip)

    # Activations: add post-module hooks
    for m in model.modules():
        if isinstance(m, target_types):
            try:
                # PyTorch ≥ 2.0
                h = m.register_forward_hook(act_hook, with_kwargs=False)
            except TypeError:
                # PyTorch 1.x fallback
                h = m.register_forward_hook(act_hook)
            ctx.hooks.append(h)

    # Weights: parametrization that rounds to fp16 every forward
    if opts.round_weights:
        for m in model.modules():
            if isinstance(m, target_types) and hasattr(m, "weight"):
                try:
                    parametrize.register_parametrization(m, "weight", _FP16WeightParam(opts.stochastic))
                    ctx.weight_parametrized.append((m, "weight"))
                except Exception:
                    pass
    return ctx

