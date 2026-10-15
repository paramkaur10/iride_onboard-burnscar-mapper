"""One training, the same for every model (search candidates, final models, baselines, ablations, teacher).

    settings = Settings.from_config(cfg)              # [Train] (the teacher: [Train] with [Teacher] on top)
    result, history, info = train(model, batches, spec, settings, device, example)

Schedule: AdamW, linear warm-up, cosine decay to lr_floor x lr, gradient clipping. Precision:
    fp16  mixed precision in FP16 (autocast, FP32 master weights, dynamic loss scaling): the network computes in
          the precision of the Myriad X
    bf16  mixed precision in bfloat16 (models that stay on the ground)
    fp32  no mixed precision
The last phase starts at final_phase x steps: the BatchNorm layers are folded into the convolutions
(precision.fold, outputs checked unchanged) when fold is true, or frozen at their statistics otherwise; a new
optimizer continues the schedule, and the weights are averaged (EMA) over the phase. The averaged weights are the
result: no checkpoint is chosen on any held-out set.
"""
from __future__ import annotations

import contextlib
import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..space.network import encoder_parameters
from . import losses
from .data import augment, class_lut, normalise, teacher_logp
from .precision import fold, freeze_bn


@dataclass
class Settings:
    steps: int = 10000
    batch: int = 16
    lr: float = 2e-3
    encoder_lr_scale: float = 1.0
    weight_decay: float = 1e-2
    warmup: int = 100
    lr_floor: float = 0.02
    grad_clip: float = 1.0
    precision: str = "fp16"
    final_phase: float = 0.8
    fold: bool = True
    ema_decay: float = 0.999
    kd_weight: float = 0.8
    kd_temperature: float = 1.0
    dice_weight: float = 0.0
    seed: int = 0
    eval_every: int = 0
    log_every: int = 200
    time_limit_minutes: float = 0.0

    @classmethod
    def from_config(cls, cfg, section="Train", over=None):
        """[Train], with the options of section `over` (e.g. Teacher) that [Train] also has taking precedence."""
        def get(opt, cast):
            sec = over if (over and cfg.has_option(over, opt)) else section
            raw = cfg.get(sec, opt)
            return raw.strip().lower() in ("1", "true", "yes", "on") if cast is bool else cast(raw)
        kw = {f: get(f, t) for f, t in (("steps", int), ("batch", int), ("lr", float), ("encoder_lr_scale", float),
                                          ("weight_decay", float), ("warmup", int), ("lr_floor", float),
                                          ("grad_clip", float), ("precision", str), ("final_phase", float),
                                          ("fold", bool), ("ema_decay", float), ("kd_weight", float),
                                          ("kd_temperature", float), ("dice_weight", float), ("seed", int),
                                          ("eval_every", int), ("log_every", int), ("time_limit_minutes", float))}
        kw["precision"] = kw["precision"].strip().lower()
        if kw["precision"] not in ("fp16", "bf16", "fp32"):
            raise ValueError(f"precision {kw['precision']}: fp16, bf16 or fp32")
        return cls(**kw)


class Ema:
    """Exponential moving average of the parameters (buffers are taken from the model as they are). The decay
    starts low and grows to `decay`, so that the first steps do not weigh for ever."""

    def __init__(self, model, decay=0.999):
        self.decay, self.n = decay, 0
        self.params = [p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        torch._foreach_lerp_(self.params, [p.detach() for p in model.parameters()], 1.0 - d)

    @torch.no_grad()
    def copy_of(self, model):
        m = copy.deepcopy(model)
        for p, e in zip(m.parameters(), self.params):
            p.copy_(e)
        return m


def _factor(step, s):
    if step < s.warmup:
        return (step + 1) / s.warmup
    p = min((step - s.warmup) / max(1, s.steps - s.warmup), 1.0)
    return s.lr_floor + (1 - s.lr_floor) * 0.5 * (1 + math.cos(math.pi * p))


def _optimizer(model, s):
    enc = {id(p) for p in encoder_parameters(model)}
    rest = [p for p in model.parameters() if id(p) not in enc and p.requires_grad]
    groups = [dict(params=rest, base_lr=s.lr)]
    encp = [p for p in model.parameters() if id(p) in enc and p.requires_grad]
    if encp:
        groups.append(dict(params=encp, base_lr=s.lr * s.encoder_lr_scale))
    for g in groups:
        g["lr"] = g["base_lr"]
    return torch.optim.AdamW(groups, lr=s.lr, weight_decay=s.weight_decay)


def _scaler(enabled):
    """Dynamic loss scaling for FP16 (torch.amp in recent PyTorch, torch.cuda.amp before)."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _mode(model, last_phase):
    model.train()
    return freeze_bn(model) if last_phase else model


def train(model, batches, spec, s, device, example, eval_fn=None, log=None, name=""):
    """Train model on the batches (data.loader: stored values, masks, teacher arrays, indices). example: a batch
    of network inputs on the device, for the BatchNorm folding check. eval_fn(model, step) -> dict, called every
    s.eval_every steps with a copy of the current result. Returns (result model, history DataFrame, info dict)."""
    torch.manual_seed(s.seed)
    np.random.seed(s.seed)
    cuda = device.type == "cuda"
    precision = s.precision
    if precision == "bf16" and cuda and not torch.cuda.is_bf16_supported():
        precision = "fp16"                          # GPUs without bfloat16: FP16 mixed precision with loss scaling
    amp = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[precision]
    scaler = _scaler(precision == "fp16" and cuda)
    model = model.to(device)
    _mode(model, False)
    opt = _optimizer(model, s)
    phase_at = int(round(s.final_phase * s.steps)) if s.final_phase and s.final_phase < 1 else s.steps
    weights = torch.as_tensor(spec.class_weights, dtype=torch.float32, device=device) if spec.class_weights else None
    lut = class_lut(spec, device)
    gen = torch.Generator()
    gen.manual_seed(s.seed)
    use_teacher = s.kd_weight > 0
    ema, pairs, last_phase = None, [], False
    hist, t0 = [], time.time()
    acc = dict(loss=0.0, ce=0.0, kd=0.0, n=0)
    info = dict(not_finite_steps=0, folded_pairs=0, last_phase_step=None, stopped=None, precision=precision)

    def result():
        return ema.copy_of(model) if ema is not None else copy.deepcopy(model)

    step = 0
    for step, (r, m, t, _) in enumerate(batches, 1):
        if step - 1 == phase_at and not last_phase:
            if s.fold:
                model, pairs = fold(model, example)
                info["folded_pairs"] = len(pairs)
                model = model.to(device)
                opt = _optimizer(model, s)
            last_phase = True
            info["last_phase_step"] = step - 1
            _mode(model, True)
            ema = Ema(model, s.ema_decay)
        r, m = r.to(device, non_blocking=True), m.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True) if (use_teacher and t.numel()) else None
        refl = r.float() / spec.scale
        if spec.augment:
            refl, m, t = augment(refl, m, t, gen, spec)
        x = normalise(refl, m, spec)
        y = lut[m.long()]
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * _factor(step - 1, s)
        with (torch.autocast(device.type, dtype=amp) if (amp is not None and cuda) else contextlib.nullcontext()):
            out = model(x)
        loss, ce, kd = losses.loss(out, y, teacher_logp(t) if t is not None else None, (m != spec.nodata).float(),
                                   weights, s.kd_weight, s.kd_temperature, s.dice_weight)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            info["not_finite_steps"] += 1          # skipped: the forward pass overflowed (FP16 range)
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), s.grad_clip)
        scaler.step(opt)
        scaler.update()
        if ema is not None:
            ema.update(model)
        acc["loss"] += float(loss.detach())
        acc["ce"] += float(ce.detach())
        acc["kd"] += float(kd.detach())
        acc["n"] += 1
        last = step == s.steps
        if (s.log_every and step % s.log_every == 0) or last:
            k = max(acc["n"], 1)
            hist.append(dict(step=step, loss=acc["loss"] / k, ce=acc["ce"] / k, kd=acc["kd"] / k,
                             lr=opt.param_groups[0]["lr"], scale=float(scaler.get_scale()) if scaler.is_enabled() else 1.0,
                             not_finite=info["not_finite_steps"], seconds=time.time() - t0))
            acc = dict(loss=0.0, ce=0.0, kd=0.0, n=0)
            if log:
                log(f"{name} step {step}/{s.steps}  loss {hist[-1]['loss']:.4f}  ce {hist[-1]['ce']:.4f}  "
                    f"kd {hist[-1]['kd']:.4f}  {hist[-1]['seconds']:.0f} s")
        if eval_fn is not None and s.eval_every and step % s.eval_every == 0 and not last:
            ev = eval_fn(result(), step)
            if not hist or hist[-1]["step"] != step:
                hist.append(dict(step=step, seconds=time.time() - t0))
            hist[-1].update(ev)
            _mode(model, last_phase)
        if s.time_limit_minutes and time.time() - t0 > 60 * s.time_limit_minutes:
            info["stopped"] = f"time limit at step {step}"
            break
    if not last_phase:                              # stopped before the last phase: fold / freeze now
        if s.fold:
            model, pairs = fold(model, example)
            info["folded_pairs"] = len(pairs)
        info["last_phase_step"] = step
    info.update(steps_done=step, seconds=time.time() - t0, folded=bool(s.fold), pairs=[list(p) for p in pairs])
    return result().eval(), pd.DataFrame(hist), info
