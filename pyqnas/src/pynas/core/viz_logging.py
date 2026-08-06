# viz_logging.py
from __future__ import annotations

from pathlib import Path
import torch
import torch.nn.functional as F
import torch.fx as fx
from torch.fx.passes.graph_drawer import FxGraphDrawer

# Optional deps
try:
    from torch.fx.experimental.proxy_tensor import make_fx
    _HAS_MAKE_FX = True
except Exception:
    _HAS_MAKE_FX = False

try:
    from torchviz import make_dot
    _HAS_TORCHVIZ = True
except Exception:
    _HAS_TORCHVIZ = False


# --------------------------------------------------------------------------------------
# Public helper: build an example input tensor from the datamodule's declared input_shape
# --------------------------------------------------------------------------------------
'''
def example_input_from_dm(dm):
    """
    Create a single-batch example input from dm.input_shape.
    Returns a CPU tensor; caller can move it to device if needed.
    """
    shape = dm.input_shape
    if len(shape) == 3:  # (C,H,W)
        shape = (1, *shape)
    return torch.randn(*shape)
'''
def example_input_from_dm(dm, device="cpu"):
    shape = dm.input_shape
    if len(shape) == 3:  # (C,H,W)
        shape = (1, *shape)
    return torch.randn(*shape, device=device)




# --------------------------------------------------------------------------------------
# Torchviz: create a scalar loss so autograd graph expands fully (classification/seg)
# --------------------------------------------------------------------------------------
def make_viz_loss(pl_module, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass + construct a scalar loss so torchviz expands a meaningful autograd graph.
    Works with both classification and segmentation Lightning modules in this repo.
    """
    model = pl_module.model
    device = next(model.parameters()).device
    x = x.to(device)
    y = model(x)

    # If the Lightning module exposes a criterion, use it for realism
    crit = getattr(pl_module, "loss_fn", None)
    if crit is not None:
        # classification logits: (N, C) or segmentation: (N, C, H, W)
        if y.dim() == 2:
            n, c = y.shape
            target = torch.randint(0, c, (n,), device=device)
            return crit(y, target)
        elif y.dim() == 4:
            n, c, h, w = y.shape
            target = torch.randint(0, c, (n, h, w), device=device)
            return crit(y, target)

    # Fallback: any scalar is fine
    return y.float().sum()


# --------------------------------------------------------------------------------------
# FX tracing: treat our large custom modules as leaves to keep graphs legible
# --------------------------------------------------------------------------------------
def _is_custom_leaf(m: torch.nn.Module) -> bool:
    try:
        # Local import to avoid circular imports
        from pynas.core.generic_unet import GenericUNetNetwork, UNetDecoder
        return isinstance(m, (GenericUNetNetwork, UNetDecoder))
    except Exception:
        return False


class LeafyTracer(fx.Tracer):
    def is_leaf_module(self, m: torch.nn.Module, qualname: str) -> bool:
        return _is_custom_leaf(m) or super().is_leaf_module(m, qualname)


# --------------------------------------------------------------------------------------
# FX writers: always IR text; attempt SVG via pydot+graphviz; never break training
# --------------------------------------------------------------------------------------
def _write_ir_and_svg(gm: fx.GraphModule, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1) textual IR (diffable and robust)
    (out_dir / f"{stem}.ir.txt").write_text(str(gm.graph))

    # 2) SVG (visual) — requires pydot + graphviz dot
    try:
        drawer = FxGraphDrawer(gm, stem)
        # New API path (PyTorch ≥ 1.13 / 2.x)
        drawer.get_dot_graph().write_svg(str(out_dir / f"{stem}.svg"))
    except Exception as e:
        # Don’t fail training — log why SVG wasn’t produced
        (out_dir / "_fx_svg_error.txt").write_text(repr(e))


def dump_fx_graph_with_example(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    out_dir: Path,
    stem: str = "fx",
) -> None:
    """
    Robust FX dump:
      1) Try input-aware tracing (make_fx) for better accuracy on dynamic models.
      2) Fall back to LeafyTracer() symbolic trace.
      Always writes <stem>.ir.txt; tries <stem>.svg; if both fail, writes _fx_error.txt.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    m = model.eval().cpu()
    ex = example_input.detach().cpu()

    # Path 1: make_fx (input-aware) if available
    if _HAS_MAKE_FX:
        try:
            gm = make_fx(m)(ex)
            _write_ir_and_svg(gm, out_dir, stem)
            return
        except Exception as e1:
            # Continue to fallback path
            last_make_fx_err = e1
    else:
        last_make_fx_err = RuntimeError("torch.fx.experimental.proxy_tensor.make_fx unavailable")

    # Path 2: LeafyTracer symbolic trace
    try:
        graph = LeafyTracer().trace(m)
        gm = fx.GraphModule(m, graph)
        _write_ir_and_svg(gm, out_dir, stem)
        return
    except Exception as e2:
        # Both paths failed — record both reasons
        (out_dir / "_fx_error.txt").write_text(
            "make_fx failed:\n" + repr(last_make_fx_err) + "\n\n"
            "LeafyTracer failed:\n" + repr(e2) + "\n"
        )


# --------------------------------------------------------------------------------------
# Torchviz: autograd graph rooted at scalar loss
# --------------------------------------------------------------------------------------
def dump_torchviz_png(
    pl_module,
    example_input: torch.Tensor,
    out_dir: Path,
    name: str = "torchviz",
) -> Path | None:
    """
    Render a torchviz PNG rooted at a scalar loss (not a raw tensor).
    Returns path if created, else None.
    """
    if not _HAS_TORCHVIZ:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        loss = make_viz_loss(pl_module, example_input)
        dot = make_dot(loss, params=dict(pl_module.model.named_parameters()))
        png_path = out_dir / f"{name}.png"
        # cleanup=True removes the intermediate .dot file
        dot.render(str(png_path.with_suffix("")), format="png", cleanup=True)
        return png_path
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Lightweight numeric per-parameter stats (good for papers & diffs)
# --------------------------------------------------------------------------------------
def collect_weight_stats(model: torch.nn.Module) -> dict:
    stats = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        t = p.detach().float().view(-1)
        if t.numel() == 0:
            continue
        stats[n] = {
            "numel": int(t.numel()),
            "min": float(t.min()),
            "max": float(t.max()),
            "mean": float(t.mean()),
            "std": float(t.std(unbiased=False)),
            "l2": float(torch.linalg.vector_norm(t)),
        }
    return stats
