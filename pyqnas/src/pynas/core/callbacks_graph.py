# callbacks_graph.py
import json
from pathlib import Path
import pytorch_lightning as pl
from .viz_logging import (
    example_input_from_dm,
    dump_fx_graph_with_example,
    dump_torchviz_png,
    collect_weight_stats,
)

class GraphOnce(pl.Callback):
    """Log the computation graph once to TensorBoard (if available)."""
    def __init__(self, example_input):
        self.example = example_input
        self.done = False

    def on_fit_start(self, trainer, pl_module):
        if self.done:
            return
        try:
            tb = getattr(trainer.logger, "experiment", None)
            if tb is not None and hasattr(tb, "add_graph"):
                x = self.example.to(pl_module.device)
                tb.add_graph(pl_module.model, x)
        except Exception:
            pass
        self.done = True


class WeightHists(pl.Callback):
    """Periodically log weight/grad histograms to TensorBoard."""
    def __init__(self, every_n_epochs=1):
        self.n = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.n != 0:
            return
        tb = getattr(trainer.logger, "experiment", None)
        if tb is None:
            return
        global_step = trainer.global_step
        for name, p in pl_module.model.named_parameters():
            try:
                tb.add_histogram(f"weights/{name}", p.detach().cpu(), global_step)
                if p.grad is not None:
                    tb.add_histogram(f"grads/{name}", p.grad.detach().cpu(), global_step)
            except Exception:
                # keep training even if a particular tensor can't be histogrammed
                pass


class GraphSnapshot(pl.Callback):
    """
    Save FX IR (always) + FX SVG (if possible) + torchviz PNG (loss-rooted) + weight stats JSON.
    Files are placed under: <out_dir>/snapshots_<tag>/epoch_###/
    """
    def __init__(self, out_dir: Path, example_input, every_n_epochs=1, tag="fp32"):
        self.out_dir = Path(out_dir) / f"snapshots_{tag}"
        self.every = every_n_epochs
        self.example = example_input  # CPU tensor; moved to device in viz helpers

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every != 0:
            return

        epoch_dir = self.out_dir / f"epoch_{trainer.current_epoch + 1:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # 1) FX graph (operator-level). Always get IR text; SVG if dot is present.
        try:
            dump_fx_graph_with_example(pl_module.model, self.example, epoch_dir, stem="fx")
            dump_torchviz_png(pl_module, self.example, epoch_dir, name="torchviz")
        except Exception:
            pass

        # 2) torchviz graph rooted at a scalar loss (much more informative than tensor-only)
        try:
            dump_torchviz_png(pl_module, self.example, epoch_dir, name="torchviz")
        except Exception:
            with open(epoch_dir / "_fx_error.txt", "w") as f:
                f.write(repr(e))

        # 3) numeric weight stats for reproducible comparisons
        try:
            stats = collect_weight_stats(pl_module.model)
            with open(epoch_dir / "weight_stats.json", "w") as f:
                json.dump(stats, f, indent=2)
        except Exception:
            pass
