# training/final_retrain.py
import argparse, os, sys, json, time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=int, required=True)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--params", type=int, required=True)
    parser.add_argument("--fp32-epochs", type=int, default=50)
    parser.add_argument("--fp16-epochs", type=int, default=15)
    parser.add_argument("--fp32-patience", type=int, default=8)
    parser.add_argument("--fp16-patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--class-weights", type=str, default=None)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--metrics-log", type=str, required=True)
    parser.add_argument("--progress-pct", type=int, default=10)
    parser.add_argument("--skip-fp32-training", action="store_true")
    args = parser.parse_args()

    PROJECT_ROOT = os.path.expanduser("~/projects/iride_onboard-burnscar-mapper")
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "training"))
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "pyqnas", "src"))

    import pickle
    import torch
    import pytorch_lightning as pl
    from pynas.core.config import load_default_config
    from pynas.core.population import Population
    from pynas.core.generic_lightning_module import GenericLightningSegmentationNetwork
    from pynas.core.qat_utils import read_fp16aware_opts, prepare_fp16_aware
    from pynas.train.losses import FocalLoss
    from datamodule import BalancedTileDataModule

    SAVE_DIR, TILES_ROOT = "models_traced", "processed/dataset_v1"
    INDEX_CSV = f"{TILES_ROOT}/tiles_index.csv"
    config = load_default_config()
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(seed=config.getint("Computation", "seed"), workers=True)

    LR = args.lr if args.lr is not None else 10 ** config.getfloat("Search Space", "default_log_lr", fallback=-3.0)
    w = [float(x) for x in args.class_weights.split(",")] if args.class_weights else [0.08,0.22,0.28,0.14,0.14,0.14]
    assert len(w) == 6
    CLASS_WEIGHTS = torch.tensor(w, dtype=torch.float32)

    class BestModelSaver(pl.Callback):
        """Saves model.state_dict() to `path` ONLY when val_iou improves on the
        best value seen so far. Unlike naively saving whatever epoch happens to
        be live when training stops, `path` always holds the best-epoch weights.
        Rank-0-only under DDP -- otherwise two ranks (with slightly different
        val_iou due to the unsynced validation-split gap) would race to write
        the same file."""
        def __init__(self, path, monitor="val_iou", mode="max"):
            self.path = path
            self.monitor = monitor
            self.mode = mode
            self.best = float("-inf") if mode == "max" else float("inf")

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking or not trainer.is_global_zero:
                return
            val = trainer.callback_metrics.get(self.monitor)
            if val is None:
                return
            val = float(val)
            improved = (val > self.best) if self.mode == "max" else (val < self.best)
            if improved:
                self.best = val
                torch.save(pl_module.model.state_dict(), self.path)
                print(f"  [BestModelSaver] New best {self.monitor}={val:.4f} -> saved {self.path}", flush=True)

    class InformativeProgress(pl.Callback):
        def __init__(self, stage_name, print_every_pct=10):
            self.stage_name = stage_name
            self.print_every_pct = max(1, print_every_pct)
            self._t0 = None
            self._last_pct = -1
            self._total = None

        def on_train_epoch_start(self, trainer, pl_module):
            if not trainer.is_global_zero: return
            self._t0 = time.time()
            self._last_pct = -1
            self._total = trainer.num_training_batches
            print(f"\n[{self.stage_name}] Epoch {trainer.current_epoch+1}/{trainer.max_epochs} "
                  f"starting ({self._total} batches)", flush=True)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if not trainer.is_global_zero or not self._total:
                return
            pct = int(100 * (batch_idx + 1) / self._total)
            is_last = (batch_idx + 1) == self._total
            if pct < self._last_pct + self.print_every_pct and not is_last:
                return
            self._last_pct = pct
            elapsed = time.time() - self._t0
            rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (self._total - (batch_idx + 1)) / rate if rate > 0 else 0
            loss = None
            try:
                loss = float(outputs["loss"]) if isinstance(outputs, dict) else float(outputs)
            except Exception:
                pass
            loss_str = f"loss={loss:.4f}  " if loss is not None else ""
            bar_len = 20
            filled = int(bar_len * (batch_idx + 1) / self._total)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"  [{self.stage_name}] Epoch {trainer.current_epoch+1}/{trainer.max_epochs} "
                  f"[{bar}] {pct:3d}%  batch {batch_idx+1}/{self._total}  "
                  f"{loss_str}elapsed={elapsed:5.0f}s  ETA={eta:5.0f}s", flush=True)

        def on_train_epoch_end(self, trainer, pl_module):
            if not trainer.is_global_zero: return
            elapsed = time.time() - self._t0
            print(f"[{self.stage_name}] Epoch {trainer.current_epoch+1}/{trainer.max_epochs} "
                  f"COMPLETE in {elapsed:.0f}s", flush=True)

        def on_validation_epoch_end(self, trainer, pl_module):
            if not trainer.is_global_zero or trainer.sanity_checking: return
            m = trainer.callback_metrics
            vi, vl = m.get("val_iou"), m.get("val_loss")
            lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else None
            print(f"[{self.stage_name}] Epoch {trainer.current_epoch+1} VALIDATION -> "
                  f"val_iou={float(vi):.4f}  val_loss={float(vl):.4f}  lr={lr:.2e}"
                  if vi is not None and vl is not None and lr is not None else
                  f"[{self.stage_name}] Epoch {trainer.current_epoch+1} VALIDATION -> (metrics pending)",
                  flush=True)

    class EpochJSONLLogger(pl.Callback):
        def __init__(self, path, stage):
            self.path = path
            self.stage = stage
        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking or not trainer.is_global_zero:
                return
            m = trainer.callback_metrics
            row = {"stage": self.stage, "epoch": trainer.current_epoch,
                  "val_iou": float(m.get("val_iou", float("nan"))),
                  "val_loss": float(m.get("val_loss", float("nan"))),
                  "lr": trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else None}
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")

    def _shape_signature(sd):
        return tuple(sorted((k, tuple(v.shape)) for k, v in sd.items()))

    class LightDM: input_shape=(7,256,256); num_classes=6
    pop = Population(n_individuals=30, max_layers=7, dm=LightDM(),
                     max_parameters=5_000_000, min_parameters=200_000, save_directory=SAVE_DIR)
    pop.cfg = config

    pkl = f"{SAVE_DIR}/src/population_{args.gen}.pkl"
    if not os.path.exists(pkl): pkl = f"{SAVE_DIR}/src/population_0.pkl"
    plist = pickle.load(open(pkl, "rb"))
    matches = [ind for ind in plist if getattr(ind, "model_size", None) == args.params]
    assert len(matches) >= 1
    if len(matches) == 1:
        individual = matches[0]
    else:
        ckpt_path = f"{SAVE_DIR}/generation_{args.gen}/model_{args.idx}/model_fp32.pt"
        ckpt_sig = _shape_signature(torch.load(ckpt_path, map_location="cpu"))
        resolved = [ind for ind in matches
                   if _shape_signature(pop.build_model(ind.parsed_layers, task="segmentation")[0].state_dict()) == ckpt_sig]
        assert len(resolved) == 1
        individual = resolved[0]
    print(f"Architecture: {individual.model_size:,} params", flush=True)

    dm = BalancedTileDataModule(TILES_ROOT, INDEX_CSV, batch_size=config.getint("GA", "batch_size"),
                                num_workers=args.num_workers, nas_subset_n=None)
    dm.setup()
    print(f"Full dataset -- train: {len(dm.train_ds):,}  val: {len(dm.val_ds):,}", flush=True)

    gen_dir = Path(SAVE_DIR) / "final_model"
    gen_dir.mkdir(parents=True, exist_ok=True)
    loss_fn = FocalLoss(alpha=1.0, gamma=2.0, weight=CLASS_WEIGHTS.cuda(), ignore_index=-1)

    def make_trainer_kw():
        # Fresh strategy object every call -- reusing one across two Trainer()
        # calls caused "accelerator set through both strategy class and
        # accelerator flag" on the second call.
        return dict(
            accelerator="gpu", devices=args.devices,
            strategy=(pl.strategies.DDPStrategy(process_group_backend="gloo")
                     if args.devices > 1 else "auto"),
            accumulate_grad_batches=1 if args.devices > 1 else 2,
            logger=False, enable_checkpointing=False,
            enable_progress_bar=False,
            log_every_n_steps=50,
            gradient_clip_val=1.0,
        )

    fp32_ckpt = gen_dir / "model_fp32.pt"
    if args.skip_fp32_training:
        assert fp32_ckpt.exists(), f"--skip-fp32-training set but {fp32_ckpt} does not exist"
        print(f"\n=== Scenario 1: SKIPPED — resuming from {fp32_ckpt} ===", flush=True)
        fp32_state = torch.load(fp32_ckpt, map_location="cpu")
        fp32_iou = None
    else:
        print("\n=== Scenario 1: FP32 ===", flush=True)
        model, valid = pop.build_model(individual.parsed_layers, task="segmentation")
        assert valid
        LM = GenericLightningSegmentationNetwork(model=model, learning_rate=LR)
        LM.loss_fn = loss_fn
        LM.scheduler_max_epochs = args.fp32_epochs
        early1  = pl.callbacks.EarlyStopping(monitor="val_iou", mode="max", patience=args.fp32_patience, min_delta=0.001, verbose=True)
        jsonl1  = EpochJSONLLogger(args.metrics_log, stage="fp32")
        prog1   = InformativeProgress("FP32", args.progress_pct)
        best1   = BestModelSaver(str(fp32_ckpt))
        trainer = pl.Trainer(max_epochs=args.fp32_epochs, callbacks=[early1, jsonl1, prog1, best1], **make_trainer_kw())
        trainer.fit(LM, dm)

        # Reload the genuinely-best epoch (BestModelSaver already wrote it to
        # disk during training) before testing/handing off to FP16-aware --
        # trainer.fit() leaves LM.model at the LAST epoch, not the best one.
        best_state = torch.load(fp32_ckpt, map_location=LM.device)
        LM.model.load_state_dict(best_state)
        res32 = trainer.test(LM, dm, verbose=False)[0]
        fp32_iou = float(res32.get("test_iou", 0.0))
        fp32_state = {k: v.cpu().clone() for k, v in LM.model.state_dict().items()}
        print(f"FP32 -> test_iou={fp32_iou:.4f} (best epoch, val_iou={best1.best:.4f})", flush=True)
        del LM, trainer, model
        torch.cuda.empty_cache()

    print("\n=== Scenario 2: FP16-aware ===", flush=True)
    fp16_opts = read_fp16aware_opts(config)
    fp16_opts.finetune_epochs = args.fp16_epochs
    model_ft, _ = pop.build_model(individual.parsed_layers, task="segmentation")
    model_ft.load_state_dict(fp32_state)
    ctx = prepare_fp16_aware(model_ft, fp16_opts)
    LM_ft = GenericLightningSegmentationNetwork(model=model_ft, learning_rate=LR * 0.1)
    LM_ft.loss_fn = loss_fn
    LM_ft.scheduler_max_epochs = args.fp16_epochs
    fp16_ckpt = gen_dir / "model_fp16aware.pt"
    early2  = pl.callbacks.EarlyStopping(monitor="val_iou", mode="max", patience=args.fp16_patience, min_delta=0.001, verbose=True)
    jsonl2  = EpochJSONLLogger(args.metrics_log, stage="fp16")
    prog2   = InformativeProgress("FP16", args.progress_pct)
    best2   = BestModelSaver(str(fp16_ckpt))
    trainer_ft = pl.Trainer(max_epochs=args.fp16_epochs, callbacks=[early2, jsonl2, prog2, best2], **make_trainer_kw())
    trainer_ft.fit(LM_ft, dm)

    best_state_16 = torch.load(fp16_ckpt, map_location=LM_ft.device)
    LM_ft.model.load_state_dict(best_state_16)
    res16 = trainer_ft.test(LM_ft, dm, verbose=False)[0]
    fp16_iou = float(res16.get("test_iou", 0.0))
    ctx.close()
    print(f"FP16-aware -> test_iou={fp16_iou:.4f} (best epoch, val_iou={best2.best:.4f})", flush=True)

    json.dump({"params": individual.model_size, "fp32_iou": fp32_iou, "fp16_iou": fp16_iou,
              "lr": LR, "class_weights": CLASS_WEIGHTS.tolist()},
             open(gen_dir / "metrics.json", "w"), indent=2)
    print("\nDONE", flush=True)

if __name__ == "__main__":
    main()
