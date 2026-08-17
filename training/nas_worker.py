# training/nas_worker.py
import argparse, os, sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--gen", type=int, required=True)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--dataset", choices=["nas","full"], default="nas")
    parser.add_argument("--fp32-epochs", type=int, default=None)
    parser.add_argument("--fp16-epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

    PROJECT_ROOT = os.path.expanduser("~/projects/iride_onboard-burnscar-mapper")
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "training"))
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "pyqnas", "src"))

    import pickle, json
    import torch, torch.nn as nn
    import pytorch_lightning as pl
    from pynas.core.config import load_default_config
    from pynas.core.population import Population
    from pynas.core.generic_lightning_module import GenericLightningSegmentationNetwork
    from pynas.core.qat_utils import read_fp16aware_opts, prepare_fp16_aware
    from pynas.train.losses import FocalLoss           # ← switched from plain CrossEntropyLoss
    from datamodule import BalancedTileDataModule

    config = load_default_config()
    SAVE_DIR, TILES_ROOT = "models_traced", "processed/dataset_v1"
    INDEX_CSV = f"{TILES_ROOT}/tiles_index.csv"
    BATCH_SIZE   = config.getint("GA", "batch_size")
    MAX_PARAMS   = config.getint("GA", "max_parameters")
    MIN_PARAMS   = config.getint("GA", "min_parameters", fallback=0)
    MAX_LAYERS   = config.getint("NAS", "max_layers", fallback=5)
    NAS_SUBSET_N = config.getint("GA", "nas_subset_n", fallback=15_000)
    LR = args.lr if args.lr is not None else 10 ** config.getfloat("Search Space", "default_log_lr", fallback=-3.0)
    FP32_EPOCHS = args.fp32_epochs or config.getint("Precision", "fp32_epochs")
    FP16_EPOCHS = args.fp16_epochs or config.getint("QAT", "finetune_epochs")

    # softened weights — old_burn was nearly tied with clear's effective
    # (P x weight) pull, which is what caused tonight's collapse
    CLASS_WEIGHTS = torch.tensor([0.08, 0.22, 0.28, 0.14, 0.14, 0.14], dtype=torch.float32)

    pkl = f"{SAVE_DIR}/src/population_{args.gen}.pkl"
    if not os.path.exists(pkl): pkl = f"{SAVE_DIR}/src/population_0.pkl"
    population = pickle.load(open(pkl, "rb"))
    individual = population[args.idx]

    class LightDM:
        input_shape = (7, 256, 256); num_classes = 6

    pop = Population(n_individuals=len(population), max_layers=MAX_LAYERS, dm=LightDM(),
                     max_parameters=MAX_PARAMS, min_parameters=MIN_PARAMS, save_directory=SAVE_DIR)
    pop.population, pop.cfg, pop.generation = population, config, args.gen

    dm = BalancedTileDataModule(TILES_ROOT, INDEX_CSV, batch_size=BATCH_SIZE,
                                num_workers=args.workers,
                                nas_subset_n=(NAS_SUBSET_N if args.dataset=="nas" else None))
    dm.setup()

    gen_dir = f"{SAVE_DIR}/generation_{args.gen}/model_{args.idx}"
    os.makedirs(gen_dir, exist_ok=True)

    # Focal loss: down-weights predictions the model is already confident
    # about, so "collapse to one class, predict it confidently everywhere"
    # stops being a comfortable stable minimum the way plain weighted CE allowed.
    loss_fn = FocalLoss(alpha=1.0, gamma=2.0, weight=CLASS_WEIGHTS.cuda(), ignore_index=-1)

    TRAINER_KW = dict(accelerator="gpu", devices=1, accumulate_grad_batches=2,
                      logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, log_every_n_steps=999, callbacks=[], gradient_clip_val=1.0)

    print(f"[gpu{args.gpu}] idx={args.idx} FP32 ({FP32_EPOCHS} epochs)")
    model, valid = pop.build_model(individual.parsed_layers, task="segmentation")
    if not valid:
        json.dump({"gen":args.gen,"idx":args.idx,"fitness":0.0,"error":"invalid"},
                   open(f"{gen_dir}/metrics.json","w")); return

    LM = GenericLightningSegmentationNetwork(model=model, learning_rate=LR)
    LM.loss_fn = loss_fn
    trainer = pl.Trainer(max_epochs=FP32_EPOCHS, **TRAINER_KW)
    trainer.fit(LM, dm)
    r32 = trainer.test(LM, dm, verbose=False)[0]
    fp32_iou, fp32_fps = float(r32.get("test_iou",0)), float(r32.get("test_fps",1))
    torch.save(LM.model.state_dict(), f"{gen_dir}/model_fp32.pt")
    fp32_state = {k: v.cpu().clone() for k,v in LM.model.state_dict().items()}
    print(f"[gpu{args.gpu}] idx={args.idx} FP32 -> {fp32_iou:.4f}")

    print(f"[gpu{args.gpu}] idx={args.idx} FP16 ({FP16_EPOCHS} epochs)")
    fp16_opts = read_fp16aware_opts(config); fp16_opts.finetune_epochs = FP16_EPOCHS
    model_ft, _ = pop.build_model(individual.parsed_layers, task="segmentation")
    model_ft.load_state_dict(fp32_state)
    ctx = prepare_fp16_aware(model_ft, fp16_opts)
    LM_ft = GenericLightningSegmentationNetwork(model=model_ft, learning_rate=LR*0.1)
    LM_ft.loss_fn = loss_fn
    trainer_ft = pl.Trainer(max_epochs=FP16_EPOCHS, **TRAINER_KW)
    trainer_ft.fit(LM_ft, dm)
    r16 = trainer_ft.test(LM_ft, dm, verbose=False)[0]
    fp16_iou, fp16_fps = float(r16.get("test_iou",0)), float(r16.get("test_fps",1))
    torch.save(LM_ft.model.state_dict(), f"{gen_dir}/model_fp16aware.pt")
    ctx.close()
    print(f"[gpu{args.gpu}] idx={args.idx} FP16 -> {fp16_iou:.4f}")

    fitness = 1.0*fp16_iou + 0.2*min(fp16_fps/30.0, 1.0)
    json.dump({"gen":args.gen,"idx":args.idx,"params":individual.model_size,
              "fp32_iou":fp32_iou,"fp32_fps":fp32_fps,"fp16_iou":fp16_iou,"fp16_fps":fp16_fps,
              "fitness":fitness,"fp32_model":f"{gen_dir}/model_fp32.pt",
              "fp16_model":f"{gen_dir}/model_fp16aware.pt"},
             open(f"{gen_dir}/metrics.json","w"), indent=2)
    print(f"[gpu{args.gpu}] idx={args.idx} DONE fitness={fitness:.4f}")


if __name__ == "__main__":
    main()