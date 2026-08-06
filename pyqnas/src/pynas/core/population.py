import pandas as pd
import numpy as np
import pickle
import copy
from copy import deepcopy
import tqdm, os, shlex
import json
import subprocess
import tempfile
import shutil
import logging
from typing import Optional, Callable, List

from ..blocks.heads import MultiInputClassifier
from .individual import Individual
from .generic_unet import GenericUNetNetwork
from ..opt.evo import single_point_crossover, gene_mutation
from .generic_lightning_module import GenericLightningSegmentationNetwork, GenericLightningNetwork
from .callbacks_graph import GraphOnce, WeightHists, GraphSnapshot
from .viz_logging import example_input_from_dm
from .qat_utils import (
    read_qat_opts, prepare_qat, freeze_observers, convert_quantized, export_qdq_onnx,
    read_fp16aware_opts, prepare_fp16_aware, FP16AwareOpts,
)
from .config import load_default_config
from .quant import prepare_hailo_quant, read_quant_opts, freeze_all_ranges

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.multiprocessing as mp
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from IPython.display import clear_output
from pathlib import Path

# --------------------------------------------------------------------------------------
# Module-level configuration
# --------------------------------------------------------------------------------------

DOCKER_OV_IMAGE = os.environ.get("PYNAS_OV_IMAGE", "ubuntu20_ov:2022.3.1")
REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = REPO_ROOT / "models_traced"

try:
    from pytorch_lightning.loggers import WandbLogger  # noqa: F401
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

try:
    import wandb  # type: ignore
    _HAS_WANDB_LIB = True
except Exception:
    wandb = None  # type: ignore
    _HAS_WANDB_LIB = False

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


# --------------------------------------------------------------------------------------
# Precision registry
# --------------------------------------------------------------------------------------
# Maps a precision name (as written in config [Precision]) to the qat_mode used
# by _train_one_variant. fp32 is always the base: it trains first, from scratch,
# and every other precision fine-tunes from its weights.

PRECISION_QAT_MODE = {
    "fp32":        None,            # plain FP32 training (base, from scratch)
    "fp32_ft":     None,            # CONTROL: plain FP32 finetune from base weights,
                                    # same extra budget as the int8 variants — isolates
                                    # quantization effect from extra-training effect
    "fp16":        "fp16_finetune", # FP16-aware finetune (existing qat_utils path)
    "int8_torch":  "int8_torch",    # torch.ao library QAT (paper-2 baseline)
    "int8_custom": "int8_custom",   # our hand-written Hailo-scheme fake quant
}

# Precisions whose prepared models cannot go through TorchScript/ONNX export
# (fake-quant wrapper modules / torch.ao observers present).
NON_EXPORTABLE_PRECISIONS = {"int8_torch", "int8_custom"}


# --------------------------------------------------------------------------------------
# Helpers: paths & I/O
# --------------------------------------------------------------------------------------

def scenario_dir(gen: int, idx: int, scenario: str) -> Path:
    """Create (if needed) and return the path for a scenario folder."""
    base = ARTIFACTS_DIR / f"generation_{gen}" / f"model_{idx}" / "scenarios" / scenario
    (base / "pytorch").mkdir(parents=True, exist_ok=True)
    (base / "graphs").mkdir(parents=True, exist_ok=True)
    (base / "onnx").mkdir(parents=True, exist_ok=True)
    (base / "openvino_fp16").mkdir(parents=True, exist_ok=True)
    return base


def write_metrics_json(folder: Path, payload: dict) -> None:
    with open(folder / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)


def write_results_txt(folder: Path, results: dict) -> None:
    p = folder / "pytorch" / "results.txt"
    with open(p, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")


class QuietValProgress(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar


# --------------------------------------------------------------------------------------
# Population class
# --------------------------------------------------------------------------------------
class Population:
    """GA population orchestrating model build/train/test across precisions.

    Config-driven behaviour (config.ini):

      [Precision]
      train   = fp32, int8_torch, int8_custom   # which variants to TRAIN
      test    = fp32, int8_torch, int8_custom   # which trained variants to TEST on GPU
      fitness = fp32                            # which precision's GPU IoU drives the GA

      Available precisions: fp32 | fp16 | int8_torch | int8_custom
        fp32        - plain training (always runs first; base weights for the rest)
        fp16        - FP16-aware finetune (existing qat_utils implementation)
        int8_torch  - torch.ao library QAT finetune (library baseline)
        int8_custom - hand-written Hailo-scheme fake-quant finetune (ours)

      [Myriad]
      enabled = false   # ONNX export + Docker/OpenVINO eval (fp32/fp16 only)

      [QuantSim]        # scheme knobs for int8_custom (see quant/prepare.py)
    """

    def __init__(
        self,
        n_individuals: int,
        max_layers: int,
        dm,
        max_parameters: int = 400_000,
        save_directory: Optional[str] = None,
        external_logger_factory: Optional[Callable] = None,
        **kwargs,
    ):
        if not isinstance(n_individuals, int) or n_individuals <= 0:
            raise ValueError(f"n_individuals must be a positive integer, got {n_individuals}")
        if not isinstance(max_layers, int) or max_layers <= 0:
            raise ValueError(f"max_layers must be a positive integer, got {max_layers}")
        if dm is None:
            raise ValueError("Data module (dm) cannot be None")
        if not isinstance(max_parameters, int) or max_parameters <= 0:
            raise ValueError(f"max_parameters must be a positive integer, got {max_parameters}")

        self.dm = dm
        self.n_individuals = n_individuals
        self.max_layers = max_layers
        self.max_parameters = max_parameters
        self.external_logger_factory = external_logger_factory

        self.generation = 0
        self.population = []
        self.df = None

        self.save_directory = save_directory or "../models_traced"
        os.makedirs(os.path.join(self.save_directory, "src"), exist_ok=True)
        os.makedirs(os.path.join(self.save_directory, "backups"), exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = self.setup_logger()

        # Config loaded once; can be overridden by setting pop.cfg afterwards.
        self.cfg = load_default_config()

        self.logger.info(
            f"Initialized population with {n_individuals} individuals, "
            f"max_layers={max_layers}, max_parameters={max_parameters}, device={self.device}"
        )
        self.logger.info(
            f"Precisions — train: {self.precisions_train}, test: {self.precisions_test}, "
            f"fitness: {self.fitness_precision}, Myriad: {self.myriad_enabled}"
        )

    # ------------------------------------------------------------------
    # Config helpers (never raise)
    # ------------------------------------------------------------------

    def _get_flag(self, section: str, option: str, fallback: bool = False) -> bool:
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            return fallback
        try:
            return cfg.getboolean(section, option, fallback=fallback)
        except (ValueError, TypeError):
            self.logger.warning(
                f"Config [{section}] {option} invalid boolean; defaulting to {fallback}"
            )
            return fallback

    def _get_list(self, section: str, option: str, fallback: str) -> List[str]:
        cfg = getattr(self, "cfg", None)
        raw = fallback
        if cfg is not None:
            try:
                raw = cfg.get(section, option, fallback=fallback)
            except Exception:
                raw = fallback
        items = [x.strip().lower() for x in raw.split(",") if x.strip()]
        return items

    @property
    def precisions_train(self) -> List[str]:
        """Ordered precision list to train. fp32 is forced to be present & first."""
        items = self._get_list("Precision", "train", fallback="fp32")
        valid = [p for p in items if p in PRECISION_QAT_MODE]
        for p in items:
            if p not in PRECISION_QAT_MODE:
                self.logger.warning(f"Unknown precision '{p}' in [Precision] train — ignored")
        if "fp32" in valid:
            valid.remove("fp32")
        return ["fp32"] + valid

    @property
    def precisions_test(self) -> List[str]:
        items = self._get_list("Precision", "test", fallback="fp32")
        return [p for p in items if p in PRECISION_QAT_MODE]

    @property
    def fitness_precision(self) -> str:
        items = self._get_list("Precision", "fitness", fallback="fp32")
        p = items[0] if items else "fp32"
        return p if p in PRECISION_QAT_MODE else "fp32"

    @property
    def myriad_enabled(self) -> bool:
        return self._get_flag("Myriad", "enabled", fallback=False)

    # ------------------------------------------------------------------
    # Fitness function (config-driven, [Fitness] section)
    # ------------------------------------------------------------------

    def _fitness_params(self):
        cfg = self.cfg
        mode = "iou"
        try:
            mode = cfg.get("Fitness", "mode", fallback="iou").strip().lower()
        except Exception:
            pass
        def _gf(opt, fb):
            try:
                return cfg.getfloat("Fitness", opt, fallback=fb)
            except Exception:
                return fb
        return {
            "mode": mode,
            "alpha": _gf("alpha", 1.0),
            "beta": _gf("beta", 0.2),
            "lambda": _gf("lambda", 0.5),
            "fps_target": _gf("fps_target", 100.0),
        }

    def _compute_fitness(self, results_by_p: dict, test_list: List[str]):
        """Compute composite fitness from the configured [Fitness] mode.

        Modes:
          iou               : fitness = IoU(fitness_precision)
          iou_fps           : fitness = alpha*IoU + beta*min(FPS/fps_target, 1)
          iou_fps_retention : the above - lambda*(1 - IoU/IoU_fp32)
                              (relative retention penalty; requires fp32 tested,
                               silently skipped otherwise)

        Returns (fitness, iou, fps, fit_precision_used).
        """
        p = self._fitness_params()
        fit_p = self.fitness_precision
        if fit_p not in results_by_p or fit_p not in test_list:
            self.logger.warning(
                f"Fitness precision '{fit_p}' not trained+tested; falling back to fp32"
            )
            fit_p = "fp32"

        t = results_by_p.get(fit_p, {}).get("test", {})
        iou = float(t.get("test_iou", 0.0) or 0.0)
        fps = float(t.get("test_fps", 0.0) or 0.0)

        fps_norm = min(fps / p["fps_target"], 1.0) if p["fps_target"] > 0 else 0.0

        mode = p["mode"]
        if mode == "iou":
            fitness = iou
        elif mode == "iou_fps":
            fitness = p["alpha"] * iou + p["beta"] * fps_norm
        elif mode == "iou_fps_retention":
            fitness = p["alpha"] * iou + p["beta"] * fps_norm
            fp32_t = results_by_p.get("fp32", {}).get("test", {})
            iou_fp32 = float(fp32_t.get("test_iou", 0.0) or 0.0)
            if iou_fp32 > 1e-6 and "fp32" in test_list:
                retention_loss = max(0.0, 1.0 - iou / iou_fp32)
                fitness -= p["lambda"] * retention_loss
            else:
                self.logger.warning(
                    "Retention penalty skipped: fp32 not tested or IoU_fp32=0"
                )
        else:
            self.logger.warning(f"Unknown [Fitness] mode '{mode}'; using plain IoU")
            fitness = iou

        return fitness, iou, fps, fit_p

    # ------------------------------------------------------------------
    # Notebook output helpers
    # ------------------------------------------------------------------

    def _banner(self, text: str, char: str = "=") -> None:
        """Print a visually distinct banner to the notebook and the log file."""
        line = char * 78
        print(f"\n{line}\n{text}\n{line}")
        self.logger.info(text)

    # ------------------------------------------------------------------
    # W&B (config-driven, crash-proof: can never kill a run)
    # ------------------------------------------------------------------

    def _wandb_enabled(self) -> bool:
        return self._get_flag("WANDB", "enabled", fallback=False) and _HAS_WANDB_LIB

    def _init_wandb_run(self, gen: int, idx: int, scenario: str):
        """Start a W&B run for one scenario. Never raises.

        [WANDB] mode = offline  is the escape hatch when auth/network is
        flaky: runs log locally and can be uploaded later via `wandb sync`.
        """
        if not self._wandb_enabled():
            return None
        try:
            if wandb.run is not None:
                wandb.finish()
            cfg = self.cfg
            run = wandb.init(
                project=cfg.get("WANDB", "project", fallback="pynas"),
                entity=cfg.get("WANDB", "entity", fallback=None) or None,
                mode=cfg.get("WANDB", "mode", fallback="online"),
                tags=[t.strip() for t in cfg.get("WANDB", "tags", fallback="").split(",") if t.strip()],
                name=f"gen{gen}_model{idx}_{scenario}",
                group=f"generation_{gen}",
                reinit=True,
            )
            return run
        except Exception as e:
            self.logger.warning(f"W&B init failed (continuing without): {e}")
            return None

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------

    @staticmethod
    def setup_logger(log_file: str = '../logs/population.log', log_level: int = logging.DEBUG):
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        if os.path.exists(log_file):
            base, ext = os.path.splitext(log_file)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{base}_{timestamp}{ext}"

        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger

    # ------------------------------------------------------------------
    # Population lifecycle (unchanged)
    # ------------------------------------------------------------------

    def initial_poll(self) -> None:
        self.population = self.create_population()
        self._checkpoint()
        self.report_diversity()

    def report_diversity(self) -> dict:
        """Log and print population diversity statistics.

        Reports: unique architecture count, block-type usage frequency, and
        depth distribution. This is the paper's search-space coverage figure —
        run it after initial_poll and after each evolve() to track diversity
        over generations.
        """
        if not self.population:
            return {}

        from collections import Counter
        block_counter = Counter()
        depth_counter = Counter()
        archs = set()

        for ind in self.population:
            arch = getattr(ind, "architecture", None) or str(ind.parsed_layers)
            archs.add(arch)
            layers = getattr(ind, "parsed_layers", []) or []
            depth_counter[len(layers)] += 1
            for layer in layers:
                lt = layer.get("layer_type", "?") if isinstance(layer, dict) else str(layer)
                block_counter[lt] += 1

        n = len(self.population)
        n_unique = len(archs)
        total_blocks = sum(block_counter.values()) or 1

        print(f"\n  DIVERSITY — GEN {self.generation}")
        print(f"    unique architectures: {n_unique}/{n}")
        print(f"    depth distribution:   " +
              ", ".join(f"{d}L:{c}" for d, c in sorted(depth_counter.items())))
        print(f"    block usage:")
        for bt, c in block_counter.most_common():
            print(f"      {bt:<16} {c:>4}  ({100.0 * c / total_blocks:.1f}%)")
        print()

        stats = {
            "generation": self.generation,
            "unique": n_unique,
            "total": n,
            "blocks": dict(block_counter),
            "depths": dict(depth_counter),
        }
        self.logger.info(f"Diversity: {json.dumps(stats)}")
        return stats

    def create_random_individual(self, max_attempts: int = 5) -> Individual:
        for attempt in range(max_attempts):
            try:
                individual = Individual(max_layers=self.max_layers)
                if not hasattr(individual, 'parsed_layers') or not individual.parsed_layers:
                    self.logger.warning(
                        f"Created individual has invalid parsed_layers (attempt {attempt+1}/{max_attempts})"
                    )
                    continue
                self.logger.debug(
                    f"Successfully created random individual with {len(individual.parsed_layers)} layers"
                )
                return individual
            except Exception as e:
                self.logger.warning(
                    f"Failed to create random individual (attempt {attempt+1}/{max_attempts}): {str(e)}"
                )
        raise RuntimeError(f"Failed to create valid random individual after {max_attempts} attempts")

    def _sort_population(self):
        if not hasattr(self, 'population') or not self.population:
            self.logger.warning("Cannot sort population: population is empty or not initialized")
            return []
        try:
            valid_individuals = []
            invalid_count = 0
            for individual in self.population:
                if (
                    hasattr(individual, 'fitness') and
                    individual.fitness is not None and
                    not np.isnan(individual.fitness)
                ):
                    valid_individuals.append(individual)
                else:
                    invalid_count += 1

            if invalid_count > 0:
                self.logger.warning(f"Found {invalid_count} individuals with invalid fitness values")
            if not valid_individuals:
                self.logger.error("No individuals with valid fitness values found!")
                return self.population

            sorted_population = sorted(valid_individuals, key=lambda ind: ind.fitness, reverse=True)
            self.population = sorted_population

            if sorted_population:
                top_fitness = [ind.fitness for ind in sorted_population[:min(3, len(sorted_population))]]
                self.logger.info(f"Top fitness values after sorting: {top_fitness}")
            try:
                self._checkpoint()
            except Exception as e:
                print(f"Error during checkpointing after sorting: {str(e)}")
                self.logger.error(f"Failed to checkpoint after sorting: {str(e)}")
            return sorted_population
        except Exception as e:
            print(f"Population sorting failed with error: {str(e)}")
            self.logger.error(f"Population sorting failed with error: {str(e)}")
            return self.population

    def _checkpoint(self) -> bool:
        if not hasattr(self, 'population') or not self.population:
            self.logger.error("Cannot checkpoint: population is empty or not initialized")
            return False
        try:
            os.makedirs(self.save_directory, exist_ok=True)
            backup_dir = os.path.join(self.save_directory, f"backups/gen_{self.generation}")
            os.makedirs(backup_dir, exist_ok=True)

            for file_type in ["population", "df_population"]:
                src_path = f'{self.save_directory}/src/{file_type}_{self.generation}.pkl'
                if os.path.exists(src_path):
                    backup_path = (
                        f'{backup_dir}/{file_type}_{self.generation}_'
                        f'{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pkl'
                    )
                    try:
                        shutil.copy2(src_path, backup_path)
                        self.logger.debug(f"Backed up {src_path} to {backup_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to backup {src_path}: {e}")

            try:
                self._update_df()
                self.logger.debug("Updated population dataframe")
            except Exception as e:
                self.logger.error(f"Failed to update dataframe: {e}")
                return False

            save_success = True
            try:
                self.save_population()
            except Exception as e:
                self.logger.error(f"Failed to save population: {e}")
                save_success = False
            try:
                self.save_dataframe()
            except Exception as e:
                self.logger.error(f"Failed to save dataframe: {e}")
                save_success = False

            if save_success:
                self.logger.info(f"Successfully checkpointed population at generation {self.generation}")
                return True
            else:
                self.logger.warning(f"Checkpoint at generation {self.generation} was incomplete")
                return False
        except Exception as e:
            self.logger.error(f"Checkpoint failed with error: {e}")
            return False

    def check_individual(self, individual: Individual) -> bool:
        if individual is None:
            self.logger.error("Cannot check individual: received None")
            return False
        if not hasattr(individual, 'parsed_layers') or not individual.parsed_layers:
            self.logger.error("Individual is missing parsed_layers attribute or it's empty")
            return False
        try:
            model_representation, is_valid = self.build_model(individual.parsed_layers)
            if not is_valid:
                self.logger.warning("Model building failed: build_model returned is_valid=False")
                return False
            modelSize = self.evaluate_parameters(model_representation)
            individual.model_size = modelSize
            if modelSize <= 0:
                self.logger.warning(f"Invalid model size: {modelSize} (must be positive)")
                return False
            if modelSize >= self.max_parameters:
                self.logger.warning(f"Model too large: {modelSize} parameters (max: {self.max_parameters})")
                return False
            if modelSize is None:
                self.logger.warning("Model size is None")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Unexpected error checking individual: {str(e)}")
            return False

    def create_population(self, max_attempts: int = 200, timeout_seconds: int = 300):
        import time
        start_time = time.time()
        population = []
        attempts = 0
        failed_attempts = 0
        additional_attempts = 0

        with tqdm.tqdm(total=self.n_individuals, desc="Generating Population") as pbar:
            while len(population) < self.n_individuals:
                if time.time() - start_time > timeout_seconds:
                    self.logger.warning(
                        f"Population generation timed out after {timeout_seconds} seconds. "
                        f"Created {len(population)}/{self.n_individuals} individuals."
                    )
                    break
                if attempts >= max_attempts:
                    self.logger.warning(
                        f"Reached maximum attempts ({max_attempts}) for population generation. "
                        f"Created {len(population)}/{self.n_individuals} individuals."
                    )
                    break

                attempts += 1
                try:
                    candidate = self.create_random_individual()
                    if self.check_individual(candidate):
                        population.append(candidate)
                        pbar.update(1)
                        self.logger.debug(
                            f"Added individual {len(population)}/{self.n_individuals} "
                            f"(attempt {attempts}, failed: {failed_attempts})"
                        )
                    else:
                        failed_attempts += 1
                except Exception as e:
                    failed_attempts += 1
                    self.logger.warning(f"Failed to create individual on attempt {attempts}: {e}")

                if attempts % 10 == 0:
                    self.logger.info(
                        f"Population generation: {len(population)}/{self.n_individuals} created "
                        f"(attempts: {attempts}, failed: {failed_attempts})"
                    )

        original_count = len(population)
        self.logger.info(
            f"Initial population created with {original_count} individuals, removing duplicates..."
        )
        population = self.remove_duplicates(population)

        if len(population) < self.n_individuals:
            self.logger.warning(
                f"Population size after duplicate removal: {len(population)}/{self.n_individuals}"
            )
            self.logger.info(
                f"Attempting to generate additional {self.n_individuals - len(population)} unique individuals"
            )
            with tqdm.tqdm(total=self.n_individuals - len(population), desc="Filling Missing") as pbar:
                additional_attempts = 0
                fill_start_time = time.time()
                while len(population) < self.n_individuals:
                    if time.time() - fill_start_time > timeout_seconds / 2:
                        self.logger.warning("Timed out while filling population after duplicate removal")
                        break
                    if additional_attempts >= max_attempts / 2:
                        self.logger.warning("Max attempts while filling population after duplicate removal")
                        break
                    additional_attempts += 1
                    try:
                        existing_archs = set(
                            getattr(ind, 'architecture', str(ind.parsed_layers)) for ind in population
                        )
                        candidate = self.create_random_individual()
                        if self.check_individual(candidate):
                            new_arch = getattr(candidate, 'architecture', str(candidate.parsed_layers))
                            if new_arch not in existing_archs:
                                population.append(candidate)
                                existing_archs.add(new_arch)
                                pbar.update(1)
                                self.logger.debug(
                                    f"Added missing individual {len(population)}/{self.n_individuals}"
                                )
                    except Exception as e:
                        self.logger.warning(f"Failed while filling population: {e}")

        final_unique_count = len(set(getattr(ind, 'architecture', str(ind.parsed_layers)) for ind in population))
        if final_unique_count < len(population):
            self.logger.warning(
                f"Final population still contains duplicates: "
                f"{len(population) - final_unique_count} duplicates detected"
            )

        self.logger.info(
            f"Population generation completed. Created {len(population)}/{self.n_individuals} individuals "
            f"in {time.time() - start_time:.1f} seconds "
            f"(attempts: {attempts + additional_attempts}, success rate: "
            f"{len(population)/(attempts + additional_attempts):.1%})"
        )

        if len(population) < self.n_individuals:
            self.logger.error(
                f"Unable to create required population size. Created only "
                f"{len(population)}/{self.n_individuals} individuals."
            )
            if len(population) < self.n_individuals * 0.5:
                raise RuntimeError(
                    f"Failed to create a viable population. Only generated "
                    f"{len(population)}/{self.n_individuals} individuals."
                )
        return population

    def elite_models(self, k_best: int = 1):
        valid_individuals = [
            ind for ind in self.population
            if hasattr(ind, 'fitness') and ind.fitness is not None and not np.isnan(ind.fitness)
        ]
        if not valid_individuals:
            self.logger.warning("No valid individuals with fitness values found!")
            return []
        sorted_pop = self._sort_population()
        k_best = min(k_best, len(sorted_pop))
        topModels = [deepcopy(sorted_pop[i]) for i in range(k_best)]
        for i, model in enumerate(topModels):
            self.logger.info(f"Selected elite model for next generation. Idx {i} with fitness: {model.fitness}")
        return topModels

    def evolve(
        self,
        mating_pool_cutoff: float = 0.5,
        mutation_probability: float = 0.85,
        k_best: int = 1,
        n_random: int = 3,
    ) -> None:
        new_population = []
        self.generation += 1
        self.topModels = self.elite_models(k_best=k_best)

        sorted_pop = sorted(self, key=lambda individual: individual.fitness, reverse=True)
        mating_pool = sorted_pop[:int(np.floor(mating_pool_cutoff * self.n_individuals))].copy()
        assert len(mating_pool) > 0, "Mating pool is empty."

        while len(new_population) < self.n_individuals - n_random - k_best:
            try:
                parent1 = np.random.choice(mating_pool)
                parent2 = np.random.choice(mating_pool)
                assert parent1.parsed_layers != parent2.parsed_layers, "Parents are the same individual."
            except Exception as e:
                self.logger.error(f"Error selecting parents: {e}")
                continue

            children = single_point_crossover([parent1, parent2])
            mutated_children = gene_mutation(children, mutation_probability)

            for kid in mutated_children:
                kid.reset()
                if self.check_individual(kid):
                    new_population.append(kid)

        while len(new_population) < self.n_individuals - k_best:
            try:
                individual = self.create_random_individual()
                model_representation, is_valid = self.build_model(individual.parsed_layers)
                if is_valid:
                    individual.model_size = int(self.evaluate_parameters(model_representation))
                    assert individual.model_size > 0
                    assert individual.model_size < self.max_parameters
                    assert individual.model_size is not None
                    new_population.append(individual)
            except Exception as e:
                self.logger.error(f"Error encountered when evolving population: {e}")
                continue

        new_population.extend(self.topModels)
        assert len(new_population) == self.n_individuals, (
            f"Population size is {len(new_population)}, expected {self.n_individuals}"
        )
        self.population = new_population
        self.df = None
        self._checkpoint()
        self.report_diversity()

    def remove_duplicates(self, population):
        unique_architectures = set()
        updated_population = []
        for individual in population:
            arch = getattr(individual, 'architecture', None)
            if arch is None:
                arch = str(individual.parsed_layers)
            if arch not in unique_architectures:
                unique_architectures.add(arch)
                updated_population.append(individual)
            else:
                for _ in range(50):
                    new_individual = Individual(max_layers=self.max_layers)
                    new_arch = getattr(new_individual, 'architecture', None)
                    if new_arch is None:
                        new_arch = str(new_individual.parsed_layers)
                    if new_arch not in unique_architectures:
                        unique_architectures.add(new_arch)
                        updated_population.append(new_individual)
                        break
                else:
                    updated_population.append(individual)
        return updated_population

    # ------------------------------------------------------------------
    # Modeling (unchanged)
    # ------------------------------------------------------------------

    def build_model(self, parsed_layers, task: str = "segmentation"):
        def shape_tracer(self_, encoder):
            dummy_input = torch.randn(1, *self_.dm.input_shape).to(self_.device)
            with torch.no_grad():
                output = encoder(dummy_input)
            shapes = []
            if isinstance(output, (list, tuple)):
                for o in output:
                    shapes.append(tuple(o.shape[1:]))
            else:
                shapes.append(tuple(output.shape[1:]))
            self_.logger.debug(f"Shape tracer output: {shapes}")
            return shapes

        self.task = task
        if task == "segmentation":
            model = GenericUNetNetwork(
                parsed_layers,
                input_channels=self.dm.input_shape[0],
                input_height=self.dm.input_shape[1],
                input_width=self.dm.input_shape[2],
                num_classes=self.dm.num_classes,
                encoder_only=False,
            )
            valid = True
        elif task == "classification":
            encoder = GenericUNetNetwork(
                parsed_layers,
                input_channels=self.dm.input_shape[0],
                input_height=self.dm.input_shape[1],
                input_width=self.dm.input_shape[2],
                num_classes=self.dm.num_classes,
                encoder_only=True,
            )
            valid = True
            head = MultiInputClassifier(shape_tracer(self, encoder.to(self.device)), num_classes=self.dm.num_classes)
            head = head.to(self.device)
            model = nn.Sequential(encoder, head)
        else:
            raise ValueError(f"Task {task} not supported.")
        return model, valid

    def evaluate_parameters(self, model) -> int:
        return sum(p.numel() for p in model.parameters())

    # ------------------------------------------------------------------
    # DataFrame I/O
    # ------------------------------------------------------------------

    def _update_df(self) -> None:
        """Rebuild core columns from live data; preserve populated extras."""
        core_columns = ["Generation", "Layers", "Params"]
        data = []
        for individual in self.population:
            parsed_layers = json.dumps(individual.parsed_layers, default=str)
            params = getattr(individual, "model_size", None)
            data.append([self.generation, parsed_layers, params])

        new_df = pd.DataFrame(data, columns=core_columns)

        if self.df is not None:
            extra_columns = [c for c in self.df.columns if c not in core_columns]
            for c in extra_columns:
                if len(self.df) == len(new_df):
                    new_df[c] = self.df[c].values
                else:
                    new_df[c] = np.nan

        self.df = new_df

    def save_dataframe(self) -> None:
        path = f'{self.save_directory}/src/df_population_{self.generation}.pkl'
        try:
            self.df.to_pickle(path)
            self.logger.info(f"DataFrame saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving DataFrame to {path}: {e}")

    def load_dataframe(self, generation: int):
        path = f'{self.save_directory}/src/df_population_{generation}.pkl'
        try:
            return pd.read_pickle(path)
        except Exception as e:
            self.logger.error(f"Error loading DataFrame from {path}: {e}")
            return None

    def save_population(self) -> None:
        os.makedirs(os.path.join(self.save_directory, "src"), exist_ok=True)
        path = f'{self.save_directory}/src/population_{self.generation}.pkl'
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.population, f)
            self.logger.info(f"Population saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving population to {path}: {e}")

    def load_population(self, generation: int):
        path = f'{self.save_directory}/src/population_{generation}.pkl'
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading population from {path}: {e}")
            return None

    # ------------------------------------------------------------------
    # Training orchestration — precision-suite dispatcher
    # ------------------------------------------------------------------

    def train_individual(self, idx, task, epochs: int = 20, lr: float = 1e-3, batch_size: Optional[int] = None):
        """Backward-compatible entrypoint."""
        return self.run_precision_suite(idx=idx, task=task, epochs=epochs, lr=lr, batch_size=batch_size)

    # legacy name kept as an alias so old notebooks keep working
    def train_and_test_all_three(self, idx, task, epochs: int = 20, lr: float = 1e-3, batch_size: Optional[int] = None):
        return self.run_precision_suite(idx=idx, task=task, epochs=epochs, lr=lr, batch_size=batch_size)

    def train_generation(self, task='segmentation', lr=0.001, epochs=4, batch_size=32) -> None:
        """Train every individual through the configured precision suite.

        Per-individual exceptions are caught so one bad architecture never
        kills a whole generation.
        """
        self._banner(
            f"GENERATION {self.generation} — {len(self)} models | task={task} | "
            f"epochs={epochs} | train={self.precisions_train} | "
            f"test={self.precisions_test} | fitness={self.fitness_precision}",
            char="#",
        )

        for idx in range(len(self)):
            if (
                self.df is not None and
                'Fitness' in self.df.columns and
                not pd.isna(self.df.loc[idx, 'Fitness']) and
                self.df.loc[idx, 'Fitness'] != 0
            ):
                print(f"  .. GEN {self.generation} MODEL {idx}: already trained "
                      f"(fitness={self.df.loc[idx, 'Fitness']:.4f}) — skipping")
                continue
            try:
                self.run_precision_suite(idx=idx, task=task, epochs=epochs, lr=lr, batch_size=batch_size)
            except Exception as e:
                self.logger.error(
                    f"[gen {self.generation} idx {idx}] Training failed: {e}. "
                    f"Setting fitness=0 and continuing."
                )
                individual = self.population[idx]
                individual.fitness = 0.0
                individual.metric = 0.0
                individual.fps = 0.0
                if self.df is not None:
                    self.df.loc[idx, "Fitness"] = 0.0
                    self.df.loc[idx, "Metric"] = 0.0
                    self.df.loc[idx, "FPS"] = 0.0
                self.save_dataframe()
                self.save_population()

    def run_precision_suite(
        self,
        idx: int,
        task: str,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: Optional[int] = None,
    ):
        """Train/test one individual across all configured precisions.

        Flow:
          1. fp32 trains from scratch (always). Tested if 'fp32' in [Precision] test.
          2. Base FP32 weights are snapshotted.
          3. Every other precision in [Precision] train fine-tunes from those
             weights (lr * 0.1, standard QAT recipe) and is tested if listed
             in [Precision] test.
          4. Fitness comes from [Precision] fitness (falls back to fp32).
          5. If [Myriad] enabled: fp32/fp16 additionally get exported + device-
             evaluated (int8 variants are not exportable through this path yet).
        """
        individual = self.population[idx]
        gen = self.generation
        train_list = self.precisions_train
        test_list = self.precisions_test
        results_by_p = {}

        self._banner(
            f"GEN {gen} | MODEL {idx}/{len(self)-1} | "
            f"params={getattr(individual, 'model_size', 0):,} | "
            f"arch={getattr(individual, 'architecture', '?')}"
        )

        base_state_dict = None
        # Per-phase budgets: the fp32 base should train toward convergence so
        # finetuned variants' extra epochs don't confound the quantization gap.
        try:
            fp32_epochs = self.cfg.getint("Precision", "fp32_epochs", fallback=epochs)
        except Exception:
            fp32_epochs = epochs
        try:
            ft_epochs = self.cfg.getint("Precision", "finetune_epochs", fallback=epochs)
        except Exception:
            ft_epochs = epochs

        for precision in train_list:
            qat_mode = PRECISION_QAT_MODE[precision]
            scen = scenario_dir(gen, idx, f"gpu_{precision}")
            do_test = precision in test_list

            variant_epochs = fp32_epochs if precision == "fp32" else ft_epochs

            self._banner(
                f"GEN {gen} | MODEL {idx} | PRECISION: {precision.upper()} "
                f"({'train+test' if do_test else 'train only'}, {variant_epochs} epochs)",
                char="-",
            )

            variant_lr = lr if precision == "fp32" else lr * 0.1
            results = self._train_one_variant(
                idx=idx,
                task=task,
                epochs=variant_epochs,
                lr=variant_lr,
                batch_size=batch_size,
                qat_mode=qat_mode,
                scenario_folder=scen,
                init_state_dict=None if precision == "fp32" else base_state_dict,
                do_test=do_test,
            )
            results_by_p[precision] = results

            if precision == "fp32":
                base_state_dict = copy.deepcopy(self.LM.model.state_dict())

            # record columns
            iou = float(results.get("test", {}).get("test_iou", 0.0) or 0.0)
            fps = float(results.get("test", {}).get("test_fps", 0.0) or 0.0)
            self.df.loc[idx, f"{precision}_train_loss"] = results.get("train", {}).get("loss")
            self.df.loc[idx, f"{precision}_val_loss"]   = results.get("val", {}).get("loss")
            self.df.loc[idx, f"{precision}_gpu_iou"]    = iou if do_test else np.nan
            self.df.loc[idx, f"{precision}_gpu_fps"]    = fps if do_test else np.nan

            if do_test:
                print(f"  >> GEN {gen} MODEL {idx} [{precision}]  IoU={iou:.4f}  FPS={fps:.1f}")
            else:
                print(f"  >> GEN {gen} MODEL {idx} [{precision}]  trained (test skipped per config)")

            # optional Myriad eval for exportable precisions
            if self.myriad_enabled and precision not in NON_EXPORTABLE_PRECISIONS:
                scen_myr = scenario_dir(gen, idx, f"myriad_{precision}")
                res_myr = self._export_and_eval_myriad(
                    idx=idx, task=task, scenario_folder=scen_myr,
                    xml_from_folder=scen_myr / "openvino_fp16",
                )
                self.df.loc[idx, f"{precision}_myriad_iou"] = res_myr["test"].get("mean_iou", 0.0)
                self.df.loc[idx, f"{precision}_myriad_fps"] = res_myr["test"].get("fps", 0.0)

        # legacy convenience columns
        fp32_res = results_by_p.get("fp32", {})
        self.df.loc[idx, "GPU_IoU"] = float(fp32_res.get("test", {}).get("test_iou", 0.0) or 0.0)
        self.df.loc[idx, "GPU_FPS"] = float(fp32_res.get("test", {}).get("test_fps", 0.0) or 0.0)

        # composite fitness from [Fitness] config
        fitness, fit_iou, fit_fps, fit_p = self._compute_fitness(results_by_p, test_list)

        individual.fitness = fitness
        individual.metric = fit_iou
        individual.fps = fit_fps
        self.df.loc[idx, "Fitness"] = fitness
        self.df.loc[idx, "Metric"]  = fit_iou
        self.df.loc[idx, "FPS"]     = fit_fps

        # per-model summary block
        print(f"\n  SUMMARY — GEN {gen} MODEL {idx}")
        for p in train_list:
            r = results_by_p.get(p, {})
            t = r.get("test", {})
            if p in test_list:
                print(f"    {p:<12} IoU={t.get('test_iou', float('nan')):.4f}  "
                      f"FPS={t.get('test_fps', float('nan')):.1f}")
            else:
                print(f"    {p:<12} (trained, not tested)")
        # quantization gap (analysis metric, printed whenever both ends exist)
        fp32_iou = float(results_by_p.get("fp32", {}).get("test", {}).get("test_iou", 0.0) or 0.0)
        if fit_p != "fp32" and fp32_iou > 0 and "fp32" in test_list:
            gap = fp32_iou - fit_iou
            self.df.loc[idx, f"gap_fp32_{fit_p}"] = gap
            print(f"    gap (fp32 - {fit_p}) = {gap:+.4f}")
        # budget-matched gap vs the fp32_ft control (the honest quantization cost)
        ft_iou = float(results_by_p.get("fp32_ft", {}).get("test", {}).get("test_iou", 0.0) or 0.0)
        if fit_p not in ("fp32", "fp32_ft") and ft_iou > 0 and "fp32_ft" in test_list:
            gap_bm = ft_iou - fit_iou
            self.df.loc[idx, f"gap_fp32ft_{fit_p}"] = gap_bm
            print(f"    gap budget-matched (fp32_ft - {fit_p}) = {gap_bm:+.4f}")
        print(f"    fitness [{self._fitness_params()['mode']} on {fit_p}] = {fitness:.4f}\n")

        self.save_dataframe()
        self.save_population()
        self._checkpoint()
        return results_by_p

    # ------------------------------------------------------------------
    # Training / export subroutines
    # ------------------------------------------------------------------

    def _make_logger(self, run_dir: Path, gen: int, idx: int, scenario_name: str):
        active_logger = None
        if callable(self.external_logger_factory):
            try:
                active_logger = self.external_logger_factory(run_dir, gen, idx, scenario_name)
            except TypeError:
                try:
                    active_logger = self.external_logger_factory(run_dir, gen, idx)
                except Exception as e:
                    self.logger.warning(f"external_logger_factory failed: {e}")
            except Exception as e:
                self.logger.warning(f"external_logger_factory failed: {e}")

        if active_logger is None:
            kind = "csv"
            try:
                kind = self.cfg.get("Logging", "fallback_logger", fallback="csv").strip().lower()
            except Exception:
                pass
            if kind == "tensorboard":
                active_logger = TensorBoardLogger(save_dir=str(run_dir), name="tb")
            elif kind == "none":
                active_logger = False  # Lightning accepts False = no logging
            else:
                try:
                    active_logger = CSVLogger(save_dir=str(run_dir), name="metrics")
                except Exception:
                    active_logger = False
        return active_logger

    def _train_one_variant(
        self,
        idx: int,
        task: str,
        epochs: int,
        lr: float,
        batch_size: Optional[int],
        qat_mode: Optional[str],
        scenario_folder: Path,
        init_state_dict: Optional[dict] = None,
        do_test: bool = True,
    ):
        """Train (and optionally GPU-test) one model variant.

        qat_mode:
          None             : pure FP32 training
          "fp16_finetune"  : FP16-aware finetune from init_state_dict
          "fp16_scratch"   : FP16-aware from random init
          "int8_torch"     : torch.ao library QAT finetune (baseline)
          "int8_custom"    : hand-written Hailo-scheme fake-quant finetune (ours)
        """
        gen = self.generation
        idx_dir = ARTIFACTS_DIR / f"generation_{gen}" / f"model_{idx}"
        idx_dir.mkdir(parents=True, exist_ok=True)

        # Config-driven W&B: one run per scenario, never fatal on failure.
        self._init_wandb_run(gen, idx, scenario_folder.name)

        active_logger = self._make_logger(idx_dir, gen, idx, scenario_folder.name)

        model, _ = self.build_model(self.population[idx].parsed_layers, task=task)

        if init_state_dict is not None:
            missing, unexpected = model.load_state_dict(init_state_dict, strict=False)
            if missing or unexpected:
                self.logger.warning(
                    f"init_state_dict load: missing={missing}, unexpected={unexpected}"
                )

        if task == "segmentation":
            LM = GenericLightningSegmentationNetwork(model=model, learning_rate=lr)
        elif task == "classification":
            LM = GenericLightningNetwork(model=model, learning_rate=lr, num_classes=self.dm.num_classes)
        else:
            raise ValueError(f"Task {task} not supported.")

        model_device = "cuda" if torch.cuda.is_available() else "cpu"
        LM.to(model_device)

        # [QAT] section is only consumed by the fp16 precision path.
        qat = read_qat_opts(self.cfg) if qat_mode in ("fp16_finetune", "fp16_scratch") else None

        example = example_input_from_dm(self.dm, device=model_device)
        # NOTE: fp32_ft shares qat_mode=None with fp32 — distinguished by
        # scenario folder name, not snapshot tag.
        snapshot_tag = {
            None: "fp32",
            "fp16_finetune": "fp16aware_finetune",
            "fp16_scratch": "fp16aware_scratch",
            "int8_torch": "int8_torch_qat",
            "int8_custom": "int8_custom_hailo_sim",
        }.get(qat_mode, "unknown")

        try:
            if hasattr(active_logger, "experiment") and active_logger.experiment is not None:
                exp = active_logger.experiment
                if hasattr(exp, "config"):
                    exp.config.update({
                        "generation": gen,
                        "model_idx": idx,
                        "task": task,
                        "epochs": epochs,
                        "lr": lr,
                        "batch_size": (batch_size or getattr(self.dm, "batch_size", None)),
                        "scenario": scenario_folder.name,
                        "qat_mode": qat_mode,
                    }, allow_val_change=True)
                if hasattr(active_logger, "watch"):
                    active_logger.watch(LM, log="all", log_freq=100)
        except Exception as e:
            self.logger.warning(f"Logger attach failed (non-fatal): {e}")

        graph_cb = GraphSnapshot(
            out_dir=scenario_folder / "graphs",
            example_input=example,
            every_n_epochs=1,
            tag=snapshot_tag,
        )
        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=False),
            GraphOnce(example_input=example),
            WeightHists(every_n_epochs=1),
            graph_cb,
            QuietValProgress(),
        ]
        common_trainer_kwargs = dict(
            accelerator="gpu",
            devices=1,
            callbacks=callbacks,
            logger=active_logger,
            enable_progress_bar=True,
        )

        if batch_size is not None:
            self.dm.batch_size = batch_size

        # ----- training flow -----
        if qat_mode in ("fp16_finetune", "fp16_scratch"):
            fp16_opts = FP16AwareOpts(
                enabled=True,
                stochastic=getattr(qat, "fp16_stochastic", False),
                round_weights=getattr(qat, "fp16_round_weights", True),
                act_clip=getattr(qat, "fp16_act_clip", None),
                tag=getattr(qat, "fp16_tag", "fp16aware"),
                finetune_epochs=qat.finetune_epochs,
            )
            fp16_ctx = prepare_fp16_aware(
                LM.model, fp16_opts, target_types=(torch.nn.Conv2d, torch.nn.Linear)
            )
            max_epochs = getattr(qat, "finetune_epochs", 5) if qat_mode == "fp16_finetune" else epochs
            trainer_fp16 = pl.Trainer(max_epochs=max_epochs, **common_trainer_kwargs)
            trainer_fp16.fit(LM, datamodule=self.dm)
            fp16_ctx.close()

        elif qat_mode == "int8_torch":
            # torch.ao library QAT — the paper-2 baseline. Eager mode: fuse
            # nothing (library default behavior on custom nets), attach the
            # default QAT qconfig, swap supported modules for fake-quant ones.
            import torch.ao.quantization as tq
            backend = self.cfg.get(
                "TorchQAT", "backend",
                fallback=self.cfg.get("QAT", "backend", fallback="qnnpack"),
            )
            try:
                torch.backends.quantized.engine = backend
            except Exception as e:
                self.logger.warning(f"Could not set quantized engine '{backend}': {e}")
            LM.model.train()
            LM.model.qconfig = tq.get_default_qat_qconfig(backend)
            tq.prepare_qat(LM.model, inplace=True)
            self.logger.info(f"[int8_torch] torch.ao prepare_qat applied (backend={backend})")
            trainer_q = pl.Trainer(max_epochs=epochs, **common_trainer_kwargs)
            trainer_q.fit(LM, datamodule=self.dm)
            # fake-quant modules stay installed: the GPU test below measures
            # the simulated-quantized model, same as int8_custom.

        elif qat_mode == "int8_custom":
            # Hand-written Hailo-scheme fake quantization — ours.
            qopts = read_quant_opts(self.cfg)
            qctx = prepare_hailo_quant(LM.model, qopts)
            self.logger.info(
                f"[int8_custom] Installed hand-written quant on {qctx.num_quantized} modules "
                f"(W{qopts.weight_bits}A{qopts.act_bits}, per_channel={qopts.weight_per_channel})"
            )
            trainer_q = pl.Trainer(max_epochs=epochs, **common_trainer_kwargs)
            trainer_q.fit(LM, datamodule=self.dm)
            # wrappers stay installed for the quantized GPU test.

        else:
            trainer = pl.Trainer(max_epochs=epochs, **common_trainer_kwargs)
            trainer.fit(LM, datamodule=self.dm)

        # ----- GPU test (optional per config) -----
        if do_test:
            test_trainer = pl.Trainer(
                accelerator="gpu", devices=1, logger=active_logger, enable_progress_bar=False
            )
            results = test_trainer.test(LM, self.dm)
            res_gpu = results[0] if isinstance(results, list) else results
        else:
            results = []
            res_gpu = {}
            self.logger.info(f"[{scenario_folder.name}] GPU test skipped per [Precision] test config")

        gpu_iou = float(res_gpu.get("test_iou", 0.0))
        lat = res_gpu.get("test_latency_ms", res_gpu.get("test_latency", 0.0))
        gpu_latency_ms = float(lat)
        gpu_fps_logged = res_gpu.get("test_fps", None)
        gpu_fps = float(gpu_fps_logged) if gpu_fps_logged is not None else (
            1000.0 / gpu_latency_ms if gpu_latency_ms > 0 else 0.0
        )

        self.idx = idx
        self.LM = LM
        self.results = results

        if do_test:
            write_results_txt(scenario_folder, res_gpu)

        # ----- artifact saving -----
        int8_mode = qat_mode in ("int8_torch", "int8_custom")
        if not int8_mode:
            self.save_model(
                LM,
                save_torchscript=True,
                ts_save_path=scenario_folder / "pytorch" / "model_and_architecture.pt",
                save_standard=True,
                std_save_path=scenario_folder / "pytorch" / "model.pth",
                save_myriad=self.myriad_enabled,
                openvino_save_path=scenario_folder / "openvino_fp16",
                onnx_override_path=None,
                export_model=None,
            )
        else:
            # Quantization wrappers/observers present: TorchScript/ONNX export
            # is not supported through save_model. Save a plain state_dict
            # (contains original float weights + observer state).
            arch_code = getattr(self.population[idx], "architecture", None)
            save_dict = {"state_dict": LM.model.state_dict()}
            if arch_code is not None:
                save_dict["architecture_code"] = arch_code
            torch.save(save_dict, str(scenario_folder / "pytorch" / "model.pth"))
            self.logger.info(f"[{qat_mode}] Saved state_dict (export skipped for quantized model)")

        payload = {
            "scenario": scenario_folder.name,
            "train": {"loss": float(getattr(LM, "last_train_loss", float("nan")))},
            "val":   {"loss": float(getattr(LM, "last_val_loss",  float("nan")))},
            "test":  {"test_iou": gpu_iou, "test_latency_ms": gpu_latency_ms, "test_fps": gpu_fps},
            "extras": {"classes": self.dm.num_classes, "tested": do_test},
        }
        write_metrics_json(scenario_folder, payload)

        try:
            if wandb is not None and wandb.run is not None and do_test:
                scen_prefix = scenario_folder.name
                wandb.log({
                    "gpu_iou": gpu_iou,
                    "gpu_fps": gpu_fps,
                    f"{scen_prefix}_gpu_iou": gpu_iou,
                    f"{scen_prefix}_gpu_fps": gpu_fps,
                })
        except Exception as e:
            self.logger.warning(f"W&B logging for GPU metrics failed (non-fatal): {e}")

        return {"train": payload["train"], "val": payload["val"], "test": payload["test"]}

    def _export_and_eval_myriad(self, idx: int, task: str, scenario_folder: Path, xml_from_folder: Path):
        """ONNX+IR export and Docker-based Myriad eval (fp32/fp16 models only)."""
        self.save_model(
            self.LM,
            save_torchscript=False,
            save_standard=False,
            save_myriad=True,
            openvino_save_path=scenario_folder / "openvino_fp16",
            onnx_override_path=None,
        )
        scen_name = scenario_folder.name
        self.logger.info(
            f"[Myriad] Starting eval for gen={self.generation}, idx={idx}, scenario={scen_name}"
        )
        try:
            xml_path_host = scenario_folder / "openvino_fp16" / f"temp_model_{self.idx}.xml"
            if not xml_path_host.exists():
                xmls = list((scenario_folder / "openvino_fp16").glob("*.xml"))
                assert len(xmls) > 0, f"No IR xml found in {scenario_folder / 'openvino_fp16'}"
                xml_path_host = xmls[0]

            xml_rel = xml_path_host.resolve().relative_to(REPO_ROOT).as_posix()
            inner_eval = (
                'bash -lc '
                f"'python3 -m pip show openvino >/dev/null 2>&1 || "
                'python3 -m pip install --no-cache-dir \"openvino==2022.3.0\"; '
                f"python3 scripts/eval_myriad.py "
                f"--xml \"/work/{xml_rel}\" "
                '--images "/work/data/Phisat2Simulation/Test/numpy_images/*.npy" '
                '--masks  "/work/data/Phisat2Simulation/Test/numpy_masks/*.npy" '
                f"--classes {self.dm.num_classes} "
                '--max_batches 32 '
                "--device MYRIAD'"
            )
            cmd = (
                'docker run --rm --privileged -u 0 '
                '--device /dev/bus/usb:/dev/bus/usb '
                '-v /dev:/dev '
                '-e OPENVINO_CONF_IGNORE=YES '
                f'-v "{REPO_ROOT}":/work -w /work '
                f'{DOCKER_OV_IMAGE} {inner_eval}'
            )
            proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
            if proc.returncode != 0:
                print("[DOCKER EVAL STDOUT]\n", proc.stdout)
                print("[DOCKER EVAL STDERR]\n", proc.stderr)
                raise RuntimeError(f"Myriad eval failed with code {proc.returncode}")

            last = proc.stdout.strip().splitlines()[-1]
            metrics = json.loads(last)
            mean_iou = float(metrics.get("mean_iou", 0.0))
            fps = float(metrics.get("fps", 0.0))

            payload = {
                "scenario": scen_name,
                "train": {"loss": None, "iou": 0.0},
                "val": {"loss": None, "iou": 0.0},
                "test": {"mean_iou": mean_iou, "fps": fps},
            }
            write_metrics_json(scenario_folder, payload)
            self.logger.info(f"[Myriad] Eval done: mean_iou={mean_iou:.4f}, fps={fps:.2f}")

            if wandb is not None and getattr(wandb, "run", None) is not None:
                try:
                    wandb.log({
                        "myriad_iou": mean_iou, "myriad_fps": fps,
                        f"{scen_name}_myriad_iou": mean_iou,
                        f"{scen_name}_myriad_fps": fps,
                    })
                except Exception as e:
                    self.logger.warning(f"W&B logging for Myriad metrics failed (non-fatal): {e}")
            return payload

        except Exception as e:
            self.logger.error(f"[Myriad Eval in Docker] Failed: {e}")
            gpu_iou = float(self.df.loc[self.idx, "GPU_IoU"]) if "GPU_IoU" in self.df.columns else 0.0
            gpu_fps = float(self.df.loc[self.idx, "GPU_FPS"]) if "GPU_FPS" in self.df.columns else 0.0
            payload = {
                "scenario": scen_name,
                "train": {"loss": None, "iou": 0.0},
                "val": {"loss": None, "iou": 0.0},
                "test": {"mean_iou": gpu_iou, "fps": gpu_fps},
                "extras": {"note": "fallback to GPU metrics due to Myriad eval failure"},
            }
            write_metrics_json(scenario_folder, payload)
            if wandb is not None and getattr(wandb, "run", None) is not None:
                try:
                    wandb.log({
                        f"{scen_name}_myriad_iou_fallback_gpu": gpu_iou,
                        f"{scen_name}_myriad_fps_fallback_gpu": gpu_fps,
                    })
                except Exception:
                    pass
            return payload

    # ------------------------------------------------------------------
    # Export (fp32/fp16 models only — refuses quantization-prepared models)
    # ------------------------------------------------------------------

    def save_model(
        self, LM,
        save_torchscript: bool = True,
        ts_save_path: Optional[Path] = None,
        save_standard: bool = True,
        std_save_path: Optional[Path] = None,
        save_myriad: bool = False,
        openvino_save_path: Optional[Path] = None,
        onnx_override_path: Optional[Path] = None,
        export_model=None,
    ) -> None:
        model_for_save = export_model if export_model is not None else LM.model
        model_for_save = copy.deepcopy(model_for_save).eval()

        def _looks_like_qat(m: torch.nn.Module) -> bool:
            # torch.ao observers OR our hand-written wrappers
            from .quant.qmodules import QuantConv2d, QuantLinear
            for mod in m.modules():
                if hasattr(mod, "activation_post_process"):
                    return True
                if isinstance(mod, (QuantConv2d, QuantLinear)):
                    return True
            return False

        if export_model is None and _looks_like_qat(model_for_save):
            raise RuntimeError(
                "Refusing to export a quantization-prepared model. "
                "Pass a clean float model via export_model=... ."
            )

        gen = self.generation
        gen_dir = ARTIFACTS_DIR / f"generation_{gen}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        if ts_save_path is None:
            ts_save_path = gen_dir / f"model_and_architecture_{self.idx}.pt"
        if std_save_path is None:
            std_save_path = gen_dir / f"model_{self.idx}.pth"
        if openvino_save_path is None:
            openvino_save_path = gen_dir / f"openvino_model_{self.idx}"
        openvino_save_path = Path(openvino_save_path)
        openvino_save_path.mkdir(parents=True, exist_ok=True)

        self.ts_save_path = str(ts_save_path)
        self.std_save_path = str(std_save_path)

        results_txt = openvino_save_path.parent / f"results_model_{self.idx}.txt"
        if hasattr(self, "results") and self.results:
            try:
                with open(results_txt, "w") as f:
                    res0 = self.results[0] if isinstance(self.results, list) else self.results
                    f.write("Test Results:\n")
                    for key, value in res0.items():
                        f.write(f"{key}: {value}\n")
            except Exception:
                pass

        input_shape = self.dm.input_shape
        if len(input_shape) == 3:
            input_shape = (1,) + input_shape
        shape_str = f"[{input_shape[0]},{input_shape[1]},{input_shape[2]},{input_shape[3]}]"

        if save_torchscript:
            ts_target = model_for_save
            ts_device = (
                next(ts_target.parameters()).device
                if any(p.requires_grad or p.is_leaf for p in ts_target.parameters())
                else torch.device("cpu")
            )
            if not list(ts_target.parameters()):
                ts_device = torch.device("cpu")
            example_ts = torch.randn(*input_shape, device=ts_device)
            ts_target.eval()
            with torch.inference_mode():
                traced_model = torch.jit.trace(ts_target, example_ts)
            traced_model.save(str(ts_save_path))
            print(f"Scripted (TorchScript) model saved at {ts_save_path}")

        if save_standard:
            arch_code = getattr(self.population[self.idx], "architecture", None)
            save_dict = {"state_dict": model_for_save.state_dict()}
            if arch_code is not None:
                save_dict["architecture_code"] = arch_code
            torch.save(save_dict, str(std_save_path))
            print(f"Standard model saved at {std_save_path}")

        if save_myriad:
            onnx_path_host = openvino_save_path / f"temp_model_{self.idx}.onnx"
            onnx_path_host.parent.mkdir(parents=True, exist_ok=True)

            model_for_onnx = copy.deepcopy(model_for_save).cpu().eval()
            dummy_cpu = torch.randn(*input_shape, device="cpu")
            with torch.inference_mode():
                torch.onnx.export(
                    model_for_onnx, dummy_cpu, str(onnx_path_host),
                    opset_version=13,
                    input_names=["input"],
                    output_names=["logits"],
                    training=torch.onnx.TrainingMode.EVAL,
                )
            print(f"[OpenVINO] ONNX saved at {onnx_path_host}")

            if onnx_override_path is not None:
                try:
                    int8_dir_host = openvino_save_path.parent / f"openvino_int8_model_{self.idx}"
                    int8_dir_host.mkdir(parents=True, exist_ok=True)
                    onnx_qdq_rel = Path(onnx_override_path).relative_to(REPO_ROOT).as_posix()
                    int8_rel = int8_dir_host.relative_to(REPO_ROOT).as_posix()
                    inner_mo_int8 = (
                        'bash -lc \'set -euo pipefail; '
                        'python3 -m pip install --no-cache-dir '
                        '"openvino==2022.3.1" "openvino-dev==2022.3.1" "openvino-telemetry==2022.1.0" && '
                        'python3 -m openvino.tools.mo '
                        f'--input_model "/work/{onnx_qdq_rel}" '
                        f'--output_dir  "/work/{int8_rel}" '
                        f'--input_shape "{shape_str}"\''
                    )
                    cmd_int8 = (
                        'docker run --rm -u 0 -e OPENVINO_CONF_IGNORE=YES '
                        f'-v "{os.path.expanduser("~")}/.cache/pip":/root/.cache/pip '
                        f'-v "{REPO_ROOT}":/work -w /work '
                        f'{DOCKER_OV_IMAGE} {inner_mo_int8}'
                    )
                    print(f"[OpenVINO][INT8] Using image: {DOCKER_OV_IMAGE}")
                    out = subprocess.check_output(cmd_int8, shell=True, text=True, stderr=subprocess.STDOUT)
                    print("[OpenVINO][INT8] Model Optimizer completed.")
                    print(out)
                except subprocess.CalledProcessError as e:
                    print("[ERROR][INT8] MO failed. Full log follows:\n" + e.output)
                except Exception as e:
                    print(f"[ERROR][INT8] Unexpected INT8 export error: {e}")

            onnx_rel = onnx_path_host.relative_to(REPO_ROOT).as_posix()
            output_dir_cont = f"/work/{openvino_save_path.relative_to(REPO_ROOT).as_posix()}"
            inner_mo = (
                'bash -lc \'set -euo pipefail; '
                'python3 -m pip install --no-cache-dir '
                '"openvino==2022.3.1" "openvino-dev==2022.3.1" "openvino-telemetry==2022.1.0" && '
                'python3 -m openvino.tools.mo '
                f'--input_model "/work/{onnx_rel}" '
                f'--output_dir  "{output_dir_cont}" '
                f'--input_shape "{shape_str}" '
                '--data_type FP16\''
            )
            cmd = (
                'docker run --rm -u 0 -e OPENVINO_CONF_IGNORE=YES '
                f'-v "{os.path.expanduser("~")}/.cache/pip":/root/.cache/pip '
                f'-v "{REPO_ROOT}":/work -w /work '
                f'{DOCKER_OV_IMAGE} {inner_mo}'
            )
            print(f"[OpenVINO] Using image: {DOCKER_OV_IMAGE}")
            try:
                out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                print("[OpenVINO] Model Optimizer completed.")
                print(out)
            except subprocess.CalledProcessError as e:
                print("[ERROR] MO failed. Full log follows:\n" + e.output)
                raise

    # ------------------------------------------------------------------
    # Python protocol
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        return self.population[index]

    def __len__(self):
        return len(self.population)

    def __iter__(self):
        return iter(self.population)