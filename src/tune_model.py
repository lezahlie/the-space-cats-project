from src.utils.logger import get_logger, set_logger_level, log_execution_time
from src.utils.common import argparse, os, copy, Path, pt, time, np, read_from_json, save_to_json, validate_tensor
from src.utils.device import SetupDevice
from train_model import ModelTrainer
from preprocess_data import Normalize, PrepareDataset, PrepareDatasets

import itertools
import random
import h5py
import numpy as np
 
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
 
# ==================================================
# CONTRIBUTION START: HyperparameterSearch, main()
# Contributor: Wen Yu (Helpers, internal helpers and HyperparameterSearch, stage 1-2, entry run and run stage)
# ==================================================
 
def process_args():
    parser = argparse.ArgumentParser(description="Tune Model Executable", formatter_class= argparse.RawTextHelpFormatter)
    parser.add_argument('--debug', '-d', dest='debug', action='store_true', 
                help="Enables debug option and verbose printing | default: Off")
    parser.add_argument('--random-seed', dest='random_seed', type=int, default=42,
                help="Random seed for selecting samples | default: None")
    parser.add_argument('--input-folder', dest="input_folder", type=str, required=True, 
                help="Input path/to/directory where the preprocessed datasets are saved | required")
    parser.add_argument('--output-folder', dest="output_folder", type=str, required=True, 
                help="Output path/to/directory to save experiment results to | required")
    parser.add_argument('--config-file', dest="config_file", type=str, required=True, 
                help="Input config file file path | required")
    parser.add_argument('--gpu-device-list', dest='gpu_device_list', type=int, nargs='+', default=[0], 
                help='Specify which GPU(s) to use; Provide multiple GPUs as space-separated values, e.g., "0 1" | default: 0 (if CUDA is available)')
    parser.add_argument('--gpu-memory-fraction', dest='gpu_memory_fraction', type=float, default=0.5,
                help='Fraction of GPU memory to allocate per process | default: 0.5 (if CUDA is available)')
    parser.add_argument('--cpu-device-only', dest="cpu_device_only", action='store_true',
                help="PyTorch device can only use default CPU; Overrides other device options | default: Off")
    parser.add_argument('--num-cores', dest="num_cores", type=int, default=1, 
                help="Number of cpu cores (tasks) to run in parallel. If multi-threading is enabled, max threads is set to (num_tasks * 2) | default: 1")
    
    # --- tuning-specific arguments ---
    parser.add_argument('--tune-epochs', dest="tune_epochs", type=int, default=30,
                help="Number of epochs per trial during tuning stages | default: 30")
    parser.add_argument('--final-epochs', dest="final_epochs", type=int, default=200,
                help="Number of epochs for Stage 5 full training | default: 200")
    parser.add_argument('--arch-trials', dest="arch_trials", type=int, default=30,
                help="Number of random/optuna trials for Stage 3 architecture search | default: 30")
    parser.add_argument('--use-optuna', dest="use_optuna", action='store_true',
                help="Use Optuna for Stage 3 architecture search (requires optuna) | default: Off")
    parser.add_argument('--start-stage', dest="start_stage", type=int, default=1,
                help="Stage to start from (1-5), useful for resuming | default: 1")
    parser.add_argument('--end-stage', dest="end_stage", type=int, default=5,
                help="Stage to stop after (1-5), inclusive | default: 2")
    parser.add_argument('--resume-config', dest="resume_config", type=str, default=None,
                help="Path to a best_config JSON to resume from a previous stage | default: None")

    """
    please add the remaining arguments you need to do the things
    """
    args = parser.parse_args()

    # --- validation ---
    if isinstance(args.gpu_device_list, int):
        args.gpu_device_list = [args.gpu_device_list]
    if len(args.gpu_device_list) < 1:
        raise ValueError("[--gpu-device-list] must have at least 1 device, the default gpu is '0'")
    if not (0.0 < args.gpu_memory_fraction <= 1.0):
        raise ValueError("[--gpu-memory-fraction] must be in the range (0.0, 1.0]")
    if not (1 <= args.num_cores < os.cpu_count()):
        raise ValueError(f"[--num-cores] must be an INT between [1, {os.cpu_count() - 1}]")
    if not Path(args.input_folder).is_dir():
        raise FileNotFoundError(f"[--input-folder] '{args.input_folder}' does not exist")
    if not (0 < len(str(args.output_folder)) < 256):
        raise ValueError(f"[--output-folder] path is invalid")
    if not (1 <= args.tune_epochs):
        raise ValueError("[--tune-epochs] must be >= 1")
    if not (1 <= args.final_epochs):
        raise ValueError("[--final-epochs] must be >= 1")
    if not (1 <= args.arch_trials):
        raise ValueError("[--arch-trials] must be >= 1")
    if not (1 <= args.start_stage <= 5):
        raise ValueError("[--start-stage] must be between 1 and 5")
    if args.use_optuna and not OPTUNA_AVAILABLE:
        raise ImportError("[--use-optuna] requires optuna: pip install optuna")
    if args.resume_config is not None and not Path(args.resume_config).is_file():
        raise FileNotFoundError(f"[--resume-config] '{args.resume_config}' does not exist")

    """
    please validate the arguments you added so it breaks now instead of later
    """

    return args


# --------------------------------------------------
# Helpers
# Contributor: Wen Yu (Grid Search, Random Search, Log Neighbors)
# --------------------------------------------------
 
def _grid_combinations(param_grid: dict) -> list:
    """All combinations of a param_grid dict."""
    keys, values = list(param_grid.keys()), list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
 
 
def _random_sample(param_grid: dict, n: int, seed: int = 42) -> list:
    """n random combinations (without replacement where possible)."""
    rng = random.Random(seed)
    all_combos = _grid_combinations(param_grid)
    return rng.sample(all_combos, min(n, len(all_combos)))
 
 
def _log_neighbors(best_val: float, candidates: list) -> list:
    """Return [prev, best, next] neighbors around best_val in candidates."""
    candidates = sorted(set(candidates) | {best_val})
    idx = candidates.index(best_val)
    return candidates[max(0, idx - 1) : idx + 2]
 
 


# --------------------------------------------------
# Stage definitions
# Each entry: (stage_name, param_grid_keys_or_callable)
# Stage 3 is handled separately (random/optuna, not plain grid search)
# --------------------------------------------------
 
def _get_stage_grids(full_grid: dict, best_lr: float) -> dict:
    """
    Returns the param_grid for each stage as a dict keyed by stage number.
    Stage 3 is None here — handled separately via random/optuna.
    """
    fine_lr = _log_neighbors(best_lr, list(full_grid["learn_rate"]))
    return {
        1: {
            "ssim_loss_weight": full_grid["ssim_loss_weight"],
        },
        2: {
            "learn_rate":   full_grid["learn_rate"],
            "weight_decay": full_grid["weight_decay"],
            "optim_beta1":  full_grid["optim_beta1"],
            "optim_beta2":  full_grid["optim_beta2"],
            "batch_size":   full_grid["batch_size"],
        },
        3: {   # architecture — used only by random/optuna, not run_stage()
            "hidden_layers":       full_grid["hidden_layers"],
            "hidden_dims":         full_grid["hidden_dims"],
            "latent_dims":         full_grid["latent_dims"],
            "activation_function": full_grid["activation_function"],
            "norm_layer":          full_grid["norm_layer"],
            "conv_kernel":         full_grid["conv_kernel"],
            "ascending_channels":  full_grid["ascending_channels"],
        },
        4: {
            "learn_rate":            fine_lr,
            "lr_scheduler":          full_grid["lr_scheduler"],
            "lr_scheduler_patience": full_grid["lr_scheduler_patience"],
            "lr_scheduler_factor":   full_grid["lr_scheduler_factor"],
            "lr_scheduler_min_lr":   full_grid["lr_scheduler_min_lr"],
        },
    }
 
 
# --------------------------------------------------
# HyperparameterSearch
# Contributor: Wen Yu (HyperparameterSearch, stage 1,2 and 4, entry run and run stage)
# --------------------------------------------------

class HyperparameterSearch(ModelTrainer):
    def __init__(self, config, input_folder, output_folder, device,
                 tune_epochs, final_epochs, arch_trials, use_optuna, random_seed):
        """Tunes the model with staged grid search (+ random/Optuna for architecture).

        Stage 1: Loss function     – ssim_loss_weight
        Stage 2: Optimizer         – learn_rate, weight_decay, betas, batch_size
        Stage 3: Architecture      – hidden_layers/dims, latent_dims, activation,
                                     norm_layer, conv_kernel, ascending_channels
                                     (random search or Optuna TPE)
        Stage 4: LR Scheduler      – scheduler type + params, fine-tune lr neighbors
        Stage 5: Full Training     – train from scratch with best config + more epochs,
                                     save train/valid/test outputs to HDF5
        """
        super().__init__(config, input_folder, output_folder, device)

        self.tune_epochs  = tune_epochs
        self.final_epochs = final_epochs
        self.arch_trials  = arch_trials
        self.use_optuna   = use_optuna
        self.random_seed  = random_seed

        self.base_config         = dict(self.config)
        self.best_stage_config   = dict(self.config)
        self.best_overall_config = dict(self.config)
        self.best_overall_loss   = float("inf")

        self.tuning_dir = Path(output_folder) / "tuning"
        self.tuning_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Skeleton methods from the original design
    # Contributor: Wen Yu
    # --------------------------------------------------
    def make_trainer(self, config, trial_label: str = "") -> ModelTrainer:
        """Spin up a fresh ModelTrainer in its own sub-folder."""
        trial_output = self.tuning_dir / trial_label.replace(" ", "_").replace("/", "-")
        trial_output.mkdir(parents=True, exist_ok=True)

        return ModelTrainer(
            config=config,
            input_folder=self.input_folder,
            output_folder=trial_output,
            device=self.device,
        )

    def save_best_so_far(self, stage_name: str = ""):
        """Save best_stage_config to disk after each stage."""
        path = self.tuning_dir / f"best_config_after_{stage_name}.json"
        save_to_json(path, self.best_stage_config)
        self.logger.info(f"[TUNING] Saved best config after {stage_name} -> {path}")

    def update_best_configs(self, new_params: dict, stage_loss: float):
        """Carry best settings forward; also update global best if improved."""
        self.best_stage_config.update(new_params)
        if stage_loss < self.best_overall_loss:
            self.best_overall_loss   = stage_loss
            self.best_overall_config = dict(self.best_stage_config)
            self.logger.info(f"[TUNING] New global best loss = {stage_loss:.8f}")

    def override_best_(self, new_params: dict, new_loss: float):
        """Hard-override best configs and loss (e.g. when resuming)."""
        self.best_stage_config.update(new_params)
        self.best_overall_config.update(new_params)
        self.best_overall_loss = new_loss

    def save_model_state(self, save_model_path: str):
        """Save best model weights to disk."""
        pt.save(self.best_model_state, save_model_path)
        self.logger.info(f"[TUNING] Model state saved -> {save_model_path}")

    def load_model_state(self, load_model_path: str):
        """Load model weights from disk into mae_model."""
        state = pt.load(load_model_path, map_location=self.device)
        self.mae_model.load_state_dict(state)
        self.logger.info(f"[TUNING] Model state loaded <- {load_model_path}")

    def load_best_config(self, path: str):
        """Load a previously saved best_config JSON to resume tuning."""
        cfg = read_from_json(path, as_dict=True)
        self.best_stage_config.update(cfg)
        self.best_overall_config.update(cfg)
        self.logger.info(f"[TUNING] Resumed best config from {path}")

    # --------------------------------------------------
    # Internal helpers: build scalar config for a trial
    # Contributor: Wen Yu
    # --------------------------------------------------

    def _scalar_config(self, overrides: dict) -> dict:
        """
        Build a scalar (runnable) config from best_stage_config:
          - list values not in overrides → take first element as default
          - apply overrides on top
          - set num_epochs = tune_epochs
        """
        cfg = {k: (v[0] if isinstance(v, (list, tuple)) else v)
               for k, v in self.best_stage_config.items()}
        cfg.update(overrides)
        cfg["num_epochs"] = self.tune_epochs
        return cfg


    # --------------------------------------------------
    # run_stage: generic grid search loop (Stages 1, 2, 4)
    # Contributor: Wen Yu
    # --------------------------------------------------
 
    def run_stage(self, param_grid: dict, stage_name: str) -> tuple:
        """
        Generic stage runner: grid-search over param_grid, return
        (best_params, best_loss).

        Each combination is run via make_trainer() + train_model().
        """
        combos     = _grid_combinations(param_grid)
        best_loss  = float("inf")
        best_params = {}
        self.logger.info(f"[{stage_name}] {len(combos)} combinations to evaluate")

        for i, params in enumerate(combos, 1):
            cfg   = self._scalar_config(params)
            label = f"{stage_name}_trial_{i:04d}"
            self.logger.info(f"[{stage_name}] Trial {i}/{len(combos)} | {params}")
            try:
                trainer = self.make_trainer(cfg, label)
                trainer.train_model()
                loss = trainer.best_valid_loss
            except Exception as exc:
                self.logger.warning(f"[{stage_name}] Trial {i} failed: {exc}")
                loss = float("inf")

            self.logger.info(f"[{stage_name}] Trial {i} valid_loss = {loss:.8f}")
            if loss < best_loss:
                best_loss, best_params = loss, params

        return best_params, best_loss


    # --------------------------------------------------
    # Stage 3 only: random search or Optuna
    # --------------------------------------------------
    def _run_stage3(self, arch_grid: dict) -> tuple:
        """
        Random or Optuna search for architecture params. 
        Returns (best_params, best_loss).
        """
        return NotImplemented   

    # --------------------------------------------------
    # Stage 5: full retraining + save HDF5 outputs
    # --------------------------------------------------
    def _run_stage5(self):
        """
        Retrain from scratch with best config + final_epochs, then save outputs.
        """
        return NotImplemented  


    # --------------------------------------------------
    # Stage 5: full retraining + save HDF5 outputs
    # --------------------------------------------------
 
    def _run_stage5(self):
        """
        Retrain from scratch with best config + final_epochs, then save outputs.
        """
        return NotImplemented  

    # --------------------------------------------------
    # run: top-level entry point
    # Contributor: Wen Yu
    # --------------------------------------------------
 
    def run(self, full_grid: dict, start_stage: int = 1, end_stage: int = 5):
        """
        Run stages start_stage..end_stage (inclusive).
 
        Stages 1, 2, 4  → generic run_stage() grid search
        Stage  3        → random search or Optuna (_run_stage3)
        Stage  5        → full retrain + HDF5 output (_run_stage5)
        """
        stage_names = {1: "stage1_loss", 2: "stage2_optimizer",
                       3: "stage3_architecture", 4: "stage4_scheduler"}
 
        # build per-stage grids (Stage 4 needs best_lr from Stage 2)
        best_lr    = self.best_stage_config.get("learn_rate", full_grid["learn_rate"][0])
        stage_grids = _get_stage_grids(full_grid, best_lr)
 
        for stage in range(start_stage, min(end_stage, 4) + 1):
            self.logger.info("=" * 60)
            self.logger.info(f"STAGE {stage}: {stage_names[stage].upper()}")
            self.logger.info("=" * 60)
 
            if stage == 3:
                best_params, best_loss = self._run_stage3(stage_grids[3])
            else:
                best_params, best_loss = self.run_stage(stage_grids[stage], stage_names[stage])
 
            # recompute stage 4 grid now that we have the real best_lr from stage 2
            if stage == 2:
                best_lr     = best_params.get("learn_rate", best_lr)
                stage_grids = _get_stage_grids(full_grid, best_lr)
 
            self.logger.info(f"[Stage {stage}] Best: {best_params} | loss = {best_loss:.8f}")
            self.update_best_configs(best_params, best_loss)
            self.save_best_so_far(stage_names[stage])
 
        if start_stage <= 5 <= end_stage:
            self.logger.info("=" * 60)
            self.logger.info("STAGE 5: FULL TRAINING")
            self.logger.info("=" * 60)
            self._run_stage5()
 
        save_to_json(
            Path(self.output_folder) / "best_overall_config.json",
            self.best_overall_config,
        )
        self.logger.info(
            f"[TUNING] Complete. Global best loss = {self.best_overall_loss:.8f}"
        )
 
 
@log_execution_time
def main(args):
    if args.debug:
        set_logger_level(10)

    logger = get_logger()

    device = SetupDevice.setup_torch_device(
        args.num_cores,
        args.cpu_device_only,
        args.gpu_device_list,
        args.gpu_memory_fraction,
        args.random_seed
    )
    logger.info(f"Using device = {device}")

    full_grid = read_from_json(args.config_file)
    # build scalar base_config for ModelTrainer.__init__
    # list values → take first element; scalars → keep as-is
    base_config = {k: (v[0] if isinstance(v, (list, tuple)) else v)
                   for k, v in full_grid.items()}

    base_config.update({
        "debug":       args.debug,
        "num_workers": args.num_cores,
        "random_seed": args.random_seed,
        "device":      device,   
        "num_epochs":  args.tune_epochs,
    })

    # assert all(isinstance(x, (tuple, list, dict)) for x in  base_config.values())
    search = HyperparameterSearch(
        config        = base_config,
        input_folder  = args.input_folder,
        output_folder = args.output_folder,
        device        = device,
        tune_epochs   = args.tune_epochs,
        final_epochs  = args.final_epochs,
        arch_trials   = args.arch_trials,
        use_optuna    = args.use_optuna,
        random_seed   = args.random_seed,
    )

    if args.resume_config is not None:
        search.load_best_config(args.resume_config)
 
    search.run(full_grid, start_stage=args.start_stage, end_stage=args.end_stage)

    """
    please implement the rest, you can change whatever you want
    """

if __name__ == "__main__":
    from utils.logger import init_shared_logger
    logger = init_shared_logger(__file__, log_stdout=True, log_stderr=True)
    try:
        pt.multiprocessing.set_sharing_strategy('file_system')
        args = process_args()
        main(args)
    except Exception as e:
        logger.error(e)


# make sure it works so far:
"""

# -------------------------------------------------------
# Usage examples
# -------------------------------------------------------
#
# 1) Full run — all 5 stages:
#   python src/tune_model.py \
#       --config-file configs/tune_config.json \
#       --input-folder "data/galaxiesml_tiny" \
#       --output-folder experiments/tune_mask0.0 \
#       --tune-epochs 30 \
#       --final-epochs 300 \
#       --arch-trials 30 \
#       --num-cores 2 \
#       --gpu-memory-fraction 0.9 \
#       --debug
#
# 2) Only Stage 1 + 2 (tune loss + optimizer, no arch search or final train):
#   python src/tune_model.py \
#       --config-file configs/tune_config.json \
#       --input-folder "data/galaxiesml_tiny" \
#       --output-folder experiments/tune_mask0.0 \
#       --tune-epochs 30 \
#       --end-stage 2 \
#       --num-cores 2 \
#       --gpu-memory-fraction 0.9 \
#       --debug
#
# 3) Resume from Stage 3 after running example 2:
#   python src/tune_model.py \
#       --config-file configs/tune_config.json \
#       --input-folder "data/galaxiesml_tiny" \
#       --output-folder experiments/tune_mask0.0 \
#       --tune-epochs 30 \
#       --final-epochs 300 \
#       --arch-trials 30 \
#       --start-stage 3 \
#       --resume-config experiments/tune_mask0.0/tuning/best_config_after_stage2_optimizer.json \
#       --num-cores 2 \
#       --gpu-memory-fraction 0.9
#
# 4) Use Optuna for Stage 3:
#   python src/tune_model.py ... --use-optuna --arch-trials 50
 
"""