from src.utils.logger import get_logger, set_logger_level, log_execution_time
from src.utils.common import argparse, os, copy, Path, pt, datetime, json, pd, np, time, read_from_json, save_to_json
from src.utils.device import SetupDevice
from src.train_model import ModelTrainer
from src.preprocess_data import Normalize, PrepareDataset, PrepareDatasets
from src.utils.config import merge_config

import itertools


# ==================================================
# CONTRIBUTION START: HyperparameterSearch, main()
# Contributor: Wen Yu
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

    parser.add_argument('--gpu-device-list', dest='gpu_device_list', type=int, nargs='+', default=[0], 
                help='Specify which GPU(s) to use; Provide multiple GPUs as space-separated values, e.g., "0 1" | default: 0 (if CUDA is available)')
    parser.add_argument('--gpu-memory-fraction', dest='gpu_memory_fraction', type=float, default=0.5,
                help='Fraction of GPU memory to allocate per process | default: 0.5 (if CUDA is available)')
    parser.add_argument('--cpu-device-only', dest="cpu_device_only", action='store_true',
                help="PyTorch device can only use default CPU; Overrides other device options | default: Off")
    parser.add_argument('--num-cores', dest="num_cores", type=int, default=1, 
                help="Number of cpu cores (tasks) to run in parallel. If multi-threading is enabled, max threads is set to (num_tasks * 2) | default: 1")
    parser.add_argument('--deterministic',  dest='deterministic', action='store_true', 
                help="Enables deterministic algorithms, trades reproducibility for faster torch ops and sometimes causes errors | default: True")

    # --- tuning-specific arguments ---
    parser.add_argument('--tune-optimizer-steps', dest="tune_optimizer_steps", type=int, default=500,
                help="Maximum optimizer steps per tuning trial | default: 500")
    parser.add_argument('--validate-every-steps', dest="validate_every_steps", type=int, default=50,
                help="Validate every N optimizer steps during tuning | default: 50")
    parser.add_argument('--tune-patience', dest="tune_patience", type=int, default=5,
                help="Early stopping/pruning patience in validation checks during tuning | default: 5")
    parser.add_argument('--config-file', dest="config_file", type=str,
                help="Input config file path to override base tuning config with | default: None")
    
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
    if args.tune_optimizer_steps < 1:
        raise ValueError("[--tune-optimizer-steps] must be >= 1")
    if args.validate_every_steps < 1:
        raise ValueError("[--validate-every-steps] must be >= 1")

    max_validation_checks = max(1, int(np.ceil(args.tune_optimizer_steps / args.validate_every_steps)))

    if not (1 <= args.tune_patience <= max_validation_checks):
        raise ValueError(
            f"[--tune-patience] must be between 1 and {max_validation_checks} (max_validation_checks) "
            f"for tune_optimizer_steps={args.tune_optimizer_steps} "
            f"and validate_every_steps={args.validate_every_steps}"
        )
    
    if hasattr(args, "config_file") and args.config_file is not None:
        config_file = Path(args.config_file)
        if not config_file.is_file():
            raise FileNotFoundError(f"[--config-file] '{config_file}' does not exist")

    """
    please validate the arguments you added so it breaks now instead of later
    """

    return args


# --------------------------------------------------
# Helpers
# Contributor: Wen Yu
# --------------------------------------------------

def _grid_combinations(param_grid) -> list:
    """Return explicit stage combos or Cartesian combos."""
    if isinstance(param_grid, list):
        if not all(isinstance(combo, dict) for combo in param_grid):
            raise ValueError("param_grid list must contain only dictionaries")
        return param_grid

    if not isinstance(param_grid, dict):
        raise TypeError("param_grid must be a dict or list[dict]")

    keys, values = list(param_grid.keys()), list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _get_neighbors(best_val: float, candidates: list) -> list:
    """Return [prev, best, next] neighbors around best_val in candidates."""
    candidates = sorted(set(candidates) | {best_val})
    idx = candidates.index(best_val)
    return candidates[max(0, idx - 1) : idx + 2]


# ==================================================
# CONTRIBUTION START: log_neighbors, linear_neighbors, squared_neighbors, get_stage_grids
# Contributor: Leslie Horace
# ==================================================

def log_neighbors(best_val, sigs=(1.0, 3.0, 5.0), lower=None, upper=None) -> list[float]:
    if best_val <= 0:
        raise ValueError("best_val must be > 0")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("lower must be <= upper")

    base = 10.0 ** np.floor(np.log10(best_val))
    values = np.asarray([float(sig) * base for sig in sigs], dtype=float)
    values = np.clip(values, a_min=lower, a_max=upper)

    return sorted(set(values.tolist()))


def linear_neighbors(best_val, offsets=(-0.1, 0.0, 0.1), lower=None, upper=None) -> list:
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("lower must be <= upper")

    values = np.asarray([best_val + offset for offset in offsets], dtype=float)
    values = np.clip(values, a_min=lower, a_max=upper)

    if isinstance(best_val, (int, np.integer)) and not isinstance(best_val, bool):
        values = np.rint(values).astype(int)

    return sorted(set(values.tolist()))


def base2_neighbors(best_val: int, lower=8, upper=256) -> list[int]:
    if best_val < 1:
        raise ValueError("best_val must be >= 1")

    if lower is not None and upper is not None and lower > upper:
        raise ValueError("lower must be <= upper")
    
    best_base = int(2 ** np.round(np.log2(best_val)))
    values = np.asarray([best_base // 2, best_base, best_base * 2], dtype=int)
    values = np.clip(values, a_min=lower, a_max=upper).astype(int)

    return sorted(set(values.tolist()))


def get_stage_grids(best_config: dict) -> dict:

    best_lr = best_config.get("learn_rate", 3e-4)
    fine_lr = log_neighbors(best_lr, sigs=(1.0, 3.0, 5.0), lower=1e-5, upper=1e-3)

    best_batch_size = best_config.get("batch_size", 64)
    fine_batch_size = base2_neighbors(best_batch_size, lower=16, upper=128)

    best_weight_decay = best_config.get("weight_decay", 0.0)
    fine_weight_decay = (
        [0.0, 1e-6, 1e-5]
        if best_weight_decay == 0.0
        else log_neighbors(best_weight_decay, sigs=(1.0, 3.0, 5.0), lower=1e-8, upper=1e-3)
    )

    if best_config.get("debug", False):
        return {
            1: {"name": "stage1_learn_rate", "grid": {"learn_rate": [1e-4, 1e-3]}},
            2: {"name": "stage2_batch_size", "grid": {"batch_size": [64, 128]}},
        }

    return {
        1: {
            "name": "stage1_learning",
            "grid": {
                "learn_rate": [1e-4, 3e-4, 5e-4],
                "batch_size": [32, 64, 128],
            },
        },
        2: {
            "name": "stage2_lr_scheduler",
            "grid": {
                "lr_scheduler": ["plateau"],
                "lr_scheduler_patience": [1, 2, 3],
                "lr_scheduler_factor": [0.1, 0.2, 0.3]
            },
        },
        3: {
            "name": "stage3_capacity",
            "grid": [
                {"hidden_layers": 2, "hidden_dims": 128, "latent_dims": 32},
                {"hidden_layers": 2, "hidden_dims": 128, "latent_dims": 64},
                {"hidden_layers": 2, "hidden_dims": 128, "latent_dims": 128},

                {"hidden_layers": 2, "hidden_dims": 256, "latent_dims": 64},
                {"hidden_layers": 2, "hidden_dims": 256, "latent_dims": 128},
                {"hidden_layers": 2, "hidden_dims": 256, "latent_dims": 256},

                {"hidden_layers": 3, "hidden_dims": 128, "latent_dims": 32},
                {"hidden_layers": 3, "hidden_dims": 128, "latent_dims": 64},
                {"hidden_layers": 3, "hidden_dims": 128, "latent_dims": 128},

                {"hidden_layers": 3, "hidden_dims": 256, "latent_dims": 64},
                {"hidden_layers": 3, "hidden_dims": 256, "latent_dims": 128},
                {"hidden_layers": 3, "hidden_dims": 256, "latent_dims": 256},
            ],
        },
        4: {
            "name": "stage4_network",
            "grid": {
                "conv_kernel": [3, 5],
                "activation_function": ["relu", "leaky"]
            },
        },
        5: {
            "name": "stage5_ssim",
            "grid": {
                "ssim_loss_weight": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            },
        },
        6: {
            "name": "stage6_optimizer",
            "grid": {
                "weight_decay": [0.0, 1e-5, 5e-5, 1e-4],
                "optim_beta1": [0.85, 0.9],
                "optim_beta2": [0.99, 0.999],
            },
        },
        7: {
            "name": "stage7_best_fine",
            "grid": {
                "learn_rate": fine_lr,
                "batch_size": fine_batch_size,
                "weight_decay": fine_weight_decay
            },
        },
    }

# --------------------------------------------------
# HyperparameterSearch
# Contributor: Wen Yu (HyperparameterSearch, stage 1,2 and 4, entry run and run stage)
# --------------------------------------------------

class HyperparameterSearch(ModelTrainer):
    def __init__(
        self,
        config,
        input_folder,
        output_folder,
        device,
        tune_optimizer_steps=800,
        validate_every_steps=50,
        tune_patience=10,
    ):
        """Tunes the model with staged grid search
        To save time, we prune (early stop) trials when the loss is not any better
        than the current best at the same number of optimizer steps (gradient updates)

        After all stages, best_overall_config.json is saved and can be passed
        directly into train_model.py for full training.
        """

        super().__init__(config, input_folder, output_folder, device, make_subdirs=False)

        self.tune_optimizer_steps = int(tune_optimizer_steps)
        self.validate_every_steps = int(validate_every_steps)
        self.tune_patience = int(tune_patience)
        self.random_seed = pt.initial_seed()
        
        self.base_config = dict(self.config)
        self.best_stage_config = dict(self.config)

        self.best_tuned_params = {}
        self.best_overall_config = dict(self.base_config)
        self.best_overall_loss = float("inf")
        self.best_overall_optimizer_step = -1
        
        self.tuning_dir = Path(output_folder)
        self.tuning_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_trainer(self)
        SetupDevice.free_memory()
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
            make_subdirs=True
        )

    def save_best_so_far(self, stage_name: str = ""):
        """Save best_stage_config to disk after each stage."""
        path = self.tuning_dir / f"best_config_after_{stage_name}.json"
        save_to_json(path, self.best_stage_config)
        self.logger.info(f"[TUNING] Saved best config after {stage_name} -> {path}")

    def update_best_configs(self, new_params: dict, stage_loss: float, stage_optimizer_step: int):
        """Carry best settings forward; also update global best if improved."""
        self.best_stage_config.update(new_params)

        self.best_tuned_params.update(new_params)
        self.best_overall_config = dict(self.base_config)
        self.best_overall_config.update(self.best_tuned_params)

        if stage_loss < self.best_overall_loss:
            self.best_overall_loss = stage_loss
            self.best_overall_optimizer_step = int(stage_optimizer_step)
            self.logger.info(
                f"[TUNING] New global best loss = {stage_loss:.8f} "
                f"at optimizer_step = {self.best_overall_optimizer_step}"
            )


    def override_best_(self, new_params: dict, new_loss: float, new_optimizer_step: int):
        """Hard-override best configs and loss, e.g. when resuming."""
        self.best_stage_config.update(new_params)
        self.best_tuned_params.update(new_params)
        self.best_overall_config = dict(self.base_config)
        self.best_overall_config.update(self.best_tuned_params)
        self.best_overall_loss = float(new_loss)
        self.best_overall_optimizer_step = int(new_optimizer_step)

    def load_best_config(self, path: str):
        """Load a previously saved best_config JSON to resume tuning."""
        cfg = read_from_json(path)
        self.best_stage_config.update(cfg)
        self.best_tuned_params = {
            key: value
            for key, value in cfg.items()
            if key in self.base_config and self.base_config[key] != value
        }
        self.best_overall_config = dict(self.base_config)
        self.best_overall_config.update(self.best_tuned_params)
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
        cfg["num_epochs"] = max(1, int(self.base_config.get("num_epochs", 50)))
        cfg["earlystop_patience"] = self.tune_patience
        cfg.update(overrides)
        return cfg

    # ==================================================
    # CONTRIBUTION START: sanity logging and trial pruning
    # Contributor: Leslie Horace
    # ==================================================
    def _stage_csv_path(self, stage_name: str) -> Path:
        return self.tuning_dir / f"{stage_name}_results.csv"

    def _resume_state_path(self) -> Path:
        return self.tuning_dir / "tuning_state.json"

    def _get_hyper_params(self, params: dict) -> str:
        return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)

    def _read_stage_results(self, stage_name: str) -> list[dict]:
        csv_path = self._stage_csv_path(stage_name)
        if not csv_path.is_file():
            return []

        df = pd.read_csv(csv_path).fillna("")
        return df.to_dict(orient="records")

    def save_resume_state(self, last_completed_stage_id: int, last_completed_stage_name: str):
        state = {
            "last_completed_stage_id": last_completed_stage_id,
            "last_completed_stage_name": last_completed_stage_name,
            "best_stage_config": self.best_stage_config,
            "best_tuned_params": self.best_tuned_params,
            "best_overall_config": self.best_overall_config,
            "best_overall_loss": self.best_overall_loss,
            "best_overall_optimizer_step": self.best_overall_optimizer_step,
        }
        save_to_json(self._resume_state_path(), state)
        self.logger.info(f"[TUNING] Saved resume state -> {self._resume_state_path()}")


    def load_resume_state(self) -> dict | None:
        path = self._resume_state_path()
        if not path.is_file():
            return None

        state = read_from_json(path)

        self.best_stage_config = dict(state.get("best_stage_config", self.best_stage_config))
        self.best_tuned_params = dict(state.get("best_tuned_params", self.best_tuned_params))
        self.best_overall_config = dict(state.get("best_overall_config", self.best_overall_config))
        self.best_overall_loss = float(state.get("best_overall_loss", self.best_overall_loss))

        self.best_overall_optimizer_step = int(
            state.get(
                "best_overall_optimizer_step",
                state.get("best_overall_epoch", self.best_overall_optimizer_step),
            )
        )

        self.logger.info(
            f"[TUNING] Loaded resume state <- {path} "
            f"last_stage={state.get('last_completed_stage_id', 'none')} "
            f"best_loss={self.best_overall_loss:.8f} "
            f"best_optimizer_step={self.best_overall_optimizer_step}"
        )
        return state

    def _existing_stage_state(self, stage_name: str):
        rows = self._read_stage_results(stage_name)

        completed = {}
        best_loss = float("inf")
        best_params = {}
        best_trial_id = ""
        best_optimizer_step = 0

        for row in rows:
            hyper_params = row.get("hyper_params", "")
            status = str(row.get("status", "")).strip().lower()

            if pd.notna(hyper_params) and hyper_params != "" and status in {"complete", "ok", "pruned"}:
                completed[str(hyper_params)] = row

            loss_value = row.get("trial_best_loss", row.get("loss_value", "inf"))

            try:
                loss = float(loss_value)
            except (TypeError, ValueError):
                loss = float("inf")

            if status in {"complete", "ok"} and np.isfinite(loss) and loss < best_loss:
                best_loss = loss
                best_params = json.loads(str(hyper_params)) if pd.notna(hyper_params) and hyper_params != "" else {}
                best_trial_id = str(row.get("trial_id", ""))

                try:
                    best_optimizer_step = int(row.get("trial_best_step", row.get("optimizer_step", 0)))
                except (TypeError, ValueError):
                    best_optimizer_step = 0

        return completed, best_loss, best_params, best_trial_id, best_optimizer_step
    
    def _append_stage_result(self, stage_name: str, row: dict):
        csv_path = self._stage_csv_path(stage_name)
        fieldnames = [
            # trial meta
            "datetime",
            "stage_id",
            "stage_name",
            "trial_id",
            "status",

            # trial result
            "trial_best_loss",
            "trial_best_step",
            "trial_seconds",
            "trial_peak_ram_mb",
            "trial_peak_vram_mb",

            # best so far stage/trial  
            "best_so_far",


            # reproducibility / debugging
            "hyper_params",
            "config_path",
            "error_message",
        ]

        row_df = pd.DataFrame([row], columns=fieldnames)

        if csv_path.exists():
            existing_df = pd.read_csv(csv_path)
            for col in fieldnames:
                if col not in existing_df.columns:
                    existing_df[col] = ""
            combined_df = pd.concat([existing_df[fieldnames], row_df], ignore_index=True)
        else:
            combined_df = row_df

        combined_df.to_csv(csv_path, index=False)

    def _cleanup_trainer(self, trainer):
        if trainer is None:
            return

        try:
            prepare_datasets = getattr(trainer, "prepare_datasets", None)
            if prepare_datasets is not None and hasattr(prepare_datasets, "close"):
                try:
                    prepare_datasets.close()
                except Exception:
                    pass

            if hasattr(trainer, "mae_model"):
                try:
                    trainer.mae_model.to("cpu")
                except Exception:
                    pass

            for attr in (
                "mae_model",
                "optimizer",
                "scheduler",
                "criterion",
                "best_model_state",
                "prepare_datasets",
                "train_loader",
                "valid_loader",
                "test_loader",
                "transform",
                "history",
            ):
                if hasattr(trainer, attr):
                    delattr(trainer, attr)

        finally:
            SetupDevice.free_memory()

    def save_stage_grids(self, stage_defs: dict, file_name: str = "resolved_tuning_stages.json"):
        manifest = {
            "total_trials": 0,
            "stages": []
        }

        for stage_id in sorted(stage_defs.keys()):
            stage = stage_defs[stage_id]
            stage_name = stage["name"]
            grid = stage["grid"]
            combos = _grid_combinations(grid)

            manifest["total_trials"] += len(combos)
            manifest["stages"].append({
                "stage_id": int(stage_id),
                "stage_name": stage_name,
                "num_trials": len(combos),
                "grid": grid,
                "expanded_trials": combos,
            })

        path = self.tuning_dir / file_name
        save_to_json(path, manifest)

        self.logger.info(
            f"[TUNING] Saved stage grid manifest -> {path} "
            f"total_trials={manifest['total_trials']}"
        )

        return path

    def _run_trial_with_pruning(
        self,
        trainer: ModelTrainer,
        current_best_loss: float,
        current_best_optimizer_step: int,
    ):

        trainer.reset_trial_memory_peak()
        trainer.mae_model = trainer.mae_model.to(trainer.device)
        trainer.update_trial_memory_peak()
        num_epochs = trainer.config["num_epochs"]
        total_batches = len(trainer.train_loader)

        if total_batches < 1:
            raise ValueError("train_loader has no batches; cannot tune.")

        total_possible_optimizer_steps = total_batches * num_epochs
        min_optimizer_steps = total_batches

        optimizer_step_budget = min(
            total_possible_optimizer_steps,
            max(min_optimizer_steps, int(self.tune_optimizer_steps)),
        )

        validate_every_steps = max(1, int(self.validate_every_steps))

        train_iter = iter(enumerate(trainer.train_loader, start=1))

        optimizer_steps_done = 0
        epoch = 1
        validation_checks = 0

        best_loss = float("inf")
        best_optimizer_step = 0
        not_improved_count = 0
        pruned = False

        if np.isfinite(current_best_loss) and current_best_optimizer_step > 0:
            prune_after_step = min(
                optimizer_step_budget,
                current_best_optimizer_step + self.tune_patience * validate_every_steps,
            )
        else:
            prune_after_step = None

        while epoch <= num_epochs and optimizer_steps_done < optimizer_step_budget:
            remaining_steps = optimizer_step_budget - optimizer_steps_done
            steps_this_round = min(validate_every_steps, remaining_steps)

            train_metrics, train_iter, epochs_completed = trainer.train_steps(
                trainer.mae_model,
                train_iter=train_iter,
                max_optimizer_steps=steps_this_round,
            )

            optimizer_steps_done += int(train_metrics["optimizer_steps"])
            epoch = min(num_epochs, epoch + epochs_completed)
            validation_checks += 1

            epoch_progress = round(optimizer_steps_done / max(1, total_batches), 3)

            valid_metrics = trainer.evaluate(
                trainer.mae_model,
                is_validation=True,
                epoch=epoch_progress,
                optimizer_step=optimizer_steps_done
            )

            valid_loss = float(valid_metrics["objective_loss"])
            trainer.scheduler_step(valid_loss)
            trainer.update_trial_memory_peak()

            if valid_loss < (best_loss - trainer.min_delta):
                best_loss = valid_loss
                best_optimizer_step = optimizer_steps_done
                not_improved_count = 0
            else:
                not_improved_count += 1

            self.logger.info(
                f"[TUNE] check={validation_checks} "
                f"optimizer_step={optimizer_steps_done}/{optimizer_step_budget} "
                f"valid_loss={valid_loss:.8f} "
                f"best_loss={best_loss:.8f} "
                f"current_best_loss={current_best_loss:.8f}"
            )

            if (
                prune_after_step is not None
                and optimizer_steps_done >= prune_after_step
                and best_loss >= current_best_loss
            ):
                pruned = True
                self.logger.info(
                    f"[TUNE] Pruned trial at optimizer_step={optimizer_steps_done}. "
                    f"best_loss={best_loss:.8f} did not beat current_best_loss={current_best_loss:.8f}"
                )
                break

            if trainer.earlystop and not_improved_count >= trainer.patience:
                self.logger.info(
                    f"[TUNE] Early stopped trial at optimizer_step={optimizer_steps_done}. "
                    f"best_loss={best_loss:.8f}"
                )
                break

        trainer.evaluate(
            trainer.mae_model,
            is_validation=True,
            epoch="trial_end",
            optimizer_step=optimizer_steps_done,
        )

        memory_stats = trainer.log_trial_memory("trial_end")

        return best_loss, best_optimizer_step, memory_stats, pruned

    # ==================================================
    # CONTRIBUTION END: sanity logging and trial pruning
    # ==================================================

    # --------------------------------------------------
    # run_trials: generic grid search loop with state tracking 
    # Contributor: Wen Yu, Leslie Horace (Tested + Extended)
    # --------------------------------------------------

    def run_stage(self, stage_id: int, param_grid: dict, stage_name: str) -> tuple:
        combos = _grid_combinations(param_grid)
        completed, best_loss, best_params, best_trial_id, best_so_far_optimizer_step = self._existing_stage_state(stage_name)

        if all(self._get_hyper_params(params) in completed for params in combos):
            self.logger.info(
                f"[{stage_name}] all {len(combos)} trials already completed; "
                f"best_trial={best_trial_id}, "
                f"best_loss={best_loss:.8f}, "
                f"best_optimizer_step={best_so_far_optimizer_step}"
            )
            return best_params, best_loss, best_trial_id, best_so_far_optimizer_step

        running_best_loss = self.best_overall_loss
        running_best_optimizer_step = self.best_overall_optimizer_step
        best_so_far_stage = "previous"
        best_so_far_trial = ""

        if np.isfinite(best_loss) and (
            not np.isfinite(running_best_loss) or best_loss < running_best_loss
        ):
            running_best_loss = best_loss
            running_best_optimizer_step = best_so_far_optimizer_step
            best_so_far_stage = stage_name
            best_so_far_trial = best_trial_id

        for i, params in enumerate(combos, 1):
            trial_id = f"{i:04d}"
            hyper_params = self._get_hyper_params(params)

            if hyper_params in completed:
                continue

            cfg = self._scalar_config(params)
            label = f"{stage_name}_trial_{trial_id}"

            loss = float("inf")
            best_optimizer_step = 0
            status = "failed"
            error_message = ""
            config_path = ""
            memory_stats = {
                "trial_peak_ram_mb": 0.0,
                "trial_peak_vram_mb": 0.0,
            }
            start_time = time.perf_counter()

            for attempt in range(1, 3):
                trainer = None

                try:
                    if attempt == 2:
                        self.logger.warning(
                            f"[{stage_name}] retrying trial={trial_id} once after failure"
                        )

                    trainer = self.make_trainer(cfg, label)
                    config_path = str(trainer.output_folder / "resolved_train_config.json")

                    loss, best_optimizer_step, memory_stats, pruned = self._run_trial_with_pruning(
                        trainer=trainer,
                        current_best_loss=running_best_loss,
                        current_best_optimizer_step=running_best_optimizer_step,
                    )

                    status = "pruned" if pruned else "complete"
                    error_message = ""
                    break

                except Exception as exc:
                    loss = float("inf")
                    best_optimizer_step = 0
                    status = "failed"
                    error_message = str(exc)

                    self.logger.warning(
                        f"[{stage_name}] trial={trial_id} attempt={attempt}/2 failed: {error_message}"
                    )

                finally:
                    self._cleanup_trainer(trainer)
                    trainer = None

            if np.isfinite(loss) and loss < best_loss:
                best_loss = loss
                best_params = params
                best_trial_id = trial_id
                best_so_far_optimizer_step = best_optimizer_step

            if np.isfinite(loss) and loss < running_best_loss:
                running_best_loss = loss
                running_best_optimizer_step = best_optimizer_step
                best_so_far_stage = stage_name
                best_so_far_trial = trial_id

            best_so_far = {
                "stage": best_so_far_stage,
                "trial": best_so_far_trial,
                "loss": f"{running_best_loss:.8f}" if np.isfinite(running_best_loss) else "inf",
                "step": running_best_optimizer_step,
            }

            self._append_stage_result(
                stage_name,
                {
                    # trial meta
                    "datetime": datetime.now().isoformat(),
                    "stage_id": stage_id,
                    "stage_name": stage_name,
                    "trial_id": trial_id,
                    "status": status,

                    # trial result
                    "trial_best_loss": f"{loss:.8f}" if np.isfinite(loss) else "inf",
                    "trial_best_step": best_optimizer_step,
                    "trial_seconds": f"{time.perf_counter() - start_time:.2f}",
                    "trial_peak_ram_mb": (
                        f"{memory_stats['trial_peak_ram_mb']:.1f}"
                        if "trial_peak_ram_mb" in memory_stats else ""
                    ),
                    "trial_peak_vram_mb": (
                        f"{memory_stats['trial_peak_vram_mb']:.1f}"
                        if "trial_peak_vram_mb" in memory_stats else ""
                    ),

                    # best so far
                    "best_so_far": json.dumps(best_so_far, sort_keys=True),

                    # reproducibility / debugging
                    "hyper_params": hyper_params,
                    "config_path": config_path,
                    "error_message": error_message,
                },
            )

            self.logger.info(
                f"[{stage_name}] trial={trial_id} status={status} "
                f"trial_best_loss={loss:.8f} "
                f"trial_best_step={best_optimizer_step} "
                f"best_so_far={best_so_far} "
                f"trial_seconds={time.perf_counter() - start_time:.2f} "
                f"peak_ram_mb={memory_stats.get('trial_peak_ram_mb', '')} "
                f"peak_vram_mb={memory_stats.get('trial_peak_vram_mb', '')}"
            )

        return best_params, best_loss, best_trial_id, best_so_far_optimizer_step
    
    # --------------------------------------------------
    # run: top-level entry point with auto state tracking
    # Contributor: Wen Yu, Leslie Horace (Tested + Extended)
    # --------------------------------------------------
        
    def tune_model(self):
        state = self.load_resume_state()
        stage_defs = get_stage_grids(self.best_stage_config)
        stage_ids = [s for s in sorted(stage_defs.keys())]

        self.save_stage_grids(stage_defs, "resolved_tuning_stages.json")
        
        if state is not None:
            last_completed = state.get("last_completed_stage_id")
            if isinstance(last_completed, int):
                stage_ids = [s for s in stage_ids if s > last_completed]

        for stage_id in stage_ids:
            stage_defs = get_stage_grids(self.best_stage_config)
            stage_name = stage_defs[stage_id]["name"]
            param_grid = stage_defs[stage_id]["grid"]

            best_params, best_loss, best_trial_id, best_optimizer_step = self.run_stage(
                stage_id,
                param_grid,
                stage_name,
            )

            self.logger.info(
                f"[Stage {stage_id}] name={stage_name} "
                f"best_trial={best_trial_id} "
                f"best_loss={best_loss:.8f} "
                f"best_optimizer_step={best_optimizer_step} "
                f"best_params={best_params}"
            )

            self.update_best_configs(best_params, best_loss, best_optimizer_step)
            self.save_best_so_far(stage_name)
            self.save_resume_state(stage_id, stage_name)
            self.logger.info(f"[Overall] best_trial_id={best_trial_id}, best_loss={self.best_overall_loss:.8f}")

        best_tuned_params_path = self.output_folder / "best_tuned_params.json"
        save_to_json(best_tuned_params_path, self.best_tuned_params)

        self.best_overall_config.update({
            "num_epochs": 500,
            "lr_scheduler": "plateau",
            "lr_scheduler_patience": 4,
            "lr_scheduler_factor": 0.2,
            "lr_scheduler_min_lr": 1e-6,

            "enable_earlystop": True,
            "earlystop_patience": 12,
            "earlystop_min_delta": 0.0,

            "log_epoch_frequency": 1,
            "log_batch_frequency": 0,
            "plot_last_batch_frequency": 500,
            "plot_last_batch_limit": 3
        })

        best_overall_path = self.output_folder / "best_overall_config.json"
        save_to_json(best_overall_path, self.best_overall_config)


        self.logger.info(
            f"[Done] best_loss={self.best_overall_loss:.8f} "
            f"saved={best_overall_path} tuned_only={best_tuned_params_path}"
        )

# ==================================================
# CONTRIBUTION END: HyperparameterSearch
# ==================================================


@log_execution_time
def main(args):
    if args.debug:
        set_logger_level(10)

    logger = get_logger()
    device = SetupDevice.setup_torch_device(
        args.num_cores,
        args.cpu_device_only,
        gpu_list=args.gpu_device_list,
        gpu_memory=args.gpu_memory_fraction,
        random_seed=args.random_seed,
        deterministic=args.deterministic
    )

    logger.info(f"Using device = {device}")

    override_config = {}
    if args.config_file is not None:
        override_config = read_from_json(args.config_file)

    base_config = merge_config(override_config)

    base_config.update({
        "debug":       args.debug,
        "num_workers": args.num_cores,
        "random_seed": args.random_seed
    })

    # assert all(isinstance(x, (tuple, list, dict)) for x in  base_config.values())
    search = HyperparameterSearch(
        config=base_config,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        device=device,
        tune_optimizer_steps=args.tune_optimizer_steps,
        validate_every_steps=args.validate_every_steps,
        tune_patience=args.tune_patience,
    )
    search.tune_model()


if __name__ == "__main__":
    from src.utils.logger import init_shared_logger
    logger = init_shared_logger(__file__, log_stdout=True, log_stderr=True)
    try:
        pt.multiprocessing.set_sharing_strategy('file_system')
        args = process_args()
        main(args)
    except Exception as e:
        logger.error(e)


# Run fake test grid to make sure it works
'''
python src/tune_model.py \
--input-folder "data/preprocessed/galaxiesml_tiny" \
--output-folder experiments/tune_debug_small \
--gpu-memory-fraction 0.9 \
--num-cores 2 \
--tune-optimizer-steps 250 \
--validate-every-steps 10 \
--tune-patience 5 \
--debug
'''

# Run actual tuning grids with your preprocessed mask 
'''
python src/tune_model.py \
--input-folder "data/preprocessed/galaxiesml_small" \
--output-folder experiments/tune_mae_<first_name>_<mask_ratio>  \
--gpu-memory-fraction 0.9
--num-cores 5 \
--tune-optimizer-steps 1000
--validate-every-steps 100
--tune-patience 5
'''