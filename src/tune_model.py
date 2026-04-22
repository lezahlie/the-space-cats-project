from src.utils.logger import get_logger, set_logger_level, log_execution_time
from src.utils.common import argparse, os, copy, Path, pt, datetime, json, pd, np, read_from_json, save_to_json
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
    parser.add_argument('--tune-epochs', dest="tune_epochs", type=int, default=20,
                help="Number of epochs to run per tuning trial | default: 20")
    parser.add_argument('--epoch-patience', dest="epoch_patience", type=int, default=4,
                help="Minimum number of epochs to run before pruning a trial that is not beating the current stage best | default: 4")
    
    parser.add_argument('--start-stage', dest="start_stage", type=int, default=1,
                help="Stage to start from (1-6), useful for resuming | default: 1")
    parser.add_argument('--end-stage', dest="end_stage", type=int, default=6,
                help="Stage to stop after (1-6), inclusive | default: 6")
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
    if not (1 <= args.tune_epochs):
        raise ValueError("[--tune-epochs] must be >= 1")
    if not (1 <= args.epoch_patience):
        raise ValueError("[--epoch-patience] must be >= 1")
    if not (1 <= args.start_stage <= 6):
        raise ValueError("[--start-stage] must be between 1 and 6")
    if not (1 <= args.end_stage <= 6):
        raise ValueError("[--end-stage] must be between 1 and 6")
    if args.start_stage > args.end_stage:
        raise ValueError("[--start-stage] must be <= [--end-stage]")
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
 
def _grid_combinations(param_grid: dict) -> list:
    """All combinations of a param_grid dict."""
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

    best_lr = best_config.get("learn_rate", 5e-4)
    fine_lr = log_neighbors(best_lr, sigs=(1.0, 3.0, 5.0), lower=1e-5, upper=1e-3)

    best_min_lr = best_config.get("lr_scheduler_min_lr", 1e-6)
    fine_min_lr = log_neighbors(best_min_lr, sigs=(1.0, 3.0, 5.0), lower=1e-7, upper=5e-6)

    best_batch_size = best_config.get("batch_size", 64)
    fine_batch_size = base2_neighbors(best_batch_size, lower=8, upper=256)

    best_weight_decay = best_config.get("weight_decay", 0.0)
    fine_weight_decay = [0.0, 1e-6, 1e-5] if best_weight_decay == 0.0 else log_neighbors(best_weight_decay, sigs=(1.0, 3.0, 5.0), lower=1e-8, upper=1e-3)

    if best_config.get("debug", False):
        return {
            1: {"name": "stage1_debug", "grid": {"learn_rate": [1e-4, 1e-3], "num_epochs": [1]}},
            2: {"name": "stage2_debug", "grid": {"ascending_channels": [True, False], "num_epochs": [1]}},
        }

    return {
        1: {
            "name": "stage1_scheduler_coarse",
            "grid": {
                "learn_rate": [5e-5, 1e-4, 5e-4],
                "lr_scheduler_min_lr": [1e-6, 5e-6, 1e-5],
                "lr_scheduler_patience": [0, 3, 5],
                "lr_scheduler_factor": [0.1, 0.3, 0.5],
            },
        },
        2: {
            "name": "stage2_optimizer_coarse",
            "grid": {
                "batch_size": [32, 64],
                "weight_decay": [0.0, 1e-5],
                "optim_beta1": [0.85, 0.9, 0.95],
                "optim_beta2": [0.99, 0.999, 0.9999],
            },
        },
        3: {
            "name": "stage3_capacity",
            "grid": {
                "hidden_layers": [1, 2, 3, 4],
                "hidden_dims": [32, 64, 128, 256],
                "latent_dims": [32, 64, 128, 256],
            },
        },
        4: {
            "name": "stage4_network",
            "grid": {
                "conv_kernel": [3, 5],
                "activation_function": ["relu", "leaky"],
                "ascending_channels": [True, False],
            },
        },
        5: {
            "name": "stage5_ssim_coarse",
            "grid": {
                "ssim_loss_weight": np.round(np.linspace(0.0, 1.0, num=11), 1).tolist(),
            },
        },
        6: {
            "name": "stage6_best_fine",
            "grid": {
                "learn_rate": fine_lr,
                "lr_scheduler_min_lr": fine_min_lr,
                "weight_decay": fine_weight_decay,
                "batch_size": fine_batch_size,
            },
        },
    }


# --------------------------------------------------
# HyperparameterSearch
# Contributor: Wen Yu (HyperparameterSearch, stage 1,2 and 4, entry run and run stage)
# --------------------------------------------------

class HyperparameterSearch(ModelTrainer):
    def __init__(self, config, input_folder, output_folder, device, tune_epochs=20, epoch_patience=4):
        """Tunes the model with staged grid search
        To save time, we prune (early stop) trials when the loss is not any better
        than the current best at the same number of epochs

        After all stages, best_overall_config.json is saved and can be passed
        directly into train_model.py for full training.
        """

        super().__init__(config, input_folder, output_folder, device, make_subdirs=False)

        self.tune_epochs = tune_epochs
        self.epoch_patience = epoch_patience
        self.random_seed = pt.initial_seed()
        
        self.base_config = dict(self.config)
        self.best_stage_config = dict(self.config)

        self.best_tuned_params = {}
        self.best_overall_config = dict(self.base_config)
        self.best_overall_loss = float("inf")

        self.tuning_dir = Path(output_folder)
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
            make_subdirs=True
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
            self.best_overall_loss = stage_loss
            self.best_tuned_params.update(new_params)
            self.best_overall_config = dict(self.base_config)
            self.best_overall_config.update(self.best_tuned_params)
            self.logger.info(f"[TUNING] New global best loss = {stage_loss:.8f}")

    def override_best_(self, new_params: dict, new_loss: float):
        """Hard-override best configs and loss (e.g. when resuming)."""
        self.best_stage_config.update(new_params)
        self.best_tuned_params.update(new_params)
        self.best_overall_config = dict(self.base_config)
        self.best_overall_config.update(self.best_tuned_params)
        self.best_overall_loss = new_loss

    def load_best_config(self, path: str):
        """Load a previously saved best_config JSON to resume tuning."""
        cfg = read_from_json(path, as_dict=True)
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
        cfg["num_epochs"] = self.tune_epochs
        cfg.update(overrides)
        return cfg

    # ==================================================
    # CONTRIBUTION START: sanity logging and trial pruning
    # Contributor: Leslie Horace
    # ==================================================
    def _stage_csv_path(self, stage_name: str) -> Path:
        return self.tuning_dir / f"{stage_name}_results.csv"

    def _get_hyper_params(self, params: dict) -> str:
        return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)

    def _read_stage_results(self, stage_name: str) -> list[dict]:
        csv_path = self._stage_csv_path(stage_name)
        if not csv_path.is_file():
            return []

        df = pd.read_csv(csv_path).fillna("")
        return df.to_dict(orient="records")

    def _existing_stage_state(self, stage_name: str):
        rows = self._read_stage_results(stage_name)

        completed = {}
        best_loss = float("inf")
        best_params = {}
        best_trial_id = ""
        best_so_far_epoch = 0

        for row in rows:
            hyper_params = row.get("hyper_params", "")
            status = str(row.get("status", "")).strip().lower()

            if pd.notna(hyper_params) and hyper_params != "" and status in {"complete", "ok"}:
                completed[str(hyper_params)] = row

            loss_value = row.get("loss_value", "inf")
            try:
                loss = float(loss_value)
            except (TypeError, ValueError):
                loss = float("inf")

            if status in {"complete", "ok"} and np.isfinite(loss) and loss < best_loss:
                best_loss = loss
                best_params = json.loads(str(hyper_params)) if pd.notna(hyper_params) and hyper_params != "" else {}
                best_trial_id = str(row.get("trial_id", ""))
                try:
                    best_so_far_epoch = int(row.get("best_so_far_epoch", 0))
                except (TypeError, ValueError):
                    best_so_far_epoch = 0

        return completed, best_loss, best_params, best_trial_id, best_so_far_epoch
    
    def _append_stage_result(self, stage_name: str, row: dict):
        csv_path = self._stage_csv_path(stage_name)
        fieldnames = [
            "datetime",
            "stage_name",
            "stage_id",
            "trial_id",
            "loss_value",
            "best_so_far_trial",
            "best_so_far_loss",
            "best_so_far_epoch",
            "hyper_params",
            "config_path",
            "status",
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


    def _run_trial_with_pruning(self, trainer: ModelTrainer, stage_best_loss: float):
        trainer.mae_model = trainer.mae_model.to(trainer.device)

        for epoch in range(1, trainer.config["num_epochs"] + 1):
            trainer.train(trainer.mae_model, epoch=epoch)
            valid_metrics = trainer.evaluate(trainer.mae_model, is_validation=True, epoch=epoch)

            valid_loss = float(valid_metrics["objective_loss"])
            trainer.scheduler_step(valid_loss)
            trainer.check_improvement(valid_loss, epoch)

            if epoch >= self.epoch_patience and trainer.best_valid_loss >= stage_best_loss:
                return trainer.best_valid_loss, trainer.best_model_epoch, True

            if trainer._should_earlystop():
                break

        return trainer.best_valid_loss, trainer.best_model_epoch, False

    # ==================================================
    # CONTRIBUTION END: sanity logging and trial pruning
    # ==================================================

    # --------------------------------------------------
    # run_trials: generic grid search loop (Stages 1, 2, 4)
    # Contributor: Wen Yu
    # --------------------------------------------------

    def run_stage(self, stage_id: int, param_grid: dict, stage_name: str) -> tuple:
        combos = _grid_combinations(param_grid)
        completed, best_loss, best_params, best_trial_id, best_so_far_epoch = self._existing_stage_state(stage_name)

        self.logger.info(f"[{stage_name}] stage_trials={len(combos)} already_complete={len(completed)}")

        if completed:
            self.logger.info(
                f"[{stage_name}] found_existing_complete_trials={len(completed)} "
                f"recovered_best_trial={best_trial_id or 'none'} "
                f"recovered_best_loss={best_loss:.8f}"
            )
        else:
            self.logger.info(f"[{stage_name}] found_existing_complete_trials=0")

        for i, params in enumerate(combos, 1):
            trial_id = f"{i:04d}"
            hyper_params = self._get_hyper_params(params)

            if hyper_params in completed:
                self.logger.info(f"[{stage_name}] trial={trial_id} status=skipped reason=already_complete")
                continue

            cfg = self._scalar_config(params)
            label = f"{stage_name}_trial_{trial_id}"

            config_path = ""
            status = "complete"
            error_message = ""

            try:
                trainer = self.make_trainer(cfg, label)
                config_path = str((Path(trainer.output_folder) / "resolved_train_config.json").resolve())

                loss, best_epoch, pruned = self._run_trial_with_pruning(
                    trainer=trainer,
                    stage_best_loss=best_loss,
                )
                status = "pruned" if pruned else "complete"

            except Exception as exc:
                loss = float("inf")
                best_epoch = 0
                status = "failed"
                error_message = str(exc)
                self.logger.warning(f"[{stage_name}] trial={trial_id} failed={exc}")

            if np.isfinite(loss) and loss < best_loss:
                best_loss = loss
                best_params = params
                best_trial_id = trial_id
                best_so_far_epoch = best_epoch

            self._append_stage_result(
                stage_name,
                {
                    "datetime": datetime.now().isoformat(),
                    "stage_name": stage_name,
                    "stage_id": stage_id,
                    "trial_id": trial_id,
                    "loss_value": f"{loss:.8f}" if np.isfinite(loss) else "inf",
                    "best_so_far_trial": best_trial_id,
                    "best_so_far_loss": f"{best_loss:.8f}" if np.isfinite(best_loss) else "inf",
                    "best_so_far_epoch": best_so_far_epoch,
                    "hyper_params": hyper_params,
                    "config_path": config_path,
                    "status": status,
                    "error_message": error_message,
                },
            )

            self.logger.info(
                f"[{stage_name}] trial={trial_id} status={status} "
                f"loss={loss:.8f} best_trial={best_trial_id} best_loss={best_loss:.8f}"
            )

        return best_params, best_loss, best_trial_id
    # --------------------------------------------------
    # run: top-level entry point
    # Contributor: Wen Yu
    # --------------------------------------------------
        
    def tune_model(self, start_stage: int = 1, end_stage: int | None = None):
        stage_defs = get_stage_grids(self.best_stage_config)
        max_stage = max(stage_defs.keys())

        if end_stage is None:
            end_stage = max_stage

        stage_ids = [s for s in sorted(stage_defs.keys()) if start_stage <= s <= end_stage]

        if not stage_ids:
            raise ValueError(f"No stages selected between {start_stage} and {end_stage}")

        for stage_id in stage_ids:
            stage_defs = get_stage_grids(self.best_stage_config)
            stage_name = stage_defs[stage_id]["name"]
            param_grid = stage_defs[stage_id]["grid"]

            best_params, best_loss, best_trial_id = self.run_stage(stage_id, param_grid, stage_name)
            self.logger.info(
                f"[Stage {stage_id}] name={stage_name} "
                f"best_trial={best_trial_id} best_loss={best_loss:.8f} best_params={best_params}"
            )

            self.update_best_configs(best_params, best_loss)
            self.save_best_so_far(stage_name)
            self.logger.info(f"[Overall] best_loss={self.best_overall_loss:.8f} best_config={self.best_overall_config}")

        best_overall_path = self.output_folder / "best_overall_config.json"
        save_to_json(best_overall_path, self.best_overall_config)

        best_tuned_params_path = self.output_folder / "best_tuned_params.json"
        save_to_json(best_tuned_params_path, self.best_tuned_params)

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
        "random_seed": args.random_seed,
    })

    # assert all(isinstance(x, (tuple, list, dict)) for x in  base_config.values())
    search = HyperparameterSearch(
        config=base_config,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        device=device,
        tune_epochs = args.tune_epochs,
        epoch_patience=args.epoch_patience
    )

    search.tune_model(start_stage=args.start_stage, end_stage=args.end_stage)


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
    --output-folder experiments/tune_debug_grid \
    --num-cores 2 \
    --gpu-memory-fraction 0.9 \
    --debug
'''

# Run actual tuning grids with your preprocessed mask 
'''
python src/tune_model.py \
    --input-folder "data/preprocessed/galaxiesml_medium" \
    --output-folder experiments/tune_mae_<first_name>_<mask_ratio>  \
    --num-cores 4 \
    --gpu-memory-fraction 0.9
'''