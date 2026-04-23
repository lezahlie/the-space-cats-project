"""
KNN Regressor for Redshift Prediction
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid

from src.utils.logger import get_logger, set_logger_level, log_execution_time
from src.utils.common import argparse, os, Path, np, h5py, save_to_json, save_to_yaml, load_from_yaml


SPLIT_FILES = {
    "train": "training_outputs_best.hdf5",
    "valid": "validation_outputs_best.hdf5",
    "test": "testing_outputs_best.hdf5",
}

PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 10, 15, 20, 30],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
}

SPLIT_COLORS = {"train": "steelblue", "valid": "darkorange", "test": "seagreen"}


def process_args():
    parser = argparse.ArgumentParser(
        description="KNN Regressor for Redshift Prediction Executable",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--debug', '-d', dest='debug', action='store_true',
                help="Enables debug option and verbose printing | default: Off")
    parser.add_argument('--params-file', dest="params_file", type=str, default=None,
                help=(
                    "Path to a pre-saved knn_best_params.yaml.\n"
                    "If not present, tune hyperparameters and generate yaml | default: None"
                ))
    parser.add_argument('--input-folder', dest="input_folder", type=str, required=True,
                help=(
                    "Directory containing HDF5 model output files.\n"
                    "Expected files: training_outputs_best.hdf5, validation_outputs_best.hdf5, testing_outputs_best.hdf5 | required"
                ))
    parser.add_argument('--output-folder', dest="output_folder", type=str, required=True,
                help="Output path/to/directory to save KNN results and plots | required")
    parser.add_argument('--random-seed', dest="random_seed", type=int, default=42,
                help="Random seed for reproducibility | default: 42")

    args = parser.parse_args()

    if args.params_file is not None and not os.path.isfile(args.params_file):
        raise FileNotFoundError(f"[--params-file] '{args.params_file}' does not exist")

    if not os.path.isdir(args.input_folder):
        raise FileNotFoundError(f"'{args.input_folder}' does not exist")
    for fname in SPLIT_FILES.values():
        fpath = os.path.join(args.input_folder, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"missing '{fname}' in '{args.input_folder}'")

    return args


# ==================================================
# CONTRIBUTION START: load_split, tune_knn, evaluate_knn, plot_* functions, main
# Contributor: Charlie Faber
# ==================================================

def load_split(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        X = f['z_latent_vector'][:].astype(np.float32)
        y = f['y_specz_redshift'][:].astype(np.float32)
    return X, y


def tune_knn(X_train, y_train, X_valid, y_valid, param_grid=None):
    logger = get_logger()

    if param_grid is None:
        param_grid = PARAM_GRID

    combos = list(ParameterGrid(param_grid))
    total = len(combos)
    best_mae = float('inf')
    best_params = None
    results = []

    for i, params in enumerate(combos, start=1):
        model = KNeighborsRegressor(**params)
        model.fit(X_train, y_train)
        mae = float(mean_absolute_error(y_valid, model.predict(X_valid)))
        results.append({**params, "valid_mae": mae})
        logger.info(f"[TUNE {i}/{total}] {params} => valid_mae={mae:.6f}")

        if mae < best_mae:
            best_mae = mae
            best_params = params

    logger.info(f"[TUNE] best_params={best_params}, best_valid_mae={best_mae:.6f}")
    return best_params, results


def evaluate_knn(model, X, y, split_name):
    y_pred = model.predict(X)
    metrics = {
        "split": split_name,
        "mae": float(mean_absolute_error(y, y_pred)),
        "mse": float(mean_squared_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
        "n_samples": int(len(y)),
    }
    return metrics, y_pred


def plot_predictions(y_true_dict, y_pred_dict, save_path):
    splits = list(y_true_dict.keys())

    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4), squeeze=False)

    for ax, split in zip(axes[0], splits):
        y_true = y_true_dict[split]
        y_pred = y_pred_dict[split]
        ax.scatter(y_true, y_pred, alpha=0.3, s=5, color=SPLIT_COLORS.get(split, "gray"))
        vmin = min(y_true.min(), y_pred.min())
        vmax = max(y_true.max(), y_pred.max())
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1, label="y = x")
        ax.set_xlabel("True Redshift (normalized)")
        ax.set_ylabel("Predicted Redshift (normalized)")
        ax.set_title(f"{split.title()} Split")
        ax.legend(fontsize=8)

    fig.suptitle("KNN Predicted vs Actual Redshift", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_kde(y_true_dict, y_pred_dict, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))

    for split, y_true in y_true_dict.items():
        errors = y_pred_dict[split] - y_true
        sns.kdeplot(errors, ax=ax, label=split.title(), color=SPLIT_COLORS.get(split, "gray"), fill=True, alpha=0.3)

    ax.axvline(0, color='black', linestyle='--', linewidth=1, label="zero error")
    ax.set_xlabel("Prediction Error (predicted - true, normalized)")
    ax.set_ylabel("Density")
    ax.set_title("KDE of KNN Prediction Errors")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tuning_results(tune_results, save_path):
    labels = [f"k={r['n_neighbors']}, w={r['weights'][:3]}, m={r['metric'][:3]}" for r in tune_results]
    maes = [r["valid_mae"] for r in tune_results]
    best_idx = int(np.argmin(maes))
    bar_colors = ["seagreen" if i == best_idx else "steelblue" for i in range(len(maes))]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.4), 5))
    ax.bar(range(len(labels)), maes, color=bar_colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Validation MAE (normalized)")
    ax.set_title("KNN Hyperparameter Tuning — Validation MAE")
    ax.legend(handles=[
        mpatches.Patch(color="seagreen", label=f"Best: {labels[best_idx]}"),
        mpatches.Patch(color="steelblue", label="Other combinations"),
    ])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(comparison_results, save_path):
    labels = [r["label"] for r in comparison_results]
    r2s = [r["test_r2"] for r in comparison_results]
    maes = [r["test_mae"] for r in comparison_results]

    best_r2_idx = int(np.argmax(r2s))
    best_mae_idx = int(np.argmin(maes))

    def bar_colors(best_idx):
        return ["seagreen" if i == best_idx else "steelblue" for i in range(len(labels))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(labels, r2s, color=bar_colors(best_r2_idx))
    ax1.set_xlabel("Mask Ratio")
    ax1.set_ylabel("Test R²")
    ax1.set_title("Test R² by Mask Ratio")
    ax1.legend(handles=[
        mpatches.Patch(color="seagreen", label=f"Best: {labels[best_r2_idx]}"),
        mpatches.Patch(color="steelblue", label="Other"),
    ], fontsize=8)

    ax2.bar(labels, maes, color=bar_colors(best_mae_idx))
    ax2.set_xlabel("Mask Ratio")
    ax2.set_ylabel("Test MAE (normalized)")
    ax2.set_title("Test MAE by Mask Ratio")
    ax2.legend(handles=[
        mpatches.Patch(color="seagreen", label=f"Best: {labels[best_mae_idx]}"),
        mpatches.Patch(color="steelblue", label="Other"),
    ], fontsize=8)

    fig.suptitle("Latent Quality: KNN Probe Across Mask Ratios", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_model(input_folder, output_folder, fixed_params=None):
    """Evaluate KNN for a single model's latent vectors. Returns test metrics.

    If fixed_params is provided, skips tuning and uses them directly.
    """
    logger = get_logger()
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split, fname in SPLIT_FILES.items():
        fpath = Path(input_folder) / fname
        X, y = load_split(fpath)
        splits[split] = (X, y)
        logger.info(f"[{split:>5}] Loaded X={X.shape}, y={y.shape} from {fpath.name}")

    X_train, y_train = splits["train"]
    X_valid, y_valid = splits["valid"]

    if fixed_params is None:
        logger.info("Tuning KNN hyperparameters...")
        best_params, tune_results = tune_knn(X_train, y_train, X_valid, y_valid)
        save_to_json(output_folder / "knn_tune_results.json", tune_results)
        plot_tuning_results(tune_results, output_folder / "knn_tuning.png")
    else:
        best_params = fixed_params
        logger.info(f"Using fixed params from baseline: {best_params}")

    save_to_yaml(output_folder / "knn_best_params.yaml", best_params)

    # Fit final model on train+valid combined with best params, evaluate on test
    X_trainval = np.concatenate([X_train, X_valid], axis=0)
    y_trainval = np.concatenate([y_train, y_valid], axis=0)
    final_model = KNeighborsRegressor(**best_params)
    final_model.fit(X_trainval, y_trainval)
    logger.info(f"Fitted final KNN on train+valid with {best_params}")

    # Also fit a train-only model for reporting train and valid metrics separately
    train_model = KNeighborsRegressor(**best_params)
    train_model.fit(X_train, y_train)

    y_true_dict = {}
    y_pred_dict = {}
    all_metrics = []

    for split, (X, y) in splits.items():
        model = train_model if split in ("train", "valid") else final_model
        metrics, y_pred = evaluate_knn(model, X, y, split)
        all_metrics.append(metrics)
        y_true_dict[split] = y
        y_pred_dict[split] = y_pred
        logger.info(
            f"[{split.upper():>5}] r2={metrics['r2']:.4f}, "
            f"mae={metrics['mae']:.6f}, mse={metrics['mse']:.6f} "
            f"(n={metrics['n_samples']})"
        )

    save_to_json(output_folder / "knn_eval_metrics.json", all_metrics)
    logger.info(f"Evaluation metrics saved to: {output_folder / 'knn_eval_metrics.json'}")

    plot_predictions(y_true_dict, y_pred_dict, output_folder / "knn_scatter.png")
    plot_error_kde(y_true_dict, y_pred_dict, output_folder / "knn_error_kde.png")
    logger.info(f"Plots saved to: {output_folder}")

    return next(m for m in all_metrics if m["split"] == "test")


@log_execution_time
def main(args):
    if args.debug:
        set_logger_level(10)

    np.random.seed(args.random_seed)

    logger = get_logger()
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.params_file is not None:
        fixed_params = load_from_yaml(args.params_file)
        logger.info(f"Loaded fixed params from {args.params_file}: {fixed_params}")
    else:
        fixed_params = None

    run_model(args.input_folder, output_folder, fixed_params=fixed_params)


# ==================================================
# CONTRIBUTION END
# ==================================================

if __name__ == "__main__":
    from src.utils.logger import init_shared_logger
    logger = init_shared_logger(__file__, log_stdout=True, log_stderr=True)
    try:
        args = process_args()
        main(args)
    except Exception as e:
        logger.error(e)


"""
Example usage (baseline - self-tunes, saves knn_best_params.yaml):
python -m src.analysis.knn_regressor \
--input-folder experiments/train_mae_leslie_0.0/artifacts/samples \
--output-folder experiments/knn_results/train_mae_leslie_0.0 \
--debug

Example usage (everyone else - uses fixed params from baseline):
python -m src.analysis.knn_regressor \
--params-file configs/knn_best_params.yaml \
--input-folder experiments/train_mae_charlie_0.25/artifacts/samples \
--output-folder experiments/knn_results/train_mae_charlie_0.25 \
--debug
"""
