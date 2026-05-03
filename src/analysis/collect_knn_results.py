from src.analysis.knn_regressor import plot_model_comparison, plot_test_scatter_grid
from src.utils.common import argparse, Path, save_to_json, read_from_json
import numpy as np
import shutil

RUNS = [
    ("leslie", "0.0"),
    ("charlie", "0.25"),
    ("chris", "0.5"),
    ("wen", "0.75"),
]


def process_args():
    parser = argparse.ArgumentParser(
        description="Collect KNN metrics across mask-ratio runs and plot comparison."
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project root directory. Default: current directory.",
    )
    return parser.parse_args()


def load_test_metrics(path):
    metrics = read_from_json(path)

    for row in metrics:
        if row.get("split") == "test":
            return row

    raise ValueError(f"No test split found in {path}")


def main():
    args = process_args()
    project_dir = Path(args.project_dir).resolve()

    results_dir = project_dir / "results" / "knn"
    results_dir.mkdir(parents=True, exist_ok=True)

    ablation_dir = project_dir / "experiments" / "knn_results" / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    comparison_results = []
    scatter_data = {}
    missing = []
    missing_npz = []

    for person, mask_ratio in RUNS:
        run_name = f"train_mae_medium_{person}_mask_{mask_ratio}"
        run_dir = project_dir / "experiments" / "knn_results" / run_name

        src_path = run_dir / "knn_eval_metrics.json"
        dst_path = results_dir / f"{person}_{mask_ratio}.json"

        if not src_path.is_file():
            missing.append((person, mask_ratio, src_path))
            continue

        shutil.copy2(src_path, dst_path)
        test = load_test_metrics(dst_path)

        comparison_results.append(
            {
                "label": mask_ratio,
                "person": person,
                "mask_ratio": mask_ratio,
                "test_r2": float(test["r2"]),
                "test_mae": float(test["mae"]),
                "n_samples": int(test.get("n_samples", -1)),
                "input_path": str(src_path),
            }
        )

        print(
            f"mask_ratio={mask_ratio}: "
            f"test_r2={test['r2']:.4f}, "
            f"test_mae={test['mae']:.6f}"
        )

        npz_path = run_dir / "knn_test_predictions.npz"
        if npz_path.is_file():
            data = np.load(npz_path)
            scatter_data[mask_ratio] = (data["y_true"], data["y_pred"])
        else:
            missing_npz.append((person, mask_ratio, npz_path))

    if missing:
        print("\nMissing KNN result files:")
        for person, mask_ratio, path in missing:
            print(f"  person={person} mask_ratio={mask_ratio}: {path}")

        raise FileNotFoundError(
            "Cannot make final KNN comparison until all four knn_eval_metrics.json files exist."
        )

    if missing_npz:
        print("\nMissing knn_test_predictions.npz (re-run knn_regressor.py to generate):")
        for person, mask_ratio, path in missing_npz:
            print(f"  person={person} mask_ratio={mask_ratio}: {path}")

    save_to_json(ablation_dir / "knn_comparison.json", comparison_results)
    plot_model_comparison(comparison_results, ablation_dir / "knn_comparison.png")
    plot_test_scatter_grid(scatter_data, ablation_dir / "knn_test_scatter_grid.png")

    print(f"\nSaved comparison JSON:  {ablation_dir / 'knn_comparison.json'}")
    print(f"Saved comparison plot:  {ablation_dir / 'knn_comparison.png'}")
    print(f"Saved test scatter grid: {ablation_dir / 'knn_test_scatter_grid.png'}")


if __name__ == "__main__":
    main()