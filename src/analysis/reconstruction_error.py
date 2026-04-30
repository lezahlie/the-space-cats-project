import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolor
import matplotlib.lines as mline
import seaborn as sns

from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.functional.pairwise import pairwise_cosine_similarity as cosine_sim
from scipy.stats import gaussian_kde

from src.utils.common import np, pd, pt, Path, GalaxiesMLDataset


MASK_ORDER = ["0.0", "0.25", "0.5", "0.75"]
MASK_POS = {mask_ratio: i for i, mask_ratio in enumerate(MASK_ORDER)}
COLOR_PALETTE = dict(zip(MASK_ORDER, sns.color_palette("Set2", n_colors=len(MASK_ORDER))))

METRICS = {
    "ssim": {
        "label": "SSIM (higher = more similar)",
        "title": "SSIM",
        "bounds": (0.0, 1.0),
    },
    "psnr": {
        "label": "PSNR",
        "title": "PSNR",
        "bounds": None,
    },
    "cosine": {
        "label": "Cosine Similarity (higher = more similar)",
        "title": "Cosine Similarity",
        "bounds": (-1.0, 1.0),
    },
}


PLOT_RC = {
    "figure.figsize": (10, 8),
    # "figure.dpi": 120,
    # "savefig.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
    "grid.alpha": 0.20,
}

sns.set_theme(
    context="paper",
    style="whitegrid",
    rc=PLOT_RC,
)


def make_dataset(path):
    return GalaxiesMLDataset(
        path,
        input_key="y_recon_image",
        target_key="y_target_image",
    )


def make_image_batch(x):
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(0)

    return x.to(pt.float32)


def compute_metrics_one(x_recon, x_target, ssim_metric, psnr_metric):
    x_recon = make_image_batch(x_recon)
    x_target = make_image_batch(x_target)

    ssim_value = ssim_metric(x_recon, x_target).item()
    psnr_value = psnr_metric(x_recon, x_target).item()

    x_recon_flat = x_recon.flatten(start_dim=1)
    x_target_flat = x_target.flatten(start_dim=1)

    cosine_value = cosine_sim(x_recon_flat, x_target_flat).diag()[0].item()

    return ssim_value, psnr_value, cosine_value


def collect_metric_rows(datasets):
    rows = []

    ssim_metric = SSIM(data_range=1.0)
    psnr_metric = PSNR(data_range=1.0)

    for run_name, dataset in datasets.items():
        mask_ratio = run_name.split()[-1]

        print(f"\nComputing metrics for {run_name}: {len(dataset)} samples")

        for idx in range(len(dataset)):
            x_recon, x_target = dataset[idx]

            ssim_value, psnr_value, cosine_value = compute_metrics_one(
                x_recon,
                x_target,
                ssim_metric,
                psnr_metric,
            )

            rows.append(
                {
                    "run": run_name,
                    "mask_ratio": mask_ratio,
                    "idx": idx,
                    "ssim": ssim_value,
                    "psnr": psnr_value,
                    "cosine": cosine_value,
                }
            )

    return pd.DataFrame(rows)



def add_mask_ratio_column(df):
    df = df.copy()

    if "mask_ratio" not in df.columns:
        df["mask_ratio"] = df["run"].str.extract(r"(\d+(?:\.\d+)?)$")[0]

    df["mask_ratio"] = df["mask_ratio"].astype(str)

    return df

def get_metric_xlim(df, metric):
    values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()

    if metric == "ssim":
        x_min = values.quantile(0.01)
        x_max = values.quantile(0.995)

        pad = 0.08 * (x_max - x_min)
        if pad == 0:
            pad = 0.002

        return max(0.0, x_min - pad), min(1.0, x_max + pad)

    if metric == "cosine":
        return -1.0, 1.0

    if metric == "psnr":
        x_min = values.quantile(0.01)
        x_max = values.quantile(0.99)

        pad = 0.08 * (x_max - x_min)
        if pad == 0:
            pad = 1.0

        return x_min - pad, x_max + pad

    return values.min(), values.max()

def plot_metric_box(df, metric, out_path):
    df = add_mask_ratio_column(df)

    present_order = [m for m in MASK_ORDER if m in set(df["mask_ratio"])]
    n_samples = len(df.dropna(subset=[metric, "mask_ratio"]))

    fig, ax = plt.subplots(figsize=(10, max(4, 1.5 * len(present_order))))
    
    sns.boxplot(
        data=df,
        x=metric,
        y="mask_ratio",
        order=present_order,
        hue="mask_ratio",
        hue_order=present_order,
        palette=COLOR_PALETTE,
        width=0.45,
        showfliers=True,
        linewidth=1.4,
        medianprops={"color": "black", "linewidth": 2.0},
        legend=False,
        ax=ax,
    )

    ax.set_xlim(*get_metric_xlim(df, metric))
    ax.set_title(
        f"Testing Set {METRICS[metric]['title']} by Mask Ratio (n={n_samples})"
    )
    ax.set_xlabel(METRICS[metric]["label"])
    ax.set_ylabel("Mask Ratio")

    ax.grid(True, axis="x")
    ax.grid(False, axis="y")

    sns.despine(ax=ax)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_path)


def plot_metric_hexbin(df, metric, out_path):
    df = add_mask_ratio_column(df)

    present_order = [m for m in MASK_ORDER if m in set(df["mask_ratio"])]
    present_pos = {mask_ratio: i for i, mask_ratio in enumerate(present_order)}

    plot_df = df.dropna(subset=[metric, "mask_ratio"]).copy()
    plot_df["mask_pos"] = plot_df["mask_ratio"].map(present_pos)

    rng = np.random.default_rng(0)
    plot_df["mask_pos_jitter"] = plot_df["mask_pos"] + rng.normal(
        loc=0.0,
        scale=0.05,
        size=len(plot_df),
    )

    x_min, x_max = get_metric_xlim(plot_df, metric)

    fig, ax = plt.subplots(figsize=(12, max(5, 2.5 * len(present_order))))

    y_pad = 0.35
    y_min = -y_pad
    y_max = len(present_order) - 1 + y_pad

    hb = ax.hexbin(
        plot_df[metric],
        plot_df["mask_pos_jitter"],
        bins=100,
        gridsize=(75, max(20, 24 * len(present_order))),
        extent=(x_min, x_max, y_min, y_max),
        mincnt=1,
        cmap="turbo",
        linewidths=0.05,
    )

    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label(f"Count (n={len(plot_df)})")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, len(present_order) - 0.5)

    ax.set_yticks(list(present_pos.values()))
    ax.set_yticklabels(present_order)

    ax.set_title(f"Testing Set {METRICS[metric]['title']} Density by Mask Ratio")
    ax.set_xlabel(METRICS[metric]["label"])
    ax.set_ylabel("Mask Ratio")

    ax.grid(True, axis="x")
    ax.grid(False, axis="y")

    sns.despine(ax=ax)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_path)


def plot_all_metrics(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["ssim", "psnr", "cosine"]:
        if metric not in df.columns:
            print(f"Skipping {metric}. Missing column.")
            continue

        plot_metric_box(df, metric, out_dir / f"{metric}_box.png")

        plot_metric_hexbin(df, metric, out_dir / f"{metric}_hexbin.png")


def main():
    project_path = Path(__file__).resolve().parents[2]

    analysis_dir = project_path / "analysis" / "reconstruction_error"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        "Leslie 0.0": ("leslie", "0.0"),
        "Charlie 0.25": ("charlie", "0.25"),
        "Chris 0.5": ("chris", "0.5"),
        "Wen 0.75": ("wen", "0.75"),
    }

    datasets = {}

    for run_label, (first_name, mask_ratio) in runs.items():
        path = (
            project_path
            / "experiments"
            / f"train_mae_{first_name}_{mask_ratio}"
            / "artifacts"
            / "samples"
            / "testing_outputs_best.hdf5"
        )

        if not path.exists():
            print(f"Skipping {run_label}. Missing: {path}")
            continue

        print(f"Loading {run_label}: {path}")
        datasets[run_label] = make_dataset(path)

    if len(datasets) == 0:
        print("\nNo datasets found. Nothing plotted.")
        print("Run this to inspect available files:")
        print("find experiments -name 'testing_outputs_best.hdf5'")
        return

    df = collect_metric_rows(datasets)

    csv_path = analysis_dir / "reconstruction_metrics.csv"

    plot_all_metrics(df, analysis_dir)

    df.to_csv(csv_path, index=False)
    print("Saved CSV:", csv_path)

    print("\nMetric summary:")
    print(df.groupby("run")[["ssim", "psnr", "cosine"]].describe())

    print("\nSample counts:")
    print(df.groupby(["run", "mask_ratio"]).size())

if __name__ == "__main__":
    main()