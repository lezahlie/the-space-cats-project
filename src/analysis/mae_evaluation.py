import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import seaborn as sns

from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.functional.pairwise import pairwise_cosine_similarity as cosine_sim
from src.utils.common import np, pd, pt, Path, GalaxiesMLDataset


MASK_ORDER = ["0.0", "0.25", "0.5", "0.75"]
MASK_POS = {mask_ratio: i for i, mask_ratio in enumerate(MASK_ORDER)}
COLOR_PALETTE = dict(zip(MASK_ORDER, sns.color_palette("Set2", n_colors=len(MASK_ORDER))))

METRICS = {
    "ssim": {
        "label": "SSIM",
        "title": "Reconstruction SSIM",
        "direction": "higher",
    },
    "psnr": {
        "label": "PSNR",
        "title": "Reconstruction PSNR",
        "direction": "higher",
    },
    "cosine": {
        "label": "Cosine Similarity",
        "title": "Reconstruction Cosine Sim",
        "direction": "higher",
    },
    "mse": {
        "label": "Full-image MSE",
        "title": "Full-image MSE",
        "direction": "lower",
    },
    "mae": {
        "label": "Full-image MAE",
        "title": "Full-image MAE",
        "direction": "lower",
    },
    "masked_mse": {
        "label": "Masked-region MSE",
        "title": "Masked-region MSE",
        "direction": "lower",
    },
    "masked_mae": {
        "label": "Masked-region MAE",
        "title": "Masked-region MAE",
        "direction": "lower",
    },
    "ssim_error": {
        "label": "1 - SSIM",
        "title": "Reconstruction SSIM Error",
        "direction": "lower",
    },
    "cosine_error": {
        "label": "1 - Cosine Sim",
        "title": "Reconstruction Cosine Sim Error",
        "direction": "lower",
    },
}

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 17
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 12
LINEWIDTH = 2.4
BOX_LINEWIDTH = 1.5
KDE_ALPHA = 0.50
DPI = 300

BOX_FIGSIZE = (10.5, 6.5)
KDE_FIGSIZE = (10.5, 6.5)
SUMMARY_FIGSIZE = (10.5, 6.5)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update(
    {
        "font.size": TICK_FONTSIZE,
        "font.weight": "bold",
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.titleweight": "bold",
        "axes.labelsize": LABEL_FONTSIZE,
        "axes.labelweight": "bold",
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "legend.title_fontsize": LEGEND_FONTSIZE,
        "savefig.bbox": "tight",
    }
)


def make_dataset(path):
    return GalaxiesMLDataset(
        path,
        data_keys=[
            "y_recon_image",
            "y_target_image",
            "masked_region_map",
            "original_id",
        ],
        return_dict=True,
    )


def make_image_batch(x):
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(0)

    return x.to(pt.float32)


def make_metric_tensors(x_recon, x_target):
    x_recon = pt.as_tensor(x_recon).to(pt.float32)
    x_target = pt.as_tensor(x_target).to(pt.float32)

    if x_recon.shape != x_target.shape:
        raise ValueError(
            f"x_recon and x_target must have same shape, got "
            f"{tuple(x_recon.shape)} and {tuple(x_target.shape)}"
        )

    return x_recon, x_target


def make_mask(masked_region_map, image):
    if masked_region_map is None:
        return None

    mask = pt.as_tensor(masked_region_map)

    if mask.ndim == 0:
        return None

    mask = mask.to(pt.bool)

    while mask.ndim < image.ndim:
        mask = mask.unsqueeze(0)

    if mask.shape == image.shape:
        return mask

    if mask.shape[-2:] == image.shape[-2:]:
        if mask.shape[0] == 1 and image.shape[0] > 1:
            return mask.expand_as(image)

    raise ValueError(
        f"masked_region_map shape {tuple(mask.shape)} is not compatible "
        f"with image shape {tuple(image.shape)}"
    )


def compute_error_metrics(x_recon, x_target, masked_region_map=None):
    x_recon, x_target = make_metric_tensors(x_recon, x_target)

    diff = x_recon - x_target

    mse_value = pt.mean(diff ** 2).item()
    mae_value = pt.mean(pt.abs(diff)).item()

    mask = make_mask(masked_region_map, x_target)

    if mask is None or mask.sum().item() == 0:
        masked_mse_value = np.nan
        masked_mae_value = np.nan
    else:
        masked_diff = diff[mask]
        masked_mse_value = pt.mean(masked_diff ** 2).item()
        masked_mae_value = pt.mean(pt.abs(masked_diff)).item()

    return mse_value, mae_value, masked_mse_value, masked_mae_value


def compute_metrics_one(x_recon, x_target, masked_region_map, ssim_metric, psnr_metric):
    x_recon_img = make_image_batch(x_recon)
    x_target_img = make_image_batch(x_target)

    ssim_value = ssim_metric(x_recon_img, x_target_img).item()
    psnr_value = psnr_metric(x_recon_img, x_target_img).item()

    x_recon_flat = x_recon_img.flatten(start_dim=1)
    x_target_flat = x_target_img.flatten(start_dim=1)

    cosine_value = cosine_sim(x_recon_flat, x_target_flat).diag()[0].item()

    mse_value, mae_value, masked_mse_value, masked_mae_value = compute_error_metrics(
        x_recon=x_recon,
        x_target=x_target,
        masked_region_map=masked_region_map,
    )

    return (
        ssim_value,
        psnr_value,
        cosine_value,
        mse_value,
        mae_value,
        masked_mse_value,
        masked_mae_value,
    )


def collect_metric_rows(datasets):
    rows = []

    ssim_metric = SSIM(data_range=1.0)
    psnr_metric = PSNR(data_range=1.0)

    for run_name, dataset in datasets.items():
        mask_ratio = run_name.split()[-1]

        print(f"\nComputing metrics for {run_name}: {len(dataset)} samples")

        for idx in range(len(dataset)):
            sample = dataset[idx]
            x_recon = sample.y_recon_image
            x_target = sample.y_target_image
            masked_region_map = sample.get("masked_region_map", None)
            original_id = sample.get("original_id", idx)

            if hasattr(original_id, "item"):
                original_id = original_id.item()

            (
                ssim_value,
                psnr_value,
                cosine_value,
                mse_value,
                mae_value,
                masked_mse_value,
                masked_mae_value,
            ) = compute_metrics_one(
                x_recon,
                x_target,
                masked_region_map,
                ssim_metric,
                psnr_metric,
            )

            rows.append(
                {
                    "run": run_name,
                    "mask_ratio": mask_ratio,
                    "original_id": original_id,
                    "idx": idx,
                    "ssim": ssim_value,
                    "psnr": psnr_value,
                    "cosine": cosine_value,
                    "mse": mse_value,
                    "mae": mae_value,
                    "masked_mse": masked_mse_value,
                    "masked_mae": masked_mae_value,
                }
            )

    return pd.DataFrame(rows)


def add_mask_ratio_column(df):
    df = df.copy()

    if "mask_ratio" not in df.columns:
        df["mask_ratio"] = df["run"].str.extract(r"(\d+(?:\.\d+)?)$")[0]

    df["mask_ratio"] = df["mask_ratio"].astype(str)

    return df


def add_paper_plot_metrics(df):
    df = df.copy()

    if "ssim" in df.columns:
        df["ssim_error"] = 1.0 - df["ssim"]

    if "cosine" in df.columns:
        df["cosine_error"] = 1.0 - df["cosine"]

    return df


def get_mask_counts(df, metric=None):
    if metric is None:
        count_df = df.dropna(subset=["mask_ratio"]).copy()
    else:
        count_df = df.dropna(subset=["mask_ratio", metric]).copy()

    counts = count_df.groupby("mask_ratio").size()
    present_order = [m for m in MASK_ORDER if m in set(count_df["mask_ratio"])]

    return present_order, counts


def mask_ratio_labels(present_order):
    return [str(mask_ratio) for mask_ratio in present_order]


def apply_mask_ratio_xticklabels(ax, present_order, counts=None, rotation=0):
    labels = []

    for mask_ratio in present_order:
        if counts is None:
            labels.append(str(mask_ratio))
        else:
            n = int(counts.loc[mask_ratio]) if mask_ratio in counts.index else 0
            labels.append(f"{mask_ratio} (n={n})")

    ax.set_xticks(np.arange(len(present_order)))
    ax.set_xticklabels(
        labels,
        rotation=rotation,
        ha="center",
        va="top",
    )



def set_categorical_x_margins(ax, n_categories, margin_frac=0.50):
    tick_spacing = 1.0

    if n_categories <= 1:
        left = -margin_frac * tick_spacing
        right = margin_frac * tick_spacing
    else:
        left = 0.0 - margin_frac * tick_spacing
        right = float(n_categories - 1) + margin_frac * tick_spacing

    ax.set_xlim(left, right)

def style_visible_axes(ax):
    for side in ["left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.1)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=6,
        width=1.1,
        colors="black",
        bottom=True,
        left=True,
        top=False,
        right=False,
    )

def make_kde_legend_handles(plotted_masks, counts):
    handles = []

    for mask_ratio in plotted_masks:
        n = int(counts.loc[mask_ratio]) if mask_ratio in counts.index else 0
        handles.append(
            mlines.Line2D(
                [0],
                [0],
                color=COLOR_PALETTE[mask_ratio],
                linewidth=LINEWIDTH,
                marker="s",
                markersize=10,
                markerfacecolor=COLOR_PALETTE[mask_ratio],
                markeredgecolor="black",
                markeredgewidth=0.6,
                alpha=0.90,
                label=f"{mask_ratio}  n={n}",
            )
        )

    return handles


def save_paper_figure(fig, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")

    print("Saved:", out_path)
    print("Saved:", out_path.with_suffix(".pdf"))



def style_axis(ax, numeric_axis="y", min_ticks=7):
    ax.grid(True, alpha=0.7, zorder=0)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)

    if numeric_axis == "y":
        ax.yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=7, min_n_ticks=min_ticks, prune=None)
        )
    elif numeric_axis == "x":
        ax.xaxis.set_major_locator(
            mticker.MaxNLocator(nbins=7, min_n_ticks=min_ticks, prune=None)
        )


def style_legend(legend):
    if legend is None:
        return

    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("black")
    frame.set_alpha(0.92)
    frame.set_linewidth(0.8)

    if legend.get_title() is not None:
        legend.get_title().set_fontweight("bold")

    for text in legend.get_texts():
        text.set_fontweight("bold")


def format_numeric_axis(ax, metric, axis="y"):
    axis_obj = ax.yaxis if axis == "y" else ax.xaxis

    if metric in {"ssim_error", "cosine_error", "mse", "mae", "masked_mse", "masked_mae"}:
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        axis_obj.set_major_formatter(formatter)

    elif metric == "psnr":
        axis_obj.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    elif metric in {"ssim", "cosine"}:
        axis_obj.set_major_formatter(mticker.FormatStrFormatter("%.5f"))

    ax.xaxis.get_offset_text().set_fontsize(TICK_FONTSIZE)
    ax.yaxis.get_offset_text().set_fontsize(TICK_FONTSIZE)


def plot_metric_box(df, metric, out_path):
    df = add_mask_ratio_column(df)
    df = add_paper_plot_metrics(df)

    plot_df = df.dropna(subset=[metric, "mask_ratio"]).copy()
    present_order, mask_counts = get_mask_counts(plot_df, metric=metric)

    if len(plot_df) == 0 or len(present_order) == 0:
        print(f"Skipping {metric} box plot. No valid values.")
        return

    fig, ax = plt.subplots(figsize=BOX_FIGSIZE)

    sns.boxplot(
        data=plot_df,
        x="mask_ratio",
        y=metric,
        order=present_order,
        hue="mask_ratio",
        hue_order=present_order,
        palette=COLOR_PALETTE,
        width=0.50,
        showfliers=False,
        linewidth=BOX_LINEWIDTH,
        medianprops={"color": "black", "linewidth": 2.0},
        legend=False,
        ax=ax,
    )

    direction = METRICS[metric]["direction"]

    ax.set_title(f"{METRICS[metric]['title']} by Mask Ratio", pad=14)
    ax.set_xlabel("Mask Ratio", labelpad=10)
    ax.set_ylabel(f"{METRICS[metric]['label']} ({direction} is better)", labelpad=10)

    apply_mask_ratio_xticklabels(
        ax,
        present_order,
        counts=mask_counts,
        rotation=0,
    )

    format_numeric_axis(ax, metric, axis="y")
    style_axis(ax, numeric_axis="y", min_ticks=7)
    style_visible_axes(ax)

    fig.subplots_adjust(
        left=0.14,
        right=0.97,
        top=0.89,
        bottom=0.26,
    )

    save_paper_figure(fig, out_path)
    plt.close(fig)

def plot_metric_kde(df, metric, out_path):
    df = add_mask_ratio_column(df)
    df = add_paper_plot_metrics(df)

    plot_df = df.dropna(subset=[metric, "mask_ratio"]).copy()
    present_order, mask_counts = get_mask_counts(plot_df, metric=metric)

    if len(plot_df) == 0 or len(present_order) == 0:
        print(f"Skipping {metric} KDE plot. No valid values.")
        return

    fig, ax = plt.subplots(figsize=KDE_FIGSIZE)

    plotted_any = False
    plotted_masks = []

    for mask_ratio in present_order:
        values = plot_df.loc[plot_df["mask_ratio"] == mask_ratio, metric]
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        n_mask = int(mask_counts.loc[mask_ratio]) if mask_ratio in mask_counts.index else len(values)

        if len(values) < 3 or values.nunique() < 2:
            continue

        sns.kdeplot(
            x=values.to_numpy(dtype=float),
            ax=ax,
            color=COLOR_PALETTE[mask_ratio],
            linewidth=LINEWIDTH,
            fill=True,
            alpha=KDE_ALPHA,
            label="_nolegend_",
            warn_singular=False,
        )

        plotted_any = True
        plotted_masks.append(mask_ratio)

    if not plotted_any:
        print(f"Skipping {metric} KDE plot. Not enough variation.")
        plt.close(fig)
        return

    direction = METRICS[metric]["direction"]

    ax.set_title(f"{METRICS[metric]['title']} Distribution", pad=14)
    ax.set_xlabel(f"{METRICS[metric]['label']} ({direction} is better)", labelpad=10)
    ax.set_ylabel("Density", labelpad=10)

    format_numeric_axis(ax, metric, axis="x")
    style_axis(ax, numeric_axis="x", min_ticks=7)

    legend = ax.legend(
        handles=make_kde_legend_handles(plotted_masks, mask_counts),
        loc="best",
        frameon=True,
    )
    style_legend(legend)

    fig.subplots_adjust(
        left=0.14,
        right=0.97,
        top=0.89,
        bottom=0.14,
    )

    save_paper_figure(fig, out_path)
    plt.close(fig)

def plot_masked_error_summary(df, out_path):
    df = add_mask_ratio_column(df)

    metrics = ["masked_mae", "masked_mse"]
    plot_df = df.dropna(subset=["mask_ratio"] + metrics).copy()

    if len(plot_df) == 0:
        print("Skipping masked error summary. No valid masked-region values.")
        return

    long_df = plot_df.melt(
        id_vars=["run", "mask_ratio", "original_id", "idx"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )

    long_df["metric"] = long_df["metric"].map(
        {
            "masked_mae": "Masked MAE",
            "masked_mse": "Masked MSE",
        }
    )

    present_order, mask_counts = get_mask_counts(plot_df, metric=None)

    fig, ax = plt.subplots(figsize=SUMMARY_FIGSIZE)

    sns.boxplot(
        data=long_df,
        x="mask_ratio",
        y="value",
        hue="metric",
        order=present_order,
        showfliers=False,
        linewidth=BOX_LINEWIDTH,
        medianprops={"color": "black", "linewidth": 2.0},
        ax=ax,
    )

    ax.set_title("Masked-Region Reconstruction Error", pad=14)
    ax.set_xlabel("Mask Ratio", labelpad=10)
    ax.set_ylabel("Error (lower is better)", labelpad=10)
    ax.set_yscale("log")

    apply_mask_ratio_xticklabels(
        ax,
        present_order,
        counts=mask_counts,
        rotation=0,
    )

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)
    style_axis(ax, numeric_axis="y", min_ticks=7)
    style_visible_axes(ax)
    legend = ax.legend(loc="best", frameon=True)
    style_legend(legend)


    fig.subplots_adjust(
        left=0.14,
        right=0.97,
        top=0.89,
        bottom=0.26,
    )

    save_paper_figure(fig, out_path)
    plt.close(fig)


def plot_all_metrics(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = [
        "ssim_error",
        "psnr",
        "cosine_error",
        "mse",
        "mae",
        "masked_mse",
        "masked_mae",
    ]

    for metric in metrics_to_plot:
        if metric not in df.columns and metric not in {"ssim_error", "cosine_error"}:
            print(f"Skipping {metric}. Missing column.")
            continue

        plot_metric_box(
            df,
            metric,
            out_dir / f"{metric}_box_plot.png",
        )

        plot_metric_kde(
            df,
            metric,
            out_dir / f"{metric}_kde_plot.png",
        )

    plot_masked_error_summary(
        df,
        out_dir / "masked_error_summary_box_plot.png",
    )


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
            / f"train_mae_medium_{first_name}_mask_{mask_ratio}"
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

    summary_metrics = [
        "ssim",
        "psnr",
        "cosine",
        "mse",
        "mae",
        "masked_mse",
        "masked_mae",
    ]

    print("\nMetric summary:")
    print(df.groupby("run")[summary_metrics].describe())

    summary_path = analysis_dir / "reconstruction_metric_summary.csv"
    summary = (
        df.groupby(["run", "mask_ratio"])[summary_metrics]
        .agg(["mean", "std", "median", "min", "max", "count"])
    )
    summary.to_csv(summary_path)
    print("Saved summary CSV:", summary_path)

    print("\nSample counts:")
    print(df.groupby(["run", "mask_ratio"]).size())


if __name__ == "__main__":
    main()
