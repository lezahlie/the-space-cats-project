import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import matplotlib.lines as mlines
import matplotlib.ticker as mtick
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, VPacker, TextArea
from src.utils.common import argparse, Path, pd, np, read_from_json, save_to_json

# ==================================================
# CONTRIBUTION START: MAE Learning Curves and Train/Val Gap
# Contributor: Leslie Horace
# ==================================================

RUNS = [
    ("leslie", "0.0"),
    ("charlie", "0.25"),
    ("chris", "0.5"),
    ("wen", "0.75"),
]

MASK_ORDER = ["0.0", "0.25", "0.5", "0.75"]
METRICS = ["objective_loss", "smooth_l1", "ssim_loss"]

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 17
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 12
ANNOTATION_FONTSIZE = 11
LINEWIDTH = 3.0
BEST_MARKER_SIZE = 230
GAP_VIOLIN_ALPHA = 0.50
DPI = 300

SPLIT_STYLES = {
    "training": "-",
    "validation": "--",
}

METRIC_LABELS = {
    "objective_loss": r"$\ell_{\mathrm{overall}}$",
    "smooth_l1": r"$\ell_{\mathrm{smooth\ L1}}$",
    "ssim_loss": r"$\ell_{\mathrm{SSIM}}$",
}

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
        "savefig.bbox": "tight",
    }
)


def process_args():
    parser = argparse.ArgumentParser(
        description="Plot MAE learning curves and train/validation gap violins."
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project root directory. Default: current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/learning_curves",
        help="Output directory for learning-curve and gap plots.",
    )
    parser.add_argument(
        "--post-best-step-padding",
        type=int,
        default=50,
        help="Number of gradient steps to show after the latest best model step.",
    )
    parser.add_argument(
        "--no-step-limit",
        action="store_true",
        help="Plot all optimizer steps instead of clipping after the latest best step.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRICS,
        default=METRICS,
        help="Metrics to plot. Default: all metrics.",
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Save PNG figures only. Default saves both PNG and PDF.",
    )
    return parser.parse_args()


def load_run(project_dir, person, mask_ratio):
    run_dir = project_dir / "experiments" / f"train_mae_medium_{person}_mask_{mask_ratio}"

    history_path = run_dir / "artifacts" / "metrics" / "model_history.json"
    metadata_path = run_dir / "result_metadata.json"

    missing = []
    if not history_path.is_file():
        missing.append(history_path)
    if not metadata_path.is_file():
        missing.append(metadata_path)

    if missing:
        return run_dir, None, None, missing

    history = read_from_json(history_path)
    metadata = read_from_json(metadata_path)

    return run_dir, history, metadata, []


def flatten_history(person, mask_ratio, run_dir, history):
    rows = []

    for row in history:
        epoch = row.get("epoch", None)
        optimizer_step = row.get("optimizer_step", None)
        learning_rate = row.get("learning_rate", None)

        split_metrics = {
            "training": row.get("training_metrics", {}),
            "validation": row.get("validation_metrics", {}),
        }

        for split, metrics_dict in split_metrics.items():
            for metric in METRICS:
                if metric not in metrics_dict:
                    continue

                rows.append(
                    {
                        "person": person,
                        "mask_ratio": mask_ratio,
                        "run_name": run_dir.name,
                        "split": split,
                        "epoch": epoch,
                        "optimizer_step": optimizer_step,
                        "learning_rate": learning_rate,
                        "metric": metric,
                        "value": metrics_dict[metric],
                    }
                )

    return rows


def flatten_metadata(person, mask_ratio, run_dir, metadata):
    test_metrics = metadata.get("test_metrics", {})

    rows = []
    for metric in METRICS:
        rows.append(
            {
                "person": person,
                "mask_ratio": mask_ratio,
                "run_name": run_dir.name,
                "metric": metric,
                "test_value": test_metrics.get(metric, None),
                "best_epoch": metadata.get("best_epoch", None),
                "best_optimizer_step": metadata.get("best_optimizer_step", None),
                "best_valid_loss": metadata.get("best_valid_loss", None),
                "stop_reason": metadata.get("stop_reason", None),
                "optimizer_steps_total": metadata.get("optimizer_steps_total", None),
                "optimizer_step_budget": metadata.get("optimizer_step_budget", None),
                "validation_checks": metadata.get("validation_checks", None),
            }
        )

    return rows


def coerce_curve_columns(curves_df):
    curves_df = curves_df.copy()
    for col in ["epoch", "optimizer_step", "learning_rate", "value"]:
        if col in curves_df.columns:
            curves_df[col] = pd.to_numeric(curves_df[col], errors="coerce")
    return curves_df


def coerce_summary_columns(summary_df):
    summary_df = summary_df.copy()
    for col in [
        "test_value",
        "best_epoch",
        "best_optimizer_step",
        "best_valid_loss",
        "optimizer_steps_total",
        "optimizer_step_budget",
        "validation_checks",
    ]:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")
    return summary_df


def load_all_runs(project_dir):
    curve_rows = []
    summary_rows = []
    missing_by_run = []

    for person, mask_ratio in RUNS:
        run_dir, history, metadata, missing = load_run(project_dir, person, mask_ratio)

        if missing:
            missing_by_run.append((person, mask_ratio, missing))
            continue

        curve_rows.extend(flatten_history(person, mask_ratio, run_dir, history))
        summary_rows.extend(flatten_metadata(person, mask_ratio, run_dir, metadata))

    if missing_by_run:
        print("\nWARNING: skipping missing runs:")
        for person, mask_ratio, paths in missing_by_run:
            print(f"\nperson={person} mask_ratio={mask_ratio}")
            for path in paths:
                print(f"  missing: {path}")

    curves_df = coerce_curve_columns(pd.DataFrame(curve_rows))
    summary_df = coerce_summary_columns(pd.DataFrame(summary_rows))

    if curves_df.empty:
        raise ValueError("No learning-curve rows found. Nothing to plot.")

    return curves_df, summary_df


def get_auto_max_optimizer_step(summary_df, post_best_step_padding=50):
    if summary_df.empty or "best_optimizer_step" not in summary_df.columns:
        return None

    best_steps = pd.to_numeric(summary_df["best_optimizer_step"], errors="coerce").dropna()

    if best_steps.empty:
        return None

    return int(best_steps.max()) + int(post_best_step_padding)


def apply_plot_limits(df, max_optimizer_step=None):
    df = df.copy()

    if max_optimizer_step is not None and "optimizer_step" in df.columns:
        df = df[df["optimizer_step"] <= max_optimizer_step]

    return df


def point_is_visible(row, max_optimizer_step=None):
    best_step = row["best_optimizer_step"]

    if pd.isna(best_step):
        return False

    if max_optimizer_step is not None and best_step > max_optimizer_step:
        return False

    return True


def get_best_validation_marker(curves_df, row, metric, max_optimizer_step=None):
    curves_df.to_csv("curves_df.csv")
    row_mask_ratio = str(row["mask_ratio"])

    valid_df = curves_df[
        (curves_df["mask_ratio"].astype(str) == row_mask_ratio)
        & (curves_df["split"] == "validation")
        & (curves_df["metric"] == metric)
    ].copy()

    if "person" in row and not pd.isna(row["person"]):
        valid_df = valid_df[valid_df["person"] == row["person"]]

    valid_df = valid_df.dropna(subset=["optimizer_step", "value"])

    if max_optimizer_step is not None:
        valid_df = valid_df[valid_df["optimizer_step"] <= max_optimizer_step]

    if valid_df.empty:
        return None

    best_step = row.get("best_optimizer_step", np.nan)

    if not pd.isna(best_step) and (
        max_optimizer_step is None or float(best_step) <= float(max_optimizer_step)
    ):
        valid_df["step_distance"] = (valid_df["optimizer_step"] - float(best_step)).abs()
        best_row = valid_df.sort_values(["step_distance", "optimizer_step"]).iloc[0]
    else:
        # Fallback: if metadata does not provide a visible checkpoint step, mark the
        # best visible validation point for this metric. Ties use the earliest step so
        # a flat zero-loss plateau still gets a visible marker near the first minimum.
        best_row = valid_df.sort_values(["value", "optimizer_step"]).iloc[0]

    if pd.isna(best_row["value"]):
        return None

    return {
        "optimizer_step": best_row["optimizer_step"],
        "value": best_row["value"],
    }


def format_step_tick(x):
    if x >= 1000:
        return f"{x / 1000:.1f}K"
    return f"{x:.0f}"


def format_metric_label(metric):
    return METRIC_LABELS.get(metric, metric)


def mask_label(mask_ratio):
    return f"mask_ratio={mask_ratio}"


def ordered_masks(df):
    present = set(df["mask_ratio"].astype(str)) if "mask_ratio" in df.columns else set()
    return [m for m in MASK_ORDER if m in present]


def mask_split_palette(mask_order):
    paired = sns.color_palette("Paired", n_colors=max(2 * len(mask_order), 2))
    colors = {}
    for i, mask_ratio in enumerate(mask_order):
        colors[(mask_ratio, "validation")] = paired[2 * i]
        colors[(mask_ratio, "training")] = paired[2 * i + 1]
    return colors


def mask_fill_palette(mask_order):
    paired = sns.color_palette("Paired", n_colors=max(2 * len(mask_order), 2))
    return {mask_order[i]: paired[2 * i + 1] for i in range(len(mask_order))}


def save_figure(fig, out_path, save_pdf=True):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    if save_pdf:
        print(f"Saved: {out_path.with_suffix('.pdf')}")


def style_axis(ax):
    ax.grid(True, color="grey", alpha=0.28, linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=7, min_n_ticks=5, prune=None))
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_alpha(0.9)


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

def add_learning_curve_legend(ax, present_order, colors):
    def make_line_box(color, linestyle):
        box = DrawingArea(44, 14, 0, 0)
        line = mlines.Line2D(
            [3, 41],
            [7, 7],
            color=color,
            linewidth=LINEWIDTH,
            linestyle=linestyle,
            solid_capstyle="round",
        )
        box.add_artist(line)
        return box

    def make_star_box(color):
        box = DrawingArea(28, 18, 0, 0)
        star = mlines.Line2D(
            [14],
            [9],
            color=color,
            marker="*",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=1.25,
            markersize=15,
            linestyle="None",
        )
        box.add_artist(star)
        return box

    legend_rows = []

    for mask_ratio in present_order:
        train_color = colors[(mask_ratio, "training")]
        val_color = colors[(mask_ratio, "validation")]

        row = HPacker(
            children=[
                TextArea(
                    f"{mask_label(mask_ratio)}",
                    textprops={
                        "fontsize": LEGEND_FONTSIZE,
                        "fontweight": "bold",
                    },
                ),
                make_line_box(train_color, "-"),
                TextArea(
                    "training",
                    textprops={
                        "fontsize": LEGEND_FONTSIZE,
                        "fontweight": "bold",
                    },
                ),
                make_line_box(val_color, "--"),
                TextArea(
                    "validation",
                    textprops={
                        "fontsize": LEGEND_FONTSIZE,
                        "fontweight": "bold",
                    },
                ),
                make_star_box(val_color),
                TextArea(
                    "best step",
                    textprops={
                        "fontsize": LEGEND_FONTSIZE,
                        "fontweight": "bold",
                    },
                ),
            ],
            align="center",
            pad=0,
            sep=8,
        )
        legend_rows.append(row)

    legend_box = VPacker(
        children=legend_rows,
        align="left",
        pad=0,
        sep=5,
    )

    anchored_legend = AnchoredOffsetbox(
        loc="upper right",
        child=legend_box,
        pad=0.35,
        borderpad=0.55,
        frameon=True,
    )
    anchored_legend.patch.set_facecolor("white")
    anchored_legend.patch.set_alpha(0.92)
    anchored_legend.patch.set_edgecolor("black")
    anchored_legend.patch.set_linewidth(1.0)

    ax.add_artist(anchored_legend)


def plot_learning_curves_combined(
    curves_df,
    summary_df,
    metric,
    out_path,
    max_optimizer_step=None,
    save_pdf=True,
):
    plot_df = curves_df[curves_df["metric"] == metric].copy()
    test_df = summary_df[summary_df["metric"] == metric].copy()
    metric_label = format_metric_label(metric)

    plot_df = apply_plot_limits(
        plot_df,
        max_optimizer_step=max_optimizer_step,
    )

    if plot_df.empty:
        print(f"Skipping {metric}: no curve rows after plot limits.")
        return

    present_order = ordered_masks(plot_df)
    colors = mask_split_palette(present_order)

    fig, ax = plt.subplots(figsize=(12, 8))

    for mask_ratio in present_order:
        mask_df = plot_df[plot_df["mask_ratio"] == mask_ratio]

        for split in ["training", "validation"]:
            split_df = mask_df[mask_df["split"] == split].sort_values("optimizer_step")

            if split_df.empty:
                continue

            ax.plot(
                split_df["optimizer_step"].to_numpy(),
                split_df["value"].to_numpy(),
                color=colors[(mask_ratio, split)],
                linestyle=SPLIT_STYLES[split],
                linewidth=LINEWIDTH,
                alpha=0.95,
                solid_capstyle="round",
                label="_nolegend_",
                zorder=3 if split == "validation" else 2,
            )

    x_max = max_optimizer_step
    if x_max is None:
        x_max = pd.to_numeric(plot_df["optimizer_step"], errors="coerce").max()

    if not pd.isna(x_max) and x_max > 0:
        x_buffer = max(float(x_max) * 0.04, 25.0)
        ax.set_xlim(left=-x_buffer, right=float(x_max) + x_buffer)

        ticks = np.linspace(
            0,
            x_max,
            num=min(9, int(x_max) + 1),
            dtype=int,
        )
        ax.set_xticks(ticks)
        ax.set_xticklabels([format_step_tick(x) for x in ticks])

    for _, row in test_df.iterrows():
        if not point_is_visible(row, max_optimizer_step=max_optimizer_step):
            continue

        mask_ratio = str(row["mask_ratio"])

        if (mask_ratio, "validation") not in colors:
            continue

        marker = get_best_validation_marker(
            curves_df=curves_df,
            row=row,
            metric=metric,
            max_optimizer_step=max_optimizer_step,
        )

        if marker is None:
            continue

        best_marker_color = colors[(mask_ratio, "validation")]

        ax.scatter(
            marker["optimizer_step"],
            marker["value"],
            facecolors=best_marker_color,
            edgecolors="black",
            marker="*",
            s=BEST_MARKER_SIZE,
            linewidths=1.25,
            zorder=20,
            clip_on=False,
            label="_nolegend_",
        )

    style_axis(ax)

    ax.set_title(
        f"MAE Learning Curves by Mask Ratio: {metric_label}",
        pad=14,
    )
    ax.set_xlabel("Gradient Step", labelpad=10)
    ax.set_ylabel(metric_label, labelpad=10)

    add_learning_curve_legend(
        ax=ax,
        present_order=present_order,
        colors=colors,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        top=0.90,
        bottom=0.12,
    )

    save_figure(
        fig,
        out_path,
        save_pdf=save_pdf,
    )


def build_train_validation_gap_df(curves_df):
    needed_cols = [
        "person",
        "mask_ratio",
        "run_name",
        "metric",
        "epoch",
        "optimizer_step",
        "split",
        "value",
    ]
    missing_cols = [col for col in needed_cols if col not in curves_df.columns]
    if missing_cols:
        raise KeyError(f"Cannot build train/validation gap. Missing columns: {missing_cols}")

    base_df = curves_df[needed_cols].copy()
    base_df = base_df[base_df["split"].isin(["training", "validation"])]
    base_df = base_df.dropna(subset=["optimizer_step", "value"])

    index_cols = ["person", "mask_ratio", "run_name", "metric", "epoch", "optimizer_step"]
    wide_df = base_df.pivot_table(
        index=index_cols,
        columns="split",
        values="value",
        aggfunc="mean",
    ).reset_index()
    wide_df.columns.name = None

    if "training" not in wide_df.columns or "validation" not in wide_df.columns:
        return pd.DataFrame()

    wide_df = wide_df.dropna(subset=["training", "validation"]).copy()
    if wide_df.empty:
        return wide_df

    wide_df["gap"] = wide_df["validation"] - wide_df["training"]
    wide_df["abs_gap"] = wide_df["gap"].abs()
    wide_df["gap_ratio"] = wide_df["validation"] / wide_df["training"].replace(0, np.nan)
    wide_df["val_above_train"] = wide_df["gap"] > 0

    return wide_df.sort_values(["metric", "mask_ratio", "optimizer_step"]).reset_index(drop=True)


def summarize_train_validation_gap(gap_df):
    if gap_df.empty:
        return pd.DataFrame()

    grouped = gap_df.groupby(["mask_ratio", "metric"], sort=False)
    rows = []

    for (mask_ratio, metric), group in grouped:
        gap = pd.to_numeric(group["gap"], errors="coerce").dropna()
        abs_gap = pd.to_numeric(group["abs_gap"], errors="coerce").dropna()

        if gap.empty:
            continue

        rows.append(
            {
                "mask_ratio": mask_ratio,
                "metric": metric,
                "n": int(gap.size),
                "mean_gap": float(gap.mean()),
                "median_gap": float(gap.median()),
                "q25_gap": float(gap.quantile(0.25)),
                "q75_gap": float(gap.quantile(0.75)),
                "mean_abs_gap": float(abs_gap.mean()) if not abs_gap.empty else np.nan,
                "median_abs_gap": float(abs_gap.median()) if not abs_gap.empty else np.nan,
                "pct_val_above_train": float((gap > 0).mean() * 100.0),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["mask_ratio"] = pd.Categorical(out["mask_ratio"], categories=MASK_ORDER, ordered=True)
    out["metric"] = pd.Categorical(out["metric"], categories=METRICS, ordered=True)
    out = out.sort_values(["metric", "mask_ratio"]).reset_index(drop=True)
    out["mask_ratio"] = out["mask_ratio"].astype(str)
    out["metric"] = out["metric"].astype(str)
    return out


def plot_train_validation_gap_violin(
    gap_df,
    metric,
    out_path,
    save_pdf=True,
):
    plot_df = gap_df[gap_df["metric"] == metric].copy()

    if plot_df.empty:
        print(f"Skipping train/validation gap violin for {metric}: no rows.")
        return

    present_order = ordered_masks(plot_df)
    colors = mask_fill_palette(present_order)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.grid(True, axis="y", alpha=0.45, zorder=0)
    ax.grid(False, axis="x")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sns.violinplot(
            data=plot_df,
            x="mask_ratio",
            y="gap",
            order=present_order,
            palette=[colors[m] for m in present_order],
            cut=0,
            inner=None,
            linewidth=1.5,
            saturation=1.0,
            ax=ax,
            zorder=2,
        )

    for violin_body in ax.collections:
        violin_body.set_alpha(GAP_VIOLIN_ALPHA)

    # Match the simpler final_gen_gap style: clean violins plus a median marker.
    medians = plot_df.groupby("mask_ratio")["gap"].median().reindex(present_order)
    positions = np.arange(len(present_order))
    ax.scatter(
        positions,
        medians.to_numpy(),
        marker="*",
        s=120,
        color="black",
        zorder=4,
        label="Median gap",
    )

    ax.axhline(0.0, color="black", linewidth=1.2, linestyle=":", alpha=0.95, zorder=1)

    style_axis(ax)
    ax.grid(True, axis="y", alpha=0.45, zorder=0)
    ax.grid(False, axis="x")
    ax.set_title(f"MAE Generalization Gap by Mask Ratio: {format_metric_label(metric)}", pad=14)
    ax.set_xlabel("Mask Ratio", labelpad=10)
    ax.set_ylabel(r"Gap (Valid $\ell_{\mathrm{overall}}$ - Train $\ell_{\mathrm{overall}}$", labelpad=10)
    ax.set_xticks(positions)
    ax.set_xticklabels([m for m in present_order])

    legend = ax.legend(loc="best", frameon=True)
    style_legend(legend)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.89, bottom=0.14)
    save_figure(fig, out_path, save_pdf=save_pdf)


def scalar_for_json(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dataframe_records_for_json(df):
    records = []
    for row in df.to_dict(orient="records"):
        records.append({key: scalar_for_json(value) for key, value in row.items()})
    return records


def save_dataframe_json(df, path):
    save_to_json(path, dataframe_records_for_json(df))


def get_curve_value_at_best_step(curves_df, person, mask_ratio, metric, split, best_step):
    if pd.isna(best_step):
        return None

    split_df = curves_df[
        (curves_df["person"] == person)
        & (curves_df["mask_ratio"] == mask_ratio)
        & (curves_df["metric"] == metric)
        & (curves_df["split"] == split)
    ].copy()

    if split_df.empty:
        return None

    split_df["step_distance"] = (split_df["optimizer_step"] - float(best_step)).abs()
    row = split_df.sort_values("step_distance").iloc[0]

    if pd.isna(row["value"]):
        return None

    return float(row["value"])


def save_best_model_loss_table(curves_df, summary_df, output_dir):
    rows = []

    for mask_ratio in MASK_ORDER:
        run_df = summary_df[summary_df["mask_ratio"] == mask_ratio]

        if run_df.empty:
            continue

        base = run_df.iloc[0]

        row_out = {
            "person": base["person"],
            "mask_ratio": base["mask_ratio"],
            "run_name": base["run_name"],
            "best_epoch": base["best_epoch"],
            "best_optimizer_step": base["best_optimizer_step"],
            "stop_reason": base["stop_reason"],
        }

        for _, metric_row in run_df.iterrows():
            metric = metric_row["metric"]
            person = metric_row["person"]
            best_step = metric_row["best_optimizer_step"]

            row_out[f"{metric}_training"] = get_curve_value_at_best_step(
                curves_df=curves_df,
                person=person,
                mask_ratio=mask_ratio,
                metric=metric,
                split="training",
                best_step=best_step,
            )

            row_out[f"{metric}_validation"] = get_curve_value_at_best_step(
                curves_df=curves_df,
                person=person,
                mask_ratio=mask_ratio,
                metric=metric,
                split="validation",
                best_step=best_step,
            )

            row_out[f"{metric}_test"] = metric_row["test_value"]

        rows.append(row_out)

    table_df = pd.DataFrame(rows)

    if table_df.empty:
        print("Skipping best model loss table: no rows.")
        return

    table_df = table_df.sort_values("mask_ratio")

    csv_path = output_dir / "best_model_losses.csv"
    table_df.to_csv(csv_path, index=False)
    save_dataframe_json(table_df, output_dir / "best_model_losses.json")

    print(f"Saved: {csv_path}")


def main():
    args = process_args()

    project_dir = Path(args.project_dir).resolve()
    output_dir = (project_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    curves_df, summary_df = load_all_runs(project_dir)

    max_optimizer_step = None
    if not args.no_step_limit:
        max_optimizer_step = get_auto_max_optimizer_step(
            summary_df,
            post_best_step_padding=args.post_best_step_padding,
        )

    limited_curves_df = apply_plot_limits(curves_df, max_optimizer_step=max_optimizer_step)

    curves_csv = output_dir / "learning_curve_rows.csv"
    limited_curves_csv = output_dir / "learning_curve_rows_limited.csv"
    summary_csv = output_dir / "learning_curve_summary.csv"

    curves_df.to_csv(curves_csv, index=False)
    limited_curves_df.to_csv(limited_curves_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    save_dataframe_json(summary_df, output_dir / "learning_curve_summary.json")

    save_best_model_loss_table(curves_df, summary_df, output_dir)

    print(f"Saved: {curves_csv}")
    print(f"Saved: {limited_curves_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Auto plot limit: max_optimizer_step={max_optimizer_step}")

    gap_df = build_train_validation_gap_df(curves_df)
    limited_gap_df = apply_plot_limits(gap_df, max_optimizer_step=max_optimizer_step)
    gap_summary_df = summarize_train_validation_gap(limited_gap_df)

    gap_csv = output_dir / "train_validation_gap_rows.csv"
    limited_gap_csv = output_dir / "train_validation_gap_rows_limited.csv"
    gap_summary_csv = output_dir / "train_validation_gap_summary.csv"

    gap_df.to_csv(gap_csv, index=False)
    limited_gap_df.to_csv(limited_gap_csv, index=False)
    gap_summary_df.to_csv(gap_summary_csv, index=False)
    save_dataframe_json(gap_summary_df, output_dir / "train_validation_gap_summary.json")

    print(f"Saved: {gap_csv}")
    print(f"Saved: {limited_gap_csv}")
    print(f"Saved: {gap_summary_csv}")

    save_pdf = not args.png_only
    for metric in args.metrics:
        plot_learning_curves_combined(
            curves_df=curves_df,
            summary_df=summary_df,
            metric=metric,
            out_path=output_dir / f"learning_curves_{metric}.png",
            max_optimizer_step=max_optimizer_step,
            save_pdf=save_pdf,
        )

        plot_train_validation_gap_violin(
            gap_df=limited_gap_df,
            metric=metric,
            out_path=output_dir / f"train_validation_gap_violin_{metric}.png",
            save_pdf=save_pdf,
        )

    print(f"\nDone. Output directory: {output_dir}")


# ==================================================
# CONTRIBUTION END: MAE Learning Curves and Train/Val Gap
# ==================================================


if __name__ == "__main__":
    main()
