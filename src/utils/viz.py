import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from utils.common import Path, np

# ==================================================
# CONTRIBUTION START: Sampled Images and Learning Curves
# Contributor: Leslie Horace
# ==================================================
def tensor_to_image(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x)


def set_masked_values(x_images, mask_map, value=0.0):
    x_images = np.asarray(x_images, dtype=np.float32).copy()
    mask = np.asarray(mask_map, dtype=bool)

    if mask.ndim == 2:
        mask = np.broadcast_to(mask_map[None, :, :], x_images.shape)
    elif mask.ndim == 3 and mask.shape[0] == 1:
        mask = np.broadcast_to(mask, x_images.shape)
    elif mask.ndim == 3 and mask.shape[0] == x_images.shape[0]:
        pass
    else:
        raise ValueError(f"Unsupported masked_map shape {mask.shape} for x_masked shape {x_images.shape}")

    x_images[mask] = value
    return x_images


def plot_single_sample(
    masked_map,
    x_masked,
    y_target,
    y_recon,
    save_path,
    figure_title=None,
    band_names=("g", "r", "i", "z", "y"),
    cmap_name="inferno",
):
    title_fs = 24
    col_title_fs = 20
    row_label_fs = 20
    cbar_tick_fs = 16
    
    x_masked_nan = set_masked_values(x_masked, masked_map, value=np.nan)
    y_target = np.asarray(y_target, dtype=np.float32)
    y_recon = np.asarray(y_recon, dtype=np.float32)
    target_vminmax = np.nanmin(y_target).astype(float), np.nanmax(y_target).astype(float)
    recon_vminmax = np.nanmin(y_recon).astype(float), np.nanmax(y_recon).astype(float)

    row_data = [
        ("Masked X", x_masked_nan, target_vminmax),
        ("Target Y", y_target, target_vminmax),
        ("Recon Y", y_recon, recon_vminmax)
    ]

    num_rows = len(row_data)
    num_bands = len(x_masked)

    def _set_cbar_ticks(cbar, vmin=None, vmax=None):
        ticks = np.linspace(vmin, vmax, 5)
        cbar.set_ticks(ticks)
        cbar.formatter = mtick.FormatStrFormatter("%.3g")
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=cbar_tick_fs, rotation=-20)

    def _style_axis(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _draw_panel(ax, image, row_label=None, col_title=None, vmin=None, vmax=None):
        im = ax.imshow(
            np.asarray(image, dtype=np.float32),
            cmap=cmap_name,
            vmin=vmin,
            vmax=vmax,
        )
        _style_axis(ax)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _set_cbar_ticks(cbar, vmin, vmax)

        if col_title is not None:
            ax.set_title(col_title, fontsize=col_title_fs, pad=10)

        if row_label is not None:
            ax.set_ylabel(
                row_label,
                fontsize=row_label_fs,
                rotation=90,
                labelpad=18,
                va="center",
            )


    fig, axes = plt.subplots(num_rows, num_bands, figsize=(5 * num_bands, num_rows*4), squeeze=False)

    for row_idx, (row_name, row_images, (vmin_val, vmax_val)) in enumerate(row_data):
        for band_idx, band_name in enumerate(band_names):
            row_label = row_name if band_idx == 0 else None
            col_title = f"Channel {band_name.upper()}" if row_idx == 0 else None

            _draw_panel(
                axes[row_idx, band_idx],
                row_images[band_idx],
                row_label=row_label,
                col_title=col_title,
                vmin=vmin_val,
                vmax=vmax_val
            )

    if figure_title:
        fig.suptitle(figure_title, fontsize=title_fs, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.965])
    else:
        fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_image_samples(
    original_id,
    masked_map,
    x_masked,
    y_target,
    y_recon,
    y_redshift,
    save_path,
    figure_title=None,
    band_names=("g", "r", "i", "z", "y"),
    cmap_name="inferno",
):
    num_samples = len(original_id)

    if num_samples < 1:
        raise ValueError("No samples available to plot")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    stem = save_path.stem
    suffix = save_path.suffix if save_path.suffix else ".png"

    for sample_idx in range(num_samples):
        sample_save_path = save_path.parent / f"{stem}_sample_{sample_idx + 1}{suffix}"

        sample_original_id = original_id[sample_idx]
        if hasattr(sample_original_id, "item"):
            sample_original_id = sample_original_id.item()

        sample_title = f"{figure_title} | Original_id={sample_original_id}"
        if y_redshift is not None:
            sample_redshift = y_redshift[sample_idx]
            if hasattr(y_redshift[sample_idx], "item"):
                sample_redshift = sample_redshift.item()
            sample_title += f" | Redshift={sample_redshift:.2f}"

        plot_single_sample(
            masked_map=masked_map[sample_idx],
            x_masked=x_masked[sample_idx],
            y_target=y_target[sample_idx],
            y_recon=y_recon[sample_idx],
            save_path=sample_save_path,
            figure_title=sample_title,
            band_names=band_names,
            cmap_name=cmap_name,
        )
        saved_paths.append(sample_save_path)

    return saved_paths


def plot_learning_curves(history, save_path):
    if not history:
        return

    epochs = [x["epoch"] for x in history]

    train_loss = [x["train"]["objective_loss"] for x in history]
    valid_loss = [x["validation"]["objective_loss"] for x in history]

    train_smooth_l1 = [x["train"]["smooth_l1"] for x in history]
    valid_smooth_l1 = [x["validation"]["smooth_l1"] for x in history]

    train_ssim = [x["train"]["ssim_loss"] for x in history]
    valid_ssim = [x["validation"]["ssim_loss"] for x in history]

    title_fs = 22
    label_fs = 18
    tick_fs = 16
    legend_fs = 14
    line_w = 2.5

    train_color = "royalblue"
    valid_color = "firebrick"

    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)

    axes[0].plot(epochs, train_loss, label="Training", color=train_color, linewidth=line_w)
    axes[0].plot(epochs, valid_loss, label="Validation", color=valid_color, linewidth=line_w)
    axes[0].set_ylabel("Overall Loss", fontsize=label_fs)
    axes[0].set_title("Learning Curves", fontsize=title_fs, pad=14)
    axes[0].legend(fontsize=legend_fs, frameon=False)

    axes[1].plot(epochs, train_smooth_l1, label="Training", color=train_color, linewidth=line_w)
    axes[1].plot(epochs, valid_smooth_l1, label="Validation", color=valid_color, linewidth=line_w)
    axes[1].set_ylabel("Smooth-L1 Loss", fontsize=label_fs)
    axes[1].legend(fontsize=legend_fs, frameon=False)

    axes[2].plot(epochs, train_ssim, label="Training", color=train_color, linewidth=line_w)
    axes[2].plot(epochs, valid_ssim, label="Validation", color=valid_color, linewidth=line_w)
    axes[2].set_xlabel("Epoch", fontsize=label_fs)
    axes[2].set_ylabel("1-SSIM Loss", fontsize=label_fs)
    axes[2].legend(fontsize=legend_fs, frameon=False)
    axes[2].set_xticks(epochs)
    axes[2].set_xticklabels([str(e) for e in epochs], fontsize=tick_fs)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

# ==================================================
# CONTRIBUTION End: Sampled Images and Learning Curves
# ==================================================