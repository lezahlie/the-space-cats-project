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


def ensure_band_first_image(x, expected_bands=None):
    x = tensor_to_image(x)

    if x.ndim == 2:
        x = x[None, :, :]
    elif x.ndim == 3:
        if x.shape[0] <= 8:
            pass
        elif x.shape[-1] <= 8:
            x = np.transpose(x, (2, 0, 1))
        else:
            raise ValueError(f"Could not infer band dimension from shape={x.shape}")
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape={x.shape}")

    if expected_bands is not None and x.shape[0] != expected_bands:
        raise ValueError(f"Expected {expected_bands} bands, got shape={x.shape}")

    return x


def ensure_band_first_mask(mask, num_bands):
    mask = tensor_to_image(mask)

    if mask.ndim == 2:
        mask = np.repeat(mask[None, :, :], num_bands, axis=0)
    elif mask.ndim == 3:
        if mask.shape[0] == num_bands:
            pass
        elif mask.shape[-1] == num_bands:
            mask = np.transpose(mask, (2, 0, 1))
        elif mask.shape[0] == 1:
            mask = np.repeat(mask, num_bands, axis=0)
        elif mask.shape[-1] == 1:
            mask = np.transpose(mask, (2, 0, 1))
            mask = np.repeat(mask, num_bands, axis=0)
        else:
            raise ValueError(f"Mask shape {mask.shape} does not match num_bands={num_bands}")
    else:
        raise ValueError(f"Expected 2D or 3D mask, got shape={mask.shape}")

    return mask

def make_mask_overlay(mask_2d):
    mask_2d = np.asarray(mask_2d, dtype=np.float32)
    overlay = np.zeros((mask_2d.shape[0], mask_2d.shape[1], 4), dtype=np.float32)
    overlay[..., :3] = 1.0
    overlay[..., 3] = (mask_2d > 0.5).astype(np.float32)
    return overlay


def plot_single_sample(
    masked_input,
    mask,
    target,
    reconstruction,
    save_path,
    figure_title=None,
    band_names=("g", "r", "i", "z", "y"),
    cmap_name="inferno",
):
    num_bands = len(band_names)

    title_fs = 24
    col_title_fs = 20
    row_label_fs = 20
    cbar_tick_fs = 16

    masked_input = ensure_band_first_image(masked_input, expected_bands=num_bands)
    target = ensure_band_first_image(target, expected_bands=num_bands)
    reconstruction = ensure_band_first_image(reconstruction, expected_bands=num_bands)

    _ = mask

    def _set_five_data_ticks(cbar, vmin, vmax):
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
        _set_five_data_ticks(cbar, vmin, vmax)

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

    fig, axes = plt.subplots(
        3,
        num_bands,
        figsize=(5.2 * num_bands, 12),
        squeeze=False,
    )

    row_data = [
        ("Masked X", masked_input),
        ("Target Y", target),
        ("Recon Y", reconstruction),
    ]

    for row_idx, (row_name, row_images) in enumerate(row_data):
        row_images_np = np.asarray(row_images, dtype=np.float32)
        row_vmin = float(row_images_np.min())
        row_vmax = float(row_images_np.max())

        for band_idx, band_name in enumerate(band_names):
            row_label = row_name if band_idx == 0 else None
            col_title = f"Channel {band_name.upper()}" if row_idx == 0 else None

            _draw_panel(
                axes[row_idx, band_idx],
                row_images[band_idx],
                row_label=row_label,
                col_title=col_title,
                vmin=row_vmin,
                vmax=row_vmax,
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
    masked_inputs,
    masks,
    targets,
    reconstructions,
    save_path,
    redshifts=None,
    max_samples=4,
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
    sample_indices = np.random.choice(num_samples, size=max_samples, replace=False)
    for sample_idx in sample_indices:
        sample_save_path = save_path.parent / f"{stem}_sample_{sample_idx + 1}{suffix}"

        sample_original_id = original_id[sample_idx]
        if hasattr(sample_original_id, "item"):
            sample_original_id = sample_original_id.item()

        sample_title = f"{figure_title} | Original_id={sample_original_id}"
        if redshifts is not None:
            sample_redshift = redshifts[sample_idx]
            if hasattr(redshifts[sample_idx], "item"):
                sample_redshift = sample_redshift.item()
            sample_title += f" | Redshift={sample_redshift}"


        plot_single_sample(
            masked_input=masked_inputs[sample_idx],
            mask=masks[sample_idx],
            target=targets[sample_idx],
            reconstruction=reconstructions[sample_idx],
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