from src.utils.common import *
from src.utils.viz import *

MASK_ORDER = ["0.0", "0.25", "0.5", "0.75"]
BAND_NAMES = ("g", "r", "i", "z", "y")

SPLIT_FILES = {
    "training": "training_outputs_best.hdf5",
    "validation": "validation_outputs_best.hdf5",
    "testing": "testing_outputs_best.hdf5",
}

SPLIT_ALIASES = {
    "train": "training",
    "valid": "validation",
    "val": "validation",
    "test": "testing",
}

RUNS = {
    "0.0": ("leslie", "train_mae_medium_leslie_mask_0.0"),
    "0.25": ("charlie", "train_mae_medium_charlie_mask_0.25"),
    "0.5": ("chris", "train_mae_medium_chris_mask_0.5"),
    "0.75": ("wen", "train_mae_medium_wen_mask_0.75"),
}


def normalize_split_name(split):
    split = str(split).lower().strip()
    split = SPLIT_ALIASES.get(split, split)

    if split not in SPLIT_FILES:
        raise ValueError(
            f"split must be one of {sorted(SPLIT_FILES.keys())} "
            f"or aliases {sorted(SPLIT_ALIASES.keys())}, not {split!r}"
        )

    return split


def build_default_mask_paths(
    project_dir=".",
    split="testing",
    experiments_dir="experiments",
    run_template=None,
):
    project_dir = Path(project_dir)
    split = normalize_split_name(split)
    file_name = SPLIT_FILES[split]

    mask_to_hdf5_path = {}
    missing = []

    for mask_ratio, (person, default_run_name) in RUNS.items():
        if run_template is None:
            run_name = default_run_name
        else:
            run_name = run_template.format(
                person=person,
                mask_ratio=mask_ratio,
            )

        h5_path = (
            project_dir
            / experiments_dir
            / run_name
            / "artifacts"
            / "samples"
            / file_name
        )

        mask_to_hdf5_path[mask_ratio] = h5_path

        if not h5_path.is_file():
            missing.append(h5_path)

    if missing:
        msg = "Missing required HDF5 files:\n"
        msg += "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(msg)

    return mask_to_hdf5_path


def make_cross_mask_output_path(output_dir, split, original_id=None, sample_idx=0):
    output_dir = Path(output_dir)
    split = normalize_split_name(split)

    if original_id is not None:
        file_name = f"mae_cross_mask_{split}_original_id_{original_id}.png"
    else:
        file_name = f"mae_cross_mask_{split}_sample_{sample_idx}.png"

    return output_dir / file_name

def _as_scalar(x):
    x = tensor_to_image(x)
    if x.shape == ():
        return x.item()
    if x.size == 1:
        return x.reshape(-1)[0].item()
    return x


def _require_hdf5_keys(h5_path, required_keys):
    with h5py.File(h5_path, "r") as f:
        missing = [key for key in required_keys if key not in f]

    if missing:
        raise KeyError(f"{h5_path} is missing required HDF5 keys: {missing}")


def find_common_original_ids(mask_to_hdf5_path):
    common_ids = None

    for mask_ratio, h5_path in mask_to_hdf5_path.items():
        _require_hdf5_keys(h5_path, ["original_id"])

        with h5py.File(h5_path, "r") as f:
            ids = set(f["original_id"][:].tolist())

        common_ids = ids if common_ids is None else common_ids.intersection(ids)

    if not common_ids:
        raise ValueError("No shared original_id values across mask-ratio files")

    return sorted(common_ids)


def load_hdf5_sample_by_original_id(h5_path, original_id=None, sample_idx=0):
    h5_path = Path(h5_path)

    required_keys = [
        "original_id",
        "masked_region_map",
        "x_masked_image",
        "y_target_image",
        "y_recon_image",
    ]
    _require_hdf5_keys(h5_path, required_keys)

    with h5py.File(h5_path, "r") as f:
        original_ids = f["original_id"][:]

        if original_id is None:
            idx = int(sample_idx)
        else:
            matches = np.where(original_ids == original_id)[0]
            if len(matches) == 0:
                raise ValueError(f"original_id={original_id} not found in {h5_path}")
            idx = int(matches[0])

        sample = {
            "original_id": _as_scalar(f["original_id"][idx]),
            "masked_region_map": f["masked_region_map"][idx],
            "x_masked_image": f["x_masked_image"][idx],
            "y_target_image": f["y_target_image"][idx],
            "y_recon_image": f["y_recon_image"][idx],
        }

        if "y_specz_redshift" in f:
            sample["y_specz_redshift"] = _as_scalar(f["y_specz_redshift"][idx])
        else:
            sample["y_specz_redshift"] = None

    return sample


def plot_original_mask_recon_by_ratio(
    mask_to_hdf5_path,
    save_path,
    original_id=None,
    sample_idx=0,
    band_names=BAND_NAMES,
    cmap_name="inferno",
    save_pdf=True,
):
    mask_to_hdf5_path = {str(k): Path(v) for k, v in mask_to_hdf5_path.items()}
    present_masks = [m for m in MASK_ORDER if m in mask_to_hdf5_path]

    if not present_masks:
        raise ValueError("No valid mask ratios found in mask_to_hdf5_path")

    if original_id is None:
        original_id = find_common_original_ids(mask_to_hdf5_path)[int(sample_idx)]

    samples = {
        mask_ratio: load_hdf5_sample_by_original_id(
            mask_to_hdf5_path[mask_ratio],
            original_id=original_id,
        )
        for mask_ratio in present_masks
    }

    target = np.asarray(samples[present_masks[0]]["y_target_image"], dtype=np.float32)

    rows = [("Original", target)]

    for mask_ratio in present_masks:
        sample = samples[mask_ratio]

        x_masked_nan = set_masked_values(
            sample["x_masked_image"],
            sample["masked_region_map"],
            value=np.nan,
        )

        rows.append((f"Masked X\nmask={mask_ratio}", x_masked_nan))
        rows.append((f"Recon Y\nmask={mask_ratio}", sample["y_recon_image"]))

    num_rows = len(rows)
    num_bands = len(band_names)

    fig, axes = plt.subplots(
        num_rows,
        num_bands,
        figsize=(2.15 * num_bands, 1.75 * num_rows),
        squeeze=False,
    )

    for row_idx, (row_label, row_images) in enumerate(rows):
        for band_idx, band_name in enumerate(band_names):
            ax = axes[row_idx, band_idx]

            vmin = float(np.nanmin(target[band_idx]))
            vmax = float(np.nanmax(target[band_idx]))

            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-6

            ax.imshow(
                np.asarray(row_images[band_idx], dtype=np.float32),
                cmap=cmap_name,
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            if row_idx == 0:
                ax.set_title(f"{band_name.upper()} band", fontsize=10, pad=6)

            if band_idx == 0:
                ax.set_ylabel(
                    row_label,
                    fontsize=10,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=34,
                )

    title = (
        "Original, Masked Inputs, and Reconstructions by Mask Ratio"
        f" | original_id={original_id}"
    )

    redshift = samples[present_masks[0]].get("y_specz_redshift", None)
    if redshift is not None:
        title += f" | redshift={float(redshift):.3f}"

    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0.06, 0.0, 1.0, 0.975])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if save_pdf:
        fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {save_path}")
    if save_pdf:
        print(f"Saved: {save_path.with_suffix('.pdf')}")

    return save_path


def main_cross_mask_samples():
    parser = argparse.ArgumentParser(
        description="Plot one sample across MAE mask ratios."
    )

    parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project root directory. Default: current directory.",
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="experiments",
        help="Experiment directory under project-dir. Default: experiments.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="testing",
        help="Split to plot: training, validation, testing, or aliases train/valid/test.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/sample_plots",
        help="Output directory used when --output is not provided.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit output PNG path.",
    )
    parser.add_argument(
        "--run-template",
        type=str,
        default="train_mae_medium_{person}_mask_{mask_ratio}",
        help=(
            "Optional run-name template using {person} and {mask_ratio}. "
            "Example: train_mae_{person}_{mask_ratio}"
        ),
    )

    parser.add_argument("--mask-0", default=None, help="Optional path to mask_ratio=0.0 HDF5 output")
    parser.add_argument("--mask-25", default=None, help="Optional path to mask_ratio=0.25 HDF5 output")
    parser.add_argument("--mask-50", default=None, help="Optional path to mask_ratio=0.5 HDF5 output")
    parser.add_argument("--mask-75", default=None, help="Optional path to mask_ratio=0.75 HDF5 output")

    parser.add_argument("--original-id", type=int, default=None, help="Specific original_id to plot")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index into shared original_id list")
    parser.add_argument("--png-only", action="store_true", help="Only save PNG, not PDF")

    args = parser.parse_args()

    explicit_paths = {
        "0.0": args.mask_0,
        "0.25": args.mask_25,
        "0.5": args.mask_50,
        "0.75": args.mask_75,
    }

    has_any_explicit = any(path is not None for path in explicit_paths.values())
    has_all_explicit = all(path is not None for path in explicit_paths.values())

    if has_any_explicit and not has_all_explicit:
        missing = [mask for mask, path in explicit_paths.items() if path is None]
        raise ValueError(
            "If using explicit mask paths, provide all four. "
            f"Missing masks: {missing}"
        )

    if has_all_explicit:
        mask_to_hdf5_path = explicit_paths
        split = "custom"
    else:
        split = normalize_split_name(args.split)
        mask_to_hdf5_path = build_default_mask_paths(
            project_dir=args.project_dir,
            split=split,
            experiments_dir=args.experiments_dir,
            run_template=args.run_template,
        )

    if args.output is None:
        if split == "custom":
            save_path = Path(args.output_dir) / f"mae_cross_mask_sample_{args.sample_idx}.png"
        else:
            save_path = make_cross_mask_output_path(
                output_dir=args.output_dir,
                split=split,
                original_id=args.original_id,
                sample_idx=args.sample_idx,
            )
    else:
        save_path = Path(args.output)

    plot_original_mask_recon_by_ratio(
        mask_to_hdf5_path=mask_to_hdf5_path,
        save_path=save_path,
        original_id=args.original_id,
        sample_idx=args.sample_idx,
        save_pdf=not args.png_only,
    )

if __name__ == "__main__":
    main_cross_mask_samples()