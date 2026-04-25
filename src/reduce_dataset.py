from src.utils.logger import get_logger
from src.utils.common import  argparse,  h5py, np, pd, Path, save_to_json, make_tar_gz
import matplotlib.pyplot as plt


def process_args():
    parser = argparse.ArgumentParser(description="Create reduced GalaxiesML HDF5 datasets")

    parser.add_argument(
        "--input-folder",
        dest="input_folder",
        type=str,
        required=True,
        help="Path to folder containing raw GalaxiesML HDF5 files",
    )

    parser.add_argument(
        "--output-folder",
        dest="output_folder",
        type=str,
        required=True,
        help="Path to output folder for reduced datasets",
    )

    parser.add_argument(
        "--analysis-only",
        dest="analysis_only",
        action="store_true",
        help="Only regenerate redshift plots and stats from existing reduced HDF5 files; do not rewrite datasets or archives",
    )

    return parser.parse_args()



def read_redshift(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        return f["specz_redshift"][:]


def compute_redshift_iqr_stats(
    y,
    dataset_name,
    split_name,
    source_name,
    outlier_factor=1.5,
    extreme_factor=3.0,
):
    y = pd.Series(np.asarray(y), dtype="float64")
    y = y.replace([np.inf, -np.inf], np.nan).dropna()

    if len(y) == 0:
        raise ValueError(f"{dataset_name} {split_name} {source_name} has no finite redshift values")

    q1 = y.quantile(0.25)
    median = y.quantile(0.50)
    q3 = y.quantile(0.75)
    iqr = q3 - q1

    outlier_low = q1 - outlier_factor * iqr
    outlier_high = q3 + outlier_factor * iqr

    extreme_low = q1 - extreme_factor * iqr
    extreme_high = q3 + extreme_factor * iqr

    outlier_mask = (y < outlier_low) | (y > outlier_high)
    extreme_mask = (y < extreme_low) | (y > extreme_high)

    return {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "source_name": source_name,

        "size": int(len(y)),
        "min": float(y.min()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(y.max()),
        "iqr": float(iqr),

        "outlier_factor": float(outlier_factor),
        "outlier_low": float(outlier_low),
        "outlier_high": float(outlier_high),
        "outlier_count": int(outlier_mask.sum()),
        "outlier_fraction": float(outlier_mask.mean()),

        "extreme_factor": float(extreme_factor),
        "extreme_low": float(extreme_low),
        "extreme_high": float(extreme_high),
        "extreme_count": int(extreme_mask.sum()),
        "extreme_fraction": float(extreme_mask.mean()),
    }


def compare_redshift_hist(
    full_y,
    sampled_y,
    out_path,
    dataset_name="",
    split_name="",
    num_bins=50,
    outlier_factor=1.5,
    extreme_factor=3.0,
):
    full_y = pd.Series(np.asarray(full_y), dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    sampled_y = pd.Series(np.asarray(sampled_y), dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()

    full_stats = compute_redshift_iqr_stats(
        full_y,
        dataset_name=dataset_name,
        split_name=split_name,
        source_name="full",
        outlier_factor=outlier_factor,
        extreme_factor=extreme_factor,
    )

    sampled_stats = compute_redshift_iqr_stats(
        sampled_y,
        dataset_name=dataset_name,
        split_name=split_name,
        source_name="sampled",
        outlier_factor=outlier_factor,
        extreme_factor=extreme_factor,
    )

    stats_df = pd.DataFrame([full_stats, sampled_stats])

    # stats_text = (
    #     f"Full Size = {full_stats['size']:,}\sample_size"
    #     f"Sample Size = {sampled_stats['size']:,}\sample_size"
    #     f"Sample Fraction = {sampled_stats['size'] / full_stats['size']:.3f}\sample_size"
    #     f"Full IQR = {full_stats['iqr']:.5f}\sample_size"
    #     f"Sample IQR = {sampled_stats['iqr']:.5f}\sample_size"
    #     f"IQR Difference = {sampled_stats['iqr'] - full_stats['iqr']:+.5f}\sample_size"
    #     f"Outlier Rule = {outlier_factor:.1f} * IQR\sample_size"
    #     f"Extreme Rule = {extreme_factor:.1f} * IQR\sample_size"
    #     f"Full Outliers = {full_stats['outlier_count']:,} ({full_stats['outlier_fraction']:.3f})\sample_size"
    #     f"Sample Outliers = {sampled_stats['outlier_count']:,} ({sampled_stats['outlier_fraction']:.3f})\sample_size"
    #     f"Full Extreme = {full_stats['extreme_count']:,} ({full_stats['extreme_fraction']:.3f})\sample_size"
    #     f"Sample Extreme = {sampled_stats['extreme_count']:,} ({sampled_stats['extreme_fraction']:.3f})"
    # )
    
    stats_text = (
        f"Full N = {full_stats['size']:,}\n"
        f"Sample N = {sampled_stats['size']:,}\n"
        f"IQR Diff = {sampled_stats['iqr'] - full_stats['iqr']:+.5f}\n"
        f"Outliers = {sampled_stats['outlier_fraction']:.3f} "
        f"vs {full_stats['outlier_fraction']:.3f}\n"
        f"Extreme = {sampled_stats['extreme_fraction']:.3f} "
        f"vs {full_stats['extreme_fraction']:.3f}"
    )

    title = "Full vs sampled redshift distribution"
    if dataset_name:
        title = f"{title} - {dataset_name}"
    if split_name:
        title = f"{title} - {split_name}"

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")
    ax.set_facecolor("white")

    ax.hist(full_y, bins=num_bins, density=True, alpha=0.5, label="Full", color="red")
    ax.hist(sampled_y, bins=num_bins, density=True, alpha=0.5, label="Sampled", color="blue")
    ax.grid(True)
    ax.set_xlabel("Specz_Redshift", fontsize=18)
    ax.set_ylabel("Density", fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.legend()
    ax.tick_params(axis="both", labelsize=16)

    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black",
            alpha=1.0,
        ),
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stats_df


def stratified_sample_indices(y, sample_size, num_bins=20, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    sample_size = min(sample_size, len(y))

    # Quantile bins keep roughly equal population per bin.
    # This is better than fixed-width bins if redshift is highly skewed.
    edges = np.quantile(y, np.linspace(0.0, 1.0, num_bins + 1))
    edges = np.unique(edges)

    selected = []

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]

        if i == len(edges) - 2:
            bin_idx = np.where((y >= lo) & (y <= hi))[0]
        else:
            bin_idx = np.where((y >= lo) & (y < hi))[0]

        if len(bin_idx) == 0:
            continue

        # proportional allocation
        k = int(round(sample_size * len(bin_idx) / len(y)))
        k = min(k, len(bin_idx))

        if k > 0:
            selected.append(rng.choice(bin_idx, size=k, replace=False))

    selected = np.concatenate(selected) if selected else np.array([], dtype=np.int64)

    # Fix rounding mismatch.
    if len(selected) > sample_size:
        selected = rng.choice(selected, size=sample_size, replace=False)
    elif len(selected) < sample_size:
        remaining = np.setdiff1d(np.arange(len(y)), selected)
        extra = rng.choice(remaining, size=sample_size - len(selected), replace=False)
        selected = np.concatenate([selected, extra])

    return np.sort(selected)


def save_reduced_hdf5_with_ids(
    src_hdf5_path,
    dst_hdf5_path,
    sample_size=1000,
    image_dtype=np.float32,
    chunk_size=512,
    save_sample_id=False,
    seed=42,
    stratify=True,
    num_bins=50
):
    with h5py.File(src_hdf5_path, "r") as f_in:
        images = f_in["image"]
        redshift = f_in["specz_redshift"]
        object_id = f_in["object_id"]

        total = len(images)
        sample_size = min(sample_size, total)

        full_redshift = redshift[:]

        if stratify:
            selected_indices = stratified_sample_indices(
                y=full_redshift,
                sample_size=sample_size,
                num_bins=num_bins,
                seed=seed,
            )
        else:
            rng = np.random.default_rng(seed)
            selected_indices = np.sort(rng.choice(total, size=sample_size, replace=False))

        image_shape = images.shape[1:]
        with h5py.File(dst_hdf5_path, "w") as f_out:
            dset_image = f_out.create_dataset(
                "image",
                shape=(sample_size, *image_shape),
                dtype=image_dtype,
                chunks=(min(chunk_size, sample_size), *image_shape),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            dset_redshift = f_out.create_dataset(
                "specz_redshift",
                shape=(sample_size,),
                dtype=np.float32,
                chunks=(min(chunk_size, sample_size),),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            dset_object_id = f_out.create_dataset(
                "object_id",
                shape=(sample_size,),
                dtype=np.int64,
                chunks=(min(chunk_size, sample_size),),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            dset_source_index = f_out.create_dataset(
                "source_index",
                shape=(sample_size,),
                dtype=np.int64,
                chunks=(min(chunk_size, sample_size),),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            if save_sample_id:
                str_dt = h5py.string_dtype(encoding="utf-8")
                dset_sample_id = f_out.create_dataset(
                    "sample_id",
                    shape=(sample_size,),
                    dtype=str_dt,
                )

            for out_start in range(0, sample_size, chunk_size):
                out_end = min(out_start + chunk_size, sample_size)
                src_idx = selected_indices[out_start:out_end]

                x = images[src_idx].astype(np.float32)

                dset_image[out_start:out_end] = x.astype(image_dtype)
                dset_redshift[out_start:out_end] = redshift[src_idx].astype(np.float32)
                dset_object_id[out_start:out_end] = object_id[src_idx]
                dset_source_index[out_start:out_end] = src_idx.astype(np.int64)

                if save_sample_id:
                    dset_sample_id[out_start:out_end] = np.array(
                        [f"sample_{i:06d}" for i in src_idx],
                        dtype=object,
                    )

                print(f"Wrote {out_end}/{sample_size}")

            f_out.attrs["num_samples"] = sample_size
            f_out.attrs["subset_seed"] = seed
            f_out.attrs["stratified"] = bool(stratify)
            f_out.attrs["num_bins"] = num_bins if stratify else 0

        sampled_redshift = full_redshift[selected_indices]

    return full_redshift, sampled_redshift

def main(args):
    logger = get_logger()
    all_stats = []

    dataset_splits = [
        (2000,   500,   500,  "tiny"),    # total = 3000  | smoke tests
        (5000,   1000,  1000, "small"),   # total = 7000  | tuning
        (10000,  2500,  2500, "medium"),  # total = 15000 | final/confirmation
        (20000,  5000,  5000, "large"),   # total = 30000 | optional only
    ]

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    if not input_folder.is_dir():
        raise FileNotFoundError(f"--input-folder does not exist: {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "training": input_folder / "5x64x64_training_with_morphology.hdf5",
        "validation": input_folder / "5x64x64_validation_with_morphology.hdf5",
        "testing": input_folder / "5x64x64_testing_with_morphology.hdf5",
    }

    for split_name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw {split_name} file: {path}")

    base_seed = 42

    for train_s, val_s, test_s, keyword in dataset_splits:
        dataset_name = f"galaxiesml_{keyword}"
        out_dir = output_folder / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        plots_dir = out_dir / "plots"
        metadata_dir = out_dir / "metadata"
        plots_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        split_specs = [
            (
                "training",
                train_s,
                source_paths["training"],
                out_dir / f"5x64x64_training_reduced_{keyword}.hdf5",
            ),
            (
                "validation",
                val_s,
                source_paths["validation"],
                out_dir / f"5x64x64_validation_reduced_{keyword}.hdf5",
            ),
            (
                "testing",
                test_s,
                source_paths["testing"],
                out_dir / f"5x64x64_testing_reduced_{keyword}.hdf5",
            ),
        ]

        dataset_stats = []

        for split_name, split_n, src_path, dst_path in split_specs:
            if args.analysis_only:
                if not dst_path.is_file():
                    raise FileNotFoundError(f"Missing reduced {split_name} file for analysis-only mode: {dst_path}")

                full_y = read_redshift(src_path)
                sampled_y = read_redshift(dst_path)

            else:
                full_y, sampled_y = save_reduced_hdf5_with_ids(
                    src_hdf5_path=src_path,
                    dst_hdf5_path=dst_path,
                    sample_size=split_n,
                    chunk_size=1024,
                    image_dtype=np.float32,
                    seed=base_seed,
                    stratify=True,
                    num_bins=50,
                )

            stats_df = compare_redshift_hist(
                full_y=full_y,
                sampled_y=sampled_y,
                out_path=plots_dir / f"{split_name}_redshift_hist.png",
                dataset_name=dataset_name,
                split_name=split_name,
                num_bins=50,
                outlier_factor=1.5,
                extreme_factor=3.0,
            )

            dataset_stats.append(stats_df)
            all_stats.append(stats_df)

        dataset_stats_df = pd.concat(dataset_stats, ignore_index=True)
        dataset_stats_path = metadata_dir / "redshift_iqr_stats.csv"
        dataset_stats_df.to_csv(dataset_stats_path, index=False)
        logger.info(f"Saved dataset_stats to {dataset_stats_path}")

        metadata_path = metadata_dir / "redshift_iqr_stats.json"
        save_to_json(metadata_path, dataset_stats_df.to_dict(orient="records"),)
        logger.info(f"Saved metadata to {metadata_path}")

        archive_path = output_folder / f"{dataset_name}.tar.gz"
        make_tar_gz(out_dir, archive_path)
        logger.info(f"Saved archive to {archive_path}")

    all_stats_df = pd.concat(all_stats, ignore_index=True)
    all_stats_df.to_csv(output_folder / "redshift_iqr_stats.csv", index=False)
    save_to_json(
        output_folder / "redshift_iqr_stats.json",
        all_stats_df.to_dict(orient="records"),
    )

    logger.info(f"Saved redshift stats to {output_folder / 'redshift_iqr_stats.csv'}")
    logger.info(f"Saved redshift stats to {output_folder / 'redshift_iqr_stats.json'}")

if __name__ == "__main__":
    from src.utils.logger import init_shared_logger
    logger = init_shared_logger(__file__, log_stdout=True, log_stderr=True)
    try:
        args = process_args()
        main(args)
    except Exception as e:
        logger.error(e)
