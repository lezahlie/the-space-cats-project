#!/usr/bin/env python3
import shutil
from src.utils.common import argparse, h5py, np, Path



SPLIT_FILES = [
    "training_outputs_best.hdf5",
    "validation_outputs_best.hdf5",
    "testing_outputs_best.hdf5",
]


def first_unique_indices(ids):
    seen = set()
    keep = []

    for idx, value in enumerate(ids.tolist()):
        if value in seen:
            continue
        seen.add(value)
        keep.append(idx)

    return np.asarray(keep, dtype=np.int64)


def backup_path_for(path):
    backup_path = path.with_name(path.stem + ".with_duplicates.bak" + path.suffix)

    if not backup_path.exists():
        return backup_path

    i = 1
    while True:
        candidate = path.with_name(path.stem + f".with_duplicates.bak{i}" + path.suffix)
        if not candidate.exists():
            return candidate
        i += 1


def copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def dataset_kwargs(src_dataset, out_shape):
    kwargs = {}

    if src_dataset.chunks is not None and len(out_shape) > 0:
        chunks = list(src_dataset.chunks)
        chunks[0] = min(chunks[0], out_shape[0])
        kwargs["chunks"] = tuple(chunks)

    if src_dataset.compression is not None:
        kwargs["compression"] = src_dataset.compression
        kwargs["compression_opts"] = src_dataset.compression_opts

    if src_dataset.shuffle:
        kwargs["shuffle"] = True

    if src_dataset.fletcher32:
        kwargs["fletcher32"] = True

    return kwargs


def copy_group_dedup(src_group, dst_group, keep_idx, original_n_rows, row_chunk_size):
    copy_attrs(src_group, dst_group)

    for name, item in src_group.items():
        if isinstance(item, h5py.Group):
            out_group = dst_group.create_group(name)
            copy_group_dedup(
                item,
                out_group,
                keep_idx=keep_idx,
                original_n_rows=original_n_rows,
                row_chunk_size=row_chunk_size,
            )
            continue

        if not isinstance(item, h5py.Dataset):
            continue

        should_filter = len(item.shape) > 0 and item.shape[0] == original_n_rows

        if not should_filter:
            src_group.copy(item, dst_group, name=name)
            continue

        out_shape = (len(keep_idx),) + item.shape[1:]
        out_dataset = dst_group.create_dataset(
            name,
            shape=out_shape,
            dtype=item.dtype,
            **dataset_kwargs(item, out_shape),
        )
        copy_attrs(item, out_dataset)

        for start in range(0, len(keep_idx), row_chunk_size):
            stop = min(start + row_chunk_size, len(keep_idx))
            idx = keep_idx[start:stop]
            out_dataset[start:stop] = item[idx]


def dedupe_hdf5(path, apply=False, row_chunk_size=512):
    path = Path(path)

    if not path.exists():
        print(f"SKIP missing: {path}")
        return

    with h5py.File(path, "r") as f:
        if "original_id" not in f:
            print(f"SKIP no original_id: {path}")
            return

        ids = f["original_id"][:]

    keep_idx = first_unique_indices(ids)

    n_rows = len(ids)
    n_unique = len(keep_idx)
    n_duplicates = n_rows - n_unique

    if n_duplicates == 0:
        print(f"OK {path} rows={n_rows} unique={n_unique} duplicates=0")
        return

    print(f"FIX {path} rows={n_rows} unique={n_unique} duplicates={n_duplicates}")

    if not apply:
        return

    tmp_path = path.with_name(path.stem + ".dedup.tmp" + path.suffix)

    if tmp_path.exists():
        tmp_path.unlink()

    with h5py.File(path, "r") as src, h5py.File(tmp_path, "w") as dst:
        copy_group_dedup(
            src,
            dst,
            keep_idx=keep_idx,
            original_n_rows=n_rows,
            row_chunk_size=row_chunk_size,
        )

    with h5py.File(tmp_path, "r") as f:
        fixed_ids = f["original_id"][:]
        fixed_unique = len(set(fixed_ids.tolist()))

    if len(fixed_ids) != n_unique or fixed_unique != n_unique:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Verification failed for {path}: "
            f"rows={len(fixed_ids)}, unique={fixed_unique}, expected={n_unique}"
        )

    backup_path = backup_path_for(path)
    shutil.move(str(path), str(backup_path))
    shutil.move(str(tmp_path), str(path))
    shutil.copymode(backup_path, path)

    print(f"  backup: {backup_path}")
    print(f"  saved:  {path}")


def find_result_files(project_dir):
    experiments_dir = project_dir / "experiments"

    run_dirs = sorted(experiments_dir.glob("train_mae_medium_*_mask_*"))

    for run_dir in run_dirs:
        samples_dir = run_dir / "artifacts" / "samples"

        for file_name in SPLIT_FILES:
            yield samples_dir / file_name


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate all MAE exported HDF5 result files by original_id."
    )
    parser.add_argument(
        "--project-dir",
        default=".",
        help="Project root directory. Default: current directory.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually save fixed files. Without this, only reports duplicates.",
    )
    parser.add_argument(
        "--row-chunk-size",
        type=int,
        default=512,
        help="Rows copied per chunk. Default: 512.",
    )

    args = parser.parse_args()
    project_dir = Path(args.project_dir).resolve()

    for path in find_result_files(project_dir):
        dedupe_hdf5(
            path,
            apply=args.apply,
            row_chunk_size=args.row_chunk_size,
        )


if __name__ == "__main__":
    main()