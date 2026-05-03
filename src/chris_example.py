from src.utils.common import pt, Path, GalaxiesMLDataset

# need to import Normalize to load/use transform object
from src.preprocess_data import Normalize


def _original_id_to_key(original_id):
    if hasattr(original_id, "detach"):
        original_id = original_id.detach().cpu()

        if original_id.numel() == 1:
            original_id = original_id.item()
        else:
            original_id = tuple(original_id.flatten().tolist())

    elif hasattr(original_id, "item"):
        original_id = original_id.item()

    if isinstance(original_id, bytes):
        original_id = original_id.decode("utf-8")

    return original_id


def _get_sample_original_id(sample):
    if isinstance(sample, dict):
        return _original_id_to_key(sample["original_id"])

    return _original_id_to_key(sample.original_id)


def _build_original_id_index(dataset):
    id_to_idx = {}

    for idx in range(len(dataset)):
        original_id = _get_sample_original_id(dataset[idx])

        if original_id in id_to_idx:
            raise ValueError(f"Duplicate original_id found: {original_id}")

        id_to_idx[original_id] = idx

    return id_to_idx


def load_results_example(project_path, batch_size=32):

    # for getting result folders
    runs = {
        "leslie": "0.0",
        "charlie": "0.25",
        "chris": "0.5",
        "wen": "0.75",
    }

    # for getting the dataset results from the best model
    split_files = {
        "train": "training_outputs_best.hdf5",
        "valid": "validation_outputs_best.hdf5",
        "test": "testing_outputs_best.hdf5",
    }

    # OPTIONAL: load normalization object from preprocessing
    transform_path = Path(project_path) / "data/preprocessed/galaxiesml_tiny/normalize_transform.pth"

    # for loading the transform object
    transform = None
    if transform_path.exists():
        transform = pt.load(transform_path, map_location="cpu", weights_only=False)
    else:
        print("Skipping transform. Missing:", transform_path)

    datasets = {}

    for first_name, mask_ratio in runs.items():

        # path to everyones results
        results_path = Path(project_path) / f"experiments/train_mae_medium_{first_name}_mask_{mask_ratio}/artifacts/samples"

        # pytorch datasets
        datasets[first_name] = {}

        for split, file_name in split_files.items():
            data_path = results_path / file_name

            if not data_path.exists():
                print(f"Skipping {first_name} {split}. Missing: {data_path}")
                continue

            # load results and original data
            dataset = GalaxiesMLDataset(
                data_path,
                data_keys={
                    "x_masked_image": "x_masked_image",         # Masked Galaxy Image (Encoder Input)
                    "x_recon_image": "y_recon_image",           # Recon Galaxy Image (Decoder Output)
                    "x_original_image": "y_target_image",       # Original Galaxy Image
                    "y_original_redshift": "y_specz_redshift",  # Original Redshift Scalar
                    "original_id": "original_id",               # Original Id from source dataset
                },
                return_dict=True,
            )

            datasets[first_name][split] = dataset

    # match original_ids across all four mask ratios
    matched_loaders = {}
    matched_ids_by_split = {}

    for split in split_files.keys():

        # make sure every mask ratio has this split
        available_names = [
            first_name
            for first_name in runs.keys()
            if split in datasets[first_name]
        ]

        if len(available_names) != len(runs):
            print(f"Skipping {split}. Only found {available_names}, expected all runs.")
            continue

        # map each original_id to its dataset index
        id_maps = {
            first_name: _build_original_id_index(datasets[first_name][split])
            for first_name in available_names
        }

        # keep only original_ids that exist for every mask ratio
        common_ids = set.intersection(*[
            set(id_map.keys())
            for id_map in id_maps.values()
        ])

        # sort so every mask ratio has the same order
        matched_ids = sorted(common_ids)
        matched_ids_by_split[split] = matched_ids

        print(f"\n{split.upper()} matched original_ids: {len(matched_ids)}")

        matched_loaders[split] = {}

        for first_name in runs.keys():

            # get indices in the same original_id order for each mask ratio
            matched_indices = [
                id_maps[first_name][original_id]
                for original_id in matched_ids
            ]

            matched_dataset = pt.utils.data.Subset(
                datasets[first_name][split],
                matched_indices,
            )

            # pytorch dataloaders
            matched_loaders[split][first_name] = pt.utils.data.DataLoader(
                matched_dataset,
                batch_size=batch_size,
                shuffle=False,
            )

    # load the data and make sure it is correct
    for split, split_loaders in matched_loaders.items():

        print(f"\n[{split.upper()} Split]")

        for first_name, loader in split_loaders.items():

            print(f"\n{first_name.upper()} mask_ratio={runs[first_name]}")

            for batch_idx, batch in enumerate(loader):
                x_masked_image = batch.x_masked_image
                x_recon_image = batch.x_recon_image
                x_original_image = batch.x_original_image
                y_original_redshift = batch.y_original_redshift
                original_id = batch.original_id

                print("Batch:", batch_idx)
                print("original_id:", original_id[:3])
                print("x_masked_image shape:", x_masked_image.shape)
                print("x_recon_image shape:", x_recon_image.shape)
                print("x_original_image shape:", x_original_image.shape)
                print("y_original_redshift shape:", y_original_redshift.shape)

                print("\nNormalized values")
                print("x_masked_image min/max:", x_masked_image.min().item(), x_masked_image.max().item())
                print("x_recon_image min/max:", x_recon_image.min().item(), x_recon_image.max().item())
                print("x_original_image min/max:", x_original_image.min().item(), x_original_image.max().item())
                print("y_original_redshift min/max:", y_original_redshift.min().item(), y_original_redshift.max().item())

                if transform is not None:
                    x_masked_raw = transform.inverse_transform(x_masked_image)
                    x_recon_raw = transform.inverse_transform(x_recon_image)
                    x_original_raw = transform.inverse_transform(x_original_image)
                    y_redshift_raw = transform.inverse_transform_specz(y_original_redshift)

                    print("\nInverse transformed values")
                    print("x_masked_raw min/max:", x_masked_raw.min().item(), x_masked_raw.max().item())
                    print("x_recon_raw min/max:", x_recon_raw.min().item(), x_recon_raw.max().item())
                    print("x_original_raw min/max:", x_original_raw.min().item(), x_original_raw.max().item())
                    print("y_redshift_raw min/max:", y_redshift_raw.min().item(), y_redshift_raw.max().item())

                break

    return matched_loaders, matched_ids_by_split


# update to project root: /path/to/the-space-cats-project
project_path = Path(__file__).resolve().parents[1]

# run the example
matched_loaders, matched_ids_by_split = load_results_example(project_path, batch_size=32)


def _batch_ids_to_list(original_id):
    if hasattr(original_id, "detach"):
        original_id = original_id.detach().cpu()

        if original_id.ndim == 1:
            return [_original_id_to_key(x) for x in original_id]

        return [_original_id_to_key(x) for x in original_id]

    return [_original_id_to_key(x) for x in original_id]


def check_matched_loaders(matched_loaders, matched_ids_by_split):

    for split, split_loaders in matched_loaders.items():

        print(f"\nChecking {split.upper()}")

        expected_ids = matched_ids_by_split[split]
        seen_ids = []

        loader_items = list(split_loaders.items())

        for batch_idx, batches in enumerate(zip(*[loader for _, loader in loader_items])):

            batch_ids_by_run = {}

            for (first_name, _), batch in zip(loader_items, batches):
                batch_ids_by_run[first_name] = _batch_ids_to_list(batch.original_id)

            reference_name = loader_items[0][0]
            reference_ids = batch_ids_by_run[reference_name]

            for first_name, original_ids in batch_ids_by_run.items():
                assert original_ids == reference_ids, (
                    f"Mismatch in {split}, batch {batch_idx}: "
                    f"{reference_name} ids do not match {first_name} ids"
                )

            seen_ids.extend(reference_ids)

        assert seen_ids == expected_ids, (
            f"{split} IDs are aligned across loaders, but do not match matched_ids_by_split"
        )

        print(f"PASSED: {split} has {len(seen_ids)} matched original_ids across all mask ratios")


check_matched_loaders(matched_loaders, matched_ids_by_split)