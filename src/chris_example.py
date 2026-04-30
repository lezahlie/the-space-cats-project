from src.utils.common import pt, Path, GalaxiesMLDataset

# need to import Normalize to load/use transform object
from src.preprocess_data import Normalize

def main():
    # Update batchsize
    batch_size = 32

    # got up one level from the script path
    project_path = Path(__file__).resolve().parents[1]

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
    transform_path = Path(project_path) / "data/preprocessed/galaxiesml_medium/normalize_transform.pth"

    # for loading the transform object
    transform = None
    if transform_path.exists():
        transform = pt.load(transform_path, map_location="cpu", weights_only=False)
    else:
        print("Skipping transform. Missing:", transform_path)

    loaders = {}

    for first_name, mask_ratio in runs.items():

        # path to everyones results
        results_path = Path(project_path) / f"experiments/train_mae_{first_name}_{mask_ratio}/artifacts/samples"

        # pytorch dataloaders
        loaders[first_name] = {}

        for split, file_name in split_files.items():
            data_path = results_path / file_name

            if not data_path.exists():
                print(f"Skipping {first_name} {split}. Missing: {data_path}")
                continue

            # load results and original data
            dataset = GalaxiesMLDataset(
                data_path,
                input_key="y_recon_image",          # reconstructed galaxy image
                target_key="y_specz_redshift",      # ground truth redshift
                auxiliary_key="y_target_image",     # original galaxy image
            )

            # pytorch datalaoders
            loaders[first_name][split] = pt.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
            )

    # load the data and make sure it's correct
    for first_name, split_loaders in loaders.items():
        if len(split_loaders) == 0:
            continue

        print(f"\n{first_name.upper()}")

        for split, loader in split_loaders.items():
            print(f"\n[{split.upper()} Split]")

            for batch_idx, (x_recon, y_redshift, x_original) in enumerate(loader):
                print("Batch:", batch_idx)
                print("x_recon shape:", x_recon.shape)
                print("y_redshift shape:", y_redshift.shape)
                print("x_original shape:", x_original.shape)

                print("\nNormalized values")
                print("x_recon min/max:", x_recon.min().item(), x_recon.max().item())
                print("x_original min/max:", x_original.min().item(), x_original.max().item())
                print("y_redshift min/max:", y_redshift.min().item(), y_redshift.max().item())

                if transform is not None:
                    x_recon_raw = transform.inverse_transform(x_recon)
                    x_original_raw = transform.inverse_transform(x_original)
                    y_redshift_raw = transform.inverse_transform_specz(y_redshift)

                    print("\nInverse transformed values")
                    print("x_recon_raw min/max:", x_recon_raw.min().item(), x_recon_raw.max().item())
                    print("x_original_raw min/max:", x_original_raw.min().item(), x_original_raw.max().item())
                    print("y_redshift_raw min/max:", y_redshift_raw.min().item(), y_redshift_raw.max().item())

                break


if __name__ == "__main__":
    main()