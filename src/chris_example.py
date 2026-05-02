from src.utils.common import pt, Path, GalaxiesMLDataset

# need to import Normalize to load/use transform object
from src.preprocess_data import Normalize

def load_results_example(project_path, batch_size = 32):

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

    loaders = {}

    for first_name, mask_ratio in runs.items():

        # path to everyones results
        results_path = Path(project_path) / f"experiments/train_mae_medium_{first_name}_mask_{mask_ratio}/artifacts/samples"

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
                data_keys={
                    "x_masked_image": "x_masked_image",         # Masked Galaxy Image (Encoder Input)
                    "x_recon_image": "y_recon_image",           # Recon Galaxy Image (Decoder Output)
                    "x_original_image": "y_target_image",       # Original Galaxy Image    
                    "y_original_redshift": "y_specz_redshift",  # Original Redshift Scalar
                    "original_id": "original_id",               # Original Id from source dataset
                },
                return_dict=True,
            )

            # pytorch datalaoders
            loaders[first_name][split] = pt.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
            )


    # load the data and make sure it is correct
    for first_name, split_loaders in loaders.items():
        if len(split_loaders) == 0:
            continue

        print(f"\n{first_name.upper()}")

        for split, loader in split_loaders.items():
            print(f"\n[{split.upper()} Split]")

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




# update to project root: /path/to/the-space-cats-project
project_path = Path(__file__).resolve().parents[1]

# run the example
load_results_example(project_path, batch_size=32)