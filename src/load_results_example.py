from src.utils.common import pt, Path, GalaxiesMLDataset
from src.preprocess_data import Normalize

def main():

    dataset_size="tiny"

    # Update these for whomever's results
    first_name = "leslie"
    mask_ratio = "0.0"

    # Update batchsize to whatever 
    batch_size = 32

    # this assumes you are running the program from the project root
    project_path = "."

    # load the results with reconstructed galaxies, original galaxy images, and original redshift values
    results_path = Path(project_path) / f"experiments/train_mae_{dataset_size}_{first_name}_{mask_ratio}/artifacts/samples"
    train_path = results_path / "training_outputs_best.hdf5"
    valid_path = results_path / "validation_outputs_best.hdf5"
    test_path = results_path / "testing_outputs_best.hdf5"


    train_data = GalaxiesMLDataset(
        train_path,
        input_key="y_recon_image",
        target_key="y_specz_redshift",
        auxiliary_key="y_target_image"
    )

    valid_data = GalaxiesMLDataset(
        valid_path,
        input_key="y_recon_image",
        target_key="y_specz_redshift",
        auxiliary_key="y_target_image",
    )

    test_data = GalaxiesMLDataset(
        test_path,
        input_key="y_recon_image",
        target_key="y_specz_redshift",
        auxiliary_key="y_target_image",
    )

    # OPTIONAL: load normalization object from preprocessing to convert normalized data to the original numerical scale
    transform_path = Path(project_path) / f"data/preprocessed/galaxiesml_{dataset_size}/normalize_transform.pth"
    transform = pt.load(transform_path, map_location="cpu", weights_only=False)

    train_loader = pt.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = pt.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = pt.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    loaders = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
    }

    for split, loader in loaders.items():
        print(f"\n{split.upper()}")

        for batch_idx, (x_recon, y_redshift, x_original) in enumerate(loader):

            # example loading the data one sample at a time
            print("Batch:", batch_idx)
            print("x_recon shape:", x_recon.shape)
            print("y_redshift shape:", y_redshift.shape)
            print("x_original shape:", x_original.shape)

            # example unnormalizing the data if you need that for evaluation/analysis/etc
            x_recon_raw = transform.inverse_transform(x_recon)
            x_original_raw = transform.inverse_transform(x_original)

            # you can use the exact method below to unnormalize your predictions too
            y_redshift_raw = transform.inverse_transform_specz(y_redshift)
    
            print("\nNormalized values")
            print("x_recon min/max:", x_recon.min().item(), x_recon.max().item())
            print("x_original min/max:", x_original.min().item(), x_original.max().item())
            print("y_redshift min/max:", y_redshift.min().item(), y_redshift.max().item())

            print("\nInverse transformed values")
            print("x_recon_raw min/max:", x_recon_raw.min().item(), x_recon_raw.max().item())
            print("x_original_raw min/max:", x_original_raw.min().item(), x_original_raw.max().item())
            print("y_redshift_raw min/max:", y_redshift_raw.min().item(), y_redshift_raw.max().item())
            
            break


if __name__ == "__main__":
    main()