from src.utils.logger import get_logger, log_execution_time
from src.utils.common import h5py, np, pd, Path, pt, GalaxiesMLDataset
from src.utils.device import SetupDevice
# need to import Normalize to load/use transform object
from src.preprocess_data import Normalize
import matplotlib.pyplot as plt


nn=pt.nn
F=nn.functional

BATCH_SIZE=32

# stats from the raw medium dataset
TRAIN_SPLIT_STATS = {
    "image_mean": 0.6587017379905387,
    "image_std": 3.159784986458364,
    "image_median": 0.08223292231559753,
    "image_min": -137.3596649169922,
    "image_max": 852.43115234375,
    "redshift_mean": 0.5946988821332343,
    "redshift_std": 0.5720175378707262,
    "redshift_median": 0.4889649897813797,
    "redshift_min": 0.013700000010430813,
    "redshift_max": 3.9086599349975586
}


# arch from CNN experiments:
class RedshiftCNN(nn.Module):
    def __init__(self, num_channels=5, output_dim=1, input_image_size=(64, 64)):
        super(RedshiftCNN, self).__init__()

        # Input: 5 channels, 64x64 image
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # Apply Conv -> ReLU -> Pool for each block
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        # Flatten the output for the fully connected layers
        x = pt.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    


@log_execution_time
def main():

    logger = get_logger()

    device = SetupDevice.setup_torch_device(
        4,
        False,
        gpu_list=[0],
        gpu_memory=0.9,
        random_seed=42,
        deterministic=False
    )

    # update to project root: /path/to/the-space-cats-project
    project_path=Path(__file__).resolve().parents[1]

    # for getting result folders
    runs = {
        "leslie": "0.0",
        "charlie": "0.25",
        "chris": "0.5",
        "wen": "0.75",
    }

    # for getting the dataset results from the best model
    test_results = "testing_outputs_best.hdf5"

    # load normalization object from preprocessing
    transform_path = project_path / "data/preprocessed/galaxiesml_medium/normalize_transform.pth"

    # for loading the transform object
    transform = None
    if transform_path.exists():
        transform = pt.load(transform_path, map_location="cpu", weights_only=False)
    else:
        print("Skipping transform. Missing:", transform_path)

    test_loaders = {}


    for first_name, mask_ratio in runs.items():

        # path to everyones results
        results_path = project_path / f"experiments/train_mae_medium_{first_name}_mask_{mask_ratio}/artifacts/samples"

        # pytorch dataloaders
        test_loaders[mask_ratio] = {}


        data_path = results_path / test_results

        if not data_path.exists():
            print(f"Skipping {first_name}. Missing: {data_path}")
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
        test_loaders[mask_ratio] = pt.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )


    # init new model
    model = RedshiftCNN() 
    # load model weight
    state_dict = pt.load(project_path / "redshift_cnn_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(device)

    # put model in eval mode
    model.eval()

    # stats for standardization
    image_mean = pt.tensor(TRAIN_SPLIT_STATS["image_mean"], dtype=pt.float32, device=device).view(1, -1, 1, 1)
    image_std = pt.tensor(TRAIN_SPLIT_STATS["image_std"], dtype=pt.float32, device=device).view(1, -1, 1, 1)

    redshift_mean = pt.tensor(TRAIN_SPLIT_STATS["redshift_mean"], dtype=pt.float32, device=device)
    redshift_std = pt.tensor(TRAIN_SPLIT_STATS["redshift_std"], dtype=pt.float32, device=device)

    with pt.no_grad():

        # load the testing splits by mask ratio
        for mask_ratio, test_loader in test_loaders.items():

            # loop over testing loader in batches
            for batch_idx, batch in enumerate(test_loader):

                # ALREADY IN SAME ORDER, WAS NOT SHUFFLED
                original_id = batch.original_id

                x_recon_image = transform.inverse_transform(batch.x_recon_image).to(device)
                x_original_image = transform.inverse_transform(batch.x_original_image).to(device)
                y_original_redshift = transform.inverse_transform_specz(batch.y_original_redshift).to(device)

                ##############################################################
                ##TODO Standardize the x_original_image,  x_recon_image, and _original_redshift
                ##############################################################
                x_recon_image_std = None
                x_original_image_std = None
    
                y_original_redshift_std = None

                ##############################################################
                ##TODO Predict the redshift values
                ##############################################################
                # only need to do this once they are all the same for every mask
                pred_redshift_orig_std = model(x_original_image_std).view(-1, 1)   

                pred_redshift_recon_std = model(x_recon_image_std).view(-1, 1)

                ##############################################################
                ##TODO Compute the losses L1 / L2 and save them somehow
                ##############################################################
                original_l2 = F.mse_loss(
                    pred_redshift_orig_std,
                    y_original_redshift_std,
                    reduction="none",
                ).view(-1)

                recon_l2 = F.mse_loss(
                    pred_redshift_recon_std,
                    y_original_redshift_std,
                    reduction="none",
                ).view(-1)

                original_l1 = F.l1_loss(
                    pred_redshift_orig_std,
                    y_original_redshift_std,
                    reduction="none",
                ).view(-1)

                recon_l1 = F.l1_loss(
                    pred_redshift_recon_std,
                    y_original_redshift_std,
                    reduction="none",
                ).view(-1)

                ##############################################################
                ##TODO IF NEEDED: unstandardize the x_original_image,  x_recon_image, and _original_redshift
                ##############################################################

                ##############################################################
                ##TODO compute other metrics and save them somehow
                ##############################################################

        ##############################################################
        ##TODO move results from device to cpu as numpy
        ##############################################################

    ##############################################################
    ##TODO make your plots for the results and write about them in paper 
    ##############################################################


if __name__ == "__main__":
    from src.utils.logger import init_shared_logger
    logger = init_shared_logger(__file__, log_stdout=True, log_stderr=True)
    try:
        pt.multiprocessing.set_sharing_strategy('file_system')
        main()
    except Exception as e:
        logger.error(e)
