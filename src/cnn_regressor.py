from src.utils.logger import get_logger, log_execution_time
from src.utils.common import h5py, np, pd, Path, pt, GalaxiesMLDataset
from src.utils.device import SetupDevice
from src.preprocess_data import Normalize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


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

# per-channel CNN normalization stats
CHANNEL_MEANS = np.array([0.21533734, 0.45367402, 0.68543416, 0.8832283,  1.0558323],  dtype=np.float32)
CHANNEL_STDS  = np.array([1.0022058,  1.9175421,  3.0707815,  3.8724658,  4.5129385],  dtype=np.float32)
 
# MAE min-max normalization stats (used for de-normalization)
MAE_ORIG_MIN = TRAIN_SPLIT_STATS["image_min"]
MAE_ORIG_MAX = TRAIN_SPLIT_STATS["image_max"]

# arch from CNN experiments:
class RedshiftCNN(nn.Module):
    def __init__(self, num_channels=5, output_dim=1, input_image_size=(64, 64)):
        super(RedshiftCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32,  kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32,  64,  kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64,  128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc1     = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.25)
        self.fc2     = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = pt.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    


@log_execution_time
def main():
    logger = get_logger()
 
    device = SetupDevice.setup_torch_device(
        4, False, gpu_list=[0], gpu_memory=0.9, random_seed=42, deterministic=False
    )
 
    project_path = Path(__file__).resolve().parents[1]
    output_dir   = project_path / "experiments" / "cnn_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
 
    runs = {
        "leslie": "0.0",
        "charlie": "0.25",
        "chris":   "0.5",
        "wen":     "0.75",
    }
 
    # load MAE transform for de-normalization
    transform_path = project_path / "data/preprocessed/galaxiesml_medium/normalize_transform.pth"
    transform = None
    if transform_path.exists():
        transform = pt.load(transform_path, map_location="cpu", weights_only=False)
    else:
        logger.warning(f"Missing transform: {transform_path} — will use manual min-max de-normalization")
 
    # build dataloaders
    test_loaders = {}
    for first_name, mask_ratio in runs.items():
        data_path = (
            project_path
            / f"experiments/train_mae_medium_{first_name}_mask_{mask_ratio}/artifacts/samples"
            / "testing_outputs_best.hdf5"
        )
        if not data_path.exists():
            logger.warning(f"Skipping {first_name} (mask={mask_ratio}): missing {data_path}")
            continue
 
        dataset = GalaxiesMLDataset(
            data_path,
            data_keys={
                "x_masked_image":      "x_masked_image",
                "x_recon_image":       "y_recon_image",
                "x_original_image":    "y_target_image",
                "y_original_redshift": "y_specz_redshift",
                "original_id":         "original_id",
            },
            return_dict=True,
        )
        test_loaders[mask_ratio] = pt.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False
        )
        logger.info(f"Loaded {first_name} mask={mask_ratio} | N={len(dataset)}")
 
    # load CNN
    model = RedshiftCNN()
    model_path = project_path / "redshift_cnn_model.pth"
    state_dict  = pt.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"Loaded CNN from {model_path}")
 
    # per-channel normalization tensors for CNN (shape: 1 x 5 x 1 x 1)
    ch_mean = pt.tensor(CHANNEL_MEANS, dtype=pt.float32, device=device).view(1, 5, 1, 1)
    ch_std  = pt.tensor(CHANNEL_STDS,  dtype=pt.float32, device=device).view(1, 5, 1, 1)
 
    # ------------------------------------------------------------------ #
    # Inference loop
    # ------------------------------------------------------------------ #
    all_rows = []
 
    with pt.no_grad():
        for mask_ratio, loader in test_loaders.items():
            logger.info(f"Running CNN inference | mask_ratio={mask_ratio}")
 
            for batch in loader:
                original_id = batch.original_id
 
                # --- de-normalize from MAE [0,1] scale back to original pixel scale ---
                if transform is not None:
                    x_orig = transform.inverse_transform(batch.x_original_image).to(device)
                    x_recon = transform.inverse_transform(batch.x_recon_image).to(device)
                    y_true  = transform.inverse_transform_specz(batch.y_original_redshift).to(device)
                else:
                    # manual min-max inversion
                    orig_span = MAE_ORIG_MAX - MAE_ORIG_MIN
                    x_orig  = (batch.x_original_image.to(device) * orig_span + MAE_ORIG_MIN)
                    x_recon = (batch.x_recon_image.to(device)    * orig_span + MAE_ORIG_MIN)
                    y_true  = batch.y_original_redshift.to(device).view(-1)
 
                # --- re-normalize for CNN (per-channel z-score) ---
                x_orig_std  = (x_orig  - ch_mean) / ch_std
                x_recon_std = (x_recon - ch_mean) / ch_std
 
                # --- predict redshift ---
                pred_orig  = model(x_orig_std).view(-1)
                pred_recon = model(x_recon_std).view(-1)
 
                # --- compute losses ---
                l1_orig  = F.l1_loss(pred_orig,  y_true, reduction="none")
                l1_recon = F.l1_loss(pred_recon, y_true, reduction="none")
                l2_orig  = F.mse_loss(pred_orig,  y_true, reduction="none")
                l2_recon = F.mse_loss(pred_recon, y_true, reduction="none")
 
                # --- move to CPU and collect ---
                for i, oid in enumerate(original_id):
                    all_rows.append({
                        "mask_ratio":       mask_ratio,
                        "original_id":      oid,
                        "actual_redshift":  y_true[i].item(),
                        "pred_orig":        pred_orig[i].item(),
                        "pred_recon":       pred_recon[i].item(),
                        "delta_orig":       (pred_orig[i]  - y_true[i]).item(),
                        "delta_recon":      (pred_recon[i] - y_true[i]).item(),
                        "l1_orig":          l1_orig[i].item(),
                        "l1_recon":         l1_recon[i].item(),
                        "l2_orig":          l2_orig[i].item(),
                        "l2_recon":         l2_recon[i].item(),
                    })
 
    # ------------------------------------------------------------------ #
    # Save per-image results
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(all_rows)
    per_image_path = output_dir / "cnn_predictions_all_images.csv"
    df.to_csv(per_image_path, index=False)
    logger.info(f"Saved per-image results -> {per_image_path}")
 
    # ------------------------------------------------------------------ #
    # Summary table (MAE, RMSE, mean error, std error per mask ratio)
    # ------------------------------------------------------------------ #
    summary_rows = []
    for mask_ratio, grp in df.groupby("mask_ratio"):
        summary_rows.append({
            "mask_ratio":       mask_ratio,
            "original_mae":     grp["l1_orig"].mean(),
            "original_rmse":    grp["l2_orig"].pow(0.5).mean() if "l2_orig" in grp else np.nan,
            "recon_mae":        grp["l1_recon"].mean(),
            "recon_rmse":       grp["l2_recon"].pow(0.5).mean() if "l2_recon" in grp else np.nan,
            "recon_mean_error": grp["delta_recon"].mean(),
            "recon_std_error":  grp["delta_recon"].std(),
        })
 
    df_summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "cnn_predictions_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"Saved summary -> {summary_path}")
    logger.info("\n" + df_summary.to_string(index=False))
 
    # ------------------------------------------------------------------ #
    # plots
    # ------------------------------------------------------------------ #
 
    # Plot 1: Average absolute delta by mask ratio (original vs recon)
    fig, ax = plt.subplots(figsize=(8, 5))
    mask_ratios_sorted = sorted(df_summary["mask_ratio"].unique(), key=float)
    orig_maes  = [df_summary[df_summary["mask_ratio"] == m]["original_mae"].values[0] for m in mask_ratios_sorted]
    recon_maes = [df_summary[df_summary["mask_ratio"] == m]["recon_mae"].values[0]    for m in mask_ratios_sorted]
 
    ax.plot(mask_ratios_sorted, orig_maes,  marker="o", label="Original Image MAE")
    ax.plot(mask_ratios_sorted, recon_maes, marker="o", label="Reconstructed Image MAE")
    ax.set_xlabel("Mask Ratio")
    ax.set_ylabel("Mean Absolute Error (Redshift)")
    ax.set_title("CNN Redshift Prediction MAE vs Mask Ratio")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
 
    plot_path = output_dir / "cnn_mae_vs_mask_ratio.pdf"
    fig.savefig(plot_path)
    plt.close(fig)
    logger.info(f"Saved plot -> {plot_path}")
 
    logger.info(f"All CNN evaluation outputs saved to: {output_dir}")
 


if __name__ == "__main__":
    from src.utils.logger import init_shared_logger
    logger = init_shared_logger(__file__, log_stdout=True, log_stderr=True)
    try:
        pt.multiprocessing.set_sharing_strategy("file_system")
        main()
    except Exception as e:
        logger.error(e)
 
