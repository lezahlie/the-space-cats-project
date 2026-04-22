from src.utils.logger import get_logger, set_logger_level, log_execution_time
from src.utils.common import argparse, os, copy, Path, pt, time, np, gc, HDF5StackWriter, read_from_json, save_to_json, validate_tensor
from src.utils.config import validate_config, merge_config
from src.utils.device import SetupDevice
from src.utils.viz import plot_image_samples, plot_learning_curves
from src.preprocess_data import Normalize, PrepareDataset, PrepareDatasets
from src.models.MaskedAutoencoder import MaskedAutoencoder
from src.utils.losses import masked_reconstruction_loss


def process_args():
    parser = argparse.ArgumentParser(description="Train Model Executable", formatter_class= argparse.RawTextHelpFormatter)
    
    parser.add_argument('--debug', '-d', dest='debug', action='store_true', 
                help="Enables debug option and verbose printing | default: Off")
    parser.add_argument('--random-seed', dest='random_seed', type=int, default=42,
                help="Random seed for selecting samples | default: None")
    parser.add_argument('--input-folder', dest="input_folder", type=str, required=True, 
                help="Input path/to/directory where the preprocessed datasets are saved | required")
    parser.add_argument('--output-folder', dest="output_folder", type=str, required=True,  
                help="Output path/to/directory to save experiment results to | required")
    parser.add_argument('--config-file', dest="config_file", type=str, required=True, 
                help="Input config file path for model training | required")
    parser.add_argument('--gpu-device-list', dest='gpu_device_list', type=int, nargs='+', default=[0], 
                help='Specify which GPU(s) to use; Provide multiple GPUs as space-separated values, e.g., "0 1" | default: 0 (if CUDA is available)')
    parser.add_argument('--gpu-memory-fraction', dest='gpu_memory_fraction', type=float, default=0.5,
                help='Fraction of GPU memory to allocate per process | default: 0.5 (if CUDA is available)')
    parser.add_argument('--cpu-device-only', dest="cpu_device_only", action='store_true',
                help="PyTorch device can only use default CPU; Overrides other device options | default: Off")
    parser.add_argument('--num-cores', dest="num_cores", type=int, default=1, 
                help="Number of cpu cores (tasks) to run in parallel. If multi-threading is enabled, max threads is set to (num_tasks * 2) | default: 1")
    parser.add_argument('--deterministic',  dest='deterministic', action='store_true', 
                help="Enables deterministic algorithms, trades reproducibility for faster torch ops and sometimes causes errors | default: True")
    
    args = parser.parse_args()


    if isinstance(args.gpu_device_list, int):
        args.gpu_device_list = [args.gpu_device_list]
    if len(args.gpu_device_list) < 1:
        raise ValueError("[--gpu-device-list] must have at least 1 device, the default gpu is '0'")
    if not (0.0 < args.gpu_memory_fraction <= 1.0):
        raise ValueError("[--gpu-memory-fraction] must be in the range (0.0, 1.0]")
    if not (1 <= args.num_cores < os.cpu_count()):
        raise ValueError(f"[--num-cores] must be an INT between [1, {os.cpu_count() - 1}]")
    input_path = Path(args.input_folder)
    if not input_path.is_dir():
        raise FileNotFoundError(f"[--input-folder] '{input_path}' does not exist")
    output_path = Path(args.output_folder)
    if not (0 < len(str(output_path)) < 256):
        raise ValueError(f"[--output-folder] '{output_path}' must have a length between [1, 255]")
    config_file = Path(args.config_file)
    if not config_file.is_file():
        raise FileNotFoundError(f"[--config-file] '{config_file}' does not exist")
    
    return args


# ==================================================
# CONTRIBUTION START: MAETrainer, main()
# Contributor: Leslie Horace
# ==================================================


class ModelTrainer:
    def __init__(
        self,
        config,
        input_folder,
        output_folder,
        device="cuda" if pt.cuda.is_available() else "cpu",
        make_subdirs=True
    ):
        self.logger = get_logger()
        self.device = device if isinstance(device, pt.device) else pt.device(device)
        self.make_subdirs = make_subdirs
        self.config = merge_config(config)

        self.input_folder = Path(input_folder).resolve()
        self.output_folder = Path(output_folder).resolve()
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.artifacts_dir = self.output_folder / "artifacts"
        self.checkpoints_dir = self.artifacts_dir / "checkpoints"
        self.plots_dir = self.artifacts_dir / "plots"
        self.metrics_dir = self.artifacts_dir / "metrics"
        self.samples_dir = self.artifacts_dir / "samples"

        prep_datasets_path = self.input_folder / "prepare_datasets.pth"
        self.prepare_datasets = PrepareDatasets.load(
            prep_datasets_path,
            batch_size=self.config["batch_size"],
            num_workers=max(1, self.config["num_workers"]),
            random_seed=self.config["random_seed"],
            pin_memory=self.device.type == "cuda"
        )

        self.train_loader = self.prepare_datasets.train_dataloader
        self.valid_loader = self.prepare_datasets.valid_dataloader
        self.test_loader = self.prepare_datasets.test_dataloader
        self.transform = self.prepare_datasets.transform
        
        prep_metadata_path = self.input_folder / "preprocessing_metadata.json"
        prep_metadata = read_from_json(prep_metadata_path)
        self.config["input_shape"] = prep_metadata["dataset_sample_shapes"]["x_masked_image"]
        self.config["mask_ratio"] = prep_metadata["dataset_masking"]["mask_ratio"]

        self.config = validate_config(self.config)

        self.mae_model = MaskedAutoencoder(self.config)
        self.logger.debug(f"CNNEncoder.hidden_per_layer: {self.mae_model.encoder.hidden_per_layer}")
        self.logger.debug(f"CNNDecoder.hidden_per_layer: {self.mae_model.decoder.hidden_per_layer}")

        self.setup_optimizer(self.mae_model)
        self.criterion = masked_reconstruction_loss

        self.best_model_state = copy.deepcopy(self.mae_model.state_dict())
        self.best_model_epoch = 0
        self.best_valid_loss = float("inf")
        self.not_improved_epochs = 0
        self.history = []

        self.patience = self.config["earlystop_patience"]
        self.earlystop = self.patience > 0 and self.config["enable_earlystop"]
        self.min_delta = abs(self.config["earlystop_min_delta"])

        self._should_earlystop = lambda: self.earlystop and (self.not_improved_epochs >= self.patience)

        self._should_log_batch = lambda batch_idx, total_batches: (
            self.config["log_batch_frequency"] > 0 and (
                batch_idx == 1
                or batch_idx == total_batches
                or batch_idx % self.config["log_batch_frequency"] == 0
            )
        )

        self._should_log_epoch = lambda epoch: (
            self.config["log_epoch_frequency"] > 0 and (
                epoch == 1
                or epoch == self.config["num_epochs"]
                or epoch % self.config["log_epoch_frequency"] == 0
            )
        )

        self._should_plot_last_batch = lambda epoch=None: (
            self.config["plot_last_batch_frequency"] > 0
            and self.config["plot_last_batch_limit"] > 0
            and (epoch is None or isinstance(epoch, str) or (isinstance(epoch, int)
                    and (epoch == 0
                        or epoch == self.config["num_epochs"]
                        or epoch % self.config["plot_last_batch_frequency"] == 0)
                    )
                )
            )

        total_params = sum(p.numel() for p in self.mae_model.encoder.parameters())
        trainable_params = sum(p.numel() for p in self.mae_model.encoder.parameters() if p.requires_grad)
        self.logger.info(f"CNNEncoder: total_params = {total_params}, trainable_params = {trainable_params}")

        total_params = sum(p.numel() for p in self.mae_model.decoder.parameters())
        trainable_params = sum(p.numel() for p in self.mae_model.decoder.parameters() if p.requires_grad)
        self.logger.info(f"CNNDecoder: total_params = {total_params}, trainable_params = {trainable_params}")

        total_params = sum(p.numel() for p in self.mae_model.parameters())
        trainable_params = sum(p.numel() for p in self.mae_model.parameters() if p.requires_grad)
        self.logger.info(f"MaskedAutoEncoder: total_params = {total_params}, trainable_params = {trainable_params}")

        if self.make_subdirs:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            self.samples_dir.mkdir(parents=True, exist_ok=True)
            save_to_json(self.output_folder / "resolved_train_config.json", self.config)
            
    def setup_optimizer(self, model):
        if self.config["optim_type"] == "adam":
            self.optimizer = pt.optim.Adam(
                model.parameters(),
                lr=self.config["learn_rate"],
                weight_decay=self.config["weight_decay"],
                betas=(self.config["optim_beta1"], self.config["optim_beta2"]),
            )
        else:
            self.optimizer = pt.optim.AdamW(
                model.parameters(),
                lr=self.config["learn_rate"],
                weight_decay=self.config["weight_decay"],
                betas=(self.config["optim_beta1"], self.config["optim_beta2"]),
            )

        scheduler_name = self.config["lr_scheduler"]
        if scheduler_name == "none":
            self.scheduler = None

        elif scheduler_name == "plateau":
            self.scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config["lr_scheduler_factor"],
                patience=self.config["lr_scheduler_patience"],
                min_lr=self.config["lr_scheduler_min_lr"],
            )

        elif scheduler_name == "cosine":
            self.scheduler = pt.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["num_epochs"],
                eta_min=self.config["lr_scheduler_min_lr"],
            )

        else:
            raise ValueError(f"lr_scheduler {scheduler_name!r} not one of 'none', 'plateau', or 'cosine'")

    def scheduler_step(self, valid_loss):
        if self.scheduler is None:
            return None

        if self.config["lr_scheduler"] == "plateau":
            return self.scheduler.step(valid_loss)

        return self.scheduler.step()

    def check_improvement(self, valid_loss, epoch):
        if valid_loss < (self.best_valid_loss - self.min_delta):
            self.best_model_state = copy.deepcopy(self.mae_model.state_dict())
            self.best_valid_loss = valid_loss
            self.best_model_epoch = epoch
            self.not_improved_epochs = 0
            return True

        self.not_improved_epochs += 1
        return False
    
    def save_model_checkpoint(self, file_name, model_state_dict, extra_data=None):
        checkpoint = {
            "model_state_dict": model_state_dict,
            "config": self.config,
        }
        if extra_data is not None:
            checkpoint.update(extra_data)

        pt.save(checkpoint, self.checkpoints_dir / file_name)

    def load_model_checkpoint(self, file_name):
        checkpoint_path = self.checkpoints_dir / file_name

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        best_checkpoint = pt.load(checkpoint_path)
        best_state = best_checkpoint["model_state_dict"]
        self.mae_model.load_state_dict(best_state)
        

    def _copy_batch_samples(self, batch, y_recon, z_latent, start=0, stop=None):
        if batch is None:
            return None

        batch_samples = {
            "original_id": np.asarray(batch.original_id[start:stop]),
            "masked_region_map": batch.masked_region_map[start:stop],
            "x_masked_image": batch.x_masked_image[start:stop],
            "y_target_image": batch.y_target_image[start:stop],
            "y_recon_image": y_recon[start:stop],
            "z_latent_vector": z_latent[start:stop],
            "y_specz_redshift": batch.y_specz_redshift[start:stop],
        }

        return batch_samples
    

    @pt.no_grad()
    def save_sample_plots(self, sample_data, split:str, inverse_transform=False, epoch=None):
        if sample_data is None:
            self.logger.warning(f"[{split}] plot data is None, skipping sample plot")
            return None

        original_id = sample_data["original_id"]
        masked_map = sample_data["masked_region_map"]

        x_masked = sample_data["x_masked_image"]
        y_target = sample_data["y_target_image"]
        y_recon = sample_data["y_recon_image"]
        y_redshift = sample_data["y_specz_redshift"]
    
        if inverse_transform:
            x_masked = self.transform.inverse_transform(x_masked)
            y_target = self.transform.inverse_transform(y_target)
            y_recon = self.transform.inverse_transform(y_recon)
            y_redshift = self.transform.inverse_transform_specz(y_redshift)

            masked_map_broadcast = masked_map.repeat(1, x_masked.shape[1], 1, 1)
            x_masked = x_masked * (masked_map_broadcast <= 0.5)

        file_name = f"{split}.png"
        figure_title = f"{split.title()} Sample"
        if epoch is not None:
            file_name = file_name.replace(".png", f"_epoch_{epoch}.png")
            figure_title += f" | Epoch #{epoch}"
            
        total_samples = len(original_id)
        max_samples = min(total_samples, self.config["plot_last_batch_limit"])
        sample_indices = np.random.choice(total_samples, size=max_samples, replace=False)

        save_path = self.plots_dir / file_name
        plot_image_samples(
            original_id=[original_id[i] for i in sample_indices],
            masked_map= masked_map[sample_indices].detach().cpu().numpy(),
            x_masked= x_masked[sample_indices].detach().cpu().numpy(),
            y_target= y_target[sample_indices].detach().cpu().numpy(),
            y_recon= y_recon[sample_indices].detach().cpu().numpy(),
            y_redshift= y_redshift[sample_indices].detach().cpu().numpy(),
            save_path=save_path,
            figure_title=figure_title,
            band_names=("g", "r", "i", "z", "y"),
            cmap_name="inferno",    
        )

        # @note need to find a perceptually uniform colormap made for visualizing redshift.
        # for now 'inferno' is good enough to see if the model is learning something.

        self.logger.info(f"[{split}] saved {max_samples} sample plots to: {self.plots_dir}")


    @pt.no_grad()
    def export_model_outputs(self, model, loader, file_name: str, chunk_size=64):
        model.eval()

        save_path = self.samples_dir / file_name
        total_rows = 0
        writer = None
        sample_batch = None
        y_recon = None
        z_latent = None
        x_input = None

        try:
            for batch_idx, batch in enumerate(loader, start=1):

                del sample_batch, y_recon, z_latent, x_input

                x_input = batch.x_masked_image.to(self.device, non_blocking=True)

                z_latent = model.encode(x_input)
                if self.config["debug"]:
                    validate_tensor("z_latent", z_latent)

                y_recon = model.decode(z_latent)
                if self.config["debug"]:
                    validate_tensor("y_recon", y_recon)

                batch_size = x_input.shape[0]
                sample_batch = self._copy_batch_samples(
                    batch=batch,
                    y_recon=y_recon,
                    z_latent=z_latent,
                )

                if writer is None:
                    writer = HDF5StackWriter(
                        hdf5_path=save_path,
                        chunk_rows=chunk_size,
                        overwrite=True,
                    )

                writer.append(sample_batch)
                total_rows += batch_size

                if self._should_log_batch(batch_idx, len(loader)):
                    self.logger.info(f"[EXPORT]: Batch[{batch_idx}/{len(loader)}] saved_rows={total_rows}")

            sample_batch = self._copy_batch_samples(
                batch=batch,
                y_recon=y_recon,
                z_latent=z_latent,
            )

            self.save_sample_plots(sample_batch, file_name.split("_")[0], epoch="best")

        finally:
            if writer is not None:
                writer.close()

        self.logger.info(f"[EXPORT] saved {total_rows} rows to: {save_path}")
        return save_path

    def save_result_metrics(self, test_metrics):
        history_path = self.metrics_dir / "model_history.json"
        plot_path = self.plots_dir / "learning_curves.png"

        save_to_json(history_path, self.history)
        plot_learning_curves(self.history, plot_path)

        result_metadata = {
            "best_epoch": self.best_model_epoch,
            "best_valid_loss": self.best_valid_loss,
            "test_metrics": test_metrics,
            "history_path": str(history_path),
            "history_plot_path": str(plot_path),
            "best_checkpoint_path": str(self.checkpoints_dir / "best_model.pth"),
            "final_checkpoint_path": str(self.checkpoints_dir / "final_model.pth"),
            "config_path": str(self.output_folder / "resolved_train_config.json"),
        }

        save_to_json(self.output_folder / "result_metadata.json", result_metadata)
        return result_metadata

    def train(self, model, epoch=None):
        model.train()
        self.logger.debug(f"[TRAIN] model.training={model.training}")

        total_loss = 0.0
        total_smooth_l1 = 0.0
        total_ssim_loss = 0.0

        epoch_start = time.perf_counter()
        total_batches = len(self.train_loader)

        with pt.enable_grad():
            for batch_idx, batch in enumerate(self.train_loader, start=1):
                x_input = batch.x_masked_image.to(self.device, non_blocking=True)
                x_mask = batch.masked_region_map.to(self.device, non_blocking=True)
                y_target = batch.y_target_image.to(self.device, non_blocking=True)

                z_latent = model.encode(x_input)
                if self.config["debug"]:
                    validate_tensor("z_latent", z_latent)

                y_recon = model.decode(z_latent)
                if self.config["debug"]:
                    validate_tensor("y_recon", y_recon)

                loss, smooth_l1, ssim_loss = self.criterion(
                    recon_image=y_recon,
                    target_image=y_target,
                    masked_region_map=x_mask,
                    ssim_weight=self.config["ssim_loss_weight"],
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.detach().item()
                total_smooth_l1 += smooth_l1.detach().item()
                total_ssim_loss += ssim_loss.detach().item()
                
                if self._should_log_batch(batch_idx, total_batches):
                    avg_loss = total_loss / batch_idx
                    self.logger.info(
                        f"[TRAIN]: Batch[{batch_idx}/{total_batches}] "
                        f"batch_loss={loss.detach().item():.6f}, "
                        f"running_loss={total_loss:.6f}, "
                        f"avg_loss={avg_loss:.6f}"
                    )

        if self.config["debug"] and self._should_plot_last_batch(epoch):
            batch_samples = self._copy_batch_samples(batch=batch, y_recon=y_recon, z_latent=z_latent)
            self.save_sample_plots(batch_samples, "training", epoch=epoch)

        metrics = {
            "objective_loss": total_loss / total_batches,
            "smooth_l1": total_smooth_l1 / total_batches,
            "ssim_loss": total_ssim_loss / total_batches,
            "train_seconds": time.perf_counter() - epoch_start
        }

        return metrics

    @pt.no_grad()
    def evaluate(self, model, is_validation=True, epoch=None):
        model.eval()

        loader = self.valid_loader if is_validation else self.test_loader
        log_label = "VALID" if is_validation else "TEST"
        split_label = "validation" if is_validation else "testing"
        self.logger.debug(f"[{log_label}] model.training={model.training}")

        total_loss = 0.0
        total_smooth_l1 = 0.0
        total_ssim_loss = 0.0

        eval_start = time.perf_counter()
        total_batches = len(loader)

        with pt.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):

                x_input = batch.x_masked_image.to(self.device, non_blocking=True)
                x_mask = batch.masked_region_map.to(self.device, non_blocking=True)
                y_target = batch.y_target_image.to(self.device, non_blocking=True)

                z_latent = model.encode(x_input)
                if self.config["debug"]:
                    validate_tensor("z_latent", z_latent)
                    
                y_recon = model.decode(z_latent)
                if self.config["debug"]:
                    validate_tensor("y_recon", y_recon)
    
                loss, smooth_l1, ssim_loss = self.criterion(
                    recon_image=y_recon,
                    target_image=y_target,
                    masked_region_map=x_mask,
                    ssim_weight=self.config["ssim_loss_weight"],
                )

                total_loss += loss.detach().item()
                total_smooth_l1 += smooth_l1.detach().item()
                total_ssim_loss += ssim_loss.detach().item()
    
                if self._should_log_batch(batch_idx, total_batches):
                    avg_loss = total_loss / batch_idx
                    self.logger.info(
                        f"[{log_label:>5}]: Batch[{batch_idx}/{total_batches}] "
                        f"batch_loss={loss.detach().item():.6f}, "
                        f"running_loss={total_loss:.6f}, "
                        f"avg_loss={avg_loss:.6f}"
                    )

        if self._should_plot_last_batch(epoch):
            batch_samples = self._copy_batch_samples(batch=batch, y_recon=y_recon, z_latent=z_latent)
            self.save_sample_plots(batch_samples, split_label, epoch=epoch)

        metrics = {
            "objective_loss": total_loss / total_batches,
            "smooth_l1": total_smooth_l1 / total_batches,
            "ssim_loss": total_ssim_loss / total_batches,
            f"{split_label}_seconds": time.perf_counter() - eval_start
        }

        return metrics

    def train_model(self):
        self.mae_model = self.mae_model.to(self.device)

        for epoch in range(1, self.config["num_epochs"] + 1):
            self.logger.info(f"========== Epoch [{epoch}/{self.config['num_epochs']}] ==========")

            train_metrics = self.train(self.mae_model, epoch=epoch)
            if self._should_log_epoch(epoch):
                self.logger.info(
                    f"[TRAIN] objective_loss={train_metrics['objective_loss']:.8f}, "
                    f"smooth_l1={train_metrics['smooth_l1']:.8f}, "
                    f"ssim_loss={train_metrics['ssim_loss']:.8f}"
                )

            valid_metrics = self.evaluate(self.mae_model, is_validation=True, epoch=epoch)
            if self._should_log_epoch(epoch):
                self.logger.info(
                    f"[VALID] objective_loss={valid_metrics['objective_loss']:.8f}, "
                    f"smooth_l1={valid_metrics['smooth_l1']:.8f}, "
                    f"ssim_loss={valid_metrics['ssim_loss']:.8f}"
                )

            valid_loss = valid_metrics["objective_loss"]
            self.scheduler_step(valid_loss)
            has_improved = self.check_improvement(valid_loss, epoch)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.debug(
                f"[EPOCH #{epoch}] "
                f"current_lr={current_lr:.6e}, "
                f"train_loss={train_metrics['objective_loss']:.6f}, "
                f"valid_loss={valid_loss:.6f}, "
                f"best_valid_loss={self.best_valid_loss:.6f}, "
                f"best_model_epoch={self.best_model_epoch}, "
                f"not_improved_epochs={self.not_improved_epochs}"
            )

            self.history.append({
                "epoch": epoch,
                "learning_rate": current_lr,
                "train": train_metrics,
                "validation": valid_metrics,
                "improved": has_improved,
            })

            if has_improved:
                self.save_model_checkpoint(
                    file_name="best_model.pth",
                    model_state_dict=self.best_model_state,
                    extra_data={
                        "best_epoch": self.best_model_epoch,
                        "best_valid_loss": self.best_valid_loss,
                        "history": self.history,
                    },
                )
                
            if self._should_earlystop():
                
                self.logger.info(
                    f"Early stopping at epoch={epoch}, "
                    f"best_epoch={self.best_model_epoch}, "
                    f"best_valid_loss={self.best_valid_loss:.8f}"
                )
                break

        self.final_epoch = epoch
        # final model results
        final_test_metrics = self.evaluate(self.mae_model, is_validation=False, epoch="final")
        self.save_model_checkpoint(
            file_name="final_model.pth",
            model_state_dict=self.mae_model.state_dict(),
            extra_data={
                "final_model_epoch": self.final_epoch,
                "final_valid_loss": valid_loss,
                "final_test_metrics": final_test_metrics,
                "history": self.history,
            },
        )

        # best model results
        self.mae_model.load_state_dict(self.best_model_state)

        best_test_metrics = self.evaluate(self.mae_model, is_validation=False, epoch="best")
        self.save_model_checkpoint(
            file_name="best_model.pth",
            model_state_dict=self.mae_model.state_dict(),
            extra_data={
                "best_model_epoch": self.best_model_epoch,
                "best_valid_loss": self.best_valid_loss,
                "best_test_metrics": best_test_metrics,
                "history": self.history,
            }
        )

        result_metadata = self.save_result_metrics(best_test_metrics)

        self.logger.info(
            f"[TEST] objective_loss={best_test_metrics['objective_loss']:.8f}, "
            f"smooth_l1={best_test_metrics['smooth_l1']:.8f}, "
            f"ssim_loss={best_test_metrics['ssim_loss']:.8f}"
        )

        return result_metadata


@log_execution_time
def main(args):
    if args.debug:
        set_logger_level(10)

    logger = get_logger()

    device = SetupDevice.setup_torch_device(
        args.num_cores,
        args.cpu_device_only,
        gpu_list=args.gpu_device_list,
        gpu_memory=args.gpu_memory_fraction,
        random_seed=args.random_seed,
        deterministic=args.deterministic
    )
    
    logger.info(f"Using device = {device}")

    train_config = read_from_json(args.config_file)
    train_config.update({
        "debug": args.debug,
        "num_workers": args.num_cores,
        "random_seed": args.random_seed,
    })

    trainer = ModelTrainer(train_config, args.input_folder, args.output_folder,device=device)
    results = trainer.train_model()

    logger.info(results)

    trainer.load_model_checkpoint("best_model.pth")

    trainer.export_model_outputs(model=trainer.mae_model, loader=trainer.train_loader, file_name="training_outputs_best.hdf5")
    trainer.export_model_outputs(model=trainer.mae_model, loader=trainer.valid_loader, file_name="validation_outputs_best.hdf5")
    trainer.export_model_outputs(model=trainer.mae_model, loader=trainer.test_loader, file_name="testing_outputs_best.hdf5")


# ==================================================
# CONTRIBUTION End: ModelTrainer, main()
# ==================================================

if __name__ == "__main__":
    from src.utils.logger import init_shared_logger
    logger = init_shared_logger(__file__, log_stdout=True, log_stderr=True)
    try:
        pt.multiprocessing.set_sharing_strategy('file_system')
        args = process_args()
        main(args)
    except Exception as e:
        logger.error(e)


# make sure it works so far:
"""
python src/train_model.py \
--config-file configs/train_config.json \
--input-folder data/preprocessed/galaxiesml_tiny \
--output-folder experiments/train_galaxiesml_tiny \
--gpu-memory-fraction 0.9 \
--num-cores 2 \
--debug
"""