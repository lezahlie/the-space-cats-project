from src.utils.logger import get_logger, set_logger_level, log_execution_time
from src.utils.common import argparse, os, Path, pt, h5py, np, json, GalaxiesMLDataset

def process_args():
    parser = argparse.ArgumentParser(description="Preprocess Dataset Executable", formatter_class= argparse.RawTextHelpFormatter)
    
    parser.add_argument('--debug', '-d', dest='debug', action='store_true', 
                help="Enables debug option and verbose printing | default: Off")
    parser.add_argument('--random-seed', dest='random_seed', type=int, default=42,
                help="Random seed for selecting samples | default: None")
    parser.add_argument('--input-folder', dest="input_folder", type=str, required=True, 
                help="Input path/to/directory where the datasets are saved | required")
    parser.add_argument('--output-folder', dest="output_folder", type=str, required=True, 
                help="Output path/to/directory to save preprocessed datasets to | required")
    parser.add_argument('--num-cores', dest="num_cores", type=int, default=1, 
                help="Number of cpu cores (tasks) to run in parallel. If multi-threading is enabled, max threads is set to (num_tasks * 2) | default: 1")
    parser.add_argument('--batch-size', dest="batch_size", type=int, default=32, 
                help="Batch size for DataLoaders | default: 32")
                
    """
    please add the remaining arguments you need to do the things
    """
    args = parser.parse_args()
    
    if not (1 <= args.num_cores < os.cpu_count()):
        raise ValueError(f"[--num-cores]  must be a INT between [1, {os.cpu_count()} - 1]")
    if not os.path.isdir(args.input_folder):
        raise FileNotFoundError(f"[--input-folder] '{args.input_folder}' does not exist")
    if not (0 < len(args.output_folder) < 256):
        raise ValueError(f"[--output-folder] '{args.output_folder}' must have a length between [1, 255]")
    if not (1 <= args.batch_size <= 4096):
        raise ValueError(f"[--batch-size] must be a INT between [1, 4096]")

    """
    please validate the arguments you added so it breaks now instead of later
    """
        
    return args


# ==================================================
# CONTRIBUTION START: Normalize, PrepareDataset, PrepareDatasets
# Contributor: Charlie Faber
# ==================================================

class PrepareDatasets:
    def __init__(self, train_dataset, valid_dataset, test_dataset, 
                 transform, output_folder, batch_size: int = 32, 
                 num_workers: int = 0, random_seed:int = 42):
        """read the datasets, clean, reformat, and save for later
        """
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.random_seed = random_seed
        self.transform = transform
        self.output_folder = Path(output_folder)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.output_folder.mkdir(parents=True, exist_ok=True)
        g = pt.Generator()
        g.manual_seed(self.random_seed)

        self.train_dataloader = pt.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=g
        )
        self.valid_dataloader = pt.utils.data.DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        self.test_dataloader = pt.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def validate(self):
        logger = get_logger()
        splits = {
            "train": self.train_dataloader,
            "validation": self.valid_dataloader,
            "testing": self.test_dataloader
        }

        for split, loader in splits.items():
            image, specz, original_id = next(iter(loader))
            assert image.ndim == 4 and image.shape[1:] == (5, 64, 64), (
                f"[{split}] image shape {tuple(image.shape)}, expected (N, 5, 64, 64)")
            assert image.dtype == pt.float32, (
                f"[{split}] image dtype {image.dtype}, expected float32")
            assert specz.ndim == 1 and specz.dtype == pt.float32, (
                f"[{split}] specz shape/dtype mismatch — got shape {tuple(specz.shape)}, dtype {specz.dtype}")
            assert isinstance(original_id, (list, tuple)) and isinstance(original_id[0], str), (
                f"[{split}] original_id should be a list[str], got {type(original_id)}")

            finite = image[pt.isfinite(image)]
            img_min = finite.min().item() if len(finite) > 0 else 0.0
            img_max = finite.max().item() if len(finite) > 0 else 1.0
            assert img_min >= -1e-4, (
                f"[{split}] image min {img_min:.6f} is below 0.0")
            assert img_max <= 1.0 + 1e-4, (
                f"[{split}] image max {img_max:.6f} is above 1.0")

            if split == 'train':
                restored = self.transform.inverse_transform(image)
                logger.info(
                    f"Inverse transform check — restored pixel range: "
                    f"[{restored.min():.4f}, {restored.max():.4f}]"
                )

            logger.info(
                f"[{split}] image={tuple(image.shape)} dtype={image.dtype} "
                f"range=[{img_min:.4f}, {img_max:.4f}] | "
                f"specz={tuple(specz.shape)} | "
                f"original_id[0]='{original_id[0]}'"
            )

        logger.info("All dataloaders validated successfully")

    def save(self):
        logger = get_logger()
        out = self.output_folder

        pt.save(self.transform, out / "transform.pth")
        pt.save(self.train_dataloader, out / "train_dataloader.pth")
        pt.save(self.valid_dataloader, out / "valid_dataloader.pth")
        pt.save(self.test_dataloader, out / "test_dataloader.pth")

        metadata = {
            "image_normalization": {
                "original_min": float(self.transform.original_min),
                "original_max": float(self.transform.original_max),
                "normalized_min": float(self.transform.normalized_min),
                "normalized_max": float(self.transform.normalized_max),
            },
            "specz_normalization": (
                {
                    "original_min": float(self.transform.specz_min),
                    "original_max": float(self.transform.specz_max),
                    "normalized_min": float(self.transform.normalized_min),
                    "normalized_max": float(self.transform.normalized_max),
                }
                if self.transform.specz_min is not None else None
            ),
            "dataset_sizes": {
                "train": len(self.train_dataset),
                "valid": len(self.valid_dataset),
                "test": len(self.test_dataset),
            },
            "dataloader_settings": {
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "random_seed": self.random_seed,
            },
            "image_shape": list(self.train_dataset.sample_shape),
        }

        with open(out / "preprocessing_metadata.json", 'w') as meta_f:
            json.dump(metadata, meta_f, indent=2)

        logger.info(f"Saved 4 .pth files and metadata JSON to: {out}")


class Normalize:
    def __init__(self, min_val=0.0, max_val=1.0, device='cpu'):
        """initialize the normalized min and max, then also the original min and max

        Args:
            min_val (float, optional): _description_. Defaults to 0.0.
            max_val (float, optional): _description_. Defaults to 1.0.
            device (str, optional): _description_. Defaults to 'cpu'.
        """
        self.device = device
        self.normalized_min  = min_val
        self.normalized_max = max_val
        self.original_min = None
        self.original_max = None
        self.specz_min = None
        self.specz_max = None

    def fit(self, train_data):
        """fit the training split by deriving the global min and max 
        from the training split ONLY to avoid data leakage.

        Returns:
            train_data: _description_
        """
        logger = get_logger()
        hdf5_path = getattr(train_data, "hdf5_path", None)
        if hdf5_path is None:
            raise ValueError("train_data must have an attribute 'hdf5_path' (GalaxiesMLDataset)")

        image_key = getattr(train_data, "input_key", None)
        n_samples = len(train_data)
        chunk_size = 512

        global_min = np.inf
        global_max = -np.inf

        with h5py.File(hdf5_path, 'r') as f:
            images = f[image_key]
            logger.info(f"Fitting on {n_samples} from {hdf5_path} with chunk size {chunk_size}")
            for i in range(0, n_samples, chunk_size):
                end = min(i + chunk_size, n_samples)
                chunk = images[i:end].astype(np.float32)
                global_min = min(global_min, chunk.min())
                global_max = max(global_max, chunk.max())
            
            self.original_min = global_min
            self.original_max = global_max
            logger.info(f"Image pixel range: min={global_min:.6f}, max={global_max:.6f}")

            if 'specz_redshift' in f:
                specz = f['specz_redshift'][:n_samples].astype(np.float32)
                valid = specz[np.isfinite(specz)]
                if len(valid) > 0:
                    self.specz_min = float(valid.min())
                    self.specz_max = float(valid.max())
                    logger.info(f"Specz range: min={self.specz_min:.6f}, max={self.specz_max:.6f}")

            return self

    def __call__(self, data):

        if self.original_min is None:
            raise ValueError("Normalize instance must be fitted")
        span = self.original_max - self.original_min
        if span == 0:
            return pt.full_like(data, self.normalized_min)
        
        normalized = (data - self.original_min) / span
        normalized = normalized * (self.normalized_max - self.normalized_min) + self.normalized_min
        return pt.clamp(normalized, self.normalized_min, self.normalized_max)
        
    def normalize_specz(self, specz: pt.Tensor) -> pt.Tensor:
        if self.specz_min is None:
            return specz
        
        span = self.specz_max - self.specz_min
        if span == 0:
            return pt.full_like(specz, self.normalized_min)
        
        normalized = (specz - self.specz_min) / span
        normalized = normalized * (self.normalized_max - self.normalized_min) + self.normalized_min
        return pt.clamp(normalized, self.normalized_min, self.normalized_max)
        

    def inverse_transform(self, data: pt.Tensor, new_device=None) -> pt.Tensor:
        """transforms normalized data back to its original scale

        Args:
            data (_type_): tensor data
            new_device (_type_, optional): move to specific device. Defaults to None.

        """

        norm_span = self.normalized_max - self.normalized_min
        orig_span = self.original_max - self.original_min

        restored = (data - self.normalized_min) / norm_span * orig_span + self.original_min
        if new_device is not None:
            restored = restored.to(new_device)
        return restored
    
    def inverse_transform_specz(self, specz: pt.Tensor, new_device=None) -> pt.Tensor:
        if self.specz_min is None:
            return specz
        
        norm_span = self.normalized_max - self.normalized_min
        orig_span = self.specz_max - self.specz_min

        restored = (specz - self.normalized_min) / norm_span * orig_span + self.specz_min
        if new_device is not None:
            restored = restored.to(new_device)
        return restored
    
class PrepareDataset(pt.utils.data.Dataset):
    """PyTorch Dataset for GalaxiesML HDF5 files.
    Returns (image, specz_redshift, original_id) per sample.
    """

    IMAGE_KEY = 'image'
    SPECZ_KEY = 'specz_redshift'
    ID_KEY = 'object_id'

    def __init__(self, hdf5_path, transform=None, max_samples=None):
        self.hdf5_path = str(hdf5_path)
        self.transform = transform
        self.max_samples = max_samples
        self.file = None

        with h5py.File(self.hdf5_path, 'r') as f:
            if self.IMAGE_KEY not in f:
                raise KeyError(f"'{self.IMAGE_KEY}' dataset not found in {self.hdf5_path}")
            self.length = f[self.IMAGE_KEY].shape[0]
            if max_samples is not None:
                self.length = min(self.length, max_samples)

            self.sample_shape = tuple(f[self.IMAGE_KEY].shape[1:])
            if self.SPECZ_KEY not in f:
                raise KeyError(f"'{self.SPECZ_KEY}' dataset not found in {self.hdf5_path}")
            if self.ID_KEY not in f:
                raise KeyError(f"'{self.ID_KEY}' dataset not found in {self.hdf5_path}")

    def _get_file(self):
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r')
        return self.file

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f = self._get_file()
        image = pt.from_numpy(f[self.IMAGE_KEY][idx].astype(np.float32))

        if self.transform is not None:
            image = self.transform(image)
        
        specz = pt.tensor(float(f[self.SPECZ_KEY][idx]), dtype=pt.float32)
        if self.transform is not None:
            specz = self.transform.normalize_specz(specz)

        raw = f[self.ID_KEY][idx]
        original_id = raw.decode('utf-8') if isinstance(raw, bytes) else str(raw)

        return image, specz, original_id
    
    def __del__(self):
        if self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass


# ==================================================
# CONTRIBUTION END
# ==================================================

@log_execution_time
def main(args):
    if args.debug:
        set_logger_level(10)

    transform = Normalize()

    logger = get_logger()

    dataset_dir = Path(args.input_folder)
    dataset_file = "5x64x64_{}_reduced_*.hdf5"

    train_matches = list(dataset_dir.glob(dataset_file.format("training")))
    if not train_matches:
        raise FileNotFoundError(f"No training file matched in {dataset_dir}")
    train_path = train_matches[0]

    valid_matches = list(dataset_dir.glob(dataset_file.format("validation")))
    if not valid_matches:
        raise FileNotFoundError(f"No validation file matched in {dataset_dir}")
    valid_path = valid_matches[0]

    test_matches = list(dataset_dir.glob(dataset_file.format("testing")))
    if not test_matches:
        raise FileNotFoundError(f"No testing file matched in {dataset_dir}")
    test_path = test_matches[0]

    logger.info(f"Training: {train_path}")
    logger.info(f"Validation: {valid_path}")
    logger.info(f"Testing: {test_path}")

    # Fit normalization on raw training data (no transform applied yet)
    transform = Normalize()
    train_raw = GalaxiesMLDataset(train_path)
    logger.info("Fitting normalization transform on training data...")
    transform.fit(train_raw)

    # Build datasets with normalization applied
    train_dataset = PrepareDataset(train_path, transform=transform)
    valid_dataset = PrepareDataset(valid_path, transform=transform)
    test_dataset  = PrepareDataset(test_path,  transform=transform)

    # Create DataLoaders, validate output format, then save everything
    output_folder = Path(args.output_folder) / dataset_dir.name
    prepared = PrepareDatasets(
        train_dataset,
        valid_dataset,
        test_dataset,
        transform=transform,
        output_folder=output_folder,
        batch_size=args.batch_size,
        num_workers=args.num_cores,
        random_seed=args.random_seed,
    )
    prepared.validate()
    prepared.save()
    logger.info("Preprocessing complete.")

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
# returns error if datasets are not downloaded yet
"""
python src/preprocess_data.py \
--input-folder data/galaxiesml_tiny \
--output-folder data/preprocessed \
--num-cores 2 \
--batch-size 32 \
--debug
"""