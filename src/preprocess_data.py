from utils.logger import get_logger, set_logger_level, log_execution_time
from utils.common import argparse, os, Path, pt, h5py, np, random, json, GalaxiesMLDataset, AttrDict


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
    parser.add_argument('--mask-ratio', '--mask-size-ratio', dest="mask_ratio", type=float, default=0.0, 
                help=(
                    "Mask size ratio between 0.0 and 1.0, represents a proportion of the full image length on one side;"
                    "Note that a ratio of 0.0 disables masking and a ratio of 1.0 applies masking over the full image | default: 0.0"
                ))
    parser.add_argument('--mask-start-seed', dest="mask_start_seed", type=int, default=20, 
                help=(
                    "Mask seed to randomly select the a channel-wise origin to apply the mask; "
                    "Seed increments by +1 to select different origins per image sample. | default: 20 "
                ))

    args = parser.parse_args()
    
    if not (1 <= args.num_cores < os.cpu_count()):
        raise ValueError(f"[--num-cores]  must be a INT between [1, {os.cpu_count()} - 1]")
    if not os.path.isdir(args.input_folder):
        raise FileNotFoundError(f"[--input-folder] '{args.input_folder}' does not exist")
    if not (0 < len(args.output_folder) < 256):
        raise ValueError(f"[--output-folder] '{args.output_folder}' must have a length between [1, 255]")
    if not (1 <= args.batch_size <= 4096):
        raise ValueError(f"[--batch-size] must be a INT between [1, 4096]")
    if not (0.0 <= args.mask_ratio <= 1.0):
        raise ValueError(f"[--mask-size-ratio] must be a FLOAT between [0.0, 1.0]")

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
        self.pin_memory = False
        self.generator = self.make_generator(self.random_seed)
        self.build_dataloaders() # default dataloaders

    # ==================================================
    # PARTIAL CONTRIBUTION START: self.make_generator(), self.seed_worker(), 
    # self.collate_batch(), self.build_dataloaders(), self.load()
    # Contributor: Leslie
    #
    # Note: These helpers were added to simplify the 
    # process of creating and iterating dataloaders with 
    # different batch sizes, num_worker, and/or seeds
    # ==================================================
    
    @staticmethod
    def make_generator(seed: int):
        gen = pt.Generator()
        gen.manual_seed(seed)
        return gen
    
    @staticmethod
    def seed_worker(worker_id: int):
        worker_seed = pt.initial_seed() % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        pt.manual_seed(worker_seed)

    @staticmethod
    def collate_batch(batch: pt.Tensor, as_dict=False):
        """Formatting to make accessing batched tensors easier
        Args:
            batch (pt.Tensor): batched sample from dataloader
            as_dict: format batch as dict to access batch["key"] instead of batch.key
        Returns:
            AttrDict OR dict: batch formatted as attributes or a dictionary
        """
        formatted = AttrDict(
            x_masked_image=pt.stack([item["x_masked_image"] for item in batch], dim=0),
            masked_region_map=pt.stack([item["masked_region_map"] for item in batch], dim=0),
            y_target_image=pt.stack([item["y_target_image"] for item in batch], dim=0),
            y_specz_redshift=pt.stack([item["y_specz_redshift"] for item in batch], dim=0),
            original_id=[item["original_id"] for item in batch],
        )

        return formatted.__dict__ if as_dict else formatted

    def build_dataloaders(self):

        self.train_dataloader = pt.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=self.generator,
            worker_init_fn=self.seed_worker,
            collate_fn=self.collate_batch,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )

        self.valid_dataloader = pt.utils.data.DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            generator=self.generator,
            worker_init_fn=self.seed_worker,
            collate_fn=self.collate_batch,
            pin_memory=self.pin_memory,
            persistent_workers=True

        )
        
        self.test_dataloader = pt.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            generator=self.generator,
            worker_init_fn=self.seed_worker,
            collate_fn=self.collate_batch,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )

    @classmethod
    def load(cls, path, batch_size: int = None, num_workers: int = None, random_seed: int = None, pin_memory=False):
        obj = None
        try:
            obj = pt.load(path, weights_only=False)
        except Exception as e:
            raise e

        obj.train_dataset.close()
        obj.valid_dataset.close()
        obj.test_dataset.close()

        rebuild_dataloaders = (
            obj.train_dataloader is None
            or obj.valid_dataloader is None
            or obj.test_dataloader is None
        )

        if isinstance(batch_size, int) and obj.batch_size != batch_size:
            obj.batch_size = batch_size
            rebuild_dataloaders = True

        if isinstance(num_workers, int) and obj.num_workers != num_workers:
            obj.num_workers = num_workers
            rebuild_dataloaders = True

        if isinstance(random_seed, int) and obj.random_seed != random_seed:
            obj.random_seed = random_seed
            rebuild_dataloaders = True

        if isinstance(pin_memory, bool) and obj.pin_memory != pin_memory:
            obj.pin_memory = pin_memory
            rebuild_dataloaders = True

        obj.generator = obj.make_generator(obj.random_seed)

        if rebuild_dataloaders:
            obj.build_dataloaders()

        return obj
    
    def close(self):
        self.train_dataset.close()
        self.valid_dataset.close()
        self.test_dataset.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["train_dataloader"] = None
        state["valid_dataloader"] = None
        state["test_dataloader"] = None
        state["generator"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None
        self.generator = self.make_generator(self.random_seed)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ==================================================
    # PARTIAL CONTRIBUTION END: self.make_generator(), self.seed_worker(), 
    # self.collate_batch(), self.build_dataloaders(), self.load(),
    # self.close(), self.__getstate__(), self.__setstate__(), self.__del__()
    # ==================================================

    def validate(self):
        logger = get_logger()
        
        splits = {
            "train": self.train_dataloader,
            "validation": self.valid_dataloader,
            "testing": self.test_dataloader
        }

        for split, loader in splits.items():
            batch = next(iter(loader))

            mask_bool = (batch.masked_region_map == 1.0).expand_as(batch.x_masked_image)
            assert pt.all(batch.x_masked_image[mask_bool] == 0.0), (
                "Masked region is not all 0.0 in masked_region_map")
            assert pt.all((batch.x_masked_image[~mask_bool] >= 0.0) & (batch.x_masked_image[~mask_bool] <= 1.0)), (
                "Unmasked region contains values outside [0.0, 1.0] in masked_region_map")
            
            assert batch.x_masked_image.ndim == 4 and batch.x_masked_image.shape[1:] == (5, 64, 64), (
                f"[{split}] x_masked_image shape {tuple(batch.x_masked_image.shape)}, expected (N, 5, 64, 64)")
            assert batch.x_masked_image.dtype == pt.float32, (
                f"[{split}] x_masked_image dtype {batch.x_masked_image.dtype}, expected float32")
            assert batch.masked_region_map.ndim == 4 and batch.masked_region_map.shape[1:] == (1, 64, 64), (
                f"[{split}] masked_region_map shape {tuple(batch.masked_region_map.shape)}, expected (N, 1, 64, 64)")
            assert batch.masked_region_map.dtype == pt.float32, (
                f"[{split}] masked_region_map dtype {batch.masked_region_map.dtype}, expected float32")

            assert batch.y_target_image.ndim == 4 and batch.y_target_image.shape[1:] == (5, 64, 64), (
                f"[{split}] y_target_image shape {tuple(batch.y_target_image.shape)}, expected (N, 5, 64, 64)")
            assert batch.y_target_image.dtype == pt.float32, (
                f"[{split}] y_target_image dtype {batch.y_target_image.dtype}, expected float32")
    
            assert batch.y_specz_redshift.ndim == 1 and batch.y_specz_redshift.dtype == pt.float32, (
                f"[{split}] y_specz_redshift shape/dtype mismatch — got shape {tuple(batch.y_specz_redshift.shape)}, dtype {batch.y_specz_redshift.dtype}")

            assert isinstance(batch.original_id, (list, tuple)) and isinstance(batch.original_id[0], str), (
                f"[{split}] original_id should be a list[str], got {type(batch.original_id)}")

            img_finite = batch.y_target_image[pt.isfinite(batch.y_target_image)]
            img_min = img_finite.min().item() if len(img_finite) > 0 else 0.0
            img_max = img_finite.max().item() if len(img_finite) > 0 else 1.0
            assert img_min >= -1e-4, (
                f"[{split}] image min {img_min:.6f} is below 0.0")
            assert img_max <= 1.0 + 1e-4, (
                f"[{split}] image max {img_max:.6f} is above 1.0")

            specz_finite = batch.y_specz_redshift[pt.isfinite(batch.y_specz_redshift)]
            specz_min = specz_finite.min().item() if len(specz_finite) > 0 else 0.0
            specz_max = specz_finite.max().item() if len(specz_finite) > 0 else 1.0
            assert specz_min >= -1e-4, (
                f"[{split}] specz_redshift min {specz_min:.6f} is below 0.0")
            assert specz_max <= 1.0 + 1e-4, (
                f"[{split}] specz_redshift max {specz_max:.6f} is above 1.0")
            
    
            restored = self.transform.inverse_transform(batch.y_target_image)
            renormed = self.transform(restored)
            logger.info(
                f"[{split}] Transform check — image pixel range: [{batch.y_target_image.min():.4f}, {batch.y_target_image.max():.4f}]"
                f"\n\t=> Inverse transform check — restored pixel range: [{restored.min():.4f}, {restored.max():.4f}]"
                f" => Re-transform check — image pixel range: [{renormed.min():.4f}, {renormed.max():.4f}]"
            )
            restored = self.transform.inverse_transform_specz(batch.y_specz_redshift)
            renormed = self.transform.normalize_specz(restored)
            logger.info(
                f"[{split}] Transform check — specz_redshift scalar range: [{batch.y_specz_redshift.min():.4f}, {batch.y_specz_redshift.max():.4f}]"
                f"\n\t=> Inverse transform check — restored scalar range: [{restored.min():.4f}, {restored.max():.4f}]"
                f" => Re-transform check — specz_redshift scalar range: [{renormed.min():.4f}, {renormed.max():.4f}]"
            )

            logger.info(f"[{split}] y_target_image={tuple(batch.y_target_image.shape)} dtype={batch.y_target_image.dtype} range=[{img_min:.4f}, {img_max:.4f}]")
            logger.info(f"[{split}] y_specz_redshift={tuple(batch.y_specz_redshift.shape)} dtype={batch.y_specz_redshift.dtype} range=[{specz_min:.4f}, {specz_max:.4f}]")
            logger.info(f"[{split}] original_id[0]='{batch.original_id[0]}' dtype={type(batch.original_id[0])}")

        logger.info("All dataloaders validated successfully")

    def save(self):
        logger = get_logger()
        out = self.output_folder
        pt.save(self, out / "prepare_datasets.pth")

        metadata = {
            "image_normalization": {
                "original_min": float(self.transform.original_min),
                "original_max": float(self.transform.original_max),
                "normalized_min": float(self.transform.normalized_min),
                "normalized_max": float(self.transform.normalized_max),
            },
            "specz_redshift_normalization": (
            {
                "original_min": float(self.transform.specz_min),
                "original_max": float(self.transform.specz_max),
                "normalized_min": float(self.transform.normalized_min),
                "normalized_max": float(self.transform.normalized_max),
            }
            if self.transform.specz_min is not None else None
            ),
            "dataloader_settings": {
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "random_seed": self.random_seed,
            },
            "dataset_sizes": {
                "train": len(self.train_dataset),
                "valid": len(self.valid_dataset),
                "test": len(self.test_dataset),
            },
            "dataset_masking": {
                "apply_mask": self.train_dataset.apply_mask,
                "mask_ratio": self.train_dataset.mask_ratio,
                "mask_start_seed": self.train_dataset.mask_start_seed,
            },
            "dataset_sample_shapes": {
                "x_masked_image": self.train_dataset[0]["x_masked_image"].shape,
                "y_target_image": self.train_dataset[0]["y_target_image"].shape,
                "y_specz_redshift": self.train_dataset[0]["y_specz_redshift"].shape,
                "masked_region_map": self.train_dataset[0]["masked_region_map"].shape,
                "original_id": list(self.train_dataset[0]["original_id"]),
            },
            "source_sample_shapes": self.train_dataset.source_sample_shapes,
            "source_sample_info": {
                self.train_dataset.IMAGE_KEY: "Galaxy image samples with 5 channels of colorband order: (g, r, i, z, y)",
                self.train_dataset.SPECZ_KEY: "Redshift scalar (float32) values for the sample image",
                self.train_dataset.ID_KEY: "Unique id for galaxy image from the original dataset source"
            }
        }

        with open(out / "preprocessing_metadata.json", 'w') as meta_f:
            json.dump(metadata, meta_f, indent=2)
    
        logger.info(f"Saved prepared_datasets.pth and preprocessing_metadata.json to: {out}")


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
    Returns(x_masked_image, masked_region_map, y_target_image, y_specz_redshift, original_id) per sample.
    """

    IMAGE_KEY = 'image'
    SPECZ_KEY = 'specz_redshift'
    ID_KEY = 'object_id'

    def __init__(self, hdf5_path, transform=None, max_samples=None, 
                    mask_ratio=0.0, mask_start_seed=10):

        self.hdf5_path = str(hdf5_path)
        self.transform = transform
        self.max_samples = max_samples
        self.file = None

        # ==================================================
        # PARTIAL CONTRIBUTION START: Random Masking Initialization
        # Contributor: Leslie Horace
        # ==================================================

        self.mask_ratio = mask_ratio
        self.mask_start_seed = mask_start_seed
        self.apply_mask = 0.0 < self.mask_ratio < 1.0

        # ==================================================
        # PARTIAL CONTRIBUTION END: Random Masking Initialization
        # ==================================================

        with h5py.File(self.hdf5_path, 'r') as f:
            if self.IMAGE_KEY not in f:
                raise KeyError(f"'{self.IMAGE_KEY}' dataset not found in {self.hdf5_path}")
            self.length = f[self.IMAGE_KEY].shape[0]
            if max_samples is not None:
                self.length = min(self.length, max_samples)

            self.source_sample_shapes = {
                self.IMAGE_KEY: tuple(f[self.IMAGE_KEY].shape[1:]),
                self.SPECZ_KEY: tuple(f[self.SPECZ_KEY].shape[1:]),
                self.ID_KEY: tuple(f[self.ID_KEY].shape[1:]),
            }

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

        masked_image, masked_region_map = self._apply_square_mask(image, idx)

        return dict(
            x_masked_image=masked_image,             # normalized image tensor (C, H, W) with masking, model input
            masked_region_map=masked_region_map,     # binary mask tensor (1, H, W), masked pixels = 1.0
            y_target_image=image,                    # normalized image tensor (C, H, W), reconstruction target
            y_specz_redshift=specz,                  # scalar normalized spectroscopic redshift target
            original_id=original_id                  # unique id from the raw unreduced dataset
        )

    def __del__(self):
        if self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass

    # ==================================================
    # PARTIAL CONTRIBUTION START: Random Masking Function, Serialization hooks
    # Contributor: Leslie Horace
    # ==================================================

    def _apply_square_mask(self, image: pt.Tensor, idx: int):
        if image.ndim != 3:
            raise ValueError(f"Expected image shape (C, H, W), not {tuple(image.shape)}")
        C, H, W = image.shape
        if H != W:
            raise ValueError(f"Expected square images, but H != W {tuple(image.shape[1:])}")

        masked_image = image.clone()
        masked_region_map = pt.zeros((1, H, W), dtype=pt.float32)

        if not self.apply_mask:
            return masked_image, masked_region_map

        mask_size = min(max(1, round(self.mask_ratio * H)), H)

        gen = pt.Generator()
        gen.manual_seed(self.mask_start_seed + int(idx))

        max_top = H - mask_size
        top_A = 0 if max_top == 0 else (
            pt.randint(
                0, max_top + 1, (1,), 
                dtype=pt.int64, 
                generator=gen).item()
            )
        top_B = top_A + mask_size

        max_left = W - mask_size
        left_A = 0 if max_left == 0 else (
            pt.randint(
                0, max_left + 1, (1,), 
                dtype=pt.int64, 
                generator=gen).item()
            )
        left_B = left_A + mask_size

        masked_image[:, top_A:top_B, left_A:left_B] = 0.0
        masked_region_map[:, top_A:top_B, left_A:left_B] = 1.0

        return masked_image, masked_region_map

    def close(self):
        if self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass
            self.file = None
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.file = None

    # ==================================================
    # PARTIAL CONTRIBUTION START: Random Masking Function, Serialization hooks
    # ==================================================


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
    train_dataset = PrepareDataset(train_path, transform=transform, 
                                    mask_ratio=args.mask_ratio, 
                                    mask_start_seed=args.mask_start_seed)
    
    valid_dataset = PrepareDataset(valid_path, transform=transform,
                                    mask_ratio=args.mask_ratio, 
                                    mask_start_seed=args.mask_start_seed)
    
    test_dataset  = PrepareDataset(test_path, transform=transform,
                                    mask_ratio=args.mask_ratio, 
                                    mask_start_seed=args.mask_start_seed)
    
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
--mask-ratio 0.5 \
--debug

python src/preprocess_data.py \
--input-folder data/galaxiesml_small \
--output-folder data/preprocessed \
--num-cores 2 \
--mask-ratio 0.5 \
--debug

python src/preprocess_data.py \
--input-folder data/galaxiesml_medium \
--output-folder data/preprocessed \
--num-cores 2 \
--mask-ratio 0.75 \
--debug
"""