from src.utils.logger import get_logger, set_logger_level, log_execution_time
from src.utils.common import argparse, os, Path, pt, h5py, GalaxiesMLDataset

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
    

    """
    please validate the arguments you added so it breaks now instead of later
    """
        
    return args

class PrepareDatasets:
    def __init__(self, train_dataset, valid_dataset, test_dataset, random_seed:int = 42):
        """read the datasets, clean, reformat, and save for later
        """
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.random_seed = random_seed
        """
        the rest is up to you
        """

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

    def fit(self, train_data):
        """fit the training split by deriving the global min and max 
        from the training split ONLY to avoid data leakage.

        Returns:
            train_data: _description_
        """
        return NotImplemented
    
    def __call__(self, data):

        return NotImplemented

    def inverse_transform(self, data, new_device=None):
        """transforms normalized data back to its original scale

        Args:
            data (_type_): tensor data
            new_device (_type_, optional): move to specific device. Defaults to None.

        """

        return NotImplemented


@log_execution_time
def main(args):
    if args.debug:
        set_logger_level(10)

    transform = Normalize()

    dataset_dir = Path(args.input_folder)
    dataset_file = "5x64x64_{}_with_morphology.hdf5"

    train_path = dataset_dir / dataset_file.format("training")
    train_raw = GalaxiesMLDataset(train_path)
    transform.fit(train_raw)

    train_data = GalaxiesMLDataset(train_path, transform=transform)
    valid_path = dataset_dir / dataset_file.format("validation")
    valid_data = GalaxiesMLDataset(valid_path, transform=transform)

    test_path = dataset_dir / dataset_file.format("testing")
    test_data = GalaxiesMLDataset(test_path, transform=transform)

    PrepareDatasets(
        train_data,
        valid_data,
        test_data,
        args.random_seed
    )

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
--debug
"""