from utils.logger import get_logger, set_logger_level, log_execution_time
logger = get_logger()
from utils.common import argparse, os, Path, pt
from utils.device import SetupDevice
Dataloader=pt.utils.data.Dataloader

def process_args():
    parser = argparse.ArgumentParser(description="Tune Model Executable", formatter_class= argparse.RawTextHelpFormatter)
    parser.add_argument('--debug', '-d', dest='debug', action='store_true', 
                help="Enables debug option and verbose printing | default: Off")
    parser.add_argument('--random-seed', dest='random_seed', type=int, default=42,
                help="Random seed for selecting samples | default: None")
    parser.add_argument('--input-folder', dest="input_folder", type=str, required=True, 
                help="Input path/to/directory where the preprocessed datasets are saved | required")
    parser.add_argument('--output-folder', dest="output_folder", type=str, required=True, 
                help="Output path/to/directory to save experiment results to | required")
    parser.add_argument('--config-file', dest="config_file", type=str, required=True, 
                help="Input config file file path | required")
    
    parser.add_argument('--gpu-memory-fraction', dest='gpu_memory_fraction', type=float, default=0.5,
                help='Fraction of GPU memory to allocate per process | default: 0.5 (if CUDA is available)')
    parser.add_argument('--cpu-device-only', dest="cpu_device_only", action='store_true',
                help="PyTorch device can only use default CPU; Overrides other device options | default: Off")
    parser.add_argument('--num-cores', dest="num_cores", type=int, default=1, 
                help="Number of cpu cores (tasks) to run in parallel. If multi-threading is enabled, max threads is set to (num_tasks * 2) | default: 1")

    """
    please add the remaining arguments you need to do the things
    """
    
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

    """
    please validate the arguments you added so it breaks now instead of later
    """
        
    return args


class HyperparameterSearch:
    def __init__(
        self, 
        config, 
        device = "cuda" if pt.cuda.is_available() else "cpu"
    ):
        """Tunes the model with an either an exhaustive grid search AND/OR use the optuna framework to speed things up.

            Optuna docs: https://optuna.readthedocs.io/en/stable/
            Optuna example: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py

        Args:
            config (dict): hyper-params for model network, optimizer, loss function, etc
        """
        self.device = device
        self.config = config

    def run_search(self):
        """placeholder function for hyperparameter searching
        you can delete this and/or add whatever you want.

        Returns:
            _type_: _description_
        """
        return NotImplementedError()

    def train(self, model, train_loader):
        return NotImplementedError()
        
    @pt.no_grad()
    def evaluate(self, model, valid_or_test_loader):
        """run validation"""
        if model.training:
            model.eval()

        raise NotImplementedError

    def save_model(self):
        """placeholder function for saving model weights
        Returns:
            _type_: _description_
        """
        return NotImplementedError()


@log_execution_time
def main(args):
    if args.debug:
        set_logger_level(10)

    device = SetupDevice.setup_torch_device(
        args.num_cores,
        args.cpu_device_only,
        args.gpu_device_list,
        args.gpu_memory_fraction,
        args.random_seed
    )

    fake_config ={
        "hidden_layers": [1, 2, 3], 
        "hidden_FACTOR": 3, 
        "hidden_dims": 128, 
        "latent_dims": 64, 
        "conv_kernel": 3, 
        "conv_stride": 1, 
        "earlystop": True, 
        "epoch_patience": 20, 
        "min_delta": 0.0,
    }

    search = HyperparameterSearch(fake_config, device)

    logger.debug(f"TEST DEBUG LOG (debug enabled? {args.debug})")
    logger.info("TEST INFO LOG")

    """
    please implement the rest, you can change whatever you want
    """

if __name__ == "__main__":
    from utils.logger import init_shared_logger
    logger = init_shared_logger(__file__, log_stdout=True, log_stderr=True)
    try:
        pt.multiprocessing.set_sharing_strategy('file_system')
        args = process_args()
        main(args)
    except Exception as e:
        logger.error(e)

'''
make sure it works so far:
0
python src/tune_model.py \
--config-file configs/tune_model_stage1.json \
--input-folder data/galaxiesml \
--output-folder experiments/tune_model_stage1 \
--num-cores 2 \
--gpu-memory-fraction 0.9 \
--debug
'''