from src.utils.logger import get_logger, set_logger_level, log_execution_time
from src.utils.common import argparse, os, Path, pt, read_from_json
from src.utils.device import SetupDevice
from train_model import ModelTrainer

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
    parser.add_argument('--gpu-device-list', dest='gpu_device_list', type=int, nargs='+', default=[0], 
                help='Specify which GPU(s) to use; Provide multiple GPUs as space-separated values, e.g., "0 1" | default: 0 (if CUDA is available)')
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


class HyperparameterSearch(ModelTrainer):
    def __init__(
        self, 
        config, 
        input_folder,
        output_folder,
        device = "cuda" if pt.cuda.is_available() else "cpu"
    ):
        """Tunes the model with an exhaustive grid search or optuna to speed things up.

        a) if doing exhaustive grid search:
            we need to figure out spitting up param grids into 4-5 stages to avoid combinatorial explosion
            then carryover the best settings from each stage into the next
            ideally these stages can run automatically with some helper functions
            for instance, to fine tune around lose we can compute the log neighbors around the best values
                so say the best_lr=3e-4, we can fine tune with this lr_grid=[1e-4, 3e-4, 5e-4]

        b) if using optuna with median/threshold trial pruning, we could have afford 2 stages for a coarse search and a fine search
            Note: we will need track each trials settings and also prune/skip duplicate runs
            Optuna docs: https://optuna.readthedocs.io/en/stable/
            Optuna example: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py

        Args:
            config (dict): hyper-params for model network, optimizer, loss function, etc
        """
        super().__init__(config, input_folder, output_folder, device)
        self.base_config = dict(config)
        self.best_stage_config = dict(config)
        self.best_overall_config = dict(config)


    def run_stage(self, param_grid):
        """placeholder function for hyperparameter searching
        you can delete this and/or add whatever you want.

        Returns:
            _type_: _description_
        """
        return NotImplemented


    def make_trainer(self, config):
        return ModelTrainer(
            config=config,
            input_folder=self.input_folder,
            output_folder=self.output_folder,
            device=self.device,
        )
    
    def save_best_so_far(self):
        """placeholder function for saving the 
            - best so far model state
            - best so far epoch
            - best so far valid loss

        Returns:
            _type_: _description_
        """
        return NotImplemented
    
    def update_best_configs(self, new_best):
        """placeholder function updating the best config 
        so after each stage, carry the best settings forward to the next stage
        also track the best global config, because it's possible for a 
        stages best valid_loss to be worse than the previous stage.

        Returns:
            _type_: _description_
        """
    
        # self.best_stage_config.update()
        # self.best_overall_config.update()
        return NotImplemented
    
    def override_best_(self):
        """placeholder function for saving the 
            - best so far model state
            - best so far epoch
            - best so far loss

        Returns:
            _type_: _description_
        """
        return NotImplemented

    def load_model_state(self, load_model_path):
        """placeholder function for saving model weights

        Args:
            load_model_path (_type_): _description_

        Returns:
            _type_: _description_
        """
        return NotImplemented


    def save_model_state(self, save_model_path):
        """placeholder function for saving model weights

        Args:
            save_model_path (_type_): _description_

        Returns:
            _type_: _description_
        """
        return NotImplemented


@log_execution_time
def main(args):
    if args.debug:
        set_logger_level(10)

    logger = get_logger()

    device = SetupDevice.setup_torch_device(
        args.num_cores,
        args.cpu_device_only,
        args.gpu_device_list,
        args.gpu_memory_fraction,
        args.random_seed
    )
    logger.info(f"Using device = {device}")

    base_config = read_from_json(args.config_file, as_dict=True)
    assert all(isinstance(x, (tuple, list, dict)) for x in  base_config.values())
    search = HyperparameterSearch(base_config, device)

    logger.debug(f"TEST DEBUG LOG (debug enabled? {args.debug})")
    logger.info("TEST INFO LOG")

    """
    please implement the rest, you can change whatever you want
    """

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
python src/tune_model.py \
--config-file configs/tune_model_stage1.json \
--input-folder "data/galaxiesml_tiny" \
--output-folder experiments/tune_model_stage1 \
--num-cores 2 \
--gpu-memory-fraction 0.9 \
--debug
"""