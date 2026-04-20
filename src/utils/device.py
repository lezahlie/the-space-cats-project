from utils.logger import get_logger
from utils.common import os, pt, np, gc, random

# ==================================================
# CONTRIBUTION START: Deterministic SetupDevice Class
# Contributor: Leslie Horace
# ==================================================

class SetupDevice:
    @staticmethod
    def setup_torch_threads(num_tasks):
        get_logger().info("Setting up PyTorch threads")
        # lets assume each core has 2 threads 

        reserved_threads = 2 if num_tasks > 2 else 1
        available_threads = (num_tasks*2) - reserved_threads

        pt.set_num_interop_threads(reserved_threads)
        pt.set_num_threads(available_threads)

        os.environ['OMP_NUM_THREADS'] = str(available_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(available_threads)
        os.environ['MKL_NUM_THREADS'] = str(available_threads)

        get_logger().info(f"Max usable CPU threads: {pt.get_num_threads()}")
        get_logger().info(f"Max usable inter-op threads: {pt.get_num_interop_threads()}")

    @staticmethod
    def free_memory():
        gc.collect()
        if pt.cuda.is_available():
            pt.cuda.empty_cache()
            pt.cuda.ipc_collect()

    @staticmethod
    def validate_modules(device):
        try:
            result_str = f"Validating NumPy and PyTorch modules"
            rand_array = np.random.rand(3, 3).astype(np.float32)
            result_str += f"\n1. Created random 3x3 numpy array: \n{rand_array}"
            rand_tensor = pt.tensor(rand_array, dtype=pt.float32, device=device)
            result_str += f"\n2. Converted NumPy array to PyTorch tensor on {device}:\n{rand_tensor}"
            result_tensor = rand_tensor + pt.ones_like(rand_tensor, device=device)
            result_str += f"\n3. Computed result of random tensor + 1 on {device}:\n{result_tensor}"
            get_logger().info(result_str)
        except Exception as e:
            get_logger().error(f"Failed to validate NumPy and PyTorch modules due to {e}")

    @staticmethod
    def setup_cuda(gpu_list, gpu_memory):
        device = None
        if pt.cuda.is_available():
            get_logger().info(f"CUDA is available, CUDA version: {pt.version.cuda}")
            avail_gpus = pt.cuda.device_count()
            get_logger().info(f"Available GPUs: {avail_gpus}")

            # Adjust GPU list if it exceeds the number of available GPUs
            if len(gpu_list) > avail_gpus:
                get_logger().warning(f"GPU list '{gpu_list}' requests more GPUs than available. Adjusting to available GPUs.")
                gpu_list = list(range(avail_gpus))

            gpu_string = ','.join(map(str, gpu_list))
            get_logger().info(f"Setting GPU list '{gpu_string}' and process memory fraction '{gpu_memory}'")

            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_string
            get_logger().info(f"Visible GPUs: {pt.cuda.device_count()}")
            
            device_list = []
            for i in gpu_list:
                device = pt.device(f'cuda:{i}')
                pt.cuda.set_per_process_memory_fraction(gpu_memory, device)
                device_list.append(device)
                get_logger().info(f"GPU {i} device: {pt.cuda.get_device_name(i)}")
                get_logger().info(f"GPU {i} memory: free = {pt.cuda.memory.mem_get_info(device)[0]}, available = {pt.cuda.memory.mem_get_info(device)[1]}")
            return device_list[0] if i==0 else device_list
        else:
            get_logger().info("CUDA is not available")
        return device
        
    @staticmethod
    def setup_mps():
        device = None
        if pt.backends.mps.is_available():
            get_logger().info(f"MPS is available")
            device = pt.device('mps')
        else:
            get_logger().warning("MPS is not available")
        return device
    
    @staticmethod
    def setup_cpu():
        get_logger().info("Defaulting to CPU")
        device = pt.device('cpu')
        return device

    @staticmethod
    def setup_generators(seed: int, deterministic: bool = False):
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        random.seed(seed)
        np.random.seed(seed)
        pt.manual_seed(seed)

        if pt.cuda.is_available():
            pt.cuda.manual_seed(seed)
            pt.cuda.manual_seed_all(seed)
            

        pt.backends.cudnn.benchmark = not deterministic
        pt.backends.cudnn.deterministic = deterministic
        pt.use_deterministic_algorithms(deterministic)


    @staticmethod
    def setup_torch_device(num_tasks, cpu_device_only, gpu_list=None, gpu_memory=None, random_seed=None, deterministic=False):
        SetupDevice.free_memory()
        SetupDevice.setup_generators(random_seed, deterministic=deterministic)

        get_logger().info(f"Pytorch version: {pt.__version__}")
        get_logger().info(f"CUDA Support: {pt.backends.cuda.is_built()}")
        get_logger().info(f"MPS Support: {pt.backends.mps.is_built()}")
        get_logger().info(f"Random Seed: {random_seed}")
        get_logger().info(f"Deterministic Algorithms: {deterministic}")
        
        SetupDevice.setup_torch_threads(num_tasks)
        pt.set_float32_matmul_precision('high')
        
        device = None
        if cpu_device_only:
            device = SetupDevice.setup_cpu()
        else:
            if pt.cuda.is_available():
                device = SetupDevice.setup_cuda(gpu_list, gpu_memory)
            elif pt.backends.mps.is_available():
                device = SetupDevice.setup_mps()
            else:
                device = SetupDevice.setup_cpu()

        SetupDevice.validate_modules(device)     

        return device 

# ==================================================
# CONTRIBUTION End: Deterministic SetupDevice Class
# ==================================================