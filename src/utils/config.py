"""
Utilities for parsing yaml and/or json configs 
"""
from src.utils.common import copy, AttrDict



DEFAULT_TRAIN_CONFIG = {
    "debug": False,
    "input_shape": [5, 64, 64],
    "num_workers": 1,
    "random_seed": 42,

    "mask_ratio": 0.0,
    "num_epochs": 50,
    "ssim_loss_weight": 0.5,

    "batch_size": 64,
    "learn_rate": 5e-4,
    "lr_scheduler": "plateau",
    "lr_scheduler_patience": 3,
    "lr_scheduler_factor": 0.2,
    "lr_scheduler_min_lr": 1e-6,

    "weight_decay": 0.0,
    "optim_beta1": 0.9,
    "optim_beta2": 0.999,
    "optim_type": "adamw",

    "hidden_layers": 3,
    "hidden_dims": 128,
    "latent_dims": 128,
    "conv_kernel": 3,
    "conv_stride": 1,

    "activation_function": "relu",
    "norm_layer": "none",   
    "negative_slope": 0.01,
    "hidden_factor": 2.0,
    "ascending_channels": False,

    "enable_earlystop": True,
    "earlystop_patience": 5,
    "earlystop_min_delta": 1e-4,

    "log_epoch_frequency": 1,
    "log_batch_frequency": 14,
    "plot_last_batch_frequency": 1,
    "plot_last_batch_limit": 3
}


def merge_config(user_config):
    if user_config is None:
        return AttrDict(copy.deepcopy(DEFAULT_TRAIN_CONFIG))

    if not isinstance(user_config, dict):
        raise TypeError(f"config must be a dict, got {type(user_config).__name__}")

    config = copy.deepcopy(DEFAULT_TRAIN_CONFIG)
    config.update(user_config)
    return AttrDict(config)


def validate_config(config):
    if not isinstance(config, (AttrDict, dict)):
        raise TypeError(f"config must be a dict or AttrDict, got {type(config).__name__}")

    config = AttrDict(config)



    # num_epochs
    if config["num_epochs"] <= 0:
        raise ValueError("num_epochs must be an INT > 0")
    num_epochs = config["num_epochs"]

    # batch_size
    if config["batch_size"] < 4:
        raise ValueError("batch_size must be an INT >= 4")

    # lr_scheduler
    config["lr_scheduler"] = str(config["lr_scheduler"]).lower().strip()
    valid_schedulers = {"none", "plateau", "cosine"}
    if config["lr_scheduler"] not in valid_schedulers:
        raise ValueError(
            f"lr_scheduler must be one of {sorted(valid_schedulers)}, "
            f"got {config['lr_scheduler']!r}"
        )

    # lr_scheduler_min_lr
    if config["lr_scheduler_min_lr"] < 0:
        raise ValueError("lr_scheduler_min_lr must be a FLOAT >= 0")

    # lr_scheduler_patience
    if config["lr_scheduler_patience"] < 0:
        raise ValueError("lr_scheduler_patience must be an INT >= 0")

    # lr_scheduler_factor
    if not (0.0 < config["lr_scheduler_factor"] < 1.0):
        raise ValueError("lr_scheduler_factor must be a FLOAT in range (0.0, 1.0)")

    # ssim_loss_weight
    if not (0.0 <= config["ssim_loss_weight"] <= 1.0):
        raise ValueError("ssim_loss_weight must be a FLOAT in range [0.0, 1.0]")

    # learn_rate
    if config["learn_rate"] <= 0:
        raise ValueError("learn_rate must be a FLOAT > 0")

    # weight_decay
    if config["weight_decay"] < 0:
        raise ValueError("weight_decay must be a FLOAT >= 0")

    # optim_beta1
    if not (0.0 < config["optim_beta1"] < 1.0):
        raise ValueError("optim_beta1 must be a FLOAT in range (0.0, 1.0)")

    # optim_beta2
    if not (0.0 < config["optim_beta2"] < 1.0):
        raise ValueError("optim_beta2 must be a FLOAT in range (0.0, 1.0)")

    # optim_type
    config["optim_type"] = str(config["optim_type"]).lower().strip()
    valid_optimizers = {"adamw"}
    if config["optim_type"] not in valid_optimizers:
        raise ValueError(
            f"optim_type must be one of {sorted(valid_optimizers)}, "
            f"got {config['optim_type']!r}"
        )

    # hidden_layers
    if config["hidden_layers"] <= 0:
        raise ValueError("hidden_layers must be an INT > 0")

    # hidden_dims
    if config["hidden_dims"] <= 0:
        raise ValueError("hidden_dims must be an INT > 0")

    # latent_dims
    if config["latent_dims"] <= 0:
        raise ValueError("latent_dims must be an INT > 0")

    # conv_kernel
    if config["conv_kernel"] <= 0:
        raise ValueError("conv_kernel must be an INT > 0")
    
    if config["conv_kernel"] % 2 == 0:
        raise ValueError("conv_kernel must be an odd INT")

    # conv_stride
    if config["conv_stride"] <= 0:
        raise ValueError("conv_stride must be an INT > 0")

    # activation_function
    config["activation_function"] = str(config["activation_function"]).lower().strip()
    valid_activations = {"relu", "leaky", "tanh"}
    if config["activation_function"] not in valid_activations:
        raise ValueError(
            f"activation_function must be one of {sorted(valid_activations)}, "
            f"got {config['activation_function']!r}"
        )

    # norm_layer
    config["norm_layer"] = str(config["norm_layer"]).lower().strip()
    valid_norm_layers = {"none", "group", "batch"}
    if config["norm_layer"] not in valid_norm_layers:
        raise ValueError(
            f"norm_layer must be one of {sorted(valid_norm_layers)}, "
            f"got {config['norm_layer']!r}"
        )
    
    
    # negative_slope
    if config["negative_slope"] < 0:
        raise ValueError("negative_slope must be a FLOAT >= 0")

    # hidden_factor
    if config["hidden_factor"] <= 0:
        raise ValueError("hidden_factor must be a FLOAT > 0")
    
    # ascending_channels
    if not isinstance(config["ascending_channels"], bool):
        raise ValueError("ascending_channels ust be a BOOL")
    
    # enable_earlystop
    if not isinstance(config["enable_earlystop"], bool):
        raise TypeError("enable_earlystop must be a BOOL")

    if config["enable_earlystop"]:
        # earlystop_patience
        if config["earlystop_patience"] < 0:
            raise ValueError(f"earlystop_patience must be an INT >= 0")

        # earlystop_min_delta
        if config["earlystop_min_delta"] < 0:
            raise ValueError("earlystop_min_delta must be a FLOAT >= 0.0")

    # plot_last_batch_frequency
    if config["plot_last_batch_frequency"] < 0:
        raise ValueError(f"plot_last_batch_frequency must be an INT >= 0")

    # plot_last_batch_limit
    if config["plot_last_batch_limit"] < 0:
        raise ValueError("plot_last_batch_limit must be an INT >= 0")

    # log_epoch_frequency
    if config["log_epoch_frequency"] < 0:
        raise ValueError(f"log_epoch_frequency must be an INT >= 0")


    # log_batch_frequency
    if config["log_batch_frequency"] < 0:
        raise ValueError("log_batch_frequency must be an INT >= 0")

    # appended fields from args or preprocessing
    if config["random_seed"] < 0:
        raise ValueError("random_seed must be an INT >= 0")

    if config["num_workers"] < 0:
        raise ValueError("num_workers must be an INT >= 0")

    if not (0.0 <= config["mask_ratio"] <= 1.0):
        raise ValueError("mask_ratio must be a FLOAT in range [0.0, 1.0]")
    
    return config