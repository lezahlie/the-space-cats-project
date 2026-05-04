
## Step 1: Local Project Setup

### A. Prerequisites

1. **Required**: POSIX compatible shell in a Linux/Unix-based environment
2. **Required**: Miniconda OR Anaconda installation that supports Python version 3.10
3. **Required**: Clone the project repo from github
4. **Required**: Checkout your branch to implement your work on

    ```bash
    git checkout branch <your first name>
    ```

    > Don't forget to pull main and backup your changes before merging to main

5. **Required**: Determine which environment your host architecture supports
   
    A. [cuda_environment.yml](./cuda_environment.yml): PyTorch `Stable` release built with `CUDA` support for Nvidia GPUs

    - If the environment file fails, try to manually install instead:

        ```bash
        conda config --set channel_priority strict
        conda clean --index-cache --tarballs --packages -y

        conda create -n spacecats-cuda -c conda-forge python=3.10 -y
        conda activate spacecats-cuda

        conda install -c conda-forge numpy pandas scipy scikit-learn scikit-image \
            torchmetrics optuna h5py jsonschema pandas pyyaml matplotlib seaborn -y

        conda install -c pytorch -c nvidia -c conda-forge pytorch=2.4.1 torchvision=0.19.1 \
            pytorch-cuda=11.8 -y
        ```

    > Note: `pytorch-cuda=11.8` often works with newer NVIDIA drivers. Only upgrade `pytorch-cuda` to `12.1` or `12.4` if `11.8` does not work on your system or your hardware requires it. If you need to change this, please update only `pytorch-cuda` first.

    B. [mps_environment.yml](./mps_environment.yml): PyTorch `Stable` release including Apple Silicon  `MPS` support (macOS 12.3+)

    - If the environment file fails, try to manually install instead:

        ```bash
        conda config --set channel_priority strict
        conda clean --index-cache --tarballs --packages -y

        conda create -n simdeck-mps -c conda-forge python=3.10 -y
        conda activate simdeck-mps

        conda install -c conda-forge numpy pandas scipy scikit-learn scikit-image \
            torchmetrics optuna h5py jsonschema pandas pyyaml matplotlib seaborn -y

        conda install -c pytorch -c conda-forge pytorch=2.4.1 torchvision=0.19.1  -y
        ```

    C. [cpu_environment.yml](./cpu_environment.yml): PyTorch `Stable` release with `CPU-ONLY` build

    - If the environment file fails, try to manually install instead:

        ```bash
        conda config --set channel_priority strict
        conda clean --index-cache --tarballs --packages -y
        
        conda create -n simdeck-cpu -c conda-forge python=3.10 -y
        conda activate simdeck-cpu

        conda install -c conda-forge numpy pandas scipy scikit-learn scikit-image \
            torchmetrics optuna h5py jsonschema pandas pyyaml matplotlib seaborn -y

        conda install -c pytorch -c conda-forge pytorch=2.4.1 torchvision=0.19.1 cpuonly -y
        ```

### B. Environment

#### Google Colab Setup

Please upload `the-space-cats-colab.ipynb` to google colab and follow the steps to setup

#### Option 1: Conda install

0. Make sure conda channel priority is strict

    ```bash
    conda config --set channel_priority strict
    ```

    > Optional commands for faster conda solver:
    > ```bash
    > conda install -n base conda-libmamba-solver
    > conda config --set solver libmamba
    > ```

1. Create new conda env with the environment file from the previous step
   
    ```bash
    conda env create -f <environment.yml>
    ```

    > Note: If conda solver is taking too long or fails try using `mamba`
    > ```bash
    > conda clean --index-cache --tarballs --packages -y
    > conda install -n base -c conda-forge mamba -y
    > mamba env create -f <environment.yml>
    > ```

2. Activate the project conda environment and add project path to `PYTHONPATH` 
    ```bash
    conda activate spacecats-<device>
    conda env config vars set PYTHONPATH="/path/to/the-space-cats-project${PYTHONPATH:+:$PYTHONPATH}"
    conda deactivate
    conda activate spacecats-<device>
    echo $PYTHONPATH
    ```

### C. Dataset Overview

#### Reduced cleaned datasets

1. Download any of the bundles from this link: https://gtvault-my.sharepoint.com/:f:/r/personal/lhorace3_gatech_edu/Documents/DLGroupProject_Datasets?csf=1&web=1&e=fL9OqB
2. Make sure you create a `data` folder and extract the reduced dataset bundle there

```bash
cd path/to/the-space-cats-project 
mkdir -p data
tar -xvf path/to/galaxiesml_<size>.tar.gz -C data/
```

- `galaxiesml_tiny.tar.gz`: Good for debugging issues and just getting stuff working
  - `5x64x64_training_reduced_tiny.hdf5`: N=2000
  - `5x64x64_validation_reduced_tiny.hdf5`: N=500
  - `5x64x64_testing_reduced_tiny.hdf5`: N=500
  - total: N=3000

- `galaxiesml_small.tar.gz`: Good for testing training/evaluation and tuning workflows
  - `5x64x64_training_reduced_small.hdf5`: N=5000
  - `5x64x64_validation_reduced_small.hdf5`: N=1000
  - `5x64x64_testing_reduced_small.hdf5`: N=1000
  - total: N=7000

- `galaxiesml_medium.tar.gz`: Good for final experimentation unless large fits
  - `5x64x64_training_reduced_medium.hdf5`: N=10000
  - `5x64x64_validation_reduced_medium.hdf5`: N=2500
  - `5x64x64_testing_reduced_medium.hdf5`: N=2500
  - total: N=15000

- `galaxiesml_large.tar.gz`: Ideal size for final experimentation
  - `5x64x64_training_reduced_large.hdf5`: N=20000
  - `5x64x64_validation_reduced_large.hdf5`: N=5000
  - `5x64x64_testing_reduced_large.hdf5`: N=5000
  - total: N=30000

> - Each reduced dataset is packaged as tar gzip archive containing reduced training, validation, and testing HDF5 files
> - In addition, each archive includes a metadata text file describing the internal HDF5 structure


#### FOR REFERENCE ONLY: Original raw datasets

Dataset download page: https://zenodo.org/records/11117528
- training download (16.9 GB, N=204573): https://zenodo.org/records/11117528/files/5x64x64_training_with_morphology.hdf5
- validation dataset (3.4 GB, N=40914): https://zenodo.org/records/11117528/files/5x64x64_training_with_morphology.hdf5
- testing download (3.4 GB, N=40914): https://zenodo.org/records/11117528/files/5x64x64_validation_with_morphology.hdf5


> For more information go here: https://datalab.astro.ucla.edu/galaxiesml.html
> - Note remember to add the citations on this page: https://datalab.astro.ucla.edu/galaxiesml.html
> - Also for local testing lets just use one of the smaller datasets

## Step 2: Masked AutoEncoder Experiments

### A. Prerequisites

> **It is extremely important that you run these experiments on a system with the following constraints**
>
> 1. A Linux-based system that can run for multiple days **without interruption**
> 2. One dedicated **CUDA-compatible GPU** with no competing workloads
> 3. **3 to 6 total CPU cores** is preferred
>    - Set `--num-cores` to **2 to 5**
>    - Always leave **1 additional CPU core** free for the host to prevent lag and crashes
>    - AMD or Intel **x64** CPU architecture is required to run the environment 
> 4. DRAM requirement: allocate **4 GB** per allocated core (plus **4 GB** reserved for the host) 
>    - `--num-cores < 2`: this will be too slow; allocate at least 2
>    - `--num-cores=2`: **12 GB** DRAM
>    - `--num-cores=3`: **16 GB** DRAM
>    - `--num-cores=4`: **20 GB** DRAM
>    - `--num-cores=5`: **24 GB** DRAM
>    - `--num-cores > 5`: this is too many cores for a single GPU run


### B. Setup CUDA environment 

> Note: It does not matter which option you choose as long as you do not change the pytorch versions

#### Option 1: Google colab setup

> **Please refer to the `the-space-cats-colab.ipynb` notebook for all remaining steps

#### Option 2: Conda or virtual env setup

1. **Conda env**: 
   
   - Run these in order
        ```bash
        conda config --set channel_priority strict
        conda env create -f cuda_environment.yaml

        conda activate spacecats-cuda
        conda env config vars set PYTHONPATH="/path/to/the-space-cats-project${PYTHONPATH:+:$PYTHONPATH}"

        conda deactivate

        conda activate spacecats-cuda
        echo $PYTHONPATH
        ```

        > Make sure to update `/path/to/the-space-cats-project` with the absolute path to the project

2. **Python virtual env**:

    - First make sure you have python version `3.10.13`, if not you will need to install it OR switch to conda setup


        ```bash
        python3 --version

        # Expected output: Python 3.10.13
        ```

    - Run these in order
  
        ```bash
        cd "/path/to/the-space-cats-project"
        mkdir -p venvs

        python3.10 -m venv venvs/spacecats-cuda
        source venvs/spacecats-cuda/bin/activate

        pip install --upgrade pip setuptools wheel
        pip install -r cuda_requirements.txt
        ```

        > Make sure to update `/path/to/the-space-cats-project` with the absolute path to the project

### C. Test the pipeline

1. Download `galaxiesml_tiny.tar.gz` from: https://gtvault-my.sharepoint.com/:u:/g/personal/lhorace3_gatech_edu/IQCavSjOG4EzSa62o0fZPaPJAUdR4WKUkWPAAOOk1E-g6YI?e=1SnoPD

    Extract it to the data folder

    ```bash
    cd "/path/to/the-space-cats-project"

    mkdir -p "data" && tar -xzf "/path/to/downloads/galaxiesml_tiny.tar.gz" -C "data/"
    ```

2. Test preprocessing works

    ```bash
    python src/preprocess_data.py \
    --input-folder "data/galaxiesml_tiny" \
    --output-folder "data/preprocessed" \
    --num-cores 2 \
    --mask-ratio 0.0 \
    --debug
    ```

3. Test tuning works

    ```bash
    python src/tune_model.py \
    --config-file "configs/tune_default.json" \
    --input-folder "data/preprocessed/galaxiesml_tiny" \
    --output-folder "experiments/tune_debug_tiny" \
    --gpu-memory-fraction 0.9 \
    --num-cores 5 \
    --tune-optimizer-steps 500 \
    --validate-every-steps 50 \
    --tune-patience 2 \
    --debug
    ```

4. Test training works

    ```bash
    python src/train_model.py \
    --config-file "configs/train_overfit.json" \
    --input-folder "data/preprocessed/galaxiesml_tiny" \
    --output-folder "experiments/train_overfit_tiny" \
    --gpu-memory-fraction 0.9 \
    --num-cores 5 \
    --tune-optimizer-steps 1000 \
    --validate-every-steps 50 \
    --max-wallclock-hours 2 \
    --checkpoint-buffer-minutes 30
    --debug
    ```

5. If you got to this point then you are ready to start tuning!

> Please keep in mind this may take several days to complete

### D. Run Preprocessing

1. Download `galaxiesml_small.tar.gz` and `galaxiesml_medium.tar.gz` from: https://gtvault-my.sharepoint.com/:u:/g/personal/lhorace3_gatech_edu/IQCavSjOG4EzSa62o0fZPaPJAUdR4WKUkWPAAOOk1E-g6YI?e=1SnoPD

    Extract it to the data folder

    ```bash
    cd "/path/to/the-space-cats-project"
    mkdir -p "data" && tar -xzf "/path/to/downloads/galaxiesml_small.tar.gz" -C "data/"
    mkdir -p "data" && tar -xzf "/path/to/downloads/galaxiesml_medium.tar.gz" -C "data/"
    ```

2. Preprocess with your assigned mask ratio 

    ```bash
    python src/preprocess_data.py \
    --input-folder "data/galaxiesml_small" \
    --output-folder "data/preprocessed" \
    --num-cores <num_cores> \
    --mask-ratio <mask_ratio>
    ```

    ```bash
    python src/preprocess_data.py \
    --input-folder "data/galaxiesml_medium" \
    --output-folder "data/preprocessed" \
    --num-cores <num_cores> \
    --mask-ratio <mask_ratio>
    ```

    > Replace <mask_ratio> with:
    >- Leslie: `mask_ratio = 0.0`
    >- Charlie: `mask_ratio = 0.25`
    >- Chris: `mask_ratio = 0.5`
    >- Wen: `mask_ratio = 0.75`

### E. Tune the model 

### Important Notes: 

1. `tune_model.py` autodetects and recovers completed stages and trials from saved CSV logs.
2. Each trial runs for at most `--tune-optimizer-steps` optimizer updates, unless capped earlier by `num_epochs * batches_per_epoch`.
3. A trial stops early if either of these happens:
   - its validation loss does not improve for `epoch_patience=4` consecutive validation checks
     - `epoch_patience=4` is defined in `configs/tune_default.json`)
   - after a current best exists, it fails to beat the current best validation loss within `tune_patience=4 * validate_every_steps=100` optimizer updates after the current best trial's best optimizer step
     - `tune_patience` and `validate_every_steps` are defined in the pace scripts and in the command below
4. Tuning only runs `64` total trials now with early stopping per-trial and across trials


```bash
python src/tune_model.py \
--config-file "configs/tune_default.json" \
--input-folder "data/preprocessed/galaxiesml_small" \
--output-folder "experiments/tune_mae_small_<first_name>_<mask>" \
--gpu-memory-fraction 0.9 \
--num-cores 5 \
--tune-optimizer-steps 750 \
--validate-every-steps 50 \
--tune-patience 3
```
> - Do not pass `--debug` or it will NOT run the full tuning grid
> - Replace `<your_name>` and `<mask_ratio>` in `[--output-folder]` in case we need to share our results
> - Replace `<num_cores>` with `2` to `5` cores, where `5` is preferred for speed
> - For system requirements refer to [A. Prerequisites](#step-2-masked-autoencoder-experiments-a-prerequisites)


### F. Train the best model

1. Copy the best overall config from tuning to the configs folder

    ```bash
    cp -p "experiments/tune_mae_small_<first_name>_<mask_ratio>/best_overall_config.json" "configs/train_best_<first_name>_<mask_ratio>.json"
    ```
    > Please put your <first_name> and <mask_ratio> and commit the new config to github

2. Train the model with the best params one more time to obtain results for analysis

    ```bash
    python src/train_model.py \
    --config-file "configs/best_config_<first_name>_mask_<mask_ratio>.json" \
    --input-folder "data/preprocessed/galaxiesml_medium" \
    --output-folder "experiments/train_mae_medium_<first_name>_mask_<mask_ratio>" \
    --gpu-memory-fraction 0.9 \
    --num-cores 5 \
    --validate-every-steps 100 \
    --max-wallclock-hours 16 \
    --checkpoint-buffer-minutes 120
    ```

    > Notes 
    > - This will run for more epochs with early stopping
    > - Save outputs for downstream regression and analysis

3. If the medium dataset doesn't work then everyone needs to use the small instead

## Step 3: KNN Latent Space Regression
### Overview
The KNN regressor evaluates whether the MAE latent space preserves redshift-relevant structure. It takes the encoder's latent vectors (`z_latent_vector`) as input and predicts spectroscopic redshift (`y_specz_redshift`). Results are compared across all four mask ratios to assess how masking affects latent representation quality.

- **Baseline (mask=0.0)**: MAE outputs with masked ratio of 0.0 are used to tune KNN hyperparameters and save a shared best config
- **Ablations (mask=0.25, 0.5, 0.75)**: All other team members use the shared best config to run KNN evaluation

### Prerequisites

- MAE training must be complete and outputs must be saved to:
  ```
  experiments/train_mae_medium_<first_name>_mask_<mask_ratio>/artifacts/samples/
  ```
- Required files per experiment: `training_outputs_best.hdf5`, `validation_outputs_best.hdf5`, `testing_outputs_best.hdf5`

### A. Run on PACE-ICE

Submit your KNN job (runs automatically after MAE training outputs are available):

```bash
# Submit your job only
bash pace/submit_knn.sh <first_name>

# Example
bash pace/submit_knn.sh wen
```

 **Note**: Baseline job (mask=0.0) must complete first — it tunes the KNN and saves shared hyperparameters to `configs/knn_best_params.yaml`. All other jobs depend on this file.

### B. Run manually

```bash
# Baseline (tunes KNN, saves best params)
python -m src.analysis.knn_regressor \
    --input-folder "experiments/train_mae_medium_leslie_mask_0.0/artifacts/samples" \
    --output-folder "experiments/knn_results/train_mae_medium_leslie_mask_0.0"

# Ablations (uses shared best params)
python -m src.analysis.knn_regressor \
    --input-folder "experiments/train_mae_medium_<first_name>_mask_<mask_ratio>/artifacts/samples" \
    --output-folder "experiments/knn_results/train_mae_medium_<first_name>_mask_<mask_ratio>" \
    --params-file "configs/knn_best_params.yaml"
```

### C. Outputs

Results are saved to `experiments/knn_results/train_mae_medium_<first_name>_mask_<mask_ratio>/`:

```
knn_results/
├── knn_best_params.yaml       ← best hyperparameters (baseline only, copied to configs/)
├── knn_predictions.csv        ← per-sample predictions and ground truth
└── knn_metrics.json           ← summary metrics (MAE, MSE, R²)
```

---

## Step 4: CNN Reconstruction Regression
### Overview

### Need to update the readme for users instead of us