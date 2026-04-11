
## Step 1: Project Setup

### A. Prerequisites

1. **Required**: POSIX compatible shell in a Linux/Unix-based environment
2. **Required**: Miniconda OR Anaconda installation that supports Python version 3.10
3. **Required**: Clone the project repo from github
4. **Required**: Determine which environment your host architecture supports
   
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

2. Activate the project conda environment and setup environment variables
    ```bash
    conda activate space-cats-<device>
    ```

### C. Download Datasets

Dataset download page: https://zenodo.org/records/11117528

- training download (3.4 GB): https://zenodo.org/records/11117528/files/5x64x64_training_with_morphology.hdf5?download=1
    ```bash
    mkdir -p data/galaxiesml && \
    wget -c "https://zenodo.org/records/11117528/files/5x64x64_training_with_morphology.hdf5?download=1" -O ../data/galaxiesml/5x64x64_testing_with_morphology.hdf5
    h5ls -r ../data/galaxiesml/5x64x64_training_with_morphology.hdf5 > ../data/galaxiesml/training_structure.txt
    ```
- validation dataset (16.9 GB): https://zenodo.org/records/11117528/files/5x64x64_training_with_morphology.hdf5?download=1
    ```bash
    mkdir -p data/galaxiesml && \
    wget -c "https://zenodo.org/records/11117528/files/5x64x64_validation_with_morphology.hdf5?download=1" -O ../data/galaxiesml/5x64x64_testing_with_morphology.hdf5 && \\
    h5ls -r ../data/galaxiesml/5x64x64_validation_with_morphology.hdf5 > ../data/galaxiesml/validation_structure.txt
    ```
- testing download (3.4 GB): https://zenodo.org/records/11117528/files/5x64x64_validation_with_morphology.hdf5?download=1
    ```bash
    mkdir -p data/galaxiesml && \
    wget -c "https://zenodo.org/records/11117528/files/5x64x64_testing_with_morphology.hdf5?download=1" -O ../data/galaxiesml/5x64x64_testing_with_morphology.hdf5 && \\
    h5ls -r ../data/galaxiesml/5x64x64_testing_with_morphology.hdf5 > ../data/galaxiesml/testing_structure.txt
    ```

> For more information go here: https://datalab.astro.ucla.edu/galaxiesml.html
> - Note remember to add the citations on this page: https://datalab.astro.ucla.edu/galaxiesml.html
> - Also for local testing lets just use one of the smaller datasets
### D. Preprocess the dataset

### E. Tune the model 

### F. Train the best model

### G. Evaluate from saved model state

### H. Analyze and plot results