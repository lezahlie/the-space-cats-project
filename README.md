
## Step 1: Project Setup

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


### C. Download Datasets

#### Reduced cleaned datasets

1. Download any of the bundles from this link: https://gtvault-my.sharepoint.com/:f:/r/personal/lhorace3_gatech_edu/Documents/DLGroupProject_Datasets?csf=1&web=1&e=fL9OqB
2. Make sure you create a `data` folder and extract the reduced dataset bundle there

```bash
cd path/to/the-space-cats-project 
mkdir -p data
tar -xvf path/to/galaxiesml_<size>.tar.gz -C data/
```

- `galaxiesml_tiny.tar.gz`: Good for debugging issues and just getting stuff working
  - `5x64x64_training_reduced_tiny.hdf5`: N=1750
  - `5x64x64_validation_reduced_tiny.hdf5`: N=500
  - `5x64x64_testing_reduced_tiny.hdf5`: N=250
  - total: N=2500

- `galaxiesml_small.tar.gz`: Good for testing training/evaluation and tuning workflows
  - `5x64x64_training_reduced_small.hdf5`: N=3500
  - `5x64x64_validation_reduced_small.hdf5`: N=1000
  - `5x64x64_testing_reduced_small.hdf5`: N=500
  - total: N=5000

- `galaxiesml_medium.tar.gz`: Good for final experimentation (unless large fits)
  - `5x64x64_training_reduced_medium.hdf5`: N=17500
  - `5x64x64_validation_reduced_medium.hdf5`: N=5000
  - `5x64x64_testing_reduced_medium.hdf5`: N=2500
  - total: N=25000

- `galaxiesml_large.tar.gz`: Ideal size for final experimentation
  - `5x64x64_training_reduced_large.hdf5`: N=35000
  - `5x64x64_validation_reduced_large.hdf5`: N=10000
  - `5x64x64_testing_reduced_large.hdf5`: N=5000
  - total: N=50000



> Each reduced dataset is packaged as tar gzip archive containing reduced training, validation, and testing HDF5 files.</br>
>  In addition, each archive includes a metadata text file describing the internal HDF5 structure.</br>
> You can regenerate the metadata text file by running `python src/inspect_data.py`

#### Original raw datasets

Dataset download page: https://zenodo.org/records/11117528
- training download (16.9 GB, N=204573): https://zenodo.org/records/11117528/files/5x64x64_training_with_morphology.hdf5
- validation dataset (3.4 GB, N=40914): https://zenodo.org/records/11117528/files/5x64x64_training_with_morphology.hdf5
- testing download (3.4 GB, N=40914): https://zenodo.org/records/11117528/files/5x64x64_validation_with_morphology.hdf5


> For more information go here: https://datalab.astro.ucla.edu/galaxiesml.html
> - Note remember to add the citations on this page: https://datalab.astro.ucla.edu/galaxiesml.html
> - Also for local testing lets just use one of the smaller datasets

### D. Preprocess the dataset

### E. Tune the model 

### F. Train the best model

### G. Evaluate from saved model state

### H. Analyze and plot results

tar -czf galaxiesml_raw.tar.gz galaxiesml_raw