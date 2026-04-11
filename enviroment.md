        ```bash
        conda config --set channel_priority strict
        conda clean --index-cache --tarballs --packages -y
        conda create -n spacecats-cpu -c conda-forge python=3.12 -y
        conda activate spacecats-cpu

        conda install -c pytorch -c conda-forge pytorch=2.4.1 torchvision=0.19.1 cpuonly -y

        conda install -c conda-forge numpy scipy scikit-learn scikit-image \
            torchmetrics optuna h5py jsonschema pandas pyyaml matplotlib seaborn -y
        ```